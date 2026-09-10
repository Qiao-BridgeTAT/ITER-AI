"""SMS delivery boundary and production-provider adapters."""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

import httpx
from alibabacloud_dypnsapi20170525 import (  # type: ignore[import-untyped]
    models as aliyun_dypns_models,
)
from alibabacloud_dypnsapi20170525.client import (  # type: ignore[import-untyped]
    Client as AliyunDypnsClient,
)
from alibabacloud_tea_openapi import models as aliyun_openapi_models  # type: ignore[import-untyped]

TENCENT_SMS_HOST = "sms.tencentcloudapi.com"
TENCENT_SMS_ENDPOINT = f"https://{TENCENT_SMS_HOST}/"
TENCENT_SMS_SERVICE = "sms"
TENCENT_SMS_ACTION = "SendSms"
TENCENT_SMS_VERSION = "2021-01-11"
TENCENT_CONTENT_TYPE = "application/json; charset=utf-8"
TENCENT_SIGNED_HEADERS = "content-type;host;x-tc-action"


class SmsDeliveryError(RuntimeError):
    """Raised when a provider cannot accept an SMS delivery request."""


class SmsProvider(Protocol):
    async def send_verification_code(self, phone: str, code: str) -> None: ...


class AliyunDypnsSmsClient(Protocol):
    async def send_sms_verify_code_async(
        self,
        request: aliyun_dypns_models.SendSmsVerifyCodeRequest,
    ) -> aliyun_dypns_models.SendSmsVerifyCodeResponse: ...


class UnavailableSmsProvider:
    """Safe default: never logs or silently pretends to send verification codes."""

    async def send_verification_code(self, phone: str, code: str) -> None:
        del phone, code
        raise SmsDeliveryError("SMS delivery is not configured")


@dataclass(frozen=True)
class TencentCloudSmsConfig:
    """Server-only identifiers and credentials for one approved SMS template."""

    secret_id: str
    secret_key: str
    sdk_app_id: str
    sign_name: str
    template_id: str
    region: str


class TencentCloudSmsProvider:
    """Send verification codes through Tencent Cloud SMS API 2021-01-11."""

    def __init__(
        self,
        config: TencentCloudSmsConfig,
        *,
        client: httpx.AsyncClient | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._config = config
        self._client = client or httpx.AsyncClient(timeout=10.0)
        self._owns_client = client is None
        self._clock = clock

    async def send_verification_code(self, phone: str, code: str) -> None:
        timestamp = int(self._clock())
        payload = {
            "PhoneNumberSet": [phone],
            "SmsSdkAppId": self._config.sdk_app_id,
            "SignName": self._config.sign_name,
            "TemplateId": self._config.template_id,
            "TemplateParamSet": [code],
        }
        body = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        headers = _tencent_headers(
            body,
            timestamp=timestamp,
            secret_id=self._config.secret_id,
            secret_key=self._config.secret_key,
            region=self._config.region,
        )
        try:
            response = await self._client.post(
                TENCENT_SMS_ENDPOINT,
                content=body.encode("utf-8"),
                headers=headers,
            )
            response.raise_for_status()
            result = response.json()
        except (httpx.HTTPError, json.JSONDecodeError, ValueError) as exc:
            raise SmsDeliveryError("SMS provider request failed") from exc

        if not _tencent_response_succeeded(result):
            raise SmsDeliveryError("SMS provider rejected the delivery request")

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


@dataclass(frozen=True)
class AliyunPhoneVerificationSmsConfig:
    """Server-only configuration for Aliyun Phone Verification SMS authentication."""

    access_key_id: str
    access_key_secret: str = field(repr=False, compare=False)
    sign_name: str = ""
    template_code: str = ""
    region: str = "cn-hangzhou"
    valid_time_seconds: int = 300
    interval_seconds: int = 60


class AliyunPhoneVerificationSmsProvider:
    """Deliver backend-generated codes with Aliyun SendSmsVerifyCode."""

    def __init__(
        self,
        config: AliyunPhoneVerificationSmsConfig,
        *,
        client: AliyunDypnsSmsClient | None = None,
    ) -> None:
        if config.valid_time_seconds <= 0:
            raise ValueError("valid_time_seconds must be greater than zero")
        if config.interval_seconds <= 0:
            raise ValueError("interval_seconds must be greater than zero")
        self._config = config
        self._client = client or AliyunDypnsClient(
            aliyun_openapi_models.Config(
                access_key_id=config.access_key_id,
                access_key_secret=config.access_key_secret,
                endpoint="dypnsapi.aliyuncs.com",
                protocol="https",
                region_id=config.region,
                connect_timeout=5_000,
                read_timeout=10_000,
            )
        )

    async def send_verification_code(self, phone: str, code: str) -> None:
        local_phone = _mainland_china_phone(phone)
        valid_minutes = max(1, (self._config.valid_time_seconds + 59) // 60)
        request = aliyun_dypns_models.SendSmsVerifyCodeRequest(
            auto_retry=1,
            country_code="86",
            duplicate_policy=1,
            interval=self._config.interval_seconds,
            phone_number=local_phone,
            return_verify_code=False,
            sign_name=self._config.sign_name,
            template_code=self._config.template_code,
            template_param=json.dumps(
                {"code": code, "min": str(valid_minutes)},
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
            valid_time=self._config.valid_time_seconds,
        )
        try:
            response = await self._client.send_sms_verify_code_async(request)
        except Exception as exc:
            raise SmsDeliveryError("SMS provider request failed") from exc

        body = response.body
        if body is None or body.code != "OK" or body.success is not True:
            raise SmsDeliveryError("SMS provider rejected the delivery request")


def _mainland_china_phone(phone: str) -> str:
    local_phone = phone.removeprefix("+86") if phone.startswith("+86") else ""
    if len(local_phone) != 11 or not local_phone.startswith("1") or not local_phone.isdigit():
        raise SmsDeliveryError("SMS provider only supports mainland China phone numbers")
    return local_phone


def _tencent_headers(
    body: str,
    *,
    timestamp: int,
    secret_id: str,
    secret_key: str,
    region: str,
) -> dict[str, str]:
    date = datetime.fromtimestamp(timestamp, tz=UTC).strftime("%Y-%m-%d")
    canonical_headers = (
        f"content-type:{TENCENT_CONTENT_TYPE}\n"
        f"host:{TENCENT_SMS_HOST}\n"
        f"x-tc-action:{TENCENT_SMS_ACTION.lower()}\n"
    )
    hashed_payload = hashlib.sha256(body.encode("utf-8")).hexdigest()
    canonical_request = (
        f"POST\n/\n\n{canonical_headers}\n{TENCENT_SIGNED_HEADERS}\n{hashed_payload}"
    )
    credential_scope = f"{date}/{TENCENT_SMS_SERVICE}/tc3_request"
    string_to_sign = (
        "TC3-HMAC-SHA256\n"
        f"{timestamp}\n"
        f"{credential_scope}\n"
        f"{hashlib.sha256(canonical_request.encode('utf-8')).hexdigest()}"
    )
    secret_date = _hmac_sha256(f"TC3{secret_key}".encode(), date)
    secret_service = _hmac_sha256(secret_date, TENCENT_SMS_SERVICE)
    secret_signing = _hmac_sha256(secret_service, "tc3_request")
    signature = hmac.new(
        secret_signing,
        string_to_sign.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    authorization = (
        "TC3-HMAC-SHA256 "
        f"Credential={secret_id}/{credential_scope}, "
        f"SignedHeaders={TENCENT_SIGNED_HEADERS}, "
        f"Signature={signature}"
    )
    return {
        "Authorization": authorization,
        "Content-Type": TENCENT_CONTENT_TYPE,
        "Host": TENCENT_SMS_HOST,
        "X-TC-Action": TENCENT_SMS_ACTION,
        "X-TC-Timestamp": str(timestamp),
        "X-TC-Version": TENCENT_SMS_VERSION,
        "X-TC-Region": region,
    }


def _hmac_sha256(key: bytes, message: str) -> bytes:
    return hmac.new(key, message.encode("utf-8"), hashlib.sha256).digest()


def _tencent_response_succeeded(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    response = payload.get("Response")
    if not isinstance(response, dict) or response.get("Error") is not None:
        return False
    statuses = response.get("SendStatusSet")
    return (
        isinstance(statuses, list)
        and bool(statuses)
        and all(isinstance(item, dict) and item.get("Code") == "Ok" for item in statuses)
    )
