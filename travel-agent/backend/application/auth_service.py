"""Phone verification, account identity protection, and opaque web sessions."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from backend.contracts.rest import AccountView, SmsSendRequest, SmsVerifyRequest
from backend.domain.account_profile import effective_nickname
from backend.persistence.models import AuthIdentity, User
from backend.persistence.redis_temporary import RedisTemporaryStore
from backend.providers.sms import SmsProvider


class AuthenticationError(RuntimeError):
    pass


class InvalidVerificationCodeError(AuthenticationError):
    pass


class AuthenticationRateLimitError(AuthenticationError):
    def __init__(self, retry_after_seconds: int) -> None:
        super().__init__("authentication rate limit exceeded")
        self.retry_after_seconds = retry_after_seconds


class AuthenticationConflictError(AuthenticationError):
    pass


@dataclass(frozen=True)
class AuthClientContext:
    ip_address: str
    device_id: str


@dataclass(frozen=True)
class VerifiedSession:
    account: AccountView
    session_token: str


@dataclass(frozen=True)
class AuthenticationPolicy:
    code_ttl_seconds: int = 300
    session_ttl_seconds: int = 2_592_000
    idempotency_ttl_seconds: int = 86_400
    send_phone_limit: int = 5
    send_ip_limit: int = 20
    send_device_limit: int = 10
    verify_phone_limit: int = 10
    verify_ip_limit: int = 40
    verify_device_limit: int = 20
    rate_window_seconds: int = 3_600


class PhoneProtector:
    """Provides deterministic lookup and authenticated encryption for phone PII."""

    def __init__(self, secret: str) -> None:
        if len(secret.strip()) < 16:
            raise ValueError("PII_ENCRYPTION_KEY must contain at least 16 characters")
        self._secret = secret.encode("utf-8")
        self._key = hashlib.sha256(b"travel-agent:phone-encryption:v1:" + self._secret).digest()

    def lookup_hash(self, phone: str) -> str:
        return hmac.new(self._secret, phone.encode("utf-8"), hashlib.sha256).hexdigest()

    def encrypt(self, phone: str) -> str:
        nonce = secrets.token_bytes(12)
        ciphertext = AESGCM(self._key).encrypt(nonce, phone.encode("utf-8"), b"phone:v1")
        return base64.urlsafe_b64encode(nonce + ciphertext).decode("ascii")

    def decrypt(self, encrypted_phone: str) -> str:
        payload = base64.urlsafe_b64decode(encrypted_phone.encode("ascii"))
        if len(payload) < 29:
            raise ValueError("encrypted phone payload is invalid")
        plaintext = AESGCM(self._key).decrypt(payload[:12], payload[12:], b"phone:v1")
        return plaintext.decode("utf-8")

    @staticmethod
    def mask(phone: str) -> str:
        if phone.startswith("+86") and len(phone) == 14:
            local_phone = phone[3:]
            return f"{local_phone[:3]}****{local_phone[-4:]}"
        visible_prefix = phone[:3] if len(phone) > 7 else phone[:2]
        return f"{visible_prefix}{'*' * max(4, len(phone) - len(visible_prefix) - 4)}{phone[-4:]}"


class AuthenticationService:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        redis: RedisTemporaryStore,
        sms_provider: SmsProvider,
        *,
        phone_protector: PhoneProtector,
        session_secret: str,
        policy: AuthenticationPolicy | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        if len(session_secret.strip()) < 16:
            raise ValueError("SESSION_SECRET must contain at least 16 characters")
        self._session_factory = session_factory
        self._redis = redis
        self._sms = sms_provider
        self._phones = phone_protector
        self._session_secret = session_secret.encode("utf-8")
        self._policy = policy or AuthenticationPolicy()
        self._clock = clock

    @property
    def phone_protector(self) -> PhoneProtector:
        return self._phones

    @property
    def session_ttl_seconds(self) -> int:
        return self._policy.session_ttl_seconds

    async def send_code(
        self,
        request: SmsSendRequest,
        context: AuthClientContext,
        idempotency_key: str,
    ) -> None:
        phone_hash = self._phones.lookup_hash(request.phone)
        fingerprint = self._fingerprint(request.model_dump(mode="json"))
        existing = await self._redis.get_idempotency("sms-send", phone_hash, idempotency_key)
        if existing is not None:
            self._assert_replay(existing, fingerprint)
            return
        await self._enforce_rate_limits("send", phone_hash, context)
        claimed = await self._redis.claim_idempotency(
            "sms-send",
            phone_hash,
            idempotency_key,
            {"status": "processing", "request_fingerprint": fingerprint},
            self._policy.idempotency_ttl_seconds,
        )
        if not claimed:
            raise AuthenticationConflictError("an identical request is being processed")
        code = f"{secrets.randbelow(1_000_000):06d}"
        digest = self._code_digest(phone_hash, code)
        try:
            await self._redis.save_sms_challenge(phone_hash, digest, self._policy.code_ttl_seconds)
            await self._sms.send_verification_code(request.phone, code)
            await self._redis.save_idempotency_result(
                "sms-send",
                phone_hash,
                idempotency_key,
                {
                    "status": "complete",
                    "request_fingerprint": fingerprint,
                    "response": None,
                },
                self._policy.idempotency_ttl_seconds,
            )
        except Exception:
            await self._redis.delete_sms_challenge(phone_hash)
            await self._redis.clear_idempotency("sms-send", phone_hash, idempotency_key)
            raise

    async def verify_code(
        self,
        request: SmsVerifyRequest,
        context: AuthClientContext,
        idempotency_key: str,
    ) -> VerifiedSession:
        phone_hash = self._phones.lookup_hash(request.phone)
        fingerprint = self._fingerprint(request.model_dump(mode="json"))
        existing = await self._redis.get_idempotency("sms-verify", phone_hash, idempotency_key)
        if existing is not None:
            self._assert_replay(existing, fingerprint)
            response = existing.get("response")
            if not isinstance(response, dict):
                raise AuthenticationConflictError("verification replay is incomplete")
            return VerifiedSession(
                account=AccountView.model_validate(response["account"]),
                session_token=str(response["session_token"]),
            )
        await self._enforce_rate_limits("verify", phone_hash, context)
        claimed = await self._redis.claim_idempotency(
            "sms-verify",
            phone_hash,
            idempotency_key,
            {"status": "processing", "request_fingerprint": fingerprint},
            self._policy.idempotency_ttl_seconds,
        )
        if not claimed:
            raise AuthenticationConflictError("an identical request is being processed")
        try:
            outcome = await self._redis.consume_sms_challenge(
                phone_hash, self._code_digest(phone_hash, request.code)
            )
            if outcome != "matched":
                raise InvalidVerificationCodeError("verification code is invalid or expired")
            account = await self._find_or_create_account(request.phone, phone_hash)
            session_token = secrets.token_urlsafe(32)
            await self._redis.save_auth_session(
                session_token,
                {"user_id": str(account.user_id)},
                self._policy.session_ttl_seconds,
            )
            result = VerifiedSession(account=account, session_token=session_token)
            saved = await self._redis.save_idempotency_result(
                "sms-verify",
                phone_hash,
                idempotency_key,
                {
                    "status": "complete",
                    "request_fingerprint": fingerprint,
                    "response": {
                        "account": account.model_dump(mode="json"),
                        "session_token": session_token,
                    },
                },
                self._policy.idempotency_ttl_seconds,
            )
            if not saved:
                await self._redis.delete_auth_session(session_token)
                raise AuthenticationConflictError("verification result could not be retained")
            return result
        except Exception:
            await self._redis.clear_idempotency("sms-verify", phone_hash, idempotency_key)
            raise

    async def resolve_user_id(self, session_token: str) -> UUID | None:
        raw = await self._redis.load_auth_session(session_token)
        if raw is None:
            return None
        try:
            return UUID(str(raw["user_id"]))
        except (KeyError, TypeError, ValueError):
            await self._redis.delete_auth_session(session_token)
            return None

    async def logout(self, session_token: str) -> None:
        await self._redis.delete_auth_session(session_token)

    async def logout_all(self, user_id: UUID) -> None:
        await self._redis.clear_user_auth_sessions(user_id)

    async def account_view(self, user_id: UUID) -> AccountView | None:
        async with self._session_factory() as session:
            row = await session.scalar(
                select(AuthIdentity)
                .where(AuthIdentity.user_id == user_id, AuthIdentity.provider == "phone")
                .order_by(AuthIdentity.created_at, AuthIdentity.id)
                .limit(1)
            )
            user = await session.get(User, user_id)
        if row is None or user is None:
            return None
        phone = self._phones.decrypt(row.encrypted_identifier)
        return AccountView(
            user_id=user.id,
            masked_phone=self._phones.mask(phone),
            nickname=effective_nickname(user.id, user.nickname),
            created_at=user.created_at.replace(tzinfo=UTC)
            if user.created_at.tzinfo is None
            else user.created_at,
        )

    async def _find_or_create_account(self, phone: str, phone_hash: str) -> AccountView:
        now = self._clock()
        user: User | None = None
        async with self._session_factory() as session, session.begin():
            identity = await session.scalar(
                select(AuthIdentity).where(
                    AuthIdentity.provider == "phone", AuthIdentity.lookup_hash == phone_hash
                )
            )
            if identity is None:
                user = User(id=uuid4(), status="active")
                session.add(user)
                identity = AuthIdentity(
                    id=uuid4(),
                    user_id=user.id,
                    provider="phone",
                    lookup_hash=phone_hash,
                    encrypted_identifier=self._phones.encrypt(phone),
                    verified_at=now,
                )
                session.add(identity)
                await session.flush()
            else:
                user = await session.get(User, identity.user_id)
                if user is None or user.status != "active":
                    raise AuthenticationError("account is unavailable")
                identity.verified_at = now
        if user is None:
            raise AuthenticationError("account is unavailable")
        return AccountView(
            user_id=user.id,
            masked_phone=self._phones.mask(phone),
            nickname=effective_nickname(user.id, user.nickname),
            created_at=(
                user.created_at.replace(tzinfo=UTC)
                if user.created_at.tzinfo is None
                else user.created_at
            ),
        )

    async def _enforce_rate_limits(
        self, operation: str, phone_hash: str, context: AuthClientContext
    ) -> None:
        limits = (
            ("phone", phone_hash, getattr(self._policy, f"{operation}_phone_limit")),
            ("ip", context.ip_address, getattr(self._policy, f"{operation}_ip_limit")),
            ("device", context.device_id, getattr(self._policy, f"{operation}_device_limit")),
        )
        for kind, subject, limit in limits:
            decision = await self._redis.consume_rate_limit(
                f"sms-{operation}:{kind}",
                subject,
                limit=limit,
                window_seconds=self._policy.rate_window_seconds,
            )
            if not decision.allowed:
                raise AuthenticationRateLimitError(decision.retry_after_seconds)

    def _code_digest(self, phone_hash: str, code: str) -> str:
        return hmac.new(
            self._session_secret,
            f"{phone_hash}:{code}".encode(),
            hashlib.sha256,
        ).hexdigest()

    @staticmethod
    def _fingerprint(value: Any) -> str:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _assert_replay(existing: dict[str, Any], fingerprint: str) -> None:
        if existing.get("request_fingerprint") != fingerprint:
            raise AuthenticationConflictError("idempotency key was reused for another request")
        if existing.get("status") != "complete":
            raise AuthenticationConflictError("an identical request is being processed")
