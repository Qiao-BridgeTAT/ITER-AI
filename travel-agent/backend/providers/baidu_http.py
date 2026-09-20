"""Shared safe HTTP and error semantics for Baidu Map Web API adapters."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import httpx

from backend.contracts.enums import ProviderCode
from backend.providers.contracts import ProviderError, ProviderFailureCode
from backend.providers.request_budget import budgeted_external_request

BAIDU_BASE_URL = "https://api.map.baidu.com"

_INVALID_REQUEST_CODES = frozenset({2, 8})
_AUTHENTICATION_CODES = frozenset({3, 5, 101, 102, 200})
_PERMISSION_CODES = frozenset({9, 201, 202, 203, 210, 211, 240, 250, 251, 252, 260, 261})
_RATE_LIMIT_CODES = frozenset({4, 302, 401})
_UNAVAILABLE_CODES = frozenset({1})


@budgeted_external_request
async def request_baidu_json(
    client: httpx.AsyncClient,
    api_key: str,
    path: str,
    parameters: Mapping[str, str],
    operation: str,
) -> dict[str, Any]:
    try:
        response = await client.get(f"{BAIDU_BASE_URL}{path}", params={"ak": api_key, **parameters})
        response.raise_for_status()
    except httpx.TimeoutException:
        raise ProviderError(
            ProviderCode.BAIDU,
            ProviderFailureCode.TIMEOUT,
            operation,
            retryable=True,
        ) from None
    except httpx.HTTPStatusError as exc:
        code = _http_failure_code(exc.response.status_code)
        raise ProviderError(
            ProviderCode.BAIDU,
            code,
            operation,
            retryable=code in {ProviderFailureCode.RATE_LIMITED, ProviderFailureCode.UNAVAILABLE},
        ) from None
    except httpx.HTTPError:
        raise ProviderError(
            ProviderCode.BAIDU,
            ProviderFailureCode.UNAVAILABLE,
            operation,
            retryable=True,
        ) from None
    try:
        payload = response.json()
    except ValueError:
        raise baidu_malformed(operation) from None
    if not isinstance(payload, dict):
        raise baidu_malformed(operation)
    parsed = cast(dict[str, Any], payload)
    ensure_baidu_success(parsed, operation)
    return parsed


def ensure_baidu_success(payload: dict[str, Any], operation: str) -> None:
    status = payload.get("status")
    if status == 0:
        return
    if isinstance(status, bool) or not isinstance(status, int):
        raise baidu_malformed(operation)
    if status in _AUTHENTICATION_CODES:
        code = ProviderFailureCode.AUTHENTICATION_FAILED
    elif status in _PERMISSION_CODES:
        code = ProviderFailureCode.PERMISSION_DENIED
    elif status in _INVALID_REQUEST_CODES:
        code = ProviderFailureCode.INVALID_REQUEST
    elif status in _RATE_LIMIT_CODES:
        code = ProviderFailureCode.RATE_LIMITED
    elif status in _UNAVAILABLE_CODES:
        code = ProviderFailureCode.UNAVAILABLE
    else:
        code = ProviderFailureCode.UPSTREAM_ERROR
    raise ProviderError(
        ProviderCode.BAIDU,
        code,
        operation,
        retryable=status != 302
        and code
        in {
            ProviderFailureCode.RATE_LIMITED,
            ProviderFailureCode.UNAVAILABLE,
            ProviderFailureCode.UPSTREAM_ERROR,
        },
    )


def baidu_malformed(operation: str) -> ProviderError:
    return ProviderError(
        ProviderCode.BAIDU,
        ProviderFailureCode.MALFORMED_RESPONSE,
        operation,
        retryable=False,
    )


def _http_failure_code(status_code: int) -> ProviderFailureCode:
    if status_code in {400, 404, 422}:
        return ProviderFailureCode.INVALID_REQUEST
    if status_code == 401:
        return ProviderFailureCode.AUTHENTICATION_FAILED
    if status_code == 403:
        return ProviderFailureCode.PERMISSION_DENIED
    if status_code == 429:
        return ProviderFailureCode.RATE_LIMITED
    if status_code >= 500:
        return ProviderFailureCode.UNAVAILABLE
    return ProviderFailureCode.UPSTREAM_ERROR
