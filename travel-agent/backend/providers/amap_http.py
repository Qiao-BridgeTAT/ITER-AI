"""Shared HTTP and error semantics for AMap Web Service adapters."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, cast

import httpx

from backend.config.settings import AmapPolygonProxySettings, AmapSearchProxySettings
from backend.contracts.enums import ProviderCode
from backend.providers.contracts import ProviderError, ProviderFailureCode
from backend.providers.request_budget import budgeted_external_request

AMAP_BASE_URL = "https://restapi.amap.com"
AMAP_PROXY_SEARCH_TIMEOUT_SECONDS = 15.0
AMAP_SEARCH_PATHS = frozenset(
    {
        "/v3/place/text",
        "/v3/place/around",
        "/v3/place/detail",
        "/v5/place/text",
        "/v5/place/around",
        "/v5/place/detail",
        "/v5/place/polygon",
    }
)
_AUTH_PARAMETER_NAMES = frozenset({"key", "ak"})


class _CredentialQueryFilter(logging.Filter):
    """HTTPX logs request URLs at INFO; omit the whole authenticated query."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.args, tuple):
            record.args = tuple(
                item.copy_with(query=None)
                if isinstance(item, httpx.URL)
                and any(name.casefold() in _AUTH_PARAMETER_NAMES for name in item.params)
                else item
                for item in record.args
            )
        return True


logging.getLogger("httpx").addFilter(_CredentialQueryFilter())

_AUTHENTICATION_CODES = frozenset({"10001", "10005", "10006", "10007", "10008", "10009", "10013"})
_PERMISSION_CODES = frozenset(
    {"10002", "10012", "10026", "10041", "20011", "40000", "40002", "40003"}
)
_INVALID_REQUEST_CODES = frozenset(
    {"10011", "20000", "20001", "20002", "20012", "20800", "20801", "20803", "40001"}
)
_RATE_LIMIT_CODES = frozenset(
    {
        "10003",
        "10004",
        "10010",
        "10014",
        "10015",
        "10019",
        "10020",
        "10021",
        # Older QPS codes may still appear in recorded provider responses.
        "10022",
        "10023",
        "10029",
        "10044",
        "10045",
    }
)
_UNAVAILABLE_CODES = frozenset({"10016", "10017"})
_NON_RECOVERABLE_QUOTA_CODES = frozenset({"10003", "10010", "10044", "10045"})


@budgeted_external_request
async def request_amap_json(
    client: httpx.AsyncClient,
    api_key: str,
    path: str,
    parameters: Mapping[str, str],
    operation: str,
    *,
    search_proxy: AmapSearchProxySettings | None = None,
    polygon_proxy: AmapPolygonProxySettings | None = None,
) -> dict[str, Any]:
    selected_proxy: AmapPolygonProxySettings | AmapSearchProxySettings | None = (
        polygon_proxy
        if path == "/v5/place/polygon" and polygon_proxy is not None
        else search_proxy
        if path in AMAP_SEARCH_PATHS
        else None
    )
    base_url = selected_proxy.base_url if selected_proxy else AMAP_BASE_URL
    auth_name = selected_proxy.key_parameter if selected_proxy else "key"
    auth_value = selected_proxy.api_key if selected_proxy else api_key
    request = client.build_request(
        "GET", f"{base_url}{path}", params={"output": "JSON", **parameters}
    )
    if selected_proxy is not None:
        request.extensions["timeout"] = httpx.Timeout(AMAP_PROXY_SEARCH_TIMEOUT_SECONDS).as_dict()
    # Strip auth aliases after HTTPX merges client-level defaults, then add exactly
    # one credential for the selected host. Business parameters remain unchanged.
    request.url = request.url.copy_with(
        params=[
            (name, value)
            for name, value in request.url.params.multi_items()
            if name.casefold() not in _AUTH_PARAMETER_NAMES
        ]
        + [(auth_name, auth_value)]
    )
    try:
        # Do not forward query credentials to an upstream redirect target.
        response = await client.send(request, follow_redirects=False)
        response.raise_for_status()
    except httpx.TimeoutException:
        raise ProviderError(
            ProviderCode.AMAP,
            ProviderFailureCode.TIMEOUT,
            operation,
            retryable=True,
        ) from None
    except httpx.HTTPStatusError as exc:
        code = _http_failure_code(exc.response.status_code)
        raise ProviderError(
            ProviderCode.AMAP,
            code,
            operation,
            retryable=code in {ProviderFailureCode.RATE_LIMITED, ProviderFailureCode.UNAVAILABLE},
            upstream_code=str(exc.response.status_code),
        ) from None
    except httpx.HTTPError:
        raise ProviderError(
            ProviderCode.AMAP,
            ProviderFailureCode.UNAVAILABLE,
            operation,
            retryable=True,
        ) from None
    try:
        payload = response.json()
    except ValueError:
        raise amap_malformed(operation) from None
    if not isinstance(payload, dict):
        raise amap_malformed(operation)
    parsed = cast(dict[str, Any], payload)
    ensure_amap_success(parsed, operation)
    return parsed


def ensure_amap_success(payload: dict[str, Any], operation: str) -> None:
    status = payload.get("status")
    if status == "1":
        return
    if status != "0":
        raise amap_malformed(operation)
    raw_infocode = str(payload.get("infocode", ""))
    # Only AMap's numeric code is safe to propagate, never arbitrary gateway text.
    infocode = (
        raw_infocode
        if len(raw_infocode) == 5 and raw_infocode.isascii() and raw_infocode.isdigit()
        else ""
    )
    if infocode in _AUTHENTICATION_CODES:
        code = ProviderFailureCode.AUTHENTICATION_FAILED
    elif infocode in _PERMISSION_CODES:
        code = ProviderFailureCode.PERMISSION_DENIED
    elif infocode in _INVALID_REQUEST_CODES:
        code = ProviderFailureCode.INVALID_REQUEST
    elif infocode in _RATE_LIMIT_CODES:
        code = ProviderFailureCode.RATE_LIMITED
    elif infocode in _UNAVAILABLE_CODES:
        code = ProviderFailureCode.UNAVAILABLE
    else:
        code = ProviderFailureCode.UPSTREAM_ERROR
    raise ProviderError(
        ProviderCode.AMAP,
        code,
        operation,
        retryable=code
        in {
            ProviderFailureCode.RATE_LIMITED,
            ProviderFailureCode.UNAVAILABLE,
            ProviderFailureCode.UPSTREAM_ERROR,
        }
        and infocode not in _NON_RECOVERABLE_QUOTA_CODES,
        upstream_code=infocode,
    )


def amap_malformed(operation: str) -> ProviderError:
    return ProviderError(
        ProviderCode.AMAP,
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
