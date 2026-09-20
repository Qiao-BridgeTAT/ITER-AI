"""Allowlisted AMap Streamable HTTP MCP transport. No implicit Web API fallback."""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Awaitable, Callable
from hashlib import sha256
from typing import Any, Protocol
from urllib.parse import urlsplit
from uuid import uuid4

import httpx
from jsonschema import Draft202012Validator  # type: ignore[import-untyped]

from backend.contracts.enums import ProviderCode
from backend.providers.amap_http import ensure_amap_success
from backend.providers.contracts import ProviderError, ProviderFailureCode
from backend.providers.request_budget import (
    RequestBudgetExceeded,
    active_request_budget,
    budgeted_external_request,
)
from backend.providers.resilience import ProviderRateLimiter

AMAP_MCP_ENDPOINT = "https://mcp.amap.com/mcp"
READ_TOOLS = frozenset(
    {
        "maps_polygon_search",
        "maps_text_search",
        "maps_around_search",
        "maps_search_detail",
        "maps_direction_walking",
        "maps_direction_driving",
        "maps_direction_transit_integrated",
        "maps_direction_bicycling",
        "maps_distance",
        "maps_geo",
        "maps_regeocode",
    }
)


BASIC_SEARCH_TOOLS = frozenset({"maps_text_search", "maps_around_search", "maps_search_detail"})
SEARCH_TOOLS = BASIC_SEARCH_TOOLS | {"maps_polygon_search"}


def validate_search_mcp_url(endpoint: str) -> str:
    parsed = urlsplit(endpoint)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"localhost", "127.0.0.1"}
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path != "/mcp"
    ):
        raise ValueError("AMAP_SEARCH_MCP_URL must be a loopback HTTP /mcp endpoint")
    try:
        if parsed.port is not None and not 1 <= parsed.port <= 65535:
            raise ValueError
    except ValueError:
        raise ValueError("AMAP_SEARCH_MCP_URL has an invalid port") from None
    return endpoint


class McpGateway(Protocol):
    _client: httpx.AsyncClient

    async def tools(self) -> dict[str, dict[str, Any]]: ...
    async def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]: ...


class AmapMcpClient:
    def __init__(
        self,
        api_key: str,
        *,
        client: httpx.AsyncClient | None = None,
        endpoint: str = AMAP_MCP_ENDPOINT,
        rate_limiter: ProviderRateLimiter | None = None,
        sleeper: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if endpoint == AMAP_MCP_ENDPOINT:
            if not api_key.strip():
                raise ValueError("AMap MCP requires a Web Service key")
        else:
            validate_search_mcp_url(endpoint)
            if api_key:
                raise ValueError("Local search MCP must not receive an upstream key")
        self.endpoint = endpoint
        self._key = api_key
        self._client = client or httpx.AsyncClient(timeout=22, follow_redirects=False)
        self._owns_client = client is None
        self._session: str | None = None
        self._protocol = "2025-03-26"
        self._tools: dict[str, dict[str, Any]] | None = None
        self._lock = asyncio.Lock()
        self._rate_limiter = rate_limiter
        self._rate_subject = sha256(api_key.encode()).hexdigest()[:20]
        self._sleeper = sleeper
        self._clock = clock
        self._pace_locks: dict[str, asyncio.Lock] = {}
        self._next_call: dict[str, float] = {}

    async def _wait(self, seconds: float) -> None:
        budget = active_request_budget.get()
        if budget is not None and budget.remaining() - 20 <= seconds:
            raise RequestBudgetExceeded("planner_external_budget_exhausted")
        if seconds > 0:
            await self._sleeper(seconds)

    async def _pace(self, name: str) -> None:
        # Dispatch spacing is distinct from the shared four-request concurrency cap.
        if self.endpoint != AMAP_MCP_ENDPOINT:
            return
        async with self._pace_locks.setdefault(name, asyncio.Lock()):
            await self._wait(max(0, self._next_call.get(name, 0) - self._clock()))
            if self._rate_limiter is not None:
                waited = 0.0
                while True:
                    decision = await self._rate_limiter.consume_rate_limit(
                        "amap-mcp-outbound",
                        f"{self._rate_subject}:{name}",
                        limit=1,
                        window_seconds=1,
                    )
                    if decision.allowed:
                        break
                    delay = max(0.1, float(decision.retry_after_seconds))
                    if waited + delay > 15:
                        raise _error("mcp_rate_queue", ProviderFailureCode.RATE_LIMITED)
                    await self._wait(delay)
                    waited += delay
            self._next_call[name] = self._clock() + 1.05

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def tools(self) -> dict[str, dict[str, Any]]:
        async with self._lock:
            if self._tools is not None:
                return self._tools
            hello = await self._rpc(
                "initialize",
                {
                    "protocolVersion": self._protocol,
                    "capabilities": {},
                    "clientInfo": {"name": "iter-planner", "version": "2"},
                },
            )
            protocol = hello.get("protocolVersion")
            if protocol not in {"2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25"}:
                raise _error("mcp_protocol", ProviderFailureCode.MALFORMED_RESPONSE)
            self._protocol = protocol
            await self._rpc("notifications/initialized", {}, notification=True)
            selected = {}
            cursor = None
            for _ in range(8):
                listing = await self._rpc("tools/list", {"cursor": cursor} if cursor else {})
                if not isinstance(listing.get("tools"), list):
                    raise _error("mcp_discovery", ProviderFailureCode.MALFORMED_RESPONSE)
                for tool in listing["tools"]:
                    if (
                        isinstance(tool, dict)
                        and tool.get("name") in READ_TOOLS
                        and isinstance(tool.get("inputSchema"), dict)
                    ):
                        selected[tool["name"]] = tool
                cursor = listing.get("nextCursor")
                if not cursor:
                    self._tools = selected
                    return selected
            raise _error("mcp_discovery", ProviderFailureCode.MALFORMED_RESPONSE)

    async def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        tools = await self.tools()
        if name not in READ_TOOLS or name not in tools:
            raise _error("mcp_tool_unavailable", ProviderFailureCode.PERMISSION_DENIED)
        if not Draft202012Validator(tools[name]["inputSchema"]).is_valid(arguments):
            raise _error("mcp_arguments", ProviderFailureCode.INVALID_REQUEST)
        for attempt in range(2):
            try:
                await self._pace(name)
                return await self._call_once(name, arguments)
            except ProviderError as error:
                error.attempts = attempt + 1
                if self.endpoint != AMAP_MCP_ENDPOINT or not error.retryable or attempt == 1:
                    raise
                await self._wait(max(1.5, error.retry_after_seconds or 0))
        raise AssertionError("MCP retry exhausted")

    async def _call_once(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        result = await self._rpc("tools/call", {"name": name, "arguments": arguments})
        if result.get("isError"):
            body = json.dumps(result)
            # Preserve only allowlisted symbolic/numeric codes, never upstream text.
            symbols = {
                "DAILY_QUERY_OVER_LIMIT": "10003",
                "ACCESS_TOO_FREQUENT": "10004",
                "IP_QUERY_OVER_LIMIT": "10010",
                "CUQPS_HAS_EXCEEDED_THE_LIMIT": "10021",
                "USER_DAILY_QUERY_OVER_LIMIT": "10044",
                "USER_MONTHLY_QUERY_OVER_LIMIT": "10045",
            }
            upstream = next(
                (
                    symbols[symbol]
                    for symbol in sorted(symbols, key=len, reverse=True)
                    if symbol in body
                ),
                None,
            )
            numeric = re.search(r'"(?:infocode|upstream_code)"\s*:\s*"?(\d{5})(?!\d)', body)
            upstream = numeric.group(1) if numeric else upstream
            if upstream:
                ensure_amap_success({"status": "0", "infocode": upstream}, "mcp_tool_error")
            structured = result.get("structuredContent")
            structured_error = structured.get("error", {}) if isinstance(structured, dict) else {}
            error_code = (
                structured_error.get("code") if isinstance(structured_error, dict) else None
            )
            code = (
                ProviderFailureCode.INVALID_REQUEST
                if error_code == "invalid_arguments"
                else ProviderFailureCode(error_code)
                if error_code in {c.value for c in ProviderFailureCode}
                else ProviderFailureCode.RATE_LIMITED
                if any(
                    value in body
                    for value in ("DAILY_QUERY_OVER_LIMIT", "CUQPS_HAS_EXCEEDED_THE_LIMIT")
                )
                else ProviderFailureCode.UPSTREAM_ERROR
            )
            raise _error("mcp_tool_error", code)
        structured = result.get("structuredContent")
        if isinstance(structured, dict):
            if str(structured.get("status")) == "0":
                ensure_amap_success(structured, "mcp_tool_error")
            return structured
        for item in result.get("content", []):
            if isinstance(item, dict) and item.get("type") == "text":
                try:
                    value = json.loads(item["text"])
                except (ValueError, TypeError, KeyError):
                    continue
                if isinstance(value, dict):
                    if str(value.get("status")) == "0":
                        ensure_amap_success(value, "mcp_tool_error")
                    return value
        raise _error("mcp_result", ProviderFailureCode.MALFORMED_RESPONSE)

    @budgeted_external_request
    async def _rpc(
        self, method: str, params: dict[str, Any], *, notification: bool = False
    ) -> dict[str, Any]:
        request_id = str(uuid4())
        body = {"jsonrpc": "2.0", "method": method, "params": params}
        if not notification:
            body["id"] = request_id
        headers = {
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": self._protocol,
        }
        if self._session:
            headers["Mcp-Session-Id"] = self._session
        try:
            response = await self._client.post(
                self.endpoint,
                params={"key": self._key} if self._key else None,
                json=body,
                headers=headers,
            )
        except httpx.TimeoutException:
            raise _error("mcp_request", ProviderFailureCode.TIMEOUT) from None
        except httpx.HTTPError:
            raise _error("mcp_request", ProviderFailureCode.UNAVAILABLE) from None
        if response.status_code >= 300:
            code = {
                401: ProviderFailureCode.AUTHENTICATION_FAILED,
                403: ProviderFailureCode.PERMISSION_DENIED,
                429: ProviderFailureCode.RATE_LIMITED,
            }.get(response.status_code, ProviderFailureCode.UPSTREAM_ERROR)
            error = _error("mcp_request", code)
            if response.status_code in {429, 503}:
                retry_after = response.headers.get("Retry-After", "")
                if retry_after.isascii() and retry_after.isdigit():
                    error.retry_after_seconds = float(retry_after)
            raise error
        if notification:
            return {}
        if len(response.content) > 2_000_000:
            raise _error("mcp_size", ProviderFailureCode.MALFORMED_RESPONSE)
        self._session = response.headers.get("Mcp-Session-Id", self._session)
        try:
            if "text/event-stream" in response.headers.get("content-type", ""):
                values = [
                    json.loads(line[5:].strip())
                    for line in response.text.splitlines()
                    if line.startswith("data:")
                ]
                value = next(v for v in values if isinstance(v, dict) and v.get("id") == request_id)
            else:
                value = response.json()
            if (
                value.get("id") != request_id
                or value.get("jsonrpc") != "2.0"
                or "error" in value
                or not isinstance(value.get("result"), dict)
            ):
                raise ValueError("invalid RPC response")
            return dict(value["result"])
        except (ValueError, KeyError, TypeError, AttributeError, StopIteration):
            raise _error("mcp_response", ProviderFailureCode.MALFORMED_RESPONSE) from None


def _error(operation: str, code: ProviderFailureCode) -> ProviderError:
    return ProviderError(
        ProviderCode.AMAP,
        code,
        operation,
        retryable=code
        in {
            ProviderFailureCode.RATE_LIMITED,
            ProviderFailureCode.TIMEOUT,
            ProviderFailureCode.UNAVAILABLE,
            ProviderFailureCode.UPSTREAM_ERROR,
        },
    )


class AmapMcpRouter:
    """Explicit per-tool server routing. A failed local search never falls back."""

    def __init__(self, official: AmapMcpClient, search: AmapMcpClient | None = None) -> None:
        self.official = official
        self.search = search
        self._client = official._client

    async def tools(self) -> dict[str, dict[str, Any]]:
        official = await self.official.tools()
        if self.search is None:
            return official
        search = await self.search.tools()
        if not search.keys() >= BASIC_SEARCH_TOOLS:
            raise _error("local_mcp_search_tools_missing", ProviderFailureCode.UNAVAILABLE)
        return {
            **{n: t for n, t in official.items() if n not in SEARCH_TOOLS},
            **{n: t for n, t in search.items() if n in SEARCH_TOOLS},
        }

    async def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        target = self.search if self.search and name in SEARCH_TOOLS else self.official
        return await target.call(name, arguments)

    def source(self, name: str) -> str:
        return "local_search_mcp" if self.search and name in SEARCH_TOOLS else "official_amap_mcp"
