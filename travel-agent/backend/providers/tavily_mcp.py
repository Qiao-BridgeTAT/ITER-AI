"""Official Tavily MCP through the locked Python MCP SDK, without API fallback."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Literal

import httpx

from backend.providers.request_budget import active_request_budget

if TYPE_CHECKING:
    from mcp import ClientSession

TAVILY_ENDPOINT = "https://mcp.tavily.com/mcp/"
TAVILY_READ_TOOLS = frozenset({"tavily_search", "tavily_extract"})


class WebSearchMcpError(RuntimeError):
    """Only a safe code, never upstream text, URLs or credentials."""


def checked_payload(value: dict[str, Any]) -> dict[str, Any]:
    # Keyless quota responses may have isError=false and no `error` field.
    # Never forward the service's account instructions into Agent context.
    if "results" not in value and (
        "retry_after_seconds" in value
        or any(word in str(value.get("code", "")).lower() for word in ("limit", "quota"))
    ):
        raise WebSearchMcpError("web_search_rate_limited")
    if "error" in value:
        raise WebSearchMcpError("web_search_upstream_error")
    return value


class TavilySession:
    def __init__(self, session: ClientSession) -> None:
        self.session = session

    async def tools(self) -> dict[str, dict[str, Any]]:
        try:
            listing = await self.session.list_tools()
        except Exception:
            raise WebSearchMcpError("web_search_discovery_unavailable") from None
        return {
            t.name: t.model_dump(mode="json", exclude_none=True)
            for t in listing.tools
            if t.name in TAVILY_READ_TOOLS
        }

    async def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name not in TAVILY_READ_TOOLS:
            raise WebSearchMcpError("web_search_tool_not_allowed")
        try:
            async with asyncio.timeout(12):
                result = await self.session.call_tool(name, arguments)
        except TimeoutError:
            raise WebSearchMcpError("web_search_timeout") from None
        except Exception:
            raise WebSearchMcpError("web_search_unavailable") from None
        if result.isError:
            raise WebSearchMcpError("web_search_upstream_error")
        if result.structuredContent is not None:
            value = dict(result.structuredContent)
            return checked_payload(value)
        for block in result.content:
            if block.type == "text" and len(block.text) <= 2_000_000:
                try:
                    value = json.loads(block.text)
                except ValueError:
                    continue
                if isinstance(value, dict):
                    return checked_payload(value)
        raise WebSearchMcpError("web_search_result_invalid")


@dataclass(frozen=True)
class TavilyMcpClient:
    api_key: str = field(default="", repr=False, compare=False)
    auth_mode: Literal["key", "keyless"] = "key"

    def __post_init__(self) -> None:
        if self.auth_mode == "key" and not self.api_key.strip():
            raise ValueError("TAVILY_API_KEY is required in key mode")
        if self.auth_mode not in {"key", "keyless"}:
            raise ValueError("TAVILY_MCP_AUTH_MODE must be key or keyless")
        if self.auth_mode == "keyless" and self.api_key:
            raise ValueError("keyless mode cannot also carry a Tavily key")

    @asynccontextmanager
    async def connect(self) -> AsyncIterator[TavilySession]:
        try:
            from mcp import ClientSession
            from mcp.client.streamable_http import streamablehttp_client
        except ImportError:
            raise WebSearchMcpError("web_search_sdk_not_installed") from None
        # Captured per planning segment, not in a process-global connection.
        budget = active_request_budget.get()

        async def before_request(request: httpx.Request) -> None:
            # Session teardown must work after the business budget is exhausted.
            if budget is not None and request.method != "DELETE":
                await budget.reserve()

        def client_factory(
            headers: dict[str, str] | None = None,
            timeout: httpx.Timeout | None = None,
            auth: httpx.Auth | None = None,
        ) -> httpx.AsyncClient:
            return httpx.AsyncClient(
                headers=headers,
                timeout=timeout or httpx.Timeout(12),
                auth=auth,
                follow_redirects=False,
                event_hooks={"request": [before_request]},
            )

        headers = (
            {"X-Tavily-Access-Mode": "keyless"}
            if self.auth_mode == "keyless"
            else {"Authorization": f"Bearer {self.api_key}"}
        )
        # SDK task groups can wrap exceptions thrown by their caller. Close them
        # normally, then re-raise the caller's original error (including cancel).
        caller_error: BaseException | None = None
        try:
            async with AsyncExitStack() as stack:
                read, write, _ = await stack.enter_async_context(
                    streamablehttp_client(
                        TAVILY_ENDPOINT,
                        headers=headers,
                        timeout=12,
                        sse_read_timeout=20,
                        httpx_client_factory=client_factory,
                    )
                )
                session = await stack.enter_async_context(
                    ClientSession(
                        read,
                        write,
                        read_timeout_seconds=timedelta(seconds=12),
                    )
                )
                await session.initialize()
                try:
                    yield TavilySession(session)
                except BaseException as error:
                    caller_error = error
        except Exception:
            if caller_error is None:
                raise WebSearchMcpError("web_search_connection_unavailable") from None
        if caller_error is not None:
            raise caller_error
