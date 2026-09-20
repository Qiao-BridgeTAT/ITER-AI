"""Official MCP protocol with a narrow REST proxy adapter for three POI tools."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from jsonschema import Draft202012Validator  # type: ignore[import-untyped]
from mcp import types
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.types import Receive, Scope, Send

from backend.config.settings import AmapSearchProxySettings
from backend.providers.amap_http import request_amap_json
from backend.providers.contracts import ProviderError

TOOL_PATHS = {
    "maps_polygon_search": "/v5/place/polygon",
    "maps_text_search": "/v3/place/text",
    "maps_around_search": "/v3/place/around",
    "maps_search_detail": "/v3/place/detail",
}


def search_tools() -> list[types.Tool]:
    """Keep official descriptions; narrow only unsupported/unsafe argument shapes."""
    snapshot = json.loads(Path(__file__).with_name("official_search_tools.json").read_text())
    result = []
    for definition in snapshot["tools"]:
        schema = definition["inputSchema"]
        properties = schema["properties"]
        # The proxy has no verified equivalent of official MCP's street-ranking strategy.
        properties.pop("strategy", None)
        schema["additionalProperties"] = False
        for name in ("keywords", "city", "id"):
            if name in properties:
                properties[name].update(minLength=1, maxLength=200)
        if "radius" in properties:
            properties["radius"].update(pattern=r"^[0-9]{1,5}$")
        if "location" in properties:
            properties["location"].update(maxLength=60)
        result.append(
            types.Tool(
                **definition,
                annotations=types.ToolAnnotations(
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=True,
                ),
            )
        )
    result.append(
        types.Tool(
            name="maps_polygon_search",
            description="本地扩展：搜索指定闭合矩形范围内的景点或餐厅，保持高德真实结果。范围仅作空间筛选，不表示实际路线或通勤时间。",
            inputSchema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "polygon": {
                        "type": "string",
                        "maxLength": 400,
                        "description": "闭合矩形，五个经度,纬度点以 | 分隔，首尾相同",
                    },
                    "types": {
                        "type": "string",
                        "enum": ["050000", "110000"],
                        "description": "050000餐厅，110000景点",
                    },
                    "keywords": {"type": "string", "minLength": 1, "maxLength": 100},
                    "page_num": {"type": "integer", "minimum": 1, "maximum": 100},
                    "page_size": {"type": "integer", "minimum": 1, "maximum": 25},
                },
                "required": ["polygon", "types"],
            },
            annotations=types.ToolAnnotations(
                readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True
            ),
        )
    )
    return result


def tool_result(data: dict[str, Any], *, error: bool = False) -> types.CallToolResult:
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=json.dumps(data, ensure_ascii=False))],
        structuredContent=data,
        isError=error,
    )


def invalid_arguments() -> types.CallToolResult:
    return tool_result(
        {"error": {"code": "invalid_arguments", "message": "请按工具参数定义提供有效查询条件。"}},
        error=True,
    )


class SearchProxyTools:
    def __init__(self, settings: AmapSearchProxySettings, client: httpx.AsyncClient) -> None:
        self.settings = settings
        self.client = client
        self.definitions = {tool.name: tool for tool in search_tools()}
        self.semaphore = asyncio.Semaphore(4)

    async def call(self, name: str, arguments: dict[str, Any]) -> types.CallToolResult:
        definition = self.definitions.get(name)
        if definition is None:
            return tool_result({"error": {"code": "unknown_tool"}}, error=True)
        if not Draft202012Validator(definition.inputSchema).is_valid(arguments):
            return invalid_arguments()
        if any(not str(v).strip() for v in arguments.values()):
            return invalid_arguments()
        if name == "maps_text_search" and arguments.get("citylimit") and not arguments.get("city"):
            return invalid_arguments()
        if name == "maps_around_search":
            try:
                longitude, latitude = map(float, arguments["location"].split(","))
                if not (-180 <= longitude <= 180 and -90 <= latitude <= 90):
                    return invalid_arguments()
                if not 0 <= int(arguments.get("radius", "3000")) <= 50000:
                    return invalid_arguments()
            except (ValueError, TypeError):
                return invalid_arguments()
        if name == "maps_polygon_search":
            from backend.contracts.enums import CoordinateSystem
            from backend.contracts.places import Gcj02Coordinates
            from backend.providers.contracts import validate_search_polygon

            try:
                points = tuple(
                    Gcj02Coordinates(
                        longitude=float(pair.split(",")[0]),
                        latitude=float(pair.split(",")[1]),
                        coord_system=CoordinateSystem.GCJ_02,
                    )
                    for pair in arguments["polygon"].split("|")
                )
                if len(points) != 5 or any(
                    len(pair.split(",")) != 2 for pair in arguments["polygon"].split("|")
                ):
                    return invalid_arguments()
                # Geometry validation needs no city lookup or external request.
                validate_search_polygon(points)
            except (ValueError, TypeError, IndexError):
                return invalid_arguments()
        parameters = {
            key: str(value).lower() if isinstance(value, bool) else str(value)
            for key, value in arguments.items()
        }
        if name == "maps_polygon_search":
            parameters.update(show_fields="business,navi,photos")
            parameters.setdefault("page_num", "1")
            parameters.setdefault("page_size", "20")
        else:
            parameters["extensions"] = "all"
            if name != "maps_search_detail":
                parameters.update(page="1", offset="20")
        try:
            # Includes queue time; no retries or fallback can silently multiply calls.
            async with asyncio.timeout(20), self.semaphore:
                payload = await request_amap_json(
                    self.client,
                    self.settings.api_key,
                    TOOL_PATHS[name],
                    parameters,
                    name,
                    search_proxy=self.settings,
                )
            pois = payload.get("pois")
            if not isinstance(pois, list) or any(not isinstance(poi, dict) for poi in pois):
                return tool_result({"error": {"code": "malformed_response"}}, error=True)
            result = {
                "pois": pois,
                "count": payload.get("count"),
                "source": "amap_search_proxy",
                "observed_at": datetime.now(UTC).isoformat(),
            }
            encoded = json.dumps(result, ensure_ascii=False)
            if len(encoded.encode()) > 2_000_000 or self.settings.api_key in encoded:
                return tool_result({"error": {"code": "unsafe_or_oversized_response"}}, error=True)
            return tool_result(result)
        except ProviderError as error:
            return tool_result(
                {
                    "error": {
                        "code": error.code.value,
                        "upstream_code": error.upstream_code,
                        "retryable": error.retryable,
                    }
                },
                error=True,
            )
        except TimeoutError:
            return tool_result({"error": {"code": "timeout", "retryable": True}}, error=True)
        except Exception:
            # Never propagate an exception that may embed an authenticated URL or payload.
            return tool_result({"error": {"code": "internal_error"}}, error=True)


def create_app(
    settings: AmapSearchProxySettings, *, client: httpx.AsyncClient | None = None
) -> Starlette:
    upstream = client or httpx.AsyncClient(timeout=15, follow_redirects=False)
    tools = SearchProxyTools(settings, upstream)
    server: Server[Any, Any] = Server("iter-amap-search-proxy", version="1.0.0")

    @server.list_tools()  # type: ignore[no-untyped-call, untyped-decorator]
    async def list_tools() -> list[types.Tool]:
        return list(tools.definitions.values())

    # Validate here so errors never echo caller-supplied secrets or raw arguments.
    @server.call_tool(validate_input=False)  # type: ignore[untyped-decorator]
    async def call_tool(name: str, arguments: dict[str, Any]) -> types.CallToolResult:
        return await tools.call(name, arguments)

    manager = StreamableHTTPSessionManager(
        server,
        stateless=True,
        json_response=True,
        max_request_body_size=64 * 1024,
        security_settings=TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=["127.0.0.1", "127.0.0.1:*", "localhost", "localhost:*"],
            allowed_origins=["http://127.0.0.1:*", "http://localhost:*"],
        ),
    )

    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        try:
            async with manager.run():
                yield
        finally:
            if client is None:
                await upstream.aclose()

    class McpEndpoint:
        async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
            await manager.handle_request(scope, receive, send)

    return Starlette(
        routes=[Route("/mcp", endpoint=McpEndpoint(), methods=["POST", "GET", "DELETE"])],
        lifespan=lifespan,
    )
