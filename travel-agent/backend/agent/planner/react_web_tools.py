"""Discovered web tools return attributed excerpts, never verified travel facts."""

from __future__ import annotations

import ipaddress
import json
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from backend.agent.model_gateway import ModelToolDefinition
from backend.agent.planner.react_runtime import AgentSession, AgentTool, ToolOutcome
from backend.agent.planner.workspace import PlannerGuardError, advance
from backend.contracts.v4.planner_workspace import PlannerWorkspaceState
from backend.providers.tavily_mcp import TAVILY_READ_TOOLS, TavilySession, WebSearchMcpError


def public_web_url(value: object) -> str | None:
    if not isinstance(value, str) or len(value) > 2048:
        return None
    try:
        parsed = urlsplit(value)
        host = (parsed.hostname or "").rstrip(".").lower()
        if (
            parsed.scheme not in {"http", "https"}
            or not host
            or parsed.username
            or parsed.password
            or parsed.port not in {None, 80, 443}
            or "." not in host
            or host.endswith((".localhost", ".local", ".internal"))
        ):
            return None
        try:
            if not ipaddress.ip_address(host).is_global:
                return None
        except ValueError:
            pass
        # Do not persist signed URLs or accidentally returned credentials.
        if any(
            token in parsed.query.lower()
            for token in ("api_key=", "apikey=", "token=", "signature=", "key=")
        ):
            return None
        return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, parsed.query, ""))
    except ValueError:
        return None


def normalize_web_result(name: str, raw: dict[str, Any]) -> dict[str, Any]:
    results = []
    rows = raw.get("results")
    if not isinstance(rows, list):
        raise PlannerGuardError("web_search_result_invalid")
    for row in rows[:5]:
        if not isinstance(row, dict) or not (url := public_web_url(row.get("url"))):
            continue
        content = row.get("content") if name == "tavily_search" else row.get("raw_content")
        if not isinstance(content, str):
            content = ""
        results.append(
            {
                "url": url,
                "title": str(row.get("title") or "")[:200],
                "excerpt": content[:2000],
                "published_at": row.get("published_date")
                if isinstance(row.get("published_date"), str)
                else None,
                "truncated": len(content) > 2000,
            }
        )
    return {
        "source": "tavily_mcp",
        "observed_at": datetime.now(UTC).isoformat(),
        "results": results,
        "status": "available" if results else "unavailable",
        "evidence_kind": "web_excerpt",
        "notice": "网页摘要是待核对资料，不等于指定日期营业、库存或可预订价格；保留原文来源。",
    }


async def discovered_web_tools(mcp: TavilySession) -> tuple[AgentTool, ...]:
    definitions = await mcp.tools()

    async def call(name: str, args: dict[str, Any], session: AgentSession) -> ToolOutcome:
        receipts = [r for r in session.state.receipts if r.call.function.name in TAVILY_READ_TOOLS]
        known_urls: set[str] = set()
        for receipt in receipts:
            if receipt.status != "completed" or not receipt.result:
                continue
            result = json.loads(receipt.result)
            if receipt.call.function.name == "tavily_search":
                known_urls.update(r["url"] for r in result.get("results", []))
            if (
                receipt.call.function.name == name
                and json.loads(receipt.call.function.arguments) == args
            ):
                observed = datetime.fromisoformat(result["observed_at"])
                if 0 <= (datetime.now(UTC) - observed).total_seconds() < 600:
                    return ToolOutcome({**result, "cached": True})
        if name == "tavily_search":
            if not 1 <= args.get("max_results", 5) <= 5:
                raise PlannerGuardError("web_search_max_results_must_be_1_to_5")
            if len(args.get("query", "")) > 400:
                raise PlannerGuardError("web_search_query_too_long_max_400")
            if args.get("include_raw_content") or args.get("include_images"):
                raise PlannerGuardError("web_search_use_excerpts_then_extract_known_url")
        else:
            if not 1 <= len(args.get("urls", [])) <= 2 or any(
                public_web_url(url) != url or url not in known_urls for url in args.get("urls", [])
            ):
                raise PlannerGuardError("web_extract_requires_1_to_2_prior_search_result_urls")
            if args.get("include_images"):
                raise PlannerGuardError("web_extract_text_only")

        def reserve(workspace: PlannerWorkspaceState) -> PlannerWorkspaceState:
            state = workspace.react_state
            assert state is not None
            if state.web_search_blocked:
                raise PlannerGuardError("web_search_rate_limited_for_this_segment")
            if state.web_search_calls >= 4:
                raise PlannerGuardError("web_search_budget_exhausted")
            return advance(
                workspace,
                react_state=state.model_copy(
                    update={"web_search_calls": state.web_search_calls + 1}
                ),
            )

        await session.update(reserve)
        try:
            async with session.budget.semaphore:
                raw = await mcp.call(name, args)
        except WebSearchMcpError as error:
            if str(error) == "web_search_rate_limited":
                await session.memory(web_search_blocked=True)
            raise PlannerGuardError(str(error)) from None
        result = {"tool": name, **normalize_web_result(name, raw)}
        return ToolOutcome(
            result,
            lambda w: advance(
                w,
                # Excerpts are reviewed evidence, not inputs to the deterministic
                # timeline/cost calculators. Retain that unchanged calculation;
                # an existing review must still be replaced for the new evidence.
                react_state=w.react_state.model_copy(update={"review": None})
                if w.react_state
                else None,
            ),
        )

    def bind(name: str, definition: dict[str, Any]) -> AgentTool:
        async def execute(args: dict[str, Any], session: AgentSession) -> ToolOutcome:
            return await call(name, args, session)

        return AgentTool(
            ModelToolDefinition(
                name=name,
                description=definition.get("description", name),
                parameters=definition["inputSchema"],
            ),
            execute,
            reviewer_allowed=True,
        )

    return tuple(
        bind(name, definition)
        for name, definition in definitions.items()
        if name in TAVILY_READ_TOOLS
    )
