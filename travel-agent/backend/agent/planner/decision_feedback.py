"""Actionable negative observations derived from durable tool receipts."""

import json
import re
from datetime import UTC, datetime
from math import asin, cos, radians, sin, sqrt
from typing import Any

from backend.contracts.v4.planner_react import ToolReceipt
from backend.contracts.v4.planner_workspace import PlannerWorkspaceState

PLACE_SEARCH_TOOLS = {"maps_around_search", "maps_text_search", "maps_polygon_search"}
EMPTY_SEARCH_ACTION = (
    "本次查询没有找到地点，不等于该区域客观上没有景点。停止在同一区域通过微调关键词、"
    "半径或中心点继续试探；先选已有候选、调整游览区域，或说明原因取舍绕路的必吃餐厅。"
)
DUPLICATE_ACTION = (
    "这两个地点不能同时安排；必须移除或替换其中一个，再用独立景点补足空档。"
    "换日期、时段、顺序或理由均不解决重复，勿再次提交相同地点组合。"
)


def repair_action(error: str | None) -> str:
    if (error or "").startswith("planner_plan_same_venue_duplicate"):
        return DUPLICATE_ACTION
    if (error or "").startswith("planner_plan_missing_required_restaurant"):
        return (
            "可安排该餐厅，也可在 capacity_tradeoffs 中说明绕路或不利于整体安排的具体原因后取舍；"
            "两餐仍须安排，不能省略正餐，也不能覆盖用户已明确回答的保留选择。"
        )
    return "修正具体字段；若属于地点组合或可行性问题，改选地点或重排，而非只改理由和措辞。"


def empty_place_search(name: str, result: dict[str, Any]) -> bool:
    return (
        name in PLACE_SEARCH_TOOLS and result.get("status") == "empty" and not result.get("error")
    )


def empty_searches(workspace: PlannerWorkspaceState) -> list[ToolReceipt]:
    now = datetime.now(UTC)
    return [
        r
        for r in (workspace.react_state.receipts if workspace.react_state else ())
        if r.status == "completed"
        and r.result
        and (r.expires_at is None or r.expires_at > now)
        and empty_place_search(r.call.function.name, json.loads(r.result))
    ]


def decision_obstacles(workspace: PlannerWorkspaceState) -> dict[str, Any]:
    conflicts: dict[tuple[str, ...], dict[str, Any]] = {}
    for receipt in workspace.react_state.receipts if workspace.react_state else ():
        if receipt.status != "failed" or not receipt.result:
            continue
        error = json.loads(receipt.result).get("error", "")
        if not error.startswith("planner_plan_same_venue_duplicate:"):
            continue
        match = re.search(r"candidate_keys=([^:]+)", error)
        if match:
            pair = tuple(sorted(match[1].split(",")))
            item = conflicts.setdefault(
                pair, {"candidate_keys": list(pair), "attempts": 0, "next_action": DUPLICATE_ACTION}
            )
            item["attempts"] += 1
    return {
        "rejected_place_combinations": list(conflicts.values()),
        "empty_place_searches": [
            {
                "tool": r.call.function.name,
                "arguments": json.loads(r.call.function.arguments),
                "next_action": EMPTY_SEARCH_ACTION,
            }
            for r in empty_searches(workspace)[-8:]
        ],
    }


def repeated_empty_search(
    name: str, arguments: dict[str, Any], workspace: PlannerWorkspaceState
) -> bool:
    """Do not spend another external call on cosmetic variants of an empty query.

    A new area, specific venue/category, or materially changed filters remain
    available. Provider errors are never treated as an empty search.
    """
    generic = {"景点", "景区", "公园", "风景", "名胜", "旅游景点"}
    for receipt in empty_searches(workspace):
        if receipt.call.function.name != name:
            continue
        previous = json.loads(receipt.call.function.arguments)
        if previous == arguments:
            return True
        if name != "maps_around_search":
            continue

        def filters(a: dict[str, Any]) -> dict[str, Any]:
            return {k: v for k, v in a.items() if k not in {"location", "radius", "keywords"}}

        if filters(previous) != filters(arguments):
            continue
        before = set(str(previous.get("keywords", "")).split())
        after = set(str(arguments.get("keywords", "")).split())
        if (
            not before
            or not after
            or not (before == after or before <= generic and after <= generic)
        ):
            continue
        try:
            lon1, lat1 = map(radians, map(float, previous["location"].split(",")))
            lon2, lat2 = map(radians, map(float, arguments["location"].split(",")))
            distance = 12_742_000 * asin(
                min(
                    1,
                    sqrt(
                        sin((lat2 - lat1) / 2) ** 2
                        + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
                    ),
                )
            )
            same_area = min(
                float(previous.get("radius", 3000)), float(arguments.get("radius", 3000)), 5000
            )
        except (KeyError, TypeError, ValueError):
            continue
        if distance <= same_area:
            return True
    return False
