"""Bind existing draft parameter rules to the current verified reference catalog."""

from copy import deepcopy
from typing import Any

from backend.agent.model_gateway import ModelToolDefinition
from backend.agent.planner.proposals import PlannerReferenceCatalog
from backend.contracts.v4.enums import CandidateEntityKind
from backend.contracts.v4.planner_workspace import PlannerWorkspaceState


def bind_plan_tool_schemas(
    tools: tuple[ModelToolDefinition, ...], workspace: PlannerWorkspaceState
) -> tuple[ModelToolDefinition, ...]:
    """Constrain program-known IDs/types; never choose places, order or a hotel."""
    catalog = PlannerReferenceCatalog(workspace)
    candidates = {
        key: entry
        for key, entry in catalog.candidates.items()
        if entry.selection_permission != "forbidden"
    }
    restaurants = [
        key
        for key, entry in candidates.items()
        if entry.entity_kind is CandidateEntityKind.RESTAURANT
    ]
    attractions = [
        key
        for key, entry in candidates.items()
        if entry.entity_kind is CandidateEntityKind.ATTRACTION
    ]
    others = [key for key in candidates if key not in restaurants]
    result = []
    for tool in tools:
        # Official MCP schemas are not rewritten. This projects the draft
        # compiler's existing parameter constraints into our own write tool.
        if tool.name == "search_hotels":
            schema = deepcopy(tool.parameters)
            schema["properties"] = {
                k: v
                for k, v in schema["properties"].items()
                if k in {"search_keyword", "activity_cluster_keys"}
            }
            schema["required"] = ["activity_cluster_keys"]
            clusters = list(catalog.clusters)
            if clusters:
                schema["properties"]["activity_cluster_keys"]["items"]["enum"] = clusters
            result.append(tool.model_copy(update={"parameters": schema}))
            continue
        if tool.name not in {"write_plan", "edit_plan"}:
            result.append(tool)
            continue
        schema = deepcopy(tool.parameters)
        definitions = schema.get("$defs", {})
        tradeoff = definitions.get("AgentCapacityTradeoff")
        if tradeoff is not None:
            keys = [
                key
                for key, entry in candidates.items()
                if entry.entity_kind
                in {CandidateEntityKind.ATTRACTION, CandidateEntityKind.RESTAURANT}
                and entry.commitment_level.value in {"strong", "soft"}
            ]
            if keys:
                tradeoff["properties"]["candidate_key"]["enum"] = keys
            else:
                schema["properties"]["capacity_tradeoffs"]["maxItems"] = 0
        stop = definitions.get("AgentPlanStop")
        if stop is not None:
            branches = []
            for keys, needs_meal, allows_onsite in (
                (others, False, True),
                (restaurants, True, False),
            ):
                if not keys:
                    continue
                branch = deepcopy(stop)
                branch["properties"]["candidate_key"]["enum"] = keys
                if needs_meal:
                    meal = branch["properties"]["meal_slot"]
                    meal["enum"] = ["lunch", "dinner"]
                    meal.pop("default", None)
                    branch["required"] = list(dict.fromkeys([*branch["required"], "meal_slot"]))
                else:
                    branch["properties"]["meal_slot"] = {"type": "null", "default": None}
                if not allows_onsite:
                    branch["properties"]["onsite_lunch"]["const"] = False
                branches.append(branch)
            if branches:
                definitions["AgentPlanStop"] = {"anyOf": branches}
            elif "AgentPlanDay" in definitions:
                definitions["AgentPlanDay"]["properties"]["stops"]["maxItems"] = 0
        estimate = definitions.get("AgentVisitDuration")
        if estimate is not None:
            schema["required"] = list(
                dict.fromkeys([*schema["required"], "visit_duration_estimates"])
            )
            schema["properties"]["visit_duration_estimates"].pop("default", None)
            if attractions:
                estimate["properties"]["candidate_key"]["enum"] = attractions
            else:
                schema["properties"]["visit_duration_estimates"]["maxItems"] = 0
        strategy = workspace.planning_strategy
        if strategy is not None and "selected_hotel_key" in schema["properties"]:
            selectable = (
                list(catalog.selectable_hotels) if strategy.lodging_policy.mode == "search" else []
            )
            hotel: dict[str, Any] = {
                "description": schema["properties"]["selected_hotel_key"].get("description", "")
            }
            if selectable:
                hotel.update(type="string", enum=selectable)
                schema["required"] = list(
                    dict.fromkeys([*schema["required"], "selected_hotel_key"])
                )
            else:
                hotel.update(type="null", default=None)
            schema["properties"]["selected_hotel_key"] = hotel
        result.append(tool.model_copy(update={"parameters": schema}))
    return tuple(result)
