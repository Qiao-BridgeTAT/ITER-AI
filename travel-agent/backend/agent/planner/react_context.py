"""Bound model context without deleting durable calls, facts or user constraints."""

import json
import re
from collections import Counter
from typing import Any

from backend.agent.model_gateway import ModelMessage, ModelRole
from backend.agent.planner.candidate_tradeoffs import candidate_answer
from backend.agent.planner.dining_context import dining_candidate_facts
from backend.agent.planner.hotel_status import hotel_query_status
from backend.agent.planner.proposals import PlannerReferenceCatalog
from backend.agent.planner.react_prompts import PROMPT_VERSION
from backend.agent.planner.react_review import evidence_digest, mcp_observations
from backend.agent.planner.timing_quality import (
    REACT_MAX_ACCEPTABLE_SCHEDULE_GAP_MINUTES,
    meal_start_windows,
    pre_dinner_window,
)
from backend.agent.planner.visit_identity import overlapping_visits
from backend.agent.planner.workspace import server_id, service_dates, task_book_references
from backend.contracts.v4.planner_observations import PlannerValidationIssue
from backend.contracts.v4.planner_refs import CandidateRef
from backend.contracts.v4.planner_workspace import PlannerWorkspaceState
from backend.contracts.v4.task_book import TaskBookV4


def compact_candidate_facts(context: dict[str, Any]) -> dict[str, Any]:
    """Factor repeated attributes losslessly; never rank, remove or select a POI."""
    candidates = context.get("candidates", {})
    if len(candidates) < 4:
        return context
    repeated_fields = (
        "eligibility",
        "feasible_dates",
        "infeasible_dates",
        "missing_fact_kinds",
        "typical_duration_minutes",
        "suggested_visit_duration",
        "hours",
        "cuisine",
        "provider_parent_place_id",
        "business_fact_reference_id",
        "business_observed_at",
        "scheduled_on",
        "omission_policy",
        "conflicts_with_scheduled_keys",
    )
    defaults = {}
    for field in repeated_fields:
        if any(field not in item for item in candidates.values()):
            continue
        values = Counter(
            json.dumps(item[field], sort_keys=True, ensure_ascii=False)
            for item in candidates.values()
            if field in item
        )
        if values:
            value, count = values.most_common(1)[0]
            if count >= 4 and count >= 0.75 * len(candidates):
                defaults[field] = json.loads(value)
    if not defaults:
        return context
    return {
        **context,
        "candidate_defaults": defaults,
        "candidate_field_rule": "候选未单列的字段沿用 candidate_defaults；"
        "null 仍是未知，不代表已核实。",
        "candidates": {
            key: {
                field: value
                for field, value in item.items()
                if field not in defaults
                or type(value) is not type(defaults[field])
                or value != defaults[field]
            }
            for key, item in candidates.items()
        },
    }


def agent_context(workspace: PlannerWorkspaceState, book: TaskBookV4) -> dict[str, Any]:
    """Project task, facts and current work, independent of the legacy decision policy.

    Short keys come from the same catalog used by native tools. User semantics,
    evidence provenance and unknown states survive; formal policy/ID bookkeeping
    stays in the checkpoint rather than being presented as another task.
    """
    catalog = PlannerReferenceCatalog(workspace)
    places = {p.canonical_entity_id: p for p in workspace.place_evidence}
    hours = {h.canonical_entity_id: h for h in workspace.hours_evidence}
    durations = {d.canonical_entity_id: d for d in workspace.visit_duration_estimates}
    by_candidate = {c.candidate_ref.candidate_id: k for k, c in catalog.candidates.items()}
    by_entity = {c.candidate_ref.canonical_entity_id: k for k, c in catalog.candidates.items()}
    by_cluster = {c.cluster_id: k for k, c in catalog.clusters.items()}
    by_fixed = {f.commitment_id: k for k, f in catalog.fixed.items()}
    by_hotel = {h.offer_ref.offer_id: k for k, h in catalog.hotels.items()}
    endpoints = by_candidate | by_cluster | by_fixed | by_hotel
    draft_view = draft_model_view(workspace)
    usage: dict[str, list[dict[str, Any]]] = {}
    for day in (draft_view or {}).get("days", []):
        for stop in day["stops"]:
            if key := stop.get("candidate_key"):
                usage.setdefault(key, []).append(
                    {
                        "day_index": day["day_index"],
                        "meal_slot": stop["meal_slot"],
                    }
                )
    candidates: dict[str, dict[str, Any]] = {}
    overlaps = overlapping_visits(workspace.place_evidence)
    conflicts: dict[str, list[str]] = {key: [] for key in catalog.candidates}
    for pair in overlaps:
        if all(entity in by_entity for entity in pair):
            left, right = (by_entity[entity] for entity in pair)
            conflicts[left].append(right)
            conflicts[right].append(left)
    for key, item in catalog.candidates.items():
        entity = item.candidate_ref.canonical_entity_id
        place, estimate = places.get(entity), durations.get(entity)
        answer = candidate_answer(workspace, entity)
        candidates[key] = {
            "name": item.display_name,
            "canonical_entity_id": entity,
            "entity_kind": item.entity_kind.value,
            "commitment": item.commitment_level.value,
            "omission_policy": (
                "not_selectable"
                if item.selection_permission == "forbidden"
                else "must_preserve"
                if answer and answer.semantic_action == "keep_required_candidate"
                else "optional"
                if answer and answer.semantic_action == "omit_required_candidate"
                else "explain_tradeoff"
                if item.commitment_level.value == "strong"
                else "optional"
            ),
            "conflicts_with_scheduled_keys": sorted(k for k in conflicts[key] if k in usage),
            "user_tradeoff": answer.semantic_action if answer else None,
            "selection_permission": (
                "allowed"
                if answer and answer.semantic_action == "omit_required_candidate"
                else item.selection_permission
            ),
            "scheduled_on": usage.get(key, []),
            "eligibility": item.eligibility,
            "clusters": [by_cluster[c] for c in item.cluster_ids],
            "feasible_dates": [d.isoformat() for d in item.feasible_dates],
            "infeasible_dates": [d.isoformat() for d in item.infeasible_dates],
            "missing_fact_kinds": item.missing_fact_kinds,
            "typical_duration_minutes": item.advisory_features.typical_duration_minutes,
            "suggested_visit_duration": estimate.model_dump(
                mode="json", exclude={"canonical_entity_id", "context_fingerprint"}
            )
            if estimate
            else None,
            "hours": hours[entity].model_dump(mode="json") if entity in hours else None,
            **dining_candidate_facts(place),
            **(
                {
                    "provider_entity_id": place.provider_entity_id,
                    "address": place.address,
                    "provider_typecode": place.provider_typecode,
                    "provider_parent_place_id": place.provider_parent_place_id,
                    "rating": place.rating,
                    "business_fact_reference_id": place.business_fact_reference_id,
                    "business_observed_at": place.business_observed_at.isoformat()
                    if place.business_observed_at
                    else None,
                    "coordinates": place.coordinates.model_dump(mode="json"),
                    "mcp_location": f"{place.coordinates.longitude},{place.coordinates.latitude}",
                    "observed_at": place.observed_at.isoformat(),
                    "fact_reference_id": place.fact_reference_id,
                }
                if place
                else {}
            ),
        }
    observation = workspace.hotel_observation
    strategy = workspace.planning_strategy
    state = workspace.react_state
    draft = workspace.working_itinerary
    validation = workspace.validation_observation
    review = state.review if state else None
    booking = book.lodging_direction.existing_booking
    return {
        "prompt_version": PROMPT_VERSION,
        "stage": "langgraph-react-2",
        "schedule_quality_policy": {
            "continuous_idle_target_minutes": REACT_MAX_ACCEPTABLE_SCHEDULE_GAP_MINUTES,
            "is_hard_limit": False,
            "preserve_good_reviewed_periods": True,
        },
        "confirmed_task_book": book.model_dump(
            mode="json",
            exclude={
                "task_book_id",
                "version",
                "based_on_state_version",
                "status",
                "created_at",
                "confirmed_at",
                "source_evidence_refs",
            },
        ),
        "service_dates": [day.isoformat() for day in service_dates(book)],
        "meal_timing_policy": {
            meal: {
                label: [f"{minute // 60:02d}:{minute % 60:02d}" for minute in windows[meal]]
                for label, windows in (
                    ("allowed_start", meal_start_windows(react=True)),
                    ("preferred_start", meal_start_windows(react=True, preferred=True)),
                )
            }
            for meal in ("lunch", "dinner")
        },
        "candidate_supply": {
            "minimum_per_kind": 3 * len(service_dates(book)),
            "available_counts": {
                kind: sum(
                    c["entity_kind"] == kind and c["selection_permission"] != "forbidden"
                    for c in candidates.values()
                )
                for kind in ("attraction", "restaurant")
            },
            "daily_attractions": {
                "usual": 3,
                "range": [2, 4],
                "single_visit_exception": "环球影城、迪士尼等实际游览覆盖全天的大型景点可只排一个；"
                "短时游览、长交通或半天空档不构成该例外。",
            },
            **(
                {
                    "unused_candidate_keys": {
                        kind: [
                            key
                            for key, c in candidates.items()
                            if c["entity_kind"] == kind
                            and not c["scheduled_on"]
                            and c["selection_permission"] != "forbidden"
                        ]
                        for kind in ("attraction", "restaurant")
                    },
                    "independent_unused_attraction_keys": [
                        key
                        for key, c in candidates.items()
                        if c["entity_kind"] == "attraction"
                        and not c["scheduled_on"]
                        and c["selection_permission"] != "forbidden"
                        and not c["conflicts_with_scheduled_keys"]
                    ],
                }
                if draft_view
                else {}
            ),
        },
        "service_calendar": [
            {
                "date": day.isoformat(),
                "iso_weekday": day.isoweekday(),
                "weekday": ("周一", "周二", "周三", "周四", "周五", "周六", "周日")[day.weekday()],
            }
            for day in service_dates(book)
        ],
        # These reference labels are accepted by hotel tools. All user text and
        # its provenance remain in the book; no duplicate serialized book here.
        "task_book_reference_keys": {
            key: ("confirmed_task_book.lodging_direction" if key == "lodging" else value)
            for key, value in task_book_references(book).items()
            if key in {"destination", "party", "lodging", "lodging_budget"}
            or key.startswith(("area:", "facility:"))
        },
        "required_lodging_policy": {
            "mode": "not_applicable"
            if book.lodging_direction.not_applicable
            else "fixed"
            if booking
            else "search",
            "fixed_commitment_key": by_fixed.get(booking.booking_id) if booking else None,
            "observation_status": observation.status if observation else "not_queried",
            "query_status": hotel_query_status(workspace),
            "selection_status": "selected" if catalog.current_hotel_key else "not_selected",
        },
        "required_hotel_stay": {
            "check_in_date": book.destination_and_dates.start_date.isoformat(),
            "check_out_date": book.destination_and_dates.end_date.isoformat(),
            "nights": (
                book.destination_and_dates.end_date - book.destination_and_dates.start_date
            ).days,
        }
        if not book.lodging_direction.not_applicable and booking is None
        else None,
        "calculation_defaults": {
            "daily_capacity": {
                **strategy.daily_capacity_policy.model_dump(mode="json"),
                **({"major_activity_target": {"minimum": 2, "maximum": 4}} if state else {}),
            },
            "transport_preferences": strategy.spatial_policy.preferred_transport_modes,
            "required_meals": [
                m.model_dump(mode="json") for m in strategy.dining_policy.required_meal_windows
            ],
        }
        if strategy
        else None,
        "candidates": candidates,
        "candidate_index": {
            kind: {
                key: item["name"] for key, item in candidates.items() if item["entity_kind"] == kind
            }
            for kind in sorted({item["entity_kind"] for item in candidates.values()})
        },
        "candidate_reference_keys": by_candidate,
        "overlapping_attraction_pairs": sorted(
            sorted(by_entity[entity] for entity in pair)
            for pair in overlaps
            if all(entity in by_entity for entity in pair)
        ),
        "missing_confirmed_entity_ids": workspace.candidate_pool.missing_required_candidate_refs,
        "fixed_commitments": {
            key: ref.model_dump(mode="json") for key, ref in catalog.fixed.items()
        },
        "clusters": {
            key: [by_candidate[r.candidate_id] for r in cluster.candidate_refs]
            for key, cluster in catalog.clusters.items()
        },
        "routes": {
            key: {
                "origin": endpoints.get(route.origin.reference_id, route.origin.reference_id),
                "destination": endpoints.get(
                    route.destination.reference_id, route.destination.reference_id
                ),
                "mode": route.transport_mode,
                "status": route.status,
                "minutes": route.duration_minutes,
                "meters": route.distance_meters,
                "transfer_count": route.transfer_count,
                "fare": route.fare.model_dump(mode="json") if route.fare else None,
                "missing_reason": route.missing_reason,
                "fact_reference_ids": route.fact_reference_ids,
            }
            for key, route in catalog.routes.items()
        },
        "route_comparisons": [r.model_dump(mode="json") for r in workspace.route_comparisons],
        "hotels": {
            key: hotel.model_dump(mode="json", exclude={"offer_ref", "commute_to_clusters"})
            | {
                "commute_to_clusters": [
                    {
                        "cluster_key": by_cluster.get(c.cluster_id),
                        "duration_minutes": c.duration_minutes,
                        "fact_reference_id": c.route_observation_ref,
                    }
                    for c in hotel.commute_to_clusters
                ],
            }
            for key, hotel in catalog.hotels.items()
        },
        "hotel_observation": observation.model_dump(mode="json", exclude={"offers", "scope"})
        if observation
        else None,
        "weather": [w.model_dump(mode="json") for w in workspace.weather_evidence],
        "tickets": [t.model_dump(mode="json") for t in workspace.ticket_evidence],
        "readiness_issues": {
            key: issue.model_dump(mode="json") for key, issue in catalog.issues.items()
        },
        "current_issue_keys": {
            key: validation_issue_view(issue, catalog)
            for key, issue in catalog.validation_issues.items()
        },
        "observations": [
            o.model_dump(
                mode="json",
                include={
                    "capability",
                    "status",
                    "reason_summary",
                    "missing_fact_kinds",
                    "service_dates",
                },
            )
            for o in workspace.capability_observations[-4:]
        ],
        "previous_queries": [
            {
                "capability": o.capability,
                "candidate_keys": [
                    by_candidate[r.candidate_id]
                    for r in o.candidate_refs
                    if r.candidate_id in by_candidate
                ],
                "service_dates": [d.isoformat() for d in o.service_dates],
                "status": o.status,
            }
            for o in workspace.capability_observations
        ],
        "mcp_observations": mcp_model_observations(workspace),
        "long_term_memory": [m.model_dump(mode="json") for m in state.long_term_memories]
        if state
        else [],
        "interaction_answers": [a.model_dump(mode="json") for a in workspace.interaction_answers],
        "change_request": workspace.plan_change_request.model_dump(mode="json")
        if workspace.plan_change_request
        else None,
        "current_draft": draft_view,
        **(
            {
                "execution_scope": (
                    "本轮仅补查路线和重算时间；"
                    "check_plan 或 review_plan 会取得缺失路线并计算，finish_plan 更新正式版本。"
                    "无需修改草稿，地点顺序、游览深度和交通方式保持原样。"
                )
            }
            if state and state.route_refresh_only
            else {}
        ),
        "calculated_schedule": schedule_model_view(workspace) if validation else None,
        "calculated_cost": workspace.cost_draft.model_dump(
            mode="json",
            exclude={
                "algorithm_version",
                "request_id",
                "schedule_request_id",
                "trip_id",
                "input_state_version",
                "task_book_id",
                "task_book_revision",
                "days",
            },
        )
        if workspace.cost_draft and validation
        else None,
        # Each concrete issue appears once, keyed for editing/interaction tools.
        "check_report": validation.model_dump(
            mode="json",
            include={
                "result",
                "draft_revision",
                "validation_fingerprint",
                "checked_at",
                "globally_affected_dates",
            },
        )
        | {"issues_ref": "current_issue_keys"}
        if validation
        else None,
        "review": {
            "draft_revision": review.draft_revision,
            "matches_current_draft": bool(
                draft
                and review.draft_revision == draft.draft_revision
                and review.draft_digest == draft.content_digest
            ),
            "matches_current_evidence": review.evidence_digest == evidence_digest(workspace),
            "matches_current_calculation": bool(
                validation and review.validation_fingerprint == validation.validation_fingerprint
            ),
            "verdict": review.verdict.model_dump(mode="json"),
        }
        if review
        else None,
    }


def schedule_model_view(workspace: PlannerWorkspaceState) -> dict[str, Any] | None:
    """Sum split visits by the actual draft node; never count meals as sightseeing."""
    schedule = workspace.materialized_schedule
    if schedule is None:
        return None
    result = schedule.model_dump(
        mode="json",
        include={
            "status",
            "days",
            "unscheduled_strong_desires",
            "degradation_reasons",
            "provider_fact_ids",
        },
    )
    catalog = PlannerReferenceCatalog(workspace)
    by_candidate = {c.candidate_ref.candidate_id: key for key, c in catalog.candidates.items()}
    nodes = (
        {
            server_id(item.draft_item_id, "node"): by_candidate[item.object_ref.candidate_id]
            for day in workspace.working_itinerary.days
            for item in day.ordered_items
            if isinstance(item.object_ref, CandidateRef)
        }
        if workspace.working_itinerary
        else {}
    )
    result["visit_duration_basis"] = (
        "visit_totals 是同一次游览各分段的纯游览分钟合计，不含午餐或交通；"
        "activities 保留每段明细，午餐后的继续游览也计入总时长。"
    )
    semantic_days = (
        {str(day.service_date): day for day in workspace.working_itinerary.days}
        if workspace.working_itinerary
        else {}
    )
    if workspace.react_state is not None:
        result["pre_dinner_window_basis"] = (
            "pre_dinner_window 从晚餐前最后一项结束算至不同用餐上限，包含交通和缓冲，"
            "不是净空档或已核验的游览容量；还须核对营业、固定安排、抵离及休息要求。"
            "允许范围内顺延不算拖延正餐，当前最早排出的晚餐时间不是游览截止点。"
        )
    scheduled_days = {str(day.service_date): day for day in schedule.days}
    for day in result["days"]:
        visits: dict[str, dict[str, Any]] = {}
        for activity in day["activities"]:
            if activity["kind"] != "attraction":
                continue
            node = activity["node_id"]
            visit = visits.setdefault(
                node,
                {
                    "candidate_key": nodes.get(node),
                    "title": activity["title"],
                    "total_visit_minutes": 0,
                    "segment_minutes": [],
                },
            )
            visit["total_visit_minutes"] += activity["duration_minutes"]
            visit["segment_minutes"].append(activity["duration_minutes"])
        day["visit_totals"] = list(visits.values())
        if workspace.react_state is not None and day["service_date"] in semantic_days:
            day["pre_dinner_window"] = pre_dinner_window(
                semantic_days[day["service_date"]],
                scheduled_days[day["service_date"]],
                react=True,
            )
    return result


def validation_issue_view(
    issue: PlannerValidationIssue, catalog: PlannerReferenceCatalog
) -> dict[str, Any]:
    """Replace repeated versioned references with the tool catalog's stable keys."""
    candidates = {c.candidate_ref.candidate_id: k for k, c in catalog.candidates.items()}
    hotels = {h.offer_ref.offer_id: k for k, h in catalog.hotels.items()}
    routes = {r.route_edge_id: k for k, r in catalog.routes.items()}
    result = issue.model_dump(
        mode="json", exclude={"issue_id", "candidate_refs", "hotel_offer_refs", "route_edge_ids"}
    )
    result["candidate_keys"] = [
        candidates.get(r.candidate_id, r.candidate_id) for r in issue.candidate_refs
    ]
    result["hotel_keys"] = [hotels.get(r.offer_id, r.offer_id) for r in issue.hotel_offer_refs]
    result["route_keys"] = [routes.get(r, r) for r in issue.route_edge_ids]
    return result


def mcp_model_observations(workspace: PlannerWorkspaceState) -> list[dict[str, Any]]:
    """Share queries once; normalized candidates replace repeated search payloads.

    Hotel POIs and unbound routes remain available for branch matching and
    comparisons. Full original results still bind the review and checkpoint.
    """
    catalog = PlannerReferenceCatalog(workspace)
    by_entity = {c.candidate_ref.canonical_entity_id: k for k, c in catalog.candidates.items()}
    result = []
    for observation in mcp_observations(workspace):
        raw = observation["result"]
        item = {"tool": observation["tool"], "arguments": observation["arguments"]}
        item.update(
            {
                k: v
                for k, v in raw.items()
                if k not in {"tool", "arguments", "data", "admitted_places"}
            }
        )
        if "admitted_places" in raw:
            admitted = raw["admitted_places"]
            item["candidate_keys"] = [
                by_entity[p["canonical_entity_id"]]
                for p in admitted
                if p["canonical_entity_id"] in by_entity
            ]
            admitted_ids = {
                p["provider_entity_id"] for p in admitted if p["canonical_entity_id"] in by_entity
            }
            # Non-candidate POIs (especially hotels) are still evidence, not
            # selectable hotel products. Never promote a place to a priced offer.
            item["other_places"] = [
                p
                for p in raw.get("data", {}).get("places", [])
                if p.get("source_place_id") not in admitted_ids
            ]
            item["candidate_details_ref"] = "candidates"
        elif "normalized_routes" not in raw:
            item["data"] = raw.get("data")
        result.append(model_projection(item))
    return result


def duplicate_candidate_issues(code: str, arguments: dict[str, Any]) -> list[dict[str, Any]]:
    """Locate a rejected reference without selecting which occurrence to change."""
    match = re.fullmatch(r"planner_plan_duplicate_candidate:(c[1-9]\d*)", code)
    if match is None:
        return []
    candidate = match.group(1)
    return [
        {
            "loc": ["days", day_index, "stops", stop_index, "candidate_key"],
            "type": "duplicate_candidate",
            "candidate_key": candidate,
            "expected": "每个候选全程只安排一次；自行保留一处，其余位置选择未使用候选或补搜",
        }
        for day_index, day in enumerate(arguments.get("days", []))
        for stop_index, stop in enumerate(day.get("stops", []))
        if stop.get("candidate_key") == candidate
    ]


def edit_candidate_feedback(
    code: str,
    arguments: dict[str, Any],
    workspace: PlannerWorkspaceState,
) -> list[dict[str, Any]]:
    """Explain a rejected insertion using the saved draft and still-unused choices."""
    match = re.match(r"edit_candidate_already_scheduled:(c[1-9]\d*):", code)
    if match is None:
        return []
    key = match.group(1)
    catalog = PlannerReferenceCatalog(workspace)
    candidate = catalog.candidates.get(key)
    draft = draft_model_view(workspace)
    if candidate is None or draft is None:
        return []
    used = {stop.get("candidate_key") for day in draft["days"] for stop in day["stops"]}
    existing = [
        {"day_index": day["day_index"], "meal_slot": stop["meal_slot"]}
        for day in draft["days"]
        for stop in day["stops"]
        if stop.get("candidate_key") == key
    ]
    alternatives = [
        {"key": k, "name": c.display_name}
        for k, c in catalog.candidates.items()
        if c.entity_kind is candidate.entity_kind
        and k not in used
        and c.selection_permission != "forbidden"
    ]
    return [
        {
            "loc": ["edits", index, "stop", "candidate_key"],
            "type": "candidate_already_scheduled",
            "candidate": {"key": key, "name": candidate.display_name, "scheduled_on": existing},
            "unused_same_kind_candidates": alternatives,
            "expected": "这批修改整体未保存，原草稿保持不变。补另一餐请选尚未使用的餐厅；"
            "更改这家餐厅原有餐次用 update_stop，并安排原餐次留下的缺口；"
            "移动地点须先 remove_stop 再 insert_stop。",
        }
        for index, edit in enumerate(arguments.get("edits", []))
        if edit.get("action") == "insert_stop" and edit.get("stop", {}).get("candidate_key") == key
    ]


def candidate_argument_feedback(
    tool_name: str,
    arguments: dict[str, Any],
    issues: list[dict[str, Any]],
    workspace: PlannerWorkspaceState,
) -> list[dict[str, Any]]:
    """Explain a rejected stop using its verified identity, never repair it by coercion."""
    if tool_name not in {"write_plan", "edit_plan", "patch_plan"}:
        return issues
    catalog = PlannerReferenceCatalog(workspace)
    result = []
    for issue in issues:
        current: Any = arguments
        candidate = None
        for part in issue.get("loc", []):
            if isinstance(current, dict):
                key = current.get("candidate_key")
                if isinstance(key, str) and key in catalog.candidates:
                    candidate = key
                current = current.get(part)
            elif isinstance(current, list) and isinstance(part, int) and 0 <= part < len(current):
                current = current[part]
            else:
                break
        detail = dict(issue)
        if candidate:
            entry = catalog.candidates[candidate]
            detail["candidate"] = {
                "key": candidate,
                "name": entry.display_name,
                "entity_kind": entry.entity_kind.value,
            }
            detail["repair"] = (
                "此编号对应上述已核验地点。若要安排该餐厅，选择午餐或晚餐；"
                "若原意是游览景点，改用 candidate_index 中相应景点的编号，缺少时先搜索。"
                if entry.entity_kind.value == "restaurant"
                else "按上述地点身份修正字段；若原意是另一个地点，从 candidate_index 选择正确编号。"
            )
        result.append(detail)
    return result


def draft_model_view(workspace: PlannerWorkspaceState) -> dict[str, Any] | None:
    """Expose choices using the same short keys accepted by editing tools.

    Keep exact windows, commitments and per-leg mode choices. Runtime IDs and
    automatically compiled policy structures stay in the durable draft.
    """
    draft = workspace.working_itinerary
    if draft is None:
        return None
    catalog = PlannerReferenceCatalog(workspace)
    candidates = {
        entry.candidate_ref.candidate_id: key for key, entry in catalog.candidates.items()
    }
    fixed = {ref.commitment_id: key for key, ref in catalog.fixed.items()}
    days = []
    for index, day in enumerate(draft.days, 1):
        stops = []
        for item in day.ordered_items:
            ref = item.object_ref
            key = (
                candidates[ref.candidate_id]
                if isinstance(ref, CandidateRef)
                else fixed[ref.commitment_id]
            )
            stops.append(
                {
                    "candidate_key"
                    if isinstance(ref, CandidateRef)
                    else "fixed_commitment_key": key,
                    "kind": item.item_kind,
                    "expected_window": item.expected_window.model_dump(mode="json"),
                    "meal_slot": item.meal_slot,
                    "duration_preference": item.duration_preference,
                    "onsite_lunch": item.onsite_lunch,
                    "commitment": item.commitment_level,
                }
            )
        days.append(
            {
                "day_index": index,
                "service_date": day.service_date.isoformat(),
                "theme": day.day_theme,
                "day_kind": day.day_kind,
                "stops": stops,
                "meal_coverage": {
                    meal: next(
                        (
                            {
                                "mode": "onsite" if stop["onsite_lunch"] else "restaurant",
                                "stop_key": stop.get("candidate_key")
                                or stop.get("fixed_commitment_key"),
                            }
                            for stop in stops
                            if stop["meal_slot"] == meal or meal == "lunch" and stop["onsite_lunch"]
                        ),
                        {"mode": "unassigned", "stop_key": None},
                    )
                    for meal in ("lunch", "dinner")
                },
                "visit_count": sum(item.item_kind == "visit" for item in day.ordered_items),
                "dining_goals": day.dining_goals,
                "transport_preferences": day.transport_preferences,
                "start_time": day.start_time.isoformat() if day.start_time else None,
                "end_time": day.end_time.isoformat() if day.end_time else None,
                "route_mode_selections": [
                    item.model_dump(mode="json") for item in day.route_mode_selections
                ],
            }
        )
    return {
        "draft_revision": draft.draft_revision,
        "content_digest": draft.content_digest,
        "days": days,
        "selected_hotel_key": catalog.current_hotel_key,
        "lodging_baseline": draft.lodging_baseline.model_dump(mode="json"),
        "unassigned_intents": [item.model_dump(mode="json") for item in draft.unassigned_intents],
    }


def model_projection(value: Any) -> Any:
    """Omit program-owned scope bookkeeping and display geometry, never evidence values."""
    if isinstance(value, dict):
        return {
            key: model_projection(item)
            for key, item in value.items()
            if key
            not in {
                "scope",
                "polyline",
                "raw_payload",
                "schema_version",
                "based_on_task_book_id",
                "based_on_task_book_version",
                "workspace_revision",
            }
        }
    if isinstance(value, (tuple, list)):
        return [model_projection(item) for item in value]
    return value


def meal_conflict_feedback(code: str | None, workspace: PlannerWorkspaceState) -> dict[str, Any]:
    """Ground an atomic edit failure in the saved plan, including historical receipts."""
    if not code or not code.startswith("planner_plan_duplicate_main_meal:path=days["):
        return {}
    try:
        index = int(code.split("days[", 1)[1].split("]", 1)[0]) + 1
    except (ValueError, IndexError):
        return {}
    draft = draft_model_view(workspace)
    if draft is None:
        return {}
    day = next((day for day in draft["days"] if day["day_index"] == index), None)
    if day is None:
        return {}
    return {
        "day_index": index,
        "saved_meals": [
            {"candidate_key": stop["candidate_key"], "meal_slot": stop["meal_slot"]}
            for stop in day["stops"]
            if stop.get("candidate_key") and stop["meal_slot"] in {"lunch", "dinner"}
        ],
        "repair": "整批编辑失败，当前草稿未改变。insert_stop 不会替换已有餐厅。"
        "更换同日同餐次时，在同一批中先 remove_stop 旧餐厅再插入新餐厅；"
        "若保留旧餐厅，删除新增的重复餐次。不要原样重交整批编辑。",
    }


def recent_dialogue(
    messages: tuple[ModelMessage, ...], *, workspace: PlannerWorkspaceState | None = None
) -> tuple[ModelMessage, ...]:
    """Keep four complete action/result batches; facts are supplied in current state.

    This is a request-only projection. Full original messages and receipts remain
    in the checkpoint for recovery, audit and cross-agent evidence reuse.
    """
    starts = [index for index, message in enumerate(messages) if message.tool_calls]
    if len(starts) > 4:
        cutoff = starts[-4]
        # Retain explicit follow-up text, including edit constraints that may be
        # more specific than the structured change request.
        selected = (
            tuple(m for m in messages[:cutoff] if m.role is ModelRole.USER) + messages[cutoff:]
        )
    else:
        selected = messages
    names = {call.id: call.function.name for m in messages for call in m.tool_calls}
    result = []
    for message in selected:
        if message.role is ModelRole.TOOL:
            value = model_projection(json.loads(message.content or "{}"))
            name = names.get(message.tool_call_id or "", "")
            if name.startswith("maps_") and isinstance(value, dict):
                value = {
                    k: v
                    for k, v in value.items()
                    if k
                    not in {"data", "admitted_places", "normalized_routes", "tool", "arguments"}
                }
                value["current_facts"] = (
                    "mcp_observations；已入池地点详见 candidates，绑定路线见 routes"
                )
            elif (
                name.startswith("tavily_")
                and isinstance(value, dict)
                and workspace is not None
                and workspace.react_state is not None
                and any(
                    r.call.id == message.tool_call_id and r.status == "completed"
                    for r in workspace.react_state.receipts
                )
            ):
                value = {k: v for k, v in value.items() if k != "results"}
                value["current_facts"] = "网页来源、摘录和查询条件完整见当前 mcp_observations。"
            elif name in {
                "lookup_hours",
                "lookup_weather",
                "lookup_tickets",
                "lookup_routes",
                "search_hotels",
                "refresh_hotel",
                "search_places",
                "lookup_place",
            } and isinstance(value, dict):
                value = {
                    k: v
                    for k, v in value.items()
                    if k not in {"places", "hours", "weather", "tickets", "hotel", "route_edges"}
                }
                value["current_facts"] = (
                    "当前 candidates、weather、tickets、hotels、hotel_observation、routes"
                )
            elif name == "review_plan" and isinstance(value, dict) and workspace is not None:
                review = value.get("review")
                draft = workspace.working_itinerary
                if isinstance(review, dict) and draft is not None:
                    same_draft = (
                        review.get("draft_revision") == draft.draft_revision
                        and review.get("draft_digest") == draft.content_digest
                    )
                    if not same_draft:
                        value = {
                            "reviewed_draft_revision": review.get("draft_revision"),
                            "superseded_by_draft_revision": draft.draft_revision,
                            "note": "此为旧版评审，草稿已修订；"
                            "不要把旧问题当作新版仍存在的问题。新版须重新评审。",
                        }
                    elif workspace.react_state and workspace.react_state.review:
                        value = {
                            "reviewed_draft_revision": review.get("draft_revision"),
                            "current_review_ref": "review",
                            "note": "当前评审结论、具体问题与版本匹配情况见当前状态 review。",
                        }
            elif name == "check_plan" and isinstance(value, dict):
                # The current calculation is included once in state. Keep past
                # check verdicts without replaying superseded full timetables.
                value.pop("schedule", None)
                value.pop("cost", None)
                report = value.get("validation")
                if isinstance(report, dict):
                    value["validation"] = {
                        k: v
                        for k, v in report.items()
                        if k in {"result", "draft_revision", "checked_at"}
                    }
                    value["issue_codes"] = list(
                        dict.fromkeys(i["code"] for i in report.get("issues", []))
                    )
                value["current_calculation"] = (
                    "参见当前状态的 calculated_schedule、calculated_cost、"
                    "current_issue_keys 与 check_report"
                )
            result.append(
                message.model_copy(
                    update={"content": json.dumps(value, ensure_ascii=False, separators=(",", ":"))}
                )
            )
        elif message.role is ModelRole.ASSISTANT and len(message.content or "") > 300:
            result.append(
                message.model_copy(
                    update={
                        "content": (message.content or "")[:300]
                        + "[此前正文省略，工具调用完整保留]"
                    }
                )
            )
        else:
            result.append(message)
    return tuple(result)
