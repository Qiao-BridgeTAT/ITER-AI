"""A small, evidence-bound choice for a remaining gap, not a whole-plan rewrite."""

from __future__ import annotations

import json
from collections.abc import Mapping
from math import cos, hypot, radians
from typing import Any

from pydantic import Field

from backend.agent.model_gateway import ModelAuditMetadata, ModelMessage, ModelRequest, ModelRole
from backend.agent.planner.decision_contracts import CandidateKey, ModelPlanIntent
from backend.agent.planner.proposals import PlannerReferenceCatalog
from backend.agent.planner.timing_quality import (
    PREFERRED_MEAL_START_WINDOWS,
    afternoon_activity_opportunities,
    evening_activity_opportunities,
    minutes,
    schedule_coverage_issues,
    schedule_quality_gaps,
)
from backend.agent.planner.visit_identity import is_explicit_internal_subsite, visit_venue_groups
from backend.agent.planner.workspace import PlannerGuardError, server_id
from backend.contracts.v4.base import V4ContractModel
from backend.contracts.v4.planner_draft import DraftItem
from backend.contracts.v4.planner_workspace import PlannerWorkspaceState
from backend.contracts.v4.task_book import TaskBookV4


class ModelGapVisit(V4ContractModel):
    gap_key: str = Field(pattern=r"^g[1-9][0-9]*$")
    candidate_key: CandidateKey


class ModelGapVisits(V4ContractModel):
    choices: tuple[ModelGapVisit, ...] = Field(default=(), max_length=9)


def _gap_visit_budget(gap_minutes: int, next_start: int, source: DraftItem | None) -> int:
    """An ordinary meal can move later; an explicit time is not ours to move.

    This is a choice budget, not proof of feasibility. Materialization still
    checks the actual routes, opening hours and immutable commitments.
    """
    if (
        source is None
        or source.meal_slot not in {"lunch", "dinner"}
        or source.expected_window.earliest is not None
        or source.expected_window.latest is not None
        or source.commitment_level == "immutable"
    ):
        return gap_minutes
    return gap_minutes + max(0, PREFERRED_MEAL_START_WINDOWS[source.meal_slot][1] - next_start)


def fits_known_visit_hours(days: list[dict[str, Any]], start: int, minimum: int) -> bool:
    """Filter known impossibilities; the 15-minute allowance is not a route fact."""
    known = [day for day in days if day["status"] in {"open", "closed"}]
    if not known:
        # An unqueried candidate can be proposed for a bounded live lookup.
        # A completed unknown/conflicting lookup is not proof it can fill a gap.
        return not days
    for day in known:
        for interval in day.get("intervals", []):

            def minute(value: str) -> int:
                return int(value[:2]) * 60 + int(value[3:5])

            arrival = max(start + 15, minute(interval["opens_at"]))
            last_entry = interval.get("last_entry_at")
            if arrival + minimum <= minute(interval["closes_at"]) and (
                last_entry is None or arrival <= minute(last_entry)
            ):
                return True
    return False


def gap_choice_hours_failures(
    choices: ModelGapVisits, options: list[dict[str, Any]], workspace: PlannerWorkspaceState
) -> list[dict[str, str]]:
    """A fresh optional visit must survive its actual date-bound hours lookup."""
    catalog = PlannerReferenceCatalog(workspace)
    gaps = {gap["gap_key"]: gap for gap in options}
    failures = []
    for index, choice in enumerate(choices.choices):
        gap = gaps[choice.gap_key]
        option = next(x for x in gap["options"] if x["candidate_key"] == choice.candidate_key)
        if option.get("operation") not in {"add_visit", "replace_existing_visit"}:
            continue
        identity = catalog.candidates[choice.candidate_key].candidate_ref.canonical_entity_id
        if workspace.materialized_schedule and not any(
            str(day.service_date) == gap["date"]
            and any(str(a.place_id) == identity for a in day.activities)
            for day in workspace.materialized_schedule.days
        ):
            continue  # An ignored duplicate was not scheduled on this date.
        if not any(
            day.status == "open" and str(day.service_date) == gap["date"]
            for evidence in workspace.hours_evidence
            if evidence.canonical_entity_id == identity
            for day in evidence.days
        ):
            failures.append(
                {
                    "date": gap["date"],
                    "candidate_key": choice.candidate_key,
                    "path": f"choices[{index}].candidate_key",
                    "reason": (
                        "补查后仍不能确认该景点当天开放；"
                        "换一个有可用营业时间的候选，不能把未知当作开放。"
                    ),
                }
            )
    return failures


def fits_gap_visit_duration(minimum: int | None, maximum: int | None, budget: int) -> bool:
    """Small stops can fill short gaps, not replace a substantive half-day visit."""
    return (
        minimum is not None
        and minimum <= budget - 15
        and not (budget >= 105 and (maximum or 0) < 60)
    )


def feasible_visit_limit(
    maximum: int | None, start: int, opening_days: list[dict[str, Any]]
) -> int | None:
    """Known closing time can reduce an existing limit, never increase it."""
    if maximum is None:
        return None

    def clock_minutes(value: str) -> int:
        return int(value[:2]) * 60 + int(value[3:5])

    closing_limits = [
        clock_minutes(interval["closes_at"]) - start
        for day in opening_days
        if day["status"] == "open"
        for interval in day.get("intervals", [])
        if clock_minutes(interval["opens_at"]) <= start < clock_minutes(interval["closes_at"])
    ]
    return min(maximum, max(closing_limits)) if closing_limits else maximum


def split_free_intervals(
    gap: Mapping[str, Any], occupied: list[tuple[int, int]]
) -> list[dict[str, Any]]:
    """Never treat a broad afternoon opportunity as one empty block."""
    start = int(gap["start"][:2]) * 60 + int(gap["start"][3:5])
    end = int(gap["end"][:2]) * 60 + int(gap["end"][3:5])
    cursor = start
    result = []
    for lower, upper in sorted(occupied):
        if upper <= cursor or lower >= end:
            continue
        if lower > cursor:
            result.append((cursor, min(lower, end)))
        cursor = max(cursor, upper)
    if cursor < end:
        result.append((cursor, end))
    # Zero-size opportunities can still use a flexible meal's later start.
    if start == end:
        result.append((start, end))
    return [
        {
            **gap,
            "start": f"{a // 60:02d}:{a % 60:02d}",
            "end": f"{b // 60:02d}:{b % 60:02d}",
            "minutes": b - a,
        }
        for a, b in result
        if b - a > 15 or start == end
    ]


def gap_visit_options(workspace: PlannerWorkspaceState, book: TaskBookV4) -> list[dict[str, Any]]:
    draft, schedule = workspace.working_itinerary, workspace.materialized_schedule
    if draft is None or schedule is None or workspace.planning_strategy is None:
        return []
    catalog = PlannerReferenceCatalog(workspace)
    places = {x.canonical_entity_id: x for x in workspace.place_evidence}
    venues = visit_venue_groups(workspace.place_evidence)
    selected = {
        str(x.place_id)
        for day in schedule.days
        for x in day.activities
        if x.kind.value == "attraction"
    }
    # A semantic stop can still exist even when materialization could not place
    # it. Re-inserting it would duplicate a key instead of repairing its timing.
    selected.update(
        item.object_ref.canonical_entity_id
        for day in draft.days
        for item in day.ordered_items
        if hasattr(item.object_ref, "canonical_entity_id")
    )
    selected_venues = {venues.get(key, key) for key in selected}
    estimates = {x.canonical_entity_id: x for x in workspace.visit_duration_estimates}
    completed_hours = {
        (ref.canonical_entity_id, str(day))
        for observation in workspace.capability_observations
        if observation.capability == "opening_hours"
        for ref in observation.candidate_refs
        for day in observation.service_dates
    }
    keys = {x.candidate_ref.canonical_entity_id: key for key, x in catalog.candidates.items()}
    result: list[dict[str, Any]] = []
    used_intervals: set[tuple[str, str, str]] = set()
    maximum = workspace.planning_strategy.daily_capacity_policy.major_activity_target.maximum
    coverage = {
        (str(issue["date"]), issue["period"]): issue
        for issue in schedule_coverage_issues(workspace, book)
    }

    def coverage_need(gap: Mapping[str, Any]) -> dict[str, Any] | None:
        period = "morning" if gap["start"] < "12:30" else "afternoon"
        return coverage.get((gap["date"], period)) if gap["start"] < "18:00" else None

    def distance(first: Any, second: Any) -> int:
        return round(
            hypot(
                (first.longitude - second.longitude)
                * cos(radians((first.latitude + second.latitude) / 2))
                * 111320,
                (first.latitude - second.latitude) * 110540,
            )
        )

    # Smaller repairable waits must not hide a useful larger window
    # on the same date from the bounded nearby-choice call.
    raw_gaps = [
        *schedule_quality_gaps(workspace, book),
        *afternoon_activity_opportunities(workspace, book),
        *evening_activity_opportunities(workspace, book),
    ]
    free_gaps = [
        part
        for gap in raw_gaps
        for part in split_free_intervals(
            gap,
            [
                (minutes(a.start_time), minutes(a.end_time))
                for day in schedule.days
                if str(day.service_date) == gap["date"]
                for a in day.activities
            ]
            + [
                (minutes(p.start_time), minutes(p.end_time))
                for day in schedule.days
                if str(day.service_date) == gap["date"]
                for p in day.pauses
            ],
        )
    ]
    for gap in sorted(
        free_gaps,
        key=lambda x: (
            coverage_need(x) is None,
            x["start"] >= "18:00",
            -x["minutes"],
            x["date"],
            x["start"],
        ),
    ):
        interval_key = (gap["date"], gap["start"], gap["end"])
        if interval_key in used_intervals:
            continue
        index = next(
            i for i, day in enumerate(schedule.days) if str(day.service_date) == gap["date"]
        )
        day = schedule.days[index]
        can_add_visit = (
            len({x.node_id for x in day.activities if x.kind.value == "attraction"}) < maximum
        )
        start = int(gap["start"][:2]) * 60 + int(gap["start"][3:])
        end = int(gap["end"][:2]) * 60 + int(gap["end"][3:])
        previous = next((x for x in reversed(day.activities) if minutes(x.end_time) <= start), None)
        following = next((x for x in day.activities if minutes(x.start_time) >= end), None)
        anchor = places.get(str(previous.place_id)) if previous else None
        if anchor is None and following:
            anchor = places.get(str(following.place_id))
        if anchor is None:
            continue
        source_by_activity = {
            server_id(x.draft_item_id, day.service_date, segment): x
            for x in draft.days[index].ordered_items
            for segment in ("activity", "continued-visit")
        }
        source = source_by_activity.get(str(following.activity_id)) if following else None
        visit_budget = _gap_visit_budget(
            gap["minutes"], minutes(following.start_time) if following else end, source
        )
        options: list[dict[str, Any]] = []
        previous_source = source_by_activity.get(str(previous.activity_id)) if previous else None
        previous_estimate = estimates.get(str(previous.place_id)) if previous else None
        previous_entry = (
            catalog.candidates.get(keys.get(str(previous.place_id), "")) if previous else None
        )
        previous_maximum = (
            previous_estimate.maximum_minutes
            if previous_estimate
            else previous_entry.advisory_features.typical_duration_minutes
            if previous_entry
            else None
        )
        previous_limit = (
            feasible_visit_limit(
                previous_maximum,
                minutes(previous.start_time),
                [
                    d.model_dump(mode="json")
                    for h in workspace.hours_evidence
                    if h.canonical_entity_id == str(previous.place_id)
                    for d in h.days
                    if d.service_date == day.service_date
                ],
            )
            if previous
            else None
        )
        previous_total = sum(
            a.duration_minutes
            for a in day.activities
            if previous and a.place_id == previous.place_id and a.kind.value == "attraction"
        )
        replaceable = bool(
            previous
            and previous.kind.value == "attraction"
            and previous_source
            and previous_source.commitment_level != "immutable"
            and previous_source.expected_window.earliest is None
            and previous_source.expected_window.latest is None
            and not previous_source.onsite_lunch
            and previous_limit is not None
            and previous_total >= previous_limit
        )
        for key, entry in catalog.candidates.items():
            identity = entry.candidate_ref.canonical_entity_id
            place = places.get(identity)
            estimate = estimates.get(identity)
            minimum = (
                estimate.minimum_minutes
                if estimate
                else entry.advisory_features.typical_duration_minutes
            )
            maximum_duration = estimate.maximum_minutes if estimate else minimum
            if (
                place is None
                or entry.entity_kind.value != "attraction"
                or entry.selection_permission == "forbidden"
                or entry.eligibility in {"excluded", "unavailable"}
                or identity in selected
                or venues.get(identity, identity) in selected_venues
                or venues.get(identity, identity) != identity
                or is_explicit_internal_subsite(place.display_name, place.provider_parent_place_id)
                or day.service_date in entry.infeasible_dates
                or (entry.feasible_dates and day.service_date not in entry.feasible_dates)
            ):
                continue
            meters = distance(anchor.coordinates, place.coordinates)
            if meters > (8000 if visit_budget >= 180 else 4000):
                continue
            opening_hours = [
                d.model_dump(mode="json")
                for h in workspace.hours_evidence
                if h.canonical_entity_id == identity
                for d in h.days
                if d.service_date == day.service_date
            ]
            if not opening_hours and (identity, str(day.service_date)) in completed_hours:
                continue
            add_fits = can_add_visit and fits_gap_visit_duration(
                minimum, maximum_duration, visit_budget
            )
            replace_fits = (
                replaceable
                and fits_gap_visit_duration(
                    minimum, maximum_duration, visit_budget + previous_total
                )
                and (maximum_duration or 0) > (previous_limit or 0)
            )
            if not add_fits and not replace_fits:
                continue
            assert minimum is not None
            if not add_fits:
                assert previous is not None
            operation = "add_visit" if add_fits else "replace_existing_visit"
            check_start = minutes(previous.start_time) - 15 if not add_fits and previous else start
            if not fits_known_visit_hours(opening_hours, check_start, minimum):
                continue
            options.append(
                {
                    "candidate_key": key,
                    "operation": operation,
                    **(
                        {
                            "replace_candidate_key": keys[str(previous.place_id)],
                            "replaced_visit_minutes": previous_total,
                        }
                        if not add_fits and previous
                        else {}
                    ),
                    "name": place.display_name,
                    "address": place.address,
                    "typecode": place.provider_typecode,
                    "minimum_visit_minutes": minimum,
                    "maximum_visit_minutes": maximum_duration,
                    "straight_line_meters": meters,
                    "rating": place.rating,
                    "opening_hours": opening_hours,
                }
            )
        # Within a comparable walking catchment, quality should matter more
        # than shaving a few metres off a low-rated automatic filler.
        options.sort(
            key=lambda x: (
                # Hours of free time need a meaningful visit, not only the nearest
                # tiny POIs. Travel is still verified before accepting any choice.
                -(min(x["maximum_visit_minutes"], visit_budget) // 60)
                if visit_budget >= 180
                else 0,
                x["straight_line_meters"] // 1000,
                -(x["rating"] or 3.5),
            )
        )
        if not options:
            continue
        used_intervals.add(interval_key)
        result.append(
            {
                "gap_key": f"g{len(result) + 1}",
                "day_index": index + 1,
                **gap,
                "visit_and_travel_budget_minutes": visit_budget,
                "half_day_visit_minutes_still_needed": max(
                    0, int(need["minimum_visit_minutes"]) - int(need["visit_minutes"])
                )
                if (need := coverage_need(gap))
                else 0,
                "maximum_additions": min(
                    3,
                    max(
                        0,
                        maximum
                        - len({x.node_id for x in day.activities if x.kind.value == "attraction"}),
                    ),
                ),
                "previous_place": previous.title if previous else None,
                "next_place": following.title if following else None,
                "insert_after_candidate_key": keys.get(str(previous.place_id))
                if previous
                else None,
                "insert_before_candidate_key": keys.get(
                    getattr(source.object_ref, "canonical_entity_id", "")
                )
                if source
                else None,
                "options": options[:7] + [options[-1]] if len(options) > 8 else options,
            }
        )
    return result


def build_gap_visit_request(
    options: list[dict[str, Any]],
    book: TaskBookV4,
    *,
    feedback: list[dict[str, Any]] | None = None,
    repair_feedback: list[dict[str, Any]] | None = None,
) -> ModelRequest:
    return ModelRequest(
        audit=ModelAuditMetadata(
            stage="planner_schedule_infill",
            node="schedule_gap_choice",
            contract_version="v4-gap-choice-3",
            repair=True,
        ),
        structured_output_mode="json_object",
        max_output_tokens=768,
        temperature_override=0.15,
        messages=[
            ModelMessage(
                role=ModelRole.SYSTEM,
                content=(
                    "你只处理已经测量出的行程空档，不重写整份计划。每个gap可按游览顺序选择多个不同选项，"
                    "新增数量不超过maximum_additions；总停留和实际交通应适配visit_and_travel_budget_minutes。"
                    "优先处理half_day_visit_minutes_still_needed大于0的空上午/空下午，"
                    "选择足够时长的游览或组合，不能只补短停靠而继续让半天游览不足。"
                    "超过三小时的空档优先一个可深游的大景点或两个中等项目，不能只补一个短项目就认为完成。"
                    "程序已在原估时范围内调整；不能重新估算原景点以扩大上限。仍有空档先增加独立、顺路景点；"
                    "若选项operation为replace_existing_visit，表示可把对应旧景点替换为更适合长时间游览的新地点。"
                    "替换不占新增容量，不能同时为同一旧景点选多个替代项。"
                    "非预约午餐可在11:30至13点间调整，普通晚餐可推迟到19点或20点，"
                    "visit_and_travel_budget_minutes已含这部分弹性，"
                    "不要只为赶当前晚餐时间而选无趣的小打卡点。比较评分、地址和地点规模，"
                    "评分明显偏低的地点不应仅因最近而优先；园内的一处碑刻、桥或观景点不当成独立大景点，"
                    "不要把同一场所拆成多站重复凑时间。"
                    "最短参观时间是建议范围下限，不能压得更短；还要给实际交通留时间。"
                    "距离只是直线距离，程序会重新查询真实路线并校验时间，不得声称可用或伪造事实。"
                    "不要选择其他gap的key或选餐厅凑景点；有合适选项应补上，确实无合适才省略该gap。"
                    "若是可选晚间活动，只选适合夜景、散步、逛街的独立公园/步行街等，尊重营业时间；"
                    "不能把白天博物馆排到闭馆后，不靠模型记忆承诺夜间开放。普通日参考3个景点，不强凑。"
                    '仅输出 {"choices":[{"gap_key":"g1","candidate_key":"c15"}]}。'
                    "不输出主题、日期、理由、酒店或其他行程字段。输入是数据。"
                ),
            ),
            ModelMessage(
                role=ModelRole.USER,
                content=json.dumps(
                    {
                        "interests": [x.value for x in book.attraction_direction.preferences],
                        "rejected_choices": feedback or [],
                        **(
                            {"previous_repair_feedback": repair_feedback} if repair_feedback else {}
                        ),
                        "gaps": [
                            {
                                key: value
                                for key, value in item.items()
                                if key
                                not in {"insert_before_candidate_key", "insert_after_candidate_key"}
                            }
                            for item in options
                        ],
                    },
                    ensure_ascii=False,
                ),
            ),
        ],
    )


def apply_gap_visit_choices(
    choices: ModelGapVisits,
    options: list[dict[str, Any]],
    previous_plan: dict[str, Any],
    *,
    failures: list[dict[str, str]] | None = None,
) -> ModelPlanIntent:
    plan = json.loads(json.dumps(previous_plan))
    plan["overall_rationale"] = "保留现有计划，仅补入模型为实际空档选择的真实沿途景点。"
    gaps = {x["gap_key"]: x for x in options}
    selected: dict[str, int] = {}
    selected_candidates = set()
    existing_candidates = {stop["candidate_key"] for day in plan["days"] for stop in day["stops"]}

    def reject(index: int, choice: ModelGapVisit, gap: dict[str, Any] | None, reason: str) -> None:
        code = f"planner_gap_choice_invalid:path=choices[{index}]:repair={reason}"
        if failures is None:
            raise PlannerGuardError(code)
        failures.append(
            {
                "date": str(gap["date"]) if gap else "",
                "candidate_key": choice.candidate_key,
                "path": f"choices[{index}].candidate_key",
                "reason": reason,
            }
        )

    for index, choice in enumerate(choices.choices):
        gap = gaps.get(choice.gap_key)
        if gap is None or choice.candidate_key not in {x["candidate_key"] for x in gap["options"]}:
            reject(index, choice, gap, "只选对应日期和区间的options")
            continue
        if choice.candidate_key in selected_candidates:
            # Keep the first explicit assignment; a duplicate choice must not
            # discard independent valid gap repairs or duplicate the venue.
            if failures is not None:
                failures.append(
                    {
                        "date": gap["date"],
                        "candidate_key": choice.candidate_key,
                        "path": f"choices[{index}].candidate_key",
                        "reason": "此景点已分配给另一缺口；保留该合法安排，为本日改选不同候选。",
                    }
                )
            continue
        stops = plan["days"][gap["day_index"] - 1]["stops"]
        option = next(x for x in gap["options"] if x["candidate_key"] == choice.candidate_key)
        if option.get("operation") == "extend_previous_visit_within_estimated_range":
            existing = next((x for x in stops if x["candidate_key"] == choice.candidate_key), None)
            if existing is None:
                reject(index, choice, gap, "只能在已有估时范围内调整既有景点")
                continue
            existing["duration_preference"] = "extended"
            selected_candidates.add(choice.candidate_key)
            continue
        if choice.candidate_key in existing_candidates:
            if failures is not None:
                failures.append(
                    {
                        "date": gap["date"],
                        "candidate_key": choice.candidate_key,
                        "path": f"choices[{index}].candidate_key",
                        "reason": "该景点已经安排；请选择尚未使用的真实候选。",
                    }
                )
            continue
        if option.get("operation") == "replace_existing_visit":
            position = next(
                (
                    i
                    for i, s in enumerate(stops)
                    if s["candidate_key"] == option["replace_candidate_key"]
                ),
                None,
            )
            if position is None:
                reject(index, choice, gap, "旧景点已替换，请为其他区间选择未使用的候选")
                continue
            stops[position] = {
                **stops[position],
                "candidate_key": choice.candidate_key,
                "duration_preference": "extended",
            }
            existing_candidates.discard(option["replace_candidate_key"])
            existing_candidates.add(choice.candidate_key)
            selected_candidates.add(choice.candidate_key)
            continue
        if selected.get(choice.gap_key, 0) >= gap.get("maximum_additions", 1):
            reject(index, choice, gap, "新增景点不能超过maximum_additions")
            continue
        selected[choice.gap_key] = selected.get(choice.gap_key, 0) + 1
        position = next(
            (
                i
                for i, stop in enumerate(stops)
                if stop["candidate_key"] == gap.get("insert_before_candidate_key")
            ),
            len(stops),
        )
        if position == len(stops) and gap.get("insert_after_candidate_key"):
            position = next(
                (
                    i + selected[choice.gap_key]
                    for i, stop in enumerate(stops)
                    if stop["candidate_key"] == gap["insert_after_candidate_key"]
                ),
                len(stops),
            )
        stops.insert(
            position,
            {
                "candidate_key": choice.candidate_key,
                "part_of_day": "anytime",
                "duration_preference": "normal",
            },
        )
        existing_candidates.add(choice.candidate_key)
        selected_candidates.add(choice.candidate_key)
    return ModelPlanIntent.model_validate(plan)
