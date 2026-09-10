"""Schedule quality is advisory; only explicitly sourced deadlines are hard."""

from __future__ import annotations

import re
from collections import Counter
from datetime import date, time
from typing import TypedDict

from backend.agent.planner.visit_identity import visit_venue_groups
from backend.agent.planner.workspace import server_id
from backend.contracts.itinerary_draft import (
    DraftScheduledActivity,
    DraftScheduledDay,
    DraftScheduledPause,
)
from backend.contracts.v4.planner_draft import WorkingItineraryDay
from backend.contracts.v4.planner_observations import (
    PlannerValidationIssue,
    PlannerValidationObservation,
)
from backend.contracts.v4.planner_refs import CandidateRef
from backend.contracts.v4.planner_workspace import PlannerWorkspaceState
from backend.contracts.v4.task_book import TaskBookV4

# Short transitions are acceptable, not missing activities or repair requests.
MAX_ACCEPTABLE_SCHEDULE_GAP_MINUTES = 15

# Broad start windows, not reservations. Meal goals must be checked even after
# a concrete restaurant has been selected.
MEAL_START_WINDOWS = {
    "breakfast": (6 * 60, 10 * 60),
    "lunch": (11 * 60, 14 * 60),
    "dinner": (17 * 60, 23 * 60 + 30),
    "snack": (14 * 60, 17 * 60),
}
PREFERRED_MEAL_START_WINDOWS = {
    **MEAL_START_WINDOWS,
    "lunch": (11 * 60 + 30, 13 * 60),
    "dinner": (18 * 60, 20 * 60),
}


def meal_time_violations(
    day: WorkingItineraryDay, scheduled: DraftScheduledDay, *, preferred: bool = False
) -> list[dict[str, object]]:
    activities = {str(item.activity_id): item for item in scheduled.activities}
    result: list[dict[str, object]] = []
    for index, item in enumerate(day.ordered_items):
        if item.item_kind != "dining" or item.meal_slot not in MEAL_START_WINDOWS:
            continue
        activity = activities.get(server_id(item.draft_item_id, day.service_date, "activity"))
        if activity is None:
            continue
        windows = PREFERRED_MEAL_START_WINDOWS if preferred else MEAL_START_WINDOWS
        earliest, latest = windows[item.meal_slot]
        actual = minutes(activity.start_time)
        outside = max(earliest - actual, actual - latest, 0)
        if outside:
            result.append(
                {
                    "date": day.service_date.isoformat(),
                    "path": f"days[{day.service_date}].ordered_items[{index}].meal_slot",
                    "draft_item_id": item.draft_item_id,
                    "meal": item.meal_slot,
                    "name": activity.title,
                    "actual_start": activity.start_time.strftime("%H:%M"),
                    "start_window": (
                        f"{earliest // 60:02d}:{earliest % 60:02d}–"
                        f"{latest // 60:02d}:{latest % 60:02d}"
                    ),
                    "outside_minutes": outside,
                }
            )
    pauses = {str(pause.pause_id): pause for pause in scheduled.pauses}
    for item in day.ordered_items:
        if not item.onsite_lunch:
            continue
        pause = pauses.get(server_id(item.draft_item_id, day.service_date, "onsite-lunch"))
        if pause is None:
            continue
        earliest, latest = (PREFERRED_MEAL_START_WINDOWS if preferred else MEAL_START_WINDOWS)[
            "lunch"
        ]
        outside = max(earliest - minutes(pause.start_time), minutes(pause.start_time) - latest, 0)
        if outside:
            result.append(
                {
                    "date": day.service_date.isoformat(),
                    "path": f"days[{day.service_date}].onsite_lunch",
                    "draft_item_id": item.draft_item_id,
                    "meal": "lunch",
                    "name": "园内午餐",
                    "actual_start": pause.start_time.strftime("%H:%M"),
                    "start_window": (
                        f"{earliest // 60:02d}:{earliest % 60:02d}–"
                        f"{latest // 60:02d}:{latest % 60:02d}"
                    ),
                    "outside_minutes": outside,
                }
            )
    return result


def schedule_meal_issues(
    workspace: PlannerWorkspaceState, *, preferred: bool = False
) -> list[dict[str, object]]:
    if workspace.working_itinerary is None or workspace.materialized_schedule is None:
        return []
    days = {item.service_date: item for item in workspace.working_itinerary.days}
    return [
        issue
        for day in workspace.materialized_schedule.days
        for issue in meal_time_violations(days[day.service_date], day, preferred=preferred)
    ]


def has_blocking_timing_issues(workspace: PlannerWorkspaceState) -> bool:
    observation = workspace.validation_observation
    return bool(schedule_meal_issues(workspace)) or bool(
        observation
        and any(
            item.severity != "warning"
            and item.code in {"meal_constraint_violation", "time_overlap", "opening_conflict"}
            for item in observation.issues
        )
    )


def may_keep_intermediate_timing_repair(
    previous: PlannerWorkspaceState, candidate: PlannerWorkspaceState
) -> bool:
    """Keep useful search progress, never grant permission to publish it.

    Remaining late meals must not worsen. An existing opening conflict on an
    unchanged visit may remain while another date is repaired, with the exact
    same time interval and factual source. New/worse errors still veto progress.
    The caller requires a strict quality improvement, and publication separately
    requires all hard errors to be resolved.
    """
    before, after = previous.validation_observation, candidate.validation_observation
    if after is None:
        return False
    if after.result == "passed":
        return True
    if before is None or before.result != "repairable" or after.result != "repairable":
        return False
    errors = [issue for issue in after.issues if issue.severity != "warning"]
    if not errors or any(
        issue.code not in {"meal_constraint_violation", "opening_conflict"} for issue in errors
    ):
        return False

    def error_keys(
        observation: PlannerValidationObservation,
    ) -> Counter[tuple[str, tuple[date, ...], tuple[str, ...], tuple[str, ...]]]:
        return Counter(
            (
                issue.code,
                issue.affected_dates,
                tuple(sorted(ref.canonical_entity_id for ref in issue.candidate_refs)),
                issue.violated_constraint_refs,
            )
            for issue in observation.issues
            if issue.severity != "warning"
        )

    if not error_keys(after) <= error_keys(before):
        return False
    opening_errors = [issue for issue in errors if issue.code == "opening_conflict"]
    for issue in opening_errors:
        if not issue.candidate_refs or not issue.fact_reference_ids:
            return False
        old_issue = next(
            (
                old
                for old in before.issues
                if old.code == issue.code
                and old.affected_dates == issue.affected_dates
                and old.violated_constraint_refs == issue.violated_constraint_refs
                and old.fact_reference_ids == issue.fact_reference_ids
                and {ref.canonical_entity_id for ref in old.candidate_refs}
                == {ref.canonical_entity_id for ref in issue.candidate_refs}
            ),
            None,
        )
        if old_issue is None:
            return False

        def intervals(
            value: PlannerWorkspaceState, source: PlannerValidationIssue
        ) -> tuple[tuple[object, ...], ...]:
            if value.materialized_schedule is None:
                return ()
            identities = {ref.canonical_entity_id for ref in source.candidate_refs}
            place_ids = identities | {server_id("candidate-place", key) for key in identities}
            return tuple(
                (
                    day.service_date,
                    str(activity.place_id),
                    activity.start_time,
                    activity.end_time,
                    activity.duration_minutes,
                )
                for day in value.materialized_schedule.days
                if day.service_date in source.affected_dates
                for activity in day.activities
                if str(activity.place_id) in place_ids
            )

        if not intervals(candidate, issue) or intervals(candidate, issue) != intervals(
            previous, issue
        ):
            return False
    old_meals = {
        (issue["date"], issue["meal"], issue["name"]): int(str(issue["outside_minutes"]))
        for issue in schedule_meal_issues(previous)
    }
    new_meals = schedule_meal_issues(candidate)
    return bool(new_meals or opening_errors) and all(
        int(str(issue["outside_minutes"]))
        <= old_meals.get((issue["date"], issue["meal"], issue["name"]), -1)
        for issue in new_meals
    )


def clock_overflow_minutes(day: DraftScheduledDay) -> int:
    """Catch lost minutes in legacy same-day projections, including the return leg."""
    lost = sum(
        max(0, duration - (minutes(end) - minutes(start)))
        for start, end, duration in (
            *((x.start_time, x.end_time, x.duration_minutes) for x in day.activities),
            *((x.start_time, x.end_time, x.duration_minutes) for x in day.pauses),
            *((x.departure_time, x.arrival_time, x.duration_minutes) for x in day.transport_legs),
        )
    )
    if day.transport_legs:
        last = day.transport_legs[-1]
        lost += max(0, minutes(last.arrival_time) + last.buffer_minutes - minutes(day.end_time))
    return lost


def day_timing_penalty(
    day: WorkingItineraryDay, scheduled: DraftScheduledDay, end: time | None
) -> int:
    meal_penalty = sum(
        int(str(item["outside_minutes"])) for item in meal_time_violations(day, scheduled)
    )
    comfortable_meal_penalty = sum(
        int(str(item["outside_minutes"]))
        for item in meal_time_violations(day, scheduled, preferred=True)
    )
    return (
        meal_penalty * 100
        + comfortable_meal_penalty * 4
        + clock_overflow_minutes(scheduled) * 1000
        + (max(0, minutes(scheduled.end_time) - minutes(end) - 60) * 3 if end else 0)
    )


def may_advance_departure(book: TaskBookV4, day: WorkingItineraryDay) -> bool:
    text = " ".join(
        item.value for item in (*book.hard_constraints, *book.pace_and_transport.pace_preferences)
    )
    return not re.search(
        r"不早于|不能提前|不要早起|最早|(?:必须|只能).{0,12}(?:出发|出门)", text
    ) and not any(not isinstance(item.object_ref, CandidateRef) for item in day.ordered_items)


def permits_taxi_tradeoff(book: TaskBookV4) -> bool:
    text = " ".join(
        item.value
        for item in (
            *book.hard_constraints,
            *book.pace_and_transport.pace_preferences,
            *book.pace_and_transport.transport_preferences,
        )
    )
    return not re.search(
        r"(?:不|别|禁止|拒绝|不能|不要|不用).{0,3}(?:打车|出租|网约车)|(?:只|仅|全程).{0,5}(?:公交|地铁|公共交通)",
        text,
    )


def explicit_end_deadline(book: TaskBookV4, service_date: date) -> tuple[time, str] | None:
    """Recognize explicit end/return deadlines, not defaults or casual preferences.

    Other prose stays in the existing unverified-constraint path. Date-specific
    requirements must never accidentally constrain all the other travel days.
    """
    values = [(f"hard:{index}", item.value) for index, item in enumerate(book.hard_constraints)]
    values += [
        (f"pace:{index}", item.value)
        for index, item in enumerate(book.pace_and_transport.pace_preferences)
    ]
    deadlines = []
    numbers = {
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
        "十一": 11,
        "十二": 12,
    }
    for reference, text in values:
        if is_station_return_requirement(text):
            # Natural-day scope currently has no verified return-to-station
            # anchor. Preserve and disclose it; a restaurant is not a station.
            continue
        strict = re.search(r"最晚|不晚于|不得晚于|必须|不能超过|不得超过|不超过", text)
        if strict is None or re.search(r"不必|不要求|不用|无需|可以超过|允许超过", text):
            continue
        if not re.search(r"结束|收尾|回到|返回|回酒店|回住宿", text):
            continue
        dates = re.findall(r"(\d{1,2})月(\d{1,2})[日号]?", text)
        if dates and (str(service_date.month), str(service_date.day)) not in {
            (str(int(month)), str(int(day))) for month, day in dates
        }:
            continue
        clock_pattern = (
            r"(下午|晚上|晚间|早上|上午)?\s*(\d{1,2}|十一|十二|[一二两三四五六七八九十])"
            r"(?:[：:](\d{2})|点(?:(\d{1,2})分|(?P<half>半))?|时)"
        )
        match = re.search(clock_pattern, text[strict.end() :])
        if match is None:
            match = re.search(
                clock_pattern + r"\s*(?:之前|以前|前)\s*(?:就)?$", text[: strict.start()]
            )
        if not match:
            continue
        hour_text = match[2]
        hour = int(hour_text) if hour_text.isdigit() else numbers[hour_text]
        minute = int(match[3] or match[4] or ("30" if match["half"] else "0"))
        if match[1] in {"下午", "晚上", "晚间"} and hour < 12:
            hour += 12
        # An unqualified "8点" is ambiguous; do not invent an 08:00 hard stop.
        if match[1] is None and hour < 12:
            continue
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            deadlines.append((time(hour, minute), reference))
    return min(deadlines) if deadlines else None


def is_station_return_requirement(text: str) -> bool:
    return bool(re.search(r"(?:回到|返回|赶回|到达|回).{0,16}(?:站|机场)", text))


def natural_day_limitations(book: TaskBookV4) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            f"暂未实现精确返站安排，未保证满足：{item.value[:180]}"
            for item in (*book.hard_constraints, *book.pace_and_transport.pace_preferences)
            if is_station_return_requirement(item.value)
        )
    )


class ScheduleQualityGap(TypedDict):
    date: str
    start: str
    end: str
    minutes: int
    next: str


def has_snack_interest(book: TaskBookV4 | None) -> bool:
    if book is None:
        return False
    text = " ".join(
        item.value
        for item in (*book.dining_direction.preferences, *book.travelers_and_trip_goal.trip_goals)
    )
    return bool(re.search(r"小吃|下午茶|茶馆|咖啡|美食", text))


def schedule_quality_gaps(
    workspace: PlannerWorkspaceState, book: TaskBookV4 | None = None
) -> list[ScheduleQualityGap]:
    """Report gaps over 15 minutes; transit, transfer buffers and meals count.

    Fixed reservations/opening windows may explain a gap. Report that context to
    the Planner rather than deleting or shifting these facts deterministically.
    """
    if workspace.materialized_schedule is None or workspace.working_itinerary is None:
        return []
    output: list[ScheduleQualityGap] = []
    semantic_days = {day.service_date: day for day in workspace.working_itinerary.days}
    for day in workspace.materialized_schedule.days:
        semantic = semantic_days[day.service_date]
        lower, upper = _sightseeing_window(semantic, day, book)
        # Measure the actual departure-to-return timeline. Half-day coverage
        # separately catches an early finish with no afternoon visit; a fully
        # planned day ending at 18:58 does not contain a fictitious 2-minute wait.
        upper = min(upper, minutes(day.end_time))
        if upper <= lower:
            continue
        snack_ids = (
            {
                server_id(item.draft_item_id, day.service_date, "activity")
                for item in semantic.ordered_items
                if item.meal_slot == "snack"
                and item.commitment_level not in {"strong", "immutable"}
            }
            if not has_snack_interest(book)
            else set()
        )
        snack_places = {
            item.place_id for item in day.activities if str(item.activity_id) in snack_ids
        }
        intervals = sorted(
            [
                (item.start_time, item.end_time, item.title)
                for item in day.activities
                if str(item.activity_id) not in snack_ids
            ]
            + [
                (item.start_time, item.end_time, item.reason)
                for item in day.pauses
                if item.kind.value != "rest"
            ]
            + [
                (
                    item.departure_time,
                    time(
                        min(1439, minutes(item.arrival_time) + item.buffer_minutes) // 60,
                        min(1439, minutes(item.arrival_time) + item.buffer_minutes) % 60,
                    ),
                    "交通与衔接",
                )
                for item in day.transport_legs
                if item.origin_place_id not in snack_places
                and item.destination_place_id not in snack_places
            ]
        )
        cursor = time(lower // 60, lower % 60)
        for start, end, label in intervals:
            gap = min(minutes(start), upper) - minutes(cursor)
            if gap > MAX_ACCEPTABLE_SCHEDULE_GAP_MINUTES:
                output.append(
                    {
                        "date": day.service_date.isoformat(),
                        "start": cursor.strftime("%H:%M"),
                        "end": time(
                            min(minutes(start), upper) // 60, min(minutes(start), upper) % 60
                        ).strftime("%H:%M"),
                        "minutes": gap,
                        "next": label,
                    }
                )
            cursor = max(cursor, end)
        # Include any actual uncovered tail before the recorded return/end.
        if upper - minutes(cursor) > MAX_ACCEPTABLE_SCHEDULE_GAP_MINUTES:
            output.append(
                {
                    "date": day.service_date.isoformat(),
                    "start": cursor.strftime("%H:%M"),
                    "end": f"{upper // 60:02d}:{upper % 60:02d}",
                    "minutes": upper - minutes(cursor),
                    "next": "可用游览时段结束",
                }
            )
    return output


def _dated_preferences(book: TaskBookV4, service_date: date) -> str:
    values = []
    for item in (*book.hard_constraints, *book.pace_and_transport.pace_preferences):
        dates = re.findall(r"(\d{1,2})月(\d{1,2})[日号]?", item.value)
        if dates and (service_date.month, service_date.day) not in {
            (int(month), int(day)) for month, day in dates
        }:
            continue
        day_number = re.search(r"第([一二三四五1-5])天", item.value)
        if day_number:
            number = "一二三四五".find(day_number[1]) + 1
            number = number or int(day_number[1])
            if (service_date - book.destination_and_dates.start_date).days + 1 != number:
                continue
        values.append(item.value)
    return " ".join(values)


def _sightseeing_window(
    semantic: WorkingItineraryDay, day: DraftScheduledDay, book: TaskBookV4 | None
) -> tuple[int, int]:
    lower, upper = minutes(day.start_time), max(19 * 60, minutes(day.end_time))
    text = _dated_preferences(book, day.service_date) if book else ""
    if re.search(r"(?:全天|整天|一天).{0,5}(?:休息|自由活动|不安排景点)", text):
        return lower, lower
    if re.search(r"(?:上午|早上).{0,5}(?:休息|自由活动|不安排景点)", text):
        lower = 13 * 60 + 30
    if re.search(r"(?:下午|午后).{0,5}(?:休息|自由活动|不安排景点)", text):
        upper = 12 * 60 + 30
    if book:
        deadline = explicit_end_deadline(book, day.service_date)
        if deadline:
            upper = min(upper, minutes(deadline[0]))
        if not may_advance_departure(book, semantic):
            lower = max(lower, minutes(day.start_time))
    actual = {str(item.activity_id): item for item in day.activities}
    for item in semantic.ordered_items:
        activity = actual.get(server_id(item.draft_item_id, day.service_date, "activity"))
        if activity is None:
            continue
        if item.item_kind == "arrival":
            lower = max(lower, minutes(activity.end_time))
        elif item.item_kind == "departure":
            upper = min(upper, minutes(activity.start_time))
    return lower, max(lower, upper)


def schedule_coverage_issues(
    workspace: PlannerWorkspaceState, book: TaskBookV4 | None = None
) -> list[dict[str, object]]:
    """Only executed visits count, never meal labels, themes, transfers or pauses.

    A short visit straddling lunch cannot masquerade as an all-day attraction.
    Fixed appointments and explicitly requested downtime reduce the usable window.
    An unknown opening time alone is not a reason to waive coverage.
    """
    if workspace.working_itinerary is None or workspace.materialized_schedule is None:
        return []
    output: list[dict[str, object]] = []
    venues = visit_venue_groups(workspace.place_evidence)
    for index, (semantic, day) in enumerate(
        zip(workspace.working_itinerary.days, workspace.materialized_schedule.days, strict=True)
    ):
        lower, upper = _sightseeing_window(semantic, day, book)
        items = {
            server_id(item.draft_item_id, day.service_date, "activity"): item
            for item in semantic.ordered_items
        }
        items.update(
            {
                server_id(item.draft_item_id, day.service_date, "continued-visit"): item
                for item in semantic.ordered_items
                if item.onsite_lunch
            }
        )
        visits = [
            item
            for item in day.activities
            if (source := items.get(str(item.activity_id))) is not None
            and source.item_kind == "visit"
        ]
        reservations = [
            item
            for item in day.activities
            if (source := items.get(str(item.activity_id))) is not None
            and source.item_kind == "fixed_event"
        ]
        covered: set[str] = set()
        meals = {
            source.meal_slot: item
            for item in day.activities
            if (source := items.get(str(item.activity_id))) is not None
            and source.meal_slot in {"lunch", "dinner"}
        }
        lunch: DraftScheduledActivity | DraftScheduledPause | None = meals.get("lunch")
        dinner = meals.get("dinner")
        onsite = next((item for item in semantic.ordered_items if item.onsite_lunch), None)
        if onsite is not None:
            lunch = next(
                (
                    p
                    for p in day.pauses
                    if str(p.pause_id)
                    == server_id(onsite.draft_item_id, day.service_date, "onsite-lunch")
                ),
                None,
            )
        windows = {
            "morning": (lower, minutes(lunch.start_time) if lunch else 12 * 60 + 30),
            "afternoon": (
                minutes(lunch.end_time) if lunch else 13 * 60 + 30,
                max(19 * 60, minutes(dinner.start_time)) if dinner else 19 * 60,
            ),
        }
        groups: dict[str, list[DraftScheduledActivity]] = {}
        period_minutes = {"morning": 0, "afternoon": 0}
        for visit in visits:
            groups.setdefault(venues.get(str(visit.place_id), str(visit.place_id)), []).append(
                visit
            )
        for group in groups.values():
            overlaps = {
                period: sum(
                    max(
                        0, min(minutes(visit.end_time), end) - max(minutes(visit.start_time), start)
                    )
                    for visit in group
                )
                for period, (start, end) in windows.items()
            }
            # Continued visits count their actual minutes in each period, not
            # another venue. The 120-minute check still rejects short straddles.
            for period, overlap in overlaps.items():
                period_minutes[period] += overlap
        covered.update(period for period, overlap in period_minutes.items() if overlap >= 120)
        for period, (start, end) in windows.items():
            start, end = max(start, lower), min(end, upper)
            reserved = sum(
                max(0, min(end, minutes(item.end_time)) - max(start, minutes(item.start_time)))
                for item in reservations
            )
            if end - start - reserved < 120 or period in covered:
                continue
            output.append(
                {
                    "code": "half_day_underfilled"
                    if period_minutes[period]
                    else "half_day_without_visit",
                    "path": f"days[{index}].stops",
                    "date": day.service_date.isoformat(),
                    "period": period,
                    "available_minutes": end - start - reserved,
                    "visit_minutes": period_minutes[period],
                    "minimum_visit_minutes": 120,
                    "actual_visits": [item.title for item in visits],
                    "repair": (
                        "以午餐结束至动态晚餐评估下午；深游现有景点或补入/跨日调整真实沿途景点，"
                        "不以短停靠、休息、绕路代替充足游览。"
                    ),
                }
            )
    return output


def validate_response_visit_claims(text: str, workspace: PlannerWorkspaceState) -> bool:
    """Do not promote a restaurant's neighbourhood or an unused POI into a visit."""
    schedule = workspace.materialized_schedule
    if schedule is None:
        return True
    visits = [
        item.title
        for day in schedule.days
        for item in day.activities
        if item.kind.value == "attraction"
    ]
    for place in workspace.place_evidence:
        if (
            place.entity_kind.value == "attraction"
            and not any(place.display_name in name for name in visits)
            and len(place.display_name) >= 3
            and place.display_name in text
        ):
            return False
    for target in re.findall(
        r"(?:漫步|游览|参观|探访|打卡|逛)([^，。；\s与和及并]+?(?:古镇|景区|路|街))", text
    ):
        if not any(target in name for name in visits):
            return False
    return True


def schedule_travel_minutes(workspace: PlannerWorkspaceState) -> int:
    return sum(
        leg.duration_minutes
        for day in (workspace.materialized_schedule.days if workspace.materialized_schedule else ())
        for leg in day.transport_legs
    )


def has_time_quality_issues(workspace: PlannerWorkspaceState, book: TaskBookV4) -> bool:
    if workspace.working_itinerary is None or workspace.materialized_schedule is None:
        return False
    end = (
        workspace.planning_strategy.daily_capacity_policy.preferred_end_window.latest
        if workspace.planning_strategy
        else None
    )
    return bool(
        any(
            issue.code == "opening_conflict" and issue.severity != "warning"
            for issue in (
                workspace.validation_observation.issues if workspace.validation_observation else ()
            )
        )
        or schedule_coverage_issues(workspace, book)
        or schedule_quality_gaps(workspace, book)
        or missing_concrete_meals(workspace)
        or evening_activity_opportunities(workspace, book)
        or afternoon_activity_opportunities(workspace, book)
        or dining_commute_issues(workspace)
        or any(
            day_timing_penalty(semantic, day, end)
            for semantic, day in zip(
                workspace.working_itinerary.days, workspace.materialized_schedule.days, strict=True
            )
        )
    )


def missing_concrete_meals(workspace: PlannerWorkspaceState) -> list[dict[str, object]]:
    if workspace.working_itinerary is None:
        return []
    return [
        {"date": str(day.service_date), "day_index": index, "meal": meal}
        for index, day in enumerate(workspace.working_itinerary.days, 1)
        for meal in ("lunch", "dinner")
        if not any(
            (item.item_kind == "dining" and item.meal_slot == meal)
            or (meal == "lunch" and item.onsite_lunch)
            for item in day.ordered_items
        )
    ]


def _day_visit_totals(workspace: PlannerWorkspaceState, day: DraftScheduledDay) -> Counter[str]:
    venues = visit_venue_groups(workspace.place_evidence)
    totals: Counter[str] = Counter()
    for activity in day.activities:
        if activity.kind.value == "attraction":
            identity = str(activity.place_id)
            totals[venues.get(identity, identity)] += activity.duration_minutes
    return totals


def afternoon_activity_opportunities(
    workspace: PlannerWorkspaceState, book: TaskBookV4 | None = None
) -> list[ScheduleQualityGap]:
    """Room before a movable dinner, including time hidden by an early finish.

    These are optional scheduling windows, not fabricated gaps in the published
    timeline. Real routes, hours and fixed commitments must still be validated.
    """
    draft, schedule, strategy = (
        workspace.working_itinerary,
        workspace.materialized_schedule,
        workspace.planning_strategy,
    )
    if draft is None or schedule is None or strategy is None:
        return []
    result: list[ScheduleQualityGap] = []
    for semantic, day in zip(draft.days, schedule.days, strict=True):
        totals = _day_visit_totals(workspace, day)
        # Full capacity still permits replacing an adjustable short visit with
        # a longer suitable one; the infill compiler enforces the capacity cap.
        if (totals and max(totals.values()) >= 300) or sum(totals.values()) >= 360:
            continue
        if any(item.item_kind in {"fixed_event", "departure"} for item in semantic.ordered_items):
            continue
        dinner_source = next((x for x in semantic.ordered_items if x.meal_slot == "dinner"), None)
        if (
            dinner_source is None
            or dinner_source.commitment_level == "immutable"
            or dinner_source.expected_window.earliest is not None
            or dinner_source.expected_window.latest is not None
        ):
            continue
        dinner = next(
            (
                x
                for x in day.activities
                if str(x.activity_id)
                == server_id(dinner_source.draft_item_id, day.service_date, "activity")
            ),
            None,
        )
        visits = [
            x
            for x in day.activities
            if x.kind.value == "attraction" and dinner and x.end_time <= dinner.start_time
        ]
        if not visits or dinner is None:
            continue
        start = max(13 * 60, minutes(visits[-1].end_time))
        _, upper = _sightseeing_window(semantic, day, book)
        end = min(upper, PREFERRED_MEAL_START_WINDOWS["dinner"][1])
        if end - start < 105:
            continue
        # Keep the insertion boundary at dinner. The choice budget separately
        # includes dinner's flexibility; it must not append this visit after it.
        boundary = max(start, minutes(dinner.start_time))
        result.append(
            {
                "date": str(day.service_date),
                "start": f"{start // 60:02d}:{start % 60:02d}",
                "end": f"{boundary // 60:02d}:{boundary % 60:02d}",
                "minutes": boundary - start,
                "next": "可用午后游览（晚餐可顺延）",
            }
        )
    return result


def evening_activity_opportunities(
    workspace: PlannerWorkspaceState, book: TaskBookV4 | None = None
) -> list[ScheduleQualityGap]:
    """Optional usable evening, not an assertion of idle time after returning home."""
    draft, schedule, strategy = (
        workspace.working_itinerary,
        workspace.materialized_schedule,
        workspace.planning_strategy,
    )
    if draft is None or schedule is None or strategy is None:
        return []
    target = 3
    result: list[ScheduleQualityGap] = []
    for semantic, day in zip(draft.days, schedule.days, strict=True):
        visits = [item for item in semantic.ordered_items if item.item_kind == "visit"]
        if len(visits) >= min(
            target, strategy.daily_capacity_policy.major_activity_target.maximum
        ) or any(item.item_kind in {"fixed_event", "departure"} for item in semantic.ordered_items):
            continue
        totals = _day_visit_totals(workspace, day)
        if (totals and max(totals.values()) >= 300) or sum(totals.values()) >= 360:
            continue
        dinner_item = next(
            (item for item in semantic.ordered_items if item.meal_slot == "dinner"), None
        )
        dinner = next(
            (
                a
                for a in day.activities
                if dinner_item
                and str(a.activity_id)
                == server_id(dinner_item.draft_item_id, day.service_date, "activity")
            ),
            None,
        )
        if dinner is None or any(
            a.kind.value == "attraction" and a.start_time >= dinner.end_time for a in day.activities
        ):
            continue
        start, end = minutes(dinner.end_time), 22 * 60
        deadline = explicit_end_deadline(book, day.service_date) if book else None
        if deadline:
            end = min(end, minutes(deadline[0]))
        return_leg = day.transport_legs[-1] if day.transport_legs else None
        return_minutes = (
            (return_leg.duration_minutes + return_leg.buffer_minutes) if return_leg else 30
        )
        end -= max(30, return_minutes)
        if start > 20 * 60 + 30 or end - start < 75:
            continue
        result.append(
            {
                "date": str(day.service_date),
                "start": f"{start // 60:02d}:{start % 60:02d}",
                "end": f"{end // 60:02d}:{end % 60:02d}",
                "minutes": end - start,
                "next": "可选晚间活动（不是已发生的空档）",
            }
        )
    return result


def selected_candidate_ids(workspace: PlannerWorkspaceState) -> set[str]:
    if workspace.working_itinerary is None:
        return set()
    return {
        item.object_ref.canonical_entity_id
        for day in workspace.working_itinerary.days
        for item in day.ordered_items
        if isinstance(item.object_ref, CandidateRef)
    }


def protected_time_quality_candidates(workspace: PlannerWorkspaceState) -> set[str]:
    """Preserve user-selected visits and strong meals, not every Agent-selected POI."""
    if workspace.working_itinerary is None:
        return set()
    return {
        item.object_ref.canonical_entity_id
        for day in workspace.working_itinerary.days
        for item in day.ordered_items
        if isinstance(item.object_ref, CandidateRef)
        and (
            item.commitment_level in {"strong", "immutable"}
            or (item.item_kind == "visit" and item.commitment_level == "soft")
        )
    }


class DiningCommuteIssue(TypedDict):
    date: str
    path: str
    name: str
    meal: str | None
    incoming_minutes: int
    incoming_meters: float
    outgoing_minutes: int
    combined_minutes: int
    commitment: str
    draft_item_id: str
    fact_reference_ids: tuple[str, ...]
    candidate_id: str
    suggestion: str


def dining_commute_issues(workspace: PlannerWorkspaceState) -> list[DiningCommuteIssue]:
    """A strong preference is not a reservation or an exemption from route review."""
    if workspace.working_itinerary is None or workspace.materialized_schedule is None:
        return []
    days = {day.service_date: day for day in workspace.working_itinerary.days}
    result: list[DiningCommuteIssue] = []
    for day in workspace.materialized_schedule.days:
        semantic = days[day.service_date]
        restaurants = {
            server_id(item.draft_item_id, day.service_date, "activity"): (index, item)
            for index, item in enumerate(semantic.ordered_items)
            if item.item_kind == "dining"
            and item.commitment_level != "immutable"
            and item.expected_window.earliest is None
            and item.expected_window.latest is None
        }
        for activity in day.activities:
            match = restaurants.get(str(activity.activity_id))
            if match is None:
                continue
            incoming = next(
                (
                    leg
                    for leg in day.transport_legs
                    if leg.destination_place_id == activity.place_id
                ),
                None,
            )
            outgoing = next(
                (leg for leg in day.transport_legs if leg.origin_place_id == activity.place_id),
                None,
            )
            if (
                incoming is None
                or incoming.availability.value != "available"
                or not incoming.source_reference_ids
            ):
                continue
            outgoing_minutes = (
                outgoing.duration_minutes
                if outgoing and outgoing.availability.value == "available"
                else 0
            )
            # A long transfer to the next attraction alone is not evidence
            # that the restaurant is off-route. Review clear remote meals,
            # or meals with long legs on both sides, not ordinary cross-area travel.
            if (
                max(incoming.duration_minutes, outgoing_minutes) <= 45
                and min(incoming.duration_minutes, outgoing_minutes) <= 30
            ):
                continue
            index, item = match
            if not isinstance(item.object_ref, CandidateRef):
                continue
            result.append(
                {
                    "date": day.service_date.isoformat(),
                    "path": f"days[{day.service_date}].ordered_items[{index}]",
                    "name": activity.title,
                    "meal": item.meal_slot,
                    "incoming_minutes": incoming.duration_minutes,
                    "incoming_meters": incoming.distance_m,
                    "outgoing_minutes": outgoing_minutes,
                    "combined_minutes": incoming.duration_minutes + outgoing_minutes,
                    "commitment": item.commitment_level,
                    "draft_item_id": item.draft_item_id,
                    "fact_reference_ids": tuple(
                        dict.fromkeys(
                            (
                                *incoming.source_reference_ids,
                                *(outgoing.source_reference_ids if outgoing else ()),
                            )
                        )
                    ),
                    "candidate_id": item.object_ref.candidate_id,
                    "suggestion": (
                        "比较换餐次或附近餐厅；非预约必吃也允许有依据地取舍，"
                        "保留原意愿，不为一餐牺牲下午游览。"
                    ),
                }
            )
    return result


def schedule_quality_score(workspace: PlannerWorkspaceState, book: TaskBookV4 | None = None) -> int:
    """Compare real improvement, without making the soft preference a hard stop."""
    score = sum(int(gap["minutes"]) for gap in schedule_quality_gaps(workspace, book))
    score += 100_000 * sum(
        issue.code == "opening_conflict" and issue.severity != "warning"
        for issue in (
            workspace.validation_observation.issues if workspace.validation_observation else ()
        )
    )
    score += 10_000 * len(schedule_coverage_issues(workspace, book))
    score += 4_000 * len(missing_concrete_meals(workspace))
    score += 300 * len(evening_activity_opportunities(workspace, book))
    score += 600 * len(afternoon_activity_opportunities(workspace, book))
    # A minute spent travelling costs more than a minute removed from a gap.
    # Otherwise replacing 17 minutes of waiting with 17 minutes on a bus "wins".
    score += 2 * schedule_travel_minutes(workspace)
    if (
        workspace.materialized_schedule is None
        or workspace.planning_strategy is None
        or workspace.working_itinerary is None
    ):
        return score
    policy = workspace.planning_strategy.daily_capacity_policy
    end = policy.preferred_end_window.latest
    days = {item.service_date: item for item in workspace.working_itinerary.days}
    score += sum(
        day_timing_penalty(days[day.service_date], day, end)
        for day in workspace.materialized_schedule.days
    )
    score += sum(
        max(0, int(str(issue["combined_minutes"])) - 45) * 4
        for issue in dining_commute_issues(workspace)
    )
    # A soft pace warning is a cost to weigh, not a veto on every repair that
    # adds a reasonable afternoon visit to an otherwise empty half-day.
    active_target = {"relaxed": 600, "balanced": 660, "intensive": 720, "custom": 600}[
        policy.pace_profile
    ]
    score += sum(
        max(0, day.active_minutes - active_target) for day in workspace.materialized_schedule.days
    )
    return score


def minutes(value: time) -> int:
    return value.hour * 60 + value.minute
