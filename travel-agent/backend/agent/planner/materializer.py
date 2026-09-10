"""Deterministically compile one semantic V4 draft into schedule and cost drafts."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from decimal import Decimal
from math import ceil, floor
from typing import Literal, overload
from uuid import UUID

from backend.agent.model_gateway import ModelCancellation
from backend.agent.planner.timing_quality import (
    MEAL_START_WINDOWS,
    PREFERRED_MEAL_START_WINDOWS,
    day_timing_penalty,
    explicit_end_deadline,
    may_advance_departure,
    meal_time_violations,
    permits_taxi_tradeoff,
)
from backend.agent.planner.workspace import PlannerGuardError, server_id
from backend.contracts.common import CnyAmountRange
from backend.contracts.cost_estimation import (
    CostCoverageStatus,
    CostPriceBasis,
    CostSubjectKind,
)
from backend.contracts.daily_scheduling import ScheduleActivityKind, SchedulePauseKind
from backend.contracts.enums import (
    AnchorRole,
    CostCategory,
    DataAvailability,
    ExcludedCostKind,
)
from backend.contracts.itinerary_draft import (
    CostValidationDraft,
    DraftAmountRange,
    DraftCategoryCostSummary,
    DraftCostEstimateLine,
    DraftDailyCostEstimate,
    DraftScheduledActivity,
    DraftScheduledDay,
    DraftScheduledPause,
    DraftScheduledTransport,
    DraftUnscheduledStrongDesire,
    ScheduleValidationDraft,
)
from backend.contracts.v4.enums import CandidateEntityKind
from backend.contracts.v4.planner_draft import (
    DraftItem,
    DraftRouteModeSelection,
    WorkingItineraryDay,
)
from backend.contracts.v4.planner_evidence import (
    PlannerHoursEvidence,
    PlannerPlaceEvidence,
    PlannerVisitDurationEstimate,
)
from backend.contracts.v4.planner_observations import SpatialRouteEdge, SpatialRouteEndpoint
from backend.contracts.v4.planner_refs import CandidateRef, FixedCommitmentRef
from backend.contracts.v4.planner_strategy import CandidatePoolEntry
from backend.contracts.v4.planner_workspace import PlannerWorkspaceState
from backend.contracts.v4.task_book import TaskBookV4
from backend.domain.party_size import parse_party_size
from backend.providers.contracts import HoursDayStatus, RouteMode

MATERIALIZER_VERSION = "4.0.8"
COST_COMPILER_VERSION = "4.0.2"
MAX_PARALLEL_DAYS = 4
MISSING_ROUTE_PLACEHOLDER_MINUTES = 45
MISSING_ROUTE_BUFFER_MINUTES = 10
TransportPreference = Literal["public_transit", "taxi", "walking", "driving"]

_PART_OF_DAY_START = {
    "morning": 9 * 60,
    "midday": 11 * 60 + 30,
    "afternoon": 13 * 60 + 30,
    "evening": 17 * 60 + 30,
    "anytime": 9 * 60,
}
_MEAL_START = {
    "breakfast": 8 * 60,
    "lunch": 12 * 60,
    "dinner": 19 * 60,
    "snack": 15 * 60 + 30,
}
_MEAL_LATEST = {meal: window[1] for meal, window in MEAL_START_WINDOWS.items()}
_MEAL_DURATION = {"breakfast": 45, "lunch": 60, "dinner": 60, "snack": 30}
_CATEGORY_ORDER = tuple(CostCategory)


@dataclass(frozen=True)
class PlannerMaterialization:
    schedule: ScheduleValidationDraft
    cost: CostValidationDraft


@dataclass(frozen=True)
class _DayResult:
    day: DraftScheduledDay
    omitted_item_ids: tuple[str, ...]
    missing_route_keys: tuple[str, ...]
    provider_fact_ids: tuple[str, ...]


@dataclass(frozen=True)
class _RouteSelection:
    edge: SpatialRouteEdge
    mode: RouteMode


class PlannerDraftMaterializer:
    """Compile formal fields without changing the Planner's semantic choices."""

    def __init__(
        self,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        max_parallel_days: int = MAX_PARALLEL_DAYS,
    ) -> None:
        if max_parallel_days < 1:
            raise ValueError("max_parallel_days must be positive")
        self._clock = clock
        self._max_parallel_days = max_parallel_days

    async def materialize(
        self,
        workspace: PlannerWorkspaceState,
        book: TaskBookV4,
        cancellation: ModelCancellation,
        *,
        input_state_version: int | None = None,
        allow_time_savings: bool = True,
    ) -> PlannerMaterialization:
        draft = workspace.working_itinerary
        strategy = workspace.planning_strategy
        if draft is None or strategy is None:
            raise PlannerGuardError("planner_materialization_prerequisites_missing")
        now = self._aware_now()
        semaphore = asyncio.Semaphore(self._max_parallel_days)

        async def build(day: WorkingItineraryDay) -> _DayResult:
            async with semaphore:
                cancellation.raise_if_cancelled("planner_materialize_day")
                return await asyncio.to_thread(
                    _materialize_day,
                    day,
                    workspace,
                    book,
                    allow_time_savings=allow_time_savings,
                )

        results = await asyncio.gather(*(build(day) for day in draft.days))
        ordered = tuple(sorted(results, key=lambda item: item.day.service_date))
        cancellation.raise_if_cancelled("planner_materialize_merge")
        degradation = tuple(
            dict.fromkeys(
                (
                    *(
                        f"物化器按授权省略 {item_id}。"
                        for result in ordered
                        for item_id in result.omitted_item_ids
                    ),
                    *(
                        f"缺少真实路线 {route_key}，未生成交通耗时。"
                        for result in ordered
                        for route_key in result.missing_route_keys
                    ),
                    *(
                        activity.missing_reason
                        for result in ordered
                        for activity in result.day.activities
                        if activity.availability is not DataAvailability.AVAILABLE
                        and activity.missing_reason is not None
                    ),
                )
            )
        )
        schedule = ScheduleValidationDraft(
            algorithm_version=MATERIALIZER_VERSION,
            request_id=UUID(server_id(draft.draft_id, draft.draft_revision, "schedule")),
            trip_id=UUID(workspace.trip_id),
            input_state_version=(
                input_state_version
                if input_state_version is not None
                else book.based_on_state_version
            ),
            task_book_id=UUID(book.task_book_id),
            task_book_revision=book.version,
            city_id=book.destination_and_dates.destination_canonical_id
            or book.destination_and_dates.destination_name,
            start_date=book.destination_and_dates.start_date,
            end_date=book.destination_and_dates.end_date,
            status=DataAvailability.PARTIAL if degradation else DataAvailability.AVAILABLE,
            days=tuple(result.day for result in ordered),
            unscheduled_strong_desires=_strong_unassigned(workspace),
            degradation_reasons=degradation,
            provider_fact_ids=tuple(
                dict.fromkeys(fact_id for result in ordered for fact_id in result.provider_fact_ids)
            ),
            generated_at=now,
        )
        cost = _compile_cost_draft(schedule, workspace, book, now)
        if allow_time_savings and _time_saving_routes_over_budget(cost, book):
            # A soft transit preference is flexible, not a blank cheque. If
            # even known reference costs exceed an explicit budget, compare
            # earlier departure without automatic taxi upgrades instead.
            return await self.materialize(
                workspace,
                book,
                cancellation,
                input_state_version=input_state_version,
                allow_time_savings=False,
            )
        return PlannerMaterialization(schedule=schedule, cost=cost)

    def _aware_now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Planner materializer clock must return an aware datetime")
        return value.astimezone(UTC)


def _materialize_day(
    day: WorkingItineraryDay,
    workspace: PlannerWorkspaceState,
    book: TaskBookV4,
    *,
    allow_time_savings: bool = True,
) -> _DayResult:
    """Compare bounded clock/route alternatives, without changing place choices.

    Reuse the exact same Provider observations. No extra model/provider calls,
    shortened visits, moved reservations or invented driving times are involved.
    """
    assert workspace.planning_strategy is not None
    policy = workspace.planning_strategy.daily_capacity_policy
    preferred_start = _time_minutes(policy.preferred_start_window.earliest) or 9 * 60
    baseline = _materialize_day_at(day, workspace, book, start_override=preferred_start)
    end = policy.preferred_end_window.latest
    baseline_penalty = day_timing_penalty(day, baseline.day, end)
    if baseline_penalty == 0 and not _known_hours_conflicts(day, baseline.day, workspace):
        return baseline
    starts = [preferred_start]
    if may_advance_departure(book, day):
        starts.extend(range(preferred_start - 15, max(7 * 60, preferred_start - 120) - 1, -15))
    candidates = [baseline]

    def bad_hours(result: _DayResult) -> int:
        return _known_hours_conflicts(day, result.day, workspace)

    for start in starts:
        for faster in (False, True):
            if (start == preferred_start and not faster) or (
                faster and (not allow_time_savings or not permits_taxi_tradeoff(book))
            ):
                continue
            candidate = _materialize_day_at(
                day, workspace, book, start_override=start, prefer_time_savings=faster
            )
            # Earlier departure may only wait for known opening, never worsen a
            # known visit window or hide an omitted activity/route gap.
            if (
                len(candidate.missing_route_keys) > len(baseline.missing_route_keys)
                or candidate.omitted_item_ids != baseline.omitted_item_ids
            ):
                continue
            if bad_hours(candidate) <= bad_hours(baseline):
                candidates.append(candidate)

    def rank(result: _DayResult) -> tuple[int, int, int]:
        # Prefer a small change; only buy meaningful time savings when needed.
        changed_modes = sum(
            a.mode != b.mode
            for a, b in zip(result.day.transport_legs, baseline.day.transport_legs, strict=False)
        )
        departure_shift = preferred_start - (_time_minutes(result.day.start_time) or 0)
        # Real opening/last-entry constraints precede preferred meal and departure
        # times. A nearby meal must not make the old invalid 09:00 baseline win
        # just because its commute no longer delays dinner.
        return (
            bad_hours(result),
            day_timing_penalty(day, result.day, end),
            departure_shift + changed_modes * 40,
        )

    best = min(candidates, key=rank)
    # Buying time before lunch need not turn the rest of the day into taxis.
    # Revert each automatic upgrade that only replaces waiting with faster travel.
    endpoints = {_item_place_id(item): _item_endpoint(item) for item in day.ordered_items}
    start_place, _, boundary = _day_boundary(workspace, day)
    if boundary is not None:
        endpoints[start_place] = boundary
    trial_day = day
    for old, new in zip(baseline.day.transport_legs, best.day.transport_legs, strict=False):
        if (
            old.mode is not RouteMode.TRANSIT
            or new.mode is not RouteMode.DRIVING
            or old.origin_place_id != new.origin_place_id
            or old.destination_place_id != new.destination_place_id
            or new.origin_place_id not in endpoints
            or new.destination_place_id not in endpoints
        ):
            continue
        trial = trial_day.model_copy(
            update={
                "route_mode_selections": (
                    *trial_day.route_mode_selections,
                    DraftRouteModeSelection(
                        origin=endpoints[new.origin_place_id],
                        destination=endpoints[new.destination_place_id],
                        transport_mode="public_transit",
                    ),
                )
            }
        )
        candidate = _materialize_day_at(
            trial,
            workspace,
            book,
            start_override=_time_minutes(best.day.start_time) or preferred_start,
            prefer_time_savings=True,
        )
        if (
            rank(candidate) < rank(best)
            and bad_hours(candidate) <= bad_hours(best)
            and candidate.omitted_item_ids == best.omitted_item_ids
            and len(candidate.missing_route_keys) <= len(best.missing_route_keys)
        ):
            best, trial_day = candidate, trial
    return best


def _known_hours_conflicts(
    day: WorkingItineraryDay,
    scheduled: DraftScheduledDay,
    workspace: PlannerWorkspaceState,
) -> int:
    """Count factual conflicts, including one continuous cross-lunch envelope.

    Unknown hours stay unknown. Neither public diagnostic wording nor preferred
    timing penalties decide whether a verified closing/last-entry time was met.
    """
    evidence = {item.canonical_entity_id: item for item in workspace.hours_evidence}
    conflicts = 0
    for item in day.ordered_items:
        if not isinstance(item.object_ref, CandidateRef):
            continue
        hours = evidence.get(item.object_ref.canonical_entity_id)
        dated = (
            next((x for x in hours.days if x.service_date == day.service_date), None)
            if hours
            else None
        )
        activities = [x for x in scheduled.activities if x.place_id == _item_place_id(item)]
        if dated is None or not activities:
            continue
        if dated.status is HoursDayStatus.CLOSED:
            conflicts += 1
        elif dated.status is HoursDayStatus.OPEN:
            start = min(x.start_time for x in activities)
            end = max(x.end_time for x in activities)
            if not any(
                window.opens_at <= start <= end <= window.closes_at
                and (window.last_entry_at is None or start <= window.last_entry_at)
                for window in dated.intervals
            ):
                conflicts += 1
    return conflicts


def _time_saving_routes_over_budget(cost: CostValidationDraft, book: TaskBookV4) -> bool:
    if cost.known_total_per_person is None:
        return False
    for item in book.hard_constraints:
        match = re.search(
            r"(总预算|人均预算)\s*(?:为|是|不超过|最多|上限)?\s*(\d+(?:\.\d+)?)\s*元", item.value
        )
        if match:
            ceiling = int(Decimal(match[2]) * 100)
            known = cost.known_total_per_person.maximum_fen
            if match[1] == "总预算":
                known *= cost.party_size
            if known > ceiling:
                return True
    return False


def _materialize_day_at(
    day: WorkingItineraryDay,
    workspace: PlannerWorkspaceState,
    book: TaskBookV4,
    *,
    start_override: int,
    prefer_time_savings: bool = False,
    duration_overrides: Mapping[str, int] | None = None,
) -> _DayResult:
    assert workspace.working_itinerary is not None
    assert workspace.planning_strategy is not None
    strategy = workspace.planning_strategy
    pool_by_id = workspace.candidate_pool.candidate_by_id()
    places = {item.canonical_entity_id: item for item in workspace.place_evidence}
    hours = {item.canonical_entity_id: item for item in workspace.hours_evidence}
    discardable = {
        item.draft_item_id: item
        for item in workspace.working_itinerary.discardable_objects
        if item.mode == "materializer_may_omit"
    }
    start_limit = start_override
    end_limit = _time_minutes(strategy.daily_capacity_policy.preferred_end_window.latest)
    if end_limit is None:
        end_limit = 20 * 60
    hard_end = explicit_end_deadline(book, day.service_date)
    duration_estimates = {
        item.canonical_entity_id: item for item in workspace.visit_duration_estimates
    }

    start_place, end_place, boundary_endpoint = _day_boundary(workspace, day)
    cursor = start_limit
    current_endpoint = boundary_endpoint
    current_place = start_place
    activities: list[DraftScheduledActivity] = []
    pauses: list[DraftScheduledPause] = []
    legs: list[DraftScheduledTransport] = []
    omitted: list[str] = []
    missing_routes: list[str] = []
    fact_ids: list[str] = []
    walking_m = 0
    cycling_m = 0
    buffer_minutes = 0

    concrete_meals = {
        item.meal_slot
        for item in day.ordered_items
        if item.item_kind == "dining" and item.meal_slot is not None
    }
    if any(item.onsite_lunch for item in day.ordered_items):
        concrete_meals.add("lunch")
    pending_meals = [meal for meal in day.dining_goals if meal not in concrete_meals]
    timeline: list[tuple[Literal["item", "meal"], DraftItem | str]] = []
    for item in day.ordered_items:
        item_start = _expected_start(item)
        while pending_meals and _MEAL_START[pending_meals[0]] <= item_start:
            timeline.append(("meal", pending_meals.pop(0)))
        timeline.append(("item", item))
    timeline.extend(("meal", meal) for meal in pending_meals)

    for timeline_index, (kind, value) in enumerate(timeline):
        if kind == "meal":
            meal = str(value)
            start = max(cursor, _MEAL_START[meal])
            duration = _MEAL_DURATION[meal]
            pauses.append(
                DraftScheduledPause(
                    pause_id=UUID(
                        server_id(workspace.working_itinerary.draft_id, day.service_date, meal)
                    ),
                    kind=SchedulePauseKind.MEAL,
                    service_date=day.service_date,
                    start_time=_as_time(start),
                    end_time=_as_time(start + duration),
                    duration_minutes=duration,
                    reason=f"草稿要求保留{_meal_label(meal)}时段，未绑定具体餐厅。",
                )
            )
            cursor = start + duration
            continue

        assert isinstance(value, DraftItem)
        item = value
        endpoint = _item_endpoint(item)
        place_id = _item_place_id(item)
        if current_endpoint is not None and current_endpoint != endpoint:
            selected = _select_route(
                current_endpoint,
                endpoint,
                day.transport_preferences,
                workspace,
                day=day,
                book=book,
                walked_m=walking_m,
                prefer_time_savings=prefer_time_savings,
            )
            if selected is None:
                missing_route_key = _route_key(current_endpoint, endpoint, day.service_date)
                missing_routes.append(missing_route_key)
                departure = cursor
                arrival = departure + MISSING_ROUTE_PLACEHOLDER_MINUTES
                legs.append(
                    _missing_route_placeholder(
                        workspace=workspace,
                        service_date=day.service_date,
                        route_key=missing_route_key,
                        origin_place_id=current_place,
                        destination_place_id=place_id,
                        departure=departure,
                        arrival=arrival,
                        mode=day.transport_preferences[0],
                        discriminator=len(legs),
                    )
                )
                cursor = arrival + MISSING_ROUTE_BUFFER_MINUTES
                buffer_minutes += MISSING_ROUTE_BUFFER_MINUTES
            else:
                edge = selected.edge
                assert edge.duration_minutes is not None
                departure = cursor
                arrival = departure + edge.duration_minutes
                distance = edge.distance_meters or 0
                leg_buffer = 10
                legs.append(
                    DraftScheduledTransport(
                        leg_id=UUID(
                            server_id(
                                workspace.working_itinerary.draft_id,
                                day.service_date,
                                edge.route_edge_id,
                                len(legs),
                            )
                        ),
                        origin_place_id=current_place,
                        destination_place_id=place_id,
                        departure_time=_as_time(departure),
                        arrival_time=_as_time(arrival),
                        mode=selected.mode,
                        availability=DataAvailability.AVAILABLE
                        if edge.status == "available"
                        else DataAvailability.PARTIAL,
                        distance_m=distance,
                        duration_minutes=edge.duration_minutes,
                        walking_m=distance if selected.mode is RouteMode.WALKING else 0,
                        buffer_minutes=leg_buffer,
                        source_reference_ids=edge.fact_reference_ids,
                        missing_reason=edge.missing_reason,
                    )
                )
                cursor = arrival + leg_buffer
                buffer_minutes += leg_buffer
                if selected.mode is RouteMode.WALKING:
                    walking_m += distance
                elif selected.mode is RouteMode.CYCLING:
                    cycling_m += distance
                fact_ids.extend(edge.fact_reference_ids)

        requested_duration = _duration_minutes(
            item,
            pool_by_id.get(_candidate_id(item)),
            duration_estimates.get(getattr(item.object_ref, "canonical_entity_id", "")),
            pace_profile=strategy.daily_capacity_policy.pace_profile,
        )
        duration = (duration_overrides or {}).get(item.draft_item_id, requested_duration)
        finish_limits: tuple[int, ...] = ()
        if item.item_kind == "visit" and timeline_index + 1 < len(timeline):
            next_kind, next_value = timeline[timeline_index + 1]
            next_meal = (
                str(next_value)
                if next_kind == "meal"
                else (next_value.meal_slot if isinstance(next_value, DraftItem) else None)
            )
            if next_meal in {"lunch", "dinner"}:
                transfer = 0
                if isinstance(next_value, DraftItem):
                    onward = _select_route(
                        endpoint,
                        _item_endpoint(next_value),
                        day.transport_preferences,
                        workspace,
                        day=day,
                        book=book,
                        walked_m=walking_m,
                        prefer_time_savings=prefer_time_savings,
                    )
                    transfer = (
                        onward.edge.duration_minutes + 10
                        if onward is not None and onward.edge.duration_minutes is not None
                        else MISSING_ROUTE_PLACEHOLDER_MINUTES + MISSING_ROUTE_BUFFER_MINUTES
                    )
                finish_limits = tuple(
                    window[next_meal][1] - transfer
                    for window in (PREFERRED_MEAL_START_WINDOWS, MEAL_START_WINDOWS)
                )
                if isinstance(next_value, DraftItem) and isinstance(
                    next_value.object_ref, CandidateRef
                ):
                    meal_hours = hours.get(next_value.object_ref.canonical_entity_id)
                    meal_date = (
                        next(
                            (x for x in meal_hours.days if x.service_date == day.service_date), None
                        )
                        if meal_hours is not None
                        else None
                    )
                    if meal_date is not None and meal_date.status is HoursDayStatus.OPEN:
                        meal_duration = _duration_minutes(
                            next_value, pool_by_id.get(_candidate_id(next_value)), None
                        )
                        latest_starts = [
                            min(
                                _time_minutes(interval.closes_at) - meal_duration,
                                _time_minutes(interval.last_entry_at) or 24 * 60,
                                _expected_latest(next_value),
                                MEAL_START_WINDOWS[next_meal][1],
                            )
                            for interval in meal_date.intervals
                            if max(
                                _time_minutes(interval.opens_at),
                                _time_minutes(next_value.expected_window.earliest) or 0,
                                MEAL_START_WINDOWS[next_meal][0],
                            )
                            <= min(
                                _time_minutes(interval.closes_at) - meal_duration,
                                _time_minutes(interval.last_entry_at) or 24 * 60,
                                _expected_latest(next_value),
                                MEAL_START_WINDOWS[next_meal][1],
                            )
                        ]
                        if latest_starts:
                            finish_limits = tuple(
                                min(limit, max(latest_starts) - transfer) for limit in finish_limits
                            )
        estimate = duration_estimates.get(getattr(item.object_ref, "canonical_entity_id", ""))
        meal_minutes = 60 if item.onsite_lunch else 0
        # A full-day visit has one entry/exit envelope but excludes lunch from
        # its pure visit duration. Fit the envelope to hours before splitting.
        envelope_estimate = (
            estimate.model_copy(
                update={
                    "minimum_minutes": estimate.minimum_minutes + meal_minutes,
                    "maximum_minutes": estimate.maximum_minutes + meal_minutes,
                }
            )
            if estimate and meal_minutes
            else estimate
        )
        (start, availability, missing_reason, sources), envelope_duration = _fit_flexible_visit(
            item,
            day.service_date,
            cursor,
            duration + meal_minutes,
            pool_by_id,
            places,
            hours,
            day_start=start_limit,
            estimate=envelope_estimate,
            finish_limits=finish_limits,
        )
        duration = envelope_duration - meal_minutes
        projected_finish = start + envelope_duration
        if (
            hard_end is not None
            and projected_finish > _time_minutes(hard_end[0])
            and item.draft_item_id in discardable
        ):
            omitted.append(item.draft_item_id)
            continue
        wait = max(0, start - cursor)
        buffer_minutes += wait
        finish = projected_finish
        activity = DraftScheduledActivity(
            activity_id=UUID(server_id(item.draft_item_id, day.service_date, "activity")),
            node_id=UUID(server_id(item.draft_item_id, "node")),
            place_id=place_id,
            kind=_activity_kind(item),
            role=_anchor_role(item, book),
            title=_item_title(item, pool_by_id, book),
            service_date=day.service_date,
            start_time=_as_time(start),
            end_time=_as_time(finish),
            duration_minutes=duration,
            availability=availability,
            source_reference_ids=sources,
            missing_reason=missing_reason,
            timing_notice=(
                f"参观时长由建议的{requested_duration}分钟调整为{duration}分钟，"
                "仍在建议范围内，以适配开放时间和用餐安排。"
                if duration < requested_duration
                else "该时段晚于建议收尾时间。"
                if finish > end_limit + 60
                else None
            ),
        )
        if item.onsite_lunch:
            # No extra journey, exit/re-entry assumption or fake restaurant.
            # Keep the original first segment ID and node for old consumers.
            lunch_start = max(start + 30, min(13 * 60, max(12 * 60, start + 90)))
            before_lunch = min(max(1, duration - 30), lunch_start - start)
            lunch_start = start + before_lunch
            activities.append(
                activity.model_copy(
                    update={
                        "end_time": _as_time(lunch_start),
                        "duration_minutes": before_lunch,
                    }
                )
            )
            pauses.append(
                DraftScheduledPause(
                    pause_id=UUID(server_id(item.draft_item_id, day.service_date, "onsite-lunch")),
                    kind=SchedulePauseKind.MEAL,
                    service_date=day.service_date,
                    start_time=_as_time(lunch_start),
                    end_time=_as_time(lunch_start + 60),
                    duration_minutes=60,
                    reason=f"{activity.title}内午餐；具体餐厅与价格待确认，不默认允许自带食物或出园再入园。",
                )
            )
            activities.append(
                activity.model_copy(
                    update={
                        "activity_id": UUID(
                            server_id(item.draft_item_id, day.service_date, "continued-visit")
                        ),
                        "title": f"{activity.title}（继续游览）",
                        "start_time": _as_time(lunch_start + 60),
                        "duration_minutes": duration - before_lunch,
                    }
                )
            )
        else:
            activities.append(activity)
        fact_ids.extend(sources)
        cursor = finish
        current_endpoint = endpoint
        current_place = place_id

    if (
        boundary_endpoint is not None
        and current_endpoint is not None
        and current_endpoint != boundary_endpoint
    ):
        selected = _select_route(
            current_endpoint,
            boundary_endpoint,
            day.transport_preferences,
            workspace,
            day=day,
            book=book,
            walked_m=walking_m,
            prefer_time_savings=prefer_time_savings,
        )
        if selected is None:
            missing_route_key = _route_key(
                current_endpoint,
                boundary_endpoint,
                day.service_date,
            )
            missing_routes.append(missing_route_key)
            departure = cursor
            arrival = departure + MISSING_ROUTE_PLACEHOLDER_MINUTES
            legs.append(
                _missing_route_placeholder(
                    workspace=workspace,
                    service_date=day.service_date,
                    route_key=missing_route_key,
                    origin_place_id=current_place,
                    destination_place_id=end_place,
                    departure=departure,
                    arrival=arrival,
                    mode=day.transport_preferences[0],
                    discriminator="return",
                )
            )
            cursor = arrival + MISSING_ROUTE_BUFFER_MINUTES
            buffer_minutes += MISSING_ROUTE_BUFFER_MINUTES
        else:
            edge = selected.edge
            assert edge.duration_minutes is not None
            departure = cursor
            arrival = departure + edge.duration_minutes
            distance = edge.distance_meters or 0
            legs.append(
                DraftScheduledTransport(
                    leg_id=UUID(
                        server_id(
                            workspace.working_itinerary.draft_id,
                            day.service_date,
                            edge.route_edge_id,
                            "return",
                        )
                    ),
                    origin_place_id=current_place,
                    destination_place_id=end_place,
                    departure_time=_as_time(departure),
                    arrival_time=_as_time(arrival),
                    mode=selected.mode,
                    availability=DataAvailability.AVAILABLE
                    if edge.status == "available"
                    else DataAvailability.PARTIAL,
                    distance_m=distance,
                    duration_minutes=edge.duration_minutes,
                    walking_m=distance if selected.mode is RouteMode.WALKING else 0,
                    buffer_minutes=10,
                    source_reference_ids=edge.fact_reference_ids,
                    missing_reason=edge.missing_reason,
                )
            )
            cursor = arrival + 10
            buffer_minutes += 10
            walking_m += distance if selected.mode is RouteMode.WALKING else 0
            cycling_m += distance if selected.mode is RouteMode.CYCLING else 0
            fact_ids.extend(edge.fact_reference_ids)

    pause_minutes = sum(item.duration_minutes for item in pauses)
    active_minutes = sum(item.duration_minutes for item in activities) + sum(
        item.duration_minutes for item in legs
    )
    if not activities and not pauses:
        cursor = start_limit + 60
    result = _DayResult(
        day=DraftScheduledDay(
            service_date=day.service_date,
            start_place_id=start_place,
            end_place_id=end_place,
            start_time=_as_time(start_limit),
            end_time=_as_time(max(cursor, start_limit + 1)),
            activities=tuple(activities),
            pauses=tuple(sorted(pauses, key=lambda item: (item.start_time, str(item.pause_id)))),
            transport_legs=tuple(legs),
            active_minutes=active_minutes,
            walking_m=walking_m,
            cycling_m=cycling_m,
            meal_minutes=sum(
                item.duration_minutes for item in pauses if item.kind is SchedulePauseKind.MEAL
            )
            + sum(
                item.duration_minutes
                for item in activities
                if item.kind is ScheduleActivityKind.RESTAURANT
            ),
            rest_minutes=pause_minutes
            - sum(item.duration_minutes for item in pauses if item.kind is SchedulePauseKind.MEAL),
            buffer_minutes=buffer_minutes,
        ),
        omitted_item_ids=tuple(omitted),
        missing_route_keys=tuple(missing_routes),
        provider_fact_ids=tuple(dict.fromkeys(fact_ids)),
    )
    if duration_overrides is not None:
        return result
    return _extend_visits_into_waits(
        day,
        workspace,
        book,
        result,
        start_override=start_override,
        prefer_time_savings=prefer_time_savings,
    )


def _visit_waits(day: DraftScheduledDay) -> list[tuple[int, int]]:
    """Measure idle time, excluding actual transit, its buffer and explicit rests."""
    result = []
    for index, activity in enumerate(day.activities):
        ready = _time_minutes(day.activities[index - 1].end_time if index else day.start_time)
        assert ready is not None
        start = _time_minutes(activity.start_time)
        assert start is not None
        ready = max(
            [
                ready,
                *(
                    _time_minutes(leg.arrival_time) + leg.buffer_minutes
                    for leg in day.transport_legs
                    if ready <= _time_minutes(leg.arrival_time) <= start
                ),
                *(
                    _time_minutes(pause.end_time)
                    for pause in day.pauses
                    if ready <= _time_minutes(pause.end_time) <= start
                ),
            ]
        )
        if start > ready:
            result.append((index, start - ready))
    return result


def _extend_visits_into_waits(
    day: WorkingItineraryDay,
    workspace: PlannerWorkspaceState,
    book: TaskBookV4,
    baseline: _DayResult,
    *,
    start_override: int,
    prefer_time_savings: bool,
) -> _DayResult:
    """Allocate existing idle time within advisory ranges, without more LLM/tools.

    Recompute the whole day after each bounded change. Meals, fixed appointments,
    known hours, transit choices and other visit durations may not pay for a longer
    stop. Multiple visits in the same meal/appointment-delimited block can deepen.
    """
    estimates = {item.canonical_entity_id: item for item in workspace.visit_duration_estimates}
    if not estimates or not _visit_waits(baseline.day):
        return baseline
    sources = {
        server_id(item.draft_item_id, day.service_date, "activity"): item
        for item in day.ordered_items
    }
    current = baseline
    attempts = 0
    # First restore normal visits, then use remaining slack for meaningful depth.
    for depth in ("normal", "extended"):
        for target_index, _ in _visit_waits(current.day):
            eligible = []
            for activity in reversed(current.day.activities[:target_index]):
                item = sources.get(str(activity.activity_id))
                if (
                    item is None
                    or item.item_kind != "visit"
                    or not isinstance(item.object_ref, CandidateRef)
                ):
                    break
                estimate = estimates.get(item.object_ref.canonical_entity_id)
                if (
                    estimate is None
                    or item.onsite_lunch
                    or item.expected_window.earliest is not None
                    or item.expected_window.latest is not None
                ):
                    continue
                target = (
                    estimate.maximum_minutes
                    if depth == "extended"
                    else min(
                        estimate.maximum_minutes,
                        ceil((estimate.minimum_minutes + estimate.maximum_minutes) / 30) * 15,
                    )
                )
                if target > activity.duration_minutes:
                    eligible.append((item, target))
            for item, target in reversed(eligible):
                wait = dict(_visit_waits(current.day)).get(target_index, 0)
                if wait < 1 or attempts >= 16:
                    break
                by_id = {str(a.activity_id): a for a in current.day.activities}
                activity = by_id[server_id(item.draft_item_id, day.service_date, "activity")]
                addition = min(wait, target - activity.duration_minutes)
                if addition <= 0:
                    continue
                overrides = {
                    sources[key].draft_item_id: value.duration_minutes
                    + (
                        by_id[
                            server_id(
                                sources[key].draft_item_id, day.service_date, "continued-visit"
                            )
                        ].duration_minutes
                        if sources[key].onsite_lunch
                        else 0
                    )
                    for key, value in by_id.items()
                    if key in sources
                }
                overrides[item.draft_item_id] = activity.duration_minutes + addition
                attempts += 1
                candidate = _materialize_day_at(
                    day,
                    workspace,
                    book,
                    start_override=start_override,
                    prefer_time_savings=prefer_time_savings,
                    duration_overrides=overrides,
                )
                if _safe_visit_extension(day, workspace, current, candidate):
                    current = candidate
    return current


def _safe_visit_extension(
    day: WorkingItineraryDay,
    workspace: PlannerWorkspaceState,
    before: _DayResult,
    after: _DayResult,
) -> bool:
    if (
        before.omitted_item_ids != after.omitted_item_ids
        or before.missing_route_keys != after.missing_route_keys
        or [a.activity_id for a in before.day.activities]
        != [a.activity_id for a in after.day.activities]
        or after.day.end_time > before.day.end_time
        or sum(wait for _, wait in _visit_waits(after.day))
        >= sum(wait for _, wait in _visit_waits(before.day))
    ):
        return False
    for preferred in (False, True):
        old_issues = {
            str(issue["draft_item_id"]): int(str(issue["outside_minutes"]))
            for issue in meal_time_violations(day, before.day, preferred=preferred)
        }
        if any(
            int(str(issue["outside_minutes"])) > old_issues.get(str(issue["draft_item_id"]), 0)
            for issue in meal_time_violations(day, after.day, preferred=preferred)
        ):
            return False
    hours = {item.canonical_entity_id: item for item in workspace.hours_evidence}
    sources = {
        server_id(item.draft_item_id, day.service_date, segment): item
        for item in day.ordered_items
        for segment in (("activity", "continued-visit") if item.onsite_lunch else ("activity",))
    }
    for old, new in zip(before.day.activities, after.day.activities, strict=True):
        if new.duration_minutes < old.duration_minutes:
            return False
        if old.kind is ScheduleActivityKind.FIXED_EVENT and (
            old.start_time != new.start_time or old.end_time != new.end_time
        ):
            return False
        source = sources.get(str(new.activity_id))
        if (
            source is not None
            and (
                source.expected_window.earliest is not None
                or source.expected_window.latest is not None
            )
            and old.start_time != new.start_time
        ):
            return False
        evidence = hours.get(str(new.place_id))
        dated = (
            next((x for x in evidence.days if x.service_date == day.service_date), None)
            if evidence
            else None
        )
        if (
            dated is not None
            and dated.status is HoursDayStatus.OPEN
            and not any(
                interval.opens_at <= new.start_time < new.end_time <= interval.closes_at
                and (
                    interval.last_entry_at is None
                    or new.start_time <= interval.last_entry_at
                    or (
                        source is not None
                        and source.onsite_lunch
                        and str(new.activity_id)
                        == server_id(source.draft_item_id, day.service_date, "continued-visit")
                    )
                )
                for interval in dated.intervals
            )
        ):
            return False
    return True


def _fit_item(
    item: DraftItem,
    service_date: date,
    cursor: int,
    duration: int,
    pool_by_id: Mapping[str, CandidatePoolEntry],
    places: Mapping[str, PlannerPlaceEvidence],
    hours: Mapping[str, PlannerHoursEvidence],
    *,
    day_start: int | None = None,
) -> tuple[int, DataAvailability, str | None, tuple[str, ...]]:
    preferred = _expected_start(item)
    if item.meal_slot == "dinner" and item.expected_window.earliest is None:
        # The preferred 18:00-20:00 band is a quality target, not permission to
        # strand travellers at a restaurant until 18:00. Actual visits can push
        # dinner later, but after 17:00 an ordinary meal may start on arrival.
        preferred = max(cursor, MEAL_START_WINDOWS["dinner"][0])
    if (
        isinstance(item.object_ref, CandidateRef)
        and item.item_kind == "visit"
        and item.expected_window.earliest is None
    ):
        # Day-part labels order experiences; they are not clock reservations.
        preferred = cursor
    if (
        not isinstance(item.object_ref, FixedCommitmentRef)
        and item.expected_window.earliest is None
        and item.meal_slot is None
        and item.expected_window.part_of_day == "midday"
    ):
        # A model's broad day-part label is not an 11:30 reservation. Let a
        # second morning visit start on arrival; dated opening hours still win.
        preferred = min(preferred, max(cursor, 10 * 60 + 30))
    if (
        day_start is not None
        and day_start < 9 * 60
        and not isinstance(item.object_ref, FixedCommitmentRef)
        and item.expected_window.earliest is None
        and item.meal_slot is None
        and item.expected_window.part_of_day in {"morning", "anytime"}
    ):
        preferred = day_start
    if (
        not isinstance(item.object_ref, FixedCommitmentRef)
        and item.expected_window.earliest is None
        and item.meal_slot != "dinner"
        and preferred - cursor > 45
    ):
        # Day parts and default mealtimes are preferences, not appointments.
        # Move within a sensible broad window instead of waiting for 13:30/18:00.
        floor = (
            PREFERRED_MEAL_START_WINDOWS[item.meal_slot][0]
            if item.meal_slot
            else {
                "morning": 9 * 60,
                "midday": 11 * 60,
                "afternoon": 12 * 60,
                "evening": 16 * 60 + 30,
                "anytime": 9 * 60,
            }[item.expected_window.part_of_day]
        )
        preferred = max(floor, min(preferred, cursor + 30))
    earliest = max(cursor, preferred)
    latest = _expected_latest(item)
    if item.meal_slot is not None:
        earliest = max(earliest, MEAL_START_WINDOWS[item.meal_slot][0])
        latest = min(latest, MEAL_START_WINDOWS[item.meal_slot][1])
    if isinstance(item.object_ref, FixedCommitmentRef):
        has_explicit_start = item.expected_window.earliest is not None
        return (
            earliest,
            (DataAvailability.AVAILABLE if has_explicit_start else DataAvailability.PARTIAL),
            (
                None
                if has_explicit_start
                else "固定安排仅确认日期，具体时间待确认；当前排程时间仅用于内部排序。"
            ),
            (item.object_ref.commitment_id,),
        )
    candidate = pool_by_id.get(item.object_ref.candidate_id)
    place = places.get(item.object_ref.canonical_entity_id)
    entry_sources = tuple(getattr(candidate, "fact_reference_ids", ()))
    intent_sources = tuple(getattr(candidate, "source_intent_refs", ()))
    if place is None:
        return (
            earliest,
            DataAvailability.MISSING,
            "缺少该地点的真实地点事实，当前时间仅是待校验草案。",
            tuple(dict.fromkeys((*intent_sources, *entry_sources))),
        )
    place_source = (place.fact_reference_id,)
    evidence = hours.get(item.object_ref.canonical_entity_id)
    if evidence is None:
        return (
            earliest,
            DataAvailability.PARTIAL,
            "未取得该日期的营业时间，当前时段必须继续核验。",
            tuple(dict.fromkeys((*intent_sources, *entry_sources, *place_source))),
        )
    date_hours = next((day for day in evidence.days if day.service_date == service_date), None)
    hours_source = (evidence.fact_reference_id,)
    sources = tuple(dict.fromkeys((*intent_sources, *entry_sources, *place_source, *hours_source)))
    if date_hours is None or date_hours.status is HoursDayStatus.UNKNOWN:
        return earliest, DataAvailability.PARTIAL, "该日期营业状态未知。", sources
    if date_hours.status is HoursDayStatus.CONFLICT:
        return earliest, DataAvailability.PARTIAL, "营业信息存在冲突。", sources
    if date_hours.status is HoursDayStatus.CLOSED:
        return earliest, DataAvailability.PARTIAL, "真实营业资料显示该日期闭馆。", sources

    for interval in date_hours.intervals:
        opening = _time_minutes(interval.opens_at)
        closing = _time_minutes(interval.closes_at)
        assert opening is not None and closing is not None
        start = max(earliest, opening)
        start_deadline = min(
            latest,
            _time_minutes(interval.last_entry_at) or latest,
            closing - duration,
        )
        if start <= start_deadline:
            return start, DataAvailability.AVAILABLE, None, sources
    if item.meal_slot is not None and item.expected_window.earliest is None:
        # A preferred dinner time must not override verified restaurant hours.
        for interval in date_hours.intervals:
            start = max(
                cursor, MEAL_START_WINDOWS[item.meal_slot][0], _time_minutes(interval.opens_at)
            )
            deadline = min(
                latest,
                _time_minutes(interval.last_entry_at) or latest,
                _time_minutes(interval.closes_at) - duration,
            )
            if start <= deadline:
                return start, DataAvailability.AVAILABLE, None, sources
    return earliest, DataAvailability.PARTIAL, "期望时段无法完整落在真实开放窗口内。", sources


def _fit_flexible_visit(
    item: DraftItem,
    service_date: date,
    cursor: int,
    duration: int,
    pool_by_id: Mapping[str, CandidatePoolEntry],
    places: Mapping[str, PlannerPlaceEvidence],
    hours: Mapping[str, PlannerHoursEvidence],
    *,
    day_start: int,
    estimate: PlannerVisitDurationEstimate | None,
    finish_limits: tuple[int, ...] = (),
) -> tuple[tuple[int, DataAvailability, str | None, tuple[str, ...]], int]:
    """Fit advisory durations to real hours, never shorten a fixed commitment.

    `extended` is a preference, not a reservation. Try the largest feasible
    duration within the existing estimate; do not invent hours or a lower bound.
    """
    fitted = _fit_item(
        item, service_date, cursor, duration, pool_by_id, places, hours, day_start=day_start
    )
    if (
        item.item_kind != "visit"
        or not isinstance(item.object_ref, CandidateRef)
        or estimate is None
    ):
        return fitted, duration
    evidence = hours.get(item.object_ref.canonical_entity_id)
    date_hours = (
        next((day for day in evidence.days if day.service_date == service_date), None)
        if evidence
        else None
    )
    known_open = date_hours is not None and date_hours.status is HoursDayStatus.OPEN
    if date_hours is not None and date_hours.status is HoursDayStatus.CLOSED:
        return fitted, duration
    if not known_open and not finish_limits:
        return fitted, duration
    hourly_capacity = (
        [
            _time_minutes(interval.closes_at) - max(fitted[0], _time_minutes(interval.opens_at))
            for interval in date_hours.intervals
        ]
        if known_open and date_hours is not None
        else [duration]
    )
    for limit in (*finish_limits, 24 * 60):
        feasible_durations = {
            min(duration, capacity, limit - fitted[0]) for capacity in hourly_capacity
        }
        for adjusted in sorted(feasible_durations, reverse=True):
            if not estimate.minimum_minutes <= adjusted <= duration:
                continue
            candidate = _fit_item(
                item, service_date, cursor, adjusted, pool_by_id, places, hours, day_start=day_start
            )
            if candidate[0] + adjusted <= limit and (
                candidate[1] is DataAvailability.AVAILABLE
                or (not known_open and candidate[1] is DataAvailability.PARTIAL)
            ):
                return candidate, adjusted
    return fitted, duration


def _duration_minutes(
    item: DraftItem,
    candidate: CandidatePoolEntry | None,
    estimate: PlannerVisitDurationEstimate | None = None,
    *,
    pace_profile: str | None = None,
) -> int:
    typical = getattr(
        getattr(candidate, "advisory_features", None), "typical_duration_minutes", None
    )
    if item.item_kind == "visit" and typical is None and estimate is not None:
        if item.duration_preference == "short":
            return estimate.minimum_minutes
        if item.duration_preference == "extended" or pace_profile == "relaxed":
            return estimate.maximum_minutes
        target = (estimate.minimum_minutes + estimate.maximum_minutes) / 2
        return max(estimate.minimum_minutes, min(estimate.maximum_minutes, ceil(target / 15) * 15))
    if item.item_kind in {"dining", "fixed_event", "arrival", "departure"}:
        base = typical or 60
    else:
        base = typical or 90
    multiplier = {"short": 0.75, "normal": 1.0, "extended": 1.25}.get(
        item.duration_preference or "normal",
        1.0,
    )
    return max(30, int(round(base * multiplier / 15)) * 15)


def _item_endpoint(item: DraftItem) -> SpatialRouteEndpoint:
    if isinstance(item.object_ref, CandidateRef):
        return SpatialRouteEndpoint(kind="candidate", reference_id=item.object_ref.candidate_id)
    return SpatialRouteEndpoint(kind="fixed_commitment", reference_id=item.object_ref.commitment_id)


def _item_place_id(item: DraftItem) -> UUID:
    if isinstance(item.object_ref, CandidateRef):
        try:
            return UUID(item.object_ref.canonical_entity_id)
        except ValueError:
            return UUID(server_id("candidate-place", item.object_ref.canonical_entity_id))
    return UUID(server_id("fixed-place", item.object_ref.commitment_id))


def _day_boundary(
    workspace: PlannerWorkspaceState,
    day: WorkingItineraryDay,
) -> tuple[UUID, UUID, SpatialRouteEndpoint | None]:
    draft = workspace.working_itinerary
    assert draft is not None
    baseline = draft.lodging_baseline
    if baseline.selected_offer_ref is not None:
        offer = baseline.selected_offer_ref
        place_id = UUID(server_id("hotel-property", offer.property_id))
        return (
            place_id,
            place_id,
            SpatialRouteEndpoint(kind="hotel_offer", reference_id=offer.offer_id),
        )
    if baseline.fixed_commitment_ref is not None:
        fixed = baseline.fixed_commitment_ref
        place_id = UUID(server_id("fixed-hotel", fixed.commitment_id))
        return (
            place_id,
            place_id,
            SpatialRouteEndpoint(kind="fixed_commitment", reference_id=fixed.commitment_id),
        )
    if day.ordered_items:
        first = day.ordered_items[0]
        last = day.ordered_items[-1]
        return _item_place_id(first), _item_place_id(last), None
    placeholder = UUID(server_id(draft.draft_id, day.service_date, "day-boundary"))
    return placeholder, placeholder, None


def _select_route(
    origin: SpatialRouteEndpoint,
    destination: SpatialRouteEndpoint,
    preferences: tuple[TransportPreference, ...],
    workspace: PlannerWorkspaceState,
    *,
    day: WorkingItineraryDay | None = None,
    book: TaskBookV4 | None = None,
    walked_m: int = 0,
    prefer_time_savings: bool = False,
) -> _RouteSelection | None:
    edges = (
        *(workspace.spatial_observation.route_edges if workspace.spatial_observation else ()),
        *workspace.route_evidence,
    )
    by_mode = {
        edge.transport_mode: edge
        for edge in edges
        if edge.origin == origin
        and edge.destination == destination
        and edge.status in {"available", "partial"}
        and edge.duration_minutes is not None
    }
    override = (
        next(
            (
                choice.transport_mode
                for choice in day.route_mode_selections
                if choice.origin == origin and choice.destination == destination
            ),
            None,
        )
        if day
        else None
    )
    if override is not None:
        edge = by_mode.get(override)
        return _RouteSelection(edge=edge, mode=_route_mode(override)) if edge is not None else None
    walking = by_mode.get("walking")
    if (
        book is not None
        and walking is not None
        and walking.distance_meters is not None
        and walking.distance_meters <= 1000
        and walking.duration_minutes is not None
        and walking.duration_minutes <= 15
        and walked_m + walking.distance_meters <= 5000
        and _allows_short_walk(book, workspace, day)
    ):
        return _RouteSelection(edge=walking, mode=RouteMode.WALKING)
    for index, preference in enumerate(preferences):
        edge = by_mode.get(preference)
        if edge is None:
            continue
        if preference == "walking":
            # A fallback walk remains a short connection, not permission for a
            # multi-hour hike when bus/taxi data is missing. A walking-first day
            # has more room; an explicit per-leg user selection above still wins.
            max_minutes, max_meters = (30, 2000) if index == 0 else (15, 1000)
            if (
                edge.duration_minutes is None
                or edge.duration_minutes > max_minutes
                or edge.distance_meters is None
                or edge.distance_meters > max_meters
                or walked_m + edge.distance_meters > 5000
                or (book is not None and not _allows_short_walk(book, workspace, day))
            ):
                continue
        taxi = by_mode.get("taxi")
        if (
            prefer_time_savings
            and preference == "public_transit"
            and "taxi" in preferences
            and taxi is not None
            and taxi.duration_minutes is not None
            and edge.duration_minutes is not None
            and edge.duration_minutes - taxi.duration_minutes >= 15
            and taxi.duration_minutes <= edge.duration_minutes * 0.75
            and (book is None or permits_taxi_tradeoff(book))
        ):
            return _RouteSelection(edge=taxi, mode=RouteMode.DRIVING)
        return _RouteSelection(edge=edge, mode=_route_mode(preference))
    return None


def _allows_short_walk(
    book: TaskBookV4, workspace: PlannerWorkspaceState, day: WorkingItineraryDay | None
) -> bool:
    text = " ".join(
        item.value
        for item in (
            *book.pace_and_transport.pace_preferences,
            *book.pace_and_transport.transport_preferences,
            *book.hard_constraints,
        )
    )
    if re.search(
        r"少走|少步行|不(?:要|能)?(?:步行|走路)|轮椅|行动不便|行李|全程.{0,3}(?:打车|出租|自驾)|只.{0,3}(?:打车|出租)",
        text,
    ):
        return False
    return not any(
        day is not None
        and weather.service_date == day.service_date
        and weather.forecast_kind == "forecast"
        and (
            re.search(
                r"雨|雪|雷|沙尘", f"{weather.condition_day or ''} {weather.condition_night or ''}"
            )
            or (weather.high_celsius is not None and weather.high_celsius >= 35)
        )
        for weather in workspace.weather_evidence
    )


def _route_mode(value: str) -> RouteMode:
    return {
        "walking": RouteMode.WALKING,
        "public_transit": RouteMode.TRANSIT,
        "taxi": RouteMode.DRIVING,
        "driving": RouteMode.DRIVING,
    }[value]


def _route_key(
    origin: SpatialRouteEndpoint,
    destination: SpatialRouteEndpoint,
    service_date: date,
) -> str:
    return server_id(
        "missing-route",
        service_date,
        origin.kind,
        origin.reference_id,
        destination.kind,
        destination.reference_id,
    )


def _missing_route_placeholder(
    *,
    workspace: PlannerWorkspaceState,
    service_date: date,
    route_key: str,
    origin_place_id: UUID,
    destination_place_id: UUID,
    departure: int,
    arrival: int,
    mode: TransportPreference,
    discriminator: int | str,
) -> DraftScheduledTransport:
    """Reserve non-zero time without presenting the placeholder as Provider evidence."""

    assert workspace.working_itinerary is not None
    return DraftScheduledTransport(
        leg_id=UUID(
            server_id(
                workspace.working_itinerary.draft_id,
                service_date,
                route_key,
                discriminator,
            )
        ),
        origin_place_id=origin_place_id,
        destination_place_id=destination_place_id,
        departure_time=_as_time(departure),
        arrival_time=_as_time(arrival),
        mode=_route_mode(mode),
        availability=DataAvailability.MISSING,
        distance_m=0,
        duration_minutes=MISSING_ROUTE_PLACEHOLDER_MINUTES,
        walking_m=0,
        buffer_minutes=MISSING_ROUTE_BUFFER_MINUTES,
        source_reference_ids=(),
        missing_reason=(
            "实时路线缺失；为避免按零耗时排程暂预留45分钟，该占位不是距离、时长或可达性事实。"
        ),
    )


def _expected_start(item: DraftItem) -> int:
    earliest = _time_minutes(item.expected_window.earliest)
    if earliest is not None:
        return earliest
    if item.meal_slot is not None:
        return _MEAL_START[item.meal_slot]
    return _PART_OF_DAY_START[item.expected_window.part_of_day]


def _expected_latest(item: DraftItem) -> int:
    latest = _time_minutes(item.expected_window.latest)
    if latest is not None:
        return latest
    if item.meal_slot is not None:
        return _MEAL_LATEST[item.meal_slot]
    return {
        "morning": 12 * 60,
        "midday": 14 * 60,
        "afternoon": 17 * 60 + 30,
        "evening": 21 * 60,
        "anytime": 21 * 60,
    }[item.expected_window.part_of_day]


def _candidate_id(item: DraftItem) -> str:
    return item.object_ref.candidate_id if isinstance(item.object_ref, CandidateRef) else ""


def _activity_kind(item: DraftItem) -> ScheduleActivityKind:
    if item.item_kind == "dining":
        return ScheduleActivityKind.RESTAURANT
    if item.item_kind in {"fixed_event", "arrival", "departure"}:
        return ScheduleActivityKind.FIXED_EVENT
    return ScheduleActivityKind.ATTRACTION


def _anchor_role(item: DraftItem, book: TaskBookV4) -> AnchorRole:
    if isinstance(item.object_ref, FixedCommitmentRef):
        return AnchorRole.FIXED_EVENT
    candidate = item.object_ref
    if candidate.entity_kind is CandidateEntityKind.RESTAURANT:
        destinations = {
            value.canonical_entity_id for value in book.dining_direction.destination_restaurants
        }
        return (
            AnchorRole.DESTINATION_RESTAURANT
            if candidate.canonical_entity_id in destinations
            else AnchorRole.CONVENIENT_RESTAURANT
        )
    must = {value.canonical_entity_id for value in book.attraction_direction.must_visit}
    wanted = {value.canonical_entity_id for value in book.attraction_direction.wanted}
    if candidate.canonical_entity_id in must:
        return AnchorRole.MUST_ATTRACTION
    if candidate.canonical_entity_id in wanted:
        return AnchorRole.WANT_ATTRACTION
    return AnchorRole.CONVENIENT_ATTRACTION


def _item_title(
    item: DraftItem,
    pool_by_id: Mapping[str, CandidatePoolEntry],
    book: TaskBookV4,
) -> str:
    if isinstance(item.object_ref, CandidateRef):
        entry = pool_by_id.get(item.object_ref.candidate_id)
        return str(getattr(entry, "display_name", "待核验地点"))
    booking = next(
        (
            value
            for value in (*book.existing_bookings,)
            if value.booking_id == item.object_ref.commitment_id
        ),
        None,
    )
    if booking is not None:
        return booking.user_description
    return {"arrival": "抵达安排", "departure": "返程安排"}.get(
        item.item_kind,
        "已确认的固定安排",
    )


def _strong_unassigned(
    workspace: PlannerWorkspaceState,
) -> tuple[DraftUnscheduledStrongDesire, ...]:
    draft = workspace.working_itinerary
    if draft is None:
        return ()
    result: list[DraftUnscheduledStrongDesire] = []
    for item in draft.unassigned_intents:
        if item.commitment_level != "strong":
            continue
        entry = workspace.candidate_pool.candidate_by_id()[item.candidate_ref.candidate_id]
        result.append(
            DraftUnscheduledStrongDesire(
                node_id=UUID(server_id(item.candidate_ref.candidate_id, "node")),
                place_id=_candidate_place_id(item.candidate_ref),
                role=AnchorRole.DESTINATION_RESTAURANT
                if item.candidate_ref.entity_kind is CandidateEntityKind.RESTAURANT
                else AnchorRole.MUST_ATTRACTION,
                reason=f"强意愿尚未安排：{item.reason_code}。",
                source_reference_ids=tuple(
                    dict.fromkeys((*entry.source_intent_refs, *item.supporting_observation_refs))
                ),
            )
        )
    return tuple(result)


def _candidate_place_id(reference: CandidateRef) -> UUID:
    try:
        return UUID(reference.canonical_entity_id)
    except ValueError:
        return UUID(server_id("candidate-place", reference.canonical_entity_id))


def _compile_cost_draft(
    schedule: ScheduleValidationDraft,
    workspace: PlannerWorkspaceState,
    book: TaskBookV4,
    now: datetime,
) -> CostValidationDraft:
    party_size = _party_size(book)
    lodging_divisor = min(2, party_size)
    lines_by_date: dict[date, list[DraftCostEstimateLine]] = {
        day.service_date: [] for day in schedule.days
    }
    places = {place.canonical_entity_id: place for place in workspace.place_evidence}
    tickets = {
        (item.canonical_entity_id, item.service_date): item for item in workspace.ticket_evidence
    }
    edges = {
        fact: edge
        for edge in (
            *workspace.route_evidence,
            *(workspace.spatial_observation.route_edges if workspace.spatial_observation else ()),
        )
        for fact in edge.fact_reference_ids
    }
    for day in schedule.days:
        billed_visits: set[UUID] = set()
        for activity in day.activities:
            if activity.kind is ScheduleActivityKind.FIXED_EVENT:
                continue
            category = (
                CostCategory.DINING
                if activity.kind is ScheduleActivityKind.RESTAURANT
                else CostCategory.ATTRACTION_TICKETS
            )
            if category is CostCategory.ATTRACTION_TICKETS:
                if activity.node_id in billed_visits:
                    continue
                billed_visits.add(activity.node_id)
            place = places.get(str(activity.place_id))
            ticket = tickets.get((str(activity.place_id), day.service_date))
            price = (
                place.average_cost
                if category is CostCategory.DINING and place is not None
                else ticket.reference_price
                if ticket is not None
                else None
            )
            if price is not None:
                refs = (
                    (place.business_fact_reference_id or place.fact_reference_id,)
                    if category is CostCategory.DINING and place is not None
                    else (ticket.fact_reference_id, *ticket.source_offer_ids)
                    if ticket is not None
                    else ()
                )
                lines_by_date[day.service_date].append(
                    _reference_cost_line(
                        subject_kind=CostSubjectKind.ACTIVITY,
                        subject_id=activity.activity_id,
                        service_date=day.service_date,
                        category=category,
                        basis=CostPriceBasis.PER_PERSON,
                        price=price,
                        divisor=1,
                        fetched_at=(
                            place.business_observed_at or place.observed_at
                            if category is CostCategory.DINING and place is not None
                            else ticket.observed_at
                            if ticket is not None
                            else now
                        ),
                        source_reference_ids=refs,
                    )
                )
                continue
            lines_by_date[day.service_date].append(
                _missing_cost_line(
                    subject_kind=CostSubjectKind.ACTIVITY,
                    subject_id=activity.activity_id,
                    service_date=day.service_date,
                    category=category,
                    basis=CostPriceBasis.PER_PERSON,
                    fetched_at=now,
                    reason=(
                        "非免费，门票价格暂缺。"
                        if ticket is not None
                        and category is CostCategory.ATTRACTION_TICKETS
                        and ticket.admission_status == "paid"
                        else "当前 Provider Observation 未提供可核验价格；不会按零元计算。"
                    ),
                    source_reference_ids=activity.source_reference_ids,
                )
            )
        assert workspace.working_itinerary is not None
        semantic_day = next(
            d for d in workspace.working_itinerary.days if d.service_date == day.service_date
        )
        for item in semantic_day.ordered_items:
            if item.onsite_lunch:
                lines_by_date[day.service_date].append(
                    _missing_cost_line(
                        subject_kind=CostSubjectKind.ACTIVITY,
                        subject_id=UUID(
                            server_id(item.draft_item_id, day.service_date, "activity")
                        ),
                        service_date=day.service_date,
                        category=CostCategory.DINING,
                        basis=CostPriceBasis.PER_PERSON,
                        fetched_at=now,
                        reason="园内午餐尚未核验具体餐厅与价格，不按免费计算。",
                        source_reference_ids=(),
                    )
                )
        for leg in day.transport_legs:
            edge = next((edges[ref] for ref in leg.source_reference_ids if ref in edges), None)
            price = edge.fare if edge is not None else None
            walking = (
                leg.mode is RouteMode.WALKING and leg.availability is not DataAvailability.MISSING
            )
            if walking:
                price = CnyAmountRange(minimum_fen=0, maximum_fen=0)
            if price is not None:
                vehicle = leg.mode is RouteMode.DRIVING
                cars = ceil(party_size / 4) if vehicle else 1
                lines_by_date[day.service_date].append(
                    _reference_cost_line(
                        subject_kind=CostSubjectKind.TRANSPORT_LEG,
                        subject_id=leg.leg_id,
                        service_date=day.service_date,
                        category=CostCategory.LOCAL_TRANSPORT,
                        basis=CostPriceBasis.PER_VEHICLE if vehicle else CostPriceBasis.PER_PERSON,
                        price=CnyAmountRange(
                            minimum_fen=price.minimum_fen * cars,
                            maximum_fen=price.maximum_fen * cars,
                        ),
                        divisor=party_size if vehicle else 1,
                        fetched_at=now,
                        source_reference_ids=leg.source_reference_ids,
                        reference_only=not walking,
                    )
                )
                continue
            lines_by_date[day.service_date].append(
                _missing_cost_line(
                    subject_kind=CostSubjectKind.TRANSPORT_LEG,
                    subject_id=leg.leg_id,
                    service_date=day.service_date,
                    category=CostCategory.LOCAL_TRANSPORT,
                    basis=CostPriceBasis.PER_VEHICLE,
                    fetched_at=now,
                    reason="路线 Observation 没有真实车费字段；不会由模型估价。",
                    source_reference_ids=leg.source_reference_ids,
                )
            )

    _add_lodging_lines(
        lines_by_date,
        workspace,
        book,
        now,
        lodging_divisor=lodging_divisor,
    )
    all_lines = tuple(line for day in schedule.days for line in lines_by_date[day.service_date])
    days = tuple(
        DraftDailyCostEstimate(
            service_date=day.service_date,
            lines=tuple(lines_by_date[day.service_date]),
            categories=tuple(
                _summarize_cost(category, lines_by_date[day.service_date])
                for category in _CATEGORY_ORDER
            ),
            known_subtotal_per_person=_sum_amounts(
                line.amount_per_person for line in lines_by_date[day.service_date]
            ),
        )
        for day in schedule.days
    )
    return CostValidationDraft(
        algorithm_version=COST_COMPILER_VERSION,
        request_id=UUID(server_id(schedule.request_id, "cost")),
        schedule_request_id=schedule.request_id,
        trip_id=schedule.trip_id,
        input_state_version=schedule.input_state_version,
        task_book_id=schedule.task_book_id,
        task_book_revision=schedule.task_book_revision,
        start_date=schedule.start_date,
        end_date=schedule.end_date,
        currency="CNY",
        per_person=True,
        party_size=party_size,
        lodging_share_divisor=lodging_divisor,
        days=days,
        categories=tuple(_summarize_cost(category, all_lines) for category in _CATEGORY_ORDER),
        known_total_per_person=_sum_amounts(line.amount_per_person for line in all_lines),
        excluded_costs=tuple(ExcludedCostKind),
        pricing_note=f"每人参考预算，按{party_size}人分摊；不含往返大交通和个人购物。",
        generated_at=now,
    )


def _add_lodging_lines(
    lines_by_date: dict[date, list[DraftCostEstimateLine]],
    workspace: PlannerWorkspaceState,
    book: TaskBookV4,
    now: datetime,
    *,
    lodging_divisor: int,
) -> None:
    draft = workspace.working_itinerary
    if draft is None:
        return
    nights = (book.destination_and_dates.end_date - book.destination_and_dates.start_date).days
    if nights <= 0:
        return
    fixed_ref = draft.lodging_baseline.fixed_commitment_ref
    if fixed_ref is not None:
        fixed_booking = book.lodging_direction.existing_booking
        source_reference_ids = (
            tuple(fixed_booking.source_operation_refs)
            if fixed_booking is not None and fixed_booking.booking_id == fixed_ref.commitment_id
            else (fixed_ref.commitment_id,)
        )
        subject_id = UUID(server_id("fixed-hotel", fixed_ref.commitment_id))
        for offset in range(nights):
            service_date = date.fromordinal(
                book.destination_and_dates.start_date.toordinal() + offset
            )
            if service_date not in lines_by_date:
                continue
            lines_by_date[service_date].append(
                _missing_cost_line(
                    subject_kind=CostSubjectKind.HOTEL_NIGHT,
                    subject_id=subject_id,
                    service_date=service_date,
                    category=CostCategory.LODGING,
                    basis=CostPriceBasis.PER_ROOM_NIGHT,
                    fetched_at=now,
                    reason=(
                        "用户已有固定住宿，但任务书没有可核验的住宿金额；"
                        "该费用不计入已知总额，也不按零元处理。"
                    ),
                    source_reference_ids=source_reference_ids,
                )
            )
        return
    if draft.lodging_baseline.selected_offer_ref is None:
        if draft.lodging_baseline.mode == "unresolved":
            for service_date in sorted(lines_by_date)[:nights]:
                lines_by_date[service_date].append(
                    _missing_cost_line(
                        subject_kind=CostSubjectKind.HOTEL_NIGHT,
                        subject_id=UUID(server_id(draft.draft_id, "unresolved-hotel")),
                        service_date=service_date,
                        category=CostCategory.LODGING,
                        basis=CostPriceBasis.PER_ROOM_NIGHT,
                        fetched_at=now,
                        reason="本次需要住宿，但尚无已核验酒店与房价；费用仍然缺失，不是无需住宿。",
                        source_reference_ids=(),
                    )
                )
        return
    observation = workspace.hotel_observation
    offer_ref = draft.lodging_baseline.selected_offer_ref
    offer = next(
        (
            value
            for value in (observation.offers if observation is not None else ())
            if value.offer_ref == offer_ref
        ),
        None,
    )
    subject_id = UUID(server_id("hotel-property", offer_ref.property_id))
    for offset in range(nights):
        service_date = date.fromordinal(book.destination_and_dates.start_date.toordinal() + offset)
        if service_date not in lines_by_date:
            continue
        price = (offer.total_price or offer.reference_price) if offer is not None else None
        if price is None or price.currency != "CNY":
            lines_by_date[service_date].append(
                _missing_cost_line(
                    subject_kind=CostSubjectKind.HOTEL_NIGHT,
                    subject_id=subject_id,
                    service_date=service_date,
                    category=CostCategory.LODGING,
                    basis=CostPriceBasis.PER_ROOM_NIGHT,
                    fetched_at=now,
                    reason="当前酒店报价缺失或币种尚未转换，不能生成金额。",
                    source_reference_ids=offer.source_reference_ids if offer is not None else (),
                )
            )
            continue
        assert offer is not None
        if offer.total_price is not None:
            minimum_per_night = maximum_per_night = offer.total_price.amount_minor / nights
        else:
            assert offer.reference_price is not None
            minimum_per_night = offer.reference_price.minimum_minor
            maximum_per_night = offer.reference_price.maximum_minor
        amount = DraftAmountRange(
            currency="CNY",
            minimum_fen=floor(minimum_per_night / lodging_divisor),
            maximum_fen=ceil(maximum_per_night / lodging_divisor),
        )
        lines_by_date[service_date].append(
            DraftCostEstimateLine(
                price_fact_id=server_id(offer_ref.offer_id, service_date, "hotel-price"),
                subject_kind=CostSubjectKind.HOTEL_NIGHT,
                subject_id=subject_id,
                service_date=service_date,
                category=CostCategory.LODGING,
                basis=CostPriceBasis.PER_ROOM_NIGHT,
                share_divisor=lodging_divisor,
                availability=DataAvailability.AVAILABLE
                if offer.total_price is not None
                else DataAvailability.PARTIAL,
                missing_reason=None
                if offer.total_price is not None
                else "每晚参考价，实际房型和入住日期价格待确认。",
                amount_per_person=amount,
                original_amount={
                    "currency": "CNY",
                    "minimum_minor": floor(minimum_per_night),
                    "maximum_minor": ceil(maximum_per_night),
                },
                source_reference_ids=tuple(
                    dict.fromkeys(
                        (
                            *offer.room_and_price_fact_refs,
                            *offer.reference_price_fact_refs,
                            *offer.source_reference_ids,
                        )
                    )
                ),
                fetched_at=offer.observed_at,
            )
        )


def _reference_cost_line(
    *,
    subject_kind: CostSubjectKind,
    subject_id: UUID,
    service_date: date,
    category: CostCategory,
    basis: CostPriceBasis,
    price: CnyAmountRange,
    divisor: int,
    fetched_at: datetime,
    source_reference_ids: tuple[str, ...],
    reference_only: bool = True,
) -> DraftCostEstimateLine:
    return DraftCostEstimateLine(
        price_fact_id=server_id(subject_id, service_date, category.value, "reference-price"),
        subject_kind=subject_kind,
        subject_id=subject_id,
        service_date=service_date,
        category=category,
        basis=basis,
        share_divisor=divisor,
        availability=DataAvailability.PARTIAL if reference_only else DataAvailability.AVAILABLE,
        amount_per_person=DraftAmountRange(
            minimum_fen=floor(price.minimum_fen / divisor),
            maximum_fen=ceil(price.maximum_fen / divisor),
            currency="CNY",
        ),
        original_amount={
            "currency": "CNY",
            "minimum_minor": price.minimum_fen,
            "maximum_minor": price.maximum_fen,
        },
        source_reference_ids=source_reference_ids,
        fetched_at=fetched_at,
        missing_reason="来源参考价，实际消费可能变化。" if reference_only else None,
    )


def _missing_cost_line(
    *,
    subject_kind: CostSubjectKind,
    subject_id: UUID,
    service_date: date,
    category: CostCategory,
    basis: CostPriceBasis,
    fetched_at: datetime,
    reason: str,
    source_reference_ids: tuple[str, ...],
) -> DraftCostEstimateLine:
    return DraftCostEstimateLine(
        price_fact_id=server_id(subject_id, service_date, category.value, "missing-price"),
        subject_kind=subject_kind,
        subject_id=subject_id,
        service_date=service_date,
        category=category,
        basis=basis,
        share_divisor=1,
        availability=DataAvailability.MISSING,
        source_reference_ids=source_reference_ids,
        fetched_at=fetched_at,
        missing_reason=reason,
    )


def _summarize_cost(
    category: CostCategory,
    lines: Iterable[DraftCostEstimateLine],
) -> DraftCategoryCostSummary:
    selected = tuple(line for line in lines if line.category is category)
    known = tuple(line.amount_per_person for line in selected if line.amount_per_person is not None)
    missing = sum(line.availability is DataAvailability.MISSING for line in selected)
    status = (
        CostCoverageStatus.NOT_APPLICABLE
        if not selected
        else CostCoverageStatus.MISSING
        if not known
        else CostCoverageStatus.PARTIAL
        if missing or any(line.availability is DataAvailability.PARTIAL for line in selected)
        else CostCoverageStatus.AVAILABLE
    )
    return DraftCategoryCostSummary(
        category=category,
        status=status,
        amount_per_person=_sum_amounts(known),
        priced_item_count=len(known),
        missing_item_count=missing,
        source_reference_ids=tuple(
            dict.fromkeys(reference for line in selected for reference in line.source_reference_ids)
        ),
        note=("部分或全部价格仍缺少真实来源。" if missing else None),
    )


def _sum_amounts(values: Iterable[DraftAmountRange | None]) -> DraftAmountRange | None:
    present = tuple(value for value in values if value is not None)
    if not present:
        return None
    return DraftAmountRange(
        currency="CNY",
        minimum_fen=sum(value.minimum_fen for value in present),
        maximum_fen=sum(value.maximum_fen for value in present),
    )


def _party_size(book: TaskBookV4) -> int:
    return parse_party_size(book.travelers_and_trip_goal.travelers)


def _meal_label(meal: str) -> str:
    return {"breakfast": "早餐", "lunch": "午餐", "dinner": "晚餐", "snack": "加餐"}[meal]


@overload
def _time_minutes(value: time) -> int: ...


@overload
def _time_minutes(value: None) -> None: ...


def _time_minutes(value: time | None) -> int | None:
    return value.hour * 60 + value.minute if value is not None else None


def _as_time(value: int) -> time:
    bounded = max(0, min(23 * 60 + 59, value))
    return time(hour=bounded // 60, minute=bounded % 60)
