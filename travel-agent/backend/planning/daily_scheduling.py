"""Deterministic V3-37 insertion scheduler over confirmed, source-backed facts."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from uuid import NAMESPACE_URL, UUID, uuid5

from backend.contracts.daily_scheduling import (
    DailyScheduleResult,
    DailyScheduleWindow,
    DailySchedulingRequest,
    ScheduleActivityKind,
    ScheduledActivity,
    ScheduledDay,
    ScheduledPause,
    ScheduledTransport,
    SchedulePauseKind,
    SchedulePlaceFact,
    ScheduleRouteFact,
    UnscheduledStrongDesire,
)
from backend.contracts.enums import AnchorRole, DataAvailability, PlaceCategory
from backend.contracts.spatial_planning import SpatialAnchor
from backend.providers.contracts import HoursDayStatus

DAILY_SCHEDULING_ALGORITHM_VERSION = "1.1.0"

_ROLE_PRIORITY = {
    AnchorRole.MUST_ATTRACTION: 0,
    AnchorRole.DESTINATION_RESTAURANT: 1,
    AnchorRole.WANT_ATTRACTION: 2,
    AnchorRole.CONVENIENT_RESTAURANT: 3,
    AnchorRole.CONVENIENT_ATTRACTION: 4,
}
_STRONG_ROLES = {
    AnchorRole.MUST_ATTRACTION,
    AnchorRole.DESTINATION_RESTAURANT,
    AnchorRole.WANT_ATTRACTION,
}


class DailySchedulingError(ValueError):
    """A deterministic conflict that cannot produce a safe schedule."""


@dataclass(frozen=True)
class _Materialized:
    day: ScheduledDay
    provider_fact_ids: tuple[str, ...]


class DailySchedulingService:
    def __init__(
        self,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._clock = clock

    def build(self, request: DailySchedulingRequest) -> DailyScheduleResult:
        request = DailySchedulingRequest.model_validate(
            request.model_dump(mode="json"), context={"today": request.business_date}
        )
        generated_at = self._aware_now()
        places = {item.place_id: item for item in request.places}
        routes = _route_index(request.routes)
        windows = {item.service_date: item for item in request.daily_windows}
        cluster_by_node = {
            node_id: cluster.cluster_id
            for cluster in request.spatial_result.clusters
            for node_id in cluster.member_node_ids
        }
        sequences: dict[date, list[SpatialAnchor]] = {service_date: [] for service_date in windows}
        for anchor in request.spatial_result.anchors:
            if anchor.role is AnchorRole.FIXED_HOTEL:
                continue
            if anchor.role is AnchorRole.FIXED_EVENT:
                fixed = anchor.fixed_time_window
                if fixed is None:
                    raise DailySchedulingError("fixed event is missing its time window")
                sequences[fixed.event_date].append(anchor)
        for sequence in sequences.values():
            sequence.sort(key=_fixed_sort_key)

        unscheduled: list[UnscheduledStrongDesire] = []
        optional = sorted(
            (
                anchor
                for anchor in request.spatial_result.anchors
                if anchor.role not in {AnchorRole.FIXED_EVENT, AnchorRole.FIXED_HOTEL}
            ),
            key=lambda item: (
                _ROLE_PRIORITY[item.role],
                str(cluster_by_node.get(item.node_id, "")),
                item.name,
                str(item.node_id),
            ),
        )

        for anchor in optional:
            best: tuple[tuple[object, ...], date, list[SpatialAnchor]] | None = None
            for service_date in anchor.available_dates:
                current_sequence = sequences.get(service_date)
                if current_sequence is None:
                    continue
                for index in range(len(current_sequence) + 1):
                    proposed = [
                        *current_sequence[:index],
                        anchor,
                        *current_sequence[index:],
                    ]
                    materialized = _materialize_day(
                        request, windows[service_date], proposed, places, routes
                    )
                    if materialized is None:
                        continue
                    same_cluster = sum(
                        cluster_by_node.get(existing.node_id) == cluster_by_node.get(anchor.node_id)
                        for existing in current_sequence
                    )
                    score = (
                        len(materialized.day.activities),
                        materialized.day.active_minutes,
                        -same_cluster,
                        service_date,
                        index,
                    )
                    if best is None or score < best[0]:
                        best = (score, service_date, proposed)
            if best is not None:
                sequences[best[1]] = best[2]
            elif anchor.role in _STRONG_ROLES:
                unscheduled.append(
                    UnscheduledStrongDesire(
                        node_id=anchor.node_id,
                        place_id=anchor.place_id,
                        role=anchor.role,
                        reason=_unscheduled_reason(anchor, places),
                        source_reference_ids=anchor.source_reference_ids,
                    )
                )

        days: list[ScheduledDay] = []
        fact_ids: list[str] = []
        degradation: list[str] = []
        for service_date in windows:
            materialized = _materialize_day(
                request,
                windows[service_date],
                sequences[service_date],
                places,
                routes,
            )
            if materialized is None:
                raise DailySchedulingError(
                    f"fixed arrangements cannot fit {service_date.isoformat()}"
                )
            days.append(materialized.day)
            fact_ids.extend(materialized.provider_fact_ids)
            for activity in materialized.day.activities:
                if activity.availability is DataAvailability.PARTIAL:
                    degradation.append(
                        f"{activity.title}：{activity.missing_reason or '营业资料不完整'}"
                    )
            if any(
                item.availability is DataAvailability.PARTIAL
                for item in materialized.day.transport_legs
            ):
                degradation.append(f"partial route facts on {service_date.isoformat()}")
        if unscheduled:
            degradation.append("one or more strong desires could not be scheduled safely")
        return DailyScheduleResult(
            algorithm_version=DAILY_SCHEDULING_ALGORITHM_VERSION,
            request_id=request.request_id,
            trip_id=request.trip_id,
            input_state_version=request.input_state_version,
            task_book_id=request.task_book.task_book_id,
            task_book_revision=request.task_book.revision,
            city_id=request.city_id,
            start_date=request.task_book.date_range.start_date,
            end_date=request.task_book.date_range.end_date,
            status=DataAvailability.PARTIAL if degradation else DataAvailability.AVAILABLE,
            days=tuple(days),
            unscheduled_strong_desires=tuple(unscheduled),
            degradation_reasons=tuple(dict.fromkeys(degradation)),
            provider_fact_ids=tuple(dict.fromkeys(fact_ids)),
            generated_at=generated_at,
        )

    def _aware_now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("daily scheduling clock must return an aware datetime")
        return value.astimezone(UTC)


def _materialize_day(
    request: DailySchedulingRequest,
    window: DailyScheduleWindow,
    sequence: list[SpatialAnchor],
    places: dict[UUID, SchedulePlaceFact],
    routes: dict[tuple[UUID, UUID], tuple[ScheduleRouteFact, ...]],
) -> _Materialized | None:
    pauses = _planned_pauses(request, window, sequence)
    current_minutes = _time_minutes(window.start_time)
    end_minutes = _time_minutes(window.end_time)
    current_place = window.start_place_id
    activities: list[ScheduledActivity] = []
    legs: list[ScheduledTransport] = []
    facts: list[str] = []
    restaurant_index = 0

    for anchor in sequence:
        place = places[anchor.place_id]
        fixed = anchor.fixed_time_window
        meal_window = None
        if place.category is PlaceCategory.RESTAURANT and fixed is None:
            meal_window = _meal_window(request, restaurant_index)
            if meal_window is None:
                return None
        latest_arrival = (
            _time_minutes(fixed.start_time)
            if fixed is not None
            else meal_window[4]
            if meal_window is not None
            else None
        )
        routed = _scheduled_route(
            request,
            window.service_date,
            current_place,
            anchor.place_id,
            current_minutes,
            pauses,
            routes,
            prefix="leg",
            latest_arrival=latest_arrival,
        )
        if current_place != anchor.place_id and routed is None:
            return None
        leg, arrival, route_fact_id = routed or (None, current_minutes, None)
        duration = (
            _time_minutes(fixed.end_time) - _time_minutes(fixed.start_time)
            if fixed is not None
            else request.preferences.meal_duration_minutes
            if place.category is PlaceCategory.RESTAURANT
            else _paced_duration(place.recommended_duration_minutes, request.preferences.pace_level)
        )
        if fixed is not None:
            start = _time_minutes(fixed.start_time)
            if arrival > start:
                return None
            finish = _time_minutes(fixed.end_time)
            (
                availability,
                missing_reason,
                opening_sources,
            ) = _assess_fixed_opening(
                place,
                window.service_date,
                start,
                finish,
            )
            activity_sources = tuple(
                dict.fromkeys(
                    (
                        *anchor.source_reference_ids,
                        *place.source_reference_ids,
                        *opening_sources,
                    )
                )
            )
            facts.extend(opening_sources)
            timing_notice = None
        else:
            earliest = arrival
            timing_notice = None
            if place.category is PlaceCategory.RESTAURANT:
                if meal_window is None:
                    return None
                meal_name, acceptable_start, ideal_start, ideal_end, acceptable_end = meal_window
                fitted = _fit_meal(
                    place,
                    window.service_date,
                    earliest,
                    duration,
                    pauses,
                    meal_name=meal_name,
                    acceptable_start=acceptable_start,
                    ideal_start=ideal_start,
                    ideal_end=ideal_end,
                    acceptable_end=acceptable_end,
                )
                restaurant_index += 1
            else:
                earliest = _avoid_pauses(earliest, duration, pauses)
                fitted = _fit_opening(
                    place,
                    window.service_date,
                    earliest,
                    duration,
                    pauses,
                )
            if fitted is None:
                return None
            start, availability, missing_reason, opening_sources, timing_notice = fitted
            finish = start + duration
            activity_sources = tuple(
                dict.fromkeys(
                    (
                        *anchor.source_reference_ids,
                        *place.source_reference_ids,
                        *opening_sources,
                    )
                )
            )
            facts.extend(opening_sources)
        if finish > end_minutes:
            return None
        if leg is not None:
            legs.append(leg)
        if route_fact_id is not None:
            facts.append(route_fact_id)
        activities.append(
            ScheduledActivity(
                activity_id=_stable_id(
                    request.trip_id, f"activity:{window.service_date}:{anchor.node_id}"
                ),
                node_id=anchor.node_id,
                place_id=anchor.place_id,
                kind=_activity_kind(anchor, place),
                role=anchor.role,
                title=anchor.name,
                service_date=window.service_date,
                start_time=_as_time(start),
                end_time=_as_time(finish),
                duration_minutes=finish - start,
                availability=availability,
                source_reference_ids=activity_sources,
                missing_reason=missing_reason,
                timing_notice=timing_notice,
            )
        )
        current_minutes = finish
        current_place = anchor.place_id

    routed = _scheduled_route(
        request,
        window.service_date,
        current_place,
        window.end_place_id,
        current_minutes,
        pauses,
        routes,
        prefix="return",
        latest_arrival=end_minutes,
    )
    if current_place != window.end_place_id and routed is None:
        return None
    if routed is not None:
        leg, arrival, route_fact_id = routed
        if arrival > end_minutes:
            return None
        if leg is not None:
            legs.append(leg)
        if route_fact_id is not None:
            facts.append(route_fact_id)

    used_pauses = tuple(
        ScheduledPause(
            pause_id=_stable_id(
                request.trip_id,
                f"pause:{window.service_date}:{kind}:{start}:{finish}",
            ),
            kind=kind,
            service_date=window.service_date,
            start_time=_as_time(start),
            end_time=_as_time(finish),
            duration_minutes=finish - start,
            reason=("预留正常用餐时间" if kind is SchedulePauseKind.MEAL else "按旅行节奏预留休息"),
        )
        for kind, start, finish in pauses
    )
    active_minutes = sum(item.duration_minutes for item in activities)
    if active_minutes > request.preferences.maximum_active_minutes:
        return None
    day = ScheduledDay(
        service_date=window.service_date,
        start_place_id=window.start_place_id,
        end_place_id=window.end_place_id,
        start_time=window.start_time,
        end_time=window.end_time,
        activities=tuple(activities),
        pauses=used_pauses,
        transport_legs=tuple(legs),
        active_minutes=active_minutes,
        walking_m=sum(item.walking_m for item in legs),
        cycling_m=sum(item.distance_m for item in legs if item.mode.value == "cycling"),
        meal_minutes=sum(
            item.duration_minutes for item in used_pauses if item.kind is SchedulePauseKind.MEAL
        ),
        rest_minutes=sum(
            item.duration_minutes for item in used_pauses if item.kind is SchedulePauseKind.REST
        ),
        buffer_minutes=sum(item.buffer_minutes for item in legs),
    )
    return _Materialized(day=day, provider_fact_ids=tuple(dict.fromkeys(facts)))


# Helper functions below deliberately contain no model or Provider calls.


def _scheduled_route(
    request: DailySchedulingRequest,
    service_date: date,
    origin: UUID,
    destination: UUID,
    current_minutes: int,
    pauses: tuple[tuple[SchedulePauseKind, int, int], ...],
    routes: dict[tuple[UUID, UUID], tuple[ScheduleRouteFact, ...]],
    *,
    prefix: str,
    latest_arrival: int | None = None,
) -> tuple[ScheduledTransport | None, int, str | None] | None:
    if origin == destination:
        return None, current_minutes, None
    route = _choose_route(
        request,
        origin,
        destination,
        current_minutes,
        pauses,
        routes,
        latest_arrival=latest_arrival,
    )
    if route is None or route.duration_minutes is None:
        return None
    duration_with_buffer = route.duration_minutes + request.preferences.route_buffer_minutes
    departure = _avoid_pauses(current_minutes, duration_with_buffer, pauses)
    arrival = departure + duration_with_buffer
    availability = (
        DataAvailability.PARTIAL
        if route.availability is DataAvailability.PARTIAL
        else DataAvailability.AVAILABLE
    )
    leg = ScheduledTransport(
        leg_id=_stable_id(
            request.trip_id,
            f"{prefix}:{service_date}:{origin}:{destination}:{departure}",
        ),
        origin_place_id=origin,
        destination_place_id=destination,
        departure_time=_as_time(departure),
        arrival_time=_as_time(arrival),
        mode=route.mode,
        availability=availability,
        distance_m=route.distance_m or 0,
        duration_minutes=route.duration_minutes,
        walking_m=route.walking_m or 0,
        buffer_minutes=request.preferences.route_buffer_minutes,
        source_reference_ids=route.source_reference_ids,
        missing_reason=(route.missing_reason if availability is DataAvailability.PARTIAL else None),
    )
    return leg, arrival, route.route_fact_id


def _planned_pauses(
    request: DailySchedulingRequest,
    window: DailyScheduleWindow,
    sequence: list[SpatialAnchor],
) -> tuple[tuple[SchedulePauseKind, int, int], ...]:
    if not sequence:
        return ()
    fixed_intervals = [
        (
            _time_minutes(anchor.fixed_time_window.start_time),
            _time_minutes(anchor.fixed_time_window.end_time),
        )
        for anchor in sequence
        if anchor.fixed_time_window is not None
    ]
    restaurant_count = sum(
        anchor.role in {AnchorRole.DESTINATION_RESTAURANT, AnchorRole.CONVENIENT_RESTAURANT}
        for anchor in sequence
    )
    proposals: list[tuple[SchedulePauseKind, int, int]] = []
    if restaurant_count == 0:
        proposals.append(
            _pause_within(
                SchedulePauseKind.MEAL,
                request.preferences.lunch_ideal_start,
                request.preferences.lunch_ideal_end,
                request.preferences.meal_duration_minutes,
            )
        )
    if (
        restaurant_count < 2
        and _time_minutes(window.end_time)
        >= _time_minutes(request.preferences.dinner_ideal_start)
        + request.preferences.meal_duration_minutes
    ):
        proposals.append(
            _pause_within(
                SchedulePauseKind.MEAL,
                request.preferences.dinner_ideal_start,
                request.preferences.dinner_ideal_end,
                request.preferences.meal_duration_minutes,
            )
        )
    rest_duration = {1: 0, 2: 0, 3: 15, 4: 30, 5: 45}[request.preferences.pace_level]
    if rest_duration:
        proposals.append((SchedulePauseKind.REST, 15 * 60, 15 * 60 + rest_duration))
    safe: list[tuple[SchedulePauseKind, int, int]] = []
    for proposal in proposals:
        _, start, finish = proposal
        if start < _time_minutes(window.start_time) or finish > _time_minutes(window.end_time):
            continue
        if any(_overlaps(start, finish, left, right) for left, right in fixed_intervals):
            continue
        if any(_overlaps(start, finish, left, right) for _, left, right in safe):
            continue
        safe.append(proposal)
    return tuple(sorted(safe, key=lambda value: value[1]))


def _pause_within(
    kind: SchedulePauseKind,
    window_start: time,
    window_end: time,
    duration: int,
) -> tuple[SchedulePauseKind, int, int]:
    start = _time_minutes(window_start)
    return kind, start, min(start + duration, _time_minutes(window_end))


def _avoid_pauses(
    start: int,
    duration: int,
    pauses: Iterable[tuple[SchedulePauseKind, int, int]],
) -> int:
    adjusted = start
    for _, pause_start, pause_end in pauses:
        if _overlaps(adjusted, adjusted + duration, pause_start, pause_end):
            adjusted = pause_end
    return adjusted


def _fit_opening(
    place: SchedulePlaceFact,
    service_date: date,
    earliest: int,
    duration: int,
    pauses: tuple[tuple[SchedulePauseKind, int, int], ...],
    *,
    latest_start: int | None = None,
) -> tuple[int, DataAvailability, str | None, tuple[str, ...], str | None] | None:
    date_status = next(
        (item for item in place.opening_dates if item.service_date == service_date), None
    )
    if date_status is not None and date_status.status is not HoursDayStatus.OPEN:
        return None
    windows = sorted(
        (item for item in place.opening_windows if item.service_date == service_date),
        key=lambda item: item.start_time,
    )
    if not windows:
        # Unknown, conflicted and closed dates do not provide a scheduling window.
        return None
    for opening in windows:
        start = max(earliest, _time_minutes(opening.start_time))
        start = _avoid_pauses(start, duration, pauses)
        if latest_start is not None and start > latest_start:
            continue
        if opening.last_entry_at is not None and start >= _time_minutes(opening.last_entry_at):
            continue
        if start + duration <= _time_minutes(opening.end_time):
            availability = (
                DataAvailability.PARTIAL
                if date_status is None and place.opening_availability is DataAvailability.PARTIAL
                else DataAvailability.AVAILABLE
            )
            return (
                start,
                availability,
                (
                    place.opening_missing_reason
                    if availability is DataAvailability.PARTIAL
                    else None
                ),
                opening.source_reference_ids,
                None,
            )
    return None


def _fit_meal(
    place: SchedulePlaceFact,
    service_date: date,
    arrival: int,
    duration: int,
    pauses: tuple[tuple[SchedulePauseKind, int, int], ...],
    *,
    meal_name: str,
    acceptable_start: int,
    ideal_start: int,
    ideal_end: int,
    acceptable_end: int,
) -> tuple[int, DataAvailability, str | None, tuple[str, ...], str | None] | None:
    ideal = _fit_opening(
        place,
        service_date,
        max(arrival, ideal_start),
        duration,
        pauses,
        latest_start=ideal_end,
    )
    if ideal is not None:
        return ideal
    acceptable = _fit_opening(
        place,
        service_date,
        max(arrival, acceptable_start),
        duration,
        pauses,
        latest_start=acceptable_end,
    )
    if acceptable is None:
        return None
    start, availability, missing_reason, sources, _ = acceptable
    if start < ideal_start:
        notice = f"{meal_name}时间稍早：为适配餐厅营业时间与当前路线，已安排在可接受时段内。"
    elif start > ideal_end:
        notice = f"{meal_name}时间稍晚：因前序游览和交通衔接，已安排在可接受时段内。"
    else:
        notice = None
    return start, availability, missing_reason, sources, notice


def _assess_fixed_opening(
    place: SchedulePlaceFact,
    service_date: date,
    start: int,
    finish: int,
) -> tuple[DataAvailability, str | None, tuple[str, ...]]:
    date_status = next(
        (item for item in place.opening_dates if item.service_date == service_date), None
    )
    if date_status is not None and date_status.status in (
        HoursDayStatus.UNKNOWN,
        HoursDayStatus.CONFLICT,
    ):
        return (
            DataAvailability.PARTIAL,
            f"固定事项保持原时间，营业规则待核实：{date_status.reason}",
            place.source_reference_ids,
        )
    windows = tuple(item for item in place.opening_windows if item.service_date == service_date)
    sources = tuple(
        dict.fromkeys(source for opening in windows for source in opening.source_reference_ids)
    )
    covering = next(
        (
            opening
            for opening in windows
            if _time_minutes(opening.start_time) <= start
            and finish <= _time_minutes(opening.end_time)
            and (opening.last_entry_at is None or start < _time_minutes(opening.last_entry_at))
        ),
        None,
    )
    if covering is not None:
        if date_status is None and place.opening_availability is DataAvailability.PARTIAL:
            return (
                DataAvailability.PARTIAL,
                place.opening_missing_reason,
                covering.source_reference_ids,
            )
        return DataAvailability.AVAILABLE, None, covering.source_reference_ids
    if place.opening_availability is DataAvailability.MISSING:
        return (
            DataAvailability.PARTIAL,
            f"固定事项营业资料缺失：{place.opening_missing_reason}",
            place.source_reference_ids,
        )
    window_text = "、".join(
        f"{item.start_time.strftime('%H:%M')}–{item.end_time.strftime('%H:%M')}" for item in windows
    )
    detail = window_text or "当日没有可用营业窗口"
    return (
        DataAvailability.PARTIAL,
        "固定事项保持原时间，但与营业时间冲突："
        f"事项为{_as_time(start).strftime('%H:%M')}–{_as_time(finish).strftime('%H:%M')}，"
        f"营业资料为{detail}",
        sources or place.source_reference_ids,
    )


def _choose_route(
    request: DailySchedulingRequest,
    origin: UUID,
    destination: UUID,
    current_minutes: int,
    pauses: tuple[tuple[SchedulePauseKind, int, int], ...],
    routes: dict[tuple[UUID, UUID], tuple[ScheduleRouteFact, ...]],
    *,
    latest_arrival: int | None = None,
) -> ScheduleRouteFact | None:
    mode_rank = {mode: index for index, mode in enumerate(request.preferences.allowed_modes)}
    options = [
        item
        for item in routes.get((origin, destination), ())
        if item.availability is not DataAvailability.MISSING
        and item.mode in mode_rank
        and (item.walking_m or 0) <= request.preferences.maximum_walking_m_per_leg
        and (
            latest_arrival is None
            or item.duration_minutes is not None
            and _avoid_pauses(
                current_minutes,
                item.duration_minutes + request.preferences.route_buffer_minutes,
                pauses,
            )
            + item.duration_minutes
            + request.preferences.route_buffer_minutes
            <= latest_arrival
        )
    ]
    if not options:
        return None
    return min(
        options,
        key=lambda item: (
            mode_rank[item.mode],
            item.duration_minutes or 10_000,
            item.walking_m or 0,
            item.route_fact_id,
        ),
    )


def _meal_window(
    request: DailySchedulingRequest,
    restaurant_index: int,
) -> tuple[str, int, int, int, int] | None:
    preferences = request.preferences
    if restaurant_index == 0:
        return (
            "午餐",
            _time_minutes(preferences.lunch_acceptable_start),
            _time_minutes(preferences.lunch_ideal_start),
            _time_minutes(preferences.lunch_ideal_end),
            _time_minutes(preferences.lunch_acceptable_end),
        )
    if restaurant_index == 1:
        return (
            "晚餐",
            _time_minutes(preferences.dinner_acceptable_start),
            _time_minutes(preferences.dinner_ideal_start),
            _time_minutes(preferences.dinner_ideal_end),
            _time_minutes(preferences.dinner_acceptable_end),
        )
    return None


def _route_index(
    routes: tuple[ScheduleRouteFact, ...],
) -> dict[tuple[UUID, UUID], tuple[ScheduleRouteFact, ...]]:
    grouped: dict[tuple[UUID, UUID], list[ScheduleRouteFact]] = {}
    for route in routes:
        grouped.setdefault((route.origin_place_id, route.destination_place_id), []).append(route)
    return {key: tuple(value) for key, value in grouped.items()}


def _paced_duration(base: int, pace_level: int) -> int:
    factor = {1: 0.75, 2: 0.9, 3: 1.0, 4: 1.15, 5: 1.3}[pace_level]
    return max(15, round(base * factor / 5) * 5)


def _activity_kind(anchor: SpatialAnchor, place: SchedulePlaceFact) -> ScheduleActivityKind:
    if anchor.role is AnchorRole.FIXED_EVENT:
        return ScheduleActivityKind.FIXED_EVENT
    if place.category is PlaceCategory.RESTAURANT:
        return ScheduleActivityKind.RESTAURANT
    return ScheduleActivityKind.ATTRACTION


def _unscheduled_reason(
    anchor: SpatialAnchor,
    places: dict[UUID, SchedulePlaceFact],
) -> str:
    place = places[anchor.place_id]
    if place.category is PlaceCategory.RESTAURANT:
        return (
            "在调整次要景点、路线和顺序后，仍无法把餐厅安排进午餐或晚餐的"
            "可接受开始时段，因此未静默生成不合理用餐时间。"
        )
    if place.opening_availability is DataAvailability.MISSING:
        return f"营业资料缺失：{place.opening_missing_reason}"
    return (
        "在可用日期、营业窗口、路线、行动限制和每日节奏上限内无法安全安排，"
        f"因此未静默加入“{anchor.name}”。"
    )


def _fixed_sort_key(anchor: SpatialAnchor) -> tuple[object, ...]:
    fixed = anchor.fixed_time_window
    return (
        fixed.start_time if fixed is not None else time.max,
        anchor.name,
        str(anchor.node_id),
    )


def _stable_id(trip_id: UUID, value: str) -> UUID:
    return uuid5(NAMESPACE_URL, f"iter:schedule:{trip_id}:{value}")


def _time_minutes(value: time) -> int:
    return value.hour * 60 + value.minute


def _as_time(minutes: int) -> time:
    if not 0 <= minutes < 24 * 60:
        raise DailySchedulingError("daily schedule cannot cross midnight")
    return time(minutes // 60, minutes % 60)


def _overlaps(left_start: int, left_end: int, right_start: int, right_end: int) -> bool:
    return left_start < right_end and right_start < left_end
