"""Deterministic V3-39 validation over the selected, costed itinerary."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import UTC, date, datetime, time
from uuid import NAMESPACE_URL, UUID, uuid5

from backend.contracts.cost_estimation import TripCostEstimate
from backend.contracts.daily_scheduling import DailyScheduleResult, ScheduleActivityKind
from backend.contracts.enums import AnchorRole, CostCategory, DataAvailability
from backend.contracts.itinerary_draft import (
    DraftAmountRange,
    DraftCategoryCostSummary,
    DraftCostEstimateLine,
    DraftScheduledDay,
)
from backend.contracts.itinerary_validation import (
    ItineraryValidationRequest,
    ItineraryValidationResult,
    RepairAction,
    ValidationIssue,
    ValidationIssueCode,
    ValidationSeverity,
    ValidationStatus,
    ValidationTargetKind,
)
from backend.providers.contracts import HoursDayStatus, RouteMode

ITINERARY_VALIDATION_ALGORITHM_VERSION = "1.1.0"


class ItineraryValidationService:
    def __init__(
        self,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._clock = clock

    def validate(self, request: ItineraryValidationRequest) -> ItineraryValidationResult:
        request = ItineraryValidationRequest.model_validate(
            request.model_dump(mode="json"),
            context={"today": request.scheduling_request.business_date},
        )
        issues: list[ValidationIssue] = []
        place_facts = {item.place_id: item for item in request.scheduling_request.places}
        preferences = request.scheduling_request.preferences
        selected_hotel = request.scheduling_request.hotel_result.selected_hotel_place_id
        schedule = request.schedule_draft
        cost = request.cost_draft

        issues.extend(self._identity_issues(request))
        issues.extend(self._cost_issues(request))

        for day in schedule.days:
            issues.extend(self._day_structure_issues(request, day))
            if day.active_minutes > preferences.maximum_active_minutes:
                issues.append(
                    self._issue(
                        request,
                        code=ValidationIssueCode.DAILY_LOAD,
                        severity=ValidationSeverity.SOFT_RISK,
                        target=ValidationTargetKind.DAY,
                        message="当天活动量超过本次节奏对应的建议上限。",
                        service_date=day.service_date,
                        repair=RepairAction.REASSIGN_DAY,
                    )
                )
            if day.walking_m > preferences.maximum_walking_m_per_day:
                issues.append(
                    self._issue(
                        request,
                        code=ValidationIssueCode.WALKING_LIMIT,
                        severity=ValidationSeverity.HARD_CONFLICT,
                        target=ValidationTargetKind.DAY,
                        message="当天步行距离超过用户接受的上限。",
                        service_date=day.service_date,
                        repair=RepairAction.CHANGE_ROUTE,
                    )
                )
            if day.cycling_m > preferences.maximum_cycling_m_per_day:
                issues.append(
                    self._issue(
                        request,
                        code=ValidationIssueCode.CYCLING_LIMIT,
                        severity=ValidationSeverity.HARD_CONFLICT,
                        target=ValidationTargetKind.DAY,
                        message="当天骑行距离超过用户接受的上限。",
                        service_date=day.service_date,
                        repair=RepairAction.CHANGE_ROUTE,
                    )
                )
            for activity in day.activities:
                place = place_facts.get(activity.place_id)
                date_status = (
                    next(
                        (
                            item
                            for item in place.opening_dates
                            if item.service_date == day.service_date
                        ),
                        None,
                    )
                    if place
                    else None
                )
                if place is not None and (
                    date_status is not None
                    and date_status.status in (HoursDayStatus.UNKNOWN, HoursDayStatus.CONFLICT)
                    or date_status is None
                    and place.opening_availability is DataAvailability.MISSING
                ):
                    issues.append(
                        self._issue(
                            request,
                            code=ValidationIssueCode.OPENING_HOURS_UNKNOWN,
                            severity=ValidationSeverity.UNKNOWN_FACT,
                            target=ValidationTargetKind.ACTIVITY,
                            message=date_status.reason
                            if date_status
                            else place.opening_missing_reason or "营业时间资料暂缺。",
                            service_date=day.service_date,
                            activity_id=activity.activity_id,
                            repair=RepairAction.REQUERY_FACT,
                            sources=place.source_reference_ids,
                        )
                    )
                elif place is not None:
                    windows = tuple(
                        item
                        for item in place.opening_windows
                        if item.service_date == day.service_date
                    )
                    fits = any(
                        _minutes(item.start_time) <= _minutes(activity.start_time)
                        and _minutes(activity.end_time) <= _minutes(item.end_time)
                        and (
                            item.last_entry_at is None
                            or _minutes(activity.start_time) < _minutes(item.last_entry_at)
                        )
                        for item in windows
                    )
                    if not fits:
                        fixed = activity.kind is ScheduleActivityKind.FIXED_EVENT
                        issues.append(
                            self._issue(
                                request,
                                code=(
                                    ValidationIssueCode.FIXED_EVENT_CONFLICT
                                    if fixed
                                    else ValidationIssueCode.OPENING_HOURS_CONFLICT
                                ),
                                severity=ValidationSeverity.HARD_CONFLICT,
                                target=ValidationTargetKind.ACTIVITY,
                                message=(
                                    "固定事项保留原时间，但与地点营业时间冲突。"
                                    if fixed
                                    else "活动时间不在已知营业窗口内。"
                                ),
                                service_date=day.service_date,
                                activity_id=activity.activity_id,
                                repair=(RepairAction.NONE if fixed else RepairAction.REASSIGN_DAY),
                                sources=activity.source_reference_ids,
                            )
                        )
                    elif (
                        date_status is None
                        and place.opening_availability is DataAvailability.PARTIAL
                    ):
                        issues.append(
                            self._issue(
                                request,
                                code=ValidationIssueCode.OPENING_HOURS_UNKNOWN,
                                severity=ValidationSeverity.UNKNOWN_FACT,
                                target=ValidationTargetKind.ACTIVITY,
                                message=place.opening_missing_reason or "营业时间资料不完整。",
                                service_date=day.service_date,
                                activity_id=activity.activity_id,
                                repair=RepairAction.REQUERY_FACT,
                                sources=place.source_reference_ids,
                            )
                        )
                if activity.timing_notice is not None:
                    issues.append(
                        self._issue(
                            request,
                            code=ValidationIssueCode.MEAL_TIMING,
                            severity=ValidationSeverity.SOFT_RISK,
                            target=ValidationTargetKind.ACTIVITY,
                            message=activity.timing_notice,
                            service_date=day.service_date,
                            activity_id=activity.activity_id,
                            repair=RepairAction.REORDER_DAY,
                            sources=activity.source_reference_ids,
                        )
                    )
            for leg in day.transport_legs:
                if leg.mode not in preferences.allowed_modes:
                    issues.append(
                        self._issue(
                            request,
                            code=(
                                ValidationIssueCode.CYCLING_LIMIT
                                if leg.mode is RouteMode.CYCLING
                                else ValidationIssueCode.WALKING_LIMIT
                            ),
                            severity=ValidationSeverity.HARD_CONFLICT,
                            target=ValidationTargetKind.TRANSPORT_LEG,
                            message="路线使用了用户未允许的交通方式。",
                            service_date=day.service_date,
                            transport_leg_id=leg.leg_id,
                            repair=RepairAction.CHANGE_ROUTE,
                            sources=leg.source_reference_ids,
                        )
                    )
                if leg.walking_m > preferences.maximum_walking_m_per_leg:
                    issues.append(
                        self._issue(
                            request,
                            code=ValidationIssueCode.WALKING_LIMIT,
                            severity=ValidationSeverity.HARD_CONFLICT,
                            target=ValidationTargetKind.TRANSPORT_LEG,
                            message="单段路线步行距离超过用户接受的上限。",
                            service_date=day.service_date,
                            transport_leg_id=leg.leg_id,
                            repair=RepairAction.CHANGE_ROUTE,
                            sources=leg.source_reference_ids,
                        )
                    )
                if leg.availability is DataAvailability.PARTIAL:
                    issues.append(
                        self._issue(
                            request,
                            code=ValidationIssueCode.PARTIAL_ROUTE,
                            severity=ValidationSeverity.UNKNOWN_FACT,
                            target=ValidationTargetKind.TRANSPORT_LEG,
                            message=leg.missing_reason or "路线资料不完整。",
                            service_date=day.service_date,
                            transport_leg_id=leg.leg_id,
                            repair=RepairAction.REQUERY_FACT,
                            sources=leg.source_reference_ids,
                        )
                    )

        if request.scheduling_request.hotel_result.night_count and (
            selected_hotel is None
            or any(
                day.start_place_id != selected_hotel or day.end_place_id != selected_hotel
                for day in schedule.days
            )
        ):
            issues.append(
                self._issue(
                    request,
                    code=ValidationIssueCode.LODGING_COVERAGE,
                    severity=ValidationSeverity.HARD_CONFLICT,
                    target=(
                        ValidationTargetKind.HOTEL
                        if selected_hotel is not None
                        else ValidationTargetKind.TRIP
                    ),
                    message="过夜行程没有由同一家最终酒店覆盖全部夜晚。",
                    hotel_place_id=selected_hotel,
                    repair=RepairAction.REPLACE_CANDIDATE,
                )
            )

        for category in cost.categories:
            if category.status.value in {"missing", "partial"}:
                issues.append(
                    self._issue(
                        request,
                        code=ValidationIssueCode.MISSING_PRICE,
                        severity=ValidationSeverity.UNKNOWN_FACT,
                        target=ValidationTargetKind.COST_CATEGORY,
                        message=category.note or "该费用类别存在未取得的价格。",
                        cost_category=category.category,
                        repair=RepairAction.REQUERY_FACT,
                        sources=category.source_reference_ids,
                    )
                )

        if not request.weather:
            for day in schedule.days:
                issues.append(
                    self._issue(
                        request,
                        code=ValidationIssueCode.MISSING_NIGHT_WEATHER,
                        severity=ValidationSeverity.UNKNOWN_FACT,
                        target=ValidationTargetKind.DAY,
                        message="当天及夜间天气资料暂缺。",
                        service_date=day.service_date,
                        repair=RepairAction.REQUERY_FACT,
                    )
                )
        for weather in request.weather:
            if (
                weather.availability is not DataAvailability.AVAILABLE
                or not weather.night_condition_available
            ):
                issues.append(
                    self._issue(
                        request,
                        code=ValidationIssueCode.MISSING_NIGHT_WEATHER,
                        severity=ValidationSeverity.UNKNOWN_FACT,
                        target=ValidationTargetKind.DAY,
                        message=weather.missing_reason or "夜间天气资料暂缺。",
                        service_date=weather.service_date,
                        repair=RepairAction.REQUERY_FACT,
                        sources=weather.source_reference_ids,
                    )
                )

        for desire in schedule.unscheduled_strong_desires:
            hard = desire.role in {
                AnchorRole.MUST_ATTRACTION,
                AnchorRole.DESTINATION_RESTAURANT,
            }
            issues.append(
                self._issue(
                    request,
                    code=ValidationIssueCode.UNSCHEDULED_STRONG_DESIRE,
                    severity=(
                        ValidationSeverity.HARD_CONFLICT if hard else ValidationSeverity.SOFT_RISK
                    ),
                    target=ValidationTargetKind.ACTIVITY,
                    message=f"未能安全安排：{desire.reason}",
                    unscheduled_node_id=desire.node_id,
                    repair=RepairAction.REASSIGN_DAY,
                    sources=desire.source_reference_ids,
                )
            )

        generated_at = self._clock()
        if generated_at.tzinfo is None or generated_at.utcoffset() is None:
            raise ValueError("itinerary validation clock must return an aware datetime")
        status = (
            ValidationStatus.BLOCKED
            if any(item.severity is ValidationSeverity.HARD_CONFLICT for item in issues)
            else ValidationStatus.REVIEW
            if issues
            else ValidationStatus.VALID
        )
        return ItineraryValidationResult(
            algorithm_version=ITINERARY_VALIDATION_ALGORITHM_VERSION,
            request_id=request.request_id,
            trip_id=request.trip_id,
            input_state_version=request.input_state_version,
            task_book_id=schedule.task_book_id,
            task_book_revision=schedule.task_book_revision,
            schedule_request_id=schedule.request_id,
            cost_request_id=cost.request_id,
            status=status,
            issues=tuple(issues),
            generated_at=generated_at.astimezone(UTC),
        )

    def publish_strict(
        self,
        request: ItineraryValidationRequest,
        result: ItineraryValidationResult,
    ) -> tuple[DailyScheduleResult, TripCostEstimate]:
        """Cross the publication boundary only after a completely valid report."""

        if (
            result.request_id != request.request_id
            or result.trip_id != request.trip_id
            or result.input_state_version != request.input_state_version
            or result.status is ValidationStatus.BLOCKED
        ):
            raise ValueError("a mismatched or blocked draft cannot become a strict result")
        schedule = DailyScheduleResult.model_validate(
            request.schedule_draft.model_dump(mode="json")
        )
        cost = TripCostEstimate.model_validate(request.cost_draft.model_dump(mode="json"))
        return schedule, cost

    def _identity_issues(self, request: ItineraryValidationRequest) -> tuple[ValidationIssue, ...]:
        base = request.scheduling_request
        schedule = request.schedule_draft
        cost = request.cost_draft
        issues: list[ValidationIssue] = []
        if any(
            value != request.trip_id for value in (base.trip_id, schedule.trip_id, cost.trip_id)
        ) or any(
            value != request.input_state_version
            for value in (
                base.input_state_version,
                schedule.input_state_version,
                cost.input_state_version,
            )
        ):
            issues.append(
                self._issue(
                    request,
                    code=ValidationIssueCode.REFERENCE_CONFLICT,
                    severity=ValidationSeverity.HARD_CONFLICT,
                    target=ValidationTargetKind.TRIP,
                    message="待校验草案与当前旅行或状态版本不一致。",
                    repair=RepairAction.FIX_REFERENCE,
                    identity_suffix="trip-version",
                )
            )
        if (
            schedule.request_id != base.request_id
            or schedule.task_book_id != base.task_book.task_book_id
            or schedule.task_book_revision != base.task_book.revision
            or schedule.city_id != base.city_id
            or cost.schedule_request_id != schedule.request_id
            or cost.task_book_id != schedule.task_book_id
            or cost.task_book_revision != schedule.task_book_revision
        ):
            issues.append(
                self._issue(
                    request,
                    code=ValidationIssueCode.REFERENCE_CONFLICT,
                    severity=ValidationSeverity.HARD_CONFLICT,
                    target=ValidationTargetKind.TRIP,
                    message="排程、费用或任务书引用不属于同一版旅行。",
                    repair=RepairAction.FIX_REFERENCE,
                    identity_suffix="planning-chain",
                )
            )
        expected_start = base.task_book.date_range.start_date
        expected_end = base.task_book.date_range.end_date
        expected_dates = tuple(
            expected_start.fromordinal(day)
            for day in range(expected_start.toordinal(), expected_end.toordinal() + 1)
        )
        draft_dates = tuple(day.service_date for day in schedule.days)
        cost_dates = tuple(day.service_date for day in cost.days)
        if (
            schedule.start_date != expected_start
            or schedule.end_date != expected_end
            or draft_dates != expected_dates
            or cost.start_date != expected_start
            or cost.end_date != expected_end
            or cost_dates != expected_dates
        ):
            issues.append(
                self._issue(
                    request,
                    code=ValidationIssueCode.DATE_CONFLICT,
                    severity=ValidationSeverity.HARD_CONFLICT,
                    target=ValidationTargetKind.TRIP,
                    message="排程或费用日期没有完整对应已确认旅行日期。",
                    repair=RepairAction.REASSIGN_DAY,
                    identity_suffix="trip-dates",
                )
            )
        weather_dates = tuple(item.service_date for item in request.weather)
        if request.weather and weather_dates != expected_dates:
            issues.append(
                self._issue(
                    request,
                    code=ValidationIssueCode.DATE_CONFLICT,
                    severity=ValidationSeverity.HARD_CONFLICT,
                    target=ValidationTargetKind.TRIP,
                    message="天气资料日期没有完整对应旅行日期。",
                    repair=RepairAction.REQUERY_FACT,
                    identity_suffix="weather-dates",
                )
            )
        return tuple(issues)

    def _day_structure_issues(
        self,
        request: ItineraryValidationRequest,
        day: DraftScheduledDay,
    ) -> tuple[ValidationIssue, ...]:
        issues: list[ValidationIssue] = []
        known_places = {item.place_id for item in request.scheduling_request.places}
        hotel = request.scheduling_request.hotel_result.selected_hotel_place_id
        if hotel is not None:
            known_places.add(hotel)
        referenced_places = {
            day.start_place_id,
            day.end_place_id,
            *(item.place_id for item in day.activities),
            *(item.origin_place_id for item in day.transport_legs),
            *(item.destination_place_id for item in day.transport_legs),
        }
        if not referenced_places <= known_places:
            issues.append(
                self._issue(
                    request,
                    code=ValidationIssueCode.REFERENCE_CONFLICT,
                    severity=ValidationSeverity.HARD_CONFLICT,
                    target=ValidationTargetKind.DAY,
                    message="当天活动或路线引用了当前旅行不存在的地点。",
                    service_date=day.service_date,
                    repair=RepairAction.FIX_REFERENCE,
                    identity_suffix="unknown-place",
                )
            )
        place_ids = [item.place_id for item in day.activities]
        if len(set(place_ids)) != len(place_ids):
            issues.append(
                self._issue(
                    request,
                    code=ValidationIssueCode.REFERENCE_CONFLICT,
                    severity=ValidationSeverity.HARD_CONFLICT,
                    target=ValidationTargetKind.DAY,
                    message="同一地点在当天被重复安排。",
                    service_date=day.service_date,
                    repair=RepairAction.REORDER_DAY,
                    identity_suffix="duplicate-place",
                )
            )
        for activity in day.activities:
            if activity.service_date != day.service_date:
                issues.append(
                    self._issue(
                        request,
                        code=ValidationIssueCode.DATE_CONFLICT,
                        severity=ValidationSeverity.HARD_CONFLICT,
                        target=ValidationTargetKind.ACTIVITY,
                        message="活动日期与所属日不一致。",
                        service_date=day.service_date,
                        activity_id=activity.activity_id,
                        repair=RepairAction.REASSIGN_DAY,
                    )
                )
            actual_duration = _minutes(activity.end_time) - _minutes(activity.start_time)
            if actual_duration <= 0 or actual_duration != activity.duration_minutes:
                issues.append(
                    self._issue(
                        request,
                        code=ValidationIssueCode.DURATION_CONFLICT,
                        severity=ValidationSeverity.HARD_CONFLICT,
                        target=ValidationTargetKind.ACTIVITY,
                        message="活动起止时间与停留时长不一致。",
                        service_date=day.service_date,
                        activity_id=activity.activity_id,
                        repair=RepairAction.ADJUST_TIME,
                    )
                )
        for pause in day.pauses:
            actual_duration = _minutes(pause.end_time) - _minutes(pause.start_time)
            if (
                pause.service_date != day.service_date
                or actual_duration <= 0
                or (actual_duration != pause.duration_minutes)
            ):
                issues.append(
                    self._issue(
                        request,
                        code=(
                            ValidationIssueCode.DATE_CONFLICT
                            if pause.service_date != day.service_date
                            else ValidationIssueCode.DURATION_CONFLICT
                        ),
                        severity=ValidationSeverity.HARD_CONFLICT,
                        target=ValidationTargetKind.DAY,
                        message="用餐或休息的日期、起止时间与时长不一致。",
                        service_date=day.service_date,
                        repair=RepairAction.ADJUST_TIME,
                        related_pause_ids=(pause.pause_id,),
                        identity_suffix=str(pause.pause_id),
                    )
                )
        for leg in day.transport_legs:
            actual_duration = _minutes(leg.arrival_time) - _minutes(leg.departure_time)
            if actual_duration <= 0 or actual_duration != leg.duration_minutes + leg.buffer_minutes:
                issues.append(
                    self._issue(
                        request,
                        code=ValidationIssueCode.INSUFFICIENT_TRAVEL_TIME,
                        severity=ValidationSeverity.HARD_CONFLICT,
                        target=ValidationTargetKind.TRANSPORT_LEG,
                        message="路线时间不足以覆盖交通耗时与缓冲。",
                        service_date=day.service_date,
                        transport_leg_id=leg.leg_id,
                        repair=RepairAction.ADJUST_BUFFER,
                        sources=leg.source_reference_ids,
                    )
                )
        events: list[tuple[int, int, str, UUID, str]] = (
            [
                (
                    _minutes(item.start_time),
                    _minutes(item.end_time),
                    item.title,
                    item.activity_id,
                    "activity",
                )
                for item in day.activities
            ]
            + [
                (
                    _minutes(item.departure_time),
                    _minutes(item.arrival_time),
                    "交通段",
                    item.leg_id,
                    "transport",
                )
                for item in day.transport_legs
            ]
            + [
                (
                    _minutes(item.start_time),
                    _minutes(item.end_time),
                    "用餐或休息",
                    item.pause_id,
                    "pause",
                )
                for item in day.pauses
            ]
        )
        events.sort(key=lambda item: (item[0], item[1], str(item[3])))
        for index, previous in enumerate(events):
            for current in events[index + 1 :]:
                if current[0] >= previous[1]:
                    break
                activity_ids = tuple(
                    item[3] for item in (previous, current) if item[4] == "activity"
                )
                leg_ids = tuple(item[3] for item in (previous, current) if item[4] == "transport")
                pause_ids = tuple(item[3] for item in (previous, current) if item[4] == "pause")
                issues.append(
                    self._issue(
                        request,
                        code=ValidationIssueCode.TIME_OVERLAP,
                        severity=ValidationSeverity.HARD_CONFLICT,
                        target=ValidationTargetKind.DAY,
                        message=f"{previous[2]}与{current[2]}发生时间重叠。",
                        service_date=day.service_date,
                        repair=RepairAction.REORDER_DAY,
                        related_activity_ids=activity_ids,
                        related_transport_leg_ids=leg_ids,
                        related_pause_ids=pause_ids,
                        identity_suffix=f"{previous[3]}:{current[3]}",
                    )
                )
        totals_match = (
            day.active_minutes == sum(item.duration_minutes for item in day.activities)
            and day.walking_m == sum(item.walking_m for item in day.transport_legs)
            and day.cycling_m
            == sum(item.distance_m for item in day.transport_legs if item.mode is RouteMode.CYCLING)
            and day.meal_minutes
            == sum(item.duration_minutes for item in day.pauses if item.kind.value == "meal")
            and day.rest_minutes
            == sum(item.duration_minutes for item in day.pauses if item.kind.value == "rest")
            and day.buffer_minutes == sum(item.buffer_minutes for item in day.transport_legs)
        )
        if not totals_match:
            issues.append(
                self._issue(
                    request,
                    code=ValidationIssueCode.SCHEDULE_TOTAL_INCONSISTENCY,
                    severity=ValidationSeverity.HARD_CONFLICT,
                    target=ValidationTargetKind.DAY,
                    message="当天活动、步行、骑行、休息或缓冲汇总与明细不一致。",
                    service_date=day.service_date,
                    repair=RepairAction.RECALCULATE_SCHEDULE,
                    identity_suffix="schedule-totals",
                )
            )
        issues.extend(self._route_sequence_issues(request, day))
        return tuple(issues)

    def _route_sequence_issues(
        self,
        request: ItineraryValidationRequest,
        day: DraftScheduledDay,
    ) -> tuple[ValidationIssue, ...]:
        activities = sorted(day.activities, key=lambda item: (item.start_time, item.end_time))
        if not activities:
            return ()
        transitions: list[tuple[UUID, UUID, time, time]] = []
        transitions.append(
            (day.start_place_id, activities[0].place_id, day.start_time, activities[0].start_time)
        )
        transitions.extend(
            (
                left.place_id,
                right.place_id,
                left.end_time,
                right.start_time,
            )
            for left, right in zip(activities, activities[1:], strict=False)
        )
        transitions.append(
            (activities[-1].place_id, day.end_place_id, activities[-1].end_time, day.end_time)
        )
        issues: list[ValidationIssue] = []
        for index, (origin, destination, earliest, latest) in enumerate(transitions):
            if origin == destination:
                continue
            matching = tuple(
                leg
                for leg in day.transport_legs
                if leg.origin_place_id == origin and leg.destination_place_id == destination
            )
            if not matching:
                issues.append(
                    self._issue(
                        request,
                        code=ValidationIssueCode.INSUFFICIENT_TRAVEL_TIME,
                        severity=ValidationSeverity.HARD_CONFLICT,
                        target=ValidationTargetKind.DAY,
                        message="相邻地点之间缺少可执行的交通段。",
                        service_date=day.service_date,
                        repair=RepairAction.CHANGE_ROUTE,
                        identity_suffix=f"missing-leg:{index}:{origin}:{destination}",
                    )
                )
                continue
            leg = matching[0]
            if leg.departure_time < earliest or leg.arrival_time > latest:
                issues.append(
                    self._issue(
                        request,
                        code=ValidationIssueCode.INSUFFICIENT_TRAVEL_TIME,
                        severity=ValidationSeverity.HARD_CONFLICT,
                        target=ValidationTargetKind.TRANSPORT_LEG,
                        message="交通段没有完整落在相邻活动之间。",
                        service_date=day.service_date,
                        transport_leg_id=leg.leg_id,
                        repair=RepairAction.ADJUST_TIME,
                        sources=leg.source_reference_ids,
                    )
                )
        return tuple(issues)

    def _cost_issues(self, request: ItineraryValidationRequest) -> tuple[ValidationIssue, ...]:
        issues: list[ValidationIssue] = []
        cost = request.cost_draft
        all_lines = tuple(line for day in cost.days for line in day.lines)
        for day in cost.days:
            if any(line.service_date != day.service_date for line in day.lines):
                issues.append(
                    self._issue(
                        request,
                        code=ValidationIssueCode.DATE_CONFLICT,
                        severity=ValidationSeverity.HARD_CONFLICT,
                        target=ValidationTargetKind.DAY,
                        message="费用明细日期与每日费用归属不一致。",
                        service_date=day.service_date,
                        repair=RepairAction.RECALCULATE_COST,
                        identity_suffix="cost-line-date",
                    )
                )
            expected = _sum_draft_ranges(
                item.amount_per_person for item in day.lines if item.amount_per_person is not None
            )
            if not _same_draft_range(day.known_subtotal_per_person, expected):
                issues.append(
                    self._issue(
                        request,
                        code=ValidationIssueCode.COST_INCONSISTENCY,
                        severity=ValidationSeverity.HARD_CONFLICT,
                        target=ValidationTargetKind.DAY,
                        message="当天费用小计与费用明细之和不一致。",
                        service_date=day.service_date,
                        repair=RepairAction.RECALCULATE_COST,
                        identity_suffix="daily-total",
                    )
                )
            issues.extend(
                self._category_cost_issues(request, day.categories, day.lines, day.service_date)
            )
        expected_total = _sum_draft_ranges(
            day.known_subtotal_per_person
            for day in cost.days
            if day.known_subtotal_per_person is not None
        )
        if not _same_draft_range(cost.known_total_per_person, expected_total):
            issues.append(
                self._issue(
                    request,
                    code=ValidationIssueCode.COST_INCONSISTENCY,
                    severity=ValidationSeverity.HARD_CONFLICT,
                    target=ValidationTargetKind.TRIP,
                    message="全程费用与每日费用小计之和不一致。",
                    repair=RepairAction.RECALCULATE_COST,
                    identity_suffix="trip-total",
                )
            )
        issues.extend(self._category_cost_issues(request, cost.categories, all_lines, None))
        return tuple(issues)

    def _category_cost_issues(
        self,
        request: ItineraryValidationRequest,
        categories: tuple[DraftCategoryCostSummary, ...],
        lines: tuple[DraftCostEstimateLine, ...],
        service_date: date | None,
    ) -> tuple[ValidationIssue, ...]:
        issues: list[ValidationIssue] = []
        by_category = {item.category: item for item in categories}
        for category in CostCategory:
            applicable = tuple(item for item in lines if item.category is category)
            summary = by_category.get(category)
            expected = _sum_draft_ranges(
                item.amount_per_person for item in applicable if item.amount_per_person is not None
            )
            invalid_source = any(
                item.availability is not DataAvailability.MISSING and not item.source_reference_ids
                for item in applicable
            )
            priced = tuple(item for item in applicable if item.amount_per_person is not None)
            missing = tuple(
                item
                for item in applicable
                if item.amount_per_person is None or item.availability is DataAvailability.PARTIAL
            )
            expected_status = (
                "not_applicable"
                if not applicable
                else "missing"
                if not priced
                else "partial"
                if missing
                else "available"
            )
            expected_sources = {
                source for item in applicable for source in item.source_reference_ids
            }
            if (
                len(categories) != len(CostCategory)
                or len(by_category) != len(CostCategory)
                or summary is None
                or not _valid_draft_range(expected)
                or not _same_draft_range(
                    summary.amount_per_person if summary is not None else None,
                    expected,
                )
                or invalid_source
                or summary.status.value != expected_status
                or summary.priced_item_count != len(priced)
                or summary.missing_item_count != len(missing)
                or set(summary.source_reference_ids) != expected_sources
            ):
                issues.append(
                    self._issue(
                        request,
                        code=ValidationIssueCode.COST_INCONSISTENCY,
                        severity=ValidationSeverity.HARD_CONFLICT,
                        target=ValidationTargetKind.COST_CATEGORY,
                        message="费用分类、明细合计或来源记录不一致。",
                        service_date=service_date,
                        cost_category=category,
                        repair=RepairAction.RECALCULATE_COST,
                        identity_suffix=(
                            f"{service_date.isoformat() if service_date else 'trip'}:{category}"
                        ),
                    )
                )
        return tuple(issues)

    def _issue(
        self,
        request: ItineraryValidationRequest,
        *,
        code: ValidationIssueCode,
        severity: ValidationSeverity,
        target: ValidationTargetKind,
        message: str,
        repair: RepairAction,
        service_date: date | None = None,
        activity_id: UUID | None = None,
        unscheduled_node_id: UUID | None = None,
        transport_leg_id: UUID | None = None,
        hotel_place_id: UUID | None = None,
        cost_category: CostCategory | None = None,
        sources: tuple[str, ...] = (),
        related_activity_ids: tuple[UUID, ...] = (),
        related_transport_leg_ids: tuple[UUID, ...] = (),
        related_pause_ids: tuple[UUID, ...] = (),
        identity_suffix: str | None = None,
    ) -> ValidationIssue:
        location = (
            activity_id
            or unscheduled_node_id
            or transport_leg_id
            or hotel_place_id
            or cost_category
            or service_date
            or "trip"
        )
        if identity_suffix is not None:
            location = f"{location}:{identity_suffix}"
        issue_id = uuid5(
            NAMESPACE_URL,
            f"iter:v3-39:{request.trip_id}:{request.input_state_version}:{code}:{location}",
        )
        return ValidationIssue(
            issue_id=issue_id,
            code=code,
            severity=severity,
            target_kind=target,
            message=message,
            service_date=service_date,
            activity_id=activity_id,
            unscheduled_node_id=unscheduled_node_id,
            transport_leg_id=transport_leg_id,
            hotel_place_id=hotel_place_id,
            cost_category=cost_category,
            related_activity_ids=related_activity_ids,
            related_transport_leg_ids=related_transport_leg_ids,
            related_pause_ids=related_pause_ids,
            repairable=repair is not RepairAction.NONE,
            repair_action=repair,
            source_reference_ids=tuple(dict.fromkeys(sources)),
        )


def _minutes(value: time) -> int:
    return value.hour * 60 + value.minute


def _valid_draft_range(value: DraftAmountRange | None) -> bool:
    return value is None or (
        value.currency == "CNY"
        and value.minimum_fen >= 0
        and value.maximum_fen >= value.minimum_fen
    )


def _same_draft_range(
    value: DraftAmountRange | None,
    expected: DraftAmountRange | None,
) -> bool:
    if not _valid_draft_range(value) or not _valid_draft_range(expected):
        return False
    if value is None or expected is None:
        return value is expected
    return (
        value.currency == expected.currency
        and value.minimum_fen == expected.minimum_fen
        and value.maximum_fen == expected.maximum_fen
    )


def _sum_draft_ranges(values: Iterable[DraftAmountRange]) -> DraftAmountRange | None:
    materialized = tuple(values)
    if not materialized:
        return None
    if any(not _valid_draft_range(item) for item in materialized):
        return DraftAmountRange(currency="INVALID", minimum_fen=-1, maximum_fen=-1)
    return DraftAmountRange(
        currency="CNY",
        minimum_fen=sum(item.minimum_fen for item in materialized),
        maximum_fen=sum(item.maximum_fen for item in materialized),
    )
