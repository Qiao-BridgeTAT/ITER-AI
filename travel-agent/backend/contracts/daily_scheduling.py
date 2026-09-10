"""V3-37 contracts for deterministic, fact-backed daily scheduling."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, time
from enum import StrEnum
from uuid import UUID

from pydantic import AwareDatetime, ConfigDict, Field, model_validator

from backend.agent.task_book_state import SemanticTaskBook, SemanticTaskBookStatus
from backend.contracts.base import ContractModel
from backend.contracts.common import NonEmptyText, ShortText
from backend.contracts.enums import AnchorRole, DataAvailability, PlaceCategory, ProviderCode
from backend.contracts.hotel_selection import (
    HotelDecisionStatus,
    HotelSelectionResult,
)
from backend.contracts.places import Gcj02Coordinates
from backend.contracts.spatial_planning import SpatialPlanningResult
from backend.providers.contracts import HoursDayStatus, RouteMode


class ImmutableScheduleModel(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ScheduleActivityKind(StrEnum):
    ATTRACTION = "attraction"
    RESTAURANT = "restaurant"
    FIXED_EVENT = "fixed_event"


class SchedulePauseKind(StrEnum):
    MEAL = "meal"
    REST = "rest"


class OpeningWindow(ImmutableScheduleModel):
    service_date: date
    start_time: time
    end_time: time
    last_entry_at: time | None = None
    source_reference_ids: tuple[NonEmptyText, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def window_is_ordered(self) -> OpeningWindow:
        if self.end_time <= self.start_time:
            raise ValueError("opening window end must be after start")
        if (
            self.last_entry_at is not None
            and not self.start_time <= self.last_entry_at <= self.end_time
        ):
            raise ValueError("last entry must be within the opening window")
        _unique(self.source_reference_ids, "opening-window sources")
        return self


class OpeningDateStatus(ImmutableScheduleModel):
    service_date: date
    status: HoursDayStatus
    reason: ShortText


class SchedulePlaceFact(ImmutableScheduleModel):
    place_id: UUID
    city_id: NonEmptyText
    name: NonEmptyText
    category: PlaceCategory
    coordinates: Gcj02Coordinates
    recommended_duration_minutes: int = Field(ge=15, le=480, strict=True)
    opening_availability: DataAvailability
    opening_windows: tuple[OpeningWindow, ...] = Field(default=(), max_length=310)
    opening_dates: tuple[OpeningDateStatus, ...] = Field(default=(), max_length=31)
    opening_missing_reason: ShortText | None = None
    source_reference_ids: tuple[NonEmptyText, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def hours_and_sources_are_consistent(self) -> SchedulePlaceFact:
        _unique(self.source_reference_ids, "schedule-place sources")
        if self.opening_dates:
            _unique(tuple(item.service_date for item in self.opening_dates), "opening dates")
            window_dates = {item.service_date for item in self.opening_windows}
            open_dates = {
                item.service_date
                for item in self.opening_dates
                if item.status is HoursDayStatus.OPEN
            }
            if window_dates != open_dates:
                raise ValueError("only verified open dates can have opening windows")
            known = any(
                item.status in (HoursDayStatus.OPEN, HoursDayStatus.CLOSED)
                for item in self.opening_dates
            )
            uncertain = any(
                item.status in (HoursDayStatus.UNKNOWN, HoursDayStatus.CONFLICT)
                for item in self.opening_dates
            )
            expected = (
                DataAvailability.PARTIAL
                if known and uncertain
                else DataAvailability.AVAILABLE
                if known
                else DataAvailability.MISSING
            )
            if self.opening_availability is not expected:
                raise ValueError("hours availability must reflect per-date evidence")
            if (self.opening_missing_reason is not None) != uncertain:
                raise ValueError("uncertain date hours require an explicit reason")
            return self
        if self.opening_availability is DataAvailability.AVAILABLE:
            if not self.opening_windows or self.opening_missing_reason is not None:
                raise ValueError("available place hours require windows and no missing reason")
        elif self.opening_availability is DataAvailability.PARTIAL:
            if not self.opening_windows or self.opening_missing_reason is None:
                raise ValueError("partial place hours require windows and a missing reason")
        elif self.opening_windows or self.opening_missing_reason is None:
            raise ValueError("missing place hours require no windows and an explicit reason")
        return self


class ScheduleRouteFact(ImmutableScheduleModel):
    route_fact_id: NonEmptyText
    origin_place_id: UUID
    destination_place_id: UUID
    mode: RouteMode
    availability: DataAvailability
    distance_m: int | None = Field(default=None, ge=0, strict=True)
    duration_minutes: int | None = Field(default=None, ge=0, le=360, strict=True)
    walking_m: int | None = Field(default=None, ge=0, strict=True)
    polyline: tuple[Gcj02Coordinates, ...] = Field(default=(), max_length=20_000)
    provider: ProviderCode | None = None
    source_reference_ids: tuple[NonEmptyText, ...] = Field(default=())
    missing_reason: ShortText | None = None

    @model_validator(mode="after")
    def route_availability_is_explicit(self) -> ScheduleRouteFact:
        if self.origin_place_id == self.destination_place_id:
            raise ValueError("schedule route requires two different places")
        _unique(self.source_reference_ids, "schedule-route sources")
        if self.availability is DataAvailability.MISSING:
            if (
                any(
                    value is not None
                    for value in (
                        self.distance_m,
                        self.duration_minutes,
                        self.walking_m,
                        self.provider,
                    )
                )
                or self.polyline
                or self.source_reference_ids
                or self.missing_reason is None
            ):
                raise ValueError("missing schedule route requires only an explicit reason")
        elif (
            self.distance_m is None
            or self.duration_minutes is None
            or self.walking_m is None
            or self.provider is None
            or not self.source_reference_ids
        ):
            raise ValueError("usable schedule route requires metrics, provider and sources")
        elif self.availability is DataAvailability.AVAILABLE and self.missing_reason is not None:
            raise ValueError("available schedule route cannot contain a missing reason")
        elif self.availability is DataAvailability.PARTIAL and self.missing_reason is None:
            raise ValueError("partial schedule route requires a missing reason")
        elif self.polyline and len(self.polyline) < 2:
            raise ValueError("schedule route polyline requires at least two points")
        return self


class DailyScheduleWindow(ImmutableScheduleModel):
    service_date: date
    start_time: time
    end_time: time
    start_place_id: UUID
    end_place_id: UUID

    @model_validator(mode="after")
    def window_is_ordered(self) -> DailyScheduleWindow:
        if self.end_time <= self.start_time:
            raise ValueError("daily schedule end must be after start")
        return self


class SchedulePreferences(ImmutableScheduleModel):
    pace_level: int = Field(ge=1, le=5, strict=True)
    allowed_modes: tuple[RouteMode, ...] = Field(min_length=1, max_length=4)
    maximum_walking_m_per_leg: int = Field(ge=0, le=20_000, strict=True)
    maximum_walking_m_per_day: int = Field(default=12_000, ge=0, le=50_000, strict=True)
    maximum_cycling_m_per_day: int = Field(default=20_000, ge=0, le=100_000, strict=True)
    route_buffer_minutes: int = Field(default=10, ge=0, le=45, strict=True)
    lunch_acceptable_start: time = time(11, 0)
    lunch_ideal_start: time = time(11, 30)
    lunch_ideal_end: time = time(13, 0)
    lunch_acceptable_end: time = time(14, 0)
    dinner_acceptable_start: time = time(17, 0)
    dinner_ideal_start: time = time(17, 30)
    dinner_ideal_end: time = time(19, 0)
    dinner_acceptable_end: time = time(20, 0)
    meal_duration_minutes: int = Field(default=45, ge=30, le=45, strict=True)

    @model_validator(mode="after")
    def modes_and_meals_are_consistent(self) -> SchedulePreferences:
        _unique(self.allowed_modes, "allowed schedule modes")
        if not (
            self.lunch_acceptable_start
            <= self.lunch_ideal_start
            <= self.lunch_ideal_end
            <= self.lunch_acceptable_end
        ):
            raise ValueError("lunch acceptable and ideal start windows must be ordered")
        if not (
            self.dinner_acceptable_start
            <= self.dinner_ideal_start
            <= self.dinner_ideal_end
            <= self.dinner_acceptable_end
        ):
            raise ValueError("dinner acceptable and ideal start windows must be ordered")
        return self

    @property
    def maximum_active_minutes(self) -> int:
        return {1: 600, 2: 540, 3: 480, 4: 390, 5: 300}[self.pace_level]


class DailySchedulingRequest(ImmutableScheduleModel):
    request_id: UUID
    trip_id: UUID
    input_state_version: int = Field(ge=0, strict=True)
    business_date: date
    city_id: NonEmptyText
    task_book: SemanticTaskBook
    spatial_result: SpatialPlanningResult
    hotel_result: HotelSelectionResult
    places: tuple[SchedulePlaceFact, ...] = Field(min_length=1, max_length=20)
    routes: tuple[ScheduleRouteFact, ...] = Field(default=(), max_length=500)
    daily_windows: tuple[DailyScheduleWindow, ...] = Field(min_length=1, max_length=5)
    preferences: SchedulePreferences

    @model_validator(mode="after")
    def planning_inputs_are_current_and_complete(self) -> DailySchedulingRequest:
        if self.task_book.status is not SemanticTaskBookStatus.CONFIRMED:
            raise ValueError("daily scheduling requires a confirmed task book")
        if self.task_book.trip_id != self.trip_id:
            raise ValueError("task book must belong to the scheduling trip")
        if self.task_book.destination.city_id != self.city_id:
            raise ValueError("task book destination must match the scheduling city")
        start_date = self.task_book.date_range.start_date
        end_date = self.task_book.date_range.end_date
        if start_date < self.business_date:
            raise ValueError("daily scheduling cannot plan a trip before its business date")
        expected_dates = tuple(
            start_date.fromordinal(day)
            for day in range(start_date.toordinal(), end_date.toordinal() + 1)
        )
        if tuple(window.service_date for window in self.daily_windows) != expected_dates:
            raise ValueError("daily windows must cover each trip date exactly once and in order")
        if (
            self.spatial_result.trip_id != self.trip_id
            or self.spatial_result.city_id != self.city_id
            or self.spatial_result.input_state_version != self.input_state_version
        ):
            raise ValueError("daily scheduling requires the current spatial result")
        if (
            self.spatial_result.task_book_id != self.task_book.task_book_id
            or self.spatial_result.task_book_revision != self.task_book.revision
        ):
            raise ValueError("spatial result must reference the confirmed task book")
        if (
            self.hotel_result.trip_id != self.trip_id
            or self.hotel_result.city_id != self.city_id
            or self.hotel_result.input_state_version != self.input_state_version
            or self.hotel_result.check_in != start_date
            or self.hotel_result.check_out != end_date
        ):
            raise ValueError("daily scheduling requires the current trip hotel result")
        _unique([place.place_id for place in self.places], "schedule place IDs")
        if any(place.city_id != self.city_id for place in self.places):
            raise ValueError("every schedule place must belong to the trip city")
        places_by_id = {place.place_id: place for place in self.places}
        anchor_place_ids = {anchor.place_id for anchor in self.spatial_result.anchors}
        if not anchor_place_ids <= set(places_by_id):
            raise ValueError("every spatial anchor requires a schedule place fact")
        for anchor in self.spatial_result.anchors:
            place = places_by_id[anchor.place_id]
            expected_category = _role_category(anchor.role)
            if expected_category is not None and place.category is not expected_category:
                raise ValueError("schedule place category must match its spatial role")
        known_places = set(places_by_id)
        selected_hotel_id = self.hotel_result.selected_hotel_place_id
        if self.hotel_result.night_count:
            if (
                self.hotel_result.decision_status is not HotelDecisionStatus.FINAL
                or selected_hotel_id is None
            ):
                raise ValueError("overnight scheduling requires one finalized hotel")
            known_places.add(selected_hotel_id)
            if any(
                window.start_place_id != selected_hotel_id
                or window.end_place_id != selected_hotel_id
                for window in self.daily_windows
            ):
                raise ValueError("every overnight day must start and end at the unique hotel")
        elif self.hotel_result.decision_status is not HotelDecisionStatus.NOT_REQUIRED:
            raise ValueError("day-trip scheduling must not contain a hotel decision")
        for window in self.daily_windows:
            if window.start_place_id not in known_places or window.end_place_id not in known_places:
                raise ValueError("daily boundaries must reference known trip places")
        _unique([route.route_fact_id for route in self.routes], "schedule route fact IDs")
        _unique(
            [
                (route.origin_place_id, route.destination_place_id, route.mode)
                for route in self.routes
            ],
            "schedule route endpoint and mode keys",
        )
        if any(
            route.origin_place_id not in known_places
            or route.destination_place_id not in known_places
            for route in self.routes
        ):
            raise ValueError("schedule routes must reference known trip places")
        for place in self.places:
            if any(window.service_date not in expected_dates for window in place.opening_windows):
                raise ValueError("opening windows must stay inside the trip dates")
            if place.opening_dates and {item.service_date for item in place.opening_dates} != set(
                expected_dates
            ):
                raise ValueError("opening date statuses must cover the trip dates")
        return self


class ScheduledActivity(ImmutableScheduleModel):
    activity_id: UUID
    node_id: UUID
    place_id: UUID
    kind: ScheduleActivityKind
    role: AnchorRole
    title: NonEmptyText
    service_date: date
    start_time: time
    end_time: time
    duration_minutes: int = Field(ge=1, le=480, strict=True)
    availability: DataAvailability
    source_reference_ids: tuple[NonEmptyText, ...] = Field(min_length=1)
    missing_reason: ShortText | None = None
    timing_notice: ShortText | None = None

    @model_validator(mode="after")
    def duration_and_availability_are_consistent(self) -> ScheduledActivity:
        if _minutes(self.start_time, self.end_time) != self.duration_minutes:
            raise ValueError("scheduled activity duration must match its time range")
        _unique(self.source_reference_ids, "scheduled activity sources")
        if self.availability is DataAvailability.MISSING:
            raise ValueError("an activity with missing facts cannot be scheduled")
        if self.availability is DataAvailability.AVAILABLE:
            if self.missing_reason is not None:
                raise ValueError("available activity cannot contain a missing reason")
        elif self.missing_reason is None:
            raise ValueError("partial scheduled activity requires an explicit missing reason")
        return self


class ScheduledPause(ImmutableScheduleModel):
    pause_id: UUID
    kind: SchedulePauseKind
    service_date: date
    start_time: time
    end_time: time
    duration_minutes: int = Field(ge=1, le=180, strict=True)
    reason: ShortText

    @model_validator(mode="after")
    def duration_matches_times(self) -> ScheduledPause:
        if _minutes(self.start_time, self.end_time) != self.duration_minutes:
            raise ValueError("schedule pause duration must match its time range")
        return self


class ScheduledTransport(ImmutableScheduleModel):
    leg_id: UUID
    origin_place_id: UUID
    destination_place_id: UUID
    departure_time: time
    arrival_time: time
    mode: RouteMode
    availability: DataAvailability
    distance_m: int
    duration_minutes: int = Field(ge=0, le=360, strict=True)
    walking_m: int = Field(ge=0, strict=True)
    buffer_minutes: int = Field(ge=0, le=45, strict=True)
    source_reference_ids: tuple[NonEmptyText, ...] = Field(min_length=1)
    missing_reason: ShortText | None = None

    @model_validator(mode="after")
    def travel_time_and_sources_are_consistent(self) -> ScheduledTransport:
        if self.origin_place_id == self.destination_place_id:
            raise ValueError("scheduled transport requires two different places")
        if _minutes(self.departure_time, self.arrival_time) != (
            self.duration_minutes + self.buffer_minutes
        ):
            raise ValueError("transport time must equal route duration plus buffer")
        _unique(self.source_reference_ids, "scheduled transport sources")
        if self.availability is DataAvailability.MISSING:
            raise ValueError("a missing route cannot become a scheduled transport leg")
        if self.availability is DataAvailability.AVAILABLE:
            if self.missing_reason is not None:
                raise ValueError("available transport cannot contain a missing reason")
        elif self.missing_reason is None:
            raise ValueError("partial transport requires an explicit missing reason")
        return self


class ScheduledDay(ImmutableScheduleModel):
    service_date: date
    start_place_id: UUID
    end_place_id: UUID
    start_time: time
    end_time: time
    activities: tuple[ScheduledActivity, ...]
    pauses: tuple[ScheduledPause, ...]
    transport_legs: tuple[ScheduledTransport, ...]
    active_minutes: int = Field(ge=0, strict=True)
    walking_m: int = Field(ge=0, strict=True)
    cycling_m: int = Field(default=0, ge=0, strict=True)
    meal_minutes: int = Field(ge=0, strict=True)
    rest_minutes: int = Field(ge=0, strict=True)
    buffer_minutes: int = Field(ge=0, strict=True)

    @model_validator(mode="after")
    def timeline_is_ordered_and_totals_match(self) -> ScheduledDay:
        events = sorted(
            [
                *((item.start_time, item.end_time) for item in self.activities),
                *((item.start_time, item.end_time) for item in self.pauses),
                *((item.departure_time, item.arrival_time) for item in self.transport_legs),
            ],
            key=lambda value: value[0],
        )
        if any(
            current[0] < previous[1] for previous, current in zip(events, events[1:], strict=False)
        ):
            raise ValueError("daily schedule events cannot overlap")
        if events and (events[0][0] < self.start_time or events[-1][1] > self.end_time):
            raise ValueError("daily schedule events must stay inside the daily window")
        if self.active_minutes != sum(item.duration_minutes for item in self.activities):
            raise ValueError("daily active minutes must equal scheduled activities")
        if self.walking_m != sum(item.walking_m for item in self.transport_legs):
            raise ValueError("daily walking must equal transport legs")
        if self.cycling_m != sum(
            item.distance_m for item in self.transport_legs if item.mode is RouteMode.CYCLING
        ):
            raise ValueError("daily cycling must equal cycling transport legs")
        if self.meal_minutes != sum(
            item.duration_minutes for item in self.pauses if item.kind is SchedulePauseKind.MEAL
        ):
            raise ValueError("daily meal minutes must equal meal pauses")
        if self.rest_minutes != sum(
            item.duration_minutes for item in self.pauses if item.kind is SchedulePauseKind.REST
        ):
            raise ValueError("daily rest minutes must equal rest pauses")
        if self.buffer_minutes != sum(item.buffer_minutes for item in self.transport_legs):
            raise ValueError("daily buffer minutes must equal route buffers")
        return self


class UnscheduledStrongDesire(ImmutableScheduleModel):
    node_id: UUID
    place_id: UUID
    role: AnchorRole
    reason: ShortText
    source_reference_ids: tuple[NonEmptyText, ...] = Field(min_length=1)


class DailyScheduleResult(ImmutableScheduleModel):
    algorithm_version: str = Field(pattern=r"^[1-9]\d*\.\d+\.\d+$")
    request_id: UUID
    trip_id: UUID
    input_state_version: int = Field(ge=0, strict=True)
    task_book_id: UUID
    task_book_revision: int = Field(ge=1, strict=True)
    city_id: NonEmptyText
    start_date: date
    end_date: date
    status: DataAvailability
    days: tuple[ScheduledDay, ...] = Field(min_length=1, max_length=5)
    unscheduled_strong_desires: tuple[UnscheduledStrongDesire, ...] = ()
    degradation_reasons: tuple[ShortText, ...] = ()
    provider_fact_ids: tuple[NonEmptyText, ...] = ()
    generated_at: AwareDatetime

    @model_validator(mode="after")
    def dates_status_and_sources_are_consistent(self) -> DailyScheduleResult:
        if self.status is DataAvailability.MISSING:
            raise ValueError("a materialized daily schedule cannot have missing status")
        expected_dates = tuple(
            self.start_date.fromordinal(day)
            for day in range(self.start_date.toordinal(), self.end_date.toordinal() + 1)
        )
        if tuple(day.service_date for day in self.days) != expected_dates:
            raise ValueError("scheduled days must cover the trip exactly once and in order")
        _unique(
            [activity.activity_id for day in self.days for activity in day.activities],
            "scheduled activity IDs",
        )
        _unique(
            [desire.node_id for desire in self.unscheduled_strong_desires],
            "unscheduled strong node IDs",
        )
        for desire in self.unscheduled_strong_desires:
            _unique(desire.source_reference_ids, "unscheduled desire sources")
        _unique(self.degradation_reasons, "schedule degradation reasons")
        _unique(self.provider_fact_ids, "schedule provider fact IDs")
        if self.status is DataAvailability.AVAILABLE:
            if self.unscheduled_strong_desires or self.degradation_reasons:
                raise ValueError("available daily schedule cannot contain degradation")
        elif not self.degradation_reasons:
            raise ValueError("partial daily schedule requires degradation reasons")
        return self


def _role_category(role: AnchorRole) -> PlaceCategory | None:
    if role in {
        AnchorRole.MUST_ATTRACTION,
        AnchorRole.WANT_ATTRACTION,
        AnchorRole.CONVENIENT_ATTRACTION,
    }:
        return PlaceCategory.ATTRACTION
    if role in {AnchorRole.DESTINATION_RESTAURANT, AnchorRole.CONVENIENT_RESTAURANT}:
        return PlaceCategory.RESTAURANT
    if role is AnchorRole.FIXED_HOTEL:
        return PlaceCategory.HOTEL
    return None


def _minutes(start: time, end: time) -> int:
    return end.hour * 60 + end.minute - start.hour * 60 - start.minute


def _unique(values: Sequence[object], label: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must be unique")


V3_DAILY_SCHEDULING_CONTRACTS: tuple[type[ContractModel], ...] = (
    DailySchedulingRequest,
    DailyScheduleResult,
)
