"""V3-39 contracts for deterministic itinerary validation and repair guidance."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import AwareDatetime, ConfigDict, Field, model_validator

from backend.contracts.base import ContractModel
from backend.contracts.common import NonEmptyText, ShortText
from backend.contracts.daily_scheduling import DailySchedulingRequest
from backend.contracts.enums import CostCategory, DataAvailability
from backend.contracts.itinerary_draft import CostValidationDraft, ScheduleValidationDraft

ITINERARY_VALIDATION_REQUEST_SCHEMA_RULE: dict[str, Any] = {
    "x-travel-itinerary-validation-request": True
}
ITINERARY_VALIDATION_RESULT_SCHEMA_RULE: dict[str, Any] = {
    "x-travel-itinerary-validation-result": True
}


class ImmutableValidationModel(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ValidationSeverity(StrEnum):
    HARD_CONFLICT = "hard_conflict"
    SOFT_RISK = "soft_risk"
    UNKNOWN_FACT = "unknown_fact"


class ValidationStatus(StrEnum):
    VALID = "valid"
    REVIEW = "review"
    BLOCKED = "blocked"


class ValidationTargetKind(StrEnum):
    TRIP = "trip"
    DAY = "day"
    ACTIVITY = "activity"
    TRANSPORT_LEG = "transport_leg"
    HOTEL = "hotel"
    COST_CATEGORY = "cost_category"


class ValidationIssueCode(StrEnum):
    DATE_CONFLICT = "date_conflict"
    REFERENCE_CONFLICT = "reference_conflict"
    OPENING_HOURS_CONFLICT = "opening_hours_conflict"
    OPENING_HOURS_UNKNOWN = "opening_hours_unknown"
    FIXED_EVENT_CONFLICT = "fixed_event_conflict"
    TIME_OVERLAP = "time_overlap"
    INSUFFICIENT_TRAVEL_TIME = "insufficient_travel_time"
    WALKING_LIMIT = "walking_limit"
    CYCLING_LIMIT = "cycling_limit"
    DAILY_LOAD = "daily_load"
    LODGING_COVERAGE = "lodging_coverage"
    COST_INCONSISTENCY = "cost_inconsistency"
    DURATION_CONFLICT = "duration_conflict"
    SCHEDULE_TOTAL_INCONSISTENCY = "schedule_total_inconsistency"
    MISSING_PRICE = "missing_price"
    PARTIAL_ROUTE = "partial_route"
    MISSING_NIGHT_WEATHER = "missing_night_weather"
    UNSCHEDULED_STRONG_DESIRE = "unscheduled_strong_desire"
    MEAL_TIMING = "meal_timing"


class RepairAction(StrEnum):
    REASSIGN_DAY = "reassign_day"
    REORDER_DAY = "reorder_day"
    CHANGE_ROUTE = "change_route"
    REPLACE_CANDIDATE = "replace_candidate"
    ADJUST_BUFFER = "adjust_buffer"
    ADJUST_TIME = "adjust_time"
    FIX_REFERENCE = "fix_reference"
    RECALCULATE_COST = "recalculate_cost"
    RECALCULATE_SCHEDULE = "recalculate_schedule"
    REQUERY_FACT = "requery_fact"
    NONE = "none"


class DailyWeatherCoverage(ImmutableValidationModel):
    service_date: date
    availability: DataAvailability
    night_condition_available: bool
    condition_day: ShortText | None = None
    condition_night: ShortText | None = None
    low_celsius: float | None = Field(default=None, ge=-80, le=70, allow_inf_nan=False)
    high_celsius: float | None = Field(default=None, ge=-80, le=70, allow_inf_nan=False)
    source_reference_ids: tuple[NonEmptyText, ...] = ()
    missing_reason: ShortText | None = None
    fetched_at: AwareDatetime | None = None

    @model_validator(mode="after")
    def weather_coverage_is_explicit(self) -> DailyWeatherCoverage:
        _unique(self.source_reference_ids, "weather sources")
        if (
            self.low_celsius is not None
            and self.high_celsius is not None
            and self.high_celsius < self.low_celsius
        ):
            raise ValueError("weather high temperature cannot be below low temperature")
        # Existing callers may have source-backed night coverage without display text.
        if self.night_condition_available != (self.condition_night is not None) and (
            self.condition_night is not None or not self.night_condition_available
        ):
            raise ValueError("night weather flag must match the normalized night condition")
        if self.availability is DataAvailability.AVAILABLE:
            if (
                not self.night_condition_available
                or not self.source_reference_ids
                or self.missing_reason is not None
                or self.fetched_at is None
            ):
                raise ValueError("available weather requires sourced daytime and night facts")
        elif self.availability is DataAvailability.PARTIAL:
            if (
                not self.source_reference_ids
                or self.missing_reason is None
                or self.fetched_at is None
            ):
                raise ValueError("partial weather requires sources, time and a reason")
        elif (
            self.night_condition_available
            or self.condition_day is not None
            or self.condition_night is not None
            or self.low_celsius is not None
            or self.high_celsius is not None
            or self.source_reference_ids
            or self.missing_reason is None
            or self.fetched_at is not None
        ):
            raise ValueError("missing weather requires only an explicit reason")
        return self


class ItineraryValidationRequest(ImmutableValidationModel):
    model_config = ConfigDict(json_schema_extra=ITINERARY_VALIDATION_REQUEST_SCHEMA_RULE)

    request_id: UUID
    trip_id: UUID
    input_state_version: int = Field(ge=0, strict=True)
    scheduling_request: DailySchedulingRequest
    schedule_draft: ScheduleValidationDraft
    cost_draft: CostValidationDraft
    weather: tuple[DailyWeatherCoverage, ...] = ()


class ValidationIssue(ImmutableValidationModel):
    issue_id: UUID
    code: ValidationIssueCode
    severity: ValidationSeverity
    target_kind: ValidationTargetKind
    message: ShortText
    service_date: date | None = None
    activity_id: UUID | None = None
    unscheduled_node_id: UUID | None = None
    transport_leg_id: UUID | None = None
    hotel_place_id: UUID | None = None
    cost_category: CostCategory | None = None
    related_activity_ids: tuple[UUID, ...] = ()
    related_transport_leg_ids: tuple[UUID, ...] = ()
    related_pause_ids: tuple[UUID, ...] = ()
    repairable: bool
    repair_action: RepairAction
    source_reference_ids: tuple[NonEmptyText, ...] = ()

    @model_validator(mode="after")
    def location_and_repair_are_consistent(self) -> ValidationIssue:
        _unique(self.source_reference_ids, "validation issue sources")
        _unique(self.related_activity_ids, "related activity IDs")
        _unique(self.related_transport_leg_ids, "related transport IDs")
        _unique(self.related_pause_ids, "related pause IDs")
        location_count = sum(
            value is not None
            for value in (
                self.activity_id,
                self.unscheduled_node_id,
                self.transport_leg_id,
                self.hotel_place_id,
                self.cost_category,
            )
        )
        if location_count > 1:
            raise ValueError("an issue must identify one concrete target")
        if (
            self.target_kind is ValidationTargetKind.ACTIVITY
            and self.activity_id is None
            and self.unscheduled_node_id is None
        ):
            raise ValueError("activity issue requires activity_id or unscheduled_node_id")
        if self.target_kind is ValidationTargetKind.TRANSPORT_LEG and self.transport_leg_id is None:
            raise ValueError("transport issue requires transport_leg_id")
        if self.target_kind is ValidationTargetKind.DAY and self.service_date is None:
            raise ValueError("day issue requires service_date")
        if self.target_kind is ValidationTargetKind.HOTEL and self.hotel_place_id is None:
            raise ValueError("hotel issue requires hotel_place_id")
        if self.target_kind is ValidationTargetKind.COST_CATEGORY and self.cost_category is None:
            raise ValueError("cost issue requires cost_category")
        if self.repairable != (self.repair_action is not RepairAction.NONE):
            raise ValueError("repairable issues require a concrete repair action")
        return self


class ItineraryValidationResult(ImmutableValidationModel):
    model_config = ConfigDict(json_schema_extra=ITINERARY_VALIDATION_RESULT_SCHEMA_RULE)

    algorithm_version: str = Field(pattern=r"^[1-9]\d*\.\d+\.\d+$")
    request_id: UUID
    trip_id: UUID
    input_state_version: int = Field(ge=0, strict=True)
    task_book_id: UUID
    task_book_revision: int = Field(ge=1, strict=True)
    schedule_request_id: UUID
    cost_request_id: UUID
    status: ValidationStatus
    issues: tuple[ValidationIssue, ...]
    generated_at: AwareDatetime

    @model_validator(mode="after")
    def status_and_issues_are_consistent(self) -> ItineraryValidationResult:
        _unique([item.issue_id for item in self.issues], "validation issue IDs")
        expected = (
            ValidationStatus.BLOCKED
            if any(item.severity is ValidationSeverity.HARD_CONFLICT for item in self.issues)
            else ValidationStatus.REVIEW
            if self.issues
            else ValidationStatus.VALID
        )
        if self.status is not expected:
            raise ValueError("validation status must be derived from issue severities")
        return self


def _unique(values: Sequence[object], label: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must be unique")


V3_ITINERARY_VALIDATION_CONTRACTS: tuple[type[ContractModel], ...] = (
    ItineraryValidationRequest,
    ItineraryValidationResult,
)
