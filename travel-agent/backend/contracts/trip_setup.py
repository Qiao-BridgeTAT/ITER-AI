"""City start, profile decision, date, event, and constraint contracts."""

from __future__ import annotations

from calendar import monthrange
from datetime import date, time
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import ConfigDict, Field, StringConstraints, ValidationInfo, model_validator

from backend.contracts.base import ContractModel
from backend.contracts.cold_start import Level
from backend.contracts.enums import (
    CityBriefStatus,
    CityCode,
    ConstraintKind,
    DateInputMode,
    DateSuggestionKind,
    DayReturn,
    DayStart,
    FixedEventKind,
    MobilityTolerance,
    OwnerType,
    PreferenceDecisionMode,
    PriorityGoal,
    TripPhase,
)

NonEmptyText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, pattern=r"\S"),
]

DATE_RANGE_SCHEMA_RULE: dict[str, Any] = {
    "x-travel-date-range": {
        "startField": "start_date",
        "endField": "end_date",
        "minimumDays": 1,
        "maximumDays": 5,
        "minimumStartOffsetDays": 0,
        "maximumStartYears": 1,
    }
}

PROFILE_DECISION_SCHEMA_RULE: dict[str, Any] = {
    "allOf": [
        {
            "if": {
                "properties": {"mode": {"const": "adjust_for_trip"}},
                "required": ["mode"],
            },
            "then": {
                "properties": {
                    "edited_profile_text": {
                        "minLength": 1,
                        "pattern": r"\S",
                        "type": "string",
                    }
                },
                "required": ["edited_profile_text"],
            },
            "else": {"properties": {"edited_profile_text": {"type": "null"}}},
        }
    ]
}

DATE_SELECTION_SCHEMA_RULE: dict[str, Any] = {
    **DATE_RANGE_SCHEMA_RULE,
    "allOf": [
        {
            "if": {
                "properties": {"input_mode": {"const": "suggestion"}},
                "required": ["input_mode"],
            },
            "then": {
                "properties": {
                    "raw_input": {"type": "null"},
                    "suggestion_kind": {
                        "enum": ["nearest_weekend", "nearby_holiday"],
                        "type": "string",
                    },
                },
                "required": ["suggestion_kind"],
            },
            "else": {
                "properties": {
                    "raw_input": {
                        "minLength": 1,
                        "pattern": r"\S",
                        "type": "string",
                    },
                    "suggestion_kind": {"type": "null"},
                },
                "required": ["raw_input"],
            },
        }
    ],
}

FIXED_EVENT_SCHEMA_RULE: dict[str, Any] = {
    "allOf": [
        {
            "if": {
                "properties": {"start_time": {"type": "string"}},
                "required": ["start_time"],
            },
            "then": {
                "properties": {"end_time": {"type": "string"}},
                "required": ["end_time"],
            },
            "else": {"properties": {"end_time": {"type": "null"}}},
        }
    ],
    "x-travel-time-range": {
        "startField": "start_time",
        "endField": "end_time",
    },
}

TRIP_SETUP_SCHEMA_RULE: dict[str, Any] = {
    "x-travel-events-within-date-range": {
        "dateSelectionField": "date_selection",
        "startField": "start_date",
        "endField": "end_date",
        "eventsField": "fixed_events",
        "eventDateField": "event_date",
    }
}


def one_year_after(value: date) -> date:
    target_year = value.year + 1
    day = min(value.day, monthrange(target_year, value.month)[1])
    return value.replace(year=target_year, day=day)


def _validation_today(info: ValidationInfo) -> date:
    context_today: object = info.context.get("today") if info.context else None
    if isinstance(context_today, date):
        return context_today
    return date.today()


def _validate_date_range(start: date, end: date, today: date) -> None:
    if start < today:
        raise ValueError("start_date cannot be in the past")
    if start > one_year_after(today):
        raise ValueError("start_date cannot be more than one year ahead")
    if end < start:
        raise ValueError("end_date cannot be before start_date")
    if (end - start).days + 1 > 5:
        raise ValueError("trip length must be between 1 and 5 days")


class TripShell(ContractModel):
    trip_id: UUID
    owner_type: OwnerType
    owner_id: NonEmptyText
    city: CityCode | None = None
    phase: TripPhase
    state_version: int = Field(ge=0)
    protocol_version: Literal["v2", "v4"] = "v2"

    @model_validator(mode="after")
    def initial_shell_phase_is_valid(self) -> TripShell:
        if (
            self.protocol_version != "v4"
            and self.city is None
            and self.phase
            not in {
                TripPhase.COLD_START,
                TripPhase.CITY_SELECTION,
            }
        ):
            raise ValueError("a cityless trip shell must be in an initial phase")
        return self


class CitySelection(ContractModel):
    city: CityCode


class CityBriefAcknowledgement(ContractModel):
    status: CityBriefStatus


class ResolvedTripPreferences(ContractModel):
    day_start: DayStart
    day_return: DayReturn
    pace_level: Level
    classic_niche_level: Level
    walking_tolerance: MobilityTolerance
    bike_tolerance: MobilityTolerance
    transit_taxi_level: Level
    priority_goals: list[PriorityGoal] = Field(default_factory=list, max_length=2)

    @model_validator(mode="after")
    def goals_are_unique(self) -> ResolvedTripPreferences:
        if len(set(self.priority_goals)) != len(self.priority_goals):
            raise ValueError("priority_goals must not contain duplicates")
        return self


class PersonalProfileDecision(ContractModel):
    model_config = ConfigDict(json_schema_extra=PROFILE_DECISION_SCHEMA_RULE)

    mode: PreferenceDecisionMode
    edited_profile_text: NonEmptyText | None = None

    @model_validator(mode="after")
    def edited_text_matches_mode(self) -> PersonalProfileDecision:
        if self.mode is PreferenceDecisionMode.ADJUST_FOR_TRIP:
            if self.edited_profile_text is None:
                raise ValueError("edited_profile_text is required for adjust_for_trip")
        elif self.edited_profile_text is not None:
            raise ValueError("edited_profile_text is only allowed for adjust_for_trip")
        return self


class DateSuggestion(ContractModel):
    model_config = ConfigDict(json_schema_extra=DATE_RANGE_SCHEMA_RULE)

    suggestion_id: NonEmptyText
    kind: DateSuggestionKind
    label: NonEmptyText
    start_date: date
    end_date: date

    @model_validator(mode="after")
    def validate_dates(self, info: ValidationInfo) -> DateSuggestion:
        _validate_date_range(
            self.start_date,
            self.end_date,
            _validation_today(info),
        )
        return self

    @property
    def day_count(self) -> int:
        return (self.end_date - self.start_date).days + 1


class DateSelection(ContractModel):
    model_config = ConfigDict(json_schema_extra=DATE_SELECTION_SCHEMA_RULE)

    input_mode: DateInputMode
    suggestion_kind: DateSuggestionKind | None = None
    raw_input: NonEmptyText | None = None
    start_date: date
    end_date: date

    @model_validator(mode="after")
    def validate_selection(self, info: ValidationInfo) -> DateSelection:
        if self.input_mode is DateInputMode.SUGGESTION:
            if self.suggestion_kind is None or self.raw_input is not None:
                raise ValueError("suggestion input requires suggestion_kind and no raw_input")
        elif self.raw_input is None or self.suggestion_kind is not None:
            raise ValueError("natural language input requires raw_input and no suggestion_kind")
        _validate_date_range(
            self.start_date,
            self.end_date,
            _validation_today(info),
        )
        return self

    @property
    def day_count(self) -> int:
        return (self.end_date - self.start_date).days + 1

    @property
    def night_count(self) -> int:
        return self.day_count - 1


class FixedEvent(ContractModel):
    model_config = ConfigDict(json_schema_extra=FIXED_EVENT_SCHEMA_RULE)

    event_id: NonEmptyText
    kind: FixedEventKind
    title: NonEmptyText
    event_date: date
    place_text: NonEmptyText | None = None
    start_time: time | None = None
    end_time: time | None = None

    @model_validator(mode="after")
    def validate_times(self) -> FixedEvent:
        if (self.start_time is None) != (self.end_time is None):
            raise ValueError("start_time and end_time must be provided together")
        if (
            self.start_time is not None
            and self.end_time is not None
            and self.end_time <= self.start_time
        ):
            raise ValueError("end_time must be after start_time")
        return self


class SpecialConstraint(ContractModel):
    kind: ConstraintKind
    description: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=500),
    ]


class TripSetupSubmission(ContractModel):
    model_config = ConfigDict(json_schema_extra=TRIP_SETUP_SCHEMA_RULE)

    date_selection: DateSelection
    fixed_events: list[FixedEvent] = Field(default_factory=list)
    special_constraints: list[SpecialConstraint] = Field(default_factory=list)

    @model_validator(mode="after")
    def fixed_events_are_within_trip(self) -> TripSetupSubmission:
        for event in self.fixed_events:
            if not (
                self.date_selection.start_date <= event.event_date <= self.date_selection.end_date
            ):
                raise ValueError("fixed event must be within the trip date range")
        return self

    @property
    def day_count(self) -> int:
        return self.date_selection.day_count

    @property
    def night_count(self) -> int:
        return self.date_selection.night_count


class CitySetupCheckpoint(ContractModel):
    city_selection: CitySelection
    city_brief: CityBriefAcknowledgement
    profile_decision: PersonalProfileDecision
    trip_setup: TripSetupSubmission


CURRENT_P0_CONTRACTS: tuple[type[ContractModel], ...] = (
    TripShell,
    CitySelection,
    CityBriefAcknowledgement,
    ResolvedTripPreferences,
    PersonalProfileDecision,
    DateSuggestion,
    DateSelection,
    FixedEvent,
    SpecialConstraint,
    TripSetupSubmission,
    CitySetupCheckpoint,
)


def schema_context(today: date) -> dict[str, Any]:
    return {"today": today}
