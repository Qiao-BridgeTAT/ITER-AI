"""P0-08 itinerary, planning issue, assumption, and cost contracts."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, time, timedelta
from typing import Any
from uuid import UUID

from pydantic import ConfigDict, Field, model_validator

from backend.contracts.base import ContractModel
from backend.contracts.common import CnyAmountRange, NonEmptyText, ShortText
from backend.contracts.enums import (
    AssumptionKind,
    CityCode,
    ConfirmationStatus,
    CostCategory,
    DataAvailability,
    EvidenceSource,
    ExcludedCostKind,
    IssueSeverity,
    ItineraryEntryKind,
    PlanningIssueCode,
    TransportMode,
)

ENTRY_TIME_SCHEMA_RULE: dict[str, Any] = {
    "x-travel-time-range": {"startField": "start_time", "endField": "end_time"}
}

DAY_PLAN_SCHEMA_RULE: dict[str, Any] = {
    "x-travel-day-plan": {
        "entriesField": "entries",
        "legsField": "transport_legs",
    }
}

COST_LINE_ITEM_SCHEMA_RULE: dict[str, Any] = {
    "allOf": [
        {
            "if": {
                "properties": {"availability": {"const": "missing"}},
                "required": ["availability"],
            },
            "then": {
                "properties": {
                    "amount_per_person": {"type": "null"},
                    "missing_reason": {
                        "minLength": 1,
                        "pattern": r"\S",
                        "type": "string",
                    },
                },
                "required": ["missing_reason"],
            },
            "else": {
                "properties": {
                    "amount_per_person": {"not": {"type": "null"}},
                    "source_fact_ids": {"type": "array", "minItems": 1},
                },
                "required": ["amount_per_person", "source_fact_ids"],
            },
        },
        {
            "if": {
                "properties": {"availability": {"const": "available"}},
                "required": ["availability"],
            },
            "then": {"properties": {"missing_reason": {"type": "null"}}},
        },
        {
            "if": {
                "properties": {"availability": {"const": "partial"}},
                "required": ["availability"],
            },
            "then": {
                "properties": {
                    "missing_reason": {
                        "minLength": 1,
                        "pattern": r"\S",
                        "type": "string",
                    }
                },
                "required": ["missing_reason"],
            },
        },
    ]
}

LODGING_SHARE_SCHEMA_RULE: dict[str, Any] = {
    "allOf": [
        {
            "if": {
                "properties": {"lodging_share_source": {"const": "system_default"}},
                "required": ["lodging_share_source"],
            },
            "then": {"properties": {"lodging_share_divisor": {"const": 2}}},
        }
    ],
    "x-travel-cost-total": {
        "itemsField": "items",
        "amountField": "amount_per_person",
        "totalField": "total_per_person",
    },
}

ITINERARY_SCHEMA_RULE: dict[str, Any] = {
    "x-travel-itinerary": {
        "startField": "start_date",
        "endField": "end_date",
        "daysField": "days",
        "dayDateField": "date",
    }
}

ITINERARY_RESULT_SCHEMA_RULE: dict[str, Any] = {
    "x-travel-itinerary-result": {
        "itineraryField": "itinerary",
        "costField": "cost_estimate",
        "daysField": "days",
        "dailyCostField": "daily_cost_per_person",
        "totalField": "total_per_person",
    }
}


def _unique(values: Sequence[str | UUID], field_name: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} must not contain duplicates")


class ItineraryEntry(ContractModel):
    model_config = ConfigDict(json_schema_extra=ENTRY_TIME_SCHEMA_RULE)

    entry_id: NonEmptyText
    kind: ItineraryEntryKind
    title: NonEmptyText
    place_id: UUID | None = None
    start_time: time
    end_time: time
    source_fact_ids: list[NonEmptyText] = Field(
        default_factory=list,
        json_schema_extra={"uniqueItems": True},
    )
    note: ShortText | None = None

    @model_validator(mode="after")
    def time_and_place_are_valid(self) -> ItineraryEntry:
        if self.end_time <= self.start_time:
            raise ValueError("itinerary entry end_time must be after start_time")
        _unique(self.source_fact_ids, "source_fact_ids")
        place_kinds = {
            ItineraryEntryKind.ATTRACTION,
            ItineraryEntryKind.RESTAURANT,
            ItineraryEntryKind.HOTEL,
        }
        if self.kind in place_kinds and self.place_id is None:
            raise ValueError("place itinerary entry requires a canonical place_id")
        if self.kind in {ItineraryEntryKind.REST, ItineraryEntryKind.BUFFER} and (
            self.place_id is not None
        ):
            raise ValueError("rest and buffer entries cannot claim a place")
        return self


class TransportLeg(ContractModel):
    leg_id: NonEmptyText
    from_entry_id: NonEmptyText
    to_entry_id: NonEmptyText
    mode: TransportMode
    availability: DataAvailability
    distance_m: int | None = Field(default=None, ge=0, strict=True)
    duration_minutes: int | None = Field(default=None, ge=0, strict=True)
    walking_m: int | None = Field(default=None, ge=0, strict=True)
    fare: CnyAmountRange | None = None
    source_fact_ids: list[NonEmptyText] = Field(
        default_factory=list,
        json_schema_extra={"uniqueItems": True},
    )
    missing_reason: ShortText | None = None

    @model_validator(mode="after")
    def route_availability_is_explicit(self) -> TransportLeg:
        _unique(self.source_fact_ids, "source_fact_ids")
        if self.from_entry_id == self.to_entry_id:
            raise ValueError("transport leg endpoints must differ")
        if self.availability is DataAvailability.MISSING:
            if any(
                value is not None
                for value in (self.distance_m, self.duration_minutes, self.walking_m, self.fare)
            ):
                raise ValueError(
                    "missing route cannot contain fabricated distance, duration, walking, or fare"
                )
            if self.missing_reason is None:
                raise ValueError("missing route requires a reason")
        elif self.distance_m is None or self.duration_minutes is None:
            raise ValueError("available or partial route requires distance and duration")
        elif not self.source_fact_ids:
            raise ValueError("available or partial route requires traceable source facts")
        return self


class DayPlan(ContractModel):
    model_config = ConfigDict(json_schema_extra=DAY_PLAN_SCHEMA_RULE)

    date: date
    theme: NonEmptyText
    entries: list[ItineraryEntry] = Field(min_length=1)
    transport_legs: list[TransportLeg] = Field(default_factory=list)
    daily_cost_per_person: CnyAmountRange
    estimated_walking_m: int | None = Field(default=None, ge=0, strict=True)
    rest_minutes: int = Field(default=0, ge=0, strict=True)
    buffer_minutes: int = Field(default=0, ge=0, strict=True)

    @model_validator(mode="after")
    def timeline_and_legs_are_consistent(self) -> DayPlan:
        _unique([entry.entry_id for entry in self.entries], "entry_id")
        _unique([leg.leg_id for leg in self.transport_legs], "leg_id")
        for previous, current in zip(self.entries, self.entries[1:], strict=False):
            if current.start_time < previous.end_time:
                raise ValueError("itinerary entries cannot overlap or run backwards")
        expected_edges = [
            (previous.entry_id, current.entry_id)
            for previous, current in zip(self.entries, self.entries[1:], strict=False)
        ]
        actual_edges = [(leg.from_entry_id, leg.to_entry_id) for leg in self.transport_legs]
        if actual_edges != expected_edges:
            raise ValueError("transport legs must connect each pair of consecutive entries")
        return self


class Itinerary(ContractModel):
    model_config = ConfigDict(json_schema_extra=ITINERARY_SCHEMA_RULE)

    city: CityCode
    start_date: date
    end_date: date
    days: list[DayPlan] = Field(min_length=1, max_length=5)

    @model_validator(mode="after")
    def days_cover_the_trip_exactly(self) -> Itinerary:
        day_count = (self.end_date - self.start_date).days + 1
        if not 1 <= day_count <= 5:
            raise ValueError("itinerary length must be between 1 and 5 days")
        expected_dates = [self.start_date + timedelta(days=index) for index in range(day_count)]
        if [day.date for day in self.days] != expected_dates:
            raise ValueError("itinerary days must cover every trip date exactly once and in order")
        return self


class PlanningIssue(ContractModel):
    issue_id: NonEmptyText
    code: PlanningIssueCode
    severity: IssueSeverity
    message: ShortText
    affected_entry_ids: list[NonEmptyText] = Field(
        default_factory=list,
        json_schema_extra={"uniqueItems": True},
    )
    resolved: bool = False

    @model_validator(mode="after")
    def affected_entries_are_unique(self) -> PlanningIssue:
        _unique(self.affected_entry_ids, "affected_entry_ids")
        return self


class Assumption(ContractModel):
    assumption_id: NonEmptyText
    kind: AssumptionKind
    description: ShortText
    source: EvidenceSource


class CostLineItem(ContractModel):
    model_config = ConfigDict(json_schema_extra=COST_LINE_ITEM_SCHEMA_RULE)

    category: CostCategory
    availability: DataAvailability
    amount_per_person: CnyAmountRange | None = None
    source_fact_ids: list[NonEmptyText] = Field(
        default_factory=list,
        json_schema_extra={"uniqueItems": True},
    )
    missing_reason: ShortText | None = None

    @model_validator(mode="after")
    def amount_matches_availability(self) -> CostLineItem:
        _unique(self.source_fact_ids, "source_fact_ids")
        if self.availability is DataAvailability.MISSING:
            if self.amount_per_person is not None or self.missing_reason is None:
                raise ValueError("missing cost requires no amount and an explicit reason")
        elif self.amount_per_person is None or not self.source_fact_ids:
            raise ValueError("available or partial cost requires an amount range and source facts")
        elif self.availability is DataAvailability.AVAILABLE and self.missing_reason is not None:
            raise ValueError("available cost cannot contain a missing-data reason")
        elif self.availability is DataAvailability.PARTIAL and self.missing_reason is None:
            raise ValueError("partial cost requires an explicit missing-data reason")
        return self


class CostEstimate(ContractModel):
    model_config = ConfigDict(json_schema_extra=LODGING_SHARE_SCHEMA_RULE)

    items: list[CostLineItem] = Field(min_length=1)
    total_per_person: CnyAmountRange
    lodging_share_divisor: int = Field(default=2, ge=1, strict=True)
    lodging_share_source: EvidenceSource = EvidenceSource.SYSTEM_DEFAULT
    excluded_costs: list[ExcludedCostKind] = Field(
        default_factory=lambda: list(ExcludedCostKind),
        json_schema_extra={"uniqueItems": True},
    )
    fetched_at_note: ShortText

    @model_validator(mode="after")
    def totals_and_scope_are_valid(self) -> CostEstimate:
        categories = [item.category for item in self.items]
        _unique(categories, "cost category")
        _unique(self.excluded_costs, "excluded_costs")
        if set(self.excluded_costs) != set(ExcludedCostKind):
            raise ValueError("airfare, rail, and intercity transport must remain excluded")
        if self.lodging_share_source is EvidenceSource.SYSTEM_DEFAULT:
            if self.lodging_share_divisor != 2:
                raise ValueError("system-default lodging cost must use a two-person split")
        elif self.lodging_share_source is not EvidenceSource.DIALOGUE:
            raise ValueError("lodging share can only come from system default or user dialogue")

        priced = [item.amount_per_person for item in self.items if item.amount_per_person]
        minimum = sum(item.minimum_fen for item in priced)
        maximum = sum(item.maximum_fen for item in priced)
        if (
            self.total_per_person.minimum_fen != minimum
            or self.total_per_person.maximum_fen != maximum
        ):
            raise ValueError("total_per_person must equal the sum of priced line items")
        return self

    @property
    def is_partial(self) -> bool:
        return any(item.availability is DataAvailability.MISSING for item in self.items)


class TaskBook(ContractModel):
    # `city` is the stage-zero compatibility key. New V3 planning uses the
    # registry-backed `city_id`, so adding a city no longer requires extending
    # the legacy Beijing/Nanjing enum.
    city: CityCode | None = None
    city_id: NonEmptyText | None = None
    start_date: date
    end_date: date
    preference_summary: NonEmptyText
    strong_attraction_ids: list[UUID] = Field(
        default_factory=list,
        json_schema_extra={"uniqueItems": True},
    )
    important_restaurant_ids: list[UUID] = Field(
        default_factory=list,
        json_schema_extra={"uniqueItems": True},
    )
    selected_hotel_id: UUID | None = None
    key_constraints: list[ShortText] = Field(default_factory=list)
    tradeoffs: list[ShortText] = Field(default_factory=list)
    omitted_strong_desires: list[ShortText] = Field(default_factory=list)
    assumptions: list[Assumption] = Field(default_factory=list)
    status: ConfirmationStatus = ConfirmationStatus.PENDING

    @model_validator(mode="after")
    def references_and_dates_are_valid(self) -> TaskBook:
        if (self.city is None) == (self.city_id is None):
            raise ValueError("task book requires exactly one city or registry city_id")
        if self.end_date < self.start_date:
            raise ValueError("task book end_date cannot be before start_date")
        _unique(self.strong_attraction_ids, "strong_attraction_ids")
        _unique(self.important_restaurant_ids, "important_restaurant_ids")
        _unique(self.key_constraints, "key_constraints")
        _unique(self.tradeoffs, "tradeoffs")
        _unique(self.omitted_strong_desires, "omitted_strong_desires")
        return self


class ItineraryResult(ContractModel):
    model_config = ConfigDict(json_schema_extra=ITINERARY_RESULT_SCHEMA_RULE)

    itinerary: Itinerary
    cost_estimate: CostEstimate
    issues: list[PlanningIssue] = Field(default_factory=list)
    assumptions: list[Assumption] = Field(default_factory=list)

    @model_validator(mode="after")
    def result_ids_are_unique(self) -> ItineraryResult:
        _unique([issue.issue_id for issue in self.issues], "issue_id")
        _unique([assumption.assumption_id for assumption in self.assumptions], "assumption_id")
        daily_minimum = sum(day.daily_cost_per_person.minimum_fen for day in self.itinerary.days)
        daily_maximum = sum(day.daily_cost_per_person.maximum_fen for day in self.itinerary.days)
        if (
            daily_minimum != self.cost_estimate.total_per_person.minimum_fen
            or daily_maximum != self.cost_estimate.total_per_person.maximum_fen
        ):
            raise ValueError("daily cost subtotals must equal the full-trip cost total")
        return self


P0_ITINERARY_CONTRACTS: tuple[type[ContractModel], ...] = (
    Itinerary,
    PlanningIssue,
    CostEstimate,
    TaskBook,
    ItineraryResult,
)
