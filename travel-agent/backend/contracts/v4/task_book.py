"""Versioned V4 task-book contract shared by Prepare and Planner."""

from __future__ import annotations

from datetime import date

from pydantic import AwareDatetime, Field, model_validator

from backend.contracts.v4.base import DisplayText, Identifier, V4ContractModel, require_unique
from backend.contracts.v4.enums import TaskBookStatus
from backend.contracts.v4.semantic_operations import (
    AttractionDisposition,
    DiningDisposition,
)


class MoneyRange(V4ContractModel):
    currency: str = Field(default="CNY", pattern=r"^[A-Z]{3}$")
    minimum_minor: int | None = Field(default=None, ge=0, strict=True)
    maximum_minor: int | None = Field(default=None, ge=0, strict=True)

    @model_validator(mode="after")
    def range_is_ordered(self) -> MoneyRange:
        if (
            self.minimum_minor is not None
            and self.maximum_minor is not None
            and self.maximum_minor < self.minimum_minor
        ):
            raise ValueError("maximum_minor cannot be below minimum_minor")
        return self


class EvidenceBackedText(V4ContractModel):
    value: DisplayText
    source_evidence_refs: list[Identifier] = Field(min_length=1)


class TaskBookEntityIntent(V4ContractModel):
    canonical_entity_id: Identifier
    display_name: DisplayText
    disposition: AttractionDisposition | DiningDisposition
    source_operation_refs: list[Identifier] = Field(min_length=1)
    facts_to_verify: list[Identifier] = Field(default_factory=list)


class DelegatedScope(V4ContractModel):
    domain: Identifier
    delegated_targets: list[Identifier] = Field(min_length=1)
    boundary_refs: list[Identifier] = Field(default_factory=list)
    source_operation_refs: list[Identifier] = Field(min_length=1)


class BookingReference(V4ContractModel):
    booking_id: Identifier
    booking_kind: Identifier
    user_description: DisplayText
    canonical_entity_id: Identifier | None = None
    start_date: date | None = None
    end_date: date | None = None
    source_operation_refs: list[Identifier] = Field(min_length=1)


class DestinationAndDates(V4ContractModel):
    destination_name: DisplayText
    destination_canonical_id: Identifier | None = None
    start_date: date
    end_date: date
    duration_days: int = Field(ge=1, le=5, strict=True)
    source_evidence_refs: list[Identifier] = Field(min_length=1)

    @model_validator(mode="after")
    def dates_match_duration(self) -> DestinationAndDates:
        if self.end_date < self.start_date:
            raise ValueError("end_date cannot be before start_date")
        if (self.end_date - self.start_date).days + 1 != self.duration_days:
            raise ValueError("duration_days must match the inclusive date range")
        return self


class TravelersAndTripGoal(V4ContractModel):
    travelers: list[DisplayText]
    trip_goals: list[EvidenceBackedText]


class PaceAndTransport(V4ContractModel):
    pace_preferences: list[EvidenceBackedText] = Field(default_factory=list)
    transport_preferences: list[EvidenceBackedText] = Field(default_factory=list)


class AttractionDirection(V4ContractModel):
    preferences: list[EvidenceBackedText] = Field(default_factory=list)
    must_visit: list[TaskBookEntityIntent] = Field(default_factory=list)
    wanted: list[TaskBookEntityIntent] = Field(default_factory=list)
    if_convenient: list[TaskBookEntityIntent] = Field(default_factory=list)
    exclusions: list[TaskBookEntityIntent] = Field(default_factory=list)
    delegated_scope: DelegatedScope | None = None

    @model_validator(mode="after")
    def intent_lists_match_attraction_dispositions(self) -> AttractionDirection:
        groups = (
            (self.must_visit, AttractionDisposition.MUST.value),
            (self.wanted, AttractionDisposition.WANT.value),
            (self.if_convenient, AttractionDisposition.IF_CONVENIENT.value),
            (self.exclusions, AttractionDisposition.AVOID.value),
        )
        all_items: list[TaskBookEntityIntent] = []
        for items, expected in groups:
            if any(item.disposition.value != expected for item in items):
                raise ValueError("attraction intent list does not match its disposition")
            all_items.extend(items)
        require_unique(
            (item.canonical_entity_id for item in all_items),
            "attraction canonical_entity_id",
        )
        return self


class DiningDirection(V4ContractModel):
    preferences: list[EvidenceBackedText] = Field(default_factory=list)
    hard_requirements: list[EvidenceBackedText] = Field(default_factory=list)
    destination_restaurants: list[TaskBookEntityIntent] = Field(default_factory=list)
    if_convenient_restaurants: list[TaskBookEntityIntent] = Field(default_factory=list)
    excluded_restaurants: list[TaskBookEntityIntent] = Field(default_factory=list)
    delegated_scope: DelegatedScope | None = None

    @model_validator(mode="after")
    def intent_lists_match_dining_dispositions(self) -> DiningDirection:
        groups = (
            (self.destination_restaurants, DiningDisposition.DESTINATION.value),
            (self.if_convenient_restaurants, DiningDisposition.IF_CONVENIENT.value),
            (self.excluded_restaurants, DiningDisposition.AVOID.value),
        )
        all_items: list[TaskBookEntityIntent] = []
        for items, expected in groups:
            if any(item.disposition.value != expected for item in items):
                raise ValueError("dining intent list does not match its disposition")
            all_items.extend(items)
        require_unique(
            (item.canonical_entity_id for item in all_items),
            "dining canonical_entity_id",
        )
        return self


class LodgingDirection(V4ContractModel):
    area_preferences: list[EvidenceBackedText] = Field(default_factory=list)
    hotel_quality_tier: str | None = Field(
        default=None,
        pattern=r"^(economy|comfort|upscale|luxury)$",
    )
    property_type_preferences: list[EvidenceBackedText] = Field(default_factory=list)
    nightly_budget: MoneyRange | None = None
    facility_requirements: list[EvidenceBackedText] = Field(default_factory=list)
    existing_booking: BookingReference | None = None
    delegated_scope: DelegatedScope | None = None
    not_applicable: bool = False

    @model_validator(mode="after")
    def not_applicable_has_no_lodging_choice(self) -> LodgingDirection:
        if self.not_applicable and any(
            (
                self.area_preferences,
                self.hotel_quality_tier,
                self.property_type_preferences,
                self.nightly_budget,
                self.facility_requirements,
                self.existing_booking,
                self.delegated_scope,
            )
        ):
            raise ValueError("not-applicable lodging cannot contain lodging preferences")
        return self


class TaskBookV4(V4ContractModel):
    task_book_id: Identifier
    version: int = Field(ge=1, strict=True)
    based_on_state_version: int = Field(ge=0, strict=True)
    status: TaskBookStatus
    created_at: AwareDatetime
    confirmed_at: AwareDatetime | None = None
    destination_and_dates: DestinationAndDates
    travelers_and_trip_goal: TravelersAndTripGoal
    pace_and_transport: PaceAndTransport
    attraction_direction: AttractionDirection
    dining_direction: DiningDirection
    lodging_direction: LodgingDirection
    hard_constraints: list[EvidenceBackedText] = Field(default_factory=list)
    existing_bookings: list[BookingReference] = Field(default_factory=list)
    tradeoffs_and_assumptions: list[EvidenceBackedText] = Field(default_factory=list)
    unresolved_non_blocking_items: list[EvidenceBackedText] = Field(default_factory=list)
    source_evidence_refs: list[Identifier] = Field(min_length=1)

    @model_validator(mode="after")
    def confirmation_and_references_are_consistent(self) -> TaskBookV4:
        if self.status is TaskBookStatus.CONFIRMED and self.confirmed_at is None:
            raise ValueError("confirmed task book requires confirmed_at")
        if self.status is not TaskBookStatus.CONFIRMED and self.confirmed_at is not None:
            raise ValueError("only a confirmed task book can contain confirmed_at")
        require_unique(self.source_evidence_refs, "source_evidence_refs")
        require_unique((item.booking_id for item in self.existing_bookings), "booking_id")
        return self


V4_TASK_BOOK_CONTRACTS = (
    TaskBookV4,
    DestinationAndDates,
    TravelersAndTripGoal,
    PaceAndTransport,
    AttractionDirection,
    DiningDirection,
    LodgingDirection,
)
