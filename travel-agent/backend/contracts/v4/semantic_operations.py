"""Typed V4 semantic-operation proposals and committed audit records."""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import AwareDatetime, Field, RootModel, field_validator, model_validator

from backend.contracts.v4.base import (
    DisplayText,
    Identifier,
    TripGoalText,
    V4ContractModel,
    require_unique,
)
from backend.contracts.v4.content_quality import (
    require_meaningful_label,
    require_meaningful_trip_goal,
)
from backend.contracts.v4.enums import ConfidenceLevel, SemanticOperationStatus


class SemanticTargetV4(StrEnum):
    TRIP_BASICS = "trip_basics"
    ATTRACTION_PREFERENCE = "attraction_preference"
    ATTRACTION_ENTITY = "attraction_entity"
    DINING_PREFERENCE = "dining_preference"
    DINING_REQUIREMENT = "dining_requirement"
    DINING_ENTITY = "dining_entity"
    LODGING_AREA = "lodging_area"
    LODGING_CLASS = "lodging_class"
    LODGING_BOOKING = "lodging_booking"
    TRANSPORT_AND_PACE = "transport_and_pace"
    GENERAL_CONSTRAINT = "general_constraint"
    FINAL_SUPPLEMENT = "final_supplement"
    TASK_BOOK_CONFIRMATION = "task_book_confirmation"
    CONFLICT = "conflict"


class SemanticDomainV4(StrEnum):
    GENERAL = "general"
    ATTRACTION = "attraction"
    DINING = "dining"
    LODGING = "lodging"
    TRANSPORT = "transport"


class AttractionDisposition(StrEnum):
    MUST = "must"
    WANT = "want"
    IF_CONVENIENT = "if_convenient"
    AVOID = "avoid"


class DiningDisposition(StrEnum):
    DESTINATION = "destination"
    IF_CONVENIENT = "if_convenient"
    AVOID = "avoid"


class SemanticOperationProposalBase(V4ContractModel):
    local_operation_key: Identifier
    target: SemanticTargetV4
    source_refs: list[Identifier] = Field(min_length=1)
    confidence: ConfidenceLevel

    @model_validator(mode="after")
    def source_refs_are_unique(self) -> SemanticOperationProposalBase:
        require_unique(self.source_refs, "source_refs")
        return self


class SetTripBasicsOperation(SemanticOperationProposalBase):
    operation_type: Literal["set_trip_basics"]
    target: Literal[SemanticTargetV4.TRIP_BASICS]
    domain: Literal[SemanticDomainV4.GENERAL]
    destination_name: DisplayText | None = None
    destination_canonical_id: Identifier | None = None
    start_date: date | None = None
    end_date: date | None = None
    duration_days: int | None = Field(default=None, ge=1, le=5, strict=True)
    travelers: list[DisplayText] | None = Field(default=None, min_length=1, max_length=20)
    trip_goals: list[TripGoalText] | None = Field(default=None, min_length=1, max_length=20)

    @field_validator("trip_goals")
    @classmethod
    def trip_goals_are_readable(cls, values: list[str] | None) -> list[str] | None:
        for value in values or []:
            require_meaningful_trip_goal(value)
        return values

    @model_validator(mode="after")
    def contains_an_explicit_basic_update(self) -> SetTripBasicsOperation:
        if not any(
            (
                self.destination_name is not None,
                self.start_date is not None,
                self.duration_days is not None,
                self.travelers is not None,
                self.trip_goals is not None,
            )
        ):
            raise ValueError("trip basics update requires at least one explicit field")
        if self.destination_canonical_id is not None and self.destination_name is None:
            raise ValueError("destination canonical ID requires destination name")
        if (self.start_date is None) != (self.end_date is None):
            raise ValueError("trip basics dates must be set together")
        if self.start_date is not None and self.end_date is not None:
            if self.end_date < self.start_date:
                raise ValueError("trip basics end date cannot be before start date")
            expected_duration = (self.end_date - self.start_date).days + 1
            if expected_duration > 5:
                raise ValueError("trip basics date range cannot exceed five days")
            if self.duration_days is not None and self.duration_days != expected_duration:
                raise ValueError("duration_days must match the inclusive date range")
        if self.travelers is not None:
            require_unique((item.casefold() for item in self.travelers), "travelers")
        if self.trip_goals is not None:
            require_unique((item.casefold() for item in self.trip_goals), "trip goals")
        return self


class SetNoPreferenceOperation(SemanticOperationProposalBase):
    operation_type: Literal["set_no_preference"]
    domain: SemanticDomainV4


class SetNotApplicableOperation(SemanticOperationProposalBase):
    operation_type: Literal["set_not_applicable"]
    reason: DisplayText


class SetDelegationScopeOperation(SemanticOperationProposalBase):
    operation_type: Literal["set_delegation_scope"]
    domain: SemanticDomainV4
    delegated_targets: list[Identifier] = Field(min_length=1)
    boundary_refs: list[Identifier] = Field(default_factory=list)


class SetExistingBookingOperation(SemanticOperationProposalBase):
    operation_type: Literal["set_existing_booking"]
    domain: SemanticDomainV4
    booking_kind: Literal["transport", "lodging", "restaurant", "attraction", "activity"]
    user_description: DisplayText
    canonical_entity_id: Identifier | None = None
    start_date: date | None = None
    end_date: date | None = None


class SelectPreferenceDirectionOperation(SemanticOperationProposalBase):
    operation_type: Literal["select_preference_direction"]
    domain: SemanticDomainV4
    direction_id: Identifier
    label: DisplayText
    description: DisplayText | None = None
    tags: list[Identifier] = Field(default_factory=list, max_length=8)
    search_query: DisplayText | None = None

    @field_validator("label")
    @classmethod
    def label_is_readable(cls, value: str) -> str:
        return require_meaningful_label(value, "preference direction label")


class ExcludePreferenceDirectionOperation(SemanticOperationProposalBase):
    operation_type: Literal["exclude_preference_direction"]
    domain: SemanticDomainV4
    direction_id: Identifier
    label: DisplayText
    description: DisplayText | None = None
    tags: list[Identifier] = Field(default_factory=list, max_length=8)
    search_query: DisplayText | None = None

    @field_validator("label")
    @classmethod
    def label_is_readable(cls, value: str) -> str:
        return require_meaningful_label(value, "preference direction label")


class SetLodgingClassPreferenceOperation(SemanticOperationProposalBase):
    operation_type: Literal["set_lodging_class_preference"]
    domain: Literal[SemanticDomainV4.LODGING]
    hotel_quality_tier: Literal["economy", "comfort", "upscale", "luxury"] | None = None
    property_type: DisplayText | None = None
    nightly_budget_minimum_minor: int | None = Field(default=None, ge=0, strict=True)
    nightly_budget_maximum_minor: int | None = Field(default=None, ge=0, strict=True)

    @model_validator(mode="after")
    def contains_a_lodging_preference(self) -> SetLodgingClassPreferenceOperation:
        if not any(
            (
                self.hotel_quality_tier,
                self.property_type,
                self.nightly_budget_minimum_minor is not None,
                self.nightly_budget_maximum_minor is not None,
            )
        ):
            raise ValueError("lodging class preference requires tier, type, or budget")
        if (
            self.nightly_budget_minimum_minor is not None
            and self.nightly_budget_maximum_minor is not None
            and self.nightly_budget_maximum_minor < self.nightly_budget_minimum_minor
        ):
            raise ValueError("lodging budget maximum cannot be below minimum")
        return self


class SelectConcreteEntityOperation(SemanticOperationProposalBase):
    operation_type: Literal["select_concrete_entity"]
    domain: Literal[SemanticDomainV4.ATTRACTION, SemanticDomainV4.DINING]
    canonical_entity_id: Identifier
    display_name: DisplayText
    disposition: Literal["must", "want", "destination", "if_convenient", "avoid"]

    @field_validator("display_name")
    @classmethod
    def display_name_is_readable(cls, value: str) -> str:
        return require_meaningful_label(value, "concrete entity display name")

    @model_validator(mode="after")
    def disposition_matches_domain(self) -> SelectConcreteEntityOperation:
        if self.domain is SemanticDomainV4.ATTRACTION and self.disposition not in {
            item.value for item in AttractionDisposition
        }:
            raise ValueError("attraction entity requires an attraction disposition")
        if self.domain is SemanticDomainV4.DINING and self.disposition not in {
            item.value for item in DiningDisposition
        }:
            raise ValueError("dining entity requires a dining disposition")
        return self


class ExcludeConcreteEntityOperation(SemanticOperationProposalBase):
    operation_type: Literal["exclude_concrete_entity"]
    domain: Literal[SemanticDomainV4.ATTRACTION, SemanticDomainV4.DINING]
    canonical_entity_id: Identifier
    display_name: DisplayText

    @field_validator("display_name")
    @classmethod
    def display_name_is_readable(cls, value: str) -> str:
        return require_meaningful_label(value, "concrete entity display name")


class RevokePriorIntentOperation(SemanticOperationProposalBase):
    operation_type: Literal["revoke_prior_intent"]
    prior_operation_id: Identifier
    reason: DisplayText


class AddConditionalRequirementOperation(SemanticOperationProposalBase):
    operation_type: Literal["add_conditional_requirement"]
    domain: SemanticDomainV4
    condition: DisplayText
    required_outcome: DisplayText
    entity_ref: Identifier | None = None


class ResolveConflictOperation(SemanticOperationProposalBase):
    operation_type: Literal["resolve_conflict"]
    conflict_id: Identifier
    accepted_operation_ids: list[Identifier] = Field(min_length=1)
    rejected_operation_ids: list[Identifier] = Field(default_factory=list)


class ConfirmFinalSupplementOperation(SemanticOperationProposalBase):
    operation_type: Literal["confirm_final_supplement"]
    target: Literal[SemanticTargetV4.FINAL_SUPPLEMENT]
    supplement_text: DisplayText | None = None
    explicitly_no_more_requirements: bool


class ConfirmTaskBookOperation(SemanticOperationProposalBase):
    operation_type: Literal["confirm_task_book"]
    target: Literal[SemanticTargetV4.TASK_BOOK_CONFIRMATION]
    task_book_id: Identifier
    task_book_version: int = Field(ge=1, strict=True)
    based_on_state_version: int = Field(ge=0, strict=True)
    confirmed_at: AwareDatetime


SemanticOperationProposalValue = Annotated[
    SetTripBasicsOperation
    | SetNoPreferenceOperation
    | SetNotApplicableOperation
    | SetDelegationScopeOperation
    | SetExistingBookingOperation
    | SelectPreferenceDirectionOperation
    | ExcludePreferenceDirectionOperation
    | SetLodgingClassPreferenceOperation
    | SelectConcreteEntityOperation
    | ExcludeConcreteEntityOperation
    | RevokePriorIntentOperation
    | AddConditionalRequirementOperation
    | ResolveConflictOperation
    | ConfirmFinalSupplementOperation
    | ConfirmTaskBookOperation,
    Field(discriminator="operation_type"),
]


class SemanticOperationProposal(RootModel[SemanticOperationProposalValue]):
    """One strict LLM/signed-card operation proposal."""


class SemanticOperationRecord(V4ContractModel):
    operation_id: Identifier
    trip_id: Identifier
    turn_id: Identifier
    applied_state_version: int = Field(ge=1, strict=True)
    proposal: SemanticOperationProposal
    status: SemanticOperationStatus
    committed_at: AwareDatetime
    supersedes_operation_ids: list[Identifier] = Field(default_factory=list)

    @model_validator(mode="after")
    def superseded_ids_are_unique(self) -> SemanticOperationRecord:
        require_unique(self.supersedes_operation_ids, "supersedes_operation_ids")
        return self


V4_SEMANTIC_OPERATION_CONTRACTS = (
    SemanticOperationProposal,
    SemanticOperationRecord,
)
