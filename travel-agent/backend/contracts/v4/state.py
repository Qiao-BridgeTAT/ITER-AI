"""Durable V4 semantic and discovery states."""

from __future__ import annotations

from datetime import date
from typing import Literal
from uuid import UUID

from pydantic import AwareDatetime, ConfigDict, Field, field_validator, model_validator

from backend.agent.state_merge import (
    SemanticInvalidation,
    SemanticMergeAudit,
    SemanticMergeConflict,
    SemanticStateEntry,
)
from backend.contracts.cold_start import ColdStartSubmission
from backend.contracts.v4.attraction_search import AttractionSearchHints
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
from backend.contracts.v4.dining_search import DiningSearchHints
from backend.contracts.v4.enums import (
    CompletionMode,
    CoverageStatus,
    DiscoverySection,
    InteractionStatus,
    PendingInteractionKind,
    TaskBookStatus,
)
from backend.contracts.v4.lodging_preferences import (
    HotelQualityTier,
    LodgingExample,
)
from backend.contracts.v4.memory import UserMemoryView
from backend.contracts.v4.planner_publication import PlannerPublishedPlan
from backend.contracts.v4.task_book import BookingReference, DelegatedScope, MoneyRange, TaskBookV4

_COVERAGE_SECTIONS = frozenset(
    {
        DiscoverySection.OTHER,
        DiscoverySection.ATTRACTION_PREFERENCE,
        DiscoverySection.ATTRACTION_SPECIFIC,
        DiscoverySection.DINING_PREFERENCE,
        DiscoverySection.DINING_SPECIFIC,
        DiscoverySection.LODGING_AREA_PREFERENCE,
        DiscoverySection.LODGING_CLASS_PREFERENCE,
        DiscoverySection.FINAL_SUPPLEMENT,
    }
)


class ColdStartProfileSnapshot(V4ContractModel):
    profile_version: int = Field(ge=1, strict=True)
    captured_at: AwareDatetime
    preferences: ColdStartSubmission
    source_evidence_refs: list[Identifier] = Field(min_length=1)


class TripBasicsProjection(V4ContractModel):
    destination_name: DisplayText | None = None
    destination_canonical_id: Identifier | None = None
    start_date: date | None = None
    end_date: date | None = None
    duration_days: int | None = Field(default=None, ge=1, le=5, strict=True)
    travelers: list[DisplayText] = Field(default_factory=list)
    trip_goals: list[TripGoalText] = Field(default_factory=list)
    destination_source_operation_refs: list[Identifier] = Field(default_factory=list)
    date_source_operation_refs: list[Identifier] = Field(default_factory=list)
    traveler_source_operation_refs: list[Identifier] = Field(default_factory=list)
    trip_goal_source_operation_refs: list[Identifier] = Field(default_factory=list)

    @field_validator("trip_goals")
    @classmethod
    def trip_goals_are_readable(cls, values: list[str]) -> list[str]:
        for value in values:
            require_meaningful_trip_goal(value)
        return values

    @model_validator(mode="after")
    def source_refs_are_unique(self) -> TripBasicsProjection:
        require_unique(
            self.destination_source_operation_refs,
            "destination source operation refs",
        )
        require_unique(
            self.date_source_operation_refs,
            "date source operation refs",
        )
        require_unique(
            self.traveler_source_operation_refs,
            "traveler source operation refs",
        )
        require_unique(
            self.trip_goal_source_operation_refs,
            "trip goal source operation refs",
        )
        return self

    @model_validator(mode="after")
    def dates_are_consistent(self) -> TripBasicsProjection:
        if (self.start_date is None) != (self.end_date is None):
            raise ValueError("start_date and end_date must be set together")
        if self.start_date is not None and self.end_date is not None:
            if self.end_date < self.start_date:
                raise ValueError("end_date cannot be before start_date")
            expected = (self.end_date - self.start_date).days + 1
            if self.duration_days is not None and self.duration_days != expected:
                raise ValueError("duration_days must match the inclusive date range")
        return self


class PreferenceDirectionState(V4ContractModel):
    direction_id: Identifier
    label: DisplayText
    description: DisplayText | None = None
    tags: list[Identifier] = Field(default_factory=list, max_length=8)
    search_query: DisplayText | None = None
    attraction_search_hints: AttractionSearchHints | None = None
    dining_search_hints: DiningSearchHints | None = None
    lodging_examples: list[LodgingExample] = Field(default_factory=list, max_length=3)
    selected: bool
    source_operation_refs: list[Identifier] = Field(min_length=1)
    coverage_eligible: bool = True

    @field_validator("label")
    @classmethod
    def label_is_readable(cls, value: str) -> str:
        return require_meaningful_label(value, "preference direction label")


class ConcreteIntentState(V4ContractModel):
    canonical_entity_id: Identifier
    display_name: DisplayText
    disposition: str = Field(pattern=r"^(must|want|destination|if_convenient|avoid)$")
    source_operation_refs: list[Identifier] = Field(min_length=1)

    @field_validator("display_name")
    @classmethod
    def display_name_is_readable(cls, value: str) -> str:
        return require_meaningful_label(value, "concrete entity display name")


class AttractionSemanticProjection(V4ContractModel):
    preference_directions: list[PreferenceDirectionState] = Field(default_factory=list)
    concrete_intents: list[ConcreteIntentState] = Field(default_factory=list)
    exclusions: list[ConcreteIntentState] = Field(default_factory=list)
    delegation: DelegatedScope | None = None


class DiningSemanticProjection(V4ContractModel):
    preference_directions: list[PreferenceDirectionState] = Field(default_factory=list)
    requirements: list[DisplayText] = Field(default_factory=list)
    allergies: list[DisplayText] = Field(default_factory=list)
    avoidances: list[DisplayText] = Field(default_factory=list)
    concrete_restaurant_intents: list[ConcreteIntentState] = Field(default_factory=list)
    exclusions: list[ConcreteIntentState] = Field(default_factory=list)
    delegation: DelegatedScope | None = None


class LodgingSemanticProjection(V4ContractModel):
    area_preferences: list[PreferenceDirectionState] = Field(default_factory=list)
    hotel_quality_tiers: list[HotelQualityTier] = Field(default_factory=list, max_length=4)
    hotel_quality_tier: str | None = Field(
        default=None,
        pattern=r"^(economy|comfort|upscale|luxury)$",
    )
    property_type_preferences: list[DisplayText] = Field(default_factory=list)
    nightly_budget: MoneyRange | None = None
    facility_requirements: list[DisplayText] = Field(default_factory=list)
    class_preference_source_operation_refs: list[Identifier] = Field(default_factory=list)
    user_named_hotel_intents: list[ConcreteIntentState] = Field(default_factory=list)
    existing_bookings: list[BookingReference] = Field(default_factory=list)
    not_applicable: bool = False
    delegation: DelegatedScope | None = None

    @model_validator(mode="after")
    def class_evidence_refs_are_unique(self) -> LodgingSemanticProjection:
        require_unique(
            self.class_preference_source_operation_refs,
            "lodging class source operation refs",
        )
        return self


class TransportAndPaceProjection(V4ContractModel):
    transport_preferences: list[DisplayText] = Field(default_factory=list)
    pace_preferences: list[DisplayText] = Field(default_factory=list)


class ConfirmedTaskBookRef(V4ContractModel):
    task_book_id: Identifier
    task_book_version: int = Field(ge=1, strict=True)
    based_on_state_version: int = Field(ge=0, strict=True)


class TripSemanticState(V4ContractModel):
    """The only V4 source of truth for this trip's user intent."""

    trip_id: Identifier
    state_version: int = Field(default=0, ge=0, strict=True)
    cold_start_profile_snapshot: ColdStartProfileSnapshot | None = None
    long_term_memory_snapshot: tuple[UserMemoryView, ...] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    trip_basics: TripBasicsProjection = Field(default_factory=TripBasicsProjection)
    attractions: AttractionSemanticProjection = Field(default_factory=AttractionSemanticProjection)
    dining: DiningSemanticProjection = Field(default_factory=DiningSemanticProjection)
    lodging: LodgingSemanticProjection = Field(default_factory=LodgingSemanticProjection)
    transport_and_pace: TransportAndPaceProjection = Field(
        default_factory=TransportAndPaceProjection
    )
    constraints: list[DisplayText] = Field(default_factory=list)
    existing_bookings: list[BookingReference] = Field(default_factory=list)
    delegations: list[DelegatedScope] = Field(default_factory=list)
    entries: list[SemanticStateEntry] = Field(default_factory=list)
    superseded_entries: list[SemanticStateEntry] = Field(default_factory=list)
    unresolved_conflicts: list[SemanticMergeConflict] = Field(default_factory=list)
    invalidations: list[SemanticInvalidation] = Field(default_factory=list)
    audit_log: list[SemanticMergeAudit] = Field(default_factory=list)
    confirmed_task_book_ref: ConfirmedTaskBookRef | None = None

    @model_validator(mode="after")
    def semantic_core_is_owned_and_versioned(self) -> TripSemanticState:
        require_unique((entry.state_key for entry in self.entries), "active semantic state_key")
        require_unique((audit.operation_id for audit in self.audit_log), "operation audit ID")
        require_unique(
            (conflict.conflict_id for conflict in self.unresolved_conflicts),
            "semantic conflict ID",
        )
        trip_id = str(self.trip_id)
        for entry in (*self.entries, *self.superseded_entries):
            if str(entry.operation.trip_id) != trip_id:
                raise ValueError("semantic entry belongs to another trip")
            if entry.applied_state_version > self.state_version:
                raise ValueError("semantic entry cannot be newer than its state")
        for conflict in self.unresolved_conflicts:
            if str(conflict.trip_id) != trip_id:
                raise ValueError("semantic conflict belongs to another trip")
            if conflict.detected_state_version > self.state_version:
                raise ValueError("semantic conflict cannot be newer than its state")
        for audit in self.audit_log:
            if audit.state_version > self.state_version:
                raise ValueError("semantic audit cannot be newer than its state")
        for invalidation in self.invalidations:
            if invalidation.state_version > self.state_version:
                raise ValueError("semantic invalidation cannot be newer than its state")
        if (
            self.confirmed_task_book_ref is not None
            and self.confirmed_task_book_ref.based_on_state_version > self.state_version
        ):
            raise ValueError("confirmed task book cannot be newer than semantic state")
        return self


class SectionCoverage(V4ContractModel):
    status: CoverageStatus = CoverageStatus.NOT_STARTED
    required_targets: list[Identifier] = Field(default_factory=list)
    covered_targets: list[Identifier] = Field(default_factory=list)
    completion_mode: CompletionMode | None = None
    evidence_refs: list[Identifier] = Field(default_factory=list)
    blocking_conflict_ids: list[Identifier] = Field(default_factory=list)
    completed_at_state_version: int | None = Field(default=None, ge=0, strict=True)

    @model_validator(mode="after")
    def completion_is_evidence_backed(self) -> SectionCoverage:
        for field_name, values in (
            ("required_targets", self.required_targets),
            ("covered_targets", self.covered_targets),
            ("evidence_refs", self.evidence_refs),
            ("blocking_conflict_ids", self.blocking_conflict_ids),
        ):
            require_unique(values, field_name)
        complete = self.status in {CoverageStatus.COMPLETE, CoverageStatus.NOT_APPLICABLE}
        if complete and (self.completion_mode is None or not self.evidence_refs):
            raise ValueError("complete coverage requires a completion mode and evidence")
        if complete and self.blocking_conflict_ids:
            raise ValueError("complete coverage cannot retain blocking conflicts")
        if complete and self.completed_at_state_version is None:
            raise ValueError("complete coverage requires completed_at_state_version")
        if not complete and self.completion_mode is not None:
            raise ValueError("incomplete coverage cannot declare a completion mode")
        if self.status is CoverageStatus.NOT_APPLICABLE:
            if self.completion_mode is not CompletionMode.NOT_APPLICABLE:
                raise ValueError("not-applicable coverage requires not_applicable completion mode")
        elif self.completion_mode is CompletionMode.NOT_APPLICABLE:
            raise ValueError("not_applicable completion mode requires not_applicable status")
        return self


_CARD_RECOVERY_SECTIONS = (
    DiscoverySection.ATTRACTION_PREFERENCE,
    DiscoverySection.ATTRACTION_SPECIFIC,
    DiscoverySection.DINING_PREFERENCE,
    DiscoverySection.DINING_SPECIFIC,
    DiscoverySection.LODGING_AREA_PREFERENCE,
    DiscoverySection.LODGING_CLASS_PREFERENCE,
)


class CardGenerationRecovery(V4ContractModel):
    """A durable retry affordance, never a selectable card or semantic answer."""

    action: Literal["retry_card"] = "retry_card"
    failure_code: Identifier


class PendingInteraction(V4ContractModel):
    model_config = ConfigDict(
        json_schema_extra={
            "allOf": [
                {
                    "if": {
                        "properties": {"recovery": {"type": "object"}},
                        "required": ["recovery"],
                    },
                    "then": {
                        "properties": {
                            "kind": {"const": "free_text_question"},
                            "section": {
                                "enum": [section.value for section in _CARD_RECOVERY_SECTIONS]
                            },
                            "option_refs": {"type": "array", "maxItems": 0},
                        }
                    },
                }
            ],
        }
    )
    interaction_id: Identifier
    kind: PendingInteractionKind
    section: DiscoverySection
    target_ids: list[Identifier] = Field(min_length=1)
    option_refs: list[Identifier] = Field(default_factory=list)
    based_on_state_version: int = Field(ge=0, strict=True)
    dependency_fingerprint: Identifier
    status: InteractionStatus = InteractionStatus.ACTIVE
    recovery: CardGenerationRecovery | None = None

    @model_validator(mode="after")
    def refs_are_unique_and_kind_is_consistent(self) -> PendingInteraction:
        require_unique(self.target_ids, "target_ids")
        require_unique(self.option_refs, "option_refs")
        if (
            self.kind
            in {
                PendingInteractionKind.PREFERENCE_CARD,
                PendingInteractionKind.SPECIFIC_CARD,
            }
            and not self.option_refs
        ):
            raise ValueError("card interaction requires option_refs")
        if self.recovery is not None and (
            self.kind is not PendingInteractionKind.FREE_TEXT_QUESTION
            or self.option_refs
            or self.section not in _CARD_RECOVERY_SECTIONS
        ):
            raise ValueError("card recovery requires a free-text interaction in a card section")
        return self


class TaskBookCandidate(V4ContractModel):
    task_book_id: Identifier
    based_on_state_version: int = Field(ge=0, strict=True)
    status: TaskBookStatus
    created_at: AwareDatetime
    unresolved_items: list[Identifier] = Field(default_factory=list)
    value: TaskBookV4

    @model_validator(mode="after")
    def value_matches_candidate(self) -> TaskBookCandidate:
        if self.value.task_book_id != self.task_book_id:
            raise ValueError("task book candidate ID does not match its value")
        if self.value.based_on_state_version != self.based_on_state_version:
            raise ValueError("task book candidate state version does not match its value")
        if self.value.status is not self.status:
            raise ValueError("task book candidate status does not match its value")
        return self


class DecisionAuditPointer(V4ContractModel):
    turn_id: Identifier
    decision_id: Identifier
    action: Identifier
    based_on_state_version: int = Field(ge=0, strict=True)
    status: str = Field(pattern=r"^(proposed|executed|rejected|superseded)$")


def initial_section_coverage() -> dict[DiscoverySection, SectionCoverage]:
    return {section: SectionCoverage() for section in _COVERAGE_SECTIONS}


class DiscoveryRuntimeState(V4ContractModel):
    trip_id: Identifier
    state_version: int = Field(default=0, ge=0, strict=True)
    current_section: DiscoverySection = DiscoverySection.OTHER
    section_coverage: dict[DiscoverySection, SectionCoverage] = Field(
        default_factory=initial_section_coverage
    )
    pending_interaction: PendingInteraction | None = None
    task_book_candidate: TaskBookCandidate | None = None
    last_decision: DecisionAuditPointer | None = None

    @model_validator(mode="after")
    def runtime_is_current_and_complete(self) -> DiscoveryRuntimeState:
        if set(self.section_coverage) != _COVERAGE_SECTIONS:
            raise ValueError(
                "section_coverage must contain every pre-task-book section exactly once"
            )
        for coverage in self.section_coverage.values():
            if (
                coverage.completed_at_state_version is not None
                and coverage.completed_at_state_version > self.state_version
            ):
                raise ValueError("coverage cannot be newer than runtime state")
        if (
            self.pending_interaction is not None
            and self.pending_interaction.based_on_state_version > self.state_version
        ):
            raise ValueError("pending interaction cannot be newer than runtime state")
        if (
            self.task_book_candidate is not None
            and self.task_book_candidate.based_on_state_version != self.state_version
            and self.task_book_candidate.status
            not in {TaskBookStatus.SUPERSEDED, TaskBookStatus.CONFIRMED}
        ):
            raise ValueError("active task book candidate must bind the current state version")
        if (
            self.last_decision is not None
            and self.last_decision.based_on_state_version > self.state_version
        ):
            raise ValueError("last decision cannot be newer than runtime state")
        return self


class V4TripStateEnvelope(V4ContractModel):
    protocol_version: str = Field(default="v4", pattern=r"^v4$")
    schema_version: str = Field(default="4.0.0", pattern=r"^4\.0\.0$")
    semantic_state: TripSemanticState
    discovery_runtime_state: DiscoveryRuntimeState
    current_plan_version_id: UUID | None = None
    published_plan: PlannerPublishedPlan | None = None

    @model_validator(mode="after")
    def states_share_identity_and_version(self) -> V4TripStateEnvelope:
        if self.semantic_state.trip_id != self.discovery_runtime_state.trip_id:
            raise ValueError("V4 states must belong to the same trip")
        if self.semantic_state.state_version != self.discovery_runtime_state.state_version:
            raise ValueError("V4 states must share one committed state_version")
        if (self.current_plan_version_id is None) != (self.published_plan is None):
            raise ValueError("V4 visible plan ID and published plan must appear together")
        if self.published_plan is not None:
            plan = self.published_plan
            confirmed = self.semantic_state.confirmed_task_book_ref
            if (
                plan.plan_version_id != self.current_plan_version_id
                or str(plan.trip_id) != self.semantic_state.trip_id
                or plan.based_on_state_version >= self.semantic_state.state_version
                or confirmed is None
                or plan.based_on_task_book_id != confirmed.task_book_id
                or plan.based_on_task_book_version != confirmed.task_book_version
            ):
                raise ValueError("V4 published plan must bind the current trip and task book")
        return self


V4_STATE_CONTRACTS = (
    ColdStartProfileSnapshot,
    TripSemanticState,
    SectionCoverage,
    PendingInteraction,
    DiscoveryRuntimeState,
    V4TripStateEnvelope,
)
