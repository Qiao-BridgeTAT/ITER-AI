"""Prepare Agent decisions, bounded tools, and candidate-turn results."""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import AwareDatetime, Field, RootModel, model_validator

from backend.contracts.v4.base import (
    Digest,
    DisplayText,
    Identifier,
    V4ContractModel,
    require_unique,
)
from backend.contracts.v4.enums import (
    DiscoverySection,
    PrepareActionKind,
    PrepareDomain,
    SectionProposalKind,
    ToolObservationStatus,
    ToolRequestPurpose,
)
from backend.contracts.v4.semantic_operations import SemanticOperationProposal
from backend.contracts.v4.state import DiscoveryRuntimeState, TripSemanticState


class PrepareToolCapability(StrEnum):
    RESOLVE_PLACE = "resolve_place"
    PLACE_FACTS = "place_facts"
    OPENING_HOURS = "opening_hours"
    TICKET_AVAILABILITY = "ticket_availability"
    WEATHER_FORECAST = "weather_forecast"
    SPATIAL_ROUTES = "spatial_routes"
    HOTEL_BOOKING_FACTS = "hotel_booking_facts"
    PLACE_PRODUCTS = "place_products"


class ToolRequestBase(V4ContractModel):
    request_id: Identifier
    depends_on: list[Identifier] = Field(default_factory=list)
    purpose: ToolRequestPurpose
    required: bool

    @model_validator(mode="after")
    def dependencies_are_valid(self) -> ToolRequestBase:
        require_unique(self.depends_on, "depends_on")
        if self.request_id in self.depends_on:
            raise ValueError("tool request cannot depend on itself")
        return self


class ResolvePlaceRequest(ToolRequestBase):
    capability: Literal[PrepareToolCapability.RESOLVE_PLACE]
    query: DisplayText
    city: DisplayText


class PlaceFactsRequest(ToolRequestBase):
    capability: Literal[PrepareToolCapability.PLACE_FACTS]
    canonical_entity_ids: list[Identifier] = Field(min_length=1, max_length=20)
    fact_kinds: list[Identifier] = Field(min_length=1)


class OpeningHoursRequest(ToolRequestBase):
    capability: Literal[PrepareToolCapability.OPENING_HOURS]
    canonical_entity_ids: list[Identifier] = Field(min_length=1, max_length=20)
    service_dates: list[date] = Field(min_length=1, max_length=5)


class TicketAvailabilityRequest(ToolRequestBase):
    capability: Literal[PrepareToolCapability.TICKET_AVAILABILITY]
    canonical_entity_ids: list[Identifier] = Field(min_length=1, max_length=20)
    service_dates: list[date] = Field(min_length=1, max_length=5)
    traveler_count_ref: Identifier


class WeatherForecastRequest(ToolRequestBase):
    capability: Literal[PrepareToolCapability.WEATHER_FORECAST]
    destination_ref: Identifier
    service_dates: list[date] = Field(min_length=1, max_length=5)


class SpatialRoutesRequest(ToolRequestBase):
    capability: Literal[PrepareToolCapability.SPATIAL_ROUTES]
    origin_ref: Identifier
    destination_ref: Identifier
    transport_modes: list[Literal["public_transit", "taxi", "walking", "driving"]] = Field(
        min_length=1,
        max_length=4,
    )
    departure_at_ref: Identifier | None = None


class HotelBookingFactsRequest(ToolRequestBase):
    capability: Literal[PrepareToolCapability.HOTEL_BOOKING_FACTS]
    user_booking_ref: Identifier
    city: DisplayText


class PlaceProductsRequest(ToolRequestBase):
    capability: Literal[PrepareToolCapability.PLACE_PRODUCTS]
    canonical_entity_ids: list[Identifier] = Field(min_length=1, max_length=20)
    service_dates: list[date] = Field(default_factory=list, max_length=5)


ToolRequestValue = Annotated[
    ResolvePlaceRequest
    | PlaceFactsRequest
    | OpeningHoursRequest
    | TicketAvailabilityRequest
    | WeatherForecastRequest
    | SpatialRoutesRequest
    | HotelBookingFactsRequest
    | PlaceProductsRequest,
    Field(discriminator="capability"),
]


class ToolRequest(RootModel[ToolRequestValue]):
    """One typed, whitelisted Prepare tool request."""


class ToolFact(V4ContractModel):
    fact_kind: Identifier
    subject_ref: Identifier
    value_summary: DisplayText
    source_reference_ids: list[Identifier] = Field(min_length=1)
    observed_at: AwareDatetime
    expires_at: AwareDatetime | None = None


class ToolObservation(V4ContractModel):
    request_id: Identifier
    capability: PrepareToolCapability
    status: ToolObservationStatus
    facts: list[ToolFact] = Field(default_factory=list)
    entity_refs: list[Identifier] = Field(default_factory=list)
    source_refs: list[Identifier] = Field(default_factory=list)
    observed_at: AwareDatetime
    safe_summary: DisplayText
    limitations: list[DisplayText] = Field(default_factory=list)

    @model_validator(mode="after")
    def status_matches_payload(self) -> ToolObservation:
        if self.status is ToolObservationStatus.SUCCESS and not self.facts:
            raise ValueError("successful tool observation requires at least one fact")
        if self.status is ToolObservationStatus.PARTIAL and (
            not self.facts or not self.limitations
        ):
            raise ValueError("partial tool observation requires facts and limitations")
        if (
            self.status
            in {
                ToolObservationStatus.UNAVAILABLE,
                ToolObservationStatus.INVALID_REQUEST,
            }
            and self.facts
        ):
            raise ValueError("unavailable or invalid request cannot claim observed facts")
        if self.facts and not self.source_refs:
            raise ValueError("observed facts require source_refs")
        for field_name, values in (
            ("entity_refs", self.entity_refs),
            ("source_refs", self.source_refs),
        ):
            require_unique(values, field_name)
        return self


class ReplyOnlyAction(V4ContractModel):
    kind: Literal[PrepareActionKind.REPLY_ONLY]
    domain: PrepareDomain | None = None
    requested_targets: list[Identifier] = Field(default_factory=list)


class UseToolAction(V4ContractModel):
    kind: Literal[PrepareActionKind.USE_TOOL]
    domain: PrepareDomain | None = None
    requested_targets: list[Identifier] = Field(min_length=1)


class ShowPreferenceCardAction(V4ContractModel):
    kind: Literal[PrepareActionKind.SHOW_PREFERENCE_CARD]
    domain: Literal[
        PrepareDomain.ATTRACTION,
        PrepareDomain.DINING,
        PrepareDomain.LODGING,
    ]
    requested_targets: list[Identifier] = Field(min_length=1)


class ShowSpecificCardAction(V4ContractModel):
    kind: Literal[PrepareActionKind.SHOW_SPECIFIC_CARD]
    domain: Literal[PrepareDomain.ATTRACTION, PrepareDomain.DINING]
    requested_targets: list[Identifier] = Field(min_length=1)


class AskClarificationAction(V4ContractModel):
    kind: Literal[PrepareActionKind.ASK_CLARIFICATION]
    domain: PrepareDomain | None = None
    requested_targets: list[Identifier] = Field(min_length=1)


class FinalSupplementAction(V4ContractModel):
    kind: Literal[PrepareActionKind.FINAL_SUPPLEMENT]
    domain: Literal[PrepareDomain.GENERAL] = PrepareDomain.GENERAL
    requested_targets: list[Identifier] = Field(min_length=1)


class GenerateTaskBookAction(V4ContractModel):
    kind: Literal[PrepareActionKind.GENERATE_TASK_BOOK]
    domain: Literal[PrepareDomain.GENERAL] = PrepareDomain.GENERAL
    requested_targets: list[Identifier] = Field(default_factory=list)


PrepareNextAction = Annotated[
    ReplyOnlyAction
    | UseToolAction
    | ShowPreferenceCardAction
    | ShowSpecificCardAction
    | AskClarificationAction
    | FinalSupplementAction
    | GenerateTaskBookAction,
    Field(discriminator="kind"),
]


class SectionProposal(V4ContractModel):
    kind: SectionProposalKind
    section: DiscoverySection
    reopen_sections: list[DiscoverySection] = Field(default_factory=list)
    evidence_refs: list[Identifier] = Field(default_factory=list)

    @model_validator(mode="after")
    def reopen_payload_matches_kind(self) -> SectionProposal:
        require_unique(self.reopen_sections, "reopen_sections")
        require_unique(self.evidence_refs, "evidence_refs")
        if self.kind is SectionProposalKind.REOPEN and not self.reopen_sections:
            raise ValueError("reopen proposal requires reopen_sections")
        if self.kind is not SectionProposalKind.REOPEN and self.reopen_sections:
            raise ValueError("only reopen proposal may contain reopen_sections")
        if self.kind is not SectionProposalKind.STAY and not self.evidence_refs:
            raise ValueError("section transition proposal requires evidence")
        return self


class ReplyGoal(V4ContractModel):
    acknowledge: list[DisplayText] = Field(default_factory=list)
    answer_questions: list[Identifier] = Field(default_factory=list)
    explain_next_step: DisplayText | None = None
    fact_requirements: list[Identifier] = Field(default_factory=list)


class ClarificationContract(V4ContractModel):
    target: Identifier
    why_blocking: DisplayText


PublishedPlanIntent = Literal["reply_only", "local_replan", "full_replan", "task_book_change"]


class PrepareDecision(V4ContractModel):
    decision_id: Identifier
    based_on_state_version: int = Field(ge=0, strict=True)
    semantic_operations: list[SemanticOperationProposal] = Field(default_factory=list)
    next_action: PrepareNextAction
    tool_requests: list[ToolRequest] = Field(default_factory=list, max_length=4)
    section_proposal: SectionProposal
    reply_goal: ReplyGoal
    clarification: ClarificationContract | None = None
    published_plan_intent: PublishedPlanIntent | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def action_specific_fields_are_strict(self) -> PrepareDecision:
        is_tool = self.next_action.kind is PrepareActionKind.USE_TOOL
        if is_tool != bool(self.tool_requests):
            raise ValueError("use_tool and tool_requests must appear together")
        is_clarification = self.next_action.kind is PrepareActionKind.ASK_CLARIFICATION
        if is_clarification != (self.clarification is not None):
            raise ValueError("ask_clarification and clarification contract must appear together")
        request_ids = [request.root.request_id for request in self.tool_requests]
        require_unique(request_ids, "tool request_id")
        known_ids = set(request_ids)
        dependencies: dict[str, tuple[str, ...]] = {}
        for request in self.tool_requests:
            if not set(request.root.depends_on).issubset(known_ids):
                raise ValueError("tool dependency must reference another request in this decision")
            dependencies[str(request.root.request_id)] = tuple(
                str(item) for item in request.root.depends_on
            )
        _validate_tool_dependency_graph(dependencies)
        return self


def _validate_tool_dependency_graph(dependencies: dict[str, tuple[str, ...]]) -> None:
    """Require an acyclic graph with at most two request layers."""

    visiting: set[str] = set()
    depths: dict[str, int] = {}

    def depth(request_id: str) -> int:
        if request_id in visiting:
            raise ValueError("tool dependency graph must be acyclic")
        if request_id in depths:
            return depths[request_id]
        visiting.add(request_id)
        request_depth = 1 + max(
            (depth(dependency_id) for dependency_id in dependencies[request_id]),
            default=0,
        )
        visiting.remove(request_id)
        depths[request_id] = request_depth
        return request_depth

    if any(depth(request_id) > 2 for request_id in dependencies):
        raise ValueError("tool dependency graph cannot exceed two layers")


class CandidateAssistantMessage(V4ContractModel):
    message_id: Identifier
    text: DisplayText
    generation_mode: Literal["qwen", "fallback"]
    content_hash: Digest
    attachment_ids: list[Identifier] = Field(default_factory=list)


class CandidateTurnResult(V4ContractModel):
    """Complete uncommitted turn bundle produced outside a DB transaction."""

    turn_id: Identifier
    trip_id: Identifier
    generation_id: Identifier
    base_state_version: int = Field(ge=0, strict=True)
    candidate_state_version: int = Field(ge=1, strict=True)
    semantic_state: TripSemanticState
    discovery_runtime_state: DiscoveryRuntimeState
    accepted_operations: list[SemanticOperationProposal] = Field(default_factory=list)
    tool_observations: list[ToolObservation] = Field(default_factory=list)
    assistant_message: CandidateAssistantMessage
    attachment_payloads: list[Identifier] = Field(default_factory=list)
    decision_audit_ref: Identifier

    @model_validator(mode="after")
    def candidate_bundle_is_one_version(self) -> CandidateTurnResult:
        if self.candidate_state_version != self.base_state_version + 1:
            raise ValueError("candidate state version must increment the base version exactly once")
        if self.semantic_state.trip_id != self.trip_id:
            raise ValueError("candidate semantic state belongs to another trip")
        if self.discovery_runtime_state.trip_id != self.trip_id:
            raise ValueError("candidate runtime state belongs to another trip")
        if self.semantic_state.state_version != self.candidate_state_version:
            raise ValueError("candidate semantic state has the wrong version")
        if self.discovery_runtime_state.state_version != self.candidate_state_version:
            raise ValueError("candidate runtime state has the wrong version")
        return self


V4_PREPARE_CONTRACTS = (
    PrepareDecision,
    ToolRequest,
    ToolObservation,
    CandidateTurnResult,
)
