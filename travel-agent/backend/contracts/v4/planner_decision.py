"""Strict six-action decision union for the V4 Planner."""

from __future__ import annotations

from datetime import date
from typing import Annotated, Literal, TypeAlias

from pydantic import Field, model_validator

from backend.contracts.v4.base import (
    Digest,
    DisplayText,
    Identifier,
    V4ContractModel,
    require_unique,
)
from backend.contracts.v4.enums import InteractionStatus
from backend.contracts.v4.plan_change import (
    FormalHotelRecommendationSet,
    SelectedHotelRecommendation,
)
from backend.contracts.v4.planner_draft import WorkingItineraryDraft
from backend.contracts.v4.planner_observations import PlannerCapabilityRequest, PlannerInteraction
from backend.contracts.v4.planner_patch import (
    ItineraryPatch,
    PlanChangeRequestPatchAuthority,
    ValidationIssuePatchAuthority,
)
from backend.contracts.v4.planner_refs import PlannerScope, require_same_scope_ownership
from backend.contracts.v4.planner_strategy import PlanningStrategy

PlannerAction = Literal[
    "build_or_update_strategy",
    "request_evidence",
    "materialize_draft",
    "revise_draft",
    "ask_user",
    "propose_finalize",
]


class PlannerInputRefs(V4ContractModel):
    strategy_revision: int | None = Field(default=None, ge=1, strict=True)
    candidate_pool_revision: int = Field(ge=1, strict=True)
    draft_revision: int | None = Field(default=None, ge=1, strict=True)
    validation_observation_id: Identifier | None = None
    readiness_observation_id: Identifier | None = None
    plan_change_request_id: Identifier | None = None
    planner_interaction_answer_id: Identifier | None = None


class PlannerCompletionAssessment(V4ContractModel):
    ready_to_finalize: bool
    blocking_issue_ids: tuple[Identifier, ...] = ()

    @model_validator(mode="after")
    def blocker_ids_are_unique(self) -> PlannerCompletionAssessment:
        require_unique(self.blocking_issue_ids, "completion blocker IDs")
        return self


class BuildOrUpdateStrategyPayload(V4ContractModel):
    action: Literal["build_or_update_strategy"] = "build_or_update_strategy"
    mode: Literal["initialize", "replace"]
    proposed_strategy: PlanningStrategy
    base_strategy_revision: int | None = Field(
        default=None,
        ge=1,
        strict=True,
        exclude_if=lambda value: value is None,
    )

    @model_validator(mode="after")
    def mode_controls_base_revision(self) -> BuildOrUpdateStrategyPayload:
        if self.mode == "initialize":
            if "base_strategy_revision" in self.model_fields_set:
                raise ValueError("initialize payload forbids base_strategy_revision")
        elif self.base_strategy_revision is None:
            raise ValueError("replace payload requires base_strategy_revision")
        return self


class RequestEvidencePayload(V4ContractModel):
    action: Literal["request_evidence"] = "request_evidence"
    capability_requests: tuple[PlannerCapabilityRequest, ...] = Field(min_length=1, max_length=4)
    resume_goal: DisplayText

    @model_validator(mode="after")
    def request_ids_are_unique(self) -> RequestEvidencePayload:
        require_unique(
            [item.request_id for item in self.capability_requests],
            "Planner capability request IDs",
        )
        return self


class MaterializeDraftPayload(V4ContractModel):
    action: Literal["materialize_draft"] = "materialize_draft"
    proposed_working_draft: WorkingItineraryDraft
    selected_hotel: SelectedHotelRecommendation | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    hotel_recommendations: FormalHotelRecommendationSet | None = None
    declared_affected_dates: tuple[date, ...] = Field(min_length=1, max_length=5)

    @model_validator(mode="after")
    def affected_dates_match_complete_draft(self) -> MaterializeDraftPayload:
        expected = tuple(day.service_date for day in self.proposed_working_draft.days)
        if tuple(self.declared_affected_dates) != expected:
            raise ValueError("initial materialization must declare every working-draft date")
        hotel_selection_count = sum(
            value is not None for value in (self.selected_hotel, self.hotel_recommendations)
        )
        selected_mode = self.proposed_working_draft.lodging_baseline.mode == "selected_offer"
        if selected_mode != (hotel_selection_count == 1):
            raise ValueError("selected lodging requires exactly one formal hotel selection")
        selected_ref = (
            self.selected_hotel.hotel_offer_ref
            if self.selected_hotel is not None
            else self.hotel_recommendations.recommended_hotel.hotel_offer_ref
            if self.hotel_recommendations is not None
            else None
        )
        if selected_ref != self.proposed_working_draft.lodging_baseline.selected_offer_ref:
            raise ValueError("the formal hotel selection must be the scheduling baseline")
        return self


class ReviseDraftPayload(V4ContractModel):
    action: Literal["revise_draft"] = "revise_draft"
    itinerary_patch: ItineraryPatch
    selected_hotel: SelectedHotelRecommendation | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    hotel_recommendations: FormalHotelRecommendationSet | None = None
    declared_affected_dates: tuple[date, ...] = Field(min_length=1, max_length=5)

    @model_validator(mode="after")
    def affected_dates_match_patch(self) -> ReviseDraftPayload:
        if tuple(self.declared_affected_dates) != self.itinerary_patch.declared_affected_dates:
            raise ValueError("revision affected dates must exactly match the ItineraryPatch")
        hotel_operations = tuple(
            operation
            for operation in self.itinerary_patch.operations
            if operation.operation == "set_hotel_baseline"
        )
        selection_count = sum(
            value is not None for value in (self.selected_hotel, self.hotel_recommendations)
        )
        if bool(hotel_operations) != (selection_count == 1):
            raise ValueError("hotel baseline revision requires one refreshed selection")
        if selection_count:
            operation = hotel_operations[0]
            selected_ref = (
                self.selected_hotel.hotel_offer_ref
                if self.selected_hotel is not None
                else self.hotel_recommendations.recommended_hotel.hotel_offer_ref  # type: ignore[union-attr]
            )
            if operation.new_baseline.selected_offer_ref != selected_ref:
                raise ValueError("revised hotel baseline must equal the formal selection")
        return self


class PlannerResumeContract(V4ContractModel):
    resume_token: Identifier
    resume_goal: DisplayText
    expected_next_actions: tuple[
        Literal[
            "build_or_update_strategy",
            "request_evidence",
            "materialize_draft",
            "revise_draft",
            "propose_finalize",
        ],
        ...,
    ] = Field(min_length=1)

    @model_validator(mode="after")
    def next_actions_are_unique(self) -> PlannerResumeContract:
        require_unique(self.expected_next_actions, "Planner resume actions")
        return self


class AskUserPayload(V4ContractModel):
    action: Literal["ask_user"] = "ask_user"
    user_decision_request: PlannerInteraction
    blocking_issue_ids: tuple[Identifier, ...] = Field(min_length=1)
    resume_contract: PlannerResumeContract

    @model_validator(mode="after")
    def request_is_current_actionable_and_resumable(self) -> AskUserPayload:
        require_unique(self.blocking_issue_ids, "ask-user blocking issue IDs")
        if set(self.blocking_issue_ids) != set(self.user_decision_request.issue_ids):
            raise ValueError("ask-user blockers must equal the interaction issue IDs")
        if self.user_decision_request.status is not InteractionStatus.ACTIVE:
            raise ValueError("ask_user requires an active PlannerInteraction")
        if self.resume_contract.resume_token != self.user_decision_request.resume_token:
            raise ValueError("resume contract token must match the PlannerInteraction")
        return self


class FinalizeArtifactRefs(V4ContractModel):
    draft_id: Identifier
    draft_revision: int = Field(ge=1, strict=True)
    draft_content_digest: Digest
    materialized_schedule_id: Identifier
    materialized_schedule_revision: int = Field(ge=1, strict=True)
    cost_draft_id: Identifier
    cost_draft_revision: int = Field(ge=1, strict=True)
    validation_observation_id: Identifier
    validation_fingerprint: Digest


class ProposeFinalizePayload(V4ContractModel):
    action: Literal["propose_finalize"] = "propose_finalize"
    final_refs: FinalizeArtifactRefs


PlannerActionPayload: TypeAlias = Annotated[
    BuildOrUpdateStrategyPayload
    | RequestEvidencePayload
    | MaterializeDraftPayload
    | ReviseDraftPayload
    | AskUserPayload
    | ProposeFinalizePayload,
    Field(discriminator="action"),
]


class PlannerDecisionProposal(V4ContractModel):
    """Model-produced decision before the service adds authoritative decision metadata."""

    scope: PlannerScope
    action: PlannerAction
    current_goal: DisplayText
    reason_summary: DisplayText
    input_refs: PlannerInputRefs
    completion_assessment: PlannerCompletionAssessment
    payload: PlannerActionPayload

    @model_validator(mode="after")
    def action_contract_is_strict(self) -> PlannerDecisionProposal:
        _validate_action_contract(self)
        return self


class PlannerDecision(PlannerDecisionProposal):
    """Validated and persisted decision with a server-owned ID."""

    decision_id: Identifier


def _validate_action_contract(decision: PlannerDecisionProposal) -> None:
    if decision.payload.action != decision.action:
        raise ValueError("PlannerDecision action must match its strict payload variant")
    is_finalize = decision.action == "propose_finalize"
    if decision.completion_assessment.ready_to_finalize != is_finalize:
        raise ValueError("only propose_finalize may set ready_to_finalize=true")
    if is_finalize and decision.completion_assessment.blocking_issue_ids:
        raise ValueError("propose_finalize cannot contain blocking issue IDs")

    refs = decision.input_refs
    payload = decision.payload
    if refs.readiness_observation_id is not None and not isinstance(
        payload, (AskUserPayload, RequestEvidencePayload)
    ):
        raise ValueError("readiness evidence is only valid for ask_user or request_evidence")
    if isinstance(payload, BuildOrUpdateStrategyPayload):
        require_same_scope_ownership(decision.scope, payload.proposed_strategy.scope)
        if payload.proposed_strategy.candidate_pool_revision != refs.candidate_pool_revision:
            raise ValueError("strategy payload and decision must use the same candidate pool")
        if any(
            value is not None
            for value in (
                refs.validation_observation_id,
                refs.plan_change_request_id,
                refs.planner_interaction_answer_id,
            )
        ):
            raise ValueError("strategy decisions forbid validation/change/interaction references")
        if payload.mode == "initialize":
            if refs.strategy_revision is not None or refs.draft_revision is not None:
                raise ValueError("strategy initialization requires an empty strategy workspace")
        elif refs.strategy_revision != payload.base_strategy_revision:
            raise ValueError("strategy replacement must use the current base revision")
    elif isinstance(payload, RequestEvidencePayload):
        for request in payload.capability_requests:
            require_same_scope_ownership(decision.scope, request.scope)
    elif isinstance(payload, MaterializeDraftPayload):
        require_same_scope_ownership(decision.scope, payload.proposed_working_draft.scope)
        if refs.strategy_revision is None or refs.draft_revision is not None:
            raise ValueError("first materialization requires strategy and no existing draft")
        if any(
            value is not None
            for value in (
                refs.validation_observation_id,
                refs.plan_change_request_id,
                refs.planner_interaction_answer_id,
            )
        ):
            raise ValueError("materialize_draft forbids validation/change/interaction references")
        if payload.proposed_working_draft.based_on_strategy_revision != refs.strategy_revision:
            raise ValueError("working draft must use the current strategy revision")
        if payload.proposed_working_draft.candidate_pool_revision != refs.candidate_pool_revision:
            raise ValueError("working draft must use the current candidate pool revision")
    elif isinstance(payload, ReviseDraftPayload):
        if refs.strategy_revision is None or refs.draft_revision is None:
            raise ValueError("revise_draft requires current strategy and draft revisions")
        require_same_scope_ownership(decision.scope, payload.itinerary_patch.scope)
        if payload.itinerary_patch.base_draft_revision != refs.draft_revision:
            raise ValueError("ItineraryPatch must use the current draft revision")
        authority_refs = (
            refs.validation_observation_id,
            refs.plan_change_request_id,
            refs.planner_interaction_answer_id,
        )
        if sum(value is not None for value in authority_refs) != 1:
            raise ValueError("revision requires exactly one current authority input reference")
        authority = payload.itinerary_patch.authority
        if isinstance(authority, ValidationIssuePatchAuthority):
            if refs.validation_observation_id != authority.reference_id:
                raise ValueError("validation Patch authority must match input_refs")
        elif isinstance(authority, PlanChangeRequestPatchAuthority):
            if refs.plan_change_request_id != authority.reference_id:
                raise ValueError("PlanChange Patch authority must match input_refs")
        elif refs.planner_interaction_answer_id != authority.user_message_or_answer_ref:
            raise ValueError("interaction Patch answer must match input_refs")
    elif isinstance(payload, AskUserPayload):
        require_same_scope_ownership(decision.scope, payload.user_decision_request.scope)
        if not any(
            value is not None
            for value in (
                refs.readiness_observation_id,
                refs.validation_observation_id,
                refs.plan_change_request_id,
                refs.planner_interaction_answer_id,
            )
        ):
            raise ValueError("ask_user requires at least one current blocking source")
        if set(payload.blocking_issue_ids) != set(
            decision.completion_assessment.blocking_issue_ids
        ):
            raise ValueError("ask_user payload and completion blockers must match")
    else:
        if (
            refs.strategy_revision is None
            or refs.draft_revision is None
            or refs.validation_observation_id is None
            or refs.planner_interaction_answer_id is not None
        ):
            raise ValueError(
                "propose_finalize requires strategy, draft and passed validation, "
                "and no active answer"
            )
        final_refs = payload.final_refs
        if final_refs.draft_revision != refs.draft_revision:
            raise ValueError("final draft revision must match input_refs")
        if final_refs.validation_observation_id != refs.validation_observation_id:
            raise ValueError("final validation reference must match input_refs")


V4_PLANNER_DECISION_CONTRACTS: tuple[type[V4ContractModel], ...] = (
    PlannerDecisionProposal,
    PlannerDecision,
)
