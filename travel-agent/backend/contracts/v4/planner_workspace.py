"""Checkpoint-safe workspace aggregate for the V4 Planner."""

from __future__ import annotations

from datetime import date

from pydantic import AwareDatetime, BaseModel, Field, model_validator

from backend.contracts.itinerary_draft import CostValidationDraft, ScheduleValidationDraft
from backend.contracts.itinerary_validation import ItineraryValidationResult
from backend.contracts.v4.base import DisplayText, Identifier, V4ContractModel, require_unique
from backend.contracts.v4.enums import InteractionStatus, PlannerStatus
from backend.contracts.v4.plan_change import (
    V4_PLAN_CHANGE_CONTRACTS,
    FormalHotelRecommendationSet,
    PlanChangeRequest,
    SelectedHotelRecommendation,
    validate_hotel_recommendations_against_observation,
    validate_selected_hotel_against_observation,
)
from backend.contracts.v4.planner_decision import (
    V4_PLANNER_DECISION_CONTRACTS,
    PlannerDecision,
    ReviseDraftPayload,
)
from backend.contracts.v4.planner_draft import (
    V4_PLANNER_DRAFT_CONTRACTS,
    UnassignedIntent,
    WorkingItineraryDraft,
    validate_draft_against_pool,
)
from backend.contracts.v4.planner_evidence import (
    V4_PLANNER_EVIDENCE_CONTRACTS,
    PlannerCandidateOrigin,
    PlannerCapabilityObservation,
    PlannerGuardObservation,
    PlannerHotelLocationEvidence,
    PlannerHoursEvidence,
    PlannerInteractionAnswer,
    PlannerPlaceEvidence,
    PlannerRouteComparisonObservation,
    PlannerTicketEvidence,
    PlannerVisitDurationEstimate,
    PlannerWeatherEvidence,
)
from backend.contracts.v4.planner_observations import (
    V4_PLANNER_OBSERVATION_CONTRACTS,
    HotelObservation,
    PlannerInteraction,
    PlannerReadinessObservation,
    PlannerValidationObservation,
    SpatialObservation,
    SpatialRouteEdge,
)
from backend.contracts.v4.planner_patch import V4_PLANNER_PATCH_CONTRACTS
from backend.contracts.v4.planner_publication import V4_PLANNER_PUBLICATION_CONTRACTS
from backend.contracts.v4.planner_refs import (
    V4_PLANNER_REF_CONTRACTS,
    PlannerScope,
    require_same_scope_ownership,
)
from backend.contracts.v4.planner_schedule_repair import PlannerScheduleRepairState
from backend.contracts.v4.planner_strategy import (
    V4_PLANNER_STRATEGY_CONTRACTS,
    CandidatePoolSummary,
    PlanningStrategy,
    validate_strategy_against_pool,
)


class VerifiedFactSummary(V4ContractModel):
    """Safe normalized fact summary; raw Provider payloads are deliberately absent."""

    fact_reference_id: Identifier
    fact_kind: Identifier
    safe_summary: DisplayText
    observed_at: AwareDatetime
    expires_at: AwareDatetime | None = None
    source_reference_ids: tuple[Identifier, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def sources_and_expiry_are_valid(self) -> VerifiedFactSummary:
        require_unique(self.source_reference_ids, "verified fact sources")
        if self.expires_at is not None and self.expires_at <= self.observed_at:
            raise ValueError("verified fact expires_at must be after observed_at")
        return self


class PlannerWorkspaceState(V4ContractModel):
    """Planner-owned exploration state; confirmed task-book identity is immutable."""

    trip_id: Identifier
    generation_id: Identifier
    based_on_task_book_id: Identifier
    based_on_task_book_version: int = Field(ge=1, strict=True)
    workspace_revision: int = Field(ge=0, strict=True)
    planning_strategy: PlanningStrategy | None = None
    candidate_pool: CandidatePoolSummary
    verified_facts: tuple[VerifiedFactSummary, ...] = ()
    candidate_origins: tuple[PlannerCandidateOrigin, ...] = ()
    place_evidence: tuple[PlannerPlaceEvidence, ...] = ()
    hours_evidence: tuple[PlannerHoursEvidence, ...] = ()
    weather_evidence: tuple[PlannerWeatherEvidence, ...] = ()
    ticket_evidence: tuple[PlannerTicketEvidence, ...] = ()
    visit_duration_estimates: tuple[PlannerVisitDurationEstimate, ...] = Field(
        default=(), exclude_if=lambda v: not v
    )
    hotel_location_evidence: tuple[PlannerHotelLocationEvidence, ...] = ()
    route_evidence: tuple[SpatialRouteEdge, ...] = ()
    route_comparisons: tuple[PlannerRouteComparisonObservation, ...] = ()
    capability_observations: tuple[PlannerCapabilityObservation, ...] = ()
    readiness_observation: PlannerReadinessObservation | None = None
    guard_observations: tuple[PlannerGuardObservation, ...] = ()
    interaction_answers: tuple[PlannerInteractionAnswer, ...] = ()
    initial_evidence_ready: bool = False
    timing_optimization_pending: bool = Field(default=False, exclude_if=lambda v: not v)
    schedule_repair_state: PlannerScheduleRepairState | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    best_effort_reasons: tuple[DisplayText, ...] = Field(default=(), exclude_if=lambda v: not v)
    recovery_observations: tuple[PlannerValidationObservation, ...] = Field(
        default=(), exclude_if=lambda v: not v
    )
    recovery_omissions: tuple[UnassignedIntent, ...] = Field(default=(), exclude_if=lambda v: not v)
    action_attempt_count: int = Field(default=0, ge=0, strict=True)
    segment_attempt_count: int = Field(default=0, ge=0, le=12, strict=True)
    segment_evidence_count: int = Field(default=0, ge=0, le=4, strict=True)
    spatial_observation: SpatialObservation | None = None
    hotel_observation: HotelObservation | None = None
    selected_hotel: SelectedHotelRecommendation | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    hotel_recommendations: FormalHotelRecommendationSet | None = None
    working_itinerary: WorkingItineraryDraft | None = None
    materialized_schedule: ScheduleValidationDraft | None = None
    cost_draft: CostValidationDraft | None = None
    validation_report: ItineraryValidationResult | None = None
    validation_observation: PlannerValidationObservation | None = None
    unresolved_decisions: tuple[Identifier, ...] = ()
    decision_trace: tuple[PlannerDecision, ...] = ()
    plan_change_request: PlanChangeRequest | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    revision_round: int = Field(default=0, ge=0, le=2, strict=True)
    user_interrupt_count: int = Field(default=0, ge=0, strict=True)
    active_interaction: PlannerInteraction | None = None
    status: PlannerStatus = PlannerStatus.PLANNING

    @model_validator(mode="after")
    def workspace_artifacts_share_ownership_and_revision(self) -> PlannerWorkspaceState:
        expected_scope = self.current_scope
        if self.plan_change_request is not None and (
            self.plan_change_request.trip_id != self.trip_id
            or self.plan_change_request.requested_scope not in {"local_replan", "full_replan"}
        ):
            raise ValueError("workspace replan authority must belong to this trip")
        _require_workspace_scope(self.candidate_pool.scope, expected_scope)
        if self.candidate_pool.scope.workspace_revision > self.workspace_revision:
            raise ValueError("candidate pool cannot be newer than the workspace")

        scoped_artifacts = (
            self.planning_strategy,
            self.spatial_observation,
            self.hotel_observation,
            self.selected_hotel,
            self.hotel_recommendations,
            self.working_itinerary,
            self.validation_observation,
            self.readiness_observation,
            self.active_interaction,
        )
        for artifact in scoped_artifacts:
            if artifact is not None:
                _require_workspace_scope(artifact.scope, expected_scope)
                if artifact.scope.workspace_revision > self.workspace_revision:
                    raise ValueError("Planner artifact cannot be newer than the workspace")
        for recovery_observation in self.recovery_observations:
            _require_workspace_scope(recovery_observation.scope, expected_scope)
            if recovery_observation.scope.workspace_revision > self.workspace_revision:
                raise ValueError("recovery observation cannot be newer than workspace")

        require_unique(
            [item.fact_reference_id for item in self.verified_facts],
            "verified fact IDs",
        )
        require_unique(self.unresolved_decisions, "unresolved Planner decisions")
        require_unique(
            [item.decision_id for item in self.decision_trace],
            "Planner decision trace IDs",
        )
        for decision in self.decision_trace:
            _require_workspace_scope(decision.scope, expected_scope)
        for observation in self.capability_observations:
            _require_workspace_scope(observation.scope, expected_scope)
        require_unique(
            [item.observation_id for item in self.route_comparisons], "route comparisons"
        )
        facts_by_id = {item.fact_reference_id: item for item in self.verified_facts}
        for comparison in self.route_comparisons:
            _require_workspace_scope(comparison.scope, expected_scope)
            if comparison.scope.workspace_revision > self.workspace_revision:
                raise ValueError("route comparison cannot be newer than workspace")
            if any(
                ref not in facts_by_id
                for edge in comparison.route_edges
                for ref in edge.fact_reference_ids
            ):
                raise ValueError("route comparison must reference owned route facts")
        require_unique(
            [item.canonical_entity_id for item in self.place_evidence], "Planner place evidence"
        )
        require_unique(
            [item.canonical_entity_id for item in self.visit_duration_estimates],
            "Planner duration estimates",
        )
        require_unique(
            [item.canonical_entity_id for item in self.candidate_origins], "candidate origins"
        )
        require_unique(
            [item.canonical_entity_id for item in self.hours_evidence], "Planner hours evidence"
        )
        places_by_id = {item.canonical_entity_id: item for item in self.place_evidence}
        for evidence in self.hours_evidence:
            place = places_by_id.get(evidence.canonical_entity_id)
            if place is None or place.provider_entity_id != evidence.provider_entity_id:
                raise ValueError("Planner hours evidence must bind to the same verified entity")
        require_unique(
            [edge.route_edge_id for edge in self.route_evidence], "Planner route evidence"
        )
        require_unique(
            [item.observation_id for item in self.capability_observations],
            "Planner capability observations",
        )
        if self.readiness_observation is not None and (
            self.readiness_observation.candidate_pool_revision != self.candidate_pool.revision
        ):
            raise ValueError("Planner readiness must use the current candidate pool")
        if self.segment_attempt_count > self.action_attempt_count:
            raise ValueError("segment attempts cannot exceed lifetime attempts")

        if (
            self.planning_strategy is not None
            and self.planning_strategy.candidate_pool_revision != self.candidate_pool.revision
        ):
            raise ValueError("workspace strategy must use the current candidate pool")
        if self.planning_strategy is not None:
            validate_strategy_against_pool(self.planning_strategy, self.candidate_pool)
        if self.spatial_observation is not None:
            pool_by_id = self.candidate_pool.candidate_by_id()
            for cluster in self.spatial_observation.clusters:
                for reference in cluster.candidate_refs:
                    entry = pool_by_id.get(reference.candidate_id)
                    if (
                        entry is None
                        or reference != entry.candidate_ref
                        or cluster.cluster_id not in entry.cluster_ids
                    ):
                        raise ValueError(
                            "Spatial cluster must reference the current exact candidate"
                        )
        if self.spatial_observation is not None and (
            self.candidate_pool.spatial_observation_id is not None
            and self.candidate_pool.spatial_observation_id
            != self.spatial_observation.observation_id
        ):
            raise ValueError("candidate pool must reference the current SpatialObservation")
        if self.working_itinerary is not None:
            validate_draft_against_pool(self.working_itinerary, self.candidate_pool)
            if self.planning_strategy is None or (
                self.working_itinerary.based_on_strategy_revision
                != self.planning_strategy.strategy_revision
            ):
                raise ValueError("working itinerary requires the current planning strategy")
            if self.working_itinerary.candidate_pool_revision != self.candidate_pool.revision:
                raise ValueError("working itinerary must use the current candidate pool")
            if self.spatial_observation is None or (
                self.working_itinerary.spatial_observation_id
                != self.spatial_observation.observation_id
            ):
                raise ValueError("working itinerary requires the current SpatialObservation")
            if self.working_itinerary.hotel_observation_id is not None and (
                self.hotel_observation is None
                or self.working_itinerary.hotel_observation_id
                != self.hotel_observation.hotel_observation_id
            ):
                raise ValueError("working itinerary uses a stale HotelObservation")
            if self.working_itinerary.lodging_baseline.mode == "selected_offer":
                if self.hotel_observation is None:
                    raise ValueError("selected lodging requires the current hotel observation")
                if (self.selected_hotel is None) == (self.hotel_recommendations is None):
                    raise ValueError(
                        "selected lodging requires exactly one current hotel selection"
                    )
                if self.selected_hotel is not None:
                    validate_selected_hotel_against_observation(
                        self.selected_hotel,
                        self.hotel_observation,
                    )
                    selected_ref = self.selected_hotel.hotel_offer_ref
                else:
                    assert self.hotel_recommendations is not None
                    validate_hotel_recommendations_against_observation(
                        self.hotel_recommendations,
                        self.hotel_observation,
                    )
                    selected_ref = self.hotel_recommendations.recommended_hotel.hotel_offer_ref
                if selected_ref != self.working_itinerary.lodging_baseline.selected_offer_ref:
                    raise ValueError("working draft hotel must equal its formal selection")
            elif self.selected_hotel is not None or self.hotel_recommendations is not None:
                raise ValueError("non-selected lodging cannot contain a hotel selection")
        elif any(
            value is not None
            for value in (
                self.materialized_schedule,
                self.cost_draft,
                self.validation_report,
                self.validation_observation,
            )
        ):
            raise ValueError("materialized and validation artifacts require a working itinerary")

        self._validate_v3_materialized_ownership()
        if self.validation_observation is not None and (
            self.working_itinerary is None
            or (
                self.validation_observation.draft_id != self.working_itinerary.draft_id
                or self.validation_observation.draft_revision
                != self.working_itinerary.draft_revision
            )
        ):
            raise ValueError("validation observation must bind to the current working draft")

        if self.revision_round > 0 and self.working_itinerary is None:
            raise ValueError("a nonzero revision round requires a working itinerary")
        if self.status is PlannerStatus.DRAFT_READY:
            if self.working_itinerary is None or not self.initial_evidence_ready:
                raise ValueError("draft_ready requires a guarded working draft and evidence")
            if self.unresolved_decisions or any(
                intent.requires_user_resolution
                for intent in self.working_itinerary.unassigned_intents
            ):
                raise ValueError("draft_ready cannot hide unresolved user-authority decisions")
        if self.status is PlannerStatus.AWAITING_USER:
            if (
                self.active_interaction is None
                or self.active_interaction.status is not InteractionStatus.ACTIVE
            ):
                raise ValueError("awaiting_user requires one active PlannerInteraction")
        elif self.active_interaction is not None and (
            self.active_interaction.status is InteractionStatus.ACTIVE
        ):
            raise ValueError("an active PlannerInteraction requires awaiting_user status")
        if self.status is PlannerStatus.READY_TO_PUBLISH:
            if any(
                value is None
                for value in (
                    self.working_itinerary,
                    self.materialized_schedule,
                    self.cost_draft,
                    self.validation_report,
                    self.validation_observation,
                )
            ):
                raise ValueError(
                    "ready_to_publish requires draft, materialization, cost and validation"
                )
            if (
                self.validation_observation is not None
                and self.validation_observation.result != "passed"
            ):
                raise ValueError("ready_to_publish requires passed Planner validation")
            if self.unresolved_decisions:
                raise ValueError("ready_to_publish cannot contain unresolved decisions")
        return self

    @property
    def accepted_local_change_dates(self) -> frozenset[date] | None:
        """Restore only this request's accepted scope, never an older patch's dates."""
        request = self.plan_change_request
        if request is None or request.requested_scope != "local_replan":
            return None
        return frozenset(
            day
            for decision in self.decision_trace
            if decision.input_refs.plan_change_request_id == request.plan_change_request_id
            and isinstance(decision.payload, ReviseDraftPayload)
            for day in decision.payload.declared_affected_dates
        )

    @property
    def current_scope(self) -> PlannerScope:
        return PlannerScope(
            trip_id=self.trip_id,
            generation_id=self.generation_id,
            task_book_id=self.based_on_task_book_id,
            task_book_version=self.based_on_task_book_version,
            task_book_state_version=self.candidate_pool.scope.task_book_state_version,
            workspace_revision=self.workspace_revision,
        )

    def _validate_v3_materialized_ownership(self) -> None:
        for artifact in (self.materialized_schedule, self.cost_draft, self.validation_report):
            if artifact is None:
                continue
            if str(artifact.trip_id) != self.trip_id:
                raise ValueError("materialized artifact belongs to another trip")
            if str(artifact.task_book_id) != self.based_on_task_book_id:
                raise ValueError("materialized artifact belongs to another task book")
            if artifact.task_book_revision != self.based_on_task_book_version:
                raise ValueError("materialized artifact uses a different task-book version")


def _require_workspace_scope(actual: PlannerScope, expected: PlannerScope) -> None:
    require_same_scope_ownership(actual, expected)


def validate_workspace_transition(
    previous: PlannerWorkspaceState,
    following: PlannerWorkspaceState,
) -> None:
    """Enforce immutable confirmed task-book identity and monotonic workspace revision."""

    if (
        previous.trip_id,
        previous.generation_id,
        previous.based_on_task_book_id,
        previous.based_on_task_book_version,
        previous.candidate_pool.scope.task_book_state_version,
    ) != (
        following.trip_id,
        following.generation_id,
        following.based_on_task_book_id,
        following.based_on_task_book_version,
        following.candidate_pool.scope.task_book_state_version,
    ):
        raise ValueError(
            "Planner workspace cannot mutate trip, generation or confirmed task-book identity"
        )
    if following.workspace_revision != previous.workspace_revision + 1:
        raise ValueError("Planner workspace revision must advance by exactly one")


V4_PLANNER_WORKSPACE_CONTRACTS: tuple[type[V4ContractModel], ...] = (PlannerWorkspaceState,)

V4_PLANNER_CONTRACTS: tuple[type[BaseModel], ...] = (
    *V4_PLANNER_REF_CONTRACTS,
    *V4_PLANNER_STRATEGY_CONTRACTS,
    *V4_PLANNER_DRAFT_CONTRACTS,
    *V4_PLANNER_PATCH_CONTRACTS,
    *V4_PLANNER_OBSERVATION_CONTRACTS,
    *V4_PLANNER_EVIDENCE_CONTRACTS,
    *V4_PLANNER_DECISION_CONTRACTS,
    *V4_PLANNER_PUBLICATION_CONTRACTS,
    *V4_PLANNER_WORKSPACE_CONTRACTS,
    *V4_PLAN_CHANGE_CONTRACTS,
)
