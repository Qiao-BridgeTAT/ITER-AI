"""Bounded LangGraph supervisor for every pre-task-book V4 user turn."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, replace
from datetime import date
from typing import Literal, TypedDict, cast
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from pydantic import ValidationError

from backend.agent.model_audit import record_model_call_annotation
from backend.agent.model_gateway import (
    ModelCancellation,
    ModelFailureCode,
    ModelGateway,
    ModelGatewayError,
)
from backend.agent.prepare.decision_contracts import (
    CardActionObservation,
    CardTextCoverageAssessment,
    CompoundTripIntakeExtraction,
    ConcreteIntentReview,
    ConcreteIntentReviewChoice,
    FinalSupplementAssessment,
    GroundedPlaceReference,
    IntakePreferenceFact,
    PaceModificationAssessment,
    PaceRequirementExtraction,
    PrepareActionRecoveryDecision,
    TaskBookActionObservation,
    TaskBookReviewModificationAssessment,
    TripBasicsAssessment,
    TripDateRangeAssessment,
    decision_contract_for_capability,
    decision_contract_for_published_plan,
)
from backend.agent.prepare.guard_recovery import (
    decision_schema_repair_instruction,
    guard_feedback,
    intake_schema_repair_instruction,
    is_action_conflict,
)
from backend.agent.prepare.progress import (
    AttractionProgress,
    DiningProgress,
    report_attraction_progress,
    report_dining_progress,
)
from backend.agent.prepare.prompts import (
    build_card_text_assessment_request,
    build_compound_trip_intake_request,
    build_concrete_intent_review_request,
    build_final_supplement_assessment_request,
    build_pace_modification_assessment_request,
    build_pace_requirement_extraction_request,
    build_place_reference_request,
    build_prepare_decision_request,
    build_task_book_review_assessment_request,
    build_trip_basics_assessment_request,
)
from backend.agent.prepare.response_composer import (
    ComposedPrepareResponse,
    compose_prepare_response,
    fallback_prepare_text,
)
from backend.agent.prepare.section_guard import DISCOVERY_ORDER, recalculate_section_coverage
from backend.agent.prepare.semantic_transaction import (
    ResolvedSelection,
    SemanticCompilationError,
    bind_selection,
    normalize_compound_basics,
    partition_pending_entities,
    resolution_choices,
)
from backend.agent.prepare.task_book import TaskBookBuilder
from backend.contracts.v4.conversation import ConversationMessageV4, V4Attachment
from backend.contracts.v4.enums import (
    CoverageStatus,
    DiscoverySection,
    PendingInteractionKind,
    PrepareActionKind,
    SectionProposalKind,
    TaskBookStatus,
    ToolObservationStatus,
)
from backend.contracts.v4.planner_publication import PlannerPublishedPlan
from backend.contracts.v4.prepare import (
    CandidateAssistantMessage,
    CandidateTurnResult,
    OpeningHoursRequest,
    PlaceFactsRequest,
    PlaceProductsRequest,
    PrepareDecision,
    ReplyGoal,
    ReplyOnlyAction,
    ResolvePlaceRequest,
    SectionProposal,
    SpatialRoutesRequest,
    TicketAvailabilityRequest,
    ToolObservation,
)
from backend.contracts.v4.semantic_operations import (
    ExcludeConcreteEntityOperation,
    SelectConcreteEntityOperation,
    SemanticDomainV4,
    SemanticOperationProposal,
    SemanticTargetV4,
    SetNotApplicableOperation,
    SetTripBasicsOperation,
)
from backend.contracts.v4.state import (
    CardGenerationRecovery,
    DecisionAuditPointer,
    DiscoveryRuntimeState,
    PendingInteraction,
    SectionCoverage,
    TripSemanticState,
)
from backend.contracts.v4.task_book import TaskBookV4
from backend.discovery.cards.candidate_composition import (
    CardGenerationContextRequired,
    CardGenerationError,
)
from backend.discovery.cards.service import PrepareCardService
from backend.discovery.tools.registry import ExecutedToolObservation, PrepareToolExecutor
from backend.domain.discovery.entity_identity import is_city_entity_ref
from backend.domain.discovery.state_merge import (
    AcceptedV4Operation,
    V4SemanticMergeError,
    advance_without_operations,
    merge_v4_operations,
    task_book_confirmation_fingerprint,
)
from backend.domain.party_size import parse_party_size
from backend.planning.city_registry import CityRegistryError, default_city_registry
from backend.planning.recall_plan import RecallPlanError

PREPARE_GRAPH_RECURSION_LIMIT = 20
OPTIONAL_TRIP_PREFERENCES_TARGET = "optional_trip_preferences"
TRIP_DATE_RANGE_TARGET = "date_range"
logger = logging.getLogger("uvicorn.error")


class PrepareGraphError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class PrepareTurnInput:
    turn_id: UUID
    trip_id: UUID
    generation_id: UUID
    assistant_message_id: UUID
    expected_state_version: int
    user_event_kind: Literal[
        "text", "trip_setup", "card_answer", "retry_interaction", "task_book_confirmation"
    ]
    user_text: str
    user_message_ref: str
    semantic_state: TripSemanticState
    runtime_state: DiscoveryRuntimeState
    recent_conversation: tuple[ConversationMessageV4, ...]
    business_date: date
    published_plan: PlannerPublishedPlan | None = None
    prevalidated_operations: tuple[SemanticOperationProposal, ...] = ()
    signed_source_refs: tuple[str, ...] = ()
    signed_entity_refs: tuple[str, ...] = ()
    required_card_section: DiscoverySection | None = None
    card_text_section: DiscoverySection | None = None


@dataclass(frozen=True, slots=True)
class PrepareGraphContext:
    cancellation: ModelCancellation
    tool_executor: PrepareToolExecutor
    card_service: PrepareCardService | None


@dataclass(frozen=True, slots=True)
class PrepareGraphResult:
    candidate: CandidateTurnResult
    accepted_operations: tuple[AcceptedV4Operation, ...]
    tool_observations: tuple[ExecutedToolObservation, ...]
    card_action_observations: tuple[CardActionObservation, ...]
    task_book_action_observations: tuple[TaskBookActionObservation, ...]
    invalidated_interaction_ids: tuple[str, ...]
    decisions: tuple[PrepareDecision, ...]
    outcome: str
    generation_mode: str
    failure_code: str | None
    trace: tuple[str, ...]
    attachments: tuple[V4Attachment, ...]


class PrepareGraphState(TypedDict, total=False):
    turn_input: PrepareTurnInput
    semantic_state: TripSemanticState
    runtime_state: DiscoveryRuntimeState
    decision: PrepareDecision
    date_range_assessment: TripDateRangeAssessment | None
    trip_basics_assessment: TripBasicsAssessment | None
    final_supplement_assessment: FinalSupplementAssessment | None
    task_book_review_assessment: TaskBookReviewModificationAssessment | None
    decisions: list[PrepareDecision]
    accepted_operations: list[AcceptedV4Operation]
    pending_entity_choices: list[dict[str, object]]
    concrete_intent_reviews: dict[str, ConcreteIntentReviewChoice]
    entity_inventory_attempted: bool
    observations: list[ExecutedToolObservation]
    card_action_observations: list[CardActionObservation]
    task_book_action_observations: list[TaskBookActionObservation]
    invalidated_interaction_ids: list[str]
    allowed_source_refs: set[str]
    known_entity_refs: set[str]
    prior_tool_request_ids: set[str]
    guard_failure_counts: dict[str, int]
    blocked_guard_contexts: set[str]
    tool_round: int
    decision_mode: Literal["natural_text", "decide_after_update", "bounded_redecide"]
    decision_failure_code: str | None
    action_failure_code: str | None
    action_redecision_required: bool
    task_book_repair_used: bool
    repair_used: bool
    outcome: str
    grounding_context: dict[str, object]
    response: ComposedPrepareResponse
    candidate: CandidateTurnResult
    attachments: list[V4Attachment]
    trace: list[str]
    node_visits: dict[str, int]


class PrepareAgentGraph:
    """Qwen decides; code validates, merges, observes, guards, and stages."""

    def __init__(
        self,
        gateway: ModelGateway,
        *,
        task_book_builder: TaskBookBuilder | None = None,
        attraction_gateway: ModelGateway | None = None,
        dining_gateway: ModelGateway | None = None,
    ) -> None:
        self._gateway = gateway
        self._attraction_gateway = attraction_gateway or gateway
        self._dining_gateway = dining_gateway or gateway
        self._task_books = task_book_builder or TaskBookBuilder()
        builder = StateGraph(PrepareGraphState, context_schema=PrepareGraphContext)
        builder.add_node("load_context", self._load_context)
        builder.add_node("apply_prevalidated_operations", self._apply_prevalidated_operations)
        builder.add_node("prepare_decision", self._prepare_decision)
        builder.add_node("begin_attraction_preferences", self._begin_attraction_preferences)
        builder.add_node("merge_working_aggregate", self._merge_working_aggregate)
        builder.add_node("execute_tools", self._execute_tools)
        builder.add_node("bounded_redecide", self._bounded_redecide)
        builder.add_node("preflight_main_action", self._preflight_main_action)
        builder.add_node("prepare_main_action", self._prepare_main_action)
        builder.add_node("compose_response", self._compose_response)
        builder.add_node("stage_candidate_turn", self._stage_candidate_turn)
        builder.add_edge(START, "load_context")
        builder.add_conditional_edges(
            "load_context",
            self._route_initial,
            {
                "signed": "apply_prevalidated_operations",
                "natural": "prepare_decision",
            },
        )
        builder.add_conditional_edges(
            "apply_prevalidated_operations",
            self._route_after_prevalidated,
            {"trip_setup": "begin_attraction_preferences", "decide": "prepare_decision"},
        )
        builder.add_edge("begin_attraction_preferences", "merge_working_aggregate")
        builder.add_edge("prepare_decision", "merge_working_aggregate")
        builder.add_conditional_edges(
            "merge_working_aggregate",
            self._route_decision,
            {"tool": "execute_tools", "action": "preflight_main_action"},
        )
        builder.add_edge("execute_tools", "bounded_redecide")
        builder.add_edge("bounded_redecide", "merge_working_aggregate")
        builder.add_conditional_edges(
            "preflight_main_action",
            self._route_after_preflight,
            {"redecide": "bounded_redecide", "execute": "prepare_main_action"},
        )
        builder.add_conditional_edges(
            "prepare_main_action",
            self._route_after_main_action,
            {"redecide": "bounded_redecide", "compose": "compose_response"},
        )
        builder.add_edge("compose_response", "stage_candidate_turn")
        builder.add_edge("stage_candidate_turn", END)
        self._compiled = builder.compile(name="v4-prepare-agent")

    @property
    def compiled_graph(
        self,
    ) -> CompiledStateGraph[
        PrepareGraphState,
        PrepareGraphContext,
        PrepareGraphState,
        PrepareGraphState,
    ]:
        return self._compiled

    async def invoke(
        self,
        turn_input: PrepareTurnInput,
        *,
        cancellation: ModelCancellation,
        tool_executor: PrepareToolExecutor,
        card_service: PrepareCardService | None = None,
    ) -> PrepareGraphResult:
        _validate_turn_input(turn_input)
        raw = cast(
            PrepareGraphState,
            await self._compiled.ainvoke(
                PrepareGraphState(
                    turn_input=turn_input,
                    semantic_state=turn_input.semantic_state.model_copy(deep=True),
                    runtime_state=turn_input.runtime_state.model_copy(deep=True),
                    decisions=[],
                    accepted_operations=[],
                    observations=[],
                    card_action_observations=[],
                    task_book_action_observations=[],
                    invalidated_interaction_ids=[],
                    allowed_source_refs={
                        turn_input.user_message_ref,
                        *turn_input.signed_source_refs,
                    },
                    known_entity_refs={
                        *_known_state_entity_refs(turn_input.semantic_state),
                        *turn_input.signed_entity_refs,
                    },
                    prior_tool_request_ids=set(),
                    guard_failure_counts={},
                    blocked_guard_contexts=set(),
                    tool_round=0,
                    decision_mode=(
                        "decide_after_update"
                        if turn_input.user_event_kind != "text"
                        else "natural_text"
                    ),
                    decision_failure_code=None,
                    action_redecision_required=False,
                    task_book_repair_used=False,
                    repair_used=False,
                    trace=[],
                    node_visits={},
                    attachments=[],
                ),
                config={"recursion_limit": PREPARE_GRAPH_RECURSION_LIMIT},
                context=PrepareGraphContext(cancellation, tool_executor, card_service),
            ),
        )
        candidate = raw.get("candidate")
        response = raw.get("response")
        if candidate is None or response is None:
            raise PrepareGraphError("Prepare graph did not stage a complete candidate turn")
        return PrepareGraphResult(
            candidate=candidate,
            accepted_operations=tuple(raw.get("accepted_operations", ())),
            tool_observations=tuple(raw.get("observations", ())),
            card_action_observations=tuple(raw.get("card_action_observations", ())),
            task_book_action_observations=tuple(raw.get("task_book_action_observations", ())),
            invalidated_interaction_ids=tuple(
                dict.fromkeys(raw.get("invalidated_interaction_ids", ()))
            ),
            decisions=tuple(raw.get("decisions", ())),
            outcome=raw.get("outcome", "answered"),
            generation_mode=response.generation_mode,
            failure_code=(
                response.failure_code
                or raw.get("decision_failure_code")
                or raw.get("action_failure_code")
            ),
            trace=tuple(raw.get("trace", ())),
            attachments=tuple(raw.get("attachments", ())),
        )

    async def _load_context(self, state: PrepareGraphState) -> PrepareGraphState:
        return _visited(state, "load_context", maximum=1)

    @staticmethod
    def _route_initial(state: PrepareGraphState) -> Literal["signed", "natural"]:
        return "signed" if state["turn_input"].prevalidated_operations else "natural"

    async def _apply_prevalidated_operations(self, state: PrepareGraphState) -> PrepareGraphState:
        turn_input = state["turn_input"]
        merged = merge_v4_operations(
            state["semantic_state"],
            state["runtime_state"],
            list(turn_input.prevalidated_operations),
            turn_id=turn_input.turn_id,
            allowed_source_refs=state["allowed_source_refs"],
            known_entity_refs=state["known_entity_refs"],
        )
        guarded = recalculate_section_coverage(
            merged.semantic_state,
            merged.runtime_state,
            merged.accepted_operations,
            affected_sections=merged.affected_sections,
        )
        return {
            "semantic_state": merged.semantic_state,
            "runtime_state": guarded.runtime_state,
            "accepted_operations": list(merged.accepted_operations),
            "invalidated_interaction_ids": list(merged.invalidated_interaction_ids),
            **_visited(state, "apply_prevalidated_operations", maximum=1),
        }

    @staticmethod
    def _route_after_prevalidated(state: PrepareGraphState) -> Literal["trip_setup", "decide"]:
        return "trip_setup" if state["turn_input"].user_event_kind == "trip_setup" else "decide"

    async def _begin_attraction_preferences(self, state: PrepareGraphState) -> PrepareGraphState:
        # A typed form submission has already been validated and merged. There is
        # no natural-language intent to assess or next action for a model to choose.
        if state["runtime_state"].current_section is not DiscoverySection.ATTRACTION_PREFERENCE:
            raise PrepareGraphError("trip setup did not reach attraction preferences")
        decision = PrepareDecision.model_validate(
            {
                "decision_id": f"trip-setup:{state['turn_input'].turn_id}",
                "based_on_state_version": state["semantic_state"].state_version,
                "semantic_operations": [],
                "tool_requests": [],
                "next_action": {
                    "kind": "show_preference_card",
                    "domain": "attraction",
                    "requested_targets": ["attraction_preference"],
                },
                "section_proposal": {"kind": "stay", "section": "attraction_preference"},
                "reply_goal": {"explain_next_step": "目的地与日期已记录，请选择感兴趣的景点方向。"},
            }
        )
        return {
            "decision": decision,
            "decisions": [decision],
            **_visited(state, "begin_attraction_preferences", maximum=1),
        }

    def _gateway_for(self, state: PrepareGraphState) -> ModelGateway:
        turn = state["turn_input"]
        # Output attachments take precedence when a turn advances into another domain.
        sections = [getattr(item.root, "section", None) for item in state.get("attachments", [])]
        section = next((item for item in sections if item is not None), None)
        section = section or turn.required_card_section or turn.card_text_section
        section = section or state["runtime_state"].current_section
        if any(
            item in {DiscoverySection.DINING_PREFERENCE, DiscoverySection.DINING_SPECIFIC}
            for item in (
                section,
                turn.required_card_section,
                turn.card_text_section,
                state["runtime_state"].current_section,
            )
        ):
            return self._dining_gateway
        if turn.user_event_kind == "trip_setup" or section in {
            DiscoverySection.ATTRACTION_PREFERENCE,
            DiscoverySection.ATTRACTION_SPECIFIC,
        }:
            return self._attraction_gateway
        return self._gateway

    async def _prepare_decision(
        self,
        state: PrepareGraphState,
        runtime: Runtime[PrepareGraphContext],
    ) -> PrepareGraphState:
        task_book_review_assessment = await _assess_task_book_review_modification(
            self._gateway_for(state),
            state,
            runtime.context.cancellation,
        )
        decision_state = cast(
            PrepareGraphState,
            {
                **state,
                "task_book_review_assessment": task_book_review_assessment,
            },
        )
        decision, failure_code, repair_used, date_range_assessment = await self._decide(
            decision_state,
            runtime.context.cancellation,
        )
        return {
            "decision": decision,
            "date_range_assessment": date_range_assessment,
            "trip_basics_assessment": decision_state.get("trip_basics_assessment"),
            "concrete_intent_reviews": decision_state.get("concrete_intent_reviews", {}),
            "entity_inventory_attempted": decision_state.get("entity_inventory_attempted", False),
            "final_supplement_assessment": decision_state.get("final_supplement_assessment"),
            "task_book_review_assessment": task_book_review_assessment,
            "decisions": [*state.get("decisions", []), decision],
            "decision_failure_code": failure_code,
            "repair_used": repair_used,
            **_visited(state, "prepare_decision", maximum=1),
        }

    async def _merge_working_aggregate(
        self,
        state: PrepareGraphState,
        runtime: Runtime[PrepareGraphContext],
    ) -> PrepareGraphState:
        decision = state["decision"]
        accepted = list(state.get("accepted_operations", []))
        seen_keys = {item.proposal.root.local_operation_key for item in accepted}
        proposals = [
            item
            for item in decision.semantic_operations
            if item.root.local_operation_key not in seen_keys
        ]
        proposals, pending_entities = partition_pending_entities(decision, proposals)
        advances_version = (
            state["semantic_state"].state_version == state["turn_input"].expected_state_version
        )
        try:
            merged = (
                merge_v4_operations(
                    state["semantic_state"],
                    state["runtime_state"],
                    proposals,
                    turn_id=state["turn_input"].turn_id,
                    allowed_source_refs=state["allowed_source_refs"],
                    known_entity_refs=state["known_entity_refs"],
                    advance_version=advances_version,
                )
                if proposals
                else (
                    advance_without_operations(state["semantic_state"], state["runtime_state"])
                    if advances_version
                    else None
                )
            )
        except V4SemanticMergeError as error:
            if state.get("repair_used", False):
                decision = _fallback_decision(
                    state["runtime_state"], state["semantic_state"].state_version
                )
                fallback_state: PrepareGraphState = {
                    **state,
                    "decision": decision,
                    "decisions": [*state.get("decisions", []), decision],
                    "decision_failure_code": "decision_semantic_validation_failed",
                    "repair_used": True,
                }
                # Even a safe fallback is a committed user turn. Re-enter once with
                # an empty operation set so the shared version advances exactly once.
                return await self._merge_working_aggregate(fallback_state, runtime)
            repaired, failure_code, _, date_range_assessment = await self._decide(
                state,
                runtime.context.cancellation,
                validation_issue=_safe_issue(error),
            )
            repaired_state = cast(
                PrepareGraphState,
                {
                    **state,
                    "decision": repaired,
                    "date_range_assessment": (
                        date_range_assessment or state.get("date_range_assessment")
                    ),
                    "decisions": [*state.get("decisions", []), repaired],
                    "decision_failure_code": failure_code,
                    "repair_used": True,
                },
            )
            return await self._merge_working_aggregate(repaired_state, runtime)

        if merged is None:
            semantic = state["semantic_state"]
            runtime_state = state["runtime_state"]
            new_operations: tuple[AcceptedV4Operation, ...] = ()
            invalidated: tuple[str, ...] = ()
            affected: frozenset[DiscoverySection] = frozenset()
        else:
            semantic = merged.semantic_state
            runtime_state = merged.runtime_state
            new_operations = merged.accepted_operations
            invalidated = merged.invalidated_interaction_ids
            affected = merged.affected_sections
        guarded = recalculate_section_coverage(
            semantic,
            runtime_state,
            new_operations,
            affected_sections=affected,
        )
        transitioned_runtime, review_invalidated = _apply_task_book_review_card_reopen(
            guarded.runtime_state,
            state.get("task_book_review_assessment"),
            decision=decision,
        )
        return {
            "semantic_state": semantic,
            "runtime_state": transitioned_runtime,
            "accepted_operations": [*accepted, *new_operations],
            "pending_entity_choices": [*state.get("pending_entity_choices", []), *pending_entities],
            "invalidated_interaction_ids": [
                *state.get("invalidated_interaction_ids", []),
                *invalidated,
                *review_invalidated,
            ],
            **_visited(state, "merge_working_aggregate", maximum=3),
        }

    @staticmethod
    def _route_decision(state: PrepareGraphState) -> Literal["tool", "action"]:
        return (
            "tool"
            if state["decision"].next_action.kind is PrepareActionKind.USE_TOOL
            and state.get("tool_round", 0) < 2
            and state.get("decision_failure_code") is None
            else "action"
        )

    async def _execute_tools(
        self,
        state: PrepareGraphState,
        runtime: Runtime[PrepareGraphContext],
    ) -> PrepareGraphState:
        runtime.context.cancellation.raise_if_cancelled("prepare_tools")
        requests = state["decision"].tool_requests
        executed = await runtime.context.tool_executor.execute_plan(
            requests,
            state["semantic_state"],
        )
        observations = [*state.get("observations", []), *executed]
        allowed = set(state["allowed_source_refs"])
        known = set(state["known_entity_refs"])
        request_ids = set(state["prior_tool_request_ids"])
        for item in executed:
            observation = item.observation
            allowed.add(f"tool:{observation.request_id}")
            allowed.update(observation.source_refs)
            known.update(observation.entity_refs)
            request_ids.add(observation.request_id)
        return {
            "observations": observations,
            "allowed_source_refs": allowed,
            "known_entity_refs": known,
            "prior_tool_request_ids": request_ids,
            "tool_round": state.get("tool_round", 0) + 1,
            **_visited(state, "execute_tools", maximum=2),
        }

    async def _bounded_redecide(
        self,
        state: PrepareGraphState,
        runtime: Runtime[PrepareGraphContext],
    ) -> PrepareGraphState:
        next_state = cast(
            PrepareGraphState,
            {**state, "decision_mode": "bounded_redecide"},
        )
        decision, failure_code, repair_used, date_range_assessment = await self._decide(
            next_state,
            runtime.context.cancellation,
        )
        return {
            "decision": decision,
            "date_range_assessment": (date_range_assessment or state.get("date_range_assessment")),
            "decisions": [*state.get("decisions", []), decision],
            "decision_failure_code": failure_code,
            "repair_used": state.get("repair_used", False) or repair_used,
            "decision_mode": "bounded_redecide",
            "concrete_intent_reviews": next_state.get("concrete_intent_reviews", {}),
            **_visited(state, "bounded_redecide", maximum=2),
        }

    @staticmethod
    def _route_after_preflight(
        state: PrepareGraphState,
    ) -> Literal["redecide", "execute"]:
        return "redecide" if state.get("action_redecision_required", False) else "execute"

    async def _preflight_main_action(self, state: PrepareGraphState) -> PrepareGraphState:
        decision = state["decision"]
        if decision.next_action.kind is not PrepareActionKind.GENERATE_TASK_BOOK:
            return {
                "action_redecision_required": False,
                **_visited(state, "preflight_main_action", maximum=2),
            }

        guard = self._task_books.preflight(state["semantic_state"], state["runtime_state"])
        if guard.ready:
            return {
                "action_redecision_required": False,
                **_visited(state, "preflight_main_action", maximum=2),
            }

        observation = _recoverable_task_book_action_observation(
            guard.blocking_reasons,
            turn_id=state["turn_input"].turn_id,
            observation_index=len(state.get("task_book_action_observations", [])),
        )
        if observation is None or state.get("task_book_repair_used", False):
            logger.warning(
                "Prepare task-book preflight rejected generation: reasons=%s",
                ",".join(guard.blocking_reasons),
            )
            raise PrepareGraphError("task book generation failed its final guard preflight")

        runtime_state = state["runtime_state"].model_copy(
            update={
                "last_decision": DecisionAuditPointer(
                    turn_id=str(state["turn_input"].turn_id),
                    decision_id=decision.decision_id,
                    action=decision.next_action.kind.value,
                    based_on_state_version=decision.based_on_state_version,
                    status="rejected",
                )
            },
            deep=True,
        )
        return {
            "runtime_state": runtime_state,
            "task_book_action_observations": [
                *state.get("task_book_action_observations", []),
                observation,
            ],
            "action_redecision_required": True,
            "task_book_repair_used": True,
            "outcome": "observed",
            "attachments": [],
            **_visited(state, "preflight_main_action", maximum=2),
        }

    @staticmethod
    def _route_after_main_action(
        state: PrepareGraphState,
    ) -> Literal["redecide", "compose"]:
        return "redecide" if state.get("action_redecision_required", False) else "compose"

    async def _prepare_main_action(
        self,
        state: PrepareGraphState,
        runtime: Runtime[PrepareGraphContext],
    ) -> PrepareGraphState:
        decision = state["decision"]
        runtime_state = state["runtime_state"]
        action = decision.next_action.kind
        attachments = list(state.get("attachments", []))
        outcome = "answered"
        pending = runtime_state.pending_interaction
        action_failure_code: str | None = None
        action_redecision_required = False
        if action in {
            PrepareActionKind.SHOW_PREFERENCE_CARD,
            PrepareActionKind.SHOW_SPECIFIC_CARD,
        }:
            try:
                if runtime.context.card_service is None:
                    raise CardGenerationError("card service is not configured")
                card = await runtime.context.card_service.generate(
                    state["semantic_state"],
                    section=runtime_state.current_section,
                    turn_id=state["turn_input"].turn_id,
                    cancellation=runtime.context.cancellation,
                )
            except (
                CardGenerationError,
                RecallPlanError,
                ModelGatewayError,
                ValidationError,
            ) as error:
                if isinstance(error, ModelGatewayError):
                    if error.code is ModelFailureCode.CANCELLED:
                        raise
                    action_failure_code = f"card_model_{error.code.value}"
                elif isinstance(error, RecallPlanError):
                    action_failure_code = "card_recall_plan_failed"
                elif isinstance(error, ValidationError):
                    action_failure_code = "card_content_invalid"
                else:
                    action_failure_code = error.code
                logger.warning(
                    "Prepare card generation failed: section=%s action=%s code=%s",
                    runtime_state.current_section.value,
                    action.value,
                    action_failure_code,
                )
                card_observation = _recoverable_card_action_observation(
                    error,
                    section=runtime_state.current_section,
                    turn_id=state["turn_input"].turn_id,
                    observation_index=len(state.get("card_action_observations", [])),
                    failure_code=action_failure_code,
                )
                if (
                    card_observation is not None
                    and state["turn_input"].user_event_kind != "trip_setup"
                ):
                    runtime_state = runtime_state.model_copy(
                        update={
                            "pending_interaction": None,
                            "last_decision": DecisionAuditPointer(
                                turn_id=str(state["turn_input"].turn_id),
                                decision_id=decision.decision_id,
                                action=action.value,
                                based_on_state_version=decision.based_on_state_version,
                                status="rejected",
                            ),
                        },
                        deep=True,
                    )
                    return {
                        "runtime_state": runtime_state,
                        "card_action_observations": [
                            *state.get("card_action_observations", []),
                            card_observation,
                        ],
                        "action_failure_code": None,
                        "action_redecision_required": True,
                        "outcome": "observed",
                        "attachments": [],
                        **_visited(state, "prepare_main_action", maximum=2),
                    }
                pending = PendingInteraction.model_validate(
                    {
                        **_free_text_pending(runtime_state, decision).model_dump(mode="json"),
                        "recovery": CardGenerationRecovery(
                            failure_code=action_failure_code
                        ).model_dump(mode="json"),
                    }
                )
                outcome = "awaiting_user"
            else:
                attachments = [V4Attachment(root=card)]
                pending = PendingInteraction(
                    interaction_id=card.interaction_id,
                    kind=(
                        PendingInteractionKind.PREFERENCE_CARD
                        if card.kind.value == "preference_card"
                        else PendingInteractionKind.SPECIFIC_CARD
                    ),
                    section=card.section,
                    # The generated card is the authoritative interaction
                    # boundary. Qwen may explain why it chose the action, but
                    # it cannot widen or duplicate the signed card targets.
                    target_ids=[card.section.value],
                    option_refs=[item.option_id for item in card.options],
                    based_on_state_version=card.based_on_state_version,
                    dependency_fingerprint=card.dependency_fingerprint,
                )
                outcome = "card_ready"
        elif action is PrepareActionKind.GENERATE_TASK_BOOK:
            try:
                task_book = self._task_books.build(state["semantic_state"], runtime_state)
            except ValueError as error:
                logger.warning("Prepare task-book final guard rejected generation")
                raise PrepareGraphError("task book generation failed its final guard") from error
            else:
                runtime_state = runtime_state.model_copy(
                    update={
                        "task_book_candidate": task_book,
                        "current_section": DiscoverySection.TASK_BOOK_REVIEW,
                    },
                    deep=True,
                )
                attachments = [V4Attachment(root=task_book.value)]
                fingerprint = task_book_confirmation_fingerprint(
                    task_book.task_book_id,
                    task_book.value.version,
                    task_book.based_on_state_version,
                )
                pending = PendingInteraction(
                    interaction_id=str(
                        uuid5(
                            NAMESPACE_URL,
                            f"v4-task-book-confirmation:{task_book.task_book_id}:{task_book.value.version}",
                        )
                    ),
                    kind=PendingInteractionKind.CONFIRMATION,
                    section=DiscoverySection.TASK_BOOK_REVIEW,
                    target_ids=[task_book.task_book_id],
                    based_on_state_version=task_book.based_on_state_version,
                    dependency_fingerprint=fingerprint,
                )
                outcome = "task_book_ready"
        elif action in {
            PrepareActionKind.ASK_CLARIFICATION,
            PrepareActionKind.FINAL_SUPPLEMENT,
            PrepareActionKind.USE_TOOL,
        }:
            pending = _free_text_pending(runtime_state, decision)
            outcome = "awaiting_user"
        runtime_state = runtime_state.model_copy(
            update={
                "pending_interaction": pending,
                "last_decision": DecisionAuditPointer(
                    turn_id=str(state["turn_input"].turn_id),
                    decision_id=decision.decision_id,
                    action=action.value,
                    based_on_state_version=decision.based_on_state_version,
                    status="rejected" if action_failure_code is not None else "executed",
                ),
            },
            deep=True,
        )
        grounding = _grounding_context(state, runtime_state, outcome)
        grounding["attachments"] = [_attachment_summary(item) for item in attachments]
        if action_failure_code is not None:
            grounding["card_generation"] = {
                "status": "unavailable",
                "section": runtime_state.current_section.value,
                "failure_code": action_failure_code,
            }
        return {
            "runtime_state": runtime_state,
            "action_failure_code": action_failure_code,
            "action_redecision_required": action_redecision_required,
            "outcome": outcome,
            "grounding_context": grounding,
            "attachments": attachments,
            **_visited(state, "prepare_main_action", maximum=2),
        }

    async def _compose_response(
        self,
        state: PrepareGraphState,
        runtime: Runtime[PrepareGraphContext],
    ) -> PrepareGraphState:
        for attachment in state.get("attachments", []):
            section = getattr(attachment.root, "section", None)
            if section is DiscoverySection.ATTRACTION_PREFERENCE:
                await report_attraction_progress(AttractionProgress.PREFERENCE_READY)
            elif section is DiscoverySection.ATTRACTION_SPECIFIC:
                await report_attraction_progress(AttractionProgress.SPECIFIC_READY)
            elif section is DiscoverySection.DINING_PREFERENCE:
                await report_dining_progress(DiningProgress.PREFERENCE_READY)
            elif section is DiscoverySection.DINING_SPECIFIC:
                await report_dining_progress(DiningProgress.SPECIFIC_READY)
        grounding = state["grounding_context"]
        failure_code = state.get("decision_failure_code")
        if failure_code is not None:
            response = ComposedPrepareResponse(
                text=fallback_prepare_text(grounding),
                generation_mode="fallback",
                failure_code=failure_code,
            )
        else:
            response = await compose_prepare_response(
                self._gateway_for(state),
                grounding,
                cancellation=runtime.context.cancellation,
            )
        return {
            "response": response,
            **_visited(state, "compose_response", maximum=1),
        }

    async def _stage_candidate_turn(self, state: PrepareGraphState) -> PrepareGraphState:
        turn_input = state["turn_input"]
        response = state["response"]
        content_hash = hashlib.sha256(response.text.encode("utf-8")).hexdigest()
        candidate = CandidateTurnResult(
            turn_id=str(turn_input.turn_id),
            trip_id=str(turn_input.trip_id),
            generation_id=str(turn_input.generation_id),
            base_state_version=turn_input.expected_state_version,
            candidate_state_version=state["semantic_state"].state_version,
            semantic_state=state["semantic_state"],
            discovery_runtime_state=state["runtime_state"],
            accepted_operations=[item.proposal for item in state.get("accepted_operations", [])],
            tool_observations=[item.observation for item in state.get("observations", [])],
            assistant_message=CandidateAssistantMessage(
                message_id=str(turn_input.assistant_message_id),
                text=response.text,
                generation_mode=cast(Literal["qwen", "fallback"], response.generation_mode),
                content_hash=content_hash,
                attachment_ids=[_attachment_id(item) for item in state.get("attachments", [])],
            ),
            attachment_payloads=[_attachment_id(item) for item in state.get("attachments", [])],
            decision_audit_ref=f"decision:{state['decision'].decision_id}",
        )
        return {
            "candidate": candidate,
            **_visited(state, "stage_candidate_turn", maximum=1),
        }

    async def _decide(
        self,
        state: PrepareGraphState,
        cancellation: ModelCancellation,
        *,
        validation_issue: str | None = None,
    ) -> tuple[PrepareDecision, str | None, bool, TripDateRangeAssessment | None]:
        failure_counts = state.setdefault("guard_failure_counts", {})
        blocked_contexts = state.setdefault("blocked_guard_contexts", set())
        evidence_context = hashlib.sha256(
            json.dumps(
                {
                    "state": state["semantic_state"].model_dump(mode="json"),
                    "observations": [
                        item.observation.model_dump(mode="json")
                        for item in state.get("observations", [])
                    ],
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        if evidence_context in blocked_contexts:
            return (
                _fallback_with_validated_operations(state, None),
                "decision_repeated_guard_failure",
                True,
                state.get("date_range_assessment"),
            )
        repair_used = validation_issue is not None
        issue = validation_issue
        card_observations = state.get("card_action_observations", [])
        latest_card_observation = card_observations[-1] if card_observations else None
        task_book_observations = state.get("task_book_action_observations", [])
        latest_task_book_observation = (
            task_book_observations[-1] if task_book_observations else None
        )
        forced_semantic_operation: (
            Literal[
                "trip_basics", "trip_basics_with_additions", "final_supplement", "pace_requirement"
            ]
            | None
        ) = None
        required_trip_basics_fields: tuple[str, ...] = ()
        trip_intake_transition_action: Literal[
            "none",
            "ask_trip_dates",
            "ask_optional_preferences",
            "show_attraction_preferences",
        ] = "none"
        forced_post_update_action: (
            Literal["show_preference_card", "show_specific_card", "final_supplement"] | None
        ) = None
        card_text_required_targets: tuple[str, ...] = ()
        card_text_requires_answer_first = False
        task_book_review_assessment = state.get("task_book_review_assessment")
        task_book_required_targets = tuple(
            item.value
            for item in (
                task_book_review_assessment.required_targets
                if task_book_review_assessment is not None
                else ()
            )
        )
        task_book_requested_card = (
            task_book_review_assessment.requested_card_section
            if task_book_review_assessment is not None
            else None
        )
        if (
            state["turn_input"].card_text_section is not None
            and state["decision_mode"] == "natural_text"
        ):
            try:
                card_text_assessment = await self._gateway_for(state).generate_structured(
                    build_card_text_assessment_request(
                        user_text=state["turn_input"].user_text,
                        section=state["turn_input"].card_text_section,
                        semantic_state=state["semantic_state"],
                    ),
                    CardTextCoverageAssessment,
                    cancellation=cancellation,
                )
                card_text_required_targets = tuple(
                    dict.fromkeys(
                        item.value for item in card_text_assessment.value.required_targets
                    )
                )
                card_text_requires_answer_first = card_text_assessment.value.requires_answer_first
            except ModelGatewayError as error:
                if error.code is ModelFailureCode.CANCELLED or error.requires_runtime_recovery:
                    raise
                return (
                    _fallback_decision(
                        state["runtime_state"], state["semantic_state"].state_version
                    ),
                    "card_text_coverage_unavailable",
                    False,
                    None,
                )
        try:
            assessment = (
                await _assess_explicit_trip_basics(
                    self._gateway_for(state),
                    state,
                    cancellation,
                )
                if validation_issue is None
                else state.get("trip_basics_assessment")
            )
        except SemanticCompilationError:
            # A decision retry cannot repair an immutable, invalid intake inventory.
            return (
                _fallback_decision(state["runtime_state"], state["semantic_state"].state_version),
                "prepare_intake_assessment_invalid",
                True,
                None,
            )
        if assessment is not None:
            state["trip_basics_assessment"] = assessment
        intake_assessment = assessment or state.get("trip_basics_assessment")
        required_additional_targets = tuple(
            item.value
            for item in (
                intake_assessment.required_additional_targets
                if intake_assessment is not None
                else []
            )
        )
        require_lodging_not_applicable = bool(
            not state["semantic_state"].lodging.not_applicable
            and assessment is not None
            and assessment.explicit_lodging_not_applicable
        )
        final_supplement_assessment = (
            await _assess_final_supplement(
                self._gateway_for(state),
                state,
                cancellation,
            )
            if validation_issue is None
            else state.get("final_supplement_assessment")
        )
        if final_supplement_assessment is not None:
            state["final_supplement_assessment"] = final_supplement_assessment
        contextual_task_book_generation = bool(
            _has_active_final_supplement_followup(state)
            and _explicitly_requests_task_book_generation(state["turn_input"].user_text)
        )
        basics_before_turn = state["semantic_state"].trip_basics
        destination_was_bound = bool(
            basics_before_turn.destination_name and basics_before_turn.destination_canonical_id
        )
        destination_will_be_bound = bool(
            destination_was_bound
            or (
                assessment is not None
                and assessment.explicit_destination
                and len(default_city_registry().mentioned_in(state["turn_input"].user_text)) == 1
            )
        )
        dates_were_bound = bool(
            basics_before_turn.start_date
            and basics_before_turn.end_date
            and basics_before_turn.duration_days
        )
        date_range_resolution = assessment.date_range_resolution if assessment is not None else None
        date_range_ready = bool(
            assessment is not None
            and (
                assessment.explicit_date_range
                or (date_range_resolution is not None and date_range_resolution.status == "ready")
            )
        )
        date_range_needs_confirmation = bool(
            date_range_resolution is not None
            and date_range_resolution.status == "needs_confirmation"
        )
        dates_will_be_bound = dates_were_bound or date_range_ready
        requires_date_followup = bool(
            destination_will_be_bound
            and (date_range_needs_confirmation or not dates_will_be_bound)
            and not _fact_capabilities(state)
        )
        if assessment is not None and any(
            (
                assessment.explicit_destination,
                date_range_ready,
                assessment.explicit_duration_days,
                assessment.explicit_travelers,
                assessment.explicit_trip_goals,
            )
        ):
            required_trip_basics_fields = tuple(
                field
                for present, field in (
                    (assessment.explicit_destination, "destination"),
                    (date_range_ready, "date_range"),
                    (assessment.explicit_duration_days, "duration_days"),
                    (assessment.explicit_travelers, "travelers"),
                    (assessment.explicit_trip_goals, "trip_goals"),
                )
                if present
            )
            forced_semantic_operation = (
                "trip_basics_with_additions" if assessment.has_additional_request else "trip_basics"
            )
            if requires_date_followup:
                trip_intake_transition_action = "ask_trip_dates"
            elif (
                destination_will_be_bound
                and state["runtime_state"].current_section is DiscoverySection.OTHER
                and not _fact_capabilities(state)
            ):
                explicitly_closed = assessment.explicitly_no_more_requirements
                if (
                    destination_was_bound
                    or explicitly_closed
                    or assessment.requests_attraction_cards
                ):
                    trip_intake_transition_action = "show_attraction_preferences"
                else:
                    trip_intake_transition_action = "ask_optional_preferences"
            explicit_fields = [
                label
                for present, label in (
                    (assessment.explicit_destination, "目的地"),
                    (date_range_ready, "已明确或确认的具体起止日期"),
                    (assessment.explicit_duration_days, "旅行天数"),
                    (assessment.explicit_travelers, "同行人"),
                    (assessment.explicit_trip_goals, "旅行目标"),
                )
                if present
            ]
            issue = (
                "窄范围语义核验确认本轮需要处理"
                + "、".join(explicit_fields)
                + "。必须忠实写入 set_trip_basics，并重新选择后续行动。"
            )
            if (
                date_range_ready
                and date_range_resolution is not None
                and date_range_resolution.status == "ready"
            ):
                issue += (
                    "日期核验给出的完整日期对为 "
                    + _assessed_date_range_summary(date_range_resolution)
                    + "。"
                    "set_trip_basics 必须同时逐值写入 start_date 与 end_date，"
                    "不能分批写入或改写该日期对。"
                )
            elif date_range_needs_confirmation and date_range_resolution is not None:
                issue += (
                    "根据用户明确的开始日期与旅行天数形成了待确认候选："
                    + _assessed_date_range_summary(date_range_resolution)
                    + "。"
                    "本轮不得写入 start_date/end_date；必须在同一个 date_range 问题中"
                    "复述候选的开始和结束日期并询问用户是否正确。"
                )
            if assessment.has_additional_request:
                issue += (
                    "本轮采用复合合同：trip_basics 字段单独填写上述明确基础信息；"
                    "其余需求仍放 semantic_operations，事实问题仍可用工具处理。"
                    "不能只写其他偏好而漏掉 trip_basics，也不能只写基础信息而丢弃其他需求。"
                )
            if require_lodging_not_applicable:
                issue += (
                    "窄范围核验也确认用户明确表示本次住宿不适用。semantic_operations 必须"
                    "包含且只包含一条对应住宿结论的 set_not_applicable："
                    "target=lodging_area，reason 忠实保留用户原话；"
                    "不能继续生成住宿区域或档次卡。"
                )
        elif requires_date_followup:
            trip_intake_transition_action = "ask_trip_dates"
            if date_range_needs_confirmation and date_range_resolution is not None:
                issue = (
                    "当前旅行已有待用户确认的完整日期候选："
                    + _assessed_date_range_summary(date_range_resolution)
                    + "。"
                    "必须只询问 date_range，在同一个问题中复述开始和结束日期并确认；"
                    "确认前不得写入任一日期端点、提前展示景点卡或生成任务书。"
                )
            else:
                issue = (
                    "当前旅行已经确认目的地，但仍缺少可执行的具体起止日期。"
                    "必须只询问 date_range；不得提前展示景点卡、生成任务书或补造日期。"
                )
        elif contextual_task_book_generation or (
            final_supplement_assessment is not None
            and final_supplement_assessment.explicitly_no_more_requirements
            and not final_supplement_assessment.has_additional_request
        ):
            if _has_active_final_supplement_followup(state):
                forced_semantic_operation = "final_supplement"
                issue = (
                    "当前激活的最终补充问题与用户明确的任务书生成指令共同确认补充已经结束。"
                    if contextual_task_book_generation
                    else "窄范围语义核验确认用户明确表示没有其他补充。"
                ) + ("必须确认 final_supplement，并请求生成旅行任务书。")
            else:
                # A missing/stale prompt cannot be repaired by retrying the same
                # forbidden confirmation. Restore its executable interaction first.
                forced_post_update_action = "final_supplement"
                issue = (
                    "尚无当前版本有效的最终补充交互。本轮只发 final_supplement 追问，"
                    "不要写任何确认或重复已有需求，不要生成任务书；待用户回答后再确认。"
                )
        elif (
            task_book_review_assessment is not None
            and task_book_review_assessment.is_change_request
            and task_book_review_assessment.required_targets
            == [SemanticTargetV4.TRANSPORT_AND_PACE]
            and task_book_requested_card is None
            and not task_book_review_assessment.has_fact_question
        ):
            forced_semantic_operation = "pace_requirement"
            issue = (
                "任务书复核节点核验确认用户正在修改旅行节奏或步行强度。"
                "必须忠实写入 transport_and_pace，并进入最终补充以生成新版任务书。"
            )
        elif task_book_review_assessment is not None and (
            task_book_review_assessment.is_change_request
            or task_book_review_assessment.requests_regeneration
        ):
            if task_book_requested_card is not None and not task_book_required_targets:
                forced_post_update_action = (
                    "show_preference_card"
                    if task_book_requested_card
                    in {
                        DiscoverySection.ATTRACTION_PREFERENCE,
                        DiscoverySection.DINING_PREFERENCE,
                        DiscoverySection.LODGING_AREA_PREFERENCE,
                        DiscoverySection.LODGING_CLASS_PREFERENCE,
                    }
                    else "show_specific_card"
                )
            described_targets = "、".join(task_book_required_targets) or "无直接语义写入"
            issue = (
                "当前位于任务书复核节点。节点核验确认用户要求修改已有任务书；"
                f"本轮至少不得遗漏这些明确目标：{described_targets}。"
                "这些目标是最低覆盖要求，不是字段白名单；同句其他明确需求仍可正常处理。"
                "任何已接受的修改都必须使旧任务书失效，并在重新确认最终补充后生成新版。"
            )
            if task_book_requested_card is not None:
                issue += (
                    f"用户明确要求重开 {task_book_requested_card.value} 卡片。"
                    "若本轮无需先解析实体或查询事实，next_action 必须立即展示这张卡；"
                    "不能只口头答应、继续停留在 task_book_review 或直接生成任务书。"
                )
            elif task_book_required_targets:
                issue += (
                    "修改写入后必须进入程序核验得到的下一环节；通常为 final_supplement，"
                    "不能沿用旧任务书的确认交互。"
                )
            else:
                issue += (
                    "用户尚未给出可写入的具体修改内容；必须只追问他要修改哪项，"
                    "不能伪造修改或废止仍有效的任务书。"
                )
        if forced_semantic_operation == "pace_requirement":
            extracted = await _extract_pace_requirement_decision(
                self._gateway_for(state),
                state,
                cancellation,
            )
            return (*extracted, date_range_resolution)
        if (
            forced_semantic_operation == "trip_basics_with_additions"
            and require_lodging_not_applicable
            and state["semantic_state"].trip_basics.destination_name
            and _required_next_tool_capability(
                state,
                [item.observation for item in state.get("observations", [])],
            )
            == "resolve_place"
        ):
            extracted = await _extract_compound_trip_intake_decision(
                self._gateway_for(state),
                state,
                cancellation,
            )
            return (*extracted, date_range_resolution)
        frozen_decision: PrepareDecision | None = None
        recovery_context: dict[str, object] | None = None
        feedback_payload: dict[str, object] | None = None
        for attempt in range(2 if not repair_used else 1):
            decision: PrepareDecision | None = None
            decision_call_id: str | None = None
            try:
                observations = [item.observation for item in state.get("observations", [])]
                required_next_tool = (
                    None
                    if latest_card_observation is not None
                    or latest_task_book_observation is not None
                    or (
                        assessment is not None
                        and assessment.explicit_destination
                        and not assessment.has_additional_request
                    )
                    else _required_next_tool_capability(state, observations)
                )
                continuation_semantics = _continuation_semantics(state)
                required_semantic_operation = (
                    forced_semantic_operation or _required_initial_semantic_operation(state)
                )
                if forced_semantic_operation is None and len(required_additional_targets) > 1:
                    required_semantic_operation = "none"
                required_concrete_choice = _required_concrete_choice(state)
                forbid_semantic_operations = _forbid_initial_semantic_operations(state)
                required_post_update_action = (
                    forced_post_update_action or _required_post_update_action(state)
                )
                result = await self._gateway_for(state).generate_structured(
                    build_prepare_decision_request(
                        published_plan_summary=_published_plan_reply_context(
                            state["turn_input"].published_plan
                        ),
                        mode=state["decision_mode"],
                        user_text=state["turn_input"].user_text,
                        business_date=state["turn_input"].business_date,
                        semantic_state=state["semantic_state"],
                        runtime_state=state["runtime_state"],
                        recent_conversation=list(state["turn_input"].recent_conversation),
                        observations=observations,
                        card_observations=card_observations,
                        task_book_observations=task_book_observations,
                        allowed_source_refs=state["allowed_source_refs"],
                        known_entity_refs=state["known_entity_refs"],
                        prior_tool_request_ids=state["prior_tool_request_ids"],
                        tool_round=state.get("tool_round", 0),
                        required_next_tool_capability=required_next_tool,
                        continuation_semantics=continuation_semantics,
                        required_semantic_operation=required_semantic_operation,
                        required_trip_basics_fields=required_trip_basics_fields,
                        date_range_resolution=(
                            date_range_resolution
                            if date_range_resolution is not None
                            and date_range_resolution.status != "none"
                            else None
                        ),
                        trip_intake_transition_action=trip_intake_transition_action,
                        required_post_update_action=required_post_update_action,
                        forbid_semantic_operations=forbid_semantic_operations,
                        required_concrete_choice=required_concrete_choice,
                        required_card_section=(
                            None
                            if latest_card_observation is not None
                            else state["turn_input"].required_card_section
                        ),
                        card_text_section=state["turn_input"].card_text_section,
                        card_text_required_targets=card_text_required_targets,
                        card_text_requires_answer_first=card_text_requires_answer_first,
                        task_book_review_assessment=task_book_review_assessment,
                        validation_issue=issue,
                        resolved_selections=[item.context() for item in _resolution_choices(state)],
                        guard_feedback=feedback_payload,
                        action_recovery=recovery_context,
                        pending_entity_choices=state.get("pending_entity_choices", []),
                        required_additional_targets=required_additional_targets,
                        requirement_facts=[
                            item.model_dump(mode="json")
                            for item in intake_assessment.requirement_facts
                        ]
                        if intake_assessment is not None
                        else [],
                    ),
                    decision_contract_for_published_plan(
                        cast(type[PrepareDecision], PrepareActionRecoveryDecision)
                        if frozen_decision is not None
                        else decision_contract_for_capability(
                            required_next_tool,
                            tools_allowed=(
                                state.get("tool_round", 0) < 2
                                and (
                                    state["decision_mode"] != "bounded_redecide"
                                    or required_next_tool is not None
                                    or continuation_semantics == "full"
                                )
                            ),
                            continuation=state["decision_mode"] == "bounded_redecide",
                            no_tool_semantics=continuation_semantics,
                            required_semantic_operation=required_semantic_operation,
                            trip_intake_transition_action=trip_intake_transition_action,
                            required_post_update_action=required_post_update_action,
                            forbid_semantic_operations=forbid_semantic_operations,
                            required_concrete_choice=required_concrete_choice,
                            card_observation_status=(
                                latest_card_observation.status
                                if latest_card_observation is not None
                                else None
                            ),
                            task_book_observation_failure_code=(
                                latest_task_book_observation.failure_code
                                if latest_task_book_observation is not None
                                else None
                            ),
                            entity_clarification_required=bool(
                                feedback_payload and feedback_payload.get("recovery") == "clarify"
                            ),
                        ),
                        required=state["turn_input"].published_plan is not None
                        and frozen_decision is None,
                    ),
                    cancellation=cancellation,
                    validation_context={"assessed_trip_date_range": date_range_resolution},
                )
                decision_call_id = result.audit_call_id
                raw_payload = result.value.model_dump(mode="json")
                if frozen_decision is not None:
                    # The retry selects an action, never re-extracts a validated transaction.
                    raw_payload["semantic_operations"] = [
                        item.model_dump(mode="json") for item in frozen_decision.semantic_operations
                    ]
                    raw_payload["section_proposal"] = frozen_decision.section_proposal.model_dump(
                        mode="json"
                    )
                    if frozen_decision.published_plan_intent is not None:
                        raw_payload["published_plan_intent"] = frozen_decision.published_plan_intent
                decision_payload = _materialize_runtime_decision_payload(
                    raw_payload,
                    state,
                    "full" if frozen_decision is not None else continuation_semantics,
                    "none" if frozen_decision is not None else required_semantic_operation,
                )
                decision = PrepareDecision.model_validate(decision_payload)
                decision = _bind_authoritative_operation_sources(decision, state)
                decision = _materialize_assessed_trip_date_range(
                    decision,
                    date_range_resolution,
                )
                decision = _strip_unchanged_contextual_destination(
                    decision,
                    state,
                    date_range_resolution,
                )
                decision = _bind_registered_trip_destination(decision, state)
                if (
                    state["decision_mode"] == "natural_text"
                    and state.get("trip_basics_assessment") is None
                    and not state.get("entity_inventory_attempted", False)
                    and any(
                        isinstance(item.root, ResolvePlaceRequest)
                        for item in decision.tool_requests
                    )
                ):
                    state["entity_inventory_attempted"] = True
                    inventory = await _assess_explicit_trip_basics(
                        self._gateway_for(state), state, cancellation, require_entity_inventory=True
                    )
                    if inventory is not None:
                        state["trip_basics_assessment"] = inventory
                decision = _complete_intake_requirement_facts(decision, state)
                decision = _complete_intake_entity_queries(decision, state)
                decision = await _ground_resolve_queries(
                    self._gateway_for(state),
                    decision,
                    state,
                    cancellation,
                )
                decision = _bind_unambiguous_tool_entity_refs(decision, state)
                decision = _normalize_tool_request_ids(decision, state)
                decision = await _review_unverified_hard_intents(
                    self._gateway_for(state), decision, state, cancellation
                )
                decision = _align_card_action_after_merge(decision, state)
                _validate_additional_semantic_coverage(decision, state, required_additional_targets)
                _validate_final_supplement_additions(decision, state)
                _validate_decision(decision, state)
                _validate_card_observation_decision(decision, state)
                _validate_task_book_observation_decision(decision, state)
                _validate_required_trip_basics_fields(
                    decision,
                    state,
                    required_trip_basics_fields,
                )
                _validate_assessed_trip_date_range(decision, date_range_resolution)
                _validate_trip_intake_transition_action(
                    decision,
                    trip_intake_transition_action,
                    state,
                )
                if require_lodging_not_applicable:
                    lodging_targets = {
                        SemanticTargetV4.LODGING_AREA,
                        SemanticTargetV4.LODGING_CLASS,
                        SemanticTargetV4.LODGING_BOOKING,
                    }
                    lodging_operations = [
                        item.root
                        for item in decision.semantic_operations
                        if item.root.target in lodging_targets
                        or getattr(item.root, "domain", None) is SemanticDomainV4.LODGING
                    ]
                    lodging_not_applicable = [
                        item
                        for item in lodging_operations
                        if isinstance(item, SetNotApplicableOperation)
                        and item.target is SemanticTargetV4.LODGING_AREA
                    ]
                    if len(lodging_not_applicable) != 1 or len(lodging_operations) != 1:
                        raise PrepareGraphError(
                            "explicit lodging not applicable requires one lodging_area operation"
                        )
                if decision.next_action.kind is not PrepareActionKind.ASK_CLARIFICATION:
                    recorded_targets = {
                        item.root.target.value for item in decision.semantic_operations
                    } | {
                        item.proposal.root.target.value
                        for item in state.get("accepted_operations", [])
                    }
                    missing_targets = set(card_text_required_targets) - recorded_targets
                    if decision.next_action.kind is PrepareActionKind.USE_TOOL:
                        missing_targets -= {"attraction_entity", "dining_entity", "lodging_booking"}
                    if missing_targets:
                        raise PrepareGraphError(
                            "card text dropped explicit semantic targets: "
                            + ",".join(sorted(missing_targets))
                        )
                card_target_by_section = {
                    DiscoverySection.ATTRACTION_PREFERENCE: "attraction_preference",
                    DiscoverySection.DINING_PREFERENCE: "dining_preference",
                    DiscoverySection.LODGING_AREA_PREFERENCE: "lodging_area",
                    DiscoverySection.LODGING_CLASS_PREFERENCE: "lodging_class",
                }
                card_section = state["turn_input"].card_text_section
                current_card_target = (
                    card_target_by_section.get(card_section) if card_section is not None else None
                )
                if (
                    current_card_target in card_text_required_targets
                    and not card_text_requires_answer_first
                    and decision.next_action.kind is PrepareActionKind.REPLY_ONLY
                ):
                    raise PrepareGraphError(
                        "card text answer requires an executable next interaction"
                    )
                await record_model_call_annotation(
                    self._gateway_for(state),
                    decision_call_id,
                    "llm_business_guard",
                    {
                        "business_guard_result": {
                            "status": "accepted",
                            "guard": "prepare_decision_guard",
                        },
                        "accepted_or_rejected": "accepted",
                        "materialized_output": decision.model_dump(mode="json"),
                    },
                )
                return decision, None, repair_used, date_range_resolution
            except ModelGatewayError as error:
                if error.code is ModelFailureCode.CANCELLED or error.requires_runtime_recovery:
                    raise
                if decision_call_id is not None:
                    await record_model_call_annotation(
                        self._gateway_for(state),
                        decision_call_id,
                        "llm_business_guard",
                        {
                            "business_guard_result": {
                                "status": "rejected",
                                "guard": "prepare_decision_guard",
                                "reason": error.code.value,
                            },
                            "accepted_or_rejected": "rejected",
                            "failure_stage": "business_guard_or_grounding",
                            "failure_code": error.code.value,
                            "materialized_output": (
                                decision.model_dump(mode="json") if decision is not None else None
                            ),
                        },
                    )
                logger.warning(
                    "Prepare decision model contract failure: attempt=%s code=%s issues=%s",
                    attempt + 1,
                    error.code.value,
                    ",".join(error.validation_issues) or "none",
                )
                issue = _decision_repair_instruction(error)
                feedback = guard_feedback(error, issue)
                feedback_payload = feedback.payload()
                fingerprint = feedback.fingerprint(evidence_context)
                failure_counts[fingerprint] = failure_counts.get(fingerprint, 0) + 1
                repeated_failure = failure_counts[fingerprint] >= 2
                if repeated_failure:
                    blocked_contexts.add(evidence_context)
                repair_used = True
                if attempt == 0 and validation_issue is None and not repeated_failure:
                    continue
                safe_continuation = _safe_non_card_continuation(decision, state, error)
                if safe_continuation is None:
                    safe_continuation = _safe_post_merge_card_continuation(frozen_decision, state)
                if safe_continuation is not None:
                    _validate_decision(safe_continuation, state)
                    return safe_continuation, None, True, date_range_resolution
                return (
                    _fallback_with_validated_operations(state, frozen_decision),
                    f"decision_{error.code.value}",
                    True,
                    date_range_resolution,
                )
            except (PrepareGraphError, ValueError) as error:
                issue = _decision_validation_repair_instruction(error)
                feedback = guard_feedback(error, issue)
                recoverable_city_tool = (
                    isinstance(error, SemanticCompilationError)
                    and error.code in {"prepare_city_is_not_poi", "prepare_category_is_not_poi"}
                    and required_next_tool is None
                )
                if decision is not None and (is_action_conflict(error) or recoverable_city_tool):
                    try:
                        _validate_additional_semantic_coverage(
                            decision, state, required_additional_targets
                        )
                        if any(
                            item.root.operation_type
                            in {"confirm_task_book", "confirm_final_supplement"}
                            for item in decision.semantic_operations
                        ):
                            raise PrepareGraphError(
                                "confirmation is not eligible for action-only recovery"
                            )
                        _validate_required_trip_basics_fields(
                            decision, state, required_trip_basics_fields
                        )
                        _validate_assessed_trip_date_range(decision, date_range_resolution)
                        preview = _preview_card_runtime(decision, state)
                    except (PrepareGraphError, ValueError):
                        pass  # A semantics failure is not eligible for an action-only retry.
                    else:
                        frozen_decision = decision
                        recovery_context = {
                            "post_merge_section": preview.current_section.value,
                            "preserved_operations": [
                                item.model_dump(mode="json")
                                for item in decision.semantic_operations
                            ],
                        }
                        next_kind = _default_action_for_section(preview.current_section)
                        if next_kind is not None:
                            domain = preview.current_section.value.split("_", 1)[0]
                            feedback = replace(
                                feedback,
                                allowed_next_action={
                                    "kind": next_kind.value,
                                    "domain": "general"
                                    if next_kind is PrepareActionKind.FINAL_SUPPLEMENT
                                    else domain,
                                    "requested_targets": [preview.current_section.value],
                                },
                            )
                        elif preview.current_section is DiscoverySection.OTHER:
                            basics = state["semantic_state"].trip_basics
                            target = (
                                TRIP_DATE_RANGE_TARGET
                                if basics.destination_canonical_id
                                else "trip_destination"
                            )
                            feedback = replace(
                                feedback,
                                allowed_next_action={
                                    "kind": "ask_clarification",
                                    "requested_targets": [target],
                                },
                            )
                feedback_payload = feedback.payload()
                fingerprint = feedback.fingerprint(evidence_context)
                failure_counts[fingerprint] = failure_counts.get(fingerprint, 0) + 1
                repeated_failure = failure_counts[fingerprint] >= 2
                if repeated_failure:
                    blocked_contexts.add(evidence_context)
                await record_model_call_annotation(
                    self._gateway_for(state),
                    decision_call_id,
                    "llm_business_guard",
                    {
                        "business_guard_result": {
                            "status": "rejected",
                            "guard": "prepare_decision_guard",
                            "reason": str(error),
                        },
                        "accepted_or_rejected": "rejected",
                        "failure_stage": "business_guard",
                        "failure_code": "decision_validation_failed",
                        "request_guard_feedback_full": (
                            {**feedback_payload, "fingerprint": fingerprint}
                        ),
                        "materialized_output": (
                            decision.model_dump(mode="json") if decision is not None else None
                        ),
                    },
                )
                logger.warning(
                    "Prepare decision business validation failure: attempt=%s issue=%s",
                    attempt + 1,
                    f"{feedback.code}:{feedback.path}",
                )
                repair_used = True
                if attempt == 0 and validation_issue is None and not repeated_failure:
                    continue
                safe_continuation = _safe_non_card_continuation(decision, state, error)
                if safe_continuation is None:
                    safe_continuation = _safe_post_merge_card_continuation(frozen_decision, state)
                if safe_continuation is not None:
                    _validate_decision(safe_continuation, state)
                    return safe_continuation, None, True, date_range_resolution
                return (
                    _fallback_with_validated_operations(state, frozen_decision),
                    "decision_validation_failed",
                    True,
                    date_range_resolution,
                )
        raise AssertionError("unreachable decision retry state")


def _decision_repair_instruction(error: ModelGatewayError) -> str:
    basics_repair = decision_schema_repair_instruction(error)
    if basics_repair is not None:
        return basics_repair
    if any(issue.endswith(":8cb5bef3b4fb") for issue in error.validation_issues):
        return (
            "next_action.kind=use_tool 与非空 tool_requests 必须同时出现；"
            "若不调用工具，tool_requests 必须为 []。请修复这一对应关系。"
        )
    if any(issue.endswith(":36dbc66a6110") for issue in error.validation_issues):
        return (
            "每条 semantic operation 的 source_refs 不能重复。请去重；"
            "若用户只在询问事实且没有表达偏好或修改，请删除该 operation。"
        )
    if any(issue.endswith(":ecf3e0401a34") for issue in error.validation_issues):
        return (
            "餐饮实体的 disposition 不合法。川菜、本帮菜、素食、咖啡馆等类别不是具体餐厅，"
            "应改用 select_preference_direction(target=dining_preference, domain=dining)；"
            "只有用户点名的具体餐厅才使用 dining_entity，必吃/专程去映射为 destination，"
            "顺路去映射为 if_convenient，排除则使用 exclude_concrete_entity。"
        )
    return (
        "结构化决策未满足合同。请逐项检查 action 对应字段、必填字段、允许枚举，"
        "并确保 use_tool 与 1～4 个 tool_requests 同时出现。"
    )


async def _review_unverified_hard_intents(
    gateway: ModelGateway,
    decision: PrepareDecision,
    state: PrepareGraphState,
    cancellation: ModelCancellation,
) -> PrepareDecision:
    if state["decision_mode"] == "decide_after_update":
        return decision  # Signed card choices already carry the user's exact enum.
    cache = state.setdefault("concrete_intent_reviews", {})
    intake = state.get("trip_basics_assessment")
    pending: dict[str, str] = {}
    for wrapped in decision.semantic_operations:
        operation = wrapped.root
        if (
            operation.operation_type != "select_concrete_entity"
            or operation.domain is not SemanticDomainV4.ATTRACTION
            or operation.disposition != "must"
        ):
            continue
        name = _normalize_place_reference(operation.display_name)
        if name in cache:
            continue
        matches = [
            item
            for item in (intake.named_entity_intents if intake is not None else [])
            if item.domain == "attraction" and _normalize_place_reference(item.query) == name
        ]
        if len(matches) == 1:
            cache[name] = ConcreteIntentReviewChoice.model_validate(
                {
                    "entity_key": name,
                    "disposition": matches[0].disposition,
                    "quote": matches[0].query,
                }
            )
        else:
            pending[name] = operation.display_name
    if pending:
        keys = {f"e{index + 1}": name for index, name in enumerate(pending)}
        # Cache unresolved entries before the call: a malformed review cannot
        # trigger the same auxiliary model call again in a decision repair.
        for name in pending:
            cache[name] = ConcreteIntentReviewChoice(entity_key=name, disposition="unclear")
        try:
            result = await gateway.generate_structured(
                build_concrete_intent_review_request(
                    user_text=state["turn_input"].user_text,
                    entities=[
                        {"entity_key": key, "name": pending[name]} for key, name in keys.items()
                    ],
                ),
                ConcreteIntentReview,
                cancellation=cancellation,
            )
            choices = result.value.choices
            if len(choices) == len(keys) and {item.entity_key for item in choices} == set(keys):
                for item in choices:
                    if item.quote and item.quote in state["turn_input"].user_text:
                        cache[keys[item.entity_key]] = item
        except ModelGatewayError as error:
            if error.code is ModelFailureCode.CANCELLED or error.requires_runtime_recovery:
                raise
            # Keep the independent semantics unresolved; ask only about that
            # place rather than silently saving an unverified hard constraint.

    operations = []
    for index, wrapped in enumerate(decision.semantic_operations):
        operation = wrapped.root
        if not isinstance(
            operation, (SelectConcreteEntityOperation, ExcludeConcreteEntityOperation)
        ):
            operations.append(wrapped)
            continue
        review = (
            cache.get(_normalize_place_reference(operation.display_name))
            if operation.domain is SemanticDomainV4.ATTRACTION
            else None
        )
        if review is None:
            operations.append(wrapped)
            continue
        if review.disposition == "unclear":
            raise SemanticCompilationError(
                "prepare_hard_intent_needs_clarification",
                f"semantic_operations[{index}].disposition",
                recovery="clarify",
            )
        if review.disposition == "not_requested":
            continue
        value = operation.model_dump(mode="json")
        if review.disposition == "avoid":
            value["operation_type"] = "exclude_concrete_entity"
            value.pop("disposition", None)
        else:
            value["operation_type"] = "select_concrete_entity"
            value["disposition"] = review.disposition
        operations.append(SemanticOperationProposal.model_validate(value))
    return decision.model_copy(update={"semantic_operations": operations})


def _decision_validation_repair_instruction(error: Exception) -> str:
    if str(error).startswith("compound intake omitted semantic targets: "):
        targets = str(error).removeprefix("compound intake omitted semantic targets: ")
        return (
            f"semantic_operations 漏掉当前用户明确需求：{targets}。"
            "保留已有 trip_basics 和正确操作，"
            "只补齐这些类别。总预算用 general_constraint 条件要求，酒店预算用 lodging_class；"
            "景点/餐饮方向使用 select_preference_direction 或 exclude_preference_direction，"
            "target 分别为 attraction_preference/dining_preference，label 保留完整偏好；"
            "trip_goals 不能替代偏好方向操作。"
            "餐厅按路线帮选用 set_delegation_scope / dining_entity，不能当作已有餐厅。"
            "未查明的具体地点仍用工具，其他独立需求不要等到工具后才写。"
        )
    if isinstance(error, SemanticCompilationError):
        if error.code == "prepare_hard_intent_needs_clarification":
            return (
                f"{error.path} 的强制意愿尚未得到可靠语义核验。"
                "删除这条待确认的地点操作，保留其他正确需求；只针对该地点 ask_clarification，"
                "确认用户是想去、必去还是不去。不要重复调用工具、擅自换档或声称已保存。"
            )
        if error.code == "prepare_unknown_tool_entity":
            return (
                f"只修复 {error.path}：实体 ID 必须逐字取自 known_entity_refs，不能自行生成或改写。"
                "若目标地点尚未解析，先 resolve_place，再用返回的实体 ID 查询事实。"
                "保留其他正确需求及每个地点原本的意愿档位。"
            )
        if error.code in {"prepare_city_is_not_poi", "prepare_category_is_not_poi"}:
            return (
                f"{error.path} 已被识别为目的地城市或类别偏好，不是具体地点。"
                "删除该 resolve_place 请求；"
                "城市只写 set_trip_basics.destination_name，由程序绑定城市 ID。"
                "保留已经提取的日期、同行人、预算及偏好。类别偏好也不需要解析 POI；"
                "没有其他真实查询时，选择合并后可执行的卡片或追问，不要强行 use_tool。"
            )
        if error.code.startswith("prepare_reopen_"):
            return (
                f"只修复 {error.path}：{error.code}。只能重开一个已经到达的卡片章节，"
                "不能借 reopen 跳过前置章节；next_action 的 domain 和 requested_targets "
                "须对应同一张卡。保留当前消息的其他明确需求。"
            )
        if error.recovery == "clarify":
            return (
                f"{error.path} 对应的查询没有唯一结果；保持其他已解析选择，"
                "不要重试相同查询或虚构 ID。只对未确定地点 ask_clarification，"
                "clarification.target 与 next_action.requested_targets 对齐。"
            )
        return (
            f"只修复 {error.path}：{error.code}。地点逐条使用 resolved_selections 的短键，"
            "每个键最多一次且分别保留原话意愿；基础信息只写 trip_basics。"
            "保留其他正确字段；真正冲突或无法唯一绑定时只针对该项澄清。"
        )
    issue = str(error)
    if issue == "final supplement additions cannot be acknowledged without handling":
        return (
            "最终补充核验确认本句还有待处理事项，不能只回复已记下。"
            "明确新需求必须写 semantic_operations；新地点尚未解析则 use_tool/resolve_place；"
            "确有歧义时 ask_clarification。事实问题在 reply_goal.answer_questions 中说明，"
            "并按事实工具边界处理。保留每个地点的独立意愿，不凭空填写身份或来源。"
        )
    if "dining entity requires a dining disposition" in issue:
        return (
            "只修复 semantic_operations 中 domain=dining 的 disposition："
            "必吃/专程去使用 destination，顺路可吃使用 if_convenient；"
            "must/want 仅用于景点，不用于餐厅。保留地点、意愿、来源及其他正确字段。"
        )
    if issue == "unconfirmed assessed date range cannot be written":
        return (
            "日期核验只形成了待用户确认的完整候选。本轮删除 set_trip_basics 中的"
            " start_date/end_date，保留其他已经明确的基础字段；下一行动只询问 date_range，"
            "并逐值复述候选开始和结束日期。确认前不得保存任何日期端点。"
        )
    if issue in {
        "ready assessed date range requires one trip basics update",
        "trip basics date range differs from assessed context",
    }:
        return (
            "日期核验已经从当前日期补充或上下文确认得到完整日期对。必须且只能在一条"
            " set_trip_basics 中同时逐值复制 date_range_resolution 的 start_date 与 end_date；"
            "不得遗漏、分批写入或改写日期，duration_days 如填写必须与首尾均计入的范围一致。"
        )
    if issue == "explicit lodging not applicable requires one lodging_area operation":
        return (
            "窄范围核验已确认用户明确表示本次住宿不适用。保留 trip_basics 与其他正确需求，"
            "在所有 lodging 相关 semantic_operations 中必须且只能保留一条"
            "对应住宿结论的 set_not_applicable："
            "target=lodging_area，reason 忠实保留当天往返、住亲友家或不需要"
            "酒店的原话。删除同轮其他 lodging domain 或 lodging_area/lodging_class/"
            "lodging_booking 操作；不能改成区域偏好、existing booking 或继续显示住宿卡。"
        )
    if issue == "operation target does not match its domain":
        return (
            "每条 semantic operation 的 target 必须使用对应 domain："
            "attraction_preference/attraction_entity -> attraction；"
            "dining_preference/dining_requirement/dining_entity -> dining；"
            "lodging_area/lodging_class/lodging_booking -> lodging；"
            "transport_and_pace -> transport；"
            "trip_basics/general_constraint/final_supplement/"
            "task_book_confirmation/conflict -> general。"
            "只修正错误 operation 的 domain，不改变 target、用户原意或其他已正确字段。"
        )
    if issue.startswith("task-book revision dropped explicit semantic targets: "):
        targets = issue.removeprefix("task-book revision dropped explicit semantic targets: ")
        return (
            "任务书复核节点的前置核验只声明了本轮至少不能漏掉的修改目标。"
            f"请保留现有正确结果，并为这些缺失目标补上对应 semantic_operations：{targets}。"
            "菜系或餐饮风格用 dining_preference，不要当作具体餐厅；"
            "具体景点、餐厅或酒店需要先完成 resolve_place。"
        )
    if issue.startswith("task-book revision requires card action "):
        action = issue.removeprefix("task-book revision requires card action ")
        return (
            f"用户明确要求重开推荐卡。本轮不再只做文字承诺，next_action.kind 必须改为 {action}，"
            "domain 与 requested_targets 必须对应"
            " task_book_review_assessment.requested_card_section。"
        )
    if issue == "task-book revision needs a concrete modification target":
        return (
            "用户表示要修改任务书但尚未给出具体内容。不要写任何 semantic_operations，"
            "只用 ask_clarification 询问他要修改日期、偏好、必去地点、指定酒店还是其他要求。"
        )
    if issue.startswith("task-book revision requires next action "):
        action = issue.removeprefix("task-book revision requires next action ")
        return (
            "保留已经提取并通过核验的任务书修改；旧任务书会由程序失效。"
            f"next_action.kind 必须改为 {action}，继续程序核验后的节点，"
            "不得停留在旧任务书确认或声称旧版本仍可确认。"
        )
    if issue == "task-book revision concrete choice changed the user disposition":
        return (
            "具体地点已经核验，但本轮实体操作没有忠实保留用户的取舍。"
            "必去景点使用 attraction/must，想去使用 attraction/want，顺路去使用"
            " attraction/if_convenient，排除景点使用 exclude_concrete_entity；"
            "必吃餐厅使用 dining/destination，顺路吃使用 dining/if_convenient，"
            "排除餐厅也使用 exclude_concrete_entity。保留同句其他已明确修改。"
        )
    concrete_followup_prefix = "concrete entity continuation requires next action "
    if issue.startswith(concrete_followup_prefix):
        action = issue.removeprefix(concrete_followup_prefix)
        return (
            "保留已经按 Observation 提出的具体地点操作；语义合并后主线仍需要真实的下一交互。"
            f"next_action.kind 必须改为 {action}，并填写该行动合同要求的 domain 与 targets。"
            "不能改成 reply_only 后只口头承诺继续，也不能删除用户的具体地点意向。"
        )
    if issue.startswith("card action is not executable after semantic merge: "):
        section = issue.removeprefix("card action is not executable after semantic merge: ")
        if section in {item.value for item in DiscoverySection}:
            expected = _default_action_for_section(DiscoverySection(section))
            if expected is not None:
                domain = section.split("_", 1)[0]
                if section == "final_supplement":
                    domain = "general"
                return (
                    f"只修复 next_action：合并后的章节为 {section}；"
                    f"kind={expected.value}，domain={domain}，requested_targets=[{section}]。"
                    "保留已正确提取的全部 semantic_operations，不要删除偏好以退回旧章节，"
                    "也不要添加虚构需求来跳过当前章节。"
                )
        return (
            f"合并本轮语义后，程序核验的实际章节仍是 {section}，不能执行当前卡片行动。"
            "请保留用户表达的全部需求，补齐遗漏的语义操作。other 进入景点探索需要"
            "已绑定 canonical ID 的目的地与可执行的具体起止日期；同行人和旅行目标不是"
            "阻塞项。目的地已确认但日期缺失时必须 ask_clarification(date_range)，不得"
            "补造日期或提前发卡。只有目的地尚未明确或无法绑定时才"
            " ask_clarification。other、final_supplement、task_book_review 没有偏好或具体卡；"
            "偏好章节使用 show_preference_card，attraction_specific/dining_specific 使用"
            " show_specific_card。不得跳过未完成章节。"
        )
    if issue == "trip intake requires the optional preferences follow-up":
        return (
            "目的地和日期已确认，other 已具备完成条件。保留本轮全部明确语义，下一行动必须是一次"
            " ask_clarification；domain=general，requested_targets 与 clarification.target 都只能是"
            " optional_trip_preferences。自然询问是否还有其他需求或偏好，并说明没有的话将推荐"
            "当地景点方向；不得重复追问日期、同行人或旅行目标。"
        )
    if issue == "trip intake requires the date range follow-up":
        return (
            "目的地已经确认，但可执行的具体起止日期仍缺失。保留本轮全部明确语义，"
            "下一行动必须是 ask_clarification；domain=general，requested_targets 与"
            " clarification.target 都只能是 date_range。自然询问开始和结束日期，"
            "不得补造日期、提前展示卡片或改问其他偏好。"
        )
    if issue == "trip intake requires the attraction preference card":
        return (
            "目的地已确认，而且用户已经明确没有其他补充、直接要求景点推荐，或正在回答此前"
            "的基础信息追问。保留本轮全部明确语义，下一行动必须是 attraction 的"
            " show_preference_card，requested_targets 包含 attraction_preference；不得重复追问。"
        )
    if issue == "optional trip preferences response requires an executable attraction interaction":
        return (
            "当前 user_text 正在回答 optional_trip_preferences。保留用户明确补充的语义；"
            "若没有事实问题或真实歧义，必须按合并后的实际章节展示景点偏好卡或具体景点卡，"
            "不能 reply_only、跳到其他领域或再次询问同一个 optional_trip_preferences。"
        )
    if issue == "card text answer requires an executable next interaction":
        return (
            "用户已经明确回答当前偏好卡，且没有需要先回答的问题。保留全部语义操作，"
            "必须选择实际 show_*_card 或确有必要的 ask_clarification/use_tool；"
            "不能 reply_only 只声称接下来推荐。景点/餐饮偏好之后为对应具体卡，"
            "住宿区域之后为住宿档次卡，住宿档次之后为 final_supplement。"
        )
    if issue.startswith("card text dropped explicit semantic targets: "):
        return (
            "卡片自由补充中有尚未记录的明确需求："
            + issue.removeprefix("card text dropped explicit semantic targets: ")
            + "。必须按原话提出对应 operation；一般风格使用 select_preference_direction，"
            "限制使用 add_conditional_requirement。不能仅换卡片或声称已保存。"
        )
    if issue == "current fact question requires a grounded tool action":
        return (
            "这是需要实时事实的用户问题，当前 Observation 尚不足。"
            "必须选择 next_action.kind=use_tool，并提供 1～4 个完整 tool_requests；"
            "若地点尚未解析先请求 resolve_place，已有 entity_refs 则请求所需事实能力。"
        )
    capability_prefix = "current fact question requires tool capability "
    if issue.startswith(capability_prefix):
        capability = issue.removeprefix(capability_prefix)
        return (
            f"当前事实问题下一步必须请求 {capability}；"
            "不要添加与问题无关的工具。使用新的 request_id，并只引用 Observation 中已有的实体 ID。"
        )
    if issue == "tool request uses an unknown entity reference":
        return (
            "工具请求中的实体 ID 必须逐字取自 known_entity_refs；"
            "不得改写、缩短或自行生成 canonical_entity_id。"
        )
    if issue == "resolve_place query is not grounded in the current user message":
        return (
            "resolve_place.query 必须逐字复制 current user_text 当前点名的地点；"
            "不要使用 recent_conversation 中的旧地点，也不要添加城市名或类别后缀。"
        )
    if issue == "resolve_place query was already resolved in this turn":
        return (
            "该地点本轮已经完成 resolve_place。停止重复解析，使用 tool_observations 的"
            " entity_refs 提出具体实体操作，或选择非工具行动。"
        )
    if issue == "trip destination must use trip basics, not resolve_place":
        return (
            "当前地点是本次旅行的目的地城市，不是景点 POI。删除 resolve_place 请求，"
            "把用户明确说出的城市写入 set_trip_basics.destination_name；"
            "canonical ID 由程序绑定。继续保留同轮的具体日期、同行人、旅行目标和其他需求。"
        )
    if issue.startswith("trip basics dropped explicit fields: "):
        return (
            "set_trip_basics 漏掉了前置核验确认的字段："
            + issue.removeprefix("trip basics dropped explicit fields: ")
            + "。只从当前 user_text 忠实补齐这些字段；旅行天数用一至五的整数，"
            "只有用户明确给出具体日期时才写 ISO 起止日期；目的地只写城市名，不生成 canonical ID。"
        )
    if issue == "trip destination is not grounded in the current user message":
        return (
            "set_trip_basics.destination_name 必须逐字来自当前 user_text 明确表达的目的地城市。"
            "删除模型推测的城市，保留其他有原话证据的基础字段；仍有歧义时 ask_clarification。"
        )
    if issue == "decision repeated an already executed tool request ID":
        return (
            "新的工具请求必须使用 prior_tool_request_ids 中从未出现过的新 request_id；"
            "不要重复执行已经产生 Observation 的请求。"
        )
    return _safe_issue(error)


def _validate_final_supplement_additions(
    decision: PrepareDecision, state: PrepareGraphState
) -> None:
    assessment = state.get("final_supplement_assessment")
    if (
        assessment is not None
        and assessment.has_additional_request
        and not decision.semantic_operations
        and not state.get("accepted_operations")
        and decision.next_action.kind
        in {
            PrepareActionKind.REPLY_ONLY,
            PrepareActionKind.FINAL_SUPPLEMENT,
            PrepareActionKind.GENERATE_TASK_BOOK,
        }
        and not decision.reply_goal.answer_questions
        and not decision.reply_goal.fact_requirements
    ):
        raise PrepareGraphError(
            "final supplement additions cannot be acknowledged without handling"
        )


def _safe_post_merge_card_continuation(
    frozen: PrepareDecision | None, state: PrepareGraphState
) -> PrepareDecision | None:
    """Finish an already-validated transaction after action-only retries fail.

    SectionGuard owns which card is executable after the model chose a card.
    At final supplement only, a saved reply may become the non-committing
    "anything else?" question. Never recover semantic errors,
    invent a new intent, execute a tool, or confirm/generate a task book here.
    """
    if frozen is None or frozen.next_action.kind not in {
        PrepareActionKind.SHOW_PREFERENCE_CARD,
        PrepareActionKind.SHOW_SPECIFIC_CARD,
        PrepareActionKind.REPLY_ONLY,
    }:
        return None
    if not frozen.semantic_operations or any(
        item.root.operation_type in {"confirm_task_book", "confirm_final_supplement"}
        for item in frozen.semantic_operations
    ):
        return None
    section = _preview_card_runtime(frozen, state).current_section
    kind = _default_action_for_section(section)
    if (
        frozen.next_action.kind is PrepareActionKind.REPLY_ONLY
        and kind is not PrepareActionKind.FINAL_SUPPLEMENT
    ):
        return None
    if kind not in {
        PrepareActionKind.SHOW_PREFERENCE_CARD,
        PrepareActionKind.SHOW_SPECIFIC_CARD,
        PrepareActionKind.FINAL_SUPPLEMENT,
    }:
        return None
    payload = frozen.model_dump(mode="json")
    payload["next_action"] = {
        "kind": kind.value,
        "domain": "general"
        if kind is PrepareActionKind.FINAL_SUPPLEMENT
        else section.value.split("_", 1)[0],
        "requested_targets": [section.value],
    }
    payload["tool_requests"] = []
    payload["clarification"] = None
    return PrepareDecision.model_validate(payload)


def _safe_non_card_continuation(
    decision: PrepareDecision | None,
    state: PrepareGraphState,
    error: Exception,
) -> PrepareDecision | None:
    """Convert a repeatedly invalid early card into one grounded question.

    Qwen keeps ownership of the semantic proposal and later composes the public
    wording. This narrow safety branch only runs after the bounded repair budget
    is exhausted and the deterministic guard proves no canonical destination is
    available for attraction exploration.
    """

    if (
        decision is None
        or str(error) != "card action is not executable after semantic merge: other"
    ):
        return None
    payload = decision.model_dump(mode="json")
    payload["next_action"] = {
        "kind": PrepareActionKind.ASK_CLARIFICATION.value,
        "requested_targets": ["trip_destination"],
    }
    payload["tool_requests"] = []
    payload["clarification"] = {
        "target": "trip_destination",
        "why_blocking": "尚未取得可绑定的目的地城市，景点推荐需要先确认目的地。",
    }
    reply_goal = payload.get("reply_goal")
    if isinstance(reply_goal, dict):
        reply_goal["explain_next_step"] = "先确认目的地城市，再继续当地景点探索。"
    return PrepareDecision.model_validate(payload)


async def _assess_explicit_trip_basics(
    gateway: ModelGateway,
    state: PrepareGraphState,
    cancellation: ModelCancellation,
    *,
    require_entity_inventory: bool = False,
) -> TripBasicsAssessment | None:
    """Let Qwen route explicit OTHER basics into a small, reliable decision schema."""

    if state["decision_mode"] != "natural_text":
        return None
    basics = state["semantic_state"].trip_basics
    if (
        not require_entity_inventory
        and state["runtime_state"].current_section is not DiscoverySection.OTHER
        and not (
            basics.destination_name
            and basics.destination_canonical_id
            and not _trip_dates_complete(state["semantic_state"])
        )
    ):
        return None
    if not require_entity_inventory and all(
        (
            basics.destination_name,
            basics.destination_canonical_id,
            basics.start_date,
            basics.end_date,
            basics.duration_days,
            basics.travelers,
            basics.trip_goals,
        )
    ):
        return None
    repair_instruction: str | None = None
    for attempt in range(2):
        call_id: str | None = None
        try:
            result = await gateway.generate_structured(
                build_trip_basics_assessment_request(
                    user_text=state["turn_input"].user_text,
                    business_date=state["turn_input"].business_date,
                    semantic_state=state["semantic_state"],
                    runtime_state=state["runtime_state"],
                    recent_conversation=list(state["turn_input"].recent_conversation),
                    repair_instruction=repair_instruction,
                ),
                TripBasicsAssessment,
                cancellation=cancellation,
            )
            call_id = result.audit_call_id
            assessment = _normalize_intake_preference_quotes(
                result.value, state["turn_input"].user_text
            )
            _validate_intake_grounding(assessment, state["turn_input"].user_text)
            _validate_intake_date_basis(assessment, state)
            if assessment.date_range_resolution.basis in {
                "contextual_completion",
                "contextual_confirmation",
            } and not _has_active_date_range_followup(state):
                return None
            await record_model_call_annotation(
                gateway,
                call_id,
                "llm_business_guard",
                {
                    "business_guard_result": {
                        "status": "accepted",
                        "guard": "prepare_intake_grounding",
                    },
                    "accepted_or_rejected": "accepted",
                },
            )
            return assessment
        except ModelGatewayError as error:
            if error.code is ModelFailureCode.CANCELLED or error.requires_runtime_recovery:
                raise
            if error.code is ModelFailureCode.MALFORMED_RESPONSE and attempt == 0:
                repair_instruction = intake_schema_repair_instruction(error)
                continue
            if repair_instruction is not None:
                raise SemanticCompilationError(
                    "prepare_intake_repair_failed", "trip_basics_assessment"
                ) from error
            return None
        except SemanticCompilationError as error:
            repair_instruction = (
                f"{error.path}：{error.code}。该字段没有当前 user_text 的原文依据。"
                "只重新核验 TripBasicsAssessment；删除仅存在于历史的事实、地点及对应"
                " required_additional_targets，重新判断 has_additional_request。"
                "quote/query 必须复制当前输入的连续原文，不改写；保留本轮有依据的其他项。"
                "日期上下文只能供 date_range_resolution 使用。不要输出主决策或语义操作。"
            )
            if error.code == "prepare_general_constraint_requires_verbatim_fact":
                repair_instruction = (
                    "required_additional_targets.general_constraint 缺少对应的 requirement_facts。"
                    "若用户确有一般硬约束，补上完整原文 target=general_constraint + quote；"
                    "若只是同行人年龄、当天不住宿、已由交通或饮食字段覆盖的要求，"
                    "仅移除多标的一般约束类别，保留其正确领域的事实，不创造新要求。"
                )
            elif error.code == "prepare_transport_requires_verbatim_fact":
                repair_instruction = (
                    "required_additional_targets.transport_and_pace 没有当前原话依据。"
                    "若本轮明确表达交通或步调要求，在 requirement_facts 中逐字引用；"
                    "若本轮只有城市、天数和日期，额外需求清单应为空，"
                    "has_additional_request=false。不要把系统示例、长期默认或未来要询问的"
                    "偏好变成本轮已表达要求；同时删除其他无依据的类别，保留明确的基础信息。"
                )
            elif error.code == "prepare_derived_date_requires_duration":
                repair_instruction = (
                    "date_range_resolution.basis=start_plus_duration 与本轮来源不符："
                    "用户本轮没有给旅行天数，existing_trip_basics 也没有已保存的天数，"
                    "不能把两个日期之间计算出的天数反过来冒充用户给定的推算依据。"
                    "重新核对当前输入与激活 date_range 追问的相邻对话："
                    "若开始日和结束日均由用户明确给出，本轮只是补齐另一端点，"
                    "应为 ready + contextual_completion；若不能唯一组成范围则为 none。"
                    "不得猜测端点、强行升级推算日期或把日期补充标成其他领域需求。"
                )
            await record_model_call_annotation(
                gateway,
                call_id,
                "llm_business_guard",
                {
                    "business_guard_result": {
                        "status": "rejected",
                        "guard": "prepare_intake_grounding",
                        "reason": str(error),
                    },
                    "accepted_or_rejected": "rejected",
                    "failure_stage": "intake_grounding",
                    "failure_code": error.code,
                    "request_guard_feedback_full": {
                        "path": error.path,
                        "repair_instruction": repair_instruction,
                    },
                },
            )
            if attempt:
                raise
    raise AssertionError("unreachable intake retry state")


def _validate_intake_date_basis(intake: TripBasicsAssessment, state: PrepareGraphState) -> None:
    """A computed duration cannot masquerade as an input to the same computation."""

    if (
        intake.date_range_resolution.basis == "start_plus_duration"
        and not intake.explicit_duration_days
        and state["semantic_state"].trip_basics.duration_days is None
    ):
        raise SemanticCompilationError(
            "prepare_derived_date_requires_duration", "date_range_resolution.basis"
        )


def _validate_intake_grounding(intake: TripBasicsAssessment, user_text: str) -> None:
    if SemanticTargetV4.GENERAL_CONSTRAINT in intake.required_additional_targets and not any(
        fact.target == "general_constraint" for fact in intake.requirement_facts
    ):
        raise SemanticCompilationError(
            "prepare_general_constraint_requires_verbatim_fact",
            "required_additional_targets.general_constraint",
        )
    if SemanticTargetV4.TRANSPORT_AND_PACE in intake.required_additional_targets and not any(
        fact.target == "transport_and_pace" for fact in intake.requirement_facts
    ):
        # A bare category is not a user instruction. Reject it at extraction,
        # before the main decision is asked to invent operations to cover it.
        raise SemanticCompilationError(
            "prepare_transport_requires_verbatim_fact",
            "required_additional_targets.transport_and_pace",
        )
    for index, fact in enumerate(intake.requirement_facts):
        if fact.quote not in user_text:
            raise SemanticCompilationError(
                "prepare_requirement_quote_not_grounded", f"requirement_facts[{index}].quote"
            )
    for index, intent in enumerate(intake.named_entity_intents):
        if _normalize_place_reference(intent.query) not in _normalize_place_reference(user_text):
            raise SemanticCompilationError(
                "assessed_entity_not_grounded", f"named_entity_intents[{index}].query"
            )


def _normalize_intake_preference_quotes(
    intake: TripBasicsAssessment, user_text: str
) -> TripBasicsAssessment:
    """Remove a redundant polarity prefix only when the remainder is verbatim input."""

    prefixes = {
        "select": ("喜欢", "偏爱", "想看", "想去", "想吃", "希望"),
        "exclude": ("不喜欢", "不想看", "不想去", "不想吃", "避开", "不要"),
    }
    facts = []
    changed = False
    for fact in intake.requirement_facts:
        if not isinstance(fact, IntakePreferenceFact) or fact.quote in user_text:
            facts.append(fact)
            continue
        normalized = fact
        for prefix in prefixes[fact.disposition]:
            if not fact.quote.startswith(prefix):
                continue
            quote = fact.quote[len(prefix) :].lstrip("：:，,、 ")
            if quote and quote in user_text:
                normalized = fact.model_copy(update={"quote": quote})
                changed = True
                break
        facts.append(normalized)
    return intake.model_copy(update={"requirement_facts": facts}) if changed else intake


def _has_active_date_range_followup(state: PrepareGraphState) -> bool:
    pending = state["runtime_state"].pending_interaction
    return bool(
        pending is not None
        and pending.status.value == "active"
        and TRIP_DATE_RANGE_TARGET in pending.target_ids
    )


def _has_active_final_supplement_followup(state: PrepareGraphState) -> bool:
    pending = state["runtime_state"].pending_interaction
    return bool(
        state["decision_mode"] == "natural_text"
        and state["runtime_state"].current_section is DiscoverySection.FINAL_SUPPLEMENT
        and pending is not None
        and pending.status.value == "active"
        and pending.kind is PendingInteractionKind.FREE_TEXT_QUESTION
        and pending.section is DiscoverySection.FINAL_SUPPLEMENT
        and pending.target_ids == ["final_supplement"]
        and pending.based_on_state_version == state["runtime_state"].state_version
    )


def _explicitly_requests_task_book_generation(user_text: str) -> bool:
    """Recognize a narrow generation command only inside the active final prompt."""

    normalized = re.sub(r"[\s，。；：、,.!！?？;:]+", "", user_text)
    return bool(
        re.fullmatch(
            r"(?:(?:好的?|可以|行|那就|现在|请|直接))*"
            r"(?:"
            r"(?:开始|立即)?(?:生成|制作|整理)(?:旅行)?任务书"
            r"|(?:开始|立即)?(?:生成|生产)"
            r")"
            r"(?:吧|了|即可|就行)?",
            normalized,
        )
    )


async def _assess_final_supplement(
    gateway: ModelGateway,
    state: PrepareGraphState,
    cancellation: ModelCancellation,
) -> FinalSupplementAssessment | None:
    """Use Qwen to decide whether the user explicitly closed discovery."""

    if state["decision_mode"] != "natural_text":
        return None
    if state["runtime_state"].current_section is not DiscoverySection.FINAL_SUPPLEMENT:
        return None
    if state["runtime_state"].section_coverage[DiscoverySection.FINAL_SUPPLEMENT].status.value in {
        "complete",
        "not_applicable",
    }:
        return None
    try:
        result = await gateway.generate_structured(
            build_final_supplement_assessment_request(
                user_text=state["turn_input"].user_text,
                semantic_state=state["semantic_state"],
            ),
            FinalSupplementAssessment,
            cancellation=cancellation,
        )
    except ModelGatewayError as error:
        if error.code is ModelFailureCode.CANCELLED or error.requires_runtime_recovery:
            raise
        return None
    return result.value


async def _assess_pace_modification(
    gateway: ModelGateway,
    state: PrepareGraphState,
    cancellation: ModelCancellation,
) -> PaceModificationAssessment | None:
    """Narrow a task-book pace revision before the broad decision contract."""

    if state["decision_mode"] != "natural_text":
        return None
    runtime_state = state["runtime_state"]
    if runtime_state.current_section is not DiscoverySection.TASK_BOOK_REVIEW:
        return None
    candidate = runtime_state.task_book_candidate
    if candidate is None or candidate.status is not TaskBookStatus.AWAITING_CONFIRMATION:
        return None
    try:
        result = await gateway.generate_structured(
            build_pace_modification_assessment_request(
                user_text=state["turn_input"].user_text,
            ),
            PaceModificationAssessment,
            cancellation=cancellation,
        )
    except ModelGatewayError as error:
        if error.code is ModelFailureCode.CANCELLED or error.requires_runtime_recovery:
            raise
        return None
    return result.value


async def _assess_task_book_review_modification(
    gateway: ModelGateway,
    state: PrepareGraphState,
    cancellation: ModelCancellation,
) -> TaskBookReviewModificationAssessment | None:
    """Read revisions only while an unconfirmed task book owns the node."""

    if state["decision_mode"] != "natural_text":
        return state.get("task_book_review_assessment")
    runtime_state = state["runtime_state"]
    if runtime_state.current_section is not DiscoverySection.TASK_BOOK_REVIEW:
        return None
    candidate = runtime_state.task_book_candidate
    if candidate is None or candidate.status is not TaskBookStatus.AWAITING_CONFIRMATION:
        return None
    try:
        result = await gateway.generate_structured(
            build_task_book_review_assessment_request(
                user_text=state["turn_input"].user_text,
                semantic_state=state["semantic_state"],
                recent_conversation=list(state["turn_input"].recent_conversation),
            ),
            TaskBookReviewModificationAssessment,
            cancellation=cancellation,
        )
    except ModelGatewayError as error:
        if error.code is ModelFailureCode.CANCELLED or error.requires_runtime_recovery:
            raise
        return None
    return result.value


async def _extract_pace_requirement_decision(
    gateway: ModelGateway,
    state: PrepareGraphState,
    cancellation: ModelCancellation,
) -> tuple[PrepareDecision, str | None, bool]:
    """Extract one pace requirement without exposing the full Prepare union to Qwen."""

    repair_instruction: str | None = None
    for attempt in range(2):
        result_call_id: str | None = None
        try:
            result = await gateway.generate_structured(
                build_pace_requirement_extraction_request(
                    user_text=state["turn_input"].user_text,
                    repair_instruction=repair_instruction,
                ),
                PaceRequirementExtraction,
                cancellation=cancellation,
            )
            result_call_id = result.audit_call_id
            payload: dict[str, object] = {
                "semantic_operations": [
                    {
                        "operation_type": "add_conditional_requirement",
                        "domain": "transport",
                        "condition": result.value.condition,
                        "required_outcome": result.value.required_outcome,
                        "confidence": "high",
                    }
                ],
                "reply_goal": {
                    "acknowledge": [result.value.required_outcome],
                    "explain_next_step": "重新确认最终补充后生成新版任务书。",
                },
            }
            materialized = _materialize_runtime_decision_payload(
                payload,
                state,
                "full",
                "pace_requirement",
            )
            decision = PrepareDecision.model_validate(materialized)
            decision = _bind_authoritative_operation_sources(decision, state)
            _validate_decision(decision, state)
            await record_model_call_annotation(
                gateway,
                result_call_id,
                "llm_business_guard",
                {
                    "business_guard_result": {
                        "status": "accepted",
                        "guard": "prepare_pace_requirement_guard",
                    },
                    "accepted_or_rejected": "accepted",
                    "materialized_output": decision.model_dump(mode="json"),
                },
            )
            return decision, None, attempt > 0
        except ModelGatewayError as error:
            if error.code is ModelFailureCode.CANCELLED or error.requires_runtime_recovery:
                raise
            if attempt == 0:
                repair_instruction = (
                    "上一次结果不符合 PaceRequirementExtraction。只输出 condition 和"
                    " required_outcome 两个完整字符串；保留用户明确的数字上限，不输出其他字段。"
                )
                continue
            return (
                _fallback_decision(state["runtime_state"], state["semantic_state"].state_version),
                f"decision_{error.code.value}",
                True,
            )
        except (PrepareGraphError, ValueError) as error:
            await record_model_call_annotation(
                gateway,
                result_call_id,
                "llm_business_guard",
                {
                    "business_guard_result": {
                        "status": "rejected",
                        "guard": "prepare_pace_requirement_guard",
                        "reason": str(error),
                    },
                    "accepted_or_rejected": "rejected",
                    "failure_stage": "business_guard",
                    "failure_code": "decision_validation_failed",
                },
            )
            logger.warning(
                "Prepare pace extraction validation failure: attempt=%s issue=%s",
                attempt + 1,
                str(error),
            )
            if attempt == 0:
                repair_instruction = (
                    "提取结果未通过状态合同。condition 必须说明适用范围，required_outcome 必须"
                    "完整保留用户明确的节奏、每日景点上限和休息要求，不得加入其他旅行内容。"
                )
                continue
            return (
                _fallback_decision(state["runtime_state"], state["semantic_state"].state_version),
                "decision_validation_failed",
                True,
            )
    raise AssertionError("unreachable pace extraction retry state")


async def _extract_compound_trip_intake_decision(
    gateway: ModelGateway,
    state: PrepareGraphState,
    cancellation: ModelCancellation,
) -> tuple[PrepareDecision, str | None, bool]:
    """Extract a complex first turn without asking Qwen to emit the full decision union."""

    turn_input = state["turn_input"]
    destination = state["semantic_state"].trip_basics.destination_name
    if destination is None:
        raise PrepareGraphError("compound trip intake requires a destination")
    source_refs = sorted({turn_input.user_message_ref, *turn_input.signed_source_refs})
    repair_instruction: str | None = None
    for attempt in range(2):
        result_call_id: str | None = None
        try:
            result = await gateway.generate_structured(
                build_compound_trip_intake_request(
                    user_text=turn_input.user_text,
                    repair_instruction=repair_instruction,
                ),
                CompoundTripIntakeExtraction,
                cancellation=cancellation,
            )
            result_call_id = result.audit_call_id
            semantic_operations: list[dict[str, object]] = [
                {
                    "operation_type": "set_not_applicable",
                    "local_operation_key": str(
                        uuid5(
                            NAMESPACE_URL,
                            f"iter-ai:lodging-not-applicable:{turn_input.turn_id}",
                        )
                    ),
                    "target": "lodging_area",
                    "reason": result.value.lodging_not_applicable_reason,
                    "source_refs": source_refs,
                    "confidence": "high",
                }
            ]
            for index, requirement in enumerate(result.value.transport_requirements):
                semantic_operations.append(
                    {
                        "operation_type": "add_conditional_requirement",
                        "local_operation_key": str(
                            uuid5(
                                NAMESPACE_URL,
                                f"iter-ai:compound-transport:{turn_input.turn_id}:{index}",
                            )
                        ),
                        "target": "transport_and_pace",
                        "domain": "transport",
                        "condition": "本次旅行",
                        "required_outcome": requirement,
                        "source_refs": source_refs,
                        "confidence": "high",
                    }
                )
            for index, requirement in enumerate(result.value.dining_requirements):
                semantic_operations.append(
                    {
                        "operation_type": "add_conditional_requirement",
                        "local_operation_key": str(
                            uuid5(
                                NAMESPACE_URL,
                                f"iter-ai:compound-dining:{turn_input.turn_id}:{index}",
                            )
                        ),
                        "target": "dining_requirement",
                        "domain": "dining",
                        "condition": "本次旅行",
                        "required_outcome": requirement,
                        "source_refs": source_refs,
                        "confidence": "high",
                    }
                )
            if result.value.delegate_dining_by_route:
                semantic_operations.append(
                    {
                        "operation_type": "set_delegation_scope",
                        "local_operation_key": str(
                            uuid5(
                                NAMESPACE_URL,
                                f"iter-ai:compound-dining-delegation:{turn_input.turn_id}",
                            )
                        ),
                        "target": "dining_entity",
                        "domain": "dining",
                        "delegated_targets": ["dining_specific"],
                        "boundary_refs": [],
                        "source_refs": source_refs,
                        "confidence": "high",
                    }
                )
            payload: dict[str, object] = {
                "trip_basics": {
                    "operation_type": "set_trip_basics",
                    "domain": "general",
                    "travelers": result.value.travelers,
                    "trip_goals": result.value.trip_goals,
                    "confidence": "high",
                },
                "semantic_operations": semantic_operations,
                "next_action": {
                    "kind": "use_tool",
                    "domain": "attraction",
                    "requested_targets": ["attraction_entity"],
                },
                "tool_requests": [
                    {
                        "capability": "resolve_place",
                        "request_id": str(
                            uuid5(
                                NAMESPACE_URL,
                                f"iter-ai:compound-intake-place:{turn_input.turn_id}",
                            )
                        ),
                        "depends_on": [],
                        "purpose": "validate_operation",
                        "required": True,
                        "query": result.value.place_query,
                        "city": destination,
                    }
                ],
                "reply_goal": {
                    "acknowledge": [
                        *result.value.trip_goals,
                        result.value.lodging_not_applicable_reason,
                    ],
                    "explain_next_step": "先核验用户指定的核心景点，再继续景点偏好探索。",
                },
            }
            materialized = _materialize_runtime_decision_payload(
                payload,
                state,
                "full",
                "trip_basics_with_additions",
            )
            decision = PrepareDecision.model_validate(materialized)
            decision = _bind_authoritative_operation_sources(decision, state)
            _validate_decision(decision, state)
            await record_model_call_annotation(
                gateway,
                result_call_id,
                "llm_business_guard",
                {
                    "business_guard_result": {
                        "status": "accepted",
                        "guard": "prepare_compound_intake_guard",
                    },
                    "accepted_or_rejected": "accepted",
                    "materialized_output": decision.model_dump(mode="json"),
                },
            )
            return decision, None, attempt > 0
        except ModelGatewayError as error:
            if error.code is ModelFailureCode.CANCELLED or error.requires_runtime_recovery:
                raise
            if attempt == 0:
                repair_instruction = (
                    "上一次结果不符合 CompoundTripIntakeExtraction。只输出合同字段；"
                    "同行人、目标、不住宿原因、核心地点及交通餐饮要求都必须来自当前用户原话。"
                )
                continue
            return (
                _fallback_decision(state["runtime_state"], state["semantic_state"].state_version),
                f"decision_{error.code.value}",
                True,
            )
        except (PrepareGraphError, ValueError) as error:
            await record_model_call_annotation(
                gateway,
                result_call_id,
                "llm_business_guard",
                {
                    "business_guard_result": {
                        "status": "rejected",
                        "guard": "prepare_compound_intake_guard",
                        "reason": str(error),
                    },
                    "accepted_or_rejected": "rejected",
                    "failure_stage": "business_guard",
                    "failure_code": "decision_validation_failed",
                },
            )
            logger.warning(
                "Prepare compound intake validation failure: attempt=%s issue=%s",
                attempt + 1,
                str(error),
            )
            if attempt == 0:
                repair_instruction = (
                    "提取结果未通过状态或地点合同。place_query 必须逐字复制唯一核心景点全名；"
                    "旅行目标必须是完整词组，不住宿原因必须忠实保留。"
                )
                continue
            return (
                _fallback_decision(state["runtime_state"], state["semantic_state"].state_version),
                "decision_validation_failed",
                True,
            )
    raise AssertionError("unreachable compound intake retry state")


def _validate_turn_input(value: PrepareTurnInput) -> None:
    if str(value.trip_id) != value.semantic_state.trip_id:
        raise PrepareGraphError("turn input semantic state belongs to another trip")
    if value.runtime_state.trip_id != value.semantic_state.trip_id:
        raise PrepareGraphError("turn input runtime state belongs to another trip")
    if value.expected_state_version != value.semantic_state.state_version:
        raise PrepareGraphError("turn input state version is stale")
    if value.runtime_state.state_version != value.expected_state_version:
        raise PrepareGraphError("turn input dual states do not share the expected version")


def _validate_decision(decision: PrepareDecision, state: PrepareGraphState) -> None:
    if decision.based_on_state_version != state["semantic_state"].state_version:
        raise PrepareGraphError("decision is based on a stale working version")
    for index, wrapped in enumerate(decision.semantic_operations):
        if wrapped.root.operation_type == "confirm_task_book":
            raise SemanticCompilationError(
                "prepare_confirmation_requires_signed_interaction",
                f"semantic_operations[{index}]",
                recovery="clarify",
            )
        if (
            wrapped.root.operation_type == "confirm_final_supplement"
            and not _has_active_final_supplement_followup(state)
        ):
            raise SemanticCompilationError(
                "prepare_final_supplement_not_active",
                f"semantic_operations[{index}]",
                recovery="redecide_action",
            )
    if state["turn_input"].user_event_kind == "task_book_confirmation":
        if decision.semantic_operations:
            raise PrepareGraphError("task book confirmation cannot add unrelated semantics")
        if decision.next_action.kind is not PrepareActionKind.REPLY_ONLY:
            raise PrepareGraphError("task book confirmation requires a reply-only acknowledgement")
    required_card = state["turn_input"].required_card_section
    if (
        required_card is not None
        and not state.get("card_action_observations")
        and _trip_dates_complete(state["semantic_state"])
    ):
        expected_action = (
            PrepareActionKind.SHOW_SPECIFIC_CARD
            if required_card
            in {
                DiscoverySection.ATTRACTION_SPECIFIC,
                DiscoverySection.DINING_SPECIFIC,
            }
            else PrepareActionKind.SHOW_PREFERENCE_CARD
        )
        if decision.semantic_operations:
            raise PrepareGraphError("card refresh cannot add semantic operations")
        if decision.next_action.kind is not expected_action:
            raise PrepareGraphError("card refresh must issue a replacement card")
    required_post_update = _required_post_update_action(state)
    if required_post_update != "none":
        expected_post_update_action = {
            "reply_only": PrepareActionKind.REPLY_ONLY,
            "show_preference_card": PrepareActionKind.SHOW_PREFERENCE_CARD,
            "show_specific_card": PrepareActionKind.SHOW_SPECIFIC_CARD,
            "final_supplement": PrepareActionKind.FINAL_SUPPLEMENT,
        }[required_post_update]
        if decision.semantic_operations or decision.tool_requests:
            raise PrepareGraphError("card answer next action cannot add semantics or tools")
        if decision.next_action.kind is not expected_post_update_action:
            raise PrepareGraphError("card answer must continue the guarded discovery mainline")
    if _is_optional_trip_preferences_response(state) and not _fact_capabilities(state):
        action = decision.next_action.kind
        repeats_optional_question = bool(
            action is PrepareActionKind.ASK_CLARIFICATION
            and OPTIONAL_TRIP_PREFERENCES_TARGET in decision.next_action.requested_targets
        )
        if repeats_optional_question or action not in {
            PrepareActionKind.SHOW_PREFERENCE_CARD,
            PrepareActionKind.SHOW_SPECIFIC_CARD,
            PrepareActionKind.USE_TOOL,
            PrepareActionKind.ASK_CLARIFICATION,
        }:
            raise PrepareGraphError(
                "optional trip preferences response requires an executable attraction interaction"
            )
    allowed = state["allowed_source_refs"]
    for wrapped in decision.semantic_operations:
        if not set(wrapped.root.source_refs).issubset(allowed):
            raise PrepareGraphError("decision operation uses unauthorized evidence")
    prior_ids = state["prior_tool_request_ids"]
    if any(item.root.request_id in prior_ids for item in decision.tool_requests):
        raise PrepareGraphError("decision repeated an already executed tool request ID")
    if state.get("tool_round", 0) >= 2 and decision.next_action.kind is PrepareActionKind.USE_TOOL:
        raise PrepareGraphError("decision exceeded the two-round tool limit")
    known_entity_refs = state["known_entity_refs"]
    for request_index, tool_request in enumerate(decision.tool_requests):
        request = tool_request.root
        if isinstance(request, ResolvePlaceRequest):
            _validate_resolve_query_kind(request, state, request_index)
            if not _resolve_query_is_grounded(request, state):
                raise PrepareGraphError(
                    "resolve_place query is not grounded in the current user message"
                )
            if _normalize_place_reference(request.query) in _executed_resolve_queries(state):
                raise PrepareGraphError("resolve_place query was already resolved in this turn")
        if isinstance(
            request,
            (
                PlaceFactsRequest,
                OpeningHoursRequest,
                TicketAvailabilityRequest,
                PlaceProductsRequest,
            ),
        ):
            references = set(request.canonical_entity_ids)
            reference_field = "canonical_entity_ids"
        elif isinstance(request, SpatialRoutesRequest):
            references = {request.origin_ref, request.destination_ref}
            reference_field = (
                "origin_ref" if request.origin_ref not in known_entity_refs else "destination_ref"
            )
        else:
            references = set()
            reference_field = ""
        if not references.issubset(known_entity_refs):
            raise SemanticCompilationError(
                "prepare_unknown_tool_entity", f"tool_requests[{request_index}].{reference_field}"
            )

    observations = [item.observation for item in state.get("observations", [])]
    required_next = _required_next_tool_capability(state, observations)
    if required_next is not None:
        if decision.next_action.kind is not PrepareActionKind.USE_TOOL:
            raise PrepareGraphError("current fact question requires a grounded tool action")
        requested_capabilities = {item.root.capability.value for item in decision.tool_requests}
        if requested_capabilities != {required_next}:
            raise PrepareGraphError(
                f"current fact question requires tool capability {required_next}"
            )
    _validate_card_action_after_merge(decision, state)
    _validate_task_book_review_modification_decision(decision, state)


def _validate_card_observation_decision(
    decision: PrepareDecision,
    state: PrepareGraphState,
) -> None:
    observations = state.get("card_action_observations", [])
    if not observations:
        return
    observation = observations[-1]
    if decision.semantic_operations or decision.tool_requests:
        raise PrepareGraphError("card observation continuation cannot add semantics or tools")
    if decision.next_action.kind not in observation.allowed_next_actions:
        raise PrepareGraphError("card observation continuation selected a forbidden action")
    if decision.next_action.kind is PrepareActionKind.ASK_CLARIFICATION:
        requested = set(decision.next_action.requested_targets)
        allowed_targets = set(observation.user_input_targets)
        if not requested or not requested <= allowed_targets:
            raise PrepareGraphError("card clarification uses an unsupported input target")
        if decision.clarification is None or decision.clarification.target not in allowed_targets:
            raise PrepareGraphError("card clarification contract uses an unsupported target")


def _validate_task_book_observation_decision(
    decision: PrepareDecision,
    state: PrepareGraphState,
) -> None:
    observations = state.get("task_book_action_observations", [])
    if not observations:
        return
    observation = observations[-1]
    if decision.semantic_operations or decision.tool_requests:
        raise PrepareGraphError("task-book recovery cannot add semantics or tools")
    if decision.next_action.kind not in observation.allowed_next_actions:
        raise PrepareGraphError("task-book recovery selected a forbidden action")
    if decision.next_action.requested_targets != observation.user_input_targets:
        raise PrepareGraphError("task-book recovery uses an unsupported input target")


def _recoverable_task_book_action_observation(
    blocking_reasons: tuple[str, ...],
    *,
    turn_id: UUID,
    observation_index: int,
) -> TaskBookActionObservation | None:
    recoverable_reasons = {
        "section_incomplete:final_supplement",
        "final_supplement_incomplete",
    }
    if not blocking_reasons or not set(blocking_reasons) <= recoverable_reasons:
        return None
    if "final_supplement_incomplete" not in blocking_reasons:
        return None
    return TaskBookActionObservation(
        observation_id=str(
            uuid5(
                NAMESPACE_URL,
                f"v4-task-book-action-observation:{turn_id}:{observation_index}",
            )
        ),
        action=PrepareActionKind.GENERATE_TASK_BOOK,
        status="blocked",
        failure_code="final_supplement_incomplete",
        blocking_reasons=list(blocking_reasons),
        allowed_next_actions=[PrepareActionKind.FINAL_SUPPLEMENT],
        user_input_targets=["final_supplement"],
        safe_summary="生成任务书前仍需确认用户已经结束最终补充。",
    )


def _recoverable_card_action_observation(
    error: Exception,
    *,
    section: DiscoverySection,
    turn_id: UUID,
    observation_index: int,
    failure_code: str,
) -> CardActionObservation | None:
    missing_fields: list[Literal["destination", "duration_days"]] = []
    input_targets: list[str]
    if isinstance(error, CardGenerationContextRequired):
        missing_fields = list(error.missing_fields)
        status: Literal["needs_input", "unavailable"] = "needs_input"
        allowed_actions = [PrepareActionKind.ASK_CLARIFICATION]
        input_targets = list(missing_fields)
        if missing_fields == ["duration_days"]:
            safe_summary = "生成当前具体候选卡还需要确认一至五天的游玩天数。"
        elif missing_fields == ["destination"]:
            safe_summary = "生成当前卡片还需要确认旅行目的地。"
        else:
            safe_summary = "生成当前具体候选卡还需要确认目的地和游玩天数。"
    elif isinstance(error, (ModelGatewayError, RecallPlanError)) or (
        isinstance(error, CardGenerationError) and error.recoverable
    ):
        status = "unavailable"
        allowed_actions = [
            PrepareActionKind.ASK_CLARIFICATION,
            PrepareActionKind.REPLY_ONLY,
        ]
        input_targets = _card_observation_input_targets(section)
        safe_summary = "当前卡片能力没有取得足够的可信候选，可请用户补充具体需求。"
    else:
        return None
    return CardActionObservation(
        observation_id=str(
            uuid5(
                NAMESPACE_URL,
                f"v4-card-action-observation:{turn_id}:{section.value}:{observation_index}",
            )
        ),
        section=section,
        status=status,
        failure_code=failure_code,
        missing_fields=missing_fields,
        allowed_next_actions=allowed_actions,
        user_input_targets=input_targets,
        safe_summary=safe_summary,
    )


def _card_observation_input_targets(section: DiscoverySection) -> list[str]:
    targets = {
        DiscoverySection.ATTRACTION_PREFERENCE: ["attraction_preference"],
        DiscoverySection.ATTRACTION_SPECIFIC: ["attraction_specific"],
        DiscoverySection.DINING_PREFERENCE: ["dining_preference", "dining_requirement"],
        DiscoverySection.DINING_SPECIFIC: ["dining_specific", "dining_requirement"],
        DiscoverySection.LODGING_AREA_PREFERENCE: ["lodging_area", "lodging_booking"],
        DiscoverySection.LODGING_CLASS_PREFERENCE: ["lodging_class", "lodging_booking"],
    }
    return targets.get(section, [section.value])


def _validate_trip_intake_transition_action(
    decision: PrepareDecision,
    required_action: Literal[
        "none",
        "ask_trip_dates",
        "ask_optional_preferences",
        "show_attraction_preferences",
    ],
    state: PrepareGraphState | None = None,
) -> None:
    """Keep the first destination transition deterministic without owning wording."""

    if required_action == "none" or decision.next_action.kind is PrepareActionKind.USE_TOOL:
        return
    if required_action == "ask_trip_dates":
        if (
            decision.next_action.kind is not PrepareActionKind.ASK_CLARIFICATION
            or decision.next_action.requested_targets != [TRIP_DATE_RANGE_TARGET]
            or decision.clarification is None
            or decision.clarification.target != TRIP_DATE_RANGE_TARGET
        ):
            raise PrepareGraphError("trip intake requires the date range follow-up")
        return
    if required_action == "ask_optional_preferences":
        if (
            decision.next_action.kind is not PrepareActionKind.ASK_CLARIFICATION
            or decision.next_action.requested_targets != [OPTIONAL_TRIP_PREFERENCES_TARGET]
            or decision.clarification is None
            or decision.clarification.target != OPTIONAL_TRIP_PREFERENCES_TARGET
        ):
            raise PrepareGraphError("trip intake requires the optional preferences follow-up")
        return
    section = (
        _preview_card_runtime(decision, state).current_section
        if state is not None
        else DiscoverySection.ATTRACTION_PREFERENCE
    )
    if (
        decision.next_action.kind is not _default_action_for_section(section)
        or getattr(decision.next_action.domain, "value", decision.next_action.domain)
        != "attraction"
        or section.value not in decision.next_action.requested_targets
        or section
        not in {
            DiscoverySection.ATTRACTION_PREFERENCE,
            DiscoverySection.ATTRACTION_SPECIFIC,
        }
    ):
        raise PrepareGraphError("trip intake requires the attraction preference card")


def _preview_card_runtime(
    decision: PrepareDecision, state: PrepareGraphState
) -> DiscoveryRuntimeState:
    """Preview the real merge without committing or advancing a version."""
    seen_keys = {
        item.proposal.root.local_operation_key for item in state.get("accepted_operations", [])
    }
    proposals = [
        item
        for item in decision.semantic_operations
        if item.root.local_operation_key not in seen_keys
    ]
    merged = merge_v4_operations(
        state["semantic_state"],
        state["runtime_state"],
        proposals,
        turn_id=state["turn_input"].turn_id,
        allowed_source_refs=state["allowed_source_refs"],
        known_entity_refs=state["known_entity_refs"],
        advance_version=False,
    )
    guarded = recalculate_section_coverage(
        merged.semantic_state,
        merged.runtime_state,
        merged.accepted_operations,
        affected_sections=merged.affected_sections,
    )
    preview_runtime, _ = _apply_task_book_review_card_reopen(
        guarded.runtime_state,
        state.get("task_book_review_assessment"),
        decision=decision,
    )
    return preview_runtime


def _align_card_action_after_merge(
    decision: PrepareDecision, state: PrepareGraphState
) -> PrepareDecision:
    """Compile the card subtype, not the model's semantic or cross-domain choice.

    Qwen still decides to offer a card. Only a same-domain preference/specific
    mismatch is mechanical: SectionGuard already owns that completion boundary.
    Never turn a reply, tool, question, or another domain into a card.
    """
    card_kinds = {PrepareActionKind.SHOW_PREFERENCE_CARD, PrepareActionKind.SHOW_SPECIFIC_CARD}
    if decision.next_action.kind not in card_kinds:
        return decision
    section = _preview_card_runtime(decision, state).current_section
    expected = _default_action_for_section(section)
    domain = section.value.split("_", 1)[0]
    if expected not in card_kinds or getattr(decision.next_action.domain, "value", None) != domain:
        return decision
    payload = decision.model_dump(mode="json")
    payload["next_action"] = {
        "kind": expected.value,
        "domain": domain,
        "requested_targets": [section.value],
    }
    return PrepareDecision.model_validate(payload)


def _validate_card_action_after_merge(decision: PrepareDecision, state: PrepareGraphState) -> None:
    """Guard the action against the actual post-merge section."""
    action = decision.next_action.kind
    concrete_continuation = (
        state["decision_mode"] == "bounded_redecide"
        and _continuation_semantics(state) == "concrete_entity"
    )
    if concrete_continuation and action in {
        PrepareActionKind.ASK_CLARIFICATION,
        PrepareActionKind.USE_TOOL,
    }:
        # Fact queries and ambiguity are legal intermediate actions. The tool
        # whitelist, references, dependencies and round budget are checked above;
        # don't simultaneously require a card while facts are still being fetched.
        return
    if not concrete_continuation and action not in {
        PrepareActionKind.SHOW_PREFERENCE_CARD,
        PrepareActionKind.SHOW_SPECIFIC_CARD,
    }:
        return
    section = _preview_card_runtime(decision, state).current_section
    expected = _default_action_for_section(section)
    if concrete_continuation:
        if expected is None:
            if action in {
                PrepareActionKind.SHOW_PREFERENCE_CARD,
                PrepareActionKind.SHOW_SPECIFIC_CARD,
            }:
                raise PrepareGraphError(
                    f"card action is not executable after semantic merge: {section.value}"
                )
            return
        if action is not expected:
            raise PrepareGraphError(
                f"concrete entity continuation requires next action {expected.value}"
            )
        return
    if (
        action is not expected
        or getattr(decision.next_action.domain, "value", None) != section.value.split("_", 1)[0]
        or decision.next_action.requested_targets != [section.value]
    ):
        raise PrepareGraphError(
            f"card action is not executable after semantic merge: {section.value}"
        )


def _validate_task_book_review_modification_decision(
    decision: PrepareDecision,
    state: PrepareGraphState,
) -> None:
    """Make the review-node assessment additive rather than a field whitelist."""

    assessment = state.get("task_book_review_assessment")
    if assessment is None or not assessment.is_change_request:
        return

    accepted_targets = {item.proposal.root.target for item in state.get("accepted_operations", [])}
    decision_targets = {item.root.target for item in decision.semantic_operations}
    recorded_targets = accepted_targets | decision_targets
    missing_targets = set(assessment.required_targets) - recorded_targets
    if decision.next_action.kind is PrepareActionKind.USE_TOOL:
        # A grounding round may precede every write in a compound revision.
        # Keep all assessed targets in graph state and enforce them together on
        # the bounded continuation instead of requiring a partial early merge.
        return
    if missing_targets:
        raise PrepareGraphError(
            "task-book revision dropped explicit semantic targets: "
            + ",".join(sorted(item.value for item in missing_targets))
        )

    requested_card = assessment.requested_card_section
    if not assessment.required_targets and requested_card is None:
        if decision.next_action.kind is not PrepareActionKind.ASK_CLARIFICATION:
            raise PrepareGraphError("task-book revision needs a concrete modification target")
        return
    _validate_task_book_review_concrete_choice(decision, state)

    if requested_card is not None:
        expected_action = (
            PrepareActionKind.SHOW_SPECIFIC_CARD
            if requested_card
            in {
                DiscoverySection.ATTRACTION_SPECIFIC,
                DiscoverySection.DINING_SPECIFIC,
            }
            else PrepareActionKind.SHOW_PREFERENCE_CARD
        )
        if decision.next_action.kind is not expected_action:
            raise PrepareGraphError(
                f"task-book revision requires card action {expected_action.value}"
            )
        if requested_card.value not in decision.next_action.requested_targets:
            raise PrepareGraphError("task-book revision card action uses the wrong target")
        expected_domain = {
            DiscoverySection.ATTRACTION_PREFERENCE: "attraction",
            DiscoverySection.ATTRACTION_SPECIFIC: "attraction",
            DiscoverySection.DINING_PREFERENCE: "dining",
            DiscoverySection.DINING_SPECIFIC: "dining",
            DiscoverySection.LODGING_AREA_PREFERENCE: "lodging",
            DiscoverySection.LODGING_CLASS_PREFERENCE: "lodging",
        }[requested_card]
        if getattr(decision.next_action.domain, "value", decision.next_action.domain) != (
            expected_domain
        ):
            raise PrepareGraphError("task-book revision card action uses the wrong domain")
        return

    if not recorded_targets:
        return
    merged = merge_v4_operations(
        state["semantic_state"],
        state["runtime_state"],
        [
            item
            for item in decision.semantic_operations
            if item.root.local_operation_key
            not in {
                accepted.proposal.root.local_operation_key
                for accepted in state.get("accepted_operations", [])
            }
        ],
        turn_id=state["turn_input"].turn_id,
        allowed_source_refs=state["allowed_source_refs"],
        known_entity_refs=state["known_entity_refs"],
        advance_version=False,
    )
    guarded = recalculate_section_coverage(
        merged.semantic_state,
        merged.runtime_state,
        merged.accepted_operations,
        affected_sections=merged.affected_sections,
    )
    post_update_action = _default_action_for_section(guarded.runtime_state.current_section)
    if post_update_action is not None and decision.next_action.kind is not post_update_action:
        raise PrepareGraphError(
            f"task-book revision requires next action {post_update_action.value}"
        )


def _validate_task_book_review_concrete_choice(
    decision: PrepareDecision,
    state: PrepareGraphState,
) -> None:
    required_choice = _required_concrete_choice(state)
    if required_choice == "none":
        return
    operations = [
        item.proposal.root.model_dump(mode="json") for item in state.get("accepted_operations", [])
    ] + [item.root.model_dump(mode="json") for item in decision.semantic_operations]
    expected = {
        "attraction_must": ("select_concrete_entity", "attraction", "must"),
        "attraction_want": ("select_concrete_entity", "attraction", "want"),
        "attraction_if_convenient": (
            "select_concrete_entity",
            "attraction",
            "if_convenient",
        ),
        "attraction_avoid": ("exclude_concrete_entity", "attraction", None),
        "dining_destination": ("select_concrete_entity", "dining", "destination"),
        "dining_if_convenient": (
            "select_concrete_entity",
            "dining",
            "if_convenient",
        ),
        "dining_avoid": ("exclude_concrete_entity", "dining", None),
    }[required_choice]
    if any(
        operation.get("operation_type") == expected[0]
        and operation.get("domain") == expected[1]
        and (expected[2] is None or operation.get("disposition") == expected[2])
        for operation in operations
    ):
        return
    raise PrepareGraphError("task-book revision concrete choice changed the user disposition")


def _apply_task_book_review_card_reopen(
    runtime_state: DiscoveryRuntimeState,
    assessment: TaskBookReviewModificationAssessment | None,
    *,
    decision: PrepareDecision,
) -> tuple[DiscoveryRuntimeState, tuple[str, ...]]:
    """Supersede the reviewed book and reopen one explicitly requested card."""

    if decision.next_action.kind not in {
        PrepareActionKind.SHOW_PREFERENCE_CARD,
        PrepareActionKind.SHOW_SPECIFIC_CARD,
    }:
        return runtime_state, ()

    requested = (
        assessment.requested_card_section
        if assessment is not None and assessment.is_change_request
        else None
    )
    if requested is None and decision.section_proposal.kind is SectionProposalKind.REOPEN:
        sections = decision.section_proposal.reopen_sections
        if len(sections) != 1:
            raise SemanticCompilationError(
                "prepare_reopen_requires_one_card", "section_proposal.reopen_sections"
            )
        requested = sections[0]
        current_index = (
            DISCOVERY_ORDER.index(runtime_state.current_section)
            if runtime_state.current_section in DISCOVERY_ORDER
            else len(DISCOVERY_ORDER)
        )
        if (
            requested not in DISCOVERY_ORDER[1:-1]
            or DISCOVERY_ORDER.index(requested) > current_index
        ):
            raise SemanticCompilationError(
                "prepare_reopen_cannot_skip_sections", "section_proposal.reopen_sections"
            )
    if requested is None:
        return runtime_state, ()
    coverage = {
        section: value.model_copy(deep=True)
        for section, value in runtime_state.section_coverage.items()
    }
    for section in (requested, DiscoverySection.FINAL_SUPPLEMENT):
        coverage[section] = SectionCoverage(
            status=CoverageStatus.REOPENED,
            required_targets=[section.value],
        )

    invalidated: tuple[str, ...] = ()
    pending = runtime_state.pending_interaction
    if pending is not None:
        invalidated = (pending.interaction_id,)
        pending = None

    task_book_candidate = runtime_state.task_book_candidate
    if task_book_candidate is not None:
        task_book_candidate = task_book_candidate.model_copy(
            update={
                "status": TaskBookStatus.SUPERSEDED,
                "value": task_book_candidate.value.model_copy(
                    update={
                        "status": TaskBookStatus.SUPERSEDED,
                        "confirmed_at": None,
                    },
                    deep=True,
                ),
            },
            deep=True,
        )
    return (
        runtime_state.model_copy(
            update={
                "current_section": requested,
                "section_coverage": coverage,
                "pending_interaction": pending,
                "task_book_candidate": task_book_candidate,
            },
            deep=True,
        ),
        invalidated,
    )


def _default_action_for_section(section: DiscoverySection) -> PrepareActionKind | None:
    return {
        DiscoverySection.ATTRACTION_PREFERENCE: PrepareActionKind.SHOW_PREFERENCE_CARD,
        DiscoverySection.ATTRACTION_SPECIFIC: PrepareActionKind.SHOW_SPECIFIC_CARD,
        DiscoverySection.DINING_PREFERENCE: PrepareActionKind.SHOW_PREFERENCE_CARD,
        DiscoverySection.DINING_SPECIFIC: PrepareActionKind.SHOW_SPECIFIC_CARD,
        DiscoverySection.LODGING_AREA_PREFERENCE: PrepareActionKind.SHOW_PREFERENCE_CARD,
        DiscoverySection.LODGING_CLASS_PREFERENCE: PrepareActionKind.SHOW_PREFERENCE_CARD,
        DiscoverySection.FINAL_SUPPLEMENT: PrepareActionKind.FINAL_SUPPLEMENT,
    }.get(section)


def _bind_authoritative_operation_sources(
    decision: PrepareDecision,
    state: PrepareGraphState,
) -> PrepareDecision:
    if not decision.semantic_operations:
        return decision
    turn_input = state["turn_input"]
    message_sources = {
        turn_input.user_message_ref,
        *turn_input.signed_source_refs,
    }
    choices = _resolution_choices(state)
    pending_queries = {
        request.query
        for wrapped in decision.tool_requests
        if isinstance((request := wrapped.root), ResolvePlaceRequest)
        and request.purpose.value == "validate_operation"
    }
    payload = decision.model_dump(mode="json")
    for index, raw_operation in enumerate(payload["semantic_operations"]):
        sources = set(message_sources)
        concrete = raw_operation["operation_type"] in {
            "select_concrete_entity",
            "exclude_concrete_entity",
        }
        if concrete or (
            raw_operation["operation_type"] == "set_existing_booking"
            and raw_operation.get("canonical_entity_id") is not None
        ):
            name = raw_operation.get("display_name") if concrete else None
            matching = [
                choice
                for choice in choices
                if len(choice.entity_refs) == 1
                and not is_city_entity_ref(choice.entity_refs[0])
                and (
                    _normalize_place_reference(choice.query) == _normalize_place_reference(name)
                    if isinstance(name, str)
                    else choice.query in raw_operation.get("user_description", "")
                )
            ]
            matching_refs = {choice.entity_refs[0] for choice in matching}
            if len(matching_refs) == 1:
                raw_operation["canonical_entity_id"] = next(iter(matching_refs))
            elif concrete and name not in pending_queries:
                # A previously accepted POI is reusable only for that named object
                # and domain. A bare known ID is not proof for another label.
                domain = raw_operation["domain"]
                semantic = state["semantic_state"]
                prior = (
                    (*semantic.attractions.concrete_intents, *semantic.attractions.exclusions)
                    if domain == "attraction"
                    else (*semantic.dining.concrete_restaurant_intents, *semantic.dining.exclusions)
                )
                entity_ref = raw_operation["canonical_entity_id"]
                if entity_ref in state["known_entity_refs"] and not any(
                    item.canonical_entity_id == entity_ref
                    and not is_city_entity_ref(entity_ref)
                    and _normalize_place_reference(item.display_name)
                    == _normalize_place_reference(str(name))
                    for item in prior
                ):
                    raise SemanticCompilationError(
                        "entity_identity_not_grounded",
                        f"semantic_operations[{index}].canonical_entity_id",
                    )
            entity_ref = raw_operation.get("canonical_entity_id")
            sources.update(
                source
                for item in state.get("observations", [])
                if entity_ref in item.observation.entity_refs
                for source in item.observation.source_refs
            )
        raw_operation["source_refs"] = sorted(sources)
    return PrepareDecision.model_validate(payload)


def _bind_registered_trip_destination(
    decision: PrepareDecision,
    state: PrepareGraphState,
) -> PrepareDecision:
    """Bind a Qwen-extracted city name to the server's supported-city identity."""

    basics_operations = [
        item.root
        for item in decision.semantic_operations
        if isinstance(item.root, SetTripBasicsOperation) and item.root.destination_name is not None
    ]
    if not basics_operations:
        return decision
    payload = decision.model_dump(mode="json")
    changed = False
    for raw_operation in payload["semantic_operations"]:
        if raw_operation.get("operation_type") != "set_trip_basics":
            continue
        destination_name = raw_operation.get("destination_name")
        if not isinstance(destination_name, str):
            continue
        if _normalize_place_reference(destination_name) not in _normalize_place_reference(
            state["turn_input"].user_text
        ):
            raise PrepareGraphError("trip destination is not grounded in the current user message")
        try:
            city = default_city_registry().resolve(destination_name)
        except CityRegistryError:
            raw_operation["destination_canonical_id"] = None
            continue
        raw_operation["destination_name"] = city.display_name
        raw_operation["destination_canonical_id"] = city.city_id
        state["known_entity_refs"].add(city.city_id)
        changed = True
    return PrepareDecision.model_validate(payload) if changed else decision


def _materialize_assessed_trip_date_range(
    decision: PrepareDecision,
    assessment: TripDateRangeAssessment | None,
) -> PrepareDecision:
    """Compile one validated ready date range into the model-selected basics update."""

    if assessment is None or assessment.status != "ready":
        return decision
    if (
        assessment.start_date is None
        or assessment.end_date is None
        or assessment.duration_days is None
    ):
        raise PrepareGraphError("assessed date range is incomplete")
    payload = decision.model_dump(mode="json")
    basics_operations = [
        operation
        for operation in payload["semantic_operations"]
        if operation.get("operation_type") == "set_trip_basics"
    ]
    if len(basics_operations) != 1:
        return decision
    basics_operations[0].update(
        {
            "start_date": assessment.start_date.isoformat(),
            "end_date": assessment.end_date.isoformat(),
            "duration_days": assessment.duration_days,
        }
    )
    return PrepareDecision.model_validate(payload)


def _strip_unchanged_contextual_destination(
    decision: PrepareDecision,
    state: PrepareGraphState,
    assessment: TripDateRangeAssessment | None,
) -> PrepareDecision:
    """Drop an unchanged saved city echoed while supplying, revising or confirming dates."""

    if assessment is None or assessment.status not in {"ready", "needs_confirmation"}:
        return decision
    saved = state["semantic_state"].trip_basics
    if not saved.destination_name or not saved.destination_canonical_id:
        return decision

    payload = decision.model_dump(mode="json")
    changed = False
    normalized_user_text = _normalize_place_reference(state["turn_input"].user_text)
    for raw_operation in payload["semantic_operations"]:
        if raw_operation.get("operation_type") != "set_trip_basics":
            continue
        destination_name = raw_operation.get("destination_name")
        if not isinstance(destination_name, str):
            continue
        if _normalize_place_reference(destination_name) in normalized_user_text:
            continue
        try:
            proposed_city = default_city_registry().resolve(destination_name)
        except CityRegistryError:
            continue
        if proposed_city.city_id != saved.destination_canonical_id:
            continue
        raw_operation["destination_name"] = None
        raw_operation["destination_canonical_id"] = None
        changed = True
    return PrepareDecision.model_validate(payload) if changed else decision


def _validate_required_trip_basics_fields(
    decision: PrepareDecision,
    state: PrepareGraphState,
    required_fields: tuple[str, ...],
) -> None:
    if not required_fields:
        return
    basics_operations = [
        item.root
        for item in decision.semantic_operations
        if isinstance(item.root, SetTripBasicsOperation)
    ]
    if len(basics_operations) != 1:
        raise PrepareGraphError("required trip basics update is missing")
    operation = basics_operations[0]
    missing: list[str] = []
    if "destination" in required_fields and operation.destination_name is None:
        missing.append("destination")
    if "date_range" in required_fields and (
        operation.start_date is None or operation.end_date is None
    ):
        missing.append("date_range")
    if "duration_days" in required_fields and operation.duration_days is None:
        missing.append("duration_days")
    if "travelers" in required_fields and not operation.travelers:
        missing.append("travelers")
    if "trip_goals" in required_fields and not operation.trip_goals:
        missing.append("trip_goals")
    if missing:
        raise PrepareGraphError("trip basics dropped explicit fields: " + ",".join(missing))
    if operation.destination_name is not None and _normalize_place_reference(
        operation.destination_name
    ) not in _normalize_place_reference(state["turn_input"].user_text):
        raise PrepareGraphError("trip destination is not grounded in the current user message")


def _validate_assessed_trip_date_range(
    decision: PrepareDecision,
    assessment: TripDateRangeAssessment | None,
) -> None:
    if assessment is None or assessment.status == "none":
        return
    basics_operations = [
        item.root
        for item in decision.semantic_operations
        if isinstance(item.root, SetTripBasicsOperation)
    ]
    if assessment.status == "needs_confirmation":
        if any(
            operation.start_date is not None or operation.end_date is not None
            for operation in basics_operations
        ):
            raise PrepareGraphError("unconfirmed assessed date range cannot be written")
        return
    if len(basics_operations) != 1:
        raise PrepareGraphError("ready assessed date range requires one trip basics update")
    operation = basics_operations[0]
    if (
        operation.start_date != assessment.start_date
        or operation.end_date != assessment.end_date
        or operation.duration_days != assessment.duration_days
    ):
        raise PrepareGraphError("trip basics date range differs from assessed context")


def _assessed_date_range_summary(assessment: TripDateRangeAssessment) -> str:
    if (
        assessment.start_date is None
        or assessment.end_date is None
        or assessment.duration_days is None
    ):
        raise PrepareGraphError("assessed date range is incomplete")
    return (
        f"{assessment.start_date.isoformat()} 至 {assessment.end_date.isoformat()}，"
        f"共 {assessment.duration_days} 天"
    )


def _materialize_runtime_decision_payload(
    payload: dict[str, object],
    state: PrepareGraphState,
    continuation_semantics: Literal["full", "none", "concrete_entity", "lodging_booking"],
    required_semantic_operation: Literal[
        "none",
        "trip_basics",
        "trip_basics_with_additions",
        "lodging_booking",
        "dining_requirement",
        "final_supplement",
        "pace_requirement",
    ],
) -> dict[str, object]:
    payload = normalize_compound_basics(payload)
    turn_input = state["turn_input"]
    action = payload.get("next_action")
    if isinstance(action, dict) and action.get("kind") == "final_supplement":
        action["requested_targets"] = ["final_supplement"]
    payload.setdefault("semantic_operations", [])
    payload.setdefault("tool_requests", [])
    payload.setdefault("clarification", None)
    if (
        required_semantic_operation == "lodging_booking"
        or continuation_semantics == "lodging_booking"
    ):
        booking = payload.pop("lodging_booking", None)
        operations = payload.get("semantic_operations")
        if not isinstance(booking, dict) or not isinstance(operations, list):
            raise PrepareGraphError("lodging_booking must contain the required booking choice")
        if any(
            isinstance(item, dict) and item.get("operation_type") == "set_existing_booking"
            for item in operations
        ):
            raise PrepareGraphError(
                "additional semantic_operations must not repeat lodging_booking"
            )
        operations.insert(0, booking)
    _materialize_conditional_requirement_targets(payload)
    payload["decision_id"] = str(
        uuid5(
            NAMESPACE_URL,
            (
                f"iter-ai:prepare-decision:{turn_input.turn_id}:"
                f"{state['decision_mode']}:{state.get('tool_round', 0)}"
            ),
        )
    )
    payload["based_on_state_version"] = state["semantic_state"].state_version
    proposal = payload.get("section_proposal")
    payload["section_proposal"] = {
        "kind": "stay",
        "section": state["runtime_state"].current_section.value,
        "reopen_sections": [],
        "evidence_refs": [],
    }
    if (
        isinstance(proposal, dict)
        and proposal.get("kind") == "reopen"
        and state["decision_mode"] == "natural_text"
    ):
        # Qwen owns the user's navigation intent. Code supplies provenance and
        # checks that reopening a card cannot skip unfinished earlier sections.
        payload["section_proposal"] = {
            "kind": "reopen",
            "section": state["runtime_state"].current_section.value,
            "reopen_sections": proposal.get("reopen_sections", []),
            "evidence_refs": [turn_input.user_message_ref],
        }
    if required_semantic_operation in {"trip_basics", "trip_basics_with_additions"}:
        raw_operations = payload.get("semantic_operations")
        if not isinstance(raw_operations, list):
            raise PrepareGraphError("trip basics requires semantic operations")
        if required_semantic_operation == "trip_basics_with_additions":
            raw_operation = payload.pop("trip_basics", None)
            raw_operations.insert(0, raw_operation)
        elif len(raw_operations) != 1:
            raise PrepareGraphError("trip basics requires one semantic choice")
        raw_operation = raw_operations[0]
        if not isinstance(raw_operation, dict):
            raise PrepareGraphError("trip basics semantic choice is malformed")
        for field in ("travelers", "trip_goals"):
            raw_value = raw_operation.get(field)
            if isinstance(raw_value, list):
                raw_operation[field] = _dedupe_display_text_values(raw_value)
        raw_operation.update(
            {
                "local_operation_key": str(
                    uuid5(
                        NAMESPACE_URL,
                        f"iter-ai:trip-basics:{turn_input.turn_id}",
                    )
                ),
                "target": "trip_basics",
                "source_refs": sorted(
                    {
                        turn_input.user_message_ref,
                        *turn_input.signed_source_refs,
                    }
                ),
            }
        )
    elif required_semantic_operation == "lodging_booking":
        turn_input = state["turn_input"]
        raw_operations = payload.get("semantic_operations")
        if not isinstance(raw_operations, list) or not raw_operations:
            raise PrepareGraphError("lodging booking requires one semantic choice")
        raw_operation = raw_operations[0]
        if not isinstance(raw_operation, dict):
            raise PrepareGraphError("lodging booking semantic choice is malformed")
        raw_operation.update(
            {
                "local_operation_key": str(
                    uuid5(
                        NAMESPACE_URL,
                        f"iter-ai:lodging-booking:{turn_input.turn_id}",
                    )
                ),
                "target": "lodging_booking",
                "source_refs": sorted(
                    {
                        turn_input.user_message_ref,
                        *turn_input.signed_source_refs,
                    }
                ),
            }
        )
    elif required_semantic_operation == "dining_requirement":
        turn_input = state["turn_input"]
        raw_operations = payload.get("semantic_operations")
        if not isinstance(raw_operations, list) or len(raw_operations) != 1:
            raise PrepareGraphError("dining requirement requires one semantic choice")
        raw_operation = raw_operations[0]
        if not isinstance(raw_operation, dict):
            raise PrepareGraphError("dining requirement semantic choice is malformed")
        raw_operation.update(
            {
                "local_operation_key": str(
                    uuid5(
                        NAMESPACE_URL,
                        f"iter-ai:dining-requirement:{turn_input.turn_id}",
                    )
                ),
                "target": "dining_requirement",
                "source_refs": sorted(
                    {
                        turn_input.user_message_ref,
                        *turn_input.signed_source_refs,
                    }
                ),
            }
        )
    elif required_semantic_operation == "pace_requirement":
        raw_operations = payload.get("semantic_operations")
        if not isinstance(raw_operations, list) or len(raw_operations) != 1:
            raise PrepareGraphError("pace requirement requires one semantic choice")
        raw_operation = raw_operations[0]
        if not isinstance(raw_operation, dict):
            raise PrepareGraphError("pace requirement semantic choice is malformed")
        raw_operation.update(
            {
                "local_operation_key": str(
                    uuid5(
                        NAMESPACE_URL,
                        f"iter-ai:pace-requirement:{turn_input.turn_id}",
                    )
                ),
                "target": "transport_and_pace",
                "source_refs": sorted(
                    {
                        turn_input.user_message_ref,
                        *turn_input.signed_source_refs,
                    }
                ),
            }
        )
        payload["next_action"] = {
            "kind": "final_supplement",
            "domain": "general",
            "requested_targets": ["final_supplement"],
        }
    if continuation_semantics == "lodging_booking":
        canonical_entity = _unambiguous_current_entity(state)
        if canonical_entity is None:
            raise PrepareGraphError("lodging booking continuation requires one resolved entity")
        turn_input = state["turn_input"]
        raw_operations = payload.get("semantic_operations")
        if not isinstance(raw_operations, list) or not raw_operations:
            raise PrepareGraphError("lodging booking continuation requires one semantic choice")
        raw_operation = raw_operations[0]
        if not isinstance(raw_operation, dict):
            raise PrepareGraphError("lodging booking continuation is malformed")
        raw_operation.update(
            {
                "local_operation_key": str(
                    uuid5(
                        NAMESPACE_URL,
                        (
                            "iter-ai:lodging-booking-resolved:"
                            f"{turn_input.turn_id}:{canonical_entity}"
                        ),
                    )
                ),
                "target": "lodging_booking",
                "canonical_entity_id": canonical_entity,
                "source_refs": sorted(
                    {
                        turn_input.user_message_ref,
                        *turn_input.signed_source_refs,
                        *state["allowed_source_refs"],
                    }
                ),
            }
        )
        return payload
    if required_semantic_operation == "final_supplement":
        payload["semantic_operations"] = [
            {
                "operation_type": "confirm_final_supplement",
                "local_operation_key": str(
                    uuid5(
                        NAMESPACE_URL,
                        f"iter-ai:final-supplement:{turn_input.turn_id}",
                    )
                ),
                "target": "final_supplement",
                "source_refs": sorted(
                    {
                        turn_input.user_message_ref,
                        *turn_input.signed_source_refs,
                    }
                ),
                "confidence": "high",
                "supplement_text": None,
                "explicitly_no_more_requirements": True,
            }
        ]
        payload["next_action"] = {
            "kind": "generate_task_book",
            "domain": "general",
            "requested_targets": [],
        }
        return payload
    if continuation_semantics != "concrete_entity":
        return payload
    choices = _resolution_choices(state)
    raw_operations = payload.get("semantic_operations")
    if not isinstance(raw_operations, list):
        raise PrepareGraphError("concrete entity continuation is malformed")
    represented_names = {
        _normalize_place_reference(str(item.get("display_name", "")))
        for item in raw_operations
        if isinstance(item, dict)
    }
    represented_keys = {
        item.get("resolution_key") for item in raw_operations if isinstance(item, dict)
    }
    pending_choices = list(state.get("pending_entity_choices", []))
    intake = state.get("trip_basics_assessment")
    for intent in intake.named_entity_intents if intake is not None else []:
        pending_choice: dict[str, object] = {
            "operation_type": "exclude_concrete_entity"
            if intent.disposition == "avoid"
            else "select_concrete_entity",
            "domain": intent.domain,
            "display_name": intent.query,
            "confidence": "medium",
        }
        if intent.disposition != "avoid":
            pending_choice["disposition"] = intent.disposition
        pending_choices.append(pending_choice)
    for pending in pending_choices:
        name = _normalize_place_reference(str(pending.get("display_name", "")))
        matches = [item for item in choices if _normalize_place_reference(item.query) == name]
        if (
            name not in represented_names
            and len(matches) == 1
            and len(matches[0].entity_refs) == 1
            and matches[0].key not in represented_keys
        ):
            # The model already selected the disposition before the lookup.
            # Bind only the returned identity; don't ask it to re-extract the intent.
            raw_operations.append({**pending, "resolution_key": matches[0].key})
            represented_names.add(name)
            represented_keys.add(matches[0].key)
    selected_keys: set[str] = set()
    selected_entities: dict[tuple[str, str], dict[str, object]] = {}
    unique_operations: list[dict[str, object]] = []
    for index, raw_operation in enumerate(raw_operations):
        if not isinstance(raw_operation, dict):
            raise PrepareGraphError("concrete entity continuation is malformed")
        choice = bind_selection(
            raw_operation, choices, index=index, allow_single_fallback=len(raw_operations) == 1
        )
        if choice.key in selected_keys:
            raise SemanticCompilationError(
                "duplicate_entity_choice", f"semantic_operations[{index}].resolution_key"
            )
        selected_keys.add(choice.key)
        domain = raw_operation.get("domain")
        if domain not in {"attraction", "dining"}:
            raise PrepareGraphError("concrete entity continuation has an invalid domain")
        canonical_entity = choice.entity_refs[0]
        identity = (domain, canonical_entity)
        previous_operation = selected_entities.get(identity)
        if previous_operation is not None:
            if any(
                previous_operation.get(field) != raw_operation.get(field)
                for field in ("operation_type", "disposition")
            ):
                raise SemanticCompilationError(
                    "entity_choice_conflict", f"semantic_operations[{index}].disposition"
                )
            previous_operation["source_refs"] = sorted(
                {*cast(list[str], previous_operation["source_refs"]), *choice.source_refs}
            )
            continue
        raw_operation.update(
            {
                "local_operation_key": str(
                    uuid5(
                        NAMESPACE_URL,
                        f"iter-ai:concrete-choice:{turn_input.turn_id}:{domain}:{canonical_entity}",
                    )
                ),
                "target": f"{domain}_entity",
                "display_name": choice.query,
                "canonical_entity_id": canonical_entity,
                "source_refs": sorted({turn_input.user_message_ref, *choice.source_refs}),
            }
        )
        selected_entities[identity] = raw_operation
        unique_operations.append(raw_operation)
    payload["semantic_operations"] = unique_operations
    # A full name and its shorter alias can resolve to the same verified POI.
    # One semantic choice covers that identity; requiring one write per lookup
    # would manufacture duplicates and make every retry fail the same guard.
    selected_identity_ids = {identity for _, identity in selected_entities}
    selected_keys.update(
        choice.key
        for choice in choices
        if len(choice.entity_refs) == 1 and choice.entity_refs[0] in selected_identity_ids
    )
    action = payload.get("next_action")
    asking = isinstance(action, dict) and action.get("kind") == "ask_clarification"
    if not asking and set(item.key for item in choices) - selected_keys:
        raise SemanticCompilationError("entity_choices_incomplete", "semantic_operations")
    return payload


def _materialize_conditional_requirement_targets(payload: dict[str, object]) -> None:
    """Derive the state target from the model-selected requirement domain.

    Conditional requirements are merged by domain, so asking the model to also
    choose the equivalent state target creates two representations of the same
    decision. Keep the semantic domain under model control and let the program
    own its one legal target mapping before public-contract validation.
    """

    target_by_domain = {
        "general": "general_constraint",
        "attraction": "attraction_preference",
        "dining": "dining_requirement",
        "lodging": "lodging_class",
        "transport": "transport_and_pace",
    }
    raw_operations = payload.get("semantic_operations")
    if not isinstance(raw_operations, list):
        return
    for raw_operation in raw_operations:
        if not isinstance(raw_operation, dict):
            continue
        if raw_operation.get("operation_type") != "add_conditional_requirement":
            continue
        domain = raw_operation.get("domain")
        target = target_by_domain.get(domain) if isinstance(domain, str) else None
        if target is not None:
            raw_operation["target"] = target


def _dedupe_display_text_values(values: list[object]) -> list[object]:
    unique: list[object] = []
    seen: set[str] = set()
    for value in values:
        key = value.casefold() if isinstance(value, str) else repr(value)
        if key in seen:
            continue
        seen.add(key)
        unique.append(value)
    return unique


def _validate_resolve_query_kind(
    request: ResolvePlaceRequest, state: PrepareGraphState, request_index: int
) -> None:
    intake = state.get("trip_basics_assessment")
    if (
        intake is not None
        and request.purpose.value == "validate_operation"
        and not any(item.query == request.query for item in intake.named_entity_intents)
        and any(
            isinstance(fact, IntakePreferenceFact) and request.query in fact.quote
            for fact in intake.requirement_facts
        )
    ):
        raise SemanticCompilationError(
            "prepare_category_is_not_poi", f"tool_requests[{request_index}].query"
        )
    try:
        requested_city = default_city_registry().resolve(request.query)
    except CityRegistryError:
        requested_city = None
    if (
        requested_city is not None
        and state["runtime_state"].current_section is DiscoverySection.OTHER
        and requested_city in default_city_registry().mentioned_in(state["turn_input"].user_text)
    ):
        raise SemanticCompilationError(
            "prepare_city_is_not_poi", f"tool_requests[{request_index}].query"
        )


async def _ground_resolve_queries(
    gateway: ModelGateway,
    decision: PrepareDecision,
    state: PrepareGraphState,
    cancellation: ModelCancellation,
) -> PrepareDecision:
    resolve_indexes = [
        index
        for index, wrapped in enumerate(decision.tool_requests)
        if isinstance(wrapped.root, ResolvePlaceRequest)
    ]
    if not resolve_indexes:
        return decision
    payload = decision.model_dump(mode="json")
    turn_input = state["turn_input"]
    destination = state["semantic_state"].trip_basics.destination_name
    changed = False
    for index in resolve_indexes:
        request = decision.tool_requests[index].root
        assert isinstance(request, ResolvePlaceRequest)
        _validate_resolve_query_kind(request, state, index)
        raw_request = payload["tool_requests"][index]
        if destination is not None and raw_request["city"] != destination:
            raw_request["city"] = destination
            changed = True
        if len(_normalize_place_reference(request.query)) >= 2 and _resolve_query_is_grounded(
            request, state
        ):
            continue
        chunks: list[str] = []
        async for chunk in gateway.stream_text(
            build_place_reference_request(
                current_user_text=turn_input.user_text,
                destination_name=destination,
                recent_conversation=list(turn_input.recent_conversation),
            ),
            cancellation=cancellation,
        ):
            chunks.append(chunk.delta)
        extracted = "".join(chunks).strip().strip("'\"`")
        grounded = GroundedPlaceReference(query=extracted)
        raw_request["query"] = grounded.query
        changed = True
    return PrepareDecision.model_validate(payload) if changed else decision


def _resolve_query_is_grounded(
    request: ResolvePlaceRequest,
    state: PrepareGraphState,
) -> bool:
    query = _normalize_place_reference(request.query)
    destination = state["semantic_state"].trip_basics.destination_name
    if destination:
        city = _normalize_place_reference(destination).removesuffix("市")
        query = query.removeprefix(city)
    current_text = _normalize_place_reference(state["turn_input"].user_text)
    if query and query in current_text:
        return True
    return bool(
        re.search(
            r"(?:它|那里|这里|这个|那个|这家|那家|刚才|前面)",
            state["turn_input"].user_text,
        )
    )


def _executed_resolve_queries(state: PrepareGraphState) -> set[str]:
    executed_ids = state["prior_tool_request_ids"]
    return {
        _normalize_place_reference(request.query)
        for decision in state.get("decisions", [])
        for wrapped in decision.tool_requests
        if isinstance((request := wrapped.root), ResolvePlaceRequest)
        and request.request_id in executed_ids
    }


def _normalize_place_reference(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def _complete_intake_entity_queries(
    decision: PrepareDecision,
    state: PrepareGraphState,
) -> PrepareDecision:
    intake = state.get("trip_basics_assessment")
    if intake is None or decision.next_action.kind is not PrepareActionKind.USE_TOOL:
        return decision
    if not all(isinstance(item.root, ResolvePlaceRequest) for item in decision.tool_requests):
        return decision
    queries = _executed_resolve_queries(state) | {
        _normalize_place_reference(item.root.query)
        for item in decision.tool_requests
        if isinstance(item.root, ResolvePlaceRequest)
    }
    city = state["semantic_state"].trip_basics.destination_name
    for item in decision.semantic_operations:
        if isinstance(item.root, SetTripBasicsOperation) and item.root.destination_name:
            city = item.root.destination_name
    if city is None:
        return decision
    payload = decision.model_dump(mode="json")
    for intent in intake.named_entity_intents:
        if len(payload["tool_requests"]) == 4:
            break
        query = _normalize_place_reference(intent.query)
        if query in queries:
            continue
        if query not in _normalize_place_reference(state["turn_input"].user_text):
            raise SemanticCompilationError("assessed_entity_not_grounded", "named_entity_intents")
        payload["tool_requests"].append(
            {
                "capability": "resolve_place",
                "request_id": f"intake-place-{len(payload['tool_requests'])}",
                "query": intent.query,
                "city": city,
                "purpose": "validate_operation",
                "required": True,
            }
        )
        queries.add(query)
    return PrepareDecision.model_validate(payload)


def _complete_intake_requirement_facts(
    decision: PrepareDecision, state: PrepareGraphState
) -> PrepareDecision:
    """Compile Qwen's grounded atomic requirements, without a second extraction call."""
    intake = state.get("trip_basics_assessment")
    if intake is None or not intake.requirement_facts:
        return decision
    payload = decision.model_dump(mode="json")
    domain_by_target = {
        "general_constraint": "general",
        "transport_and_pace": "transport",
        "lodging_class": "lodging",
        "dining_requirement": "dining",
        "attraction_preference": "attraction",
        "dining_preference": "dining",
        "lodging_area": "lodging",
    }
    existing = [
        *decision.semantic_operations,
        *(item.proposal for item in state.get("accepted_operations", [])),
    ]
    keys = {item.root.local_operation_key for item in existing}
    for index, fact in enumerate(intake.requirement_facts):
        if fact.quote not in state["turn_input"].user_text:
            raise SemanticCompilationError(
                "prepare_requirement_quote_not_grounded", f"requirement_facts[{index}].quote"
            )
        if any(
            fact.quote in json.dumps(item.model_dump(mode="json"), ensure_ascii=False)
            and item.root.target.value == fact.target
            and (
                not isinstance(fact, IntakePreferenceFact)
                or item.root.operation_type == f"{fact.disposition}_preference_direction"
            )
            for item in existing
        ):
            continue
        key = f"intake-fact-{index}"
        while key in keys:
            key += "-fact"
        keys.add(key)
        operation: dict[str, object] = {
            "operation_type": "add_conditional_requirement",
            "local_operation_key": key,
            "target": fact.target,
            "domain": domain_by_target[fact.target],
            "condition": "本次旅行",
            "required_outcome": fact.quote,
            "source_refs": [state["turn_input"].user_message_ref],
            "confidence": "high",
        }
        if isinstance(fact, IntakePreferenceFact):
            operation.pop("condition")
            operation.pop("required_outcome")
            operation.update(
                {
                    "operation_type": f"{fact.disposition}_preference_direction",
                    "direction_id": str(uuid5(NAMESPACE_URL, f"{fact.target}:{fact.quote}")),
                    "label": fact.quote,
                }
            )
        payload["semantic_operations"].append(operation)
        existing.append(SemanticOperationProposal.model_validate(operation))
    return PrepareDecision.model_validate(payload)


def _validate_additional_semantic_coverage(
    decision: PrepareDecision,
    state: PrepareGraphState,
    targets: tuple[str, ...],
) -> None:
    if decision.next_action.kind is PrepareActionKind.ASK_CLARIFICATION:
        return
    recorded = {item.root.target.value for item in decision.semantic_operations} | {
        item.proposal.root.target.value for item in state.get("accepted_operations", [])
    }
    missing = set(targets) - recorded
    intake = state.get("trip_basics_assessment")
    if decision.next_action.kind is PrepareActionKind.USE_TOOL and intake is not None:
        missing -= {f"{item.domain}_entity" for item in intake.named_entity_intents}
    if missing:
        raise PrepareGraphError(
            "compound intake omitted semantic targets: " + ",".join(sorted(missing))
        )


def _bind_unambiguous_tool_entity_refs(
    decision: PrepareDecision,
    state: PrepareGraphState,
) -> PrepareDecision:
    canonical_entity = _unambiguous_current_entity(state)
    if canonical_entity is None or not decision.tool_requests:
        return decision
    payload = decision.model_dump(mode="json")
    changed = False
    canonical_capabilities = {
        "place_facts",
        "opening_hours",
        "ticket_availability",
        "place_products",
    }
    for raw_request in payload["tool_requests"]:
        if raw_request["capability"] in canonical_capabilities:
            raw_request["canonical_entity_ids"] = [canonical_entity]
            changed = True
    return PrepareDecision.model_validate(payload) if changed else decision


def _unambiguous_current_entity(state: PrepareGraphState) -> str | None:
    current_refs = {
        entity_ref
        for item in state.get("observations", [])
        if item.observation.capability.value == "resolve_place"
        and item.observation.status
        in {
            ToolObservationStatus.SUCCESS,
            ToolObservationStatus.PARTIAL,
        }
        for entity_ref in item.observation.entity_refs
        if not is_city_entity_ref(entity_ref)
    }
    if len(current_refs) == 1:
        return next(iter(current_refs))
    known_refs = {
        entity_ref
        for entity_ref in state["known_entity_refs"]
        & _known_state_entity_refs(state["semantic_state"])
        if not is_city_entity_ref(entity_ref)
    }
    return next(iter(known_refs)) if not current_refs and len(known_refs) == 1 else None


def _resolution_choices(state: PrepareGraphState) -> list[ResolvedSelection]:
    return resolution_choices(
        state.get("decisions", []),
        [item.observation for item in state.get("observations", [])],
    )


def _normalize_tool_request_ids(
    decision: PrepareDecision,
    state: PrepareGraphState,
) -> PrepareDecision:
    if not decision.tool_requests:
        return decision
    payload = decision.model_dump(mode="json")
    raw_requests = payload["tool_requests"]
    replacements = {
        item.root.request_id: str(
            uuid5(
                NAMESPACE_URL,
                (
                    "iter-ai:prepare-tool:"
                    f"{state['turn_input'].turn_id}:{state.get('tool_round', 0)}:{index}"
                ),
            )
        )
        for index, item in enumerate(decision.tool_requests)
    }
    for raw_request in raw_requests:
        original_id = raw_request["request_id"]
        raw_request["request_id"] = replacements[original_id]
        raw_request["depends_on"] = [
            replacements[dependency_id] for dependency_id in raw_request.get("depends_on", [])
        ]
    return PrepareDecision.model_validate(payload)


def _fact_capabilities(state: PrepareGraphState) -> list[str]:
    """Read Qwen's declared fact needs, never classify the user's words again."""
    allowed = {
        "place_facts",
        "opening_hours",
        "ticket_availability",
        "weather_forecast",
        "spatial_routes",
        "hotel_booking_facts",
        "place_products",
    }
    intake = state.get("trip_basics_assessment")
    required: list[str] = list(intake.fact_capabilities) if intake is not None else []
    decisions = list(state.get("decisions", []))
    if state.get("decision") is not None:
        decisions.append(state["decision"])
    for decision in decisions:
        required.extend(item for item in decision.reply_goal.fact_requirements if item in allowed)
        required.extend(item for item in decision.reply_goal.answer_questions if item in allowed)
        required.extend(
            request.root.capability.value
            for request in decision.tool_requests
            if request.root.purpose.value == "answer_user"
            and request.root.capability.value in allowed
        )
    return list(dict.fromkeys(required))


def _required_next_fact_tool(
    state: PrepareGraphState,
    observations: list[ToolObservation],
    *,
    tool_round: int,
) -> str | None:
    completed = {item.capability.value for item in observations}
    required = next((item for item in _fact_capabilities(state) if item not in completed), None)
    if required is None:
        return None
    if _fact_lookup_cannot_progress(required, observations, tool_round=tool_round):
        return None
    resolved_place = any(
        item.capability.value == "resolve_place"
        and item.status
        in {
            ToolObservationStatus.SUCCESS,
            ToolObservationStatus.PARTIAL,
        }
        and item.entity_refs
        for item in observations
    )
    if required in {"opening_hours", "ticket_availability"} and not resolved_place:
        return "resolve_place"
    return required


def _required_next_tool_capability(
    state: PrepareGraphState,
    observations: list[ToolObservation],
) -> str | None:
    """Narrow the model schema to the one capability this turn still requires.

    An explicit named attraction must be resolved before the model can write a
    canonical choice. Mixed dining constraints remain in the same guarded turn.
    Keeping this requirement in the runtime contract avoids asking the model to
    select among every action/tool union at once, while the model still extracts
    the place query and disposition.
    """

    final_assessment = state.get("final_supplement_assessment")
    if _has_active_final_supplement_followup(state) and (
        _explicitly_requests_task_book_generation(state["turn_input"].user_text)
        or (
            final_assessment is not None
            and final_assessment.explicitly_no_more_requirements
            and not final_assessment.has_additional_request
        )
    ):
        # "Keep the must-see places/budget unchanged" is not a new POI request.
        # The same assessed closure governs both schema selection and Guard.
        return None
    intake = state.get("trip_basics_assessment")
    if intake is not None and state.get("tool_round", 0) < 2:
        executed = _executed_resolve_queries(state)
        if any(
            _normalize_place_reference(item.query) not in executed
            for item in intake.named_entity_intents
        ):
            return "resolve_place"
    # Only an LLM-grounded named entity creates a resolution obligation.
    # A target (including delegation), a category, or a word such as 忌口/不去
    # is not a place. With no grounded inventory, the main Qwen decision is
    # free to choose tools; canonical IDs are still checked before a write.
    return _required_next_fact_tool(
        state,
        observations,
        tool_round=state.get("tool_round", 0),
    )


def _continuation_semantics(
    state: PrepareGraphState,
) -> Literal["full", "none", "concrete_entity", "lodging_booking"]:
    if state["decision_mode"] != "bounded_redecide":
        return "full"
    if state.get("card_action_observations") or state.get("task_book_action_observations"):
        return "none"
    task_book_assessment = state.get("task_book_review_assessment")
    task_book_targets = (
        set(task_book_assessment.required_targets) if task_book_assessment is not None else set()
    )
    accepted_targets = {item.proposal.root.target for item in state.get("accepted_operations", [])}
    pending_task_book_targets = task_book_targets - accepted_targets
    task_book_entity_targets = task_book_targets & {
        SemanticTargetV4.ATTRACTION_ENTITY,
        SemanticTargetV4.DINING_ENTITY,
        SemanticTargetV4.LODGING_BOOKING,
    }
    semantic_resolution_requested = any(
        request.root.capability.value == "resolve_place"
        and request.root.purpose.value == "validate_operation"
        for decision in state.get("decisions", [])
        for request in decision.tool_requests
    )
    if (
        _assessment_is_fact_only(state)
        and not task_book_entity_targets
        and not state.get("pending_entity_choices")
        and not semantic_resolution_requested
    ):
        return "none"
    has_resolved_entity = any(
        item.observation.capability.value == "resolve_place"
        and item.observation.status
        in {
            ToolObservationStatus.SUCCESS,
            ToolObservationStatus.PARTIAL,
        }
        and item.observation.entity_refs
        for item in state.get("observations", [])
    )
    pending_non_entity_targets = pending_task_book_targets - {
        SemanticTargetV4.ATTRACTION_ENTITY,
        SemanticTargetV4.DINING_ENTITY,
        SemanticTargetV4.LODGING_BOOKING,
    }
    if has_resolved_entity and pending_non_entity_targets:
        # The tool round was allowed to defer a compound revision. Restore the
        # broad no-tool contract so the continuation can atomically cover the
        # grounded entity and every remaining semantic target.
        return "full"
    choices = _resolution_choices(state)
    accepted_entities = {
        item.proposal.root.canonical_entity_id
        for item in state.get("accepted_operations", [])
        if isinstance(
            item.proposal.root, (SelectConcreteEntityOperation, ExcludeConcreteEntityOperation)
        )
    }
    if choices and all(
        len(item.entity_refs) == 1 and item.entity_refs[0] in accepted_entities for item in choices
    ):
        return "full"  # Do not demand the same entity choices again after their commit.
    booking_resolution_queries = {
        request.root.query
        for decision in state.get("decisions", [])
        for request in decision.tool_requests
        if isinstance(request.root, ResolvePlaceRequest)
        and request.root.purpose.value == "validate_operation"
    }
    has_pending_booking = any(
        item.root.operation_type == "set_existing_booking"
        and getattr(item.root, "canonical_entity_id", None) is None
        and any(
            query in getattr(item.root, "user_description", "")
            for query in booking_resolution_queries
        )
        for decision in state.get("decisions", [])
        for item in decision.semantic_operations
    )
    if has_resolved_entity and has_pending_booking:
        return "lodging_booking"
    has_concrete_target = bool(
        task_book_targets
        & {
            SemanticTargetV4.ATTRACTION_ENTITY,
            SemanticTargetV4.DINING_ENTITY,
        }
    )
    has_explicit_disposition = (
        _required_concrete_choice(state) != "none"
        or bool(state.get("pending_entity_choices"))
        or semantic_resolution_requested
    )
    return (
        "concrete_entity"
        if _resolution_choices(state) and (has_explicit_disposition or has_concrete_target)
        else "full"
    )


def _required_concrete_choice(
    state: PrepareGraphState,
) -> Literal[
    "none",
    "attraction_must",
    "attraction_want",
    "attraction_if_convenient",
    "attraction_avoid",
    "dining_destination",
    "dining_if_convenient",
    "dining_avoid",
]:
    if state["decision_mode"] != "bounded_redecide":
        return "none"
    choices = _resolution_choices(state)
    if len(choices) > 1 or any(len(item.entity_refs) != 1 for item in choices):
        # A whole-message hint cannot impose one disposition/domain on every POI.
        return "none"
    # Preserve the model's per-object interpretation, never apply a negation
    # or a preference from another clause to this resolved place.
    if not choices:
        return "none"
    intake = state.get("trip_basics_assessment")
    intents: list[tuple[str, str, str]] = [
        (item.query, item.domain, item.disposition)
        for item in (intake.named_entity_intents if intake is not None else [])
    ]
    intents.extend(
        (
            str(item.get("display_name", "")),
            str(item.get("domain", "")),
            "avoid"
            if item.get("operation_type") == "exclude_concrete_entity"
            else str(item.get("disposition", "")),
        )
        for item in state.get("pending_entity_choices", [])
    )
    matches = {
        f"{domain}_{disposition}"
        for query, domain, disposition in intents
        if _normalize_place_reference(query) == _normalize_place_reference(choices[0].query)
    }
    allowed = {
        "attraction_must",
        "attraction_want",
        "attraction_if_convenient",
        "attraction_avoid",
        "dining_destination",
        "dining_if_convenient",
        "dining_avoid",
    }
    if len(matches) == 1 and matches <= allowed:
        return cast(
            Literal[
                "attraction_must",
                "attraction_want",
                "attraction_if_convenient",
                "attraction_avoid",
                "dining_destination",
                "dining_if_convenient",
                "dining_avoid",
            ],
            next(iter(matches)),
        )
    return "none"


def _required_initial_semantic_operation(
    state: PrepareGraphState,
) -> Literal["none", "lodging_booking", "dining_requirement"]:
    if state["decision_mode"] == "bounded_redecide":
        return "none"
    assessment = state.get("task_book_review_assessment")
    intake = state.get("trip_basics_assessment")
    targets = (set(assessment.required_targets) if assessment is not None else set()) | (
        set(intake.required_additional_targets) if intake is not None else set()
    )
    if targets == {SemanticTargetV4.LODGING_BOOKING}:
        return "lodging_booking"
    # Even one dining target can contain several independent requirements.
    # Do not narrow an arbitrary sentence into a single operation contract.
    return "none"


def _required_post_update_action(
    state: PrepareGraphState,
) -> Literal[
    "none",
    "reply_only",
    "show_preference_card",
    "show_specific_card",
    "final_supplement",
]:
    if state["decision_mode"] != "decide_after_update":
        return "none"
    if state["turn_input"].user_event_kind == "task_book_confirmation":
        return "reply_only"
    if not _trip_dates_complete(state["semantic_state"]):
        return "none"
    if state["turn_input"].user_event_kind not in {"card_answer", "retry_interaction"}:
        return "none"
    actions: dict[
        DiscoverySection,
        Literal[
            "show_preference_card",
            "show_specific_card",
            "final_supplement",
        ],
    ] = {
        DiscoverySection.ATTRACTION_PREFERENCE: "show_preference_card",
        DiscoverySection.ATTRACTION_SPECIFIC: "show_specific_card",
        DiscoverySection.DINING_PREFERENCE: "show_preference_card",
        DiscoverySection.DINING_SPECIFIC: "show_specific_card",
        DiscoverySection.LODGING_AREA_PREFERENCE: "show_preference_card",
        DiscoverySection.LODGING_CLASS_PREFERENCE: "show_preference_card",
        DiscoverySection.FINAL_SUPPLEMENT: "final_supplement",
    }
    return actions.get(state["runtime_state"].current_section, "none")


def _forbid_initial_semantic_operations(state: PrepareGraphState) -> bool:
    if state["decision_mode"] == "bounded_redecide":
        return False
    if _required_initial_semantic_operation(state) != "none":
        return False
    return _assessment_is_fact_only(state)


def _assessment_is_fact_only(state: PrepareGraphState) -> bool:
    """A fact lookup's purpose alone cannot erase other intents in the same turn."""
    intake = state.get("trip_basics_assessment")
    return bool(
        intake is not None
        and intake.fact_capabilities
        and not intake.required_additional_targets
        and not intake.named_entity_intents
        and not intake.requirement_facts
        and not any(
            (
                intake.explicit_destination,
                intake.explicit_date_range,
                intake.explicit_duration_days,
                intake.explicit_travelers,
                intake.explicit_trip_goals,
                intake.explicit_lodging_not_applicable,
            )
        )
    )


def _fact_lookup_cannot_progress(
    required_capability: str | None,
    observations: list[ToolObservation],
    *,
    tool_round: int,
) -> bool:
    if tool_round >= 2:
        return True
    entity_dependent = {
        "opening_hours",
        "ticket_availability",
        "spatial_routes",
        "hotel_booking_facts",
    }
    if required_capability not in entity_dependent:
        return False
    return any(
        item.capability.value == "resolve_place"
        and item.status
        in {
            ToolObservationStatus.UNAVAILABLE,
            ToolObservationStatus.INVALID_REQUEST,
        }
        for item in observations
    )


def _fallback_with_validated_operations(
    state: PrepareGraphState,
    validated: PrepareDecision | None,
) -> PrepareDecision:
    fallback = _fallback_decision(state["runtime_state"], state["semantic_state"].state_version)
    if validated is None:
        return fallback
    # Still staged and committed with the final fallback message/outbox, never early-written.
    return fallback.model_copy(update={"semantic_operations": validated.semantic_operations})


def _fallback_decision(
    runtime_state: DiscoveryRuntimeState,
    state_version: int,
) -> PrepareDecision:
    return PrepareDecision(
        decision_id=str(uuid4()),
        based_on_state_version=state_version,
        semantic_operations=[],
        next_action=ReplyOnlyAction(kind=PrepareActionKind.REPLY_ONLY),
        section_proposal=SectionProposal(
            kind=SectionProposalKind.STAY,
            section=runtime_state.current_section,
        ),
        reply_goal=ReplyGoal(explain_next_step="保持当前探索位置并安全降级。"),
    )


def _known_state_entity_refs(state: TripSemanticState) -> set[str]:
    refs = {
        item.canonical_entity_id
        for item in [
            *state.attractions.concrete_intents,
            *state.attractions.exclusions,
            *state.dining.concrete_restaurant_intents,
            *state.dining.exclusions,
            *state.lodging.user_named_hotel_intents,
        ]
    }
    refs.update(
        item.canonical_entity_id
        for item in state.existing_bookings
        if item.canonical_entity_id is not None
    )
    return refs


def _is_optional_trip_preferences_response(state: PrepareGraphState) -> bool:
    pending = state["runtime_state"].pending_interaction
    return bool(
        state["decision_mode"] == "natural_text"
        and pending is not None
        and pending.kind is PendingInteractionKind.FREE_TEXT_QUESTION
        and OPTIONAL_TRIP_PREFERENCES_TARGET in pending.target_ids
    )


def _trip_dates_complete(state: TripSemanticState) -> bool:
    basics = state.trip_basics
    return bool(basics.start_date and basics.end_date and basics.duration_days)


def _grounding_context(
    state: PrepareGraphState,
    runtime_state: DiscoveryRuntimeState,
    outcome: str,
) -> dict[str, object]:
    accepted = state.get("accepted_operations", [])
    observations = [
        item.observation.model_dump(mode="json") for item in state.get("observations", [])
    ]
    card_observations = [
        item.model_dump(mode="json") for item in state.get("card_action_observations", [])
    ]
    task_book_observations = [
        item.model_dump(mode="json") for item in state.get("task_book_action_observations", [])
    ]
    fact_capabilities = _fact_capabilities(state)
    required_fact_capability = next(iter(fact_capabilities), None)
    date_range_assessment = state.get("date_range_assessment")
    return {
        "user_text": state["turn_input"].user_text,
        "submitted_interaction": {
            "event_kind": state["turn_input"].user_event_kind,
            "section": state["turn_input"].runtime_state.current_section.value,
        },
        "verified_changes": [_operation_summary(item) for item in accepted],
        "tool_observations": observations,
        "card_action_observations": card_observations,
        "task_book_action_observations": task_book_observations,
        # A question mark alone does not imply a live Provider lookup. Facts
        # already present in the immutable published plan are trusted server
        # context; only recognized time-sensitive capabilities require a fresh
        # observation before the response may claim them.
        "fact_required": required_fact_capability is not None,
        "required_fact_capability": required_fact_capability,
        "trusted_state": {
            "trip_basics": state["semantic_state"].trip_basics.model_dump(mode="json"),
            "party_size": parse_party_size(state["semantic_state"].trip_basics.travelers)
            if state["semantic_state"].trip_basics.travelers
            else None,
            "published_plan": _published_plan_reply_context(state["turn_input"].published_plan),
        },
        "current_section": runtime_state.current_section.value,
        "next_action": state["decision"].next_action.kind.value,
        "requested_targets": list(state["decision"].next_action.requested_targets),
        "reply_goal": state["decision"].reply_goal.model_dump(mode="json"),
        "date_range_candidate": (
            date_range_assessment.model_dump(mode="json")
            if date_range_assessment is not None
            and date_range_assessment.status == "needs_confirmation"
            else None
        ),
        "date_range_resolution": (
            date_range_assessment.model_dump(mode="json")
            if date_range_assessment is not None and date_range_assessment.status != "none"
            else None
        ),
        "decision_failure_code": state.get("decision_failure_code"),
        "date_followup_active": _has_active_date_range_followup(state),
        "outcome": outcome,
    }


def _published_plan_reply_context(
    plan: PlannerPublishedPlan | None,
) -> dict[str, object] | None:
    """Expose a compact, server-verified plan projection to response Qwen."""

    if plan is None:
        return None
    schedule = plan.materialized_schedule
    known_total = plan.cost_draft.known_total_per_person
    selected_hotel_name: str | None = None
    selected_ref = plan.working_itinerary.lodging_baseline.selected_offer_ref
    if selected_ref is not None and plan.hotel_observation is not None:
        selected_hotel_name = next(
            (
                offer.property_name
                for offer in plan.hotel_observation.offers
                if offer.offer_ref == selected_ref
            ),
            None,
        )
    return {
        "status": "formal_plan",
        "day_count": len(schedule.days),
        "start_date": schedule.start_date.isoformat(),
        "end_date": schedule.end_date.isoformat(),
        "days": [
            {
                "service_date": day.service_date.isoformat(),
                "start_time": day.start_time.isoformat(),
                "end_time": day.end_time.isoformat(),
                "activities": [
                    {
                        "title": activity.title,
                        "start_time": activity.start_time.isoformat(),
                        "end_time": activity.end_time.isoformat(),
                    }
                    for activity in day.activities
                ],
            }
            for day in schedule.days
        ],
        "known_total_per_person": (
            known_total.model_dump(mode="json") if known_total is not None else None
        ),
        "pricing_note": plan.cost_draft.pricing_note,
        "selected_hotel_name": selected_hotel_name,
        "lodging_mode": plan.working_itinerary.lodging_baseline.mode,
    }


def _free_text_pending(
    runtime_state: DiscoveryRuntimeState,
    decision: PrepareDecision,
) -> PendingInteraction:
    targets = (
        ["final_supplement"]
        if decision.next_action.kind is PrepareActionKind.FINAL_SUPPLEMENT
        else list(dict.fromkeys(decision.next_action.requested_targets))
        or [runtime_state.current_section.value]
    )
    interaction_id = str(uuid4())
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "state_version": runtime_state.state_version,
                "section": runtime_state.current_section.value,
                "targets": targets,
                "decision_id": decision.decision_id,
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return PendingInteraction(
        interaction_id=interaction_id,
        kind=PendingInteractionKind.FREE_TEXT_QUESTION,
        section=runtime_state.current_section,
        target_ids=targets,
        based_on_state_version=runtime_state.state_version,
        dependency_fingerprint=fingerprint,
    )


def _attachment_id(attachment: V4Attachment) -> str:
    value = attachment.root
    if isinstance(value, PlannerPublishedPlan):
        return str(value.plan_version_id)
    return str(value.attachment_id if hasattr(value, "attachment_id") else value.task_book_id)


def _attachment_summary(attachment: V4Attachment) -> dict[str, object]:
    value = attachment.root
    if isinstance(value, PlannerPublishedPlan):
        return {
            "kind": "plan",
            "plan_version_id": str(value.plan_version_id),
            "day_count": len(value.materialized_schedule.days),
            "validation_result": value.validation_observation.result,
        }
    if not isinstance(value, TaskBookV4):
        return {
            "kind": value.kind.value,
            "section": value.section.value,
            "option_count": len(value.options),
            "labels": [item.label for item in value.options],
            "prompt": value.prompt,
            "candidate_coverage": "insufficient_verified_choices"
            if value.status.value == "partial_availability"
            else "complete",
            "coverage_explanation": (
                "这里只描述可核验候选的数量是否足够，不涉及营业、可预订状态或余票。"
            ),
        }
    return {
        "kind": "task_book",
        "task_book_id": value.task_book_id,
        "version": value.version,
        "status": value.status.value,
    }


def _operation_summary(item: AcceptedV4Operation) -> dict[str, object]:
    proposal = item.proposal.root
    payload = proposal.model_dump(mode="json")
    safe_fields = {
        key: payload[key]
        for key in (
            "operation_type",
            "target",
            "domain",
            "label",
            "display_name",
            "disposition",
            "user_description",
            "condition",
            "required_outcome",
            "supplement_text",
            "destination_name",
            "start_date",
            "end_date",
            "duration_days",
            "travelers",
            "trip_goals",
            "delegated_targets",
            "nightly_budget_minimum_minor",
            "nightly_budget_maximum_minor",
        )
        if key in payload
    }
    return {"operation_id": str(item.operation_id), **safe_fields}


def _safe_issue(error: Exception) -> str:
    text = str(error).replace("\n", " ").strip()
    return text[:180] or "决策未通过程序校验。"


def _visited(
    state: PrepareGraphState,
    node: str,
    *,
    maximum: int,
) -> PrepareGraphState:
    visits = dict(state.get("node_visits", {}))
    visits[node] = visits.get(node, 0) + 1
    if visits[node] > maximum:
        raise PrepareGraphError(f"{node} exceeded its bounded visit count")
    return {
        "trace": [*state.get("trace", []), node],
        "node_visits": visits,
    }


__all__ = [
    "PrepareAgentGraph",
    "PrepareGraphError",
    "PrepareGraphResult",
    "PrepareTurnInput",
]
