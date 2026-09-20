"""Bounded Qwen decisions around deterministic V4 draft compilation and validation."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from time import monotonic
from typing import Any, TypedDict, cast
from uuid import uuid4

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from pydantic import ValidationError

from backend.agent.model_audit import record_model_call_annotation
from backend.agent.model_gateway import (
    ModelCancellation,
    ModelFailureCode,
    ModelGateway,
    ModelGatewayError,
    ModelMessage,
    ModelRequest,
    ModelRole,
    ModelStructuredResult,
    StructuredValue,
)
from backend.agent.planner.daily_repair import (
    FINALIZE_SECONDS,
    begin_repair_batch,
    execution_remaining,
    finish_repair_batch,
    gap_matches_issue,
    next_repair_batch,
    refresh_repair_state,
)
from backend.agent.planner.decision_contracts import (
    ModelEvidenceRepairIntent,
    ModelPatchRepairIntent,
    ModelPlanChangePatchIntent,
    ModelPlanIntent,
    ModelPlannerDecision,
    ModelPlannerDecisionAfterStrategyWithIssues,
    ModelPlannerDecisionAfterStrategyWithoutIssues,
    ModelPlanStop,
    ModelRepairIntent,
    ModelStrategyDecision,
)
from backend.agent.planner.dependencies import (
    invalidate_materialized_dependencies,
    prepare_workspace_for_evidence_refresh,
    rebind_semantic_artifacts_after_evidence,
)
from backend.agent.planner.dining_repair import (
    ModelDiningReplacement,
    apply_dining_replacement,
    dining_replacement_options,
    dining_replacement_request,
)
from backend.agent.planner.dining_slot_search import (
    MAX_SLOT_CHOICES,
    MAX_SLOT_PAGES,
    slot_choice_id,
    supplement_dining_slot,
)
from backend.agent.planner.dining_slots import (
    DiningSlot,
    ModelDiningSlotChoice,
    admit_dining_place,
    dining_slot_is_resolved,
    dining_slot_request,
    dining_slots,
    restore_dining_slot_order,
)
from backend.agent.planner.evidence import (
    PlannerEvidenceBackend,
    selected_itinerary_route_requests,
)
from backend.agent.planner.finalization import build_finalize_decision
from backend.agent.planner.gap_infill import (
    ModelGapVisits,
    apply_gap_visit_choices,
    build_gap_visit_request,
    gap_choice_hours_failures,
    gap_visit_options,
)
from backend.agent.planner.guard_feedback import (
    GuardStage,
    compile_guard_violation,
    repeated_guard_failure,
)
from backend.agent.planner.guards import guard_ask_user, validate_strategy, validate_working_draft
from backend.agent.planner.long_visit_meal_repair import (
    apply_long_visit_meal_choice,
    long_visit_meal_options,
    long_visit_meal_request,
)
from backend.agent.planner.materializer import PlannerDraftMaterializer
from backend.agent.planner.meal_completion import (
    ModelMissingMeals,
    apply_missing_meals,
    missing_meal_options,
    missing_meal_request,
)
from backend.agent.planner.plan_intent_compiler import (
    build_automatic_hotel_request,
    compile_default_strategy_decision,
    compile_plan_intent_decision,
)
from backend.agent.planner.prompts import (
    build_compact_plan_request,
    build_plan_change_patch_request,
    build_planner_decision_request,
    build_planner_repair_request,
    build_planner_response_request,
)
from backend.agent.planner.proposals import PlannerReferenceCatalog, resolve_model_decision
from backend.agent.planner.repair_compiler import (
    compile_model_plan_change_intent,
    compile_model_repair_intent,
    compile_validation_interaction,
)
from backend.agent.planner.schedule_repair import (
    ModelMealAssignment,
    ModelScheduleRepair,
    apply_meal_assignment,
    available_meal_assignments,
    meal_assignment_request,
    merge_day_repair,
    normalize_day_repair,
    scoped_repair_request,
)
from backend.agent.planner.timing_quality import (
    afternoon_activity_opportunities,
    dining_commute_issues,
    evening_activity_opportunities,
    has_time_quality_issues,
    may_keep_intermediate_timing_repair,
    missing_concrete_meals,
    natural_day_limitations,
    protected_time_quality_candidates,
    schedule_coverage_issues,
    schedule_meal_issues,
    schedule_quality_gaps,
    schedule_quality_score,
    schedule_travel_minutes,
    selected_candidate_ids,
    validate_response_visit_claims,
)
from backend.agent.planner.validator import PlannerDraftValidator
from backend.agent.planner.workspace import PlannerGuardError, advance, server_id
from backend.contracts.v4.enums import PlannerStatus
from backend.contracts.v4.plan_change import PlanChangeRequest
from backend.contracts.v4.planner_decision import (
    AskUserPayload,
    BuildOrUpdateStrategyPayload,
    MaterializeDraftPayload,
    PlannerDecision,
    RequestEvidencePayload,
    ReviseDraftPayload,
)
from backend.contracts.v4.planner_draft import (
    UnassignedIntent,
    WorkingItineraryDraft,
    planning_projection_digest,
)
from backend.contracts.v4.planner_evidence import PlannerGuardObservation
from backend.contracts.v4.planner_patch import (
    PlanChangeRequestPatchAuthority,
    ValidationIssuePatchAuthority,
    atomic_apply_itinerary_patch,
)
from backend.contracts.v4.planner_refs import CandidateRef, FixedCommitmentRef
from backend.contracts.v4.planner_workspace import PlannerWorkspaceState
from backend.contracts.v4.task_book import TaskBookV4
from backend.persistence.outbox_repository import canonical_json_hash

PlannerCheckpoint = Callable[[PlannerWorkspaceState], Awaitable[None]]
PlannerProgress = Callable[[str, str], Awaitable[None]]
PLANNER_MODEL_MAX_ATTEMPTS = 2
PLANNER_MODEL_RETRY_DELAY_SECONDS = 1.0


@dataclass(frozen=True)
class PlannerGraphContext:
    book: TaskBookV4
    cancellation: ModelCancellation
    checkpoint: PlannerCheckpoint
    progress: PlannerProgress
    input_state_version: int | None = None
    allow_semantic_repair: bool = True
    deadline: float | None = None
    local_change_dates: frozenset[date] | None = None

    def remaining_seconds(self) -> float:
        return max(0.0, self.deadline - monotonic()) if self.deadline is not None else float("inf")

    def call_remaining_seconds(self) -> float:
        return max(0.0, self.remaining_seconds() - FINALIZE_SECONDS)


class PlannerGraphState(TypedDict, total=False):
    workspace: PlannerWorkspaceState
    decision: PlannerDecision | None
    proposal: ModelPlannerDecision | None
    proposal_call_id: str | None
    done: bool


class PlannerAgentGraph:
    """Program enforces legality; Qwen selects actions, strategy and day composition."""

    def __init__(
        self,
        gateway: ModelGateway,
        evidence: PlannerEvidenceBackend,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        complete_plan: bool = False,
        compact_planning: bool = False,
        estimate_visit_durations: bool = False,
        optimize_timing: bool = False,
        materializer: PlannerDraftMaterializer | None = None,
        validator: PlannerDraftValidator | None = None,
    ) -> None:
        self.gateway = gateway
        self.evidence = evidence
        self.clock = clock
        self.complete_plan = complete_plan
        self.compact_planning = compact_planning
        self.estimate_visit_durations = estimate_visit_durations
        self.optimize_timing = optimize_timing
        self.materializer = materializer or PlannerDraftMaterializer(clock=clock)
        self.validator = validator or PlannerDraftValidator(clock=clock)
        graph = StateGraph(PlannerGraphState, context_schema=PlannerGraphContext)
        graph.add_node("planner_initial_evidence", self._initialize)
        graph.add_node("planner_decide", self._decide)
        graph.add_node("planner_execute_and_observe", self._execute)
        graph.add_edge(START, "planner_initial_evidence")
        graph.add_edge("planner_initial_evidence", "planner_decide")
        graph.add_conditional_edges(
            "planner_decide",
            lambda state: "stop" if state.get("done") else "act",
            {"stop": END, "act": "planner_execute_and_observe"},
        )
        graph.add_conditional_edges(
            "planner_execute_and_observe",
            lambda state: "stop" if state.get("done") else "decide",
            {"stop": END, "decide": "planner_decide"},
        )
        self.compiled = graph.compile(name="v4-planner-agent")

    async def invoke(
        self, workspace: PlannerWorkspaceState, context: PlannerGraphContext
    ) -> PlannerWorkspaceState:
        local_dates = workspace.accepted_local_change_dates
        if local_dates is not None:
            if not local_dates:
                assert workspace.plan_change_request is not None
                return await self.apply_plan_change(
                    workspace,
                    context,
                    change_request=workspace.plan_change_request,
                    user_text=(
                        "继续完成已保存的原修改请求，"
                        "具体要求见 validated_semantic_operations，不扩大范围。"
                    ),
                )
            context = replace(context, local_change_dates=local_dates)
        if self.compact_planning:
            return await self._invoke_compact(workspace, context)
        if workspace.status is PlannerStatus.READY_TO_PUBLISH:
            return workspace
        if workspace.working_itinerary is not None:
            # A reply/transport failure after draft checkpoint is not permission
            # to choose another itinerary or append a duplicate materialization.
            context.cancellation.raise_if_cancelled("planner_restore_draft")
            validate_working_draft(
                workspace.working_itinerary, workspace, context.book, self.clock()
            )
            if self.complete_plan:
                return await self._complete_plan(workspace, context)
            restored = advance(workspace, status=PlannerStatus.DRAFT_READY)
            await context.checkpoint(restored)
            return restored
        result = await self.compiled.ainvoke(
            {"workspace": workspace, "decision": None, "proposal": None, "done": False},
            context=context,
            config={"recursion_limit": 32},
        )
        workspace = PlannerWorkspaceState.model_validate(result["workspace"])
        if self.complete_plan and workspace.status is PlannerStatus.DRAFT_READY:
            return await self._complete_plan(workspace, context)
        return workspace

    async def _invoke_compact(
        self,
        workspace: PlannerWorkspaceState,
        context: PlannerGraphContext,
    ) -> PlannerWorkspaceState:
        """Run one compact semantic planning call around the existing safety kernel."""

        if workspace.status is PlannerStatus.READY_TO_PUBLISH:
            return workspace
        if workspace.working_itinerary is not None:
            context.cancellation.raise_if_cancelled("planner_restore_plan")
            validate_working_draft(
                workspace.working_itinerary, workspace, context.book, self.clock()
            )
            if self.complete_plan:
                return await self._complete_plan(workspace, context)
            restored = advance(workspace, status=PlannerStatus.DRAFT_READY)
            await context.checkpoint(restored)
            return restored

        context.cancellation.raise_if_cancelled("planner_initial_evidence")
        if not workspace.initial_evidence_ready:
            await context.progress(
                "planner_evidence",
                "正在并行核验地点、营业信息与真实路线分区。",
            )
            workspace = await self.evidence.initialize(
                workspace,
                context.book,
                context.cancellation,
                checkpoint=context.checkpoint,
            )
            await context.checkpoint(workspace)

        readiness_issues = (
            workspace.readiness_observation.issues
            if workspace.readiness_observation is not None
            else ()
        )
        deferred = tuple(
            issue
            for issue in readiness_issues
            if self.complete_plan and issue.code == "strong_opening_conflict"
        )
        if deferred:
            workspace = advance(
                workspace,
                best_effort_reasons=tuple(
                    dict.fromkeys(
                        (
                            *workspace.best_effort_reasons,
                            *(f"未能安排：{issue.reason_summary}" for issue in deferred),
                        )
                    )
                ),
            )
            await context.checkpoint(workspace)
        user_issues = tuple(
            issue
            for issue in readiness_issues
            if issue.user_authority_required and issue not in deferred
        )
        if user_issues:
            catalog = PlannerReferenceCatalog(workspace)
            first_reason = user_issues[0].ask_user_reason
            issue_keys = tuple(
                key
                for key, issue in catalog.issues.items()
                if issue.user_authority_required and issue.ask_user_reason == first_reason
            )[:4]
            proposal = ModelPlannerDecision.model_validate(
                {
                    "action": "ask_user",
                    "current_goal": "确认无法由系统决定的硬冲突",
                    "reason_summary": user_issues[0].reason_summary,
                    "remaining_blockers": list(issue_keys),
                    "issue_keys": list(issue_keys),
                }
            )
            decision = resolve_model_decision(proposal, workspace)
            guard_ask_user(decision, workspace)
            payload = decision.payload
            assert isinstance(payload, AskUserPayload)
            awaiting = advance(
                workspace,
                active_interaction=payload.user_decision_request,
                decision_trace=(*workspace.decision_trace, decision),
                unresolved_decisions=payload.blocking_issue_ids,
                user_interrupt_count=workspace.user_interrupt_count + 1,
                status=PlannerStatus.AWAITING_USER,
            )
            await context.checkpoint(awaiting)
            return awaiting
        hotel_request = build_automatic_hotel_request(workspace, context.book)
        if hotel_request is not None:
            await context.progress(
                "planner_querying",
                "正在查询本次行程的真实酒店候选与区位信息。",
            )
            workspace = await self.evidence.execute_batch(
                (hotel_request,),
                workspace,
                context.book,
                context.cancellation,
            )
            await context.checkpoint(workspace)

        if workspace.planning_strategy is None:
            strategy_decision = compile_default_strategy_decision(workspace, context.book)
            strategy_payload = strategy_decision.payload
            assert isinstance(strategy_payload, BuildOrUpdateStrategyPayload)
            workspace = advance(
                workspace,
                planning_strategy=strategy_payload.proposed_strategy,
                decision_trace=(*workspace.decision_trace, strategy_decision),
            )
            await context.checkpoint(workspace)

        if self.estimate_visit_durations:
            from backend.agent.planner.visit_duration import ensure_visit_duration_estimates

            await context.progress(
                "planner_visit_duration", "正在结合地点规模与兴趣估算各景点的参观时长。"
            )
            workspace = await ensure_visit_duration_estimates(
                workspace, context.book, self.gateway, context.cancellation
            )
            await context.checkpoint(workspace)

        rejected_intent: ModelPlanIntent | None = None
        last_code = ""
        for attempt in range(3):
            if attempt == 2 and not (
                rejected_intent is not None
                and last_code.startswith("planner_plan_missing_required_restaurant:")
                and context.remaining_seconds() >= 20
            ):
                break
            context.cancellation.raise_if_cancelled("planner_compact_plan")
            workspace = advance(
                workspace,
                action_attempt_count=workspace.action_attempt_count + 1,
                segment_attempt_count=workspace.segment_attempt_count + 1,
            )
            await context.checkpoint(workspace)
            await context.progress(
                "planner_deciding",
                "千问正在综合全程节奏、空间与住宿形成逐日安排。",
            )
            call_id: str | None = None
            code: str | None = None
            try:
                if (
                    attempt
                    and rejected_intent
                    and last_code.startswith("planner_plan_missing_required_restaurant:")
                ):
                    key = last_code.split("candidate_key=", 1)[1].split(":", 1)[0]
                    async with asyncio.timeout(18):
                        intent, call_id = await self._assign_missing_meal(
                            rejected_intent, key, workspace, context
                        )
                else:
                    generated = await self._generate_structured_with_retry(
                        build_compact_plan_request(
                            workspace, context.book, rejected_intent=rejected_intent
                        ),
                        ModelPlanIntent,
                        context,
                    )
                    call_id = generated.audit_call_id
                    intent = ModelPlanIntent.model_validate(generated.value.model_dump(mode="json"))
                rejected_intent = intent
                decision = compile_plan_intent_decision(intent, workspace, context.book)
                payload = decision.payload
                assert isinstance(payload, MaterializeDraftPayload)
                validate_working_draft(
                    payload.proposed_working_draft,
                    workspace,
                    context.book,
                    self.clock(),
                )
                workspace = advance(
                    workspace,
                    working_itinerary=payload.proposed_working_draft,
                    selected_hotel=payload.selected_hotel,
                    hotel_recommendations=None,
                    decision_trace=(*workspace.decision_trace, decision),
                    unresolved_decisions=(),
                    status=PlannerStatus.DRAFT_READY,
                    timing_optimization_pending=self.optimize_timing,
                )
                await record_model_call_annotation(
                    self.gateway,
                    call_id,
                    "llm_business_guard",
                    {
                        "business_guard_result": {
                            "status": "accepted",
                            "guard": "planner_compact_plan_guard",
                        },
                        "accepted_or_rejected": "accepted",
                        "materialized_output": decision.model_dump(mode="json"),
                    },
                )
                await context.checkpoint(workspace)
                if self.complete_plan:
                    return await self._complete_plan(workspace, context)
                return workspace
            except ModelGatewayError as error:
                if error.code is not ModelFailureCode.MALFORMED_RESPONSE:
                    raise
                code = _model_schema_error_code(error)
                stage: GuardStage = "schema"
            except (PlannerGuardError, ValidationError, KeyError, ValueError) as error:
                code = (
                    error.code
                    if isinstance(error, PlannerGuardError)
                    else _safe_schema_error(error)
                )
                stage = "reference"
            await record_model_call_annotation(
                self.gateway,
                call_id,
                "llm_business_guard",
                {
                    "business_guard_result": {
                        "status": "rejected",
                        "guard": "planner_compact_plan_guard",
                        "reason": code,
                    },
                    "accepted_or_rejected": "rejected",
                    "failure_stage": stage,
                    "failure_code": code,
                },
            )
            workspace = self._rejected(
                workspace,
                "materialize_draft",
                code or "planner_compact_plan_invalid",
                stage=stage,
            )
            last_code = code or ""
            await context.checkpoint(workspace)
        workspace = advance(workspace, status=PlannerStatus.FAILED)
        await context.checkpoint(workspace)
        return workspace

    async def _assign_missing_meal(
        self,
        intent: ModelPlanIntent,
        key: str,
        workspace: PlannerWorkspaceState,
        context: PlannerGraphContext,
    ) -> tuple[ModelPlanIntent, str | None]:
        available = available_meal_assignments(intent, key, workspace)
        rejected_choice = None
        for _ in range(2):
            generated = await self.gateway.generate_structured(
                meal_assignment_request(intent, key, workspace, rejected_choice=rejected_choice),
                ModelMealAssignment,
                cancellation=context.cancellation,
            )
            if generated.value in available:
                return (
                    apply_meal_assignment(intent, key, generated.value, workspace),
                    generated.audit_call_id,
                )
            code = (
                "planner_meal_assignment_unavailable:path=day_index+meal_slot:allowed_values="
                + ",".join(f"{a.day_index}/{a.meal_slot}" for a in available)
            )
            await record_model_call_annotation(
                self.gateway,
                generated.audit_call_id,
                "llm_business_guard",
                {
                    "accepted_or_rejected": "rejected",
                    "failure_stage": "reference",
                    "failure_code": code,
                    "business_guard_result": {
                        "status": "rejected",
                        "guard": "planner_meal_assignment_guard",
                        "reason": code,
                    },
                },
            )
            rejected_choice = generated.value
        raise PlannerGuardError(code)

    async def apply_plan_change(
        self,
        workspace: PlannerWorkspaceState,
        context: PlannerGraphContext,
        *,
        change_request: PlanChangeRequest,
        user_text: str,
        refresh_evidence: bool = False,
    ) -> PlannerWorkspaceState:
        """Apply one validated user edit, then reuse the normal safety kernel."""

        if workspace.working_itinerary is None or workspace.planning_strategy is None:
            raise PlannerGuardError("planner_plan_change_workspace_incomplete")
        if workspace.plan_change_request != change_request:
            workspace = advance(workspace, plan_change_request=change_request)
            await context.checkpoint(workspace)
        local_dates = workspace.accepted_local_change_dates
        if local_dates:
            return await self._complete_plan(
                workspace, replace(context, local_change_dates=local_dates)
            )
        if change_request.requested_scope == "full_replan":
            workspace = await self._refresh_plan_change_evidence(
                workspace,
                context,
                preserve_semantic_artifacts=False,
                invalidate_hotel=True,
            )
            return await self.invoke(workspace, context)
        if refresh_evidence:
            workspace = await self._refresh_plan_change_evidence(
                workspace,
                context,
                preserve_semantic_artifacts=True,
                invalidate_hotel=False,
            )
        feedback: dict[str, object] | None = None
        accepted_dates: tuple[date, ...] | None = None
        rejected_intent: ModelPlanChangePatchIntent | None = None
        for attempt in range(2):
            context.cancellation.raise_if_cancelled("planner_plan_change")
            await context.progress(
                "planner_revising",
                "正在按你的要求计算最小修改范围。",
            )
            call_id: str | None = None
            try:
                generated = await self._generate_structured_with_retry(
                    build_plan_change_patch_request(
                        workspace,
                        context.book,
                        change_request,
                        user_text,
                        guard_feedback=feedback,
                        rejected_intent=rejected_intent,
                    ),
                    ModelPlanChangePatchIntent,
                    context,
                )
                call_id = generated.audit_call_id
                rejected_intent = generated.value
                decision = compile_model_plan_change_intent(
                    generated.value,
                    change_request,
                    workspace,
                )
                payload = decision.payload
                assert isinstance(payload, ReviseDraftPayload)
                authority = payload.itinerary_patch.authority
                if not isinstance(authority, PlanChangeRequestPatchAuthority):
                    raise PlannerGuardError("planner_plan_change_patch_authority_invalid")
                current_draft = workspace.working_itinerary
                if current_draft is None:
                    raise PlannerGuardError("planner_plan_change_workspace_incomplete")
                revised = atomic_apply_itinerary_patch(
                    current_draft,
                    payload.itinerary_patch,
                    candidate_pool=workspace.candidate_pool,
                    allowed_authority_refs=frozenset(change_request.semantic_operation_ids),
                )
                workspace = advance(
                    workspace,
                    working_itinerary=revised,
                    selected_hotel=(
                        payload.selected_hotel
                        if payload.selected_hotel is not None
                        else None
                        if payload.hotel_recommendations is not None
                        else workspace.selected_hotel
                    ),
                    hotel_recommendations=(
                        payload.hotel_recommendations
                        if payload.hotel_recommendations is not None
                        else None
                        if payload.selected_hotel is not None
                        else workspace.hotel_recommendations
                    ),
                    decision_trace=(*workspace.decision_trace, decision),
                    revision_round=0,
                    **invalidate_materialized_dependencies(
                        workspace,
                        affected_dates=payload.declared_affected_dates,
                    ),
                    status=PlannerStatus.PLANNING,
                )
                await record_model_call_annotation(
                    self.gateway,
                    call_id,
                    "llm_business_guard",
                    {
                        "business_guard_result": {
                            "status": "accepted",
                            "guard": "planner_plan_change_guard",
                        },
                        "accepted_or_rejected": "accepted",
                        "materialized_output": decision.model_dump(mode="json"),
                    },
                )
                await context.checkpoint(workspace)
                accepted_dates = payload.declared_affected_dates
                break
            except ModelGatewayError as error:
                if error.code is not ModelFailureCode.MALFORMED_RESPONSE:
                    raise
                code = _model_schema_error_code(error)
                stage: GuardStage = "schema"
            except (PlannerGuardError, ValidationError, ValueError, KeyError) as error:
                code = (
                    error.code
                    if isinstance(error, PlannerGuardError)
                    else _safe_schema_error(error)
                )
                stage = "reference"
            await record_model_call_annotation(
                self.gateway,
                call_id,
                "llm_business_guard",
                {
                    "business_guard_result": {
                        "status": "rejected",
                        "guard": "planner_plan_change_guard",
                        "reason": code,
                    },
                    "accepted_or_rejected": "rejected",
                    "failure_stage": stage,
                    "failure_code": code,
                },
            )
            violation = compile_guard_violation(
                workspace,
                action="revise_draft",
                stage=stage,
                raw_code=code,
            )
            feedback = violation.model_dump(mode="json")
            workspace = self._rejected(
                workspace,
                "revise_draft",
                code,
                stage=stage,
            )
            if attempt == 1 or code in {
                "planner_plan_change_prerequisites_missing",
                "planner_plan_change_scope_not_patchable",
                "planner_plan_change_trip_mismatch",
                "planner_plan_change_authority_missing",
            }:
                workspace = advance(workspace, status=PlannerStatus.FAILED)
            await context.checkpoint(workspace)
            if workspace.status is PlannerStatus.FAILED:
                break
        if accepted_dates is not None:
            # Completion errors are not malformed user choices. In particular,
            # never apply a successful move/replace a second time after a route,
            # materialization, or validation failure.
            local_context = replace(context, local_change_dates=frozenset(accepted_dates))
            return await self._complete_plan(workspace, local_context)
        return workspace

    async def _refresh_plan_change_evidence(
        self,
        workspace: PlannerWorkspaceState,
        context: PlannerGraphContext,
        *,
        preserve_semantic_artifacts: bool,
        invalidate_hotel: bool,
    ) -> PlannerWorkspaceState:
        """Refresh changed planning inputs before patching or rebuilding a plan."""

        previous = workspace
        staged = advance(
            prepare_workspace_for_evidence_refresh(
                workspace,
                invalidate_hotel=invalidate_hotel,
                reset_initial_evidence=True,
            )
        )
        if not preserve_semantic_artifacts:
            # A new complete plan must retry the original intentions. Old
            # omissions and notices belong to the previous published version.
            staged = advance(
                staged,
                decision_trace=(),
                recovery_observations=(),
                recovery_omissions=(),
                best_effort_reasons=natural_day_limitations(context.book),
            )
        await context.checkpoint(staged)
        await context.progress(
            "planner_evidence",
            "正在按已确认的修改刷新候选、路线和住宿证据。",
        )
        refreshed = await self.evidence.initialize(
            staged,
            context.book,
            context.cancellation,
            checkpoint=context.checkpoint,
        )
        if preserve_semantic_artifacts:
            refreshed = rebind_semantic_artifacts_after_evidence(
                previous, refreshed, refresh_clusters=True
            )
        await context.checkpoint(refreshed)
        return refreshed

    async def _initialize(
        self, state: PlannerGraphState, runtime: Runtime[PlannerGraphContext]
    ) -> PlannerGraphState:
        context = runtime.context
        workspace = state["workspace"]
        context.cancellation.raise_if_cancelled("planner_initial_evidence")
        if not workspace.initial_evidence_ready:
            await context.progress(
                "planner_evidence", "正在核验任务书地点、营业规则和真实路线分区。"
            )
            workspace = await self.evidence.initialize(
                workspace,
                context.book,
                context.cancellation,
                checkpoint=context.checkpoint,
            )
            await context.checkpoint(workspace)
        elif workspace.decision_trace and isinstance(
            workspace.decision_trace[-1].payload, RequestEvidencePayload
        ):
            pending = workspace.decision_trace[-1].payload.capability_requests
            observed = {item.request_id for item in workspace.capability_observations}
            if not any(request.request_id in observed for request in pending):
                # A crash after request checkpoint is resumed with the same typed
                # read-only requests. Never replay a synthetic tool result.
                workspace = await self.evidence.execute_batch(
                    pending,
                    workspace,
                    context.book,
                    context.cancellation,
                )
                await context.checkpoint(workspace)
        return {"workspace": workspace}

    async def _decide(
        self, state: PlannerGraphState, runtime: Runtime[PlannerGraphContext]
    ) -> PlannerGraphState:
        context = runtime.context
        context.cancellation.raise_if_cancelled("planner_decide")
        workspace = state["workspace"]
        if workspace.segment_attempt_count >= 12:
            workspace = self._rejected(
                workspace, "planner_decide", "planner_decision_budget_exhausted"
            )
            workspace = advance(workspace, status=PlannerStatus.FAILED)
            await context.checkpoint(workspace)
            return {"workspace": workspace, "decision": None, "done": True}
        workspace = advance(
            workspace,
            action_attempt_count=workspace.action_attempt_count + 1,
            segment_attempt_count=workspace.segment_attempt_count + 1,
        )
        await context.checkpoint(workspace)
        await context.progress("planner_deciding", "千问正在结合任务书和已查询的证据决定下一步。")
        proposal: ModelPlannerDecision | None = None
        proposal_call_id: str | None = None
        try:
            request = build_planner_decision_request(
                workspace, context.book, rejected_proposal=state.get("proposal")
            )
            if workspace.planning_strategy is None:
                # The documented first action is a complete strategy. Expose
                # only that legal contract instead of inviting an invalid tool
                # request and spending another paid model call to reject it.
                initial = await self._generate_structured_with_retry(
                    request, ModelStrategyDecision, context
                )
                proposal_call_id = initial.audit_call_id
                proposal = ModelPlannerDecision(root=initial.value)
            elif (
                workspace.readiness_observation is None
                or not workspace.readiness_observation.issues
            ):
                without_issues = await self._generate_structured_with_retry(
                    request,
                    ModelPlannerDecisionAfterStrategyWithoutIssues,
                    context,
                )
                proposal_call_id = without_issues.audit_call_id
                proposal = ModelPlannerDecision.model_validate(
                    without_issues.value.model_dump(mode="json")
                )
            else:
                generated = await self._generate_structured_with_retry(
                    request,
                    ModelPlannerDecisionAfterStrategyWithIssues,
                    context,
                )
                proposal_call_id = generated.audit_call_id
                proposal = ModelPlannerDecision.model_validate(
                    generated.value.model_dump(mode="json")
                )
            decision = resolve_model_decision(proposal, workspace)
            if workspace.planning_strategy is None and not isinstance(
                decision.payload, BuildOrUpdateStrategyPayload
            ):
                raise PlannerGuardError("planner_complete_strategy_required_first")
            return {
                "workspace": workspace,
                "decision": decision,
                "proposal": proposal,
                "proposal_call_id": proposal_call_id,
                "done": False,
            }
        except ModelGatewayError as error:
            if error.code is not ModelFailureCode.MALFORMED_RESPONSE:
                raise
            code = _model_schema_error_code(error)
        except (PlannerGuardError, ValidationError, KeyError, ValueError) as error:
            code = error.code if isinstance(error, PlannerGuardError) else _safe_schema_error(error)
            await record_model_call_annotation(
                self.gateway,
                proposal_call_id,
                "llm_business_guard",
                {
                    "business_guard_result": {
                        "status": "rejected",
                        "guard": "planner_decision_resolution",
                        "reason": code,
                    },
                    "accepted_or_rejected": "rejected",
                    "failure_stage": "business_guard",
                    "failure_code": code,
                    "materialized_output": (
                        proposal.model_dump(mode="json") if proposal is not None else None
                    ),
                },
            )
        terminal_failure = _is_terminal_hotel_failure(code, workspace)
        if terminal_failure:
            code = "planner_required_hotel_evidence_unavailable"
        workspace = self._rejected(
            workspace,
            "planner_decide",
            code,
            stage=("schema" if code.startswith("planner_model_schema_invalid") else "reference"),
        )
        repeated_failure = repeated_guard_failure(workspace)
        if terminal_failure or repeated_failure:
            workspace = advance(workspace, status=PlannerStatus.FAILED)
        await context.checkpoint(workspace)
        return {
            "workspace": workspace,
            "decision": None,
            "proposal": proposal,
            "proposal_call_id": proposal_call_id,
            "done": terminal_failure or repeated_failure,
        }

    async def _generate_structured_with_retry(
        self,
        request: ModelRequest,
        output_type: type[StructuredValue],
        context: PlannerGraphContext,
    ) -> ModelStructuredResult[StructuredValue]:
        """Retry one transient transport failure without creating another Agent action."""

        for attempt in range(PLANNER_MODEL_MAX_ATTEMPTS):
            try:
                return await self.gateway.generate_structured(
                    request,
                    output_type,
                    cancellation=context.cancellation,
                )
            except ModelGatewayError as error:
                if not error.retryable or attempt + 1 >= PLANNER_MODEL_MAX_ATTEMPTS:
                    raise
                await context.progress(
                    "planner_model_retry",
                    "千问服务暂时不可用，正在重试本次决策。",
                )
                await asyncio.sleep(PLANNER_MODEL_RETRY_DELAY_SECONDS)
                context.cancellation.raise_if_cancelled("planner_model_retry")
        raise AssertionError("planner model retry loop exhausted")  # pragma: no cover

    async def _execute(
        self, state: PlannerGraphState, runtime: Runtime[PlannerGraphContext]
    ) -> PlannerGraphState:
        context = runtime.context
        workspace, decision = state["workspace"], state.get("decision")
        proposal_call_id = state.get("proposal_call_id")
        if decision is None:
            return {"done": False}
        payload = decision.payload
        context.cancellation.raise_if_cancelled("planner_execute")
        rejected = False
        try:
            if isinstance(payload, BuildOrUpdateStrategyPayload):
                validate_strategy(payload.proposed_strategy, workspace, context.book)
                workspace = advance(
                    workspace,
                    planning_strategy=payload.proposed_strategy,
                    decision_trace=(*workspace.decision_trace, decision),
                )
            elif isinstance(payload, RequestEvidencePayload):
                self.evidence.validate_batch(payload.capability_requests, workspace, context.book)
                # Checkpoint the exact accepted requests before any external call.
                workspace = advance(workspace, decision_trace=(*workspace.decision_trace, decision))
                await context.checkpoint(workspace)
                await context.progress(
                    "planner_querying", "千问已提出证据需求，正在调用真实查询能力。"
                )
                workspace = await self.evidence.execute_batch(
                    payload.capability_requests,
                    workspace,
                    context.book,
                    context.cancellation,
                )
                hotel = workspace.hotel_observation
                if (
                    hotel is not None
                    and hotel.status == "unavailable"
                    and "provider_city_binding" in hotel.missing_fact_kinds
                ):
                    raise PlannerGuardError("planner_required_hotel_provider_unavailable")
            elif isinstance(payload, MaterializeDraftPayload):
                validate_working_draft(
                    payload.proposed_working_draft, workspace, context.book, self.clock()
                )
                workspace = advance(
                    workspace,
                    working_itinerary=payload.proposed_working_draft,
                    selected_hotel=payload.selected_hotel,
                    hotel_recommendations=payload.hotel_recommendations,
                    decision_trace=(*workspace.decision_trace, decision),
                    unresolved_decisions=(),
                    status=PlannerStatus.DRAFT_READY,
                )
            elif isinstance(payload, AskUserPayload):
                guard_ask_user(decision, workspace)
                workspace = advance(
                    workspace,
                    active_interaction=payload.user_decision_request,
                    decision_trace=(*workspace.decision_trace, decision),
                    unresolved_decisions=payload.blocking_issue_ids,
                    user_interrupt_count=workspace.user_interrupt_count + 1,
                    status=PlannerStatus.AWAITING_USER,
                )
            else:
                raise PlannerGuardError("planner_action_not_available_in_initial_draft_loop")
        except (PlannerGuardError, ValidationError, ValueError, KeyError) as error:
            rejected = True
            code = error.code if isinstance(error, PlannerGuardError) else _safe_schema_error(error)
            terminal_failure = _is_terminal_hotel_failure(code, workspace)
            if terminal_failure:
                code = "planner_required_hotel_evidence_unavailable"
            workspace = self._rejected(
                workspace,
                decision.action,
                code,
                stage="business",
            )
            if terminal_failure or repeated_guard_failure(workspace):
                workspace = advance(workspace, status=PlannerStatus.FAILED)
        await record_model_call_annotation(
            self.gateway,
            proposal_call_id,
            "llm_business_guard",
            {
                "business_guard_result": {
                    "status": "rejected" if rejected else "accepted",
                    "guard": "planner_action_guard",
                    "reason": code if rejected else None,
                },
                "accepted_or_rejected": "rejected" if rejected else "accepted",
                "failure_stage": "business_guard" if rejected else None,
                "failure_code": code if rejected else None,
                "materialized_output": {
                    "decision": decision.model_dump(mode="json"),
                    "workspace": workspace.model_dump(mode="json"),
                },
            },
        )
        await context.checkpoint(workspace)
        return {
            "workspace": workspace,
            "decision": None,
            # A typed but rejected proposal is transient repair context only.
            # It is never a committed artifact, a fact, or an accepted decision.
            "proposal": state.get("proposal") if rejected else None,
            "proposal_call_id": proposal_call_id if rejected else None,
            "done": workspace.status
            in {PlannerStatus.DRAFT_READY, PlannerStatus.AWAITING_USER, PlannerStatus.FAILED},
        }

    async def _complete_plan(
        self,
        workspace: PlannerWorkspaceState,
        context: PlannerGraphContext,
    ) -> PlannerWorkspaceState:
        """Run the fixed safety kernel around narrow, model-owned semantic repairs."""

        while True:
            context.cancellation.raise_if_cancelled("planner_complete_plan")
            if workspace.status in {
                PlannerStatus.AWAITING_USER,
                PlannerStatus.FAILED,
                PlannerStatus.CANCELLED,
                PlannerStatus.STALE,
                PlannerStatus.READY_TO_PUBLISH,
            }:
                return workspace

            if workspace.materialized_schedule is None or workspace.cost_draft is None:
                if isinstance(self.evidence, PlannerEvidenceBackend):
                    await context.progress("planner_hours", "正在核验已选景点和餐厅的营业时间。")
                    workspace = await self.evidence.ensure_selected_hours(
                        workspace,
                        context.book,
                        context.cancellation,
                    )
                    await context.progress("planner_prices", "正在补充已选景点的参考票价。")
                    workspace = await self.evidence.ensure_selected_prices(
                        workspace,
                        context.book,
                        context.cancellation,
                    )
                    await context.checkpoint(workspace)
                if selected_itinerary_route_requests(workspace, context.book):
                    await context.progress(
                        "planner_selected_routes",
                        "正在核验正式日程中酒店与每日实际相邻地点的路线。",
                    )
                    try:
                        workspace = await self.evidence.ensure_selected_itinerary_routes(
                            workspace,
                            context.book,
                            context.cancellation,
                        )
                    except (PlannerGuardError, ValidationError, ValueError, KeyError) as error:
                        return await self._fail_completion_stage(
                            workspace,
                            context,
                            action="materialize_draft",
                            stage="materialize",
                            error=error,
                        )
                    await context.checkpoint(workspace)
                await context.progress(
                    "planner_materializing",
                    "正在把每日选择编译成精确时间轴、交通段和费用覆盖。",
                )
                try:
                    materialized = await self.materializer.materialize(
                        workspace,
                        context.book,
                        context.cancellation,
                        input_state_version=context.input_state_version,
                    )
                except (PlannerGuardError, ValidationError, ValueError, KeyError) as error:
                    return await self._fail_completion_stage(
                        workspace,
                        context,
                        action="materialize_draft",
                        stage="materialize",
                        error=error,
                    )
                workspace = advance(
                    workspace,
                    materialized_schedule=materialized.schedule,
                    cost_draft=materialized.cost,
                    validation_report=None,
                    validation_observation=None,
                    status=PlannerStatus.PLANNING,
                )
                await context.checkpoint(workspace)

            if workspace.validation_observation is None or workspace.validation_report is None:
                await context.progress(
                    "planner_validating",
                    "正在并行核验营业、路线、节奏、住宿与预算。",
                )
                try:
                    validated = await self.validator.validate(
                        workspace,
                        context.book,
                        context.cancellation,
                    )
                except (PlannerGuardError, ValidationError, ValueError, KeyError) as error:
                    return await self._fail_completion_stage(
                        workspace,
                        context,
                        action="validate_draft",
                        stage="validate",
                        error=error,
                    )
                workspace = advance(
                    workspace,
                    validation_report=validated.legacy_report,
                    validation_observation=validated.observation,
                    status=PlannerStatus.PLANNING,
                )
                await context.checkpoint(workspace)

            observation = workspace.validation_observation
            assert observation is not None
            if (
                workspace.timing_optimization_pending
                and context.allow_semantic_repair
                and observation.result not in {"requires_user", "fatal"}
                and not any(
                    issue.severity in {"error", "blocking"}
                    and issue.code
                    not in {"meal_constraint_violation", "time_overlap", "opening_conflict"}
                    for issue in observation.issues
                )
            ):
                # Meal/clock/opening errors need a semantic recovery opportunity too,
                # not just plans that already pass validation.
                workspace = await self._optimize_time_quality(workspace, context)
                observation = workspace.validation_observation
                assert observation is not None
            if observation.result == "passed":
                # Final Guard re-runs the complete deterministic validator from
                # current artifacts immediately before binding publication refs.
                await context.progress(
                    "planner_final_guard",
                    "正在执行发布前最终完整校验。",
                )
                try:
                    final_validation = await self.validator.validate(
                        workspace,
                        context.book,
                        context.cancellation,
                    )
                except (PlannerGuardError, ValidationError, ValueError, KeyError) as error:
                    return await self._fail_completion_stage(
                        workspace,
                        context,
                        action="propose_finalize",
                        stage="final",
                        error=error,
                    )
                workspace = advance(
                    workspace,
                    validation_report=final_validation.legacy_report,
                    validation_observation=final_validation.observation,
                    status=PlannerStatus.PLANNING,
                )
                await context.checkpoint(workspace)
                if final_validation.observation.result != "passed":
                    continue
                try:
                    decision = build_finalize_decision(workspace)
                    workspace = advance(
                        workspace,
                        decision_trace=(*workspace.decision_trace, decision),
                        status=PlannerStatus.READY_TO_PUBLISH,
                    )
                except (PlannerGuardError, ValidationError, ValueError, KeyError) as error:
                    return await self._fail_completion_stage(
                        workspace,
                        context,
                        action="propose_finalize",
                        stage="final",
                        error=error,
                    )
                await context.checkpoint(workspace)
                return workspace

            if observation.result == "requires_user":
                try:
                    decision = compile_validation_interaction(workspace)
                    guard_ask_user(decision, workspace)
                    payload = decision.payload
                    assert isinstance(payload, AskUserPayload)
                    workspace = advance(
                        workspace,
                        active_interaction=payload.user_decision_request,
                        decision_trace=(*workspace.decision_trace, decision),
                        unresolved_decisions=payload.blocking_issue_ids,
                        user_interrupt_count=workspace.user_interrupt_count + 1,
                        status=PlannerStatus.AWAITING_USER,
                    )
                except (PlannerGuardError, ValidationError, ValueError, KeyError) as error:
                    return await self._fail_completion_stage(
                        workspace,
                        context,
                        action="ask_user",
                        stage="validate",
                        error=error,
                    )
                await context.checkpoint(workspace)
                return workspace

            if not context.allow_semantic_repair:
                # A route-menu choice authorizes recalculation, not replacing
                # attractions, meals or the hotel to make a slower leg fit.
                failed = self._rejected(
                    workspace,
                    "revise_draft",
                    "planner_transport_selection_conflicts_with_schedule",
                    stage="validate",
                )
                failed = advance(failed, status=PlannerStatus.FAILED)
                await context.checkpoint(failed)
                return failed

            if observation.result == "fatal" or workspace.revision_round >= 2:
                code = (
                    "planner_validation_fatal"
                    if observation.result == "fatal"
                    else "planner_revision_budget_exhausted"
                )
                failed = self._rejected(
                    workspace,
                    "revise_draft",
                    code,
                    stage="validate",
                )
                failed = advance(failed, status=PlannerStatus.FAILED)
                await context.checkpoint(failed)
                return failed

            if workspace.schedule_repair_state is not None and any(
                issue.kind == "hard_time" and issue.status != "resolved"
                for issue in workspace.schedule_repair_state.issues
            ):
                # The daily controller already owns these conflicts and their
                # two-attempt limit. Do not restart the legacy repair loop;
                # bounded recovery may remove an unusable adjustable item and
                # then remeasure the resulting, genuinely new gaps.
                failed = self._rejected(
                    workspace,
                    "revise_draft",
                    "planner_schedule_repair_exhausted",
                    stage="validate",
                )
                failed = advance(failed, status=PlannerStatus.FAILED)
                await context.checkpoint(failed)
                return failed

            workspace, repair_decision, proposal_call_id = await self._propose_repair(
                workspace,
                context,
            )
            if repair_decision is None:
                if workspace.status is PlannerStatus.FAILED:
                    return workspace
                continue
            workspace, accepted = await self._execute_repair(
                workspace,
                repair_decision,
                proposal_call_id,
                context,
            )
            if not accepted and workspace.status is PlannerStatus.FAILED:
                return workspace

    async def _optimize_time_quality(
        self,
        workspace: PlannerWorkspaceState,
        context: PlannerGraphContext,
    ) -> PlannerWorkspaceState:
        """Drain measured daily issues fairly; improvements are not completion."""
        current = advance(workspace, timing_optimization_pending=False)
        current = refresh_repair_state(current, context.book)
        await context.checkpoint(current)
        seen: set[str] = set()
        feedback: list[dict[str, Any]] = []
        # Failures survive checkpoints, including same-turn timeout recovery.
        if current.schedule_repair_state:
            seen.update(
                attempt.fingerprint
                for issue in current.schedule_repair_state.issues
                for attempt in issue.attempts
                if attempt.outcome != "pending"
            )
            catalog = PlannerReferenceCatalog(current)
            keys = {
                entry.candidate_ref.canonical_entity_id: key
                for key, entry in catalog.candidates.items()
            }
            feedback.extend(
                {
                    "rejected_gap_choices": [
                        {
                            "date": str(issue.service_date),
                            "candidate_key": keys[identity],
                            "path": issue.field_path,
                            "reason": attempt.failure_code or "此前方案未通过实际排程",
                        }
                    ]
                }
                for issue in current.schedule_repair_state.issues
                for attempt in issue.attempts
                for identity in attempt.rejected_candidate_ids
                if identity in keys
            )
        stop_reason = None
        for _ in range(40):
            current = refresh_repair_state(current, context.book)
            assert current.schedule_repair_state is not None
            batch = next_repair_batch(current.schedule_repair_state, context.local_change_dates)
            if not batch:
                stop_reason = (
                    "attempts_exhausted"
                    if any(i.status == "exhausted" for i in current.schedule_repair_state.issues)
                    else "no_feasible_candidate"
                )
                break
            if context.call_remaining_seconds() < 8:
                stop_reason = "budget_exhausted"
                break
            dates = frozenset(i.service_date for i in batch)
            family = batch[0].kind
            day_fallback = any(
                i.attempts
                and i.attempts[-1].action == "gap_choice"
                and i.attempts[-1].failure_code == "requires_day_repair"
                for i in batch
            )
            options = [
                gap
                for gap in gap_visit_options(current, context.book)
                if any(gap_matches_issue(gap, issue) for issue in batch)
            ]
            # At most one actual interval per issue, and no more than three in a call.
            options = list({str(gap["date"]): gap for gap in reversed(options)}.values())[:3]
            needs_discovery = family in {"coverage", "gap", "evening"} and not options
            if (
                needs_discovery
                and family != "hard_time"
                and isinstance(self.evidence, PlannerEvidenceBackend)
                and context.call_remaining_seconds() >= 35
            ):
                await context.progress(
                    "planner_schedule_choices", "正在补查待完善日期附近的真实地点。"
                )
                async with asyncio.timeout(context.call_remaining_seconds()):
                    expanded = await self.evidence.supplement_schedule_choices(
                        current,
                        context.book,
                        context.cancellation,
                        attraction_dates=set(dates)
                        if family in {"coverage", "gap", "evening"}
                        else set(),
                        dining_dates=set(dates) if family in {"meal", "dining_route"} else set(),
                    )
                    if expanded is not current:
                        if self.estimate_visit_durations and context.call_remaining_seconds() > 8:
                            from backend.agent.planner.visit_duration import (
                                ensure_visit_duration_estimates,
                            )

                            expanded = await ensure_visit_duration_estimates(
                                expanded, context.book, self.gateway, context.cancellation
                            )
                        current = await self._remeasure_time_quality(expanded, context)
                        current = refresh_repair_state(current, context.book)
                        await context.checkpoint(current)
                        options = [
                            gap
                            for gap in gap_visit_options(current, context.book)
                            if any(gap_matches_issue(gap, issue) for issue in batch)
                        ]
                        options = list(
                            {str(gap["date"]): gap for gap in reversed(options)}.values()
                        )[:3]
            if (
                options
                and family in {"coverage", "gap", "evening"}
                and isinstance(self.evidence, PlannerEvidenceBackend)
                and context.call_remaining_seconds() > 12
            ):
                catalog = PlannerReferenceCatalog(current)
                targets: dict[CandidateRef, set[date]] = {}
                for gap in options:
                    for option in gap["options"][:8]:
                        if option.get("operation") in {"add_visit", "replace_existing_visit"}:
                            ref = catalog.candidates[option["candidate_key"]].candidate_ref
                            targets.setdefault(ref, set()).add(date.fromisoformat(gap["date"]))
                async with asyncio.timeout(context.call_remaining_seconds()):
                    checked = await self.evidence.ensure_candidate_hours(
                        current, context.book, context.cancellation, targets
                    )
                if checked is not current:
                    current = await self._remeasure_time_quality(checked, context)
                    current = refresh_repair_state(current, context.book)
                    await context.checkpoint(current)
                # Identity keys remain tied to the new pool, not the old query revision.
                options = [
                    gap
                    for gap in gap_visit_options(current, context.book)
                    if any(gap_matches_issue(gap, issue) for issue in batch)
                ]
                options = list({str(gap["date"]): gap for gap in reversed(options)}.values())[:3]
            if family in {"coverage", "gap", "evening"} and not options and not day_fallback:
                current = finish_repair_batch(current, batch, [], unavailable=True)
                await context.checkpoint(current)
                continue
            action = (
                "day_repair"
                if family == "hard_time" or day_fallback
                else "meal_completion"
                if family == "meal"
                else "dining_repair"
                if family == "dining_route"
                else "gap_choice"
            )
            current = begin_repair_batch(current, batch, action)
            await context.checkpoint(current)
            feedback_start = len(feedback)
            try:
                async with asyncio.timeout(max(0.1, context.call_remaining_seconds())):
                    meal_targets = dining_slots(
                        current, context.book, allowed_dates=dates, mode=family
                    )
                    if meal_targets and isinstance(self.evidence, PlannerEvidenceBackend):
                        current = await self._repair_dining_slots(
                            current, context, seen, feedback, dates=dates, mode=family
                        )
                    else:
                        current = await self._try_time_quality(
                            current,
                            context,
                            seen,
                            feedback,
                            compact_infill=family in {"coverage", "gap", "evening"}
                            and not day_fallback,
                            dining_only=family == "dining_route",
                            missing_meals_only=family == "meal",
                            long_visit_meal=family == "hard_time"
                            and long_visit_meal_options(current, allowed_dates=dates) is not None,
                            selected_gap_options=options,
                            repair_dates=dates,
                        )
            except TimeoutError:
                # The enclosing recovery callback retains the last verified checkpoint.
                raise
            current = refresh_repair_state(current, context.book)
            current = finish_repair_batch(current, batch, feedback[feedback_start:])
            await context.checkpoint(current)
        else:
            stop_reason = "attempts_exhausted"
        current = refresh_repair_state(current, context.book, stop_reason=stop_reason)
        await context.checkpoint(current)
        return current

    async def _remeasure_time_quality(
        self, workspace: PlannerWorkspaceState, context: PlannerGraphContext
    ) -> PlannerWorkspaceState:
        materialized = await self.materializer.materialize(
            workspace,
            context.book,
            context.cancellation,
            input_state_version=context.input_state_version,
        )
        workspace = advance(
            workspace, materialized_schedule=materialized.schedule, cost_draft=materialized.cost
        )
        validated = await self.validator.validate(workspace, context.book, context.cancellation)
        return advance(
            workspace,
            validation_report=validated.legacy_report,
            validation_observation=validated.observation,
        )

    async def refresh_recovery_evidence(
        self, workspace: PlannerWorkspaceState, book: TaskBookV4, cancellation: ModelCancellation
    ) -> PlannerWorkspaceState:
        if execution_remaining(workspace, calls=True) <= 0:
            return workspace
        try:
            async with asyncio.timeout(execution_remaining(workspace, calls=True)):
                if isinstance(self.evidence, PlannerEvidenceBackend):
                    workspace = await self.evidence.ensure_selected_hours(
                        workspace, book, cancellation
                    )
                if execution_remaining(
                    workspace, calls=True
                ) > 0 and selected_itinerary_route_requests(workspace, book):
                    workspace = await self.evidence.ensure_selected_itinerary_routes(
                        workspace, book, cancellation
                    )
        except TimeoutError:
            pass  # No invented edge: materialization retains unavailable route evidence.
        return workspace

    async def repair_recovery_schedule(
        self, workspace: PlannerWorkspaceState, context: PlannerGraphContext
    ) -> PlannerWorkspaceState:
        if not self.optimize_timing or not context.allow_semantic_repair:
            return workspace
        if context.call_remaining_seconds() < 8:
            return refresh_repair_state(workspace, context.book, stop_reason="budget_exhausted")

        latest_valid = workspace

        async def provisional_checkpoint(value: PlannerWorkspaceState) -> None:
            nonlocal latest_valid
            # Persist attempts only against a measured, valid schedule. On
            # timeout return exactly the latest saved candidate, never a stale revision.
            context.cancellation.raise_if_cancelled("planner_recovery_repair")
            if (
                value.validation_observation is not None
                and value.validation_observation.result == "passed"
                and value.materialized_schedule is not None
                and value.validation_observation.materialized_schedule_id
                == str(value.materialized_schedule.request_id)
            ):
                latest_valid = value
                await context.checkpoint(value)

        try:
            async with asyncio.timeout(max(0.1, context.call_remaining_seconds())):
                return await self._optimize_time_quality(
                    workspace, replace(context, checkpoint=provisional_checkpoint)
                )
        except TimeoutError:
            return refresh_repair_state(latest_valid, context.book, stop_reason="budget_exhausted")

    async def _repair_dining_slots(
        self,
        workspace: PlannerWorkspaceState,
        context: PlannerGraphContext,
        attempted_intents: set[str],
        feedback: list[dict[str, Any]],
        *,
        dates: frozenset[date],
        mode: str,
    ) -> PlannerWorkspaceState:
        """Each meal commits independently; rejected proposals cannot roll back siblings."""
        assert isinstance(self.evidence, PlannerEvidenceBackend)
        current = workspace
        targets = dining_slots(current, context.book, allowed_dates=dates, mode=mode)
        for original in targets:
            # Recompute endpoints after every accepted meal, never reuse a stale route.
            slot = next(
                (
                    item
                    for item in dining_slots(current, context.book, allowed_dates=dates, mode=mode)
                    if item.slot_key == original.slot_key
                ),
                None,
            )
            if slot is None:
                continue
            failures: list[str] = [
                item.code
                for item in current.guard_observations
                if item.observation_id
                in {
                    slot_choice_id(current, slot.slot_key, number)
                    for number in range(1, MAX_SLOT_CHOICES + 1)
                }
            ]
            observed_ids: set[str] = set()
            for page in range(1, MAX_SLOT_PAGES + 1):
                if (
                    min(context.call_remaining_seconds(), execution_remaining(current, calls=True))
                    <= 8
                ):
                    return current
                markers = {item.observation_id for item in current.guard_observations}
                attempt = next(
                    (
                        number
                        for number in range(1, MAX_SLOT_CHOICES + 1)
                        if slot_choice_id(current, slot.slot_key, number) not in markers
                    ),
                    None,
                )
                if attempt is None:
                    break
                assert current.working_itinerary is not None
                selected = {
                    item.object_ref.canonical_entity_id
                    for day in current.working_itinerary.days
                    for item in day.ordered_items
                    if isinstance(item.object_ref, CandidateRef)
                }

                async def search_checkpoint(value: PlannerWorkspaceState) -> None:
                    nonlocal current
                    current = value
                    await context.checkpoint(value)

                try:
                    async with asyncio.timeout(min(15, context.call_remaining_seconds())):
                        searched = await supplement_dining_slot(
                            self.evidence,
                            current,
                            context.book,
                            context.cancellation,
                            slot_key=slot.slot_key,
                            service_date=slot.service_date,
                            before=slot.before,
                            after=slot.after,
                            page=page,
                            excluded_ids=selected | observed_ids,
                            checkpoint=search_checkpoint,
                        )
                except TimeoutError:
                    # A slow individual page consumes that page, not other meals'
                    # already committed improvements or their remaining budget.
                    failures.append("provider_timeout")
                    feedback.append({"failure_code": "planner_dining_provider_timeout"})
                    continue
                current = searched.workspace
                await context.checkpoint(current)
                if searched.failure_code:
                    failures.append(searched.failure_code)
                if searched.terminal:
                    feedback.append({"failure_code": f"planner_dining_{searched.failure_code}"})
                    break
                if not searched.candidates:
                    continue
                observed_ids.update(place.canonical_entity_id for place in searched.candidates)
                if (
                    min(context.call_remaining_seconds(), execution_remaining(current, calls=True))
                    <= 8
                ):
                    return current
                marker = slot_choice_id(current, slot.slot_key, attempt)
                current = advance(
                    current,
                    guard_observations=(
                        *current.guard_observations,
                        PlannerGuardObservation(
                            observation_id=marker,
                            attempted_action="dining_slot_choice",
                            code="planner_dining_choice_started",
                            message=slot.slot_key,
                            based_on_workspace_revision=current.workspace_revision,
                        ),
                    ),
                )
                await context.checkpoint(current)
                try:
                    remaining = min(
                        context.call_remaining_seconds(), execution_remaining(current, calls=True)
                    )
                    if remaining <= 0:
                        return current
                    async with asyncio.timeout(min(18, remaining)):
                        # Never wrap this request with date-bearing local-change/repair prompts.
                        choice = await self.gateway.generate_structured(
                            dining_slot_request(
                                slot,
                                searched.candidates,
                                current,
                                context.book,
                                feedback=tuple(failures),
                            ),
                            ModelDiningSlotChoice,
                            cancellation=context.cancellation,
                        )
                    if choice.value.candidate_key is None:
                        failures.append("no_suitable_candidate_in_previous_batch")
                        current = advance(
                            current,
                            guard_observations=tuple(
                                item.model_copy(update={"code": "planner_dining_choice_null"})
                                if item.observation_id == marker
                                else item
                                for item in current.guard_observations
                            ),
                        )
                        await context.checkpoint(current)
                        continue
                    selected_index = int(choice.value.candidate_key[1:]) - 1
                    if not 0 <= selected_index < len(searched.candidates):
                        raise PlannerGuardError("planner_dining_choice_outside_batch")
                    place = searched.candidates[selected_index]
                    tentative = admit_dining_place(current, context.book, place, self.clock())
                    tentative = await self._remeasure_time_quality(tentative, context)

                    selected_slot = slot

                    async def verified_checkpoint(
                        value: PlannerWorkspaceState,
                        current_slot: DiningSlot = selected_slot,
                        identity: str = place.canonical_entity_id,
                    ) -> None:
                        if dining_slot_is_resolved(value, current_slot, identity):
                            await context.checkpoint(value)

                    trial_feedback: list[dict[str, Any]] = []
                    candidate = await self._try_time_quality(
                        tentative,
                        replace(context, checkpoint=verified_checkpoint),
                        attempted_intents,
                        trial_feedback,
                        repair_dates=frozenset((slot.service_date,)),
                        resolved_dining_choice=(
                            slot,
                            place.canonical_entity_id,
                            choice.audit_call_id,
                        ),
                    )
                    if dining_slot_is_resolved(candidate, slot, place.canonical_entity_id):
                        current = candidate
                        feedback.extend(trial_feedback)
                        await context.checkpoint(current)
                        break
                    reason = (
                        str(trial_feedback[-1].get("failure_code", "meal_not_resolved"))
                        if trial_feedback
                        else "meal_not_resolved"
                    )
                    failures.append(reason.split(":", 1)[0])
                    current = advance(
                        current,
                        guard_observations=(
                            *current.guard_observations,
                            PlannerGuardObservation(
                                observation_id=server_id(
                                    marker, "rejected", place.canonical_entity_id
                                ),
                                attempted_action="dining_slot_choice",
                                code="planner_dining_candidate_rejected",
                                message=json.dumps(
                                    {
                                        "slot_key": slot.slot_key,
                                        "candidate_id": place.canonical_entity_id,
                                        "failure_code": failures[-1],
                                    }
                                ),
                                based_on_workspace_revision=current.workspace_revision,
                            ),
                        ),
                    )
                    feedback.append({"failure_code": failures[-1]})
                    await context.checkpoint(current)
                except ModelGatewayError as error:
                    if error.code in {
                        ModelFailureCode.CANCELLED,
                        ModelFailureCode.AUDIT_UNAVAILABLE,
                    }:
                        raise
                    failures.append(error.code.value)
                    feedback.append({"failure_code": f"planner_dining_{error.code.value}"})
                except TimeoutError:
                    failures.append("choice_timeout")
                    feedback.append({"failure_code": "planner_dining_choice_timeout"})
                except (PlannerGuardError, ValidationError, ValueError, KeyError) as error:
                    code = (
                        error.code if isinstance(error, PlannerGuardError) else type(error).__name__
                    )
                    failures.append(code.split(":", 1)[0])
                    feedback.append({"failure_code": failures[-1]})
        return current

    async def _try_time_quality(
        self,
        workspace: PlannerWorkspaceState,
        context: PlannerGraphContext,
        attempted_intents: set[str],
        repair_feedback: list[dict[str, object]],
        *,
        compact_infill: bool = False,
        dining_only: bool = False,
        missing_meals_only: bool = False,
        long_visit_meal: bool = False,
        selected_gap_options: list[dict[str, Any]] | None = None,
        repair_dates: frozenset[date] | None = None,
        resolved_gap_choices: tuple[ModelGapVisits, list[dict[str, Any]], str | None] | None = None,
        resolved_dining_choice: tuple[DiningSlot, str, str | None] | None = None,
    ) -> PlannerWorkspaceState:
        """Adopt only an evidence-validated improvement; retain precise rejection feedback."""
        previous = advance(workspace, timing_optimization_pending=False)
        await context.checkpoint(previous)
        gaps = schedule_quality_gaps(previous, context.book)
        schedule = previous.materialized_schedule
        validation = previous.validation_observation
        if (
            not has_time_quality_issues(previous, context.book)
            and not (
                previous.schedule_repair_state
                and any(
                    issue.status == "pending" for issue in previous.schedule_repair_state.issues
                )
            )
        ) or (previous.working_itinerary is None or schedule is None or validation is None):
            return previous
        catalog = PlannerReferenceCatalog(previous)
        candidate_keys = {
            entry.candidate_ref.canonical_entity_id: key
            for key, entry in catalog.candidates.items()
        }
        hotel_ref = previous.working_itinerary.lodging_baseline.selected_offer_ref
        hotel_key = next(
            (
                key
                for key, offer in catalog.hotels.items()
                if hotel_ref is not None and offer.offer_ref == hotel_ref
            ),
            None,
        )
        plan: dict[str, Any] = {
            "days": [
                {
                    "day_index": index,
                    "theme": day.day_theme,
                    "stops": [
                        {
                            "candidate_key": candidate_keys[item.object_ref.canonical_entity_id],
                            "part_of_day": item.expected_window.part_of_day,
                            "meal_slot": item.meal_slot
                            if item.meal_slot in {"lunch", "dinner"}
                            else None,
                            "duration_preference": item.duration_preference,
                            "onsite_lunch": item.onsite_lunch,
                        }
                        for item in day.ordered_items
                        if hasattr(item.object_ref, "canonical_entity_id")
                        and item.object_ref.canonical_entity_id in candidate_keys
                        and not (
                            item.item_kind == "dining" and item.meal_slot not in {"lunch", "dinner"}
                        )
                    ],
                }
                for index, day in enumerate(previous.working_itinerary.days, 1)
            ],
            "selected_hotel_key": hotel_key,
        }
        await context.progress(
            "planner_time_quality", "正在平衡正餐、游览和通勤，调整每日顺序与沿途餐厅。"
        )
        call_id = None
        failure_code = "planner_timing_quality_not_improved"
        comparison: dict[str, object] = {}
        omitted_keys: tuple[str, ...] = ()
        affected_dates = {
            d
            for issue in validation.issues
            if issue.severity != "warning"
            and issue.code in {"opening_conflict", "meal_constraint_violation", "time_overlap"}
            for d in issue.affected_dates
        }
        omission_allowed = {
            candidate_keys[item.object_ref.canonical_entity_id]
            for day in previous.working_itinerary.days
            if day.service_date in affected_dates
            for item in day.ordered_items
            if hasattr(item.object_ref, "canonical_entity_id")
            and item.object_ref.canonical_entity_id in candidate_keys
            and item.commitment_level != "immutable"
        }
        omission_allowed.update(
            candidate_keys[ref.canonical_entity_id]
            for issue in validation.issues
            if issue.code == "route_cost_exceeded"
            and "internal_schedule_quality:dining_detour" in issue.violated_constraint_refs
            for ref in issue.candidate_refs
            if ref.canonical_entity_id in candidate_keys
        )
        # New policy never recreates a historical third meal. The old published
        # version remains immutable; fixed bookings are handled separately.
        legacy_extra_keys = {
            candidate_keys[item.object_ref.canonical_entity_id]
            for day in previous.working_itinerary.days
            if repair_dates is None or day.service_date in repair_dates
            for item in day.ordered_items
            if isinstance(item.object_ref, CandidateRef)
            and item.item_kind == "dining"
            and item.meal_slot not in {"lunch", "dinner"}
            and item.commitment_level != "immutable"
        }
        omission_allowed.update(legacy_extra_keys)
        failure_stage = "model_choice"
        request_marker: str | None = None
        base_digest = previous.working_itinerary.content_digest

        async def generate_choice(request: ModelRequest, output_type: Any) -> Any:
            nonlocal request_marker
            if context.call_remaining_seconds() <= 0:
                raise TimeoutError("planner_call_cutoff")
            if repair_dates is not None:
                request = _with_local_change_scope(
                    request, replace(context, local_change_dates=repair_dates)
                )
            if repair_feedback:
                request = request.model_copy(
                    update={
                        "messages": [
                            *request.messages,
                            ModelMessage(
                                role=ModelRole.SYSTEM,
                                content=(
                                    "上一方案的程序校验反馈（数据）："
                                    + json.dumps(repair_feedback[-2:], ensure_ascii=False)
                                    + "。请只修复列出的日期与字段，改选可行地点或顺序；"
                                    "不要重复相同失败方案。"
                                ),
                            ),
                        ]
                    }
                )
            request_marker = server_id(
                previous.generation_id,
                "timing-choice",
                base_digest,
                canonical_json_hash(
                    [message.model_dump(mode="json") for message in request.messages]
                ),
            )
            if any(item.observation_id == request_marker for item in previous.guard_observations):
                comparison["stop_retry"] = True
                raise PlannerGuardError("planner_timing_quality_repeated_failed_request")
            async with asyncio.timeout(context.call_remaining_seconds()):
                return await self.gateway.generate_structured(
                    _with_local_change_scope(request, context),
                    output_type,
                    cancellation=context.cancellation,
                )

        try:
            if resolved_dining_choice is not None:
                slot, chosen_identity, call_id = resolved_dining_choice
                chosen_key = candidate_keys[chosen_identity]
                if slot.old_identity is not None:
                    old_key = candidate_keys[slot.old_identity]
                    old_entry = catalog.candidates[old_key]
                    if old_entry.commitment_level.value in {"strong", "soft", "immutable"}:
                        if old_key not in omission_allowed:
                            raise PlannerGuardError("planner_dining_replacement_not_authorized")
                        omitted_keys = (old_key,)
                    proposed_intent = apply_dining_replacement(
                        ModelDiningReplacement(candidate_key=chosen_key),
                        {
                            "day_index": slot.day_index,
                            "old_candidate_key": old_key,
                            "options": [{"candidate_key": chosen_key}],
                        },
                        plan,
                    )
                else:
                    proposed_intent = apply_missing_meals(
                        ModelMissingMeals.model_validate(
                            {"choices": [{"meal_key": "m1", "candidate_key": chosen_key}]}
                        ),
                        [
                            {
                                "meal_key": "m1",
                                "day_index": slot.day_index,
                                "meal": slot.meal,
                                "insert_before_candidate_key": candidate_keys.get(
                                    slot.insert_before_identity or ""
                                ),
                                "options": [{"candidate_key": chosen_key}],
                            }
                        ],
                        plan,
                    )
            elif long_visit_meal:
                long_options = long_visit_meal_options(previous, allowed_dates=repair_dates)
                if long_options is None:
                    raise PlannerGuardError("planner_long_visit_meal_no_eligible_conflict")
                async with asyncio.timeout(12):
                    choice = await generate_choice(
                        long_visit_meal_request(long_options), ModelDiningReplacement
                    )
                call_id = choice.audit_call_id
                if choice.value.candidate_key not in long_options["meal_keys"]:
                    raise PlannerGuardError(
                        f"planner_long_visit_dinner_outside_options:date={long_options['date']}:"
                        f"path=candidate_key:allowed={','.join(long_options['meal_keys'])}"
                    )
                proposed_intent, omitted_keys = apply_long_visit_meal_choice(
                    choice.value, long_options, plan
                )
                if not set(omitted_keys) <= omission_allowed:
                    raise PlannerGuardError("planner_long_visit_meal_omission_not_authorized")
            elif dining_only:
                dining_options = dining_replacement_options(
                    previous,
                    plan,
                    allowed_dates={str(d) for d in repair_dates} if repair_dates else None,
                )
                if dining_options is None:
                    raise PlannerGuardError("planner_dining_no_nearby_replacement")
                old_key = dining_options["old_candidate_key"]
                old_entry = catalog.candidates[old_key]
                if old_entry.commitment_level.value in {"strong", "soft", "immutable"}:
                    if old_key not in omission_allowed:
                        raise PlannerGuardError("planner_dining_replacement_not_authorized")
                    omitted_keys = (old_key,)
                async with asyncio.timeout(12):
                    replacement = await generate_choice(
                        dining_replacement_request(dining_options, context.book),
                        ModelDiningReplacement,
                    )
                call_id = replacement.audit_call_id
                proposed_intent = apply_dining_replacement(replacement.value, dining_options, plan)
            elif missing_meals_only:
                meal_options = missing_meal_options(
                    previous,
                    plan,
                    allowed_dates={str(d) for d in repair_dates} if repair_dates else None,
                )
                if not meal_options:
                    raise PlannerGuardError("planner_missing_meal_no_eligible_candidates")
                async with asyncio.timeout(18):
                    meals = await generate_choice(
                        missing_meal_request(meal_options, context.book), ModelMissingMeals
                    )
                call_id = meals.audit_call_id
                proposed_intent = apply_missing_meals(meals.value, meal_options, plan)
            elif compact_infill:
                rejected_choices = [
                    choice
                    for feedback in repair_feedback
                    for choice in cast(
                        list[dict[str, Any]], feedback.get("rejected_gap_choices", [])
                    )
                ]
                rejected_pairs = {
                    (choice["date"], choice["candidate_key"]) for choice in rejected_choices
                }
                options = [
                    {
                        **gap,
                        "options": [
                            option
                            for option in gap["options"]
                            if (gap["date"], option["candidate_key"]) not in rejected_pairs
                        ],
                    }
                    for gap in (
                        selected_gap_options
                        if selected_gap_options is not None
                        else gap_visit_options(previous, context.book)
                    )
                    if repair_dates is None or date.fromisoformat(gap["date"]) in repair_dates
                ]
                options = [gap for gap in options if gap["options"]]
                if not options:
                    raise PlannerGuardError("planner_gap_no_eligible_candidates")
                if resolved_gap_choices is not None:
                    chosen_gaps, options, call_id = resolved_gap_choices
                else:
                    async with asyncio.timeout(25):
                        gap_choice = await generate_choice(
                            build_gap_visit_request(
                                options,
                                context.book,
                                feedback=rejected_choices,
                                repair_feedback=[
                                    {
                                        key: value
                                        for key, value in item.items()
                                        if key
                                        in {
                                            "status",
                                            "failure_code",
                                            "repair_hint",
                                            "requires_day_repair",
                                            "rejected_gap_choices",
                                        }
                                    }
                                    for item in repair_feedback[-2:]
                                ],
                            ),
                            ModelGapVisits,
                        )
                    call_id = gap_choice.audit_call_id
                    chosen_gaps = gap_choice.value
                choice_failures: list[dict[str, str]] = []
                proposed_intent = apply_gap_visit_choices(
                    chosen_gaps, options, plan, failures=choice_failures
                )
                eligible = {
                    gap["gap_key"]: {o["candidate_key"] for o in gap["options"]} for gap in options
                }
                chosen_gaps = chosen_gaps.model_copy(
                    update={
                        "choices": tuple(
                            c
                            for c in chosen_gaps.choices
                            if c.candidate_key in eligible.get(c.gap_key, set())
                        )
                    }
                )
                if choice_failures:
                    repair_feedback.append({"rejected_gap_choices": choice_failures})
                replacements = {
                    option["candidate_key"]: option["replace_candidate_key"]
                    for gap in options
                    for option in gap["options"]
                    if option.get("operation") == "replace_existing_visit"
                }
                omitted_keys = tuple(
                    replacements[c.candidate_key]
                    for c in chosen_gaps.choices
                    if c.candidate_key in replacements
                )
                omission_allowed.update(omitted_keys)
            else:
                async with asyncio.timeout(40):
                    generated = await generate_choice(
                        scoped_repair_request(
                            build_compact_plan_request(
                                previous,
                                context.book,
                                timing_quality_review={
                                    "omission_allowed_keys": sorted(omission_allowed),
                                    "repair_priority": (
                                        "先修复饭点与营业时间冲突，再补空档；不能只补另一天的小景点而保留过晚午餐。"
                                        "若一天午餐前景点过多、另一天缺少下午游览，优先跨日移动一个完整景点，"
                                        "给两天保留充足游览；必吃不是预约，真实路线明显绕远时可取舍并换附近餐厅。"
                                    ),
                                    "gaps": gaps,
                                    "half_day_coverage_issues": schedule_coverage_issues(
                                        previous, context.book
                                    ),
                                    "previous_repair_feedback": repair_feedback[-1:],
                                    "meal_issues": schedule_meal_issues(previous),
                                    "missing_concrete_meals": missing_concrete_meals(previous),
                                    "evening_opportunities": evening_activity_opportunities(
                                        previous, context.book
                                    ),
                                    "afternoon_opportunities": afternoon_activity_opportunities(
                                        previous, context.book
                                    ),
                                    "preferred_meal_issues": schedule_meal_issues(
                                        previous, preferred=True
                                    ),
                                    "dining_commute_issues": dining_commute_issues(previous),
                                    "opening_conflicts": [
                                        {
                                            "candidate_keys": [
                                                candidate_keys[ref.canonical_entity_id]
                                                for ref in issue.candidate_refs
                                                if ref.canonical_entity_id in candidate_keys
                                            ],
                                            "dates": [str(day) for day in issue.affected_dates],
                                            "message": issue.message_summary,
                                            "allowed_actions": issue.allowed_actions,
                                        }
                                        for issue in validation.issues
                                        if issue.code == "opening_conflict"
                                        and issue.severity != "warning"
                                    ],
                                    "previous_plan": plan,
                                    "day_times": [
                                        {
                                            "date": day.service_date.isoformat(),
                                            "start": day.start_time.isoformat(),
                                            "end": day.end_time.isoformat(),
                                            "active_minutes": day.active_minutes,
                                            "activities": [
                                                {
                                                    "name": item.title,
                                                    "start": item.start_time.isoformat(),
                                                    "minutes": item.duration_minutes,
                                                }
                                                for item in day.activities
                                            ],
                                        }
                                        for day in schedule.days
                                    ],
                                },
                            )
                        ),
                        ModelScheduleRepair,
                    )
                call_id = generated.audit_call_id
                if repair_dates is not None and any(
                    d.day_index > len(previous.working_itinerary.days)
                    or previous.working_itinerary.days[d.day_index - 1].service_date
                    not in repair_dates
                    for d in generated.value.days
                ):
                    raise PlannerGuardError(
                        "planner_repair_outside_local_change_dates:path=days[].day_index"
                    )
                repair = normalize_day_repair(
                    generated.value,
                    plan,
                    protected_meal_keys={
                        key
                        for key, entry in catalog.candidates.items()
                        if entry.entity_kind.value == "restaurant"
                        and entry.commitment_level.value in {"strong", "immutable"}
                    },
                )
                free_omission_keys = {
                    key
                    for key, entry in catalog.candidates.items()
                    if entry.commitment_level.value in {"neutral", "filler"}
                }
                if not set(repair.omit_candidate_keys) <= omission_allowed | free_omission_keys:
                    raise PlannerGuardError("planner_day_repair_omission_outside_conflict")
                omitted_keys = repair.omit_candidate_keys
                proposed_intent = merge_day_repair(repair, plan)
            failure_stage = "compile"
            omitted_keys = tuple(dict.fromkeys((*omitted_keys, *legacy_extra_keys)))
            fingerprint = canonical_json_hash(
                {
                    "days": [
                        {
                            "day_index": day.day_index,
                            "stops": [stop.model_dump(mode="json") for stop in day.stops],
                        }
                        for day in proposed_intent.days
                    ],
                    "hotel": proposed_intent.selected_hotel_key,
                }
            )
            if fingerprint in attempted_intents:
                comparison["stop_retry"] = True
                raise PlannerGuardError("planner_timing_quality_repeated_intent")
            attempted_intents.add(fingerprint)
            comparison["proposal_fingerprint"] = fingerprint
            previous_fingerprint = canonical_json_hash(
                {
                    "days": [
                        {
                            "day_index": day["day_index"],
                            "stops": [
                                ModelPlanStop.model_validate(stop).model_dump(mode="json")
                                for stop in day["stops"]
                            ],
                        }
                        for day in plan["days"]
                    ],
                    "hotel": hotel_key,
                }
            )
            if fingerprint == previous_fingerprint:
                comparison["stop_retry"] = True
                raise PlannerGuardError("planner_timing_quality_unchanged_intent")
            if proposed_intent.selected_hotel_key != hotel_key:
                raise PlannerGuardError("planner_timing_quality_hotel_changed")
            staged = advance(
                previous,
                revision_round=0,
                working_itinerary=None,
                selected_hotel=None,
                hotel_recommendations=None,
                materialized_schedule=None,
                cost_draft=None,
                validation_report=None,
                validation_observation=None,
                status=PlannerStatus.PLANNING,
            )
            if omitted_keys:
                receipts = tuple(
                    UnassignedIntent(
                        candidate_ref=catalog.candidates[key].candidate_ref,
                        commitment_level="strong"
                        if catalog.candidates[key].commitment_level.value == "strong"
                        else "soft",
                        reason_code="awaiting_user",
                        supporting_observation_refs=(validation.observation_id,),
                        requires_user_resolution=catalog.candidates[key].commitment_level.value
                        == "strong",
                    )
                    for key in omitted_keys
                    if catalog.candidates[key].commitment_level.value in {"strong", "soft"}
                )
                staged = advance(
                    staged,
                    recovery_omissions=(*staged.recovery_omissions, *receipts),
                    recovery_observations=(*staged.recovery_observations, validation),
                    best_effort_reasons=tuple(
                        dict.fromkeys(
                            (
                                *staged.best_effort_reasons,
                                *(
                                    f"未安排：{catalog.candidates[key].display_name}；为解决本轮时间、营业或明显绕路问题进行了取舍，原要求保留。"
                                    for key in omitted_keys
                                ),
                            )
                        )
                    ),
                )
            decision = compile_plan_intent_decision(
                proposed_intent, staged, context.book, base_draft=previous.working_itinerary
            )
            payload = decision.payload
            assert isinstance(payload, MaterializeDraftPayload)

            # Model intent deliberately has no appointment fields. Carry them
            # from the old draft, then verify identity/date preservation below.
            appointments = {
                (day.service_date, item.object_ref.model_dump_json()): item
                for day in previous.working_itinerary.days
                for item in day.ordered_items
                if item.commitment_level == "immutable"
                or item.expected_window.earliest is not None
                or item.expected_window.latest is not None
            }
            preserved_appointments = payload.proposed_working_draft.model_copy(
                update={
                    "days": tuple(
                        day.model_copy(
                            update={
                                "ordered_items": tuple(
                                    item.model_copy(
                                        update={
                                            "expected_window": original.expected_window,
                                            "duration_preference": original.duration_preference,
                                        }
                                    )
                                    if (
                                        original := appointments.get(
                                            (day.service_date, item.object_ref.model_dump_json())
                                        )
                                    )
                                    else item
                                    for item in day.ordered_items
                                )
                            }
                        )
                        for day in payload.proposed_working_draft.days
                    )
                }
            )
            preserved_appointments = preserved_appointments.model_copy(
                update={"content_digest": planning_projection_digest(preserved_appointments)}
            )
            if resolved_dining_choice is not None:
                preserved_appointments = restore_dining_slot_order(
                    preserved_appointments,
                    previous.working_itinerary,
                    resolved_dining_choice[0],
                    resolved_dining_choice[1],
                )
            payload = payload.model_copy(update={"proposed_working_draft": preserved_appointments})
            decision = decision.model_copy(update={"payload": payload})

            if repair_dates is not None:
                # The compiler may rederive mechanical metadata for all dates;
                # an authorized small repair retains every untouched day verbatim.
                originals = {day.service_date: day for day in previous.working_itinerary.days}
                preserved = payload.proposed_working_draft.model_copy(
                    update={
                        "days": tuple(
                            day if day.service_date in repair_dates else originals[day.service_date]
                            for day in payload.proposed_working_draft.days
                        ),
                    }
                )
                kept_ids = {
                    item.object_ref.canonical_entity_id
                    for day in preserved.days
                    for item in day.ordered_items
                    if isinstance(item.object_ref, CandidateRef)
                }
                preserved = preserved.model_copy(
                    update={
                        "unassigned_intents": tuple(
                            item
                            for item in preserved.unassigned_intents
                            if item.candidate_ref.canonical_entity_id not in kept_ids
                        )
                    }
                )
                preserved = preserved.model_copy(
                    update={"content_digest": planning_projection_digest(preserved)}
                )
                payload = payload.model_copy(update={"proposed_working_draft": preserved})
                decision = decision.model_copy(update={"payload": payload})

            if context.local_change_dates is not None:
                _guard_local_day_changes(
                    previous.working_itinerary,
                    payload.proposed_working_draft,
                    context.local_change_dates,
                )

            def fixed_windows(
                draft: WorkingItineraryDraft,
            ) -> dict[tuple[date, str], tuple[str, str | None, str]]:
                return {
                    (day.service_date, item.object_ref.model_dump_json()): (
                        item.expected_window.model_dump_json(),
                        item.duration_preference,
                        item.item_kind,
                    )
                    for day in draft.days
                    for item in day.ordered_items
                    if isinstance(item.object_ref, FixedCommitmentRef)
                    or item.commitment_level == "immutable"
                    or item.expected_window.earliest is not None
                    or item.expected_window.latest is not None
                }

            if fixed_windows(payload.proposed_working_draft) != fixed_windows(
                previous.working_itinerary
            ):
                raise PlannerGuardError("planner_timing_quality_fixed_window_changed")
            failure_stage = "draft_guard"
            validate_working_draft(
                payload.proposed_working_draft, staged, context.book, self.clock()
            )
            candidate = advance(
                staged,
                working_itinerary=payload.proposed_working_draft,
                selected_hotel=payload.selected_hotel,
                decision_trace=(*staged.decision_trace, decision),
            )
            permitted_omissions = {
                catalog.candidates[key].candidate_ref.canonical_entity_id for key in omitted_keys
            }
            if not (
                protected_time_quality_candidates(previous) - permitted_omissions
            ) <= selected_candidate_ids(candidate):
                raise PlannerGuardError("planner_timing_quality_removed_selected_place")
            if selected_itinerary_route_requests(candidate, context.book):
                failure_stage = "route_evidence"
                if context.call_remaining_seconds() <= 0:
                    raise TimeoutError("planner_call_cutoff")
                candidate = await self.evidence.ensure_selected_itinerary_routes(
                    candidate, context.book, context.cancellation
                )
            if isinstance(self.evidence, PlannerEvidenceBackend):
                if context.call_remaining_seconds() <= 0:
                    raise TimeoutError("planner_call_cutoff")
                candidate = await self.evidence.ensure_selected_hours(
                    candidate, context.book, context.cancellation
                )
                # Optional price enrichment must not consume scheduling recovery.
            failure_stage = "materialize"
            materialized = await self.materializer.materialize(
                candidate,
                context.book,
                context.cancellation,
                input_state_version=context.input_state_version,
            )
            candidate = advance(
                candidate, materialized_schedule=materialized.schedule, cost_draft=materialized.cost
            )
            failure_stage = "validate"
            validated = await self.validator.validate(candidate, context.book, context.cancellation)
            candidate = advance(
                candidate,
                validation_report=validated.legacy_report,
                validation_observation=validated.observation,
            )
            infill_verified = True
            if resolved_dining_choice is not None:
                infill_verified = dining_slot_is_resolved(
                    candidate, resolved_dining_choice[0], resolved_dining_choice[1]
                )
            if compact_infill:
                blocking = [i for i in validated.observation.issues if i.severity != "warning"]
                failed_dates = {str(day) for issue in blocking for day in issue.affected_dates}
                option_by_key = {gap["gap_key"]: gap for gap in options}
                hours_failures = gap_choice_hours_failures(chosen_gaps, options, candidate)
                failed_choices = [
                    {
                        "date": option_by_key[choice.gap_key]["date"],
                        "candidate_key": choice.candidate_key,
                        "path": f"choices[{index}].candidate_key",
                        "reason": "；".join(
                            issue.message_summary
                            for issue in blocking
                            if option_by_key[choice.gap_key]["date"]
                            in {str(day) for day in issue.affected_dates}
                        ),
                    }
                    for index, choice in enumerate(chosen_gaps.choices)
                    if option_by_key[choice.gap_key]["date"] in failed_dates
                ] + hours_failures
                failed_dates.update(failure["date"] for failure in hours_failures)
                infill_verified = not failed_choices
                kept_choices = tuple(
                    choice
                    for choice in chosen_gaps.choices
                    if option_by_key[choice.gap_key]["date"] not in failed_dates
                )
                if failed_choices:
                    repair_feedback.append({"rejected_gap_choices": failed_choices})
                if (
                    failed_choices
                    and kept_choices
                    and resolved_gap_choices is None
                    and context.remaining_seconds() >= 12
                ):
                    # Keep the model's independent choices on unaffected days.
                    # Recompile/requery/revalidate the subset without another
                    # model call; never publish the failed original batch.
                    await record_model_call_annotation(
                        self.gateway,
                        call_id,
                        "llm_business_guard",
                        {
                            "accepted_or_rejected": "rejected",
                            "business_guard_result": {
                                "guard": "planner_time_quality",
                                "status": "partial_revalidation_required",
                                "rejected_gap_choices": failed_choices,
                            },
                        },
                    )
                    return await self._try_time_quality(
                        previous,
                        context,
                        attempted_intents,
                        repair_feedback,
                        compact_infill=True,
                        resolved_gap_choices=(
                            ModelGapVisits(choices=kept_choices),
                            options,
                            call_id,
                        ),
                        repair_dates=repair_dates,
                    )
            old_gap = sum(int(item["minutes"]) for item in gaps)
            new_gap = sum(
                int(item["minutes"]) for item in schedule_quality_gaps(candidate, context.book)
            )
            old_score = schedule_quality_score(previous, context.book)
            new_score = schedule_quality_score(candidate, context.book)
            old_pace = sum(issue.code == "pace_limit_exceeded" for issue in validation.issues)
            new_pace = sum(
                issue.code == "pace_limit_exceeded" for issue in validated.observation.issues
            )
            comparison = {
                "proposal_fingerprint": fingerprint,
                "gap_minutes_before": old_gap,
                "gap_minutes_after": new_gap,
                "coverage_before": schedule_coverage_issues(previous, context.book),
                "coverage_after": schedule_coverage_issues(candidate, context.book),
                "travel_minutes_before": schedule_travel_minutes(previous),
                "travel_minutes_after": schedule_travel_minutes(candidate),
                "validation_result": validated.observation.result,
                "blocking_codes": [
                    issue.code
                    for issue in validated.observation.issues
                    if issue.severity != "warning"
                ],
                "blocking_details": [
                    {
                        "code": issue.code,
                        "dates": [str(day) for day in issue.affected_dates],
                        "draft_item_ids": list(issue.draft_item_ids),
                        "candidate_keys": [
                            candidate_keys[ref.canonical_entity_id]
                            for ref in issue.candidate_refs
                            if ref.canonical_entity_id in candidate_keys
                        ],
                        "reason": issue.message_summary,
                    }
                    for issue in validated.observation.issues
                    if issue.severity != "warning"
                ],
                "pace_warnings_before": old_pace,
                "pace_warnings_after": new_pace,
                "quality_score_before": old_score,
                "quality_score_after": new_score,
                "meal_issues_before": schedule_meal_issues(previous),
                "meal_issues_after": schedule_meal_issues(candidate),
                "preferred_meal_issues_before": schedule_meal_issues(previous, preferred=True),
                "preferred_meal_issues_after": schedule_meal_issues(candidate, preferred=True),
            }
            old_missing = {
                (item["date"], item["period"])
                for item in schedule_coverage_issues(previous, context.book)
            }
            new_missing = {
                (item["date"], item["period"])
                for item in schedule_coverage_issues(candidate, context.book)
            }
            if (
                may_keep_intermediate_timing_repair(previous, candidate)
                and infill_verified
                and new_score < old_score
                and new_missing <= old_missing
            ):
                await record_model_call_annotation(
                    self.gateway,
                    call_id,
                    "llm_business_guard",
                    {
                        "accepted_or_rejected": "accepted",
                        "business_guard_result": {
                            "guard": "planner_time_quality",
                            "status": "accepted",
                            "gap_minutes_before": old_gap,
                            "gap_minutes_after": new_gap,
                            **comparison,
                        },
                    },
                )
                await context.checkpoint(candidate)
                repair_feedback.append({"status": "improved", **comparison})
                return candidate
        except ModelGatewayError as error:
            if error.code in {ModelFailureCode.CANCELLED, ModelFailureCode.AUDIT_UNAVAILABLE}:
                raise
            call_id = error.audit_call_id
            failure_code = f"planner_timing_quality_{error.code.value}"
            comparison["stop_retry"] = True
        except (TimeoutError, PlannerGuardError, ValidationError, ValueError, KeyError) as error:
            failure_code = (
                error.code
                if isinstance(error, PlannerGuardError)
                else f"planner_timing_quality_{failure_stage}_{type(error).__name__}"
            )
            comparison["failure_stage"] = failure_stage
            if isinstance(error, ValidationError):
                comparison["invalid_fields"] = [
                    {"path": ".".join(map(str, issue["loc"])), "type": issue["type"]}
                    for issue in error.errors(include_input=False, include_url=False)[:3]
                ]
            elif isinstance(error, ValueError):
                # Contract errors are controlled messages, not model payloads.
                comparison["contract_error"] = str(error)[:300]
            if isinstance(error, TimeoutError):
                comparison["stop_retry"] = True
            if failure_code.startswith("planner_plan_duplicate_candidate:"):
                comparison["repair_hint"] = (
                    "错误路径中的candidate_key已在其他日期使用。若要跨日搬移，请从原日期删除，"
                    "而不是重复加入；同时重新分配原日期的上午和下午游览，不能把空下午搬到另一天。"
                )
            elif failure_code.startswith("planner_plan_missing_required_restaurant:"):
                comparison["repair_hint"] = (
                    "保留这次调整后的景点分配，把错误中指出的必吃餐厅安排到正常午餐或晚餐，"
                    "替换普通餐厅，并按营业时间适当调整相邻景点时长；不能添加额外下午茶。"
                )
            if failure_stage == "compile" and failure_code.startswith(
                (
                    "planner_plan_same_venue_duplicate:",
                    "planner_plan_missing_required_restaurant:",
                    "planner_plan_day_capacity_exceeded:",
                )
            ):
                comparison["requires_day_repair"] = True
                comparison["repair_hint"] = (
                    "冲突位于合并后的原日程，不能靠继续增加景点修复。"
                    "下一步仅调整错误涉及的日期/项目，保留其他日期、酒店和固定预约。"
                )
        await record_model_call_annotation(
            self.gateway,
            call_id,
            "llm_business_guard",
            {
                "accepted_or_rejected": "rejected",
                "failure_code": failure_code,
                "business_guard_result": {
                    "guard": "planner_time_quality",
                    "status": "rejected",
                    **comparison,
                },
            },
        )
        repair_feedback.append({"failure_code": failure_code, **comparison})
        if (
            compact_infill
            and resolved_gap_choices is None
            and failure_stage == "compile"
            and len({g["date"] for g in options}) > 1
            and context.call_remaining_seconds() > 8
        ):
            # A compiler failure in one date is not an atomic veto of other
            # independent choices. Reuse the model result, isolate dates, and
            # revalidate each against the latest accepted workspace.
            accepted_partial = previous
            for gap in options:
                subset = tuple(c for c in chosen_gaps.choices if c.gap_key == gap["gap_key"])
                if not subset or context.call_remaining_seconds() <= 8:
                    continue
                accepted_partial = await self._try_time_quality(
                    accepted_partial,
                    context,
                    attempted_intents,
                    repair_feedback,
                    compact_infill=True,
                    resolved_gap_choices=(ModelGapVisits(choices=subset), [gap], call_id),
                    repair_dates=frozenset((date.fromisoformat(gap["date"]),)),
                )
            return accepted_partial
        if request_marker and not any(
            item.observation_id == request_marker for item in previous.guard_observations
        ):
            previous = advance(
                previous,
                guard_observations=(
                    *previous.guard_observations,
                    PlannerGuardObservation(
                        observation_id=request_marker,
                        attempted_action="timing_quality",
                        code=failure_code.split(":", 1)[0],
                        message=failure_code[:500],
                        based_on_workspace_revision=previous.workspace_revision,
                    ),
                ),
            )
            await context.checkpoint(previous)
        context.cancellation.raise_if_cancelled("planner_time_quality")
        return previous

    async def _propose_repair(
        self,
        workspace: PlannerWorkspaceState,
        context: PlannerGraphContext,
    ) -> tuple[PlannerWorkspaceState, PlannerDecision | None, str | None]:
        """Ask only for a semantic repair choice; compile all formal fields server-side."""

        observation = workspace.validation_observation
        assert observation is not None
        if workspace.segment_attempt_count >= 12:
            failed = self._rejected(
                workspace,
                "revise_draft",
                "planner_decision_budget_exhausted",
                stage="validate",
            )
            failed = advance(failed, status=PlannerStatus.FAILED)
            await context.checkpoint(failed)
            return failed, None, None

        workspace = advance(
            workspace,
            action_attempt_count=workspace.action_attempt_count + 1,
            segment_attempt_count=workspace.segment_attempt_count + 1,
        )
        await context.checkpoint(workspace)
        await context.progress(
            "planner_repairing",
            "千问正在针对当前校验问题选择最小改动。",
        )
        call_id: str | None = None
        try:
            request = _with_local_change_scope(
                build_planner_repair_request(workspace, context.book), context
            )
            if observation.result == "insufficient_evidence":
                evidence_generated = await self._generate_structured_with_retry(
                    request,
                    ModelEvidenceRepairIntent,
                    context,
                )
                call_id = evidence_generated.audit_call_id
                intent = ModelRepairIntent.model_validate(
                    evidence_generated.value.model_dump(mode="json")
                )
            else:
                patch_generated = await self._generate_structured_with_retry(
                    request,
                    ModelPatchRepairIntent,
                    context,
                )
                call_id = patch_generated.audit_call_id
                intent = ModelRepairIntent.model_validate(
                    patch_generated.value.model_dump(mode="json")
                )
            decision = compile_model_repair_intent(intent, workspace)
            if (
                context.local_change_dates is not None
                and isinstance(decision.payload, ReviseDraftPayload)
                and not set(decision.payload.declared_affected_dates) <= context.local_change_dates
            ):
                raise PlannerGuardError("planner_repair_outside_local_change_dates")
            return workspace, decision, call_id
        except ModelGatewayError as error:
            if error.code is not ModelFailureCode.MALFORMED_RESPONSE:
                raise
            code = _model_schema_error_code(error)
            stage: GuardStage = "schema"
        except (PlannerGuardError, ValidationError, ValueError, KeyError) as error:
            code = error.code if isinstance(error, PlannerGuardError) else _safe_schema_error(error)
            stage = "reference"
        await record_model_call_annotation(
            self.gateway,
            call_id,
            "llm_business_guard",
            {
                "business_guard_result": {
                    "status": "rejected",
                    "guard": "planner_repair_resolution",
                    "reason": code,
                },
                "accepted_or_rejected": "rejected",
                "failure_stage": stage,
                "failure_code": code,
            },
        )
        workspace = self._rejected(
            workspace,
            "revise_draft",
            code,
            stage=stage,
        )
        if repeated_guard_failure(workspace):
            workspace = advance(workspace, status=PlannerStatus.FAILED)
        await context.checkpoint(workspace)
        return workspace, None, call_id

    async def _execute_repair(
        self,
        workspace: PlannerWorkspaceState,
        decision: PlannerDecision,
        proposal_call_id: str | None,
        context: PlannerGraphContext,
    ) -> tuple[PlannerWorkspaceState, bool]:
        payload = decision.payload
        rejected = False
        code: str | None = None
        try:
            if isinstance(payload, RequestEvidencePayload):
                self.evidence.validate_batch(
                    payload.capability_requests,
                    workspace,
                    context.book,
                )
                workspace = advance(
                    workspace,
                    decision_trace=(*workspace.decision_trace, decision),
                )
                await context.checkpoint(workspace)
                await context.progress(
                    "planner_querying",
                    "正在补齐当前校验问题所需的真实证据。",
                )
                previous = workspace
                refreshed = await self.evidence.execute_batch(
                    payload.capability_requests,
                    workspace,
                    context.book,
                    context.cancellation,
                )
                workspace = rebind_semantic_artifacts_after_evidence(previous, refreshed)
                workspace = advance(
                    workspace,
                    **invalidate_materialized_dependencies(
                        workspace,
                        affected_dates=tuple(
                            sorted(
                                {
                                    service_date
                                    for issue in previous.validation_observation.issues
                                    for service_date in issue.affected_dates
                                }
                            )
                        )
                        if previous.validation_observation is not None
                        else (),
                    ),
                    status=PlannerStatus.PLANNING,
                )
            elif isinstance(payload, ReviseDraftPayload):
                draft = workspace.working_itinerary
                if draft is None or not isinstance(
                    payload.itinerary_patch.authority,
                    ValidationIssuePatchAuthority,
                ):
                    raise PlannerGuardError("planner_repair_patch_authority_invalid")
                authority = payload.itinerary_patch.authority
                revised = atomic_apply_itinerary_patch(
                    draft,
                    payload.itinerary_patch,
                    candidate_pool=workspace.candidate_pool,
                    allowed_authority_refs=frozenset(authority.issue_ids),
                )
                workspace = advance(
                    workspace,
                    working_itinerary=revised,
                    selected_hotel=(
                        payload.selected_hotel
                        if payload.selected_hotel is not None
                        else None
                        if payload.hotel_recommendations is not None
                        else workspace.selected_hotel
                    ),
                    hotel_recommendations=(
                        payload.hotel_recommendations
                        if payload.hotel_recommendations is not None
                        else None
                        if payload.selected_hotel is not None
                        else workspace.hotel_recommendations
                    ),
                    decision_trace=(*workspace.decision_trace, decision),
                    revision_round=workspace.revision_round + 1,
                    **invalidate_materialized_dependencies(
                        workspace,
                        affected_dates=payload.declared_affected_dates,
                    ),
                    status=PlannerStatus.PLANNING,
                )
            else:
                raise PlannerGuardError("planner_repair_action_not_supported")
        except (PlannerGuardError, ValidationError, ValueError, KeyError) as error:
            rejected = True
            code = error.code if isinstance(error, PlannerGuardError) else _safe_schema_error(error)
            workspace = self._rejected(
                workspace,
                decision.action,
                code,
                stage="business",
            )
            if repeated_guard_failure(workspace):
                workspace = advance(workspace, status=PlannerStatus.FAILED)
        await record_model_call_annotation(
            self.gateway,
            proposal_call_id,
            "llm_business_guard",
            {
                "business_guard_result": {
                    "status": "rejected" if rejected else "accepted",
                    "guard": "planner_repair_guard",
                    "reason": code,
                },
                "accepted_or_rejected": "rejected" if rejected else "accepted",
                "failure_stage": "business_guard" if rejected else None,
                "failure_code": code,
                "materialized_output": (
                    {"decision": decision.model_dump(mode="json")}
                    if rejected
                    else {
                        "decision": decision.model_dump(mode="json"),
                        "workspace": workspace.model_dump(mode="json"),
                    }
                ),
            },
        )
        await context.checkpoint(workspace)
        return workspace, not rejected

    async def _fail_completion_stage(
        self,
        workspace: PlannerWorkspaceState,
        context: PlannerGraphContext,
        *,
        action: str,
        stage: GuardStage,
        error: Exception,
    ) -> PlannerWorkspaceState:
        code = error.code if isinstance(error, PlannerGuardError) else _safe_schema_error(error)
        failed = self._rejected(workspace, action, code, stage=stage)
        failed = advance(failed, status=PlannerStatus.FAILED)
        await context.checkpoint(failed)
        return failed

    @staticmethod
    def _rejected(
        workspace: PlannerWorkspaceState,
        action: str,
        code: str,
        *,
        stage: GuardStage = "business",
    ) -> PlannerWorkspaceState:
        remedy = ""
        retry_instruction = "请根据当前证据和契约重新提出合法行动。"
        if code.startswith("planner_issue_resolution_requires_current_b_keys"):
            remedy = (
                "当前请求没有真实 readiness b 键。策略 pending_evidence 不是校验问题。"
                "普通补查请选择 complete_initial_evidence，并保持 based_on_issue_keys 为空。"
            )
        elif code.startswith("planner_strategy_no_effective_change"):
            remedy = "这份策略已经接受且没有实质变化；请实际调用所需能力或提交完整逐日草稿。"
        elif code.startswith("planner_strategy_hard_constraints_not_covered"):
            remedy = (
                "将 required_constraint_keys.hard_guard_refs 中的每个键逐字完整复制到"
                " conflict_policy.hard_guard_refs；错误代码 missing 已列出本次遗漏键。"
                "不得只挑部分约束、填写用户文本或自造引用。"
            )
        elif code.startswith("planner_strategy_dietary_constraints_not_covered"):
            remedy = (
                "将 required_constraint_keys.dietary_constraint_refs 中的每个键逐字完整复制到"
                " dining_policy.dietary_constraint_refs，并同时保留在 hard_guard_refs；"
                "错误代码 missing 已列出本次遗漏键。"
            )
        elif code.startswith("planner_strategy_unknown_assumption_source"):
            evidence_keys = list(PlannerReferenceCatalog(workspace).evidence)
            remedy = (
                "错误代码 invalid_indices 已列出 non_blocking_assumptions 中的非法数组位置。"
                "每个 source_ref 只能逐字复制 allowed_assumption_source_keys；当前可直接复制的"
                f"证据短键为 {','.join(evidence_keys) or 'none'}，任务书短键也已列在该字段中。"
                "不能填写摘要、字段路径或自造 assumption 键；没有可靠来源时删除整条"
                " assumption，不能猜测来源。"
            )
        elif code.startswith("planner_hotel_requests_must_not_overwrite_same_observation"):
            remedy = (
                "一批请求中的 hotel_search 与 hotel_offer_refresh 合计最多一项。"
                "无需同时刷新所有酒店，未知价格库存可以留待下一阶段，不阻止选择暂定空间基点。"
            )
        elif code.startswith("planner_opening_hours_already_observed"):
            remedy = (
                "这些地点与日期在本执行段已经查询；unknown/partial 是有效 Observation，"
                "不能原样重试。请保留未知，改查尚未覆盖的必要证据或提交空间工作草稿。"
            )
        elif code.startswith("planner_hotel_search_already_observed"):
            remedy = (
                "相同日期与活动簇的酒店搜索已经执行；已有 h 键可作为暂定空间基点，"
                "库存、价格和取消条款未知留待后续，不得重复搜索来伪造确定性。"
            )
        elif code.startswith("planner_required_hotel_evidence_unavailable"):
            remedy = (
                "本次酒店能力没有形成任何可选择的 h 键，且同一请求不能靠模型原样重试修复。"
                "系统已保留真实 unavailable Observation，未选择或虚构酒店。"
            )
            retry_instruction = "本执行段已停止；后续可在 Provider 恢复或城市目录更新后重新运行。"
        elif code.startswith("planner_unknown_route_endpoint"):
            remedy = (
                "上一份路线请求使用了当前 catalog 中不存在的端点。请逐项读取"
                " required_repair_contract.invalid_route_endpoint_fields：只在对应"
                " allowed_endpoint_keys 中复制一个真实键替换该字段。不能使用 hotel、lodging、"
                "area、evidence、任务书引用或地点名称；没有 h 键时若确需酒店端点，应先请求"
                " hotel_search，不能虚构酒店端点。"
            )
        elif code.startswith("planner_draft_hotel_observation_missing"):
            remedy = (
                "当前任务书需要推荐住宿，但本轮 hotel_observation 为空，不能继续重交草稿。"
                "下一行动必须改为 request_evidence，并提交一项 hotel_search；日期、party_size_ref、"
                "住宿偏好和活动簇逐字使用 required_hotel_stay、lodging 引用及当前 g 键。"
                "获得真实结果后，从 available_hotel_offer_keys 选择暂定 h 键再生成草稿。"
            )
        elif code.startswith("planner_draft_hotel_selection_required"):
            remedy = (
                "住宿策略要求搜索并选择真实结果。请从 available_hotel_offer_keys 复制一个现有 h 键"
                "写入 draft.lodging_baseline.selected_offer_key；如果该列表为空，则不得提交草稿或"
                "虚构酒店，应先请求 hotel_search。"
            )
        elif code.startswith("planner_day_exceeds_declared_capacity"):
            remedy = (
                "错误代码已给出 day_index、实际 visit 数和策略 maximum。"
                "请对照 rejected_draft_reference_checks 对应 day_index 的 capacity："
                "返回的完整草稿必须让该日 ordered_items 中 visit 数不超过"
                " required_result.ordered_visit_count_at_most，并且真的删除至少"
                " minimum_ordered_visit_removals_from_this_day 个 visit。"
                "只有 destination_days_with_free_visit_capacity 列出的日期才有移动空间；"
                "该列表只证明容量，移动前仍需核对候选 feasible_dates、路线和硬条件。"
                "没有合法移动目标时，从 soft_unassignment_contract.candidate_key_choices 自主选择"
                "足量 soft，从原日 ordered_items 删除，并按该 contract 的固定字段只放入"
                " unassigned_intents。filler/neutral 可从 directly_omittable_visit_keys 直接省略。"
                "仅新增 discardable_objects 或仅新增 unassigned_intents 都不会降低当天计数。"
                "移动不是复制，同一键不能同时已安排又未分配；不得删除 strong/immutable。"
            )
        elif code.startswith("planner_day_transport_not_in_strategy"):
            remedy = (
                "错误代码已列出 day_index 和策略允许的交通方式。请对照 "
                "rejected_draft_reference_checks 该日 transport_mode_diff："
                "从 transport_preferences 删除 remove_not_allowed，只保留 strategy_allowed "
                "的子集；与之不兼容的 route_edge_keys 也必须改选合法备选或调整顺序。"
                "不能在草稿中临时扩大已接受策略。"
            )
        elif code.startswith("planner_draft_duplicate_objects"):
            remedy = (
                "错误代码和 rejected_draft_reference_checks 已列出重复 c/f 短键及其日/位置。"
                "每个对象在整个旅行 ordered_items 中只能出现一次；请删除重复副本，"
                "不要删除唯一的 strong/immutable，也不要创造替代键。"
            )
        elif code.startswith("planner_candidate_assigned_and_unassigned"):
            remedy = (
                "错误代码 keys 已列出同时出现于 ordered_items 和 unassigned_intents 的 c 键。"
                "每个键只能二选一：能够安排就只保留在某一天；确实无法安排的 strong/soft"
                "才只保留在 unassigned_intents，并提供真实证据。不得在两处同时保留。"
            )
        elif code.startswith("planner_unknown_discardable_item_key"):
            remedy = (
                "discardable_objects.object_key 只能逐字引用本份草稿 ordered_items 中已经安排的"
                "对象；它表示物化时可舍弃的已安排项。未安排的 soft 必须改放 unassigned_intents，"
                "不能同时或改放 discardable_objects。请对照 rejected_draft_reference_checks"
                " 任一日期中的 discardable_key_diff，删除 invalid_known_keys 列出的整条记录，"
                "并删除不在 scheduled_object_keys 中的 unknown 记录；"
                "不能只改 mode 或 reason。若没有已安排项需要舍弃，请使用空数组。"
            )
        elif code.startswith("planner_unassigned_candidate_not_protected"):
            remedy = (
                "错误代码已列出误放候选及其真实 commitment。unassigned_intents.candidate_key"
                "只能逐字复制 allowed_unassigned_candidate_keys；filler/neutral 可直接不安排，"
                "不得放入 unassigned_intents，也不得改变其承诺等级。"
            )
        elif code.startswith("planner_unassigned_capacity_conflict_not_supported"):
            remedy = (
                "错误代码已列出 candidate_key。请在 rejected_unassigned_intent_checks 找到该候选："
                "capacity_support_failures 说明当前容量理由为何不成立。若"
                " available_slot_days 非空，"
                "优先自主选择其中一个合法日期，把候选加入该日 ordered_items 并从"
                " unassigned_intents 删除；同时按新顺序重建必要跨簇段与路线。"
                "若确有其他冲突，只能选择 guard_supported_alternative_reasons 中 available=true"
                " 的 reason_code 和 observation_keys。不能原样保留 capacity_conflict，"
                "也不能仅改 reason_summary。"
            )
        elif code.startswith("planner_unassigned_route_conflict_not_observed"):
            remedy = (
                "route_conflict 必须有当前 spatial Observation 和与该候选相关的真实路线。"
                "请对照 rejected_unassigned_intent_checks.guard_supported_alternative_reasons；"
                "若 route_conflict.available=false，必须把候选重新排入合法日期或选择其他已明确"
                "可用的 Guard 理由，不能把无路线证据写成路线冲突。"
            )
        elif code.startswith("planner_unknown_evidence_key:"):
            catalog = PlannerReferenceCatalog(workspace)
            common_keys = [
                key for key in ("strategy", "spatial", "hotel", "pool") if key in catalog.evidence
            ]
            remedy = (
                "observation_keys 每一项必须逐字复制本轮 evidence_keys 字典的键。"
                f"当前通用合法键为 {','.join(common_keys) or 'none'}；"
                "capacity_conflict 只用 strategy，route_conflict 只用 spatial，"
                "duplicate_experience 只用 pool，budget_conflict 使用 strategy 或 hotel；"
                "opening_conflict/infeasible_date 必须复制对应候选条目列出的具体 evidence_keys。"
                "不能填写 c/g/r/h 对象键、UUID、字段路径或自造 capacity/route/opening 名称。"
            )
        elif code.startswith("planner_cross_cluster_coverage:"):
            if ":must_cover_candidates=none:" in code:
                remedy = (
                    "错误代码 day_index 指向的日期没有任何非主簇 ordered_items，必须删除该日"
                    "所有跨簇段并将 cross_cluster_segments 整个数组设为 []；不是修改"
                    " covered_object_keys。酒店只属于 lodging_baseline，不属于 ordered_items，"
                    "即使酒店与活动簇不同也不能创建跨簇段。请逐字遵守"
                    " rejected_draft_reference_checks.cross_cluster_segments_required_value。"
                )
            else:
                remedy = (
                    "错误代码的 must_cover_candidates 已列出该日所有且仅有的非主簇 c 键。"
                    "cross_cluster_segments.covered_object_keys 的并集必须逐字等于该列表，"
                    "每个键恰好出现一次，不能包含主簇对象。请直接对照 "
                    "rejected_draft_reference_checks.coverage_diff：target_union 是精确目标，"
                    "删除 extra 和 duplicates，补全 missing；required_coverage_by_cluster "
                    "只展示你已选主簇下的确定性归属，不替你改日程决策。"
                )
        elif code.startswith("planner_cross_cluster_must_connect_primary"):
            remedy = (
                "错误代码已列出 day_index、segment_index、当天 primary 以及错误的 from/to g 键。"
                "该段必须将 from 或 to 的一端逐字改为 primary，另一端保留为被覆盖对象实际所属"
                "非主簇；对照 rejected_draft_reference_checks.segment_checks，不能连接两个非主簇。"
            )
        elif code.startswith("planner_cross_cluster_requires_distinct_clusters"):
            remedy = (
                "错误代码已列出 day_index、segment_index 和重复的 g 键。跨簇段两端必须不同，"
                "并且其中一端必须等于当天 primary_cluster_key。"
            )
        elif code.startswith("planner_cross_cluster_edge_wrong_endpoints"):
            remedy = (
                "错误代码已列出 day_index 和 segment_index。请直接执行该 segment_checks 中"
                " route_endpoint_diff 的精确差集：从本段 route_edge_keys 删除全部"
                " invalid_submitted_route_keys；然后为每个 missing_required_boundaries 从对应"
                " required_route_bindings.available_route_key_choices 恰好选择一个 r 键。"
                "from_object_key/to_object_key 必须与 ordered_items 实际有向相邻顺序完全一致；"
                "不能使用反向、其他地点、簇级、酒店或属于另一 segment 的路线。"
                "所选路线 mode 还必须出现在当天 transport_preferences。若没有匹配路线，"
                "请调整当天顺序/分簇，或在证据预算允许时请求该真实端点路线，不能借用别的边。"
            )
        elif code.startswith("planner_cross_cluster_boundary_route_missing"):
            remedy = (
                "该天每一处实际跨簇相邻边界都必须由 route_edge_keys 覆盖一次。"
                "请逐项对照 rejected_draft_reference_checks.boundary_routes；"
                "缺少可用 route_key 时调整顺序/分天，或请求真实端点路线，不能遗漏或编造。"
            )
        elif code.startswith("planner_cross_cluster_duplicate_route_alternative"):
            remedy = (
                "错误代码的 choose_exactly_one_of 已列出同一有向地点边界的路线备选键；"
                "只能从中选择一个 r 键，不能同时提交出租车、公交或步行等多种备选，也不能"
                "把备选耗时相加。请对照 rejected_draft_reference_checks.boundary_routes 的"
                " selected_route_keys 和 selection_rule，为每个实际边界保留恰好一条与当天"
                " transport_preferences 相容的路线。"
            )
        elif code.startswith(
            ("planner_cross_cluster_edge_unusable", "planner_unknown_cross_cluster_route_key")
        ):
            remedy = (
                "route_edge_keys 只能使用当前 routes 字典中的真实 r 键；路线必须 status=available、"
                "有 duration_minutes，且 mode 位于当天 transport_preferences。"
                "请从 boundary_routes 的 available_routes 复制，不能使用未知、过期或其他方式的边。"
            )
        elif code.startswith("planner_cross_cluster_has_no_strong_intent"):
            remedy = (
                "错误段的 covered_object_keys 中没有 strong/immutable，普通 want 是 soft，"
                "不能以 strong_user_intent 为跨区理由。当天主簇中的必去对象也不能为任意"
                "非主簇 soft/filler 背书。请结合 rejected_draft_reference_checks 检查具体段；"
                "只允许从 segment_checks.guard_eligible_reason_options 选择真实可用的替代理由；"
                "若 no_currently_supported_reason=true，必须重新选择分天/顺序/主簇，或将允许"
                "取舍的 soft 从 ordered_items 删除并写入 unassigned_intents；"
                "不能把想去改成必去，也不能静默删除 soft。"
            )
        elif code.startswith(
            (
                "planner_cross_cluster_reason_not_allowed",
                "planner_cross_cluster_has_no_date_specific_fact",
                "planner_cross_cluster_has_no_fixed_commitment",
                "planner_cross_cluster_anchor_missing",
            )
        ):
            remedy = (
                "当前理由不满足已接受策略或真实证据。请逐项使用"
                " rejected_draft_reference_checks.segment_checks.guard_eligible_reason_options；"
                "只能从其中选择 reason_code，并复制其 supporting_evidence_key_choices。"
                "若 no_currently_supported_reason=true，必须改变分天、顺序、主簇或合法取舍 soft，"
                "不能轮换理由枚举、升级承诺等级或编造 Observation。"
            )
        elif code.startswith("planner_contract_invalid:") and (
            "verified_global_route_improvement requires a comparison observation" in code
        ):
            allowed = (
                ",".join(
                    reason.value
                    for reason in (
                        workspace.planning_strategy.spatial_policy.allowed_cross_cluster_reasons
                    )
                )
                if workspace.planning_strategy
                else "none"
            )
            comparisons = ",".join(
                key
                for key in PlannerReferenceCatalog(workspace).evidence
                if key.startswith("comparison")
            )
            remedy = (
                "verified_global_route_improvement 只有在已接受策略允许，且本轮存在真实 comparison"
                " 短键并与整份草稿路径完全一致时才可使用。"
                f"当前策略允许理由为 {allowed or 'none'}；当前 comparison 键为"
                f" {comparisons or 'none'}。请改用 segment_checks.guard_eligible_reason_options"
                " 中的理由；若该列表为空则必须重排或合法取舍 soft，不能补造 comparison 键。"
            )
        violation = compile_guard_violation(
            workspace,
            action=action,
            stage=stage,
            raw_code=code,
        )
        observation = PlannerGuardObservation(
            observation_id=str(uuid4()),
            attempted_action=action,
            code=code[:200],
            message=f"本次提议未执行：{code[:1000]}。{remedy}{retry_instruction}",
            based_on_workspace_revision=workspace.workspace_revision,
            violations=(violation,),
            failure_fingerprint=violation.failure_fingerprint,
        )
        return advance(workspace, guard_observations=(*workspace.guard_observations, observation))

    async def compose_response(
        self,
        workspace: PlannerWorkspaceState,
        book: TaskBookV4,
        cancellation: ModelCancellation,
        *,
        change_request: PlanChangeRequest | None = None,
        gateway: ModelGateway | None = None,
    ) -> str:
        gateway = gateway or self.gateway
        chunks: list[str] = []
        call_id: str | None = None
        try:
            async for chunk in gateway.stream_text(
                build_planner_response_request(
                    workspace,
                    book,
                    change_request=change_request,
                ),
                cancellation=cancellation,
            ):
                call_id = chunk.audit_call_id or call_id
                chunks.append(chunk.delta)
                if sum(map(len, chunks)) > 2000 or chunk.finish_reason == "length":
                    raise PlannerGuardError("planner_response_too_long_or_truncated")
            text = "".join(chunks).strip()
            if len(text) < 12:
                raise PlannerGuardError("planner_response_empty_or_incomplete")
            if not validate_response_visit_claims(text, workspace):
                raise PlannerGuardError("planner_response_unscheduled_visit_claim")
            if (
                schedule_quality_gaps(workspace, book) or schedule_coverage_issues(workspace, book)
            ) and re.search(
                r"无(?:未填补时段|空档|空闲)|没有(?:空档|空闲)|全天(?:充实|排满)|每一分钟.*(?:安排|利用)",
                text,
            ):
                raise PlannerGuardError("planner_response_unfilled_time_claim")
            if (
                workspace.react_state is not None
                and workspace.status is PlannerStatus.READY_TO_PUBLISH
                and re.search(r"草稿|待确认|未形成正式行程|未产出正式行程", text)
            ):
                raise PlannerGuardError("planner_response_result_must_not_be_draft")
            if workspace.status is PlannerStatus.READY_TO_PUBLISH and "正式行程" not in text:
                text = f"正式行程已生成。\n\n{text}"
            if book.lodging_direction.not_applicable and re.search(
                r"最终酒店|酒店(?:选择|推荐|候选|价格|库存).{0,8}(?:待|尚)|(?:待|尚).{0,8}酒店",
                text,
            ):
                raise PlannerGuardError("planner_response_lodging_not_applicable")
            for claim in re.finditer(
                r"(?:已|已经)(?:成功)?(?:发布|预订|订好)|预订成功|行程全部完成", text
            ):
                prefix = text[max(0, claim.start() - 12) : claim.start()]
                if not re.search(r"不|未|尚|不能|并非", prefix):
                    raise PlannerGuardError("planner_response_cannot_claim_publication_or_booking")
        except PlannerGuardError as error:
            await record_model_call_annotation(
                gateway,
                call_id,
                "llm_business_guard",
                {
                    "business_guard_result": {
                        "status": "rejected",
                        "guard": "planner_response_guard",
                        "reason": error.code,
                    },
                    "accepted_or_rejected": "rejected",
                    "failure_stage": "business_guard",
                    "failure_code": error.code,
                    "agent_output_full": "".join(chunks).strip(),
                },
            )
            raise
        await record_model_call_annotation(
            gateway,
            call_id,
            "llm_business_guard",
            {
                "business_guard_result": {
                    "status": "accepted",
                    "guard": "planner_response_guard",
                },
                "accepted_or_rejected": "accepted",
                "agent_output_full": text,
                "materialized_output": {"text": text},
            },
        )
        return text


def _with_local_change_scope(request: ModelRequest, context: PlannerGraphContext) -> ModelRequest:
    if context.local_change_dates is None:
        return request
    return request.model_copy(
        update={
            "messages": [
                *request.messages,
                ModelMessage(
                    role=ModelRole.SYSTEM,
                    content=(
                        "本轮是已发布行程的局部编辑。后续修复仅限这些日期及其相邻路线："
                        + ",".join(str(day) for day in sorted(context.local_change_dates))
                        + "。其他日期的地点、顺序、餐次、期望时段、住宿与交通偏好保持不变。"
                    ),
                ),
            ]
        }
    )


def _guard_local_day_changes(
    before: WorkingItineraryDraft,
    after: WorkingItineraryDraft,
    allowed_dates: frozenset[date],
) -> None:
    """Ignore recompiled technical IDs, not user-visible choices or windows."""

    def protected_days(draft: WorkingItineraryDraft) -> list[dict[str, object]]:
        return [
            {
                "date": day.service_date,
                "start_time": day.start_time,
                "end_time": day.end_time,
                "items": [
                    {
                        "identity": (
                            item.object_ref.canonical_entity_id
                            if isinstance(item.object_ref, CandidateRef)
                            else item.object_ref.commitment_id
                        ),
                        "kind": item.item_kind,
                        "window": item.expected_window,
                        "meal": item.meal_slot,
                        "onsite_lunch": item.onsite_lunch,
                        "duration": item.duration_preference,
                        "commitment": item.commitment_level,
                    }
                    for item in day.ordered_items
                ],
                "modes": day.transport_preferences,
                "selected_modes": day.route_mode_selections,
            }
            for day in draft.days
            if day.service_date not in allowed_dates
        ]

    if protected_days(before) != protected_days(after):
        raise PlannerGuardError("planner_repair_outside_local_change_dates")


def _is_terminal_hotel_failure(code: str, workspace: PlannerWorkspaceState) -> bool:
    if code.startswith("planner_required_hotel_provider_unavailable"):
        return True
    hotel = workspace.hotel_observation
    return bool(
        hotel is not None
        and hotel.mode == "search"
        and hotel.status == "unavailable"
        and not hotel.offers
        and code.startswith(
            (
                "planner_hotel_search_already_observed",
                "planner_draft_hotel_selection_required",
                "planner_draft_hotel_observation_missing",
                "planner_unknown_hotel_key",
            )
        )
    )


def _safe_schema_error(error: Exception) -> str:
    if isinstance(error, ValidationError):
        return "planner_contract_invalid:" + ";".join(
            ".".join(map(str, item["loc"])) + ":" + item["type"] + ":" + item["msg"]
            for item in error.errors(include_input=False, include_context=False, include_url=False)[
                :4
            ]
        )
    if isinstance(error, KeyError):
        return "planner_unknown_reference_key"
    # Guard code messages do not contain Provider payloads or credentials.
    return "planner_context_invalid:" + str(error)[:600]


def _model_schema_error_code(error: ModelGatewayError) -> str:
    details = ";".join(error.validation_issues[:4])
    code = "planner_model_schema_invalid" + (f":{details}" if details else "")
    if any("expected_window.part_of_day:" in issue for issue in error.validation_issues):
        code += ":allowed_part_of_day=morning,midday,afternoon,evening,anytime"
    if any(
        "strategy.spatial_policy.allowed_cross_cluster_reasons" in issue
        for issue in error.validation_issues
    ):
        code += (
            ":allowed_cross_cluster_reasons=strong_user_intent,date_specific_availability,"
            "reservation_or_fixed_commitment,lodging_or_transport_anchor,"
            "verified_global_route_improvement"
        )
    return code
