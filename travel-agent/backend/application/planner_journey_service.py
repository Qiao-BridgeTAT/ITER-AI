"""Confirmed-task-book Planner admission, recovery, cancellation and committed replies."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import suppress
from datetime import UTC, datetime
from time import monotonic
from typing import Any, Literal, cast
from uuid import UUID, uuid4

from pydantic import ValidationError

from backend.agent.model_audit import (
    ModelAuditError,
    ModelAuditExecutionContext,
    ModelAuditRecorder,
    NoopModelAuditRecorder,
    activate_model_audit_execution,
    record_execution_event,
    reset_model_audit_execution,
)
from backend.agent.model_gateway import ModelCancellation, ModelFailureCode, ModelGatewayError
from backend.agent.planner.best_effort import recover_best_effort
from backend.agent.planner.daily_repair import (
    bind_execution_budget,
    execution_remaining,
    refresh_repair_state,
)
from backend.agent.planner.dependencies import rebase_workspace_for_plan_change
from backend.agent.planner.finalization import build_finalize_decision
from backend.agent.planner.graph import PlannerAgentGraph, PlannerGraphContext
from backend.agent.planner.plan_change_router import (
    build_plan_change_request,
    requests_hotel_replacement,
    route_published_plan_message,
)
from backend.agent.planner.proposals import PlannerReferenceCatalog
from backend.agent.planner.publication import build_planner_published_plan
from backend.agent.planner.timing_quality import (
    has_time_quality_issues,
    natural_day_limitations,
)
from backend.agent.planner.transport_selection import apply_transport_selection
from backend.agent.planner.workspace import (
    PlannerGuardError,
    advance,
    confirmed_task_book,
    entity_intents,
    initial_workspace,
    server_id,
    service_dates,
)
from backend.agent.prepare.task_book import project_plan_change_task_book
from backend.application.prepare_journey_service import (
    PrepareEventEmitter,
    PrepareJourneyError,
    PrepareJourneyService,
    PrepareTurnTransformContext,
)
from backend.contracts.v4.cards import SpecificCandidateCard
from backend.contracts.v4.commands import (
    V4CancelGenerationCommand,
    V4ClientCommand,
    V4PlannerAnswerCommand,
    V4PlannerResumeCommand,
    V4PlanTransportSelectionCommand,
    V4TaskBookConfirmationCommand,
    V4UserMessageCommand,
    V4UserMessagePayload,
)
from backend.contracts.v4.conversation import (
    AgentStatusEvent,
    ConversationSnapshotV4,
    ConversationView,
    PlannerInteractionPublicView,
    PlannerWorkspacePublicView,
    TurnAcceptedEvent,
    TurnCancelledEvent,
    TurnFailedEvent,
    V4Attachment,
)
from backend.contracts.v4.enums import InteractionStatus, PlannerStatus
from backend.contracts.v4.plan_change import PlanChangeRequest
from backend.contracts.v4.planner_evidence import PlannerCandidateOrigin, PlannerInteractionAnswer
from backend.contracts.v4.planner_observations import PlannerInteractionOption
from backend.contracts.v4.planner_workspace import PlannerWorkspaceState
from backend.contracts.v4.state import V4TripStateEnvelope
from backend.contracts.v4.task_book import TaskBookV4
from backend.domain.authorization import RequestActor
from backend.domain.discovery.state_merge import advance_without_operations
from backend.persistence.checkpoint_repository import (
    V4_PLANNER_CHECKPOINT_VERSION,
    CheckpointRepository,
    CheckpointStaleError,
    PlannerWorkspaceRecord,
    PlannerWorkspaceWrite,
)
from backend.persistence.outbox_repository import canonical_json_hash
from backend.persistence.redis_temporary import RedisTemporaryStore
from backend.persistence.turn_repository import (
    AssistantMessageWrite,
    TurnNotFoundError,
    TurnRepository,
    TurnResultWrite,
    UserMessageWrite,
)

PLANNER_EXECUTION_TIMEOUT_SECONDS = 300
PLANNER_RECOVERY_TIMEOUT_SECONDS = 110


class PlannerJourneyService:
    """Thin stage router: Prepare remains the owner of all task-book semantic changes."""

    def __init__(
        self,
        *,
        prepare: PrepareJourneyService,
        graph: PlannerAgentGraph,
        turns: TurnRepository,
        checkpoints: CheckpointRepository,
        temporary: RedisTemporaryStore,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        model_audit: ModelAuditRecorder | None = None,
    ) -> None:
        self.prepare = prepare
        self.graph = graph
        self.turns = turns
        self.checkpoints = checkpoints
        self.temporary = temporary
        self.clock = clock
        self.model_audit = model_audit or NoopModelAuditRecorder()
        self._cancellations: dict[str, ModelCancellation] = {}
        self._terminal_turns: set[str] = set()

    async def get_snapshot(self, actor: RequestActor, trip_id: UUID) -> ConversationSnapshotV4:
        snapshot = await self.prepare.get_snapshot(actor, trip_id)
        return await self._snapshot_with_workspace(actor, trip_id, snapshot)

    async def get_conversation_view(self, actor: RequestActor, trip_id: UUID) -> ConversationView:
        view = await self.prepare.get_conversation_view(actor, trip_id)
        snapshot = await self._snapshot_with_workspace(actor, trip_id, view.snapshot)
        return view.model_copy(update={"snapshot": snapshot})

    async def _snapshot_with_workspace(
        self,
        actor: RequestActor,
        trip_id: UUID,
        snapshot: ConversationSnapshotV4,
    ) -> ConversationSnapshotV4:
        owner = await self.prepare.resolve_owner(actor, trip_id)
        record = await self._current_record(owner, trip_id, snapshot.trip_state)
        if record is None:
            return snapshot
        workspace = PlannerWorkspaceState.model_validate(record.payload)
        active_generation = snapshot.active_generation_id
        if active_generation == workspace.generation_id and not await self.temporary.has_trip_lock(
            trip_id
        ):
            # Read-only recovery projection. A hard crash can leave a long-lived
            # generation marker; an expired lease must not hide the resume control.
            active_generation = None
        return ConversationSnapshotV4.model_validate(
            {
                **dict(snapshot),
                "planner_workspace": PlannerWorkspacePublicView(
                    trip_id=workspace.trip_id,
                    generation_id=workspace.generation_id,
                    based_on_task_book_id=workspace.based_on_task_book_id,
                    based_on_task_book_version=workspace.based_on_task_book_version,
                    workspace_revision=workspace.workspace_revision,
                    last_interaction_action=(
                        workspace.interaction_answers[-1].semantic_action
                        if workspace.interaction_answers
                        else None
                    ),
                    active_interaction=(
                        PlannerInteractionPublicView.model_validate(
                            workspace.active_interaction.model_dump(
                                mode="json",
                                include={
                                    "interaction_id",
                                    "reason_code",
                                    "option_contracts",
                                    "resume_token",
                                    "status",
                                },
                            )
                        )
                        if workspace.active_interaction is not None
                        else None
                    ),
                    status=workspace.status,
                ),
                "active_generation_id": active_generation,
            },
            context={"restore_historical_semantic_state": True},
        )

    async def handle_stream(
        self,
        actor: RequestActor,
        trip_id: UUID,
        raw_command: object,
        emit: PrepareEventEmitter,
        *,
        on_admitted: Callable[[], None] | None = None,
    ) -> None:
        try:
            command = V4ClientCommand.model_validate(raw_command).root
        except ValidationError as error:
            raise PrepareJourneyError("invalid_v4_command") from error
        owner = await self.prepare.resolve_owner(actor, trip_id)
        snapshot = await self.prepare.get_snapshot(actor, trip_id)
        record = await self._current_record(owner, trip_id, snapshot.trip_state)
        if (
            isinstance(command, V4CancelGenerationCommand)
            and record is not None
            and (command.payload.generation_id == record.generation_id)
        ):
            if on_admitted:
                on_admitted()
            await self._cancel(owner, trip_id, record, emit)
            return
        if isinstance(
            command,
            (V4PlannerResumeCommand, V4PlannerAnswerCommand, V4PlanTransportSelectionCommand),
        ):
            replay = await self.turns.find_admitted_turn(
                owner,
                trip_id,
                request_id=command.request_id,
                idempotency_key=command.idempotency_key,
                request_fingerprint=canonical_json_hash(command.model_dump(mode="json")),
            )
            if replay is not None and replay.status == "committed":
                if on_admitted:
                    on_admitted()
                committed = await self.turns.load_committed_turn(owner, replay.turn_id)
                await emit(
                    TurnAcceptedEvent(
                        event_id=str(uuid4()),
                        event_type="turn.accepted",
                        trip_id=str(trip_id),
                        turn_id=str(replay.turn_id),
                        generation_id=str(replay.generation_id),
                        sequence=0,
                        emitted_at=self.clock(),
                        base_state_version=replay.base_state_version,
                    ).model_dump(mode="json")
                )
                await self.prepare.dispatch_committed(committed.outbox_cursor, emit, replay=True)
                return
            if record is None or command.payload.generation_id != record.generation_id:
                raise PrepareJourneyError("planner_workspace_not_current")
            if command.expected_state_version != snapshot.trip_state.semantic_state.state_version:
                raise PrepareJourneyError("state_version_conflict")
            workspace = PlannerWorkspaceState.model_validate(record.payload)
            if workspace.workspace_revision != command.payload.expected_workspace_revision:
                raise PrepareJourneyError("planner_workspace_revision_conflict")
            if isinstance(command, V4PlanTransportSelectionCommand):
                plan = snapshot.trip_state.published_plan
                if plan is None:
                    raise PrepareJourneyError("planner_published_plan_required")
                try:
                    apply_transport_selection(workspace, plan, command)
                except PlannerGuardError as error:
                    raise PrepareJourneyError(error.code) from error
            if isinstance(command, V4PlannerAnswerCommand):
                _validate_answer(workspace, command)
            else:
                if workspace.status in {
                    PlannerStatus.AWAITING_USER,
                    PlannerStatus.STALE,
                }:
                    raise PrepareJourneyError("planner_resume_not_allowed")
                if workspace.interaction_answers and workspace.interaction_answers[
                    -1
                ].semantic_action in {"revise_task_book", "supply_booking_detail"}:
                    raise PrepareJourneyError("planner_task_book_reconfirmation_required")
            await self._run(
                actor,
                trip_id,
                snapshot.trip_state,
                emit,
                # Explicit retry after an exhausted unpublished attempt starts
                # a new bounded generation, not the same already-exhausted loop.
                # Existing published plans and all old checkpoints stay intact.
                record=(
                    None
                    if isinstance(command, V4PlannerResumeCommand)
                    and snapshot.trip_state.published_plan is None
                    and workspace.status is PlannerStatus.FAILED
                    and workspace.guard_observations
                    and workspace.guard_observations[-1].code == "planner_revision_budget_exhausted"
                    else record
                ),
                command=command,
                on_admitted=on_admitted,
            )
            if isinstance(command, V4PlannerAnswerCommand) and command.payload.optional_user_text:
                # Exact user text is interpreted by Prepare; Planner never edits the confirmed book.
                refreshed = await self.prepare.get_snapshot(actor, trip_id)
                forwarded = V4UserMessageCommand(
                    request_id=UUID(server_id(command.request_id, "prepare-handoff")),
                    idempotency_key=f"planner-handoff:{command.request_id}",
                    expected_state_version=refreshed.trip_state.semantic_state.state_version,
                    client_sequence=command.client_sequence + 1,
                    type="user_message",
                    payload=V4UserMessagePayload(
                        message_id=UUID(server_id(command.request_id, "text")),
                        text=command.payload.optional_user_text,
                    ),
                )
                await self.prepare.handle_stream(
                    actor, trip_id, forwarded.model_dump(mode="json"), emit
                )
            return
        # Reject stale commands before disturbing a running generation.
        if (
            record is not None
            and not isinstance(command, V4CancelGenerationCommand)
            and command.expected_state_version == snapshot.trip_state.semantic_state.state_version
            and await self.temporary.get_active_generation(trip_id) == str(record.generation_id)
        ):
            await self._cancel(owner, trip_id, record, emit)
        result_transformer = (
            self._transform_published_plan_turn
            if isinstance(command, V4UserMessageCommand)
            and snapshot.trip_state.published_plan is not None
            else None
        )
        await self.prepare.handle_stream(
            actor,
            trip_id,
            raw_command,
            emit,
            on_admitted=on_admitted,
            result_transformer=result_transformer,
        )
        if not isinstance(command, V4TaskBookConfirmationCommand):
            return
        refreshed = await self.prepare.get_snapshot(actor, trip_id)
        try:
            book = confirmed_task_book(refreshed.trip_state)
        except PlannerGuardError:
            return
        if (
            book.task_book_id != str(command.payload.task_book_id)
            or book.version != command.payload.task_book_version
        ):
            return
        if await self._current_record(owner, trip_id, refreshed.trip_state) is None:
            await self._run(actor, trip_id, refreshed.trip_state, emit, confirmation=command)

    async def _transform_published_plan_turn(
        self,
        context: PrepareTurnTransformContext,
        default_write: TurnResultWrite,
    ) -> TurnResultWrite:
        command = context.command
        prior_plan = context.previous_state.published_plan
        if not isinstance(command, V4UserMessageCommand) or prior_plan is None:
            return default_write
        route = route_published_plan_message(
            user_text=command.payload.text,
            user_message_id=command.payload.message_id,
            turn_id=context.turn_id,
            accepted_operations=context.result.accepted_operations,
            semantic_scope=next(
                (
                    decision.published_plan_intent
                    for decision in reversed(context.result.decisions)
                    if decision.published_plan_intent is not None
                ),
                None,
            ),
        )
        if route.scope in {"reply_only", "task_book_change"}:
            resulting_phase = (
                "task_reflection"
                if route.scope == "task_book_change" and default_write.state.published_plan is None
                else default_write.phase
            )
            return default_write.model_copy(
                update={
                    "phase": resulting_phase,
                    "decision_audit": {
                        **default_write.decision_audit,
                        "result_transform": "published_plan_route",
                        "plan_change_scope": route.scope,
                        "plan_change_reason": route.reason_code,
                    },
                }
            )

        book = confirmed_task_book(context.previous_state)
        planning_book = book
        if context.result.accepted_operations:
            try:
                planning_book = project_plan_change_task_book(
                    book,
                    context.result.candidate.semantic_state,
                    context.result.candidate.discovery_runtime_state,
                )
            except ValueError as error:
                raise PlannerGuardError(
                    "planner_plan_change_requires_task_book_reconfirmation"
                ) from error
        version_number = await self.turns.load_plan_version_number(
            context.owner_id,
            context.trip_id,
            prior_plan.plan_version_id,
        )
        change_request = build_plan_change_request(
            plan=prior_plan,
            base_plan_version=version_number,
            user_message_id=command.payload.message_id,
            route=route,
        )
        record = await self.checkpoints.load_latest_planner_workspace(
            context.owner_id,
            context.trip_id,
            generation_id=prior_plan.generation_id,
        )
        if record is None:
            raise PlannerGuardError("planner_plan_change_workspace_missing")
        previous_workspace = PlannerWorkspaceState.model_validate(record.payload)
        if (
            previous_workspace.working_itinerary != prior_plan.working_itinerary
            or previous_workspace.materialized_schedule != prior_plan.materialized_schedule
            or previous_workspace.cost_draft != prior_plan.cost_draft
        ):
            raise PlannerGuardError("planner_plan_change_workspace_drift")
        if (
            requests_hotel_replacement(command.payload.text)
            and not PlannerReferenceCatalog(previous_workspace).alternative_hotels
        ):
            return _unchanged_hotel_change_write(
                default_write,
                previous_state=context.previous_state,
            )
        workspace = rebase_workspace_for_plan_change(
            previous_workspace,
            generation_id=context.generation_id,
        )
        workspace = bind_execution_budget(workspace, str(context.turn_id), datetime.now(UTC))
        execution_deadline = monotonic() + execution_remaining(workspace)
        previous_revision = -1

        async def barrier() -> None:
            context.cancellation.raise_if_cancelled("planner_plan_change_barrier")
            if await self.temporary.get_active_generation(context.trip_id) != str(
                context.generation_id
            ):
                context.cancellation.cancel()
                context.cancellation.raise_if_cancelled("planner_plan_change_barrier")

        async def checkpoint(value: PlannerWorkspaceState) -> None:
            nonlocal workspace, previous_revision
            await barrier()
            saved = await self._save(
                context.owner_id,
                context.turn_id,
                context.previous_state,
                book,
                value,
                expected_revision=previous_revision,
            )
            previous_revision = saved.revision
            workspace = value

        async def progress(code: str, message: str) -> None:
            await barrier()
            await context.emit(
                AgentStatusEvent(
                    event_id=str(uuid4()),
                    event_type="agent.status",
                    trip_id=str(context.trip_id),
                    turn_id=str(context.turn_id),
                    generation_id=str(context.generation_id),
                    sequence=0,
                    emitted_at=self.clock(),
                    status_code=code,
                    message=message,
                ).model_dump(mode="json")
            )

        await checkpoint(workspace)
        # A full replan has the same bounded, validated recovery as first-time
        # planning. Local patches retain their exact scope and cannot silently
        # turn into a whole-plan best-effort rewrite.
        can_recover = getattr(self.graph, "complete_plan", False)
        try:
            async with asyncio.timeout(
                max(
                    0.1,
                    execution_deadline
                    - monotonic()
                    - (PLANNER_RECOVERY_TIMEOUT_SECONDS if can_recover else 20),
                )
            ):
                workspace = await self.graph.apply_plan_change(
                    workspace,
                    PlannerGraphContext(
                        book=planning_book,
                        cancellation=context.cancellation,
                        checkpoint=checkpoint,
                        progress=progress,
                        input_state_version=context.previous_state.semantic_state.state_version,
                        deadline=execution_deadline,
                    ),
                    change_request=change_request,
                    user_text=command.payload.text,
                    refresh_evidence=planning_book != book,
                )
        except (TimeoutError, ModelGatewayError) as error:
            if (
                not can_recover
                or workspace.working_itinerary is None
                or context.cancellation.is_cancelled
                or (
                    isinstance(error, ModelGatewayError)
                    and error.code
                    in {
                        ModelFailureCode.AUDIT_UNAVAILABLE,
                        ModelFailureCode.CANCELLED,
                        ModelFailureCode.AUTHENTICATION_FAILED,
                        ModelFailureCode.PERMISSION_DENIED,
                    }
                )
            ):
                raise
            if workspace.status is not PlannerStatus.READY_TO_PUBLISH:
                workspace = advance(workspace, status=PlannerStatus.FAILED)
                await checkpoint(workspace)
        if route.scope == "local_replan" and not workspace.accepted_local_change_dates:
            # No local patch passed. The rebased workspace still contains the
            # previous publication's artifacts/input version; neither global
            # quality repair nor publishing those artifacts is authorized.
            return _unchanged_plan_change_write(
                default_write, previous_state=context.previous_state
            )
        if can_recover and (
            workspace.status is PlannerStatus.READY_TO_PUBLISH or route.scope != "full_replan"
        ):
            workspace = await self._finish_pending_schedule_quality(
                workspace,
                PlannerGraphContext(
                    book=planning_book,
                    cancellation=context.cancellation,
                    checkpoint=checkpoint,
                    progress=progress,
                    input_state_version=context.previous_state.semantic_state.state_version,
                    deadline=execution_deadline,
                    local_change_dates=workspace.accepted_local_change_dates,
                ),
            )
        elif can_recover and workspace.status not in {
            PlannerStatus.READY_TO_PUBLISH,
            PlannerStatus.CANCELLED,
            PlannerStatus.STALE,
        }:
            await progress("planner_best_effort", "正在核验本轮可用安排并完成行程。")
            recovery_context = PlannerGraphContext(
                book=planning_book,
                cancellation=context.cancellation,
                checkpoint=checkpoint,
                progress=progress,
                input_state_version=context.previous_state.semantic_state.state_version,
                deadline=execution_deadline,
            )
            async with asyncio.timeout(max(0.1, execution_deadline - monotonic() - 2)):
                workspace = await recover_best_effort(
                    workspace,
                    planning_book,
                    context.cancellation,
                    materializer=self.graph.materializer,
                    validator=self.graph.validator,
                    checkpoint=checkpoint,
                    input_state_version=context.previous_state.semantic_state.state_version,
                    refresh_evidence=lambda value: self.graph.refresh_recovery_evidence(
                        value, planning_book, context.cancellation
                    ),
                    repair_schedule=lambda value: self.graph.repair_recovery_schedule(
                        value, recovery_context
                    ),
                )
        if workspace.status is not PlannerStatus.READY_TO_PUBLISH:
            raise PlannerGuardError("planner_plan_change_not_publishable")
        workspace = refresh_repair_state(workspace, planning_book, stop_reason="publication")
        await checkpoint(workspace)

        plan_version_id = UUID(server_id(context.turn_id, "plan-version"))
        publication_key = f"planner-plan:{context.turn_id}"
        published_plan = build_planner_published_plan(
            workspace,
            planning_book,
            plan_version_id=plan_version_id,
            publication_key=publication_key,
            based_on_state_version=context.previous_state.semantic_state.state_version,
            change_request=change_request,
            clock=self.clock,
        )
        generation_mode: Literal["qwen", "fallback"] = "qwen"
        failure_code: str | None = None
        try:
            if execution_remaining(workspace, calls=True) <= 0:
                raise TimeoutError("planner_response_budget_exhausted")
            async with asyncio.timeout(execution_remaining(workspace, calls=True)):
                response_text = await self.graph.compose_response(
                    workspace,
                    planning_book,
                    context.cancellation,
                    change_request=change_request,
                )
        except (TimeoutError, PlannerGuardError, ModelGatewayError) as error:
            if context.cancellation.is_cancelled or (
                isinstance(error, ModelGatewayError)
                and error.code is ModelFailureCode.AUDIT_UNAVAILABLE
            ):
                raise
            generation_mode = "fallback"
            failure_code = "planner_response_fallback"
            response_text = _formal_plan_fallback(workspace, planning_book)
        await barrier()
        aggregate = advance_without_operations(
            context.previous_state.semantic_state,
            context.previous_state.discovery_runtime_state,
        )
        affected_dates = (
            service_dates(planning_book)
            if route.scope == "full_replan"
            else tuple(
                sorted(
                    {
                        value
                        for decision in workspace.decision_trace
                        if decision.input_refs.plan_change_request_id
                        == change_request.plan_change_request_id
                        and hasattr(decision.payload, "declared_affected_dates")
                        for value in decision.payload.declared_affected_dates
                    }
                )
            )
        )
        return TurnResultWrite(
            state=V4TripStateEnvelope(
                semantic_state=aggregate.semantic_state,
                discovery_runtime_state=aggregate.runtime_state,
                current_plan_version_id=plan_version_id,
                published_plan=published_plan,
            ),
            phase="confirmed",
            assistant_message=AssistantMessageWrite(
                message_id=default_write.assistant_message.message_id,
                generation_id=context.generation_id,
                message_type="plan",
                text=response_text,
                attachments=[V4Attachment(root=published_plan)],
                message_metadata={
                    "stage": "V4-05",
                    "formal_plan_published": True,
                    "base_plan_version_id": str(prior_plan.plan_version_id),
                    "plan_change_request_id": change_request.plan_change_request_id,
                    "affected_dates": [value.isoformat() for value in affected_dates],
                },
            ),
            outbox_id=default_write.outbox_id,
            outbox_cursor=default_write.outbox_cursor,
            publication_key=publication_key,
            generation_mode=generation_mode,
            failure_code=failure_code,
            tool_observations=default_write.tool_observations,
            decision_audit={
                "result_transform": "published_plan_replan",
                "plan_change_scope": route.scope,
                "plan_change_reason": route.reason_code,
                "plan_change_request": change_request.model_dump(mode="json"),
                "workspace_revision": workspace.workspace_revision,
                "plan_version_id": str(plan_version_id),
            },
            plan_version_id=plan_version_id,
            outcome="plan_ready",
        )

    async def _finish_pending_schedule_quality(
        self, workspace: PlannerWorkspaceState, context: PlannerGraphContext
    ) -> PlannerWorkspaceState:
        """Spend the existing reserve on visible gaps even after safety passed.

        Safety-passed is not quality-complete. This is mutually exclusive with
        best-effort recovery, so the total execution allowance is unchanged.
        """
        if (
            workspace.plan_change_request is not None
            and workspace.plan_change_request.requested_scope == "local_replan"
            and not workspace.accepted_local_change_dates
        ):
            return workspace
        if not getattr(self.graph, "optimize_timing", False) or not has_time_quality_issues(
            workspace, context.book
        ):
            return workspace
        await context.progress("planner_time_quality", "正在补齐剩余游览空档与具体餐厅。")
        try:
            async with asyncio.timeout(max(0.1, context.remaining_seconds() - 2)):
                candidate = await self.graph.repair_recovery_schedule(workspace, context)
            if (
                candidate.validation_observation is None
                or candidate.validation_observation.result != "passed"
            ):
                return workspace
            context.cancellation.raise_if_cancelled("planner_quality_publication")
            decision = build_finalize_decision(candidate)
            candidate = advance(
                candidate,
                decision_trace=(
                    *(
                        previous
                        for previous in candidate.decision_trace
                        if previous.decision_id != decision.decision_id
                    ),
                    decision,
                ),
                status=PlannerStatus.READY_TO_PUBLISH,
            )
        except (TimeoutError, PlannerGuardError, ValidationError):
            # Never lose the already validated plan if the optional final pass
            # cannot finish. Detailed issues stay internal.
            return workspace
        await context.checkpoint(candidate)
        return candidate

    async def _current_record(
        self, owner: UUID, trip_id: UUID, state: V4TripStateEnvelope
    ) -> PlannerWorkspaceRecord | None:
        try:
            book = confirmed_task_book(state)
        except PlannerGuardError:
            return None
        record = await self.checkpoints.load_latest_planner_workspace(owner, trip_id)
        if record is None or (
            str(record.confirmed_task_book_id) != book.task_book_id
            or record.confirmed_task_book_version != book.version
            or record.confirmed_task_book_hash != canonical_json_hash(book.model_dump(mode="json"))
        ):
            return None
        return record

    async def _run(
        self,
        actor: RequestActor,
        trip_id: UUID,
        state: V4TripStateEnvelope,
        emit: PrepareEventEmitter,
        *,
        record: PlannerWorkspaceRecord | None = None,
        command: V4PlannerResumeCommand
        | V4PlannerAnswerCommand
        | V4PlanTransportSelectionCommand
        | None = None,
        confirmation: V4TaskBookConfirmationCommand | None = None,
        on_admitted: Callable[[], None] | None = None,
    ) -> None:
        owner = await self.prepare.resolve_owner(actor, trip_id)
        book = confirmed_task_book(state)  # Must precede Qwen/Provider calls.
        workspace = (
            PlannerWorkspaceState.model_validate(record.payload)
            if record
            else initial_workspace(state, uuid4(), self.clock())
        )
        if record is None:
            snapshot = await self.prepare.get_snapshot(actor, trip_id)
            if (
                snapshot.trip_state.semantic_state.state_version
                != state.semantic_state.state_version
            ):
                raise PrepareJourneyError("state_version_conflict")
            workspace = workspace.model_copy(
                update={
                    "candidate_origins": _candidate_origins(snapshot, book),
                }
            )
        if isinstance(command, V4PlanTransportSelectionCommand):
            assert state.published_plan is not None
            # A failed route-change generation is not the source of truth for
            # an unchanged published itinerary or its accepted repair receipts.
            published_record = await self.checkpoints.load_latest_planner_workspace(
                owner, trip_id, generation_id=state.published_plan.generation_id
            )
            if published_record is None:
                raise PrepareJourneyError("planner_published_workspace_missing")
            workspace = PlannerWorkspaceState.model_validate(published_record.payload)
            workspace = apply_transport_selection(workspace, state.published_plan, command)
            workspace = rebase_workspace_for_plan_change(workspace, generation_id=uuid4())
        pending_change = _pending_plan_change(workspace, state)
        generation = UUID(workspace.generation_id)
        request_id = (
            command.request_id
            if command
            else UUID(server_id(book.task_book_id, book.version, "start"))
        )
        request_key = (
            command.idempotency_key if command else f"planner:{book.task_book_id}:{book.version}"
        )
        fingerprint = (
            canonical_json_hash(command.model_dump(mode="json"))
            if command
            else canonical_json_hash(
                {
                    "confirmed_task_book": book.model_dump(mode="json"),
                    "trigger": "task_book_confirmation",
                }
            )
        )
        accepted = None
        execution_started_at = datetime.now(UTC)
        cancellation = ModelCancellation()
        heartbeat: asyncio.Task[None] | None = None
        audit_context: ModelAuditExecutionContext | None = None
        audit_token = None
        previous_revision = (
            record.revision
            if record and not isinstance(command, V4PlanTransportSelectionCommand)
            else -1
        )
        committed = False
        token = str(uuid4())
        if not await self.temporary.acquire_trip_lock(trip_id, token, 75):
            raise PrepareJourneyError("planner_already_processing", retryable=True)
        try:
            if record is not None:
                status = await self.checkpoints.planner_turn_status(owner, trip_id, record.turn_id)
                if status in {"accepted", "running"}:
                    # Lock is owned here: an abandoned execution may no longer publish.
                    await self.turns.mark_turn_failed(
                        owner,
                        record.turn_id,
                        failure_code="planner_execution_resumed",
                        cancelled=True,
                    )
            message_id = UUID(server_id(request_id, "message"))
            action_text = (
                "已确认旅行任务书，开始行程规划。"
                if command is None
                else (
                    "继续已保存的规划工作。"
                    if isinstance(command, V4PlannerResumeCommand)
                    else "将这段交通改为"
                    + {"taxi": "打车", "public_transit": "公共交通", "walking": "步行"}[
                        command.payload.transport_mode
                    ]
                    + "。"
                    if isinstance(command, V4PlanTransportSelectionCommand)
                    else "已选择规划取舍："
                    + _answer_option(workspace, command).verified_impact_summary
                )
            )
            accepted = await self.turns.accept_turn(
                owner,
                trip_id,
                request_id=request_id,
                idempotency_key=request_key,
                request_fingerprint=fingerprint,
                base_state_version=state.semantic_state.state_version,
                generation_id=generation,
                user_message=UserMessageWrite(
                    message_id=message_id,
                    client_message_id=message_id,
                    text=action_text,
                    message_metadata={
                        "trigger": "task_book_confirmation" if confirmation else "planner_control"
                    },
                ),
            )
            if on_admitted:
                on_admitted()
            await emit(
                TurnAcceptedEvent(
                    event_id=str(uuid4()),
                    event_type="turn.accepted",
                    trip_id=str(trip_id),
                    turn_id=str(accepted.turn_id),
                    generation_id=str(generation),
                    sequence=0,
                    emitted_at=self.clock(),
                    base_state_version=accepted.base_state_version,
                ).model_dump(mode="json")
            )
            if accepted.idempotent_replay:
                if accepted.status == "committed":
                    replay = await self.turns.load_committed_turn(owner, accepted.turn_id)
                    await self.prepare.dispatch_committed(replay.outbox_cursor, emit, replay=True)
                return
            audit_context = ModelAuditExecutionContext(
                trace_id=str(request_id),
                trip_id=str(trip_id),
                turn_id=str(accepted.turn_id),
                generation_id=str(generation),
                user_message_id=str(message_id),
                input_kind=(
                    "task_book_confirmation"
                    if confirmation is not None
                    else command.type
                    if command is not None
                    else "planner_start"
                ),
                user_input_full=(
                    command.model_dump(mode="json")
                    if command is not None
                    else confirmation.model_dump(mode="json")
                    if confirmation is not None
                    else {
                        "confirmed_task_book": book.model_dump(mode="json"),
                        "trigger": "task_book_confirmation",
                    }
                ),
            )
            audit_context.attach_recorder(self.model_audit)
            await record_execution_event(
                audit_context,
                "conversation_input_admitted",
                {
                    "base_state_version": accepted.base_state_version,
                    "request_id": str(request_id),
                    "planner_workspace_revision": workspace.workspace_revision,
                },
            )
            audit_token = activate_model_audit_execution(audit_context)
            await self.temporary.replace_active_generation(trip_id, generation, 3600)
            self._cancellations[str(generation)] = cancellation

            async def barrier() -> None:
                cancellation.raise_if_cancelled("planner_workspace_barrier")
                if not await self.temporary.renew_trip_lock(
                    trip_id, token, 75
                ) or await self.temporary.get_active_generation(trip_id) != str(generation):
                    cancellation.cancel()
                    cancellation.raise_if_cancelled("planner_workspace_barrier")

            async def checkpoint(value: PlannerWorkspaceState) -> None:
                nonlocal workspace, previous_revision
                await barrier()
                assert accepted is not None
                saved = await self._save(
                    owner, accepted.turn_id, state, book, value, expected_revision=previous_revision
                )
                previous_revision = saved.revision
                workspace = value

            async def progress(code: str, message: str) -> None:
                await barrier()
                assert accepted is not None
                await emit(
                    AgentStatusEvent(
                        event_id=str(uuid4()),
                        event_type="agent.status",
                        trip_id=str(trip_id),
                        turn_id=str(accepted.turn_id),
                        generation_id=str(generation),
                        sequence=0,
                        emitted_at=self.clock(),
                        status_code=code,
                        message=message,
                    ).model_dump(mode="json")
                )

            async def keep_lease() -> None:
                while True:
                    try:
                        await asyncio.wait_for(cancellation.wait_cancelled(), timeout=15)
                        return
                    except TimeoutError:
                        if (
                            not committed
                            and await self.temporary.get_active_generation(trip_id)
                            != str(generation)
                        ) or not await self.temporary.renew_trip_lock(trip_id, token, 75):
                            cancellation.cancel()
                            return

            heartbeat = asyncio.create_task(keep_lease())
            should_plan = True
            if isinstance(command, V4PlannerAnswerCommand):
                option = _answer_option(workspace, command)
                assert workspace.active_interaction is not None
                answer = PlannerInteractionAnswer(
                    answer_id=str(command.payload.answer_id),
                    interaction_id=str(command.payload.interaction_id),
                    option_id=option.option_id,
                    semantic_action=cast(
                        Literal["keep_task_book", "revise_task_book", "supply_booking_detail"],
                        option.semantic_action,
                    ),
                    user_text=command.payload.optional_user_text,
                    source_turn_id=str(accepted.turn_id),
                )
                workspace = advance(
                    workspace,
                    active_interaction=workspace.active_interaction.model_copy(
                        update={"status": InteractionStatus.ANSWERED}
                    ),
                    interaction_answers=(*workspace.interaction_answers, answer),
                    status=(
                        PlannerStatus.PLANNING
                        if option.semantic_action == "keep_task_book"
                        else PlannerStatus.CANCELLED
                    ),
                )
                should_plan = option.semantic_action == "keep_task_book"
                if should_plan:
                    workspace = advance(
                        workspace,
                        unresolved_decisions=(),
                        segment_attempt_count=0,
                        segment_evidence_count=0,
                        revision_round=0,
                        materialized_schedule=None,
                        cost_draft=None,
                        validation_report=None,
                        validation_observation=None,
                        timing_optimization_pending=getattr(self.graph, "optimize_timing", False),
                    )
            else:
                if record is not None and not isinstance(command, V4PlanTransportSelectionCommand):
                    workspace = _resume_workspace(
                        workspace, optimize_timing=getattr(self.graph, "optimize_timing", False)
                    )
            workspace = advance(
                workspace,
                best_effort_reasons=tuple(
                    dict.fromkeys((*workspace.best_effort_reasons, *natural_day_limitations(book)))
                ),
            )
            workspace = bind_execution_budget(
                workspace, str(accepted.turn_id), execution_started_at
            )
            execution_deadline = monotonic() + execution_remaining(workspace)
            await checkpoint(workspace)
            if should_plan:
                can_recover = getattr(self.graph, "complete_plan", False) and not isinstance(
                    command, V4PlanTransportSelectionCommand
                )
                try:
                    async with asyncio.timeout(
                        max(
                            0.1, execution_deadline - monotonic() - PLANNER_RECOVERY_TIMEOUT_SECONDS
                        )
                    ):
                        workspace = await self.graph.invoke(
                            workspace,
                            PlannerGraphContext(
                                book=book,
                                cancellation=cancellation,
                                checkpoint=checkpoint,
                                progress=progress,
                                input_state_version=accepted.base_state_version,
                                allow_semantic_repair=not isinstance(
                                    command, V4PlanTransportSelectionCommand
                                ),
                                local_change_dates=workspace.accepted_local_change_dates,
                                deadline=execution_deadline,
                            ),
                        )
                except (TimeoutError, ModelGatewayError) as error:
                    if (
                        not can_recover
                        or workspace.working_itinerary is None
                        or cancellation.is_cancelled
                        or (
                            isinstance(error, ModelGatewayError)
                            and error.code
                            in {
                                ModelFailureCode.AUDIT_UNAVAILABLE,
                                ModelFailureCode.CANCELLED,
                                ModelFailureCode.AUTHENTICATION_FAILED,
                                ModelFailureCode.PERMISSION_DENIED,
                            }
                        )
                    ):
                        raise
                    workspace = advance(workspace, status=PlannerStatus.FAILED)
                    await checkpoint(workspace)
                if can_recover and (
                    workspace.status is PlannerStatus.READY_TO_PUBLISH
                    or workspace.accepted_local_change_dates is not None
                ):
                    workspace = await self._finish_pending_schedule_quality(
                        workspace,
                        PlannerGraphContext(
                            book=book,
                            cancellation=cancellation,
                            checkpoint=checkpoint,
                            progress=progress,
                            input_state_version=accepted.base_state_version,
                            deadline=execution_deadline,
                            local_change_dates=workspace.accepted_local_change_dates,
                        ),
                    )
                elif can_recover and workspace.status not in {
                    PlannerStatus.READY_TO_PUBLISH,
                    PlannerStatus.CANCELLED,
                    PlannerStatus.STALE,
                }:
                    await progress("planner_best_effort", "正在保留可用安排并核验尽力完成版。")
                    recovery_context = PlannerGraphContext(
                        book=book,
                        cancellation=cancellation,
                        checkpoint=checkpoint,
                        progress=progress,
                        input_state_version=accepted.base_state_version,
                        deadline=execution_deadline,
                    )
                    async with asyncio.timeout(max(0.1, execution_deadline - monotonic() - 2)):
                        workspace = await recover_best_effort(
                            workspace,
                            book,
                            cancellation,
                            materializer=self.graph.materializer,
                            validator=self.graph.validator,
                            checkpoint=checkpoint,
                            input_state_version=accepted.base_state_version,
                            refresh_evidence=lambda value: self.graph.refresh_recovery_evidence(
                                value, book, cancellation
                            ),
                            repair_schedule=lambda value: self.graph.repair_recovery_schedule(
                                value, recovery_context
                            ),
                        )
            plan_version_id: UUID | None = None
            publication_key = f"planner-draft:{accepted.turn_id}"
            published_plan = None
            change_request = pending_change
            if isinstance(command, V4PlanTransportSelectionCommand):
                assert state.published_plan is not None
                version_number = await self.turns.load_plan_version_number(
                    owner, trip_id, state.published_plan.plan_version_id
                )
                mode_label = {"taxi": "打车", "public_transit": "公共交通", "walking": "步行"}[
                    command.payload.transport_mode
                ]
                change_request = build_plan_change_request(
                    plan=state.published_plan,
                    base_plan_version=version_number,
                    user_message_id=message_id,
                    route=route_published_plan_message(
                        user_text=f"把指定路段换成{mode_label}（路段：{command.payload.leg_id}）",
                        user_message_id=message_id,
                        turn_id=accepted.turn_id,
                        accepted_operations=(),
                    ),
                )
            if workspace.status is PlannerStatus.READY_TO_PUBLISH:
                workspace = refresh_repair_state(workspace, book, stop_reason="publication")
                await checkpoint(workspace)
                plan_version_id = UUID(server_id(accepted.turn_id, "plan-version"))
                publication_key = f"planner-plan:{accepted.turn_id}"
                published_plan = build_planner_published_plan(
                    workspace,
                    book,
                    plan_version_id=plan_version_id,
                    publication_key=publication_key,
                    based_on_state_version=accepted.base_state_version,
                    change_request=change_request,
                    clock=self.clock,
                )
            generation_mode: Literal["qwen", "fallback"] = "qwen"
            failure_code: str | None = None
            try:
                if execution_remaining(workspace, calls=True) <= 0:
                    raise TimeoutError("planner_response_budget_exhausted")
                async with asyncio.timeout(execution_remaining(workspace, calls=True)):
                    text = await self.graph.compose_response(
                        workspace, book, cancellation, change_request=change_request
                    )
            except (TimeoutError, PlannerGuardError, ModelGatewayError) as error:
                if (
                    workspace.status is not PlannerStatus.READY_TO_PUBLISH
                    or cancellation.is_cancelled
                    or (
                        isinstance(error, ModelGatewayError)
                        and error.code is ModelFailureCode.AUDIT_UNAVAILABLE
                    )
                ):
                    raise
                generation_mode = "fallback"
                failure_code = "planner_response_fallback"
                text = _formal_plan_fallback(workspace, book)
            # Detailed omissions remain in the immutable published version and
            # audit. The completion message is a concise account of actual visits.
            await barrier()
            aggregate = advance_without_operations(
                state.semantic_state, state.discovery_runtime_state
            )
            resulting_plan_version_id = (
                plan_version_id if published_plan is not None else state.current_plan_version_id
            )
            resulting_plan = published_plan if published_plan is not None else state.published_plan
            formal_plan_ready = published_plan is not None
            result = TurnResultWrite(
                state=V4TripStateEnvelope(
                    semantic_state=aggregate.semantic_state,
                    discovery_runtime_state=aggregate.runtime_state,
                    current_plan_version_id=resulting_plan_version_id,
                    published_plan=resulting_plan,
                ),
                phase=(
                    "confirmed"
                    if formal_plan_ready
                    else await self.turns.load_trip_phase(owner, trip_id)
                ),
                assistant_message=AssistantMessageWrite(
                    message_id=uuid4(),
                    generation_id=generation,
                    message_type="plan" if formal_plan_ready else "status",
                    text=text,
                    attachments=(
                        [V4Attachment(root=published_plan)] if published_plan is not None else []
                    ),
                    message_metadata={
                        "planner_workspace_revision": workspace.workspace_revision,
                        "planner_generation_id": workspace.generation_id,
                        "stage": "V4-05" if formal_plan_ready else "V4-04",
                        "formal_plan_published": formal_plan_ready,
                        **(
                            {"base_plan_version_id": str(command.payload.plan_version_id)}
                            if isinstance(command, V4PlanTransportSelectionCommand)
                            else {}
                        ),
                    },
                ),
                outbox_id=uuid4(),
                outbox_cursor=f"planner:{uuid4()}",
                publication_key=publication_key,
                generation_mode=generation_mode,
                failure_code=failure_code,
                decision_audit={
                    "workspace_revision": workspace.workspace_revision,
                    "generation_id": workspace.generation_id,
                    "status": workspace.status.value,
                    "plan_version_id": str(plan_version_id) if plan_version_id else None,
                },
                plan_version_id=plan_version_id,
                outcome=(
                    "plan_ready"
                    if formal_plan_ready
                    else "awaiting_user"
                    if workspace.status is PlannerStatus.AWAITING_USER
                    else "answered"
                ),
            )
            saved_turn = await self.turns.commit_turn_result(owner, accepted.turn_id, result)
            committed = True
            await record_execution_event(
                audit_context,
                "conversation_output_committed",
                {
                    "accepted_or_rejected": "accepted",
                    "generation_mode": generation_mode,
                    "agent_output_full": text,
                    "materialized_output": workspace.model_dump(mode="json"),
                    "attachment_refs": [str(plan_version_id)] if plan_version_id else [],
                    "authoritative_message_refs": [str(saved_turn.assistant_message_id)],
                    "outbox_ref": saved_turn.outbox_cursor,
                    "outbox_content_hash": saved_turn.content_hash,
                    "committed_state_version": saved_turn.state_version,
                    "terminal_event_ref": f"outbox:{saved_turn.outbox_id}:assistant.completed",
                    "planner_status": workspace.status.value,
                    "plan_version_id": str(plan_version_id) if plan_version_id else None,
                },
            )
            if not await self.temporary.finish_planner_execution(
                trip_id, generation, token, release_lock=False
            ):
                # The result is durable, but this worker no longer owns delivery.
                # Snapshot/outbox recovery will deliver it without disturbing a resumed turn.
                return
            await self.prepare.dispatch_committed(saved_turn.outbox_cursor, emit, replay=False)
            await record_execution_event(
                audit_context,
                "conversation_output_dispatched",
                {
                    "agent_output_full": text,
                    "outbox_ref": saved_turn.outbox_cursor,
                    "terminal_event_ref": f"outbox:{saved_turn.outbox_id}:assistant.completed",
                    "browser_terminal_status": "delivered",
                },
            )
        except asyncio.CancelledError:
            # A disconnected socket/process keeps typed checkpoints. No late result is published.
            cancellation.cancel()
            raise
        except Exception as error:
            if committed:
                raise
            cancelled = (
                isinstance(error, ModelGatewayError) and error.code is ModelFailureCode.CANCELLED
            )
            retryable = error.retryable if isinstance(error, ModelGatewayError) else True
            failure = (
                "model_audit_unavailable"
                if isinstance(error, ModelAuditError)
                else error.code.value
                if isinstance(error, ModelGatewayError)
                else error.code
                if isinstance(error, PlannerGuardError)
                else "planner_execution_timeout"
                if isinstance(error, TimeoutError)
                else "planner_execution_failed"
            )
            audit_failed = isinstance(error, ModelAuditError) or (
                isinstance(error, ModelGatewayError)
                and error.code is ModelFailureCode.AUDIT_UNAVAILABLE
            )
            if audit_context is not None and not audit_failed:
                await record_execution_event(
                    audit_context,
                    "conversation_turn_failed",
                    {
                        "accepted_or_rejected": "rejected",
                        "failure_stage": "planner_application",
                        "failure_code": failure,
                        "exception_type": type(error).__name__,
                        "failed_llm_call_id": (
                            error.audit_call_id if isinstance(error, ModelGatewayError) else None
                        ),
                    },
                )
            if accepted is not None:
                if not cancellation.is_cancelled:
                    try:
                        stopped = advance(
                            workspace,
                            status=PlannerStatus.FAILED,
                            active_interaction=workspace.active_interaction.model_copy(
                                update={"status": InteractionStatus.SUPERSEDED}
                            )
                            if workspace.active_interaction
                            else None,
                        )
                        await self._save(
                            owner,
                            accepted.turn_id,
                            state,
                            book,
                            stopped,
                            expected_revision=previous_revision,
                        )
                    except Exception:
                        pass  # A concurrent state/turn change is an expected terminal barrier.
                with suppress(TurnNotFoundError):
                    await self.turns.mark_turn_failed(
                        owner, accepted.turn_id, failure_code=failure, cancelled=cancelled
                    )
                event_data: dict[str, Any] = dict(
                    event_id=str(uuid4()),
                    trip_id=str(trip_id),
                    turn_id=str(accepted.turn_id),
                    generation_id=str(generation),
                    sequence=0,
                    emitted_at=self.clock(),
                    failure_code=failure,
                )
                event = (
                    TurnCancelledEvent(event_type="turn.cancelled", **event_data)
                    if cancelled
                    else TurnFailedEvent(
                        event_type="turn.failed",
                        retryable=retryable,
                        **event_data,
                    )
                )
                if await self.temporary.renew_trip_lock(
                    trip_id, token, 75
                ) and self._claim_terminal(accepted.turn_id):
                    await emit(event.model_dump(mode="json"))
            else:
                raise
        finally:
            if audit_token is not None:
                reset_model_audit_execution(audit_token)
            if heartbeat:
                heartbeat.cancel()
                await asyncio.gather(heartbeat, return_exceptions=True)
            if self._cancellations.get(str(generation)) is cancellation:
                self._cancellations.pop(str(generation), None)
            await self.temporary.finish_planner_execution(
                trip_id, generation, token, release_lock=True
            )

    async def _save(
        self,
        owner: UUID,
        turn_id: UUID,
        state: V4TripStateEnvelope,
        book: TaskBookV4,
        workspace: PlannerWorkspaceState,
        *,
        expected_revision: int,
    ) -> PlannerWorkspaceRecord:
        status = {
            "planning": "working",
            "awaiting_user": "awaiting_user",
            "draft_ready": "completed",
            "ready_to_publish": "completed",
            "failed": "stale",
            "stale": "stale",
            "cancelled": "stale",
        }[workspace.status.value]
        return await self.checkpoints.save_planner_workspace(
            owner,
            PlannerWorkspaceWrite(
                workspace_id=UUID(
                    server_id(workspace.generation_id, "checkpoint", workspace.workspace_revision)
                ),
                trip_id=UUID(workspace.trip_id),
                turn_id=turn_id,
                generation_id=UUID(workspace.generation_id),
                confirmed_task_book_id=UUID(book.task_book_id),
                confirmed_task_book_version=book.version,
                confirmed_task_book_hash=canonical_json_hash(book.model_dump(mode="json")),
                base_state_version=state.semantic_state.state_version,
                revision=workspace.workspace_revision,
                checkpoint_version=V4_PLANNER_CHECKPOINT_VERSION,
                payload=workspace.model_dump(mode="json"),
                status=status,
            ),
            expected_previous_revision=expected_revision,
        )

    async def _cancel(
        self, owner: UUID, trip_id: UUID, record: PlannerWorkspaceRecord, emit: PrepareEventEmitter
    ) -> None:
        generation = str(record.generation_id)
        try:
            # Marking the turn is the DB-side late-commit barrier, even on another process.
            await self.checkpoints.cancel_v4_planner_workspace(owner, trip_id, record.generation_id)
        except (TurnNotFoundError, CheckpointStaleError):
            raise PrepareJourneyError("planner_generation_not_active") from None
        cancellation = self._cancellations.get(generation)
        if cancellation:
            cancellation.cancel()
        await self.temporary.clear_active_generation_if_matches(trip_id, record.generation_id)
        if not self._claim_terminal(record.turn_id):
            return
        await emit(
            TurnCancelledEvent(
                event_id=str(uuid4()),
                event_type="turn.cancelled",
                trip_id=str(trip_id),
                turn_id=str(record.turn_id),
                generation_id=generation,
                sequence=0,
                emitted_at=self.clock(),
                failure_code="cancelled_by_user",
            ).model_dump(mode="json")
        )

    def _claim_terminal(self, turn_id: UUID) -> bool:
        """No await between membership check and insertion: cancellation races once."""
        key = str(turn_id)
        if key in self._terminal_turns:
            return False
        # Only suppress duplicate in-flight terminal delivery; durable outbox owns replay.
        if len(self._terminal_turns) >= 4096:
            self._terminal_turns.clear()
        self._terminal_turns.add(key)
        return True


def _formal_plan_fallback(workspace: PlannerWorkspaceState, book: TaskBookV4) -> str:
    """Grounded backup copy when only the optional natural-language composition fails."""

    schedule = workspace.materialized_schedule
    cost = workspace.cost_draft
    if schedule is None or cost is None:
        raise PlannerGuardError("planner_response_fallback_artifacts_missing")
    return (
        f"{book.destination_and_dates.destination_name}{len(schedule.days)}天的正式行程已生成。"
        "下方可以查看每天的游览安排、交通路线和参考预算。"
    )


def _unchanged_hotel_change_write(
    default_write: TurnResultWrite,
    *,
    previous_state: V4TripStateEnvelope,
) -> TurnResultWrite:
    """Commit an honest answer without invoking Planner when no replacement exists."""

    return default_write.model_copy(
        update={
            "state": default_write.state.model_copy(
                update={
                    "current_plan_version_id": previous_state.current_plan_version_id,
                    "published_plan": previous_state.published_plan,
                }
            ),
            "phase": "confirmed",
            "assistant_message": default_write.assistant_message.model_copy(
                update={
                    "text": (
                        "当前酒店候选中没有可替换的其他酒店，因此我没有修改这版正式行程。"
                        "你可以补充新的住宿区域、档次或预算要求，我会先刷新酒店搜索，再生成新版本。"
                    ),
                    "attachments": [],
                    "message_type": "text",
                    "message_metadata": {
                        **default_write.assistant_message.message_metadata,
                        "plan_change_applied": False,
                        "plan_change_reason": "no_alternative_hotel_offer",
                    },
                }
            ),
            "generation_mode": "fallback",
            "failure_code": None,
            "decision_audit": {
                **default_write.decision_audit,
                "result_transform": "published_plan_hotel_change_unavailable",
                "plan_change_scope": "local_replan",
                "plan_change_reason": "no_alternative_hotel_offer",
                "plan_change_applied": False,
            },
            "plan_version_id": None,
            "outcome": "answered",
        }
    )


def _unchanged_plan_change_write(
    default_write: TurnResultWrite, *, previous_state: V4TripStateEnvelope
) -> TurnResultWrite:
    """A rejected local patch is an unchanged plan, not a fresh publication."""
    unchanged = _unchanged_hotel_change_write(default_write, previous_state=previous_state)
    reason = "no_valid_local_patch"
    return unchanged.model_copy(
        update={
            "assistant_message": unchanged.assistant_message.model_copy(
                update={
                    "text": "这次修改还没找到可行的安排，原行程已保留，没有改动其他日期。",
                    "message_metadata": {
                        **unchanged.assistant_message.message_metadata,
                        "plan_change_reason": reason,
                    },
                }
            ),
            "decision_audit": {
                **unchanged.decision_audit,
                "result_transform": "published_plan_change_unavailable",
                "plan_change_reason": reason,
            },
        }
    )


def _pending_plan_change(
    workspace: PlannerWorkspaceState, state: V4TripStateEnvelope
) -> PlanChangeRequest | None:
    request = workspace.plan_change_request
    plan = state.published_plan
    if request is None or plan is None or workspace.generation_id == str(plan.generation_id):
        return None
    if request.base_plan_id != str(plan.plan_version_id):
        raise PrepareJourneyError("planner_plan_change_base_version_stale")
    return request


def _resume_workspace(
    workspace: PlannerWorkspaceState, *, optimize_timing: bool = False
) -> PlannerWorkspaceState:
    restarting = workspace.status in {PlannerStatus.CANCELLED, PlannerStatus.FAILED}
    changes: dict[str, Any] = {
        "status": PlannerStatus.PLANNING,
        "segment_attempt_count": 0 if restarting else workspace.segment_attempt_count,
        "segment_evidence_count": 0 if restarting else workspace.segment_evidence_count,
        "revision_round": 0 if restarting else workspace.revision_round,
    }
    if restarting:
        # Restoring an old checkpoint must run today's compiler and meal policy,
        # not republish stale timing/cost artifacts from before the repair.
        changes.update(
            materialized_schedule=None,
            cost_draft=None,
            validation_report=None,
            validation_observation=None,
            timing_optimization_pending=optimize_timing,
        )
        if workspace.interaction_answers and (
            workspace.interaction_answers[-1].semantic_action == "keep_task_book"
        ):
            changes["unresolved_decisions"] = ()
    hotel = workspace.hotel_observation
    if (
        workspace.status is PlannerStatus.FAILED
        and hotel is not None
        and hotel.status == "unavailable"
        and workspace.guard_observations
        and workspace.guard_observations[-1].code == "planner_required_hotel_evidence_unavailable"
    ):
        # The failed checkpoint remains in repository history. The new user
        # retry starts a fresh execution segment and may re-query a recovered
        # Provider instead of being trapped by the prior segment's loop guard.
        changes["hotel_observation"] = None
        changes["capability_observations"] = tuple(
            item
            for item in workspace.capability_observations
            if item.request_id != hotel.request_id
        )
    return advance(workspace, **changes)


def _candidate_origins(
    snapshot: ConversationSnapshotV4, book: TaskBookV4
) -> tuple[PlannerCandidateOrigin, ...]:
    selected = set(entity_intents(book))
    origins = {}
    for message in snapshot.messages:
        if message.status != "committed" or message.state_version > book.based_on_state_version:
            continue
        for attachment in message.attachments:
            if not isinstance(attachment.root, SpecificCandidateCard):
                continue
            for option in attachment.root.options:
                ref = option.entity_ref
                if ref is None or ref.canonical_entity_id not in selected:
                    continue
                for source in ref.provider_entity_refs:
                    if not source.startswith("provider:amap:"):
                        continue
                    try:
                        origin = PlannerCandidateOrigin(
                            canonical_entity_id=ref.canonical_entity_id,
                            provider_entity_id=source.removeprefix("provider:amap:"),
                            source_message_id=message.message_id,
                            source_option_id=option.option_id,
                        )
                    except ValidationError:
                        continue
                    origins[origin.canonical_entity_id] = origin
    return tuple(origins.values())


def _answer_option(
    workspace: PlannerWorkspaceState, command: V4PlannerAnswerCommand
) -> PlannerInteractionOption:
    interaction = workspace.active_interaction
    if interaction is None:
        raise PrepareJourneyError("planner_interaction_missing")
    option = next(
        (
            item
            for item in interaction.option_contracts
            if item.option_id == command.payload.option_id
        ),
        None,
    )
    if option is None:
        raise PrepareJourneyError("planner_option_not_current")
    return option


def _validate_answer(workspace: PlannerWorkspaceState, command: V4PlannerAnswerCommand) -> None:
    interaction = workspace.active_interaction
    if (
        workspace.status is not PlannerStatus.AWAITING_USER
        or interaction is None
        or interaction.status is not InteractionStatus.ACTIVE
        or interaction.interaction_id != str(command.payload.interaction_id)
        or interaction.resume_token != command.payload.resume_token
        or interaction.based_on_workspace_revision != workspace.workspace_revision
    ):
        raise PrepareJourneyError("planner_interaction_stale")
    option = _answer_option(workspace, command)
    if option.semantic_action not in {
        "keep_task_book",
        "revise_task_book",
        "supply_booking_detail",
    }:
        raise PrepareJourneyError("planner_option_action_not_supported")
    if option.semantic_action == "supply_booking_detail" and not command.payload.optional_user_text:
        raise PrepareJourneyError("planner_booking_detail_required")
    if option.semantic_action == "keep_task_book" and command.payload.optional_user_text:
        raise PrepareJourneyError("planner_keep_task_book_cannot_submit_changes")
