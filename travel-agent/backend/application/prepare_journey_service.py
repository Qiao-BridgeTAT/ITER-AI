"""Application boundary for one authoritative V4 Prepare Agent turn."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any, Literal, Protocol, cast
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5
from zoneinfo import ZoneInfo

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
from backend.agent.model_gateway import (
    ModelCancellation,
    ModelFailureCode,
    ModelGatewayError,
)
from backend.agent.prepare.graph import PrepareAgentGraph, PrepareGraphResult, PrepareTurnInput
from backend.agent.state_merge import initial_semantic_state
from backend.persistence.legacy_agent_checkpoint import StableAgentCheckpoint
from backend.application.v4_owner_resolver import (
    V4OwnerResolutionError,
    user_only_v4_owner,
)
from backend.contracts.state import TripState
from backend.contracts.v4.cards import (
    AttractionPreferenceCard,
    DiningPreferenceCard,
    DirectionSemanticValue,
    EntitySemanticValue,
    LodgingAreaPreferenceCard,
    LodgingClassPreferenceCard,
    SpecificCandidateCard,
)
from backend.contracts.v4.commands import (
    V4CancelGenerationCommand,
    V4CardAnswerCommand,
    V4ClientCommand,
    V4RetryInteractionCommand,
    V4TaskBookConfirmationCommand,
    V4UserMessageCommand,
)
from backend.contracts.v4.conversation import (
    AgentStatusEvent,
    ConversationHistoryPage,
    ConversationHistoryWindow,
    ConversationMessageV4,
    ConversationSnapshotV4,
    ConversationView,
    TurnAcceptedEvent,
    TurnCancelledEvent,
    TurnFailedEvent,
    V4Attachment,
)
from backend.contracts.v4.enums import (
    ConfidenceLevel,
    DiscoverySection,
    InteractionStatus,
    TaskBookStatus,
)
from backend.contracts.v4.semantic_operations import (
    ConfirmTaskBookOperation,
    ExcludeConcreteEntityOperation,
    ExcludePreferenceDirectionOperation,
    SelectConcreteEntityOperation,
    SelectPreferenceDirectionOperation,
    SemanticDomainV4,
    SemanticOperationProposal,
    SemanticTargetV4,
    SetDelegationScopeOperation,
    SetLodgingClassPreferenceOperation,
    SetNoPreferenceOperation,
    SetNotApplicableOperation,
)
from backend.contracts.v4.state import (
    DiscoveryRuntimeState,
    PendingInteraction,
    V4TripStateEnvelope,
)
from backend.discovery.cards.service import CardAttachment, PrepareCardService
from backend.discovery.tools.registry import PrepareToolExecutor
from backend.domain.authorization import RequestActor
from backend.domain.discovery.compatibility import preview_legacy_upgrade
from backend.persistence.outbox_repository import (
    OutboxLeaseConflictError,
    OutboxRepository,
    canonical_json_hash,
    pending_delivery_frames,
    stable_delivery_frames,
)
from backend.persistence.redis_temporary import RedisTemporaryStore
from backend.persistence.turn_repository import (
    AssistantMessageWrite,
    InvalidTurnWriteError,
    PendingInteractionWrite,
    SemanticOperationWrite,
    ToolObservationWrite,
    TurnNotFoundError,
    TurnRepository,
    TurnResultWrite,
    UserMessageWrite,
)


class PrepareJourneyError(RuntimeError):
    def __init__(self, code: str, *, retryable: bool = False) -> None:
        self.code = code
        self.retryable = retryable
        super().__init__(code)


class PrepareEventEmitter(Protocol):
    async def __call__(self, payload: dict[str, Any]) -> bool: ...


ToolExecutorFactory = Callable[[date], PrepareToolExecutor]
V4OwnerResolverCallable = Callable[[RequestActor, UUID], Awaitable[UUID]]


@dataclass(frozen=True, slots=True)
class _LoadedPrepareContext:
    state: V4TripStateEnvelope
    recent_conversation: tuple[ConversationMessageV4, ...]
    phase: str
    active_attachment: V4Attachment | None = None


@dataclass(frozen=True, slots=True)
class _PreparedCommand:
    user_event_kind: Literal["text", "card_answer", "retry_interaction", "task_book_confirmation"]
    user_text: str
    prevalidated_operations: tuple[SemanticOperationProposal, ...] = ()
    signed_source_refs: tuple[str, ...] = ()
    signed_entity_refs: tuple[str, ...] = ()
    answered_interaction_id: UUID | None = None
    interaction_answer_id: UUID | None = None
    required_card_section: DiscoverySection | None = None
    card_text_section: DiscoverySection | None = None


PrepareCommitCommand = (
    V4UserMessageCommand
    | V4CardAnswerCommand
    | V4RetryInteractionCommand
    | V4TaskBookConfirmationCommand
)


@dataclass(frozen=True, slots=True)
class PrepareTurnTransformContext:
    """Admitted in-flight turn made available to a stage-specific result transformer."""

    owner_id: UUID
    trip_id: UUID
    turn_id: UUID
    generation_id: UUID
    command: PrepareCommitCommand
    previous_state: V4TripStateEnvelope
    result: PrepareGraphResult
    cancellation: ModelCancellation
    emit: PrepareEventEmitter


PrepareTurnResultTransformer = Callable[
    [PrepareTurnTransformContext, TurnResultWrite],
    Awaitable[TurnResultWrite],
]


class PrepareJourneyService:
    """Authenticate, admit, run, commit, then dispatch one V4 Prepare turn."""

    def __init__(
        self,
        *,
        graph: PrepareAgentGraph,
        turns: TurnRepository,
        outbox: OutboxRepository,
        temporary: RedisTemporaryStore,
        tool_executor_factory: ToolExecutorFactory,
        card_service: PrepareCardService | None = None,
        timezone: str = "Asia/Shanghai",
        generation_ttl_seconds: int = 3_600,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        owner_resolver: V4OwnerResolverCallable = user_only_v4_owner,
        model_audit: ModelAuditRecorder | None = None,
    ) -> None:
        self._graph = graph
        self._turns = turns
        self._outbox = outbox
        self._temporary = temporary
        self._tool_executor_factory = tool_executor_factory
        self._card_service = card_service
        self._timezone = ZoneInfo(timezone)
        self._generation_ttl_seconds = generation_ttl_seconds
        self._clock = clock
        self._owner_resolver = owner_resolver
        self._model_audit = model_audit or NoopModelAuditRecorder()
        self._cancellations: dict[UUID, ModelCancellation] = {}
        self._generation_turns: dict[UUID, UUID] = {}
        self._terminal_generations: set[UUID] = set()
        self._cancellation_lock = asyncio.Lock()

    async def get_snapshot(
        self,
        actor: RequestActor,
        trip_id: UUID,
    ) -> ConversationSnapshotV4:
        owner_id = await self.resolve_owner(actor, trip_id)
        projection = await self._turns.load_latest_snapshot(owner_id, trip_id)
        active_generation_id = await self._temporary.get_active_generation(trip_id)
        if projection.kind == "v4":
            snapshot = await self._turns.load_conversation_snapshot(owner_id, trip_id)
            return snapshot.model_copy(update={"active_generation_id": active_generation_id})
        loaded = await self._load_prepare_context(owner_id, trip_id)
        return ConversationSnapshotV4(
            trip_state=loaded.state,
            messages=[],
            pending_interaction=loaded.state.discovery_runtime_state.pending_interaction,
            terminal_event=None,
            last_outbox_cursor=None,
            active_generation_id=active_generation_id,
            snapshot_at=self._clock(),
        )

    async def get_conversation_view(self, actor: RequestActor, trip_id: UUID) -> ConversationView:
        owner = await self.resolve_owner(actor, trip_id)
        projection = await self._turns.load_latest_snapshot(owner, trip_id)
        if projection.kind != "v4":
            snapshot = await self.get_snapshot(actor, trip_id)
            return ConversationView(
                snapshot=snapshot,
                history=ConversationHistoryWindow(
                    through_state_version=snapshot.trip_state.semantic_state.state_version,
                ),
            )
        view = await self._turns.load_conversation_view(owner, trip_id)
        active = await self._temporary.get_active_generation(trip_id)
        return view.model_copy(
            update={
                "snapshot": view.snapshot.model_copy(update={"active_generation_id": active}),
            }
        )

    async def get_conversation_history(
        self,
        actor: RequestActor,
        trip_id: UUID,
        *,
        before_state_version: int,
        through_state_version: int,
    ) -> ConversationHistoryPage:
        owner = await self.resolve_owner(actor, trip_id)
        return await self._turns.load_conversation_history(
            owner,
            trip_id,
            before_state_version=before_state_version,
            through_state_version=through_state_version,
        )

    async def get_conversation_message(
        self,
        actor: RequestActor,
        trip_id: UUID,
        message_id: UUID,
    ) -> ConversationMessageV4:
        owner = await self.resolve_owner(actor, trip_id)
        return await self._turns.load_conversation_message(owner, trip_id, message_id)

    async def handle_stream(
        self,
        actor: RequestActor,
        trip_id: UUID,
        raw_command: object,
        emit: PrepareEventEmitter,
        *,
        on_admitted: Callable[[], None] | None = None,
        result_transformer: PrepareTurnResultTransformer | None = None,
    ) -> None:
        result_committed = False
        try:
            command = V4ClientCommand.model_validate(raw_command).root
        except ValidationError as error:
            raise PrepareJourneyError("invalid_v4_command") from error
        owner_id = await self.resolve_owner(actor, trip_id)
        if isinstance(command, V4CancelGenerationCommand):
            if on_admitted is not None:
                on_admitted()
            await self._cancel_command(owner_id, trip_id, command, emit)
            return
        if not isinstance(
            command,
            (
                V4UserMessageCommand,
                V4CardAnswerCommand,
                V4RetryInteractionCommand,
                V4TaskBookConfirmationCommand,
            ),
        ):  # pragma: no cover
            raise PrepareJourneyError("unsupported_v4_command")

        generation_id = uuid4()
        admitted_message = _command_user_message(command)
        user_message_ref = f"message:{admitted_message.message_id}"
        request_fingerprint = canonical_json_hash(command.model_dump(mode="json"))
        accepted = await self._turns.accept_turn(
            owner_id,
            trip_id,
            request_id=command.request_id,
            idempotency_key=command.idempotency_key,
            request_fingerprint=request_fingerprint,
            base_state_version=command.expected_state_version,
            user_message=admitted_message,
            generation_id=generation_id,
        )
        effective_generation_id = accepted.generation_id or generation_id
        audit_context = ModelAuditExecutionContext(
            trace_id=str(command.request_id),
            trip_id=str(trip_id),
            turn_id=str(accepted.turn_id),
            generation_id=str(effective_generation_id),
            user_message_id=str(admitted_message.message_id),
            input_kind=command.type,
            user_input_full=command.model_dump(mode="json"),
        )
        audit_context.attach_recorder(self._model_audit)
        accepted_event = TurnAcceptedEvent(
            event_id=str(uuid4()),
            event_type="turn.accepted",
            trip_id=str(trip_id),
            turn_id=str(accepted.turn_id),
            generation_id=str(effective_generation_id),
            sequence=0,
            emitted_at=self._clock(),
            base_state_version=accepted.base_state_version,
        ).model_dump(mode="json")
        if accepted.idempotent_replay:
            if on_admitted is not None:
                on_admitted()
            await emit(accepted_event)
            if accepted.status == "committed":
                committed = await self._turns.load_committed_turn(owner_id, accepted.turn_id)
                await self.dispatch_committed(committed.outbox_cursor, emit, replay=True)
                return
            if accepted.status in {"accepted", "running"}:
                await self._emit_status(
                    trip_id,
                    accepted.turn_id,
                    effective_generation_id,
                    emit,
                    code="already_processing",
                    message="这条消息仍在处理中。",
                )
                return
            raise PrepareJourneyError("idempotent_turn_is_terminal")

        cancellation = ModelCancellation()
        audit_token = activate_model_audit_execution(audit_context)
        try:
            previous = await self._temporary.replace_active_generation(
                trip_id,
                generation_id,
                self._generation_ttl_seconds,
            )
            async with self._cancellation_lock:
                self._cancellations[generation_id] = cancellation
                self._generation_turns[generation_id] = accepted.turn_id
                if previous is not None:
                    prior = self._cancellations.get(UUID(previous))
                    if prior is not None:
                        prior.cancel()
            # Once the browser sees this generation it must already be
            # cancellable, including while its admission audit is being saved.
            if on_admitted is not None:
                on_admitted()
            await emit(accepted_event)
            await record_execution_event(
                audit_context,
                "conversation_input_admitted",
                {
                    "base_state_version": accepted.base_state_version,
                    "request_id": str(command.request_id),
                    "idempotency_key_hash": canonical_json_hash(
                        {"idempotency_key": command.idempotency_key}
                    ),
                },
            )
            await self._emit_status(
                trip_id,
                accepted.turn_id,
                generation_id,
                emit,
                code="deciding",
                message="正在理解你的这轮需求并决定下一步。",
            )
            loaded = await self._load_prepare_context(owner_id, trip_id)
            prepared_command = self._prepare_command(command, loaded)
            result = await self._graph.invoke(
                PrepareTurnInput(
                    turn_id=accepted.turn_id,
                    trip_id=trip_id,
                    generation_id=generation_id,
                    assistant_message_id=uuid4(),
                    expected_state_version=command.expected_state_version,
                    user_event_kind=prepared_command.user_event_kind,
                    user_text=prepared_command.user_text,
                    user_message_ref=user_message_ref,
                    semantic_state=loaded.state.semantic_state,
                    runtime_state=loaded.state.discovery_runtime_state,
                    recent_conversation=loaded.recent_conversation,
                    business_date=self._clock().astimezone(self._timezone).date(),
                    published_plan=loaded.state.published_plan,
                    prevalidated_operations=prepared_command.prevalidated_operations,
                    signed_source_refs=prepared_command.signed_source_refs,
                    signed_entity_refs=prepared_command.signed_entity_refs,
                    required_card_section=prepared_command.required_card_section,
                    card_text_section=prepared_command.card_text_section,
                ),
                cancellation=cancellation,
                tool_executor=self._tool_executor_factory(
                    self._clock().astimezone(self._timezone).date()
                ),
                card_service=self._card_service,
            )
            cancellation.raise_if_cancelled("prepare_commit_barrier")
            if await self._temporary.get_active_generation(trip_id) != str(generation_id):
                raise ModelGatewayError(
                    ModelFailureCode.CANCELLED,
                    "prepare_commit_barrier",
                    retryable=False,
                )
            await self._emit_status(
                trip_id,
                accepted.turn_id,
                generation_id,
                emit,
                code=("facts_checked" if result.tool_observations else "response_buffered"),
                message=(
                    "事实查询完成，正在提交完整回复。"
                    if result.tool_observations
                    else "回复已在服务端完成校验，正在提交。"
                ),
            )
            active_after = result.candidate.discovery_runtime_state.pending_interaction
            interaction_was_consumed = prepared_command.answered_interaction_id is not None and (
                active_after is None
                or active_after.interaction_id != str(prepared_command.answered_interaction_id)
            )
            write = _turn_result_write(
                result,
                phase=loaded.phase,
                previous_state=loaded.state,
                answered_interaction_id=(
                    prepared_command.answered_interaction_id if interaction_was_consumed else None
                ),
                interaction_answer_id=(
                    prepared_command.interaction_answer_id if interaction_was_consumed else None
                ),
            )
            if result_transformer is not None:
                write = await result_transformer(
                    PrepareTurnTransformContext(
                        owner_id=owner_id,
                        trip_id=trip_id,
                        turn_id=accepted.turn_id,
                        generation_id=generation_id,
                        command=command,
                        previous_state=loaded.state,
                        result=result,
                        cancellation=cancellation,
                        emit=emit,
                    ),
                    write,
                )
                cancellation.raise_if_cancelled("prepare_result_transformer")
            committed = await self._turns.commit_turn_result(
                owner_id,
                accepted.turn_id,
                write,
            )
            result_committed = True
            await record_execution_event(
                audit_context,
                "conversation_output_committed",
                {
                    "accepted_or_rejected": "accepted",
                    "generation_mode": write.generation_mode,
                    "agent_output_full": write.assistant_message.text,
                    "agent_decisions_full": [
                        item.model_dump(mode="json") for item in result.decisions
                    ],
                    "accepted_semantic_operations_full": [
                        item.proposal.model_dump(mode="json") for item in write.semantic_operations
                    ],
                    "attachment_refs": [
                        item.model_dump(mode="json") for item in write.assistant_message.attachments
                    ],
                    "authoritative_message_refs": [str(committed.assistant_message_id)],
                    "outbox_ref": committed.outbox_cursor,
                    "outbox_content_hash": committed.content_hash,
                    "committed_state_version": committed.state_version,
                    "terminal_event_ref": f"outbox:{committed.outbox_id}:assistant.completed",
                    "failure_code": write.failure_code,
                    "result_transform": write.decision_audit.get("result_transform"),
                },
            )
            # The model/tool transaction is complete once the authoritative
            # state, message, and outbox bundle commit. Delivery may still be
            # in progress, but it is no longer a cancellable generation. This
            # also makes a terminal-triggered snapshot recovery observe a
            # cleared active generation instead of resurrecting stale UI state.
            await self._temporary.clear_active_generation_if_matches(trip_id, generation_id)
            await self.dispatch_committed(committed.outbox_cursor, emit, replay=False)
            await record_execution_event(
                audit_context,
                "conversation_output_dispatched",
                {
                    "agent_output_full": write.assistant_message.text,
                    "outbox_ref": committed.outbox_cursor,
                    "terminal_event_ref": f"outbox:{committed.outbox_id}:assistant.completed",
                    "browser_terminal_status": "delivered",
                },
            )
        except asyncio.CancelledError:
            cancellation.cancel()
            await self._record_interrupted_turn(
                owner_id, accepted.turn_id, audit_context, result_committed=result_committed
            )
            raise
        except PrepareJourneyError as error:
            await record_execution_event(
                audit_context,
                "conversation_turn_failed",
                {
                    "accepted_or_rejected": "rejected",
                    "failure_stage": "prepare_application",
                    "failure_code": error.code,
                    "retryable": error.retryable,
                },
            )
            await self._mark_failed_safely(
                owner_id,
                accepted.turn_id,
                error.code,
                cancelled=False,
            )
            if await self._claim_terminal_emission(generation_id):
                await self._emit_terminal_failure(
                    trip_id,
                    accepted.turn_id,
                    generation_id,
                    emit,
                    failure_code=error.code,
                    cancelled=False,
                    retryable=error.retryable,
                )
        except ModelGatewayError as error:
            cancelled = error.code is ModelFailureCode.CANCELLED
            if error.code is not ModelFailureCode.AUDIT_UNAVAILABLE:
                await record_execution_event(
                    audit_context,
                    "conversation_turn_failed",
                    {
                        "accepted_or_rejected": "rejected",
                        "failure_stage": "model_gateway",
                        "failure_code": error.code.value,
                        "retryable": error.retryable,
                        "failed_llm_call_id": error.audit_call_id,
                    },
                )
            await self._mark_failed_safely(
                owner_id,
                accepted.turn_id,
                error.code.value,
                cancelled=cancelled,
            )
            if await self._claim_terminal_emission(generation_id):
                await self._emit_terminal_failure(
                    trip_id,
                    accepted.turn_id,
                    generation_id,
                    emit,
                    failure_code=error.code.value,
                    cancelled=cancelled,
                    retryable=error.retryable,
                )
        except Exception as error:
            if result_committed:
                # The authoritative message is already committed. Delivery is an
                # outbox concern and must never rewrite the turn as failed.
                raise
            failure_code = (
                "model_audit_unavailable"
                if isinstance(error, ModelAuditError)
                else _safe_internal_failure_code(error)
            )
            if not isinstance(error, ModelAuditError):
                await record_execution_event(
                    audit_context,
                    "conversation_turn_failed",
                    {
                        "accepted_or_rejected": "rejected",
                        "failure_stage": "prepare_internal",
                        "failure_code": failure_code,
                        "exception_type": type(error).__name__,
                    },
                )
            await self._mark_failed_safely(
                owner_id,
                accepted.turn_id,
                failure_code,
                cancelled=False,
            )
            if await self._claim_terminal_emission(generation_id):
                await self._emit_terminal_failure(
                    trip_id,
                    accepted.turn_id,
                    generation_id,
                    emit,
                    failure_code=failure_code,
                    cancelled=False,
                    retryable=True,
                )
            if isinstance(error, ModelAuditError):
                raise PrepareJourneyError("model_audit_unavailable", retryable=True) from error
            raise
        finally:
            reset_model_audit_execution(audit_token)
            await self._temporary.clear_active_generation_if_matches(trip_id, generation_id)
            async with self._cancellation_lock:
                self._cancellations.pop(generation_id, None)
                self._generation_turns.pop(generation_id, None)
                self._terminal_generations.discard(generation_id)

    async def resolve_owner(self, actor: RequestActor, trip_id: UUID) -> UUID:
        try:
            return await self._owner_resolver(actor, trip_id)
        except V4OwnerResolutionError as error:
            raise PrepareJourneyError(error.code) from error

    async def cancel_generation(self, generation_id: UUID) -> None:
        async with self._cancellation_lock:
            cancellation = self._cancellations.get(generation_id)
        if cancellation is not None:
            cancellation.cancel()

    async def _cancel_command(
        self,
        owner_id: UUID,
        trip_id: UUID,
        command: V4CancelGenerationCommand,
        emit: PrepareEventEmitter,
    ) -> None:
        generation_id = command.payload.generation_id
        cleared = await self._temporary.clear_active_generation_if_matches(trip_id, generation_id)
        if not cleared:
            raise PrepareJourneyError("generation_conflict", retryable=False)
        async with self._cancellation_lock:
            cancellation = self._cancellations.get(generation_id)
            turn_id = self._generation_turns.get(generation_id, command.request_id)
        if cancellation is not None:
            cancellation.cancel()
        if generation_id in self._generation_turns:
            await self._mark_failed_safely(
                owner_id,
                turn_id,
                "cancelled_by_user",
                cancelled=True,
            )
        if await self._claim_terminal_emission(generation_id):
            await emit(
                TurnCancelledEvent(
                    event_id=str(uuid4()),
                    event_type="turn.cancelled",
                    trip_id=str(trip_id),
                    turn_id=str(turn_id),
                    generation_id=str(generation_id),
                    sequence=0,
                    emitted_at=self._clock(),
                    failure_code="cancelled_by_user",
                ).model_dump(mode="json")
            )

    async def _claim_terminal_emission(self, generation_id: UUID) -> bool:
        async with self._cancellation_lock:
            if generation_id in self._terminal_generations:
                return False
            self._terminal_generations.add(generation_id)
            return True

    async def _load_prepare_context(
        self,
        owner_id: UUID,
        trip_id: UUID,
    ) -> _LoadedPrepareContext:
        projection = await self._turns.load_latest_snapshot(owner_id, trip_id)
        phase = await self._turns.load_trip_phase(owner_id, trip_id)
        if projection.kind == "v4":
            snapshot = await self._turns.load_conversation_snapshot(owner_id, trip_id)
            return _LoadedPrepareContext(
                state=snapshot.trip_state,
                recent_conversation=tuple(snapshot.messages[-6:]),
                phase=phase,
                active_attachment=_active_attachment(snapshot),
            )
        try:
            legacy_state = TripState.model_validate(projection.trip_snapshot)
            if legacy_state.agent_checkpoint is None:
                legacy_semantic = initial_semantic_state(trip_id)
            else:
                checkpoint = StableAgentCheckpoint.model_validate(
                    legacy_state.agent_checkpoint.payload,
                    context={"restore_historical_semantic_state": True},
                )
                legacy_semantic = checkpoint.semantic_state
            if legacy_semantic.trip_id != trip_id:
                raise ValueError("legacy semantic checkpoint belongs to another trip")
            if legacy_semantic.state_version != projection.state_version:
                legacy_semantic = legacy_semantic.model_copy(
                    update={"state_version": projection.state_version},
                    deep=True,
                )
            preview = preview_legacy_upgrade(legacy_state, legacy_semantic)
            semantic = preview.semantic_state
            if semantic.cold_start_profile_snapshot is None:
                # GET remains read-only. This snapshot is frozen with the first
                # committed V4 turn; later profile edits cannot rewrite the trip.
                profile = await self._turns.load_initial_cold_start_profile(owner_id, trip_id)
                if profile is not None:
                    semantic = semantic.model_copy(update={"cold_start_profile_snapshot": profile})
            envelope = V4TripStateEnvelope(
                semantic_state=semantic,
                discovery_runtime_state=preview.discovery_runtime_state,
            )
        except (ValidationError, ValueError) as error:
            raise PrepareJourneyError("legacy_state_upgrade_failed") from error
        return _LoadedPrepareContext(
            state=envelope,
            recent_conversation=(),
            phase=phase,
        )

    def _prepare_command(
        self,
        command: (
            V4UserMessageCommand
            | V4CardAnswerCommand
            | V4RetryInteractionCommand
            | V4TaskBookConfirmationCommand
        ),
        loaded: _LoadedPrepareContext,
    ) -> _PreparedCommand:
        if isinstance(command, V4UserMessageCommand):
            return _PreparedCommand(
                user_event_kind="text",
                user_text=command.payload.text,
            )
        if isinstance(command, V4RetryInteractionCommand):
            return _prepare_retry_interaction(command, loaded.state.discovery_runtime_state)
        if isinstance(command, V4TaskBookConfirmationCommand):
            runtime = loaded.state.discovery_runtime_state
            pending = runtime.pending_interaction
            candidate = runtime.task_book_candidate
            if (
                pending is None
                or pending.kind.value != "confirmation"
                or candidate is None
                or candidate.status is not TaskBookStatus.AWAITING_CONFIRMATION
                or candidate.task_book_id != str(command.payload.task_book_id)
                or candidate.value.version != command.payload.task_book_version
                or candidate.based_on_state_version != command.expected_state_version
            ):
                raise PrepareJourneyError("stale_task_book_confirmation")
            interaction_ref = f"interaction:{pending.interaction_id}"
            task_book_ref = f"task-book:{candidate.task_book_id}:{candidate.value.version}"
            operation = SemanticOperationProposal(
                root=ConfirmTaskBookOperation(
                    operation_type="confirm_task_book",
                    local_operation_key=str(
                        uuid5(
                            NAMESPACE_URL,
                            f"v4-confirm-task-book:{command.request_id}",
                        )
                    ),
                    target=SemanticTargetV4.TASK_BOOK_CONFIRMATION,
                    source_refs=[interaction_ref, task_book_ref],
                    confidence=ConfidenceLevel.HIGH,
                    task_book_id=candidate.task_book_id,
                    task_book_version=candidate.value.version,
                    based_on_state_version=candidate.based_on_state_version,
                    confirmed_at=self._clock(),
                )
            )
            return _PreparedCommand(
                user_event_kind="task_book_confirmation",
                user_text="我确认这份旅行任务书。请立即开始正式规划，不需要我再次发起规划。",
                prevalidated_operations=(operation,),
                signed_source_refs=(interaction_ref, task_book_ref),
                answered_interaction_id=UUID(pending.interaction_id),
                interaction_answer_id=command.request_id,
            )

        if self._card_service is None:
            raise PrepareJourneyError("v4_card_service_unavailable", retryable=True)
        pending = loaded.state.discovery_runtime_state.pending_interaction
        attachment = loaded.active_attachment
        card = attachment.root if attachment is not None else None
        if (
            pending is None
            or pending.interaction_id != str(command.payload.interaction_id)
            or not isinstance(
                card,
                (
                    AttractionPreferenceCard,
                    DiningPreferenceCard,
                    LodgingAreaPreferenceCard,
                    LodgingClassPreferenceCard,
                    SpecificCandidateCard,
                ),
            )
            or card.interaction_id != pending.interaction_id
        ):
            raise PrepareJourneyError("v4_interaction_not_active")
        if (
            self._card_service.dependency_fingerprint(
                loaded.state.semantic_state,
                card.section,
            )
            != card.dependency_fingerprint
        ):
            raise PrepareJourneyError("v4_interaction_stale")
        return _prepare_card_answer(command, pending, card)

    async def dispatch_committed(
        self,
        cursor: str,
        emit: PrepareEventEmitter,
        *,
        replay: bool,
    ) -> None:
        if replay:
            bundle = await self._outbox.load_by_cursor(cursor)
            if bundle is None:
                raise PrepareJourneyError("committed_outbox_missing")
            frames = stable_delivery_frames(bundle)
            for frame in frames:
                if not await emit(frame.payload):
                    return
            return

        worker_id = f"prepare:{uuid4()}"
        bundle = await self._outbox.claim_by_cursor(cursor, worker_id=worker_id)
        if bundle is None:
            raise PrepareJourneyError("committed_outbox_unavailable", retryable=True)
        try:
            for frame in pending_delivery_frames(bundle):
                if not await emit(frame.payload):
                    await self._outbox.mark_retry(
                        bundle.outbox_id,
                        worker_id=worker_id,
                        error_code="websocket_delivery_failed",
                        next_attempt_at=self._clock() + timedelta(seconds=5),
                        max_attempts=5,
                    )
                    return
                await self._outbox.mark_frame_delivered(
                    bundle.outbox_id,
                    worker_id=worker_id,
                    sequence=frame.sequence,
                    delivered_at=self._clock(),
                )
        except (asyncio.CancelledError, Exception):
            # Release only our lease. A committed reply remains recoverable;
            # never reset its saved text or contiguous delivery acknowledgements.
            with suppress(OutboxLeaseConflictError):
                await self._outbox.mark_retry(
                    bundle.outbox_id,
                    worker_id=worker_id,
                    error_code="websocket_delivery_interrupted",
                    next_attempt_at=self._clock(),
                    max_attempts=5,
                )
            raise

    async def _emit_status(
        self,
        trip_id: UUID,
        turn_id: UUID,
        generation_id: UUID,
        emit: PrepareEventEmitter,
        *,
        code: str,
        message: str,
    ) -> None:
        await emit(
            AgentStatusEvent(
                event_id=str(uuid4()),
                event_type="agent.status",
                trip_id=str(trip_id),
                turn_id=str(turn_id),
                generation_id=str(generation_id),
                sequence=0,
                emitted_at=self._clock(),
                status_code=code,
                message=message,
            ).model_dump(mode="json")
        )

    async def _emit_terminal_failure(
        self,
        trip_id: UUID,
        turn_id: UUID,
        generation_id: UUID,
        emit: PrepareEventEmitter,
        *,
        failure_code: str,
        cancelled: bool,
        retryable: bool,
    ) -> None:
        event: TurnCancelledEvent | TurnFailedEvent
        if cancelled:
            event = TurnCancelledEvent(
                event_id=str(uuid4()),
                event_type="turn.cancelled",
                trip_id=str(trip_id),
                turn_id=str(turn_id),
                generation_id=str(generation_id),
                sequence=0,
                emitted_at=self._clock(),
                failure_code=failure_code,
            )
        else:
            event = TurnFailedEvent(
                event_id=str(uuid4()),
                event_type="turn.failed",
                trip_id=str(trip_id),
                turn_id=str(turn_id),
                generation_id=str(generation_id),
                sequence=0,
                emitted_at=self._clock(),
                failure_code=failure_code,
                retryable=retryable,
            )
        await emit(event.model_dump(mode="json"))

    async def _record_interrupted_turn(
        self,
        owner_id: UUID,
        turn_id: UUID,
        audit_context: ModelAuditExecutionContext,
        *,
        result_committed: bool,
    ) -> None:
        if not result_committed:
            # Repository CAS never changes an already committed result, including
            # cancellation racing with the database commit acknowledgement.
            await self._mark_failed_safely(owner_id, turn_id, "model_cancelled", cancelled=True)
        await record_execution_event(
            audit_context,
            "conversation_delivery_interrupted" if result_committed else "conversation_turn_failed",
            {
                "failure_stage": "prepare_delivery" if result_committed else "prepare_application",
                "failure_code": "model_cancelled",
                "result_committed": result_committed,
                "retryable": not result_committed,
            },
        )

    async def _mark_failed_safely(
        self,
        owner_id: UUID,
        turn_id: UUID,
        failure_code: str,
        *,
        cancelled: bool,
    ) -> None:
        try:
            await self._turns.mark_turn_failed(
                owner_id,
                turn_id,
                failure_code=failure_code,
                cancelled=cancelled,
            )
        except (InvalidTurnWriteError, TurnNotFoundError):
            return


def _command_user_message(
    command: (
        V4UserMessageCommand
        | V4CardAnswerCommand
        | V4RetryInteractionCommand
        | V4TaskBookConfirmationCommand
    ),
) -> UserMessageWrite:
    if isinstance(command, V4UserMessageCommand):
        message_id = command.payload.message_id
        text = command.payload.text
    else:
        message_id = uuid5(NAMESPACE_URL, f"v4-command-message:{command.request_id}")
        if isinstance(command, V4CardAnswerCommand):
            text = command.payload.optional_user_text or "已提交卡片选择。"
        elif isinstance(command, V4RetryInteractionCommand):
            text = "请重新生成当前步骤的卡片。"
        else:
            text = "已确认旅行任务书。"
    return UserMessageWrite(
        message_id=message_id,
        client_message_id=message_id,
        text=text,
        message_metadata={
            "client_sequence": command.client_sequence,
            "request_id": str(command.request_id),
            "command_type": command.type,
        },
    )


def _safe_internal_failure_code(error: Exception) -> str:
    if isinstance(error, ValidationError):
        # Pydantic locations and error kinds are schema metadata, not user or
        # Provider values. Retaining the first one makes a failed live turn
        # diagnosable without persisting prompts, raw payloads, or exception
        # messages that could contain sensitive data.
        details = error.errors(include_url=False, include_input=False)
        if details:
            first = details[0]
            model_title = str(error.title or "validation")
            location = ".".join(str(item) for item in first.get("loc", ()))
            error_type = str(first.get("type", "validation_error"))
            safe_detail = re.sub(
                r"[^a-z0-9_.-]+",
                "_",
                f"{model_title}.{location}.{error_type}".casefold(),
            ).strip("_.-")
            if safe_detail:
                return f"prepare_contract_validation_error.{safe_detail}"[:96].rstrip("_.-")
        return "prepare_contract_validation_error"
    if isinstance(error, TypeError):
        return _safe_exception_origin("prepare_type_error", error)
    if isinstance(error, ValueError):
        return _safe_exception_origin("prepare_value_error", error)
    if isinstance(error, KeyError):
        return _safe_exception_origin("prepare_key_error", error)
    if isinstance(error, AssertionError):
        return _safe_exception_origin("prepare_assertion_error", error)
    return "prepare_internal_error"


def _safe_exception_origin(prefix: str, error: Exception) -> str:
    traceback = error.__traceback__
    while traceback is not None and traceback.tb_next is not None:
        traceback = traceback.tb_next
    function_name = traceback.tb_frame.f_code.co_name if traceback is not None else "unknown"
    safe_detail = re.sub(
        r"[^a-z0-9_.-]+",
        "_",
        f"{type(error).__name__}.{function_name}".casefold(),
    ).strip("_.-")
    return f"{prefix}.{safe_detail}"[:96].rstrip("_.-")


def _active_attachment(snapshot: ConversationSnapshotV4) -> V4Attachment | None:
    pending = snapshot.pending_interaction
    if pending is None:
        return None
    for message in reversed(snapshot.messages):
        for attachment in message.attachments:
            value = attachment.root
            if hasattr(value, "interaction_id") and value.interaction_id == pending.interaction_id:
                return attachment
            if (
                pending.kind.value == "confirmation"
                and hasattr(value, "task_book_id")
                and value.task_book_id in pending.target_ids
            ):
                return attachment
    return None


def _prepare_retry_interaction(
    command: V4RetryInteractionCommand,
    runtime: DiscoveryRuntimeState,
) -> _PreparedCommand:
    pending = runtime.pending_interaction
    if (
        pending is None
        or pending.interaction_id != str(command.payload.interaction_id)
        or pending.status is not InteractionStatus.ACTIVE
        or pending.recovery is None
        or pending.section is not runtime.current_section
        or command.expected_state_version != runtime.state_version
    ):
        raise PrepareJourneyError("v4_recovery_not_active")
    return _PreparedCommand(
        user_event_kind="retry_interaction",
        user_text="请重新生成当前步骤的卡片，保留我之前确认的需求。",
        required_card_section=pending.section,
        signed_source_refs=(f"interaction:{pending.interaction_id}",),
        answered_interaction_id=command.payload.interaction_id,
        interaction_answer_id=command.request_id,
    )


def _prepare_card_answer(
    command: V4CardAnswerCommand,
    pending: PendingInteraction,
    card: CardAttachment,
) -> _PreparedCommand:
    interaction_ref = f"interaction:{pending.interaction_id}"
    options = {item.option_id: item for item in card.options}
    operations: list[SemanticOperationProposal] = []
    signed_refs: list[str] = [interaction_ref]
    entity_refs: list[str] = []
    required_card_section: DiscoverySection | None = None
    user_text = command.payload.optional_user_text or "我已经完成这张卡片的选择。"

    for selection in command.payload.selections:
        option = options.get(selection.option_id)
        if option is None or option.option_id not in pending.option_refs:
            raise PrepareJourneyError("v4_card_option_not_issued")
        signed_refs.append(option.signed_operation_ref)
        source_refs = [interaction_ref, option.signed_operation_ref]
        semantic = option.semantic_value.root
        if isinstance(semantic, EntitySemanticValue):
            if selection.disposition not in semantic.allowed_dispositions:
                raise PrepareJourneyError("v4_card_disposition_not_allowed")
            assert option.entity_ref is not None
            entity_refs.append(semantic.canonical_entity_id)
            entity_domain: Literal[
                SemanticDomainV4.ATTRACTION,
                SemanticDomainV4.DINING,
            ] = (
                SemanticDomainV4.ATTRACTION
                if card.domain.value == "attraction"
                else SemanticDomainV4.DINING
            )
            entity_target = (
                SemanticTargetV4.ATTRACTION_ENTITY
                if entity_domain is SemanticDomainV4.ATTRACTION
                else SemanticTargetV4.DINING_ENTITY
            )
            if selection.disposition == "avoid":
                operations.append(
                    SemanticOperationProposal(
                        root=ExcludeConcreteEntityOperation(
                            operation_type="exclude_concrete_entity",
                            local_operation_key=option.signed_operation_ref,
                            target=entity_target,
                            source_refs=source_refs,
                            confidence=ConfidenceLevel.HIGH,
                            domain=entity_domain,
                            canonical_entity_id=semantic.canonical_entity_id,
                            display_name=option.label,
                        )
                    )
                )
            else:
                disposition = cast(
                    Literal["must", "want", "destination", "if_convenient", "avoid"],
                    selection.disposition,
                )
                operations.append(
                    SemanticOperationProposal(
                        root=SelectConcreteEntityOperation(
                            operation_type="select_concrete_entity",
                            local_operation_key=option.signed_operation_ref,
                            target=entity_target,
                            source_refs=source_refs,
                            confidence=ConfidenceLevel.HIGH,
                            domain=entity_domain,
                            canonical_entity_id=semantic.canonical_entity_id,
                            display_name=option.label,
                            disposition=disposition,
                        )
                    )
                )
            continue

        if not isinstance(semantic, DirectionSemanticValue):  # pragma: no cover
            raise PrepareJourneyError("v4_card_semantic_value_invalid")
        if card.section is DiscoverySection.LODGING_CLASS_PREFERENCE:
            if selection.disposition != "selected":
                raise PrepareJourneyError("v4_lodging_class_requires_selection")
            operations.append(
                SemanticOperationProposal(
                    root=SetLodgingClassPreferenceOperation(
                        operation_type="set_lodging_class_preference",
                        local_operation_key=option.signed_operation_ref,
                        target=SemanticTargetV4.LODGING_CLASS,
                        source_refs=source_refs,
                        confidence=ConfidenceLevel.HIGH,
                        domain=SemanticDomainV4.LODGING,
                        hotel_quality_tier=semantic.hotel_quality_tier,
                        property_type=semantic.property_type,
                        nightly_budget_minimum_minor=(semantic.nightly_budget_minimum_minor),
                        nightly_budget_maximum_minor=(semantic.nightly_budget_maximum_minor),
                    )
                )
            )
            continue
        if selection.disposition not in {"selected", "excluded"}:
            raise PrepareJourneyError("v4_preference_disposition_not_allowed")
        direction_domain, direction_target = _preference_domain_target(card.section)
        if selection.disposition == "selected":
            operations.append(
                SemanticOperationProposal(
                    root=SelectPreferenceDirectionOperation(
                        operation_type="select_preference_direction",
                        local_operation_key=option.signed_operation_ref,
                        target=direction_target,
                        source_refs=source_refs,
                        confidence=ConfidenceLevel.HIGH,
                        domain=direction_domain,
                        direction_id=semantic.direction_id,
                        label=option.label,
                        description=option.description,
                        tags=semantic.tags,
                        search_query=semantic.search_query,
                    )
                )
            )
        else:
            operations.append(
                SemanticOperationProposal(
                    root=ExcludePreferenceDirectionOperation(
                        operation_type="exclude_preference_direction",
                        local_operation_key=option.signed_operation_ref,
                        target=direction_target,
                        source_refs=source_refs,
                        confidence=ConfidenceLevel.HIGH,
                        domain=direction_domain,
                        direction_id=semantic.direction_id,
                        label=option.label,
                        description=option.description,
                        tags=semantic.tags,
                        search_query=semantic.search_query,
                    )
                )
            )

    if command.payload.control_action_id is not None:
        control = next(
            (
                item
                for item in card.control_actions
                if item.control_id == command.payload.control_action_id
            ),
            None,
        )
        if control is None:
            raise PrepareJourneyError("v4_card_control_not_issued")
        if control.signed_operation_ref is not None:
            signed_refs.append(control.signed_operation_ref)
        source_refs = [
            interaction_ref,
            *([control.signed_operation_ref] if control.signed_operation_ref is not None else []),
        ]
        control_domain, control_target = _control_domain_target(card.section)
        if control.kind == "no_preference":
            operations.append(
                SemanticOperationProposal(
                    root=SetNoPreferenceOperation(
                        operation_type="set_no_preference",
                        local_operation_key=control.signed_operation_ref or control.control_id,
                        target=control_target,
                        source_refs=source_refs,
                        confidence=ConfidenceLevel.HIGH,
                        domain=control_domain,
                    )
                )
            )
        elif control.kind == "delegate":
            operations.append(
                SemanticOperationProposal(
                    root=SetDelegationScopeOperation(
                        operation_type="set_delegation_scope",
                        local_operation_key=control.signed_operation_ref or control.control_id,
                        target=control_target,
                        source_refs=source_refs,
                        confidence=ConfidenceLevel.HIGH,
                        domain=control_domain,
                        delegated_targets=[card.section.value],
                        boundary_refs=[],
                    )
                )
            )
        elif control.kind == "not_applicable":
            if control_domain is not SemanticDomainV4.LODGING:
                raise PrepareJourneyError("v4_card_control_not_allowed")
            operations.append(
                SemanticOperationProposal(
                    root=SetNotApplicableOperation(
                        operation_type="set_not_applicable",
                        local_operation_key=control.signed_operation_ref or control.control_id,
                        target=control_target,
                        source_refs=source_refs,
                        confidence=ConfidenceLevel.HIGH,
                        reason="本次旅行不需要住宿。",
                    )
                )
            )
        elif control.kind == "refresh":
            required_card_section = card.section
            user_text = "请基于当前状态为这一章节换一批候选。"
        elif control.kind in {"free_text", "existing_booking"}:
            if command.payload.optional_user_text is None:
                raise PrepareJourneyError("v4_card_control_requires_text")
            user_text = command.payload.optional_user_text
        else:  # pragma: no cover - strict control union
            raise PrepareJourneyError("v4_card_control_not_supported")

    return _PreparedCommand(
        # A signed control identifies the card, not the meaning of the user's
        # free text. Preserve click validation, then let Qwen interpret the text
        # instead of using the no-new-semantics decide_after_update contract.
        user_event_kind=(
            "text"
            if command.payload.optional_user_text is not None and required_card_section is None
            else "card_answer"
        ),
        user_text=user_text,
        prevalidated_operations=tuple(operations),
        signed_source_refs=tuple(dict.fromkeys(signed_refs)),
        signed_entity_refs=tuple(dict.fromkeys(entity_refs)),
        answered_interaction_id=command.payload.interaction_id,
        interaction_answer_id=command.payload.answer_id,
        required_card_section=required_card_section,
        card_text_section=(
            card.section
            if command.payload.optional_user_text is not None and required_card_section is None
            else None
        ),
    )


def _preference_domain_target(
    section: DiscoverySection,
) -> tuple[SemanticDomainV4, SemanticTargetV4]:
    values = {
        DiscoverySection.ATTRACTION_PREFERENCE: (
            SemanticDomainV4.ATTRACTION,
            SemanticTargetV4.ATTRACTION_PREFERENCE,
        ),
        DiscoverySection.DINING_PREFERENCE: (
            SemanticDomainV4.DINING,
            SemanticTargetV4.DINING_PREFERENCE,
        ),
        DiscoverySection.LODGING_AREA_PREFERENCE: (
            SemanticDomainV4.LODGING,
            SemanticTargetV4.LODGING_AREA,
        ),
    }
    try:
        return values[section]
    except KeyError as error:
        raise PrepareJourneyError("v4_preference_section_invalid") from error


def _control_domain_target(
    section: DiscoverySection,
) -> tuple[SemanticDomainV4, SemanticTargetV4]:
    values = {
        DiscoverySection.ATTRACTION_PREFERENCE: (
            SemanticDomainV4.ATTRACTION,
            SemanticTargetV4.ATTRACTION_PREFERENCE,
        ),
        DiscoverySection.ATTRACTION_SPECIFIC: (
            SemanticDomainV4.ATTRACTION,
            SemanticTargetV4.ATTRACTION_ENTITY,
        ),
        DiscoverySection.DINING_PREFERENCE: (
            SemanticDomainV4.DINING,
            SemanticTargetV4.DINING_PREFERENCE,
        ),
        DiscoverySection.DINING_SPECIFIC: (
            SemanticDomainV4.DINING,
            SemanticTargetV4.DINING_ENTITY,
        ),
        DiscoverySection.LODGING_AREA_PREFERENCE: (
            SemanticDomainV4.LODGING,
            SemanticTargetV4.LODGING_AREA,
        ),
        DiscoverySection.LODGING_CLASS_PREFERENCE: (
            SemanticDomainV4.LODGING,
            SemanticTargetV4.LODGING_CLASS,
        ),
    }
    try:
        return values[section]
    except KeyError as error:
        raise PrepareJourneyError("v4_card_section_invalid") from error


def _turn_result_write(
    result: PrepareGraphResult,
    *,
    phase: str,
    previous_state: V4TripStateEnvelope,
    answered_interaction_id: UUID | None = None,
    interaction_answer_id: UUID | None = None,
) -> TurnResultWrite:
    candidate = result.candidate
    message_id = UUID(candidate.assistant_message.message_id)
    outbox_id = uuid5(NAMESPACE_URL, f"v4-outbox:{candidate.turn_id}")
    cursor = f"v4:{candidate.trip_id}:{candidate.candidate_state_version}:{outbox_id}"
    pending = candidate.discovery_runtime_state.pending_interaction
    confirmed_ref_unchanged = (
        candidate.semantic_state.confirmed_task_book_ref
        == previous_state.semantic_state.confirmed_task_book_ref
    )
    return TurnResultWrite(
        state=V4TripStateEnvelope(
            semantic_state=candidate.semantic_state,
            discovery_runtime_state=candidate.discovery_runtime_state,
            current_plan_version_id=(
                previous_state.current_plan_version_id if confirmed_ref_unchanged else None
            ),
            published_plan=previous_state.published_plan if confirmed_ref_unchanged else None,
        ),
        phase=phase,
        assistant_message=AssistantMessageWrite(
            message_id=message_id,
            text=candidate.assistant_message.text,
            generation_id=UUID(candidate.generation_id),
            attachments=list(result.attachments),
            message_type=(
                "task_book"
                if any(hasattr(item.root, "task_book_id") for item in result.attachments)
                else "card"
                if result.attachments
                else "text"
            ),
        ),
        outbox_id=outbox_id,
        outbox_cursor=cursor,
        publication_key=f"prepare:{candidate.turn_id}",
        generation_mode=candidate.assistant_message.generation_mode,
        semantic_operations=[
            SemanticOperationWrite(
                operation_id=item.operation_id,
                proposal=item.proposal,
            )
            for item in result.accepted_operations
        ],
        tool_observations=[
            ToolObservationWrite(
                observation_id=uuid5(
                    NAMESPACE_URL,
                    f"v4-observation:{candidate.turn_id}:{item.observation.request_id}",
                ),
                observation=item.observation,
                provider=item.provider,
                request_hash=canonical_json_hash(
                    {
                        "request_id": item.observation.request_id,
                        "capability": item.observation.capability.value,
                    }
                ),
                expires_at=max(
                    (
                        fact.expires_at
                        for fact in item.observation.facts
                        if fact.expires_at is not None
                    ),
                    default=None,
                ),
            )
            for item in result.tool_observations
        ],
        pending_interaction=(
            PendingInteractionWrite(
                interaction_id=UUID(pending.interaction_id),
                source_message_id=message_id,
                interaction=pending,
            )
            if pending is not None
            else None
        ),
        invalidated_interaction_ids=list(result.invalidated_interaction_ids),
        answered_interaction_id=answered_interaction_id,
        interaction_answer_id=interaction_answer_id,
        failure_code=result.failure_code,
        decision_audit={
            "trace": list(result.trace),
            "outcome": result.outcome,
            "decision_ids": [item.decision_id for item in result.decisions],
            "tool_rounds": sum(item == "execute_tools" for item in result.trace),
            "response_buffer_hash": canonical_json_hash({"text": candidate.assistant_message.text}),
        },
        outcome=result.outcome,  # type: ignore[arg-type]
    )


__all__ = [
    "PrepareJourneyError",
    "PrepareJourneyService",
    "PrepareTurnResultTransformer",
    "PrepareTurnTransformContext",
]
