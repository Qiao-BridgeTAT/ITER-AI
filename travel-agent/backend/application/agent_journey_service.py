"""Production application bridge from realtime commands to the V2 Agent graph."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Callable
from datetime import UTC, date, datetime
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import Field, model_validator

from backend.agent.graph import AgentGraphRequest, AgentGraphResult, V2AgentGraph
from backend.agent.model_gateway import ModelCancellation, ModelFailureCode, ModelGatewayError
from backend.agent.planning_graph import (
    PlanningGraphError,
    PlanningGraphRequest,
    V3PlanningGraph,
)
from backend.agent.readiness_state import CriticalQuestionResolutionProof
from backend.agent.response_selection import ResponseActionKind
from backend.agent.semantic_operations import (
    DestinationOperation,
    DiningPreferenceOperation,
    ExperiencePreferenceKind,
    ExperiencePreferenceOperation,
    OperationEvidence,
    SemanticOperation,
    SemanticOperationBatch,
    SemanticOperationKind,
    SemanticTarget,
    TaskBookConfirmationOperation,
    stable_effect_json,
)
from backend.agent.semantic_understanding import SemanticInput
from backend.agent.state_merge import SemanticTripState, initial_semantic_state
from backend.agent.task_book import TaskBookError, confirm_task_book
from backend.agent.task_book_state import SemanticTaskBook
from backend.application.realtime_command_service import (
    CommandExecutionContext,
    CommandExecutionResult,
    EventDraft,
)
from backend.contracts.base import ContractModel
from backend.contracts.commands import (
    AttachmentAnswerCommand,
    AttachmentAnswerValue,
    ColdStartSubmitCommand,
    MultiChoiceAnswer,
    RecommendationFeedbackAnswer,
    TaskBookConfirmCommand,
    UserMessageCommand,
)
from backend.contracts.conversation import (
    AttachmentAnswerRecord,
    ConversationAttachment,
    ConversationAttachmentValue,
    ConversationMessage,
    RecommendationSetAttachment,
    TaskBookReferenceAttachment,
    TextMultiChoiceAttachment,
)
from backend.contracts.enums import (
    AssumptionKind,
    Confidence,
    ConfirmationStatus,
    EventType,
    EvidenceSource,
    GenerationStatus,
    GestureKind,
    TripPhase,
)
from backend.contracts.events import (
    ErrorEventPayload,
    GenerationStatusPayload,
    GestureReadyPayload,
    StreamTokenPayload,
)
from backend.contracts.itinerary import Assumption, TaskBook
from backend.contracts.plan_modification import PendingPlanModification
from backend.contracts.state import AgentCheckpointEnvelope, TripDateRange, TripState
from backend.domain.conversation_input import (
    AttachmentAnswerConflict,
    normalize_attachment_answer,
)
from backend.planning.attraction_exploration import AttractionExplorationService
from backend.planning.attraction_feedback import recommendation_feedback_operations
from backend.planning.city_registry import default_city_registry
from backend.planning.city_theme import city_theme_answer_operations
from backend.planning.dining_recommendation import (
    DiningExplorationService,
    dining_direction_answer_operations,
    restaurant_feedback_operations,
)
from backend.planning.plan_modification import PlanModificationService
from backend.planning.runtime_backend import semantic_task_book_from_state


class AttachmentAnswerSemanticBinding(ContractModel):
    """Server-only mapping from one typed answer to its already reviewed effects."""

    attachment_id: UUID
    source_message_id: UUID
    answer: AttachmentAnswerValue
    operations: tuple[SemanticOperation, ...] = Field(min_length=1, max_length=50)
    question_resolution: CriticalQuestionResolutionProof | None = None

    @model_validator(mode="after")
    def provenance_matches_attachment(self) -> AttachmentAnswerSemanticBinding:
        operation_ids = {operation.operation_id for operation in self.operations}
        for operation in self.operations:
            evidence = operation.evidence
            if (
                evidence.source is not EvidenceSource.CARD
                or evidence.source_message_id != self.source_message_id
                or evidence.source_attachment_id != self.attachment_id
            ):
                raise ValueError("attachment binding operations require matching card evidence")
        proof = self.question_resolution
        if proof is not None:
            if proof.source_message_id != self.source_message_id:
                raise ValueError("attachment question proof must use the source message")
            if not set(proof.operation_ids) <= operation_ids:
                raise ValueError("attachment question proof references an unknown operation")
        return self


class StableAgentCheckpoint(ContractModel):
    """Private validated payload stored only after a complete graph turn."""

    semantic_state: SemanticTripState
    attachment_bindings: tuple[AttachmentAnswerSemanticBinding, ...] = ()

    @model_validator(mode="after")
    def bindings_are_unique_and_owned(self) -> StableAgentCheckpoint:
        keys = [
            (binding.attachment_id, _answer_fingerprint(binding.answer))
            for binding in self.attachment_bindings
        ]
        if len(set(keys)) != len(keys):
            raise ValueError("agent attachment answer bindings must be unique")
        trip_id = self.semantic_state.trip_id
        if any(
            operation.trip_id != trip_id
            for binding in self.attachment_bindings
            for operation in binding.operations
        ):
            raise ValueError("agent attachment binding belongs to another trip")
        return self


class AgentJourneyUseCase:
    """Runs complete V2 graph turns while preserving the legacy transport envelope."""

    def __init__(
        self,
        graph: V2AgentGraph,
        *,
        attraction_exploration: AttractionExplorationService | None = None,
        dining_exploration: DiningExplorationService | None = None,
        planning_graph: V3PlanningGraph | None = None,
        plan_modification: PlanModificationService | None = None,
        timezone: str = "Asia/Shanghai",
        today: Callable[[], date] = date.today,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._graph = graph
        self._attraction_exploration = attraction_exploration
        self._dining_exploration = dining_exploration
        self._planning_graph = planning_graph
        self._plan_modification = plan_modification or PlanModificationService(clock=clock)
        self._timezone = timezone
        self._today = today
        self._clock = clock
        self._cancellations: dict[UUID, ModelCancellation] = {}
        self._cancelled_before_start: set[UUID] = set()
        self._cancellation_lock = asyncio.Lock()

    async def cancel_generation(self, generation_id: UUID) -> None:
        """Cooperatively stop the matching graph without affecting later generations."""

        async with self._cancellation_lock:
            cancellation = self._cancellations.get(generation_id)
            if cancellation is None:
                self._cancelled_before_start.add(generation_id)
            else:
                cancellation.cancel()

    @property
    def formal_planning_graph_wired(self) -> bool:
        return self._planning_graph is not None

    async def execute(self, context: CommandExecutionContext) -> CommandExecutionResult:
        command = context.command
        if isinstance(command, ColdStartSubmitCommand):
            candidate = context.state.model_dump(mode="json")
            candidate.update(
                {
                    "personal_defaults": command.payload.model_dump(mode="json"),
                    "cold_start_completed_at": self._clock().isoformat(),
                    "phase": TripPhase.CITY_SELECTION.value,
                    "state_version": context.state.state_version + 1,
                }
            )
            return CommandExecutionResult(next_state=TripState.model_validate(candidate))
        if not isinstance(
            command,
            (UserMessageCommand, AttachmentAnswerCommand, TaskBookConfirmCommand),
        ):
            return _unsupported_command(generation_id=context.generation_id)
        if context.generation_id is None:
            return _agent_failure("agent_generation_missing", "当前请求缺少生成标识，请重试。")

        cancellation = ModelCancellation()
        async with self._cancellation_lock:
            self._cancellations[context.generation_id] = cancellation
            if context.generation_id in self._cancelled_before_start:
                cancellation.cancel()
                self._cancelled_before_start.discard(context.generation_id)
        if isinstance(command, TaskBookConfirmCommand):
            try:
                return await self._execute_planning(context, cancellation)
            finally:
                await self._release_generation(context.generation_id)

        try:
            checkpoint = _restore_checkpoint(context.state)
            graph_request = _build_graph_request(
                context,
                checkpoint,
                business_date=self._today(),
                timezone=self._timezone,
            )
        except AgentJourneyInputError as exc:
            await self._release_generation(context.generation_id)
            return _agent_failure(exc.code, str(exc))

        try:
            result = await self._graph.invoke(graph_request, cancellation=cancellation)
            cancellation.raise_if_cancelled("agent_graph")
            result = await self._enrich_exploration_response(
                context,
                result,
                previous_state=checkpoint.semantic_state,
                cancellation=cancellation,
            )
            pending_modification = None
            if context.state.published_plan is not None and result.merge_result is not None:
                pending_modification = self._plan_modification.plan(
                    context.state.published_plan,
                    result.merge_result.impacts,
                    generation_id=context.generation_id,
                    source_message_id=_source_message_id(command),
                    base_state_version=context.state.state_version,
                    base_confirmed_version_id=(
                        context.state.current_plan_version_id
                        if context.state.phase.value == "confirmed"
                        else context.state.base_confirmed_version_id
                    ),
                )
            return await self._complete_turn(
                context,
                checkpoint,
                result,
                pending_modification=pending_modification,
            )
        except ModelGatewayError as error:
            return _model_failure(error)
        finally:
            await self._release_generation(context.generation_id)

    async def _execute_planning(
        self,
        context: CommandExecutionContext,
        cancellation: ModelCancellation,
    ) -> CommandExecutionResult:
        command = context.command
        assert isinstance(command, TaskBookConfirmCommand)
        generation_id = context.generation_id
        assert generation_id is not None
        if command.payload.trip_id != context.trip_id:
            return _agent_failure("task_book_trip_conflict", "任务书不属于当前旅行。")
        if self._planning_graph is None:
            return _agent_failure(
                "planning_graph_unavailable",
                "正式规划暂不可用，任务书和已有方案均已保留。",
            )
        if context.state.task_book is None:
            return _agent_failure("task_book_missing", "当前没有可以确认的旅行任务书。")

        working_payload = context.state.model_dump(mode="json")
        task_book_payload = context.state.task_book.model_dump(mode="json")
        task_book_payload["status"] = ConfirmationStatus.CONFIRMED.value
        confirmed_semantic_state: SemanticTripState | None = None
        confirmed_semantic_task_book: SemanticTaskBook | None = None
        if context.state.agent_checkpoint is not None:
            checkpoint = _restore_checkpoint(context.state)
            semantic_task_book = checkpoint.semantic_state.task_book
            if semantic_task_book is None:
                return _agent_failure(
                    "semantic_task_book_missing",
                    "Agent 任务书状态无法恢复，最近稳定方案已保留。",
                )
            source_message_id, source_attachment_id = _task_book_reference_source(
                context.state,
                semantic_task_book.task_book_id,
            )
            if source_message_id is None or source_attachment_id is None:
                return _agent_failure(
                    "task_book_reference_missing",
                    "任务书确认入口已经失效，请刷新后重试。",
                )
            confirmation = TaskBookConfirmationOperation(
                operation_id=uuid5(
                    NAMESPACE_URL,
                    f"iter:v3-task-book-confirm:{command.request_id}",
                ),
                trip_id=context.trip_id,
                operation=SemanticOperationKind.CONFIRM,
                target=SemanticTarget.TASK_BOOK_CONFIRMATION,
                evidence=OperationEvidence(
                    source=EvidenceSource.CARD,
                    source_trip_id=context.trip_id,
                    source_message_id=source_message_id,
                    source_attachment_id=source_attachment_id,
                ),
                confidence=Confidence.HIGH,
            )
            try:
                confirmed_semantic_state = confirm_task_book(
                    checkpoint.semantic_state,
                    task_book_id=semantic_task_book.task_book_id,
                    confirmation=confirmation,
                    expected_state_version=checkpoint.semantic_state.state_version,
                    business_date=self._today(),
                ).state
                confirmed_semantic_task_book = confirmed_semantic_state.task_book
            except TaskBookError:
                return _agent_failure(
                    "task_book_confirmation_failed",
                    "任务书版本已经变化，请刷新后重新确认。",
                )
        else:
            confirmed_semantic_task_book = semantic_task_book_from_state(context.state)
        working_payload.update(
            {
                "phase": TripPhase.PLANNING.value,
                "active_generation_id": str(generation_id),
                "task_book": task_book_payload,
                # The final checkpoint is written only after the complete plan is published.
                "agent_checkpoint": None,
            }
        )
        try:
            working_state = TripState.model_validate(working_payload)

            async def emit_progress(_stage: str, message: str) -> bool:
                return await context.emit_progress(
                    EventDraft(EventType.STREAM_TOKEN, StreamTokenPayload(token=message))
                )

            if confirmed_semantic_task_book is None:
                return _agent_failure(
                    "semantic_task_book_missing",
                    "Agent 任务书状态无法恢复，最近稳定方案已保留。",
                )
            result = await self._planning_graph.invoke(
                PlanningGraphRequest(
                    state=working_state,
                    generation_id=generation_id,
                    semantic_task_book=confirmed_semantic_task_book,
                ),
                cancellation=cancellation,
                emit_progress=emit_progress,
            )
        except PlanningGraphError as exc:
            return _agent_failure(
                f"planning_{exc.stage}_failed",
                _planning_failure_message(exc),
                snapshot_required=False,
            )
        published_state = result.published_state
        if confirmed_semantic_state is not None:
            published_payload = published_state.model_dump(mode="json")
            published_payload["agent_checkpoint"] = AgentCheckpointEnvelope(
                trip_id=context.trip_id,
                trip_state_version=published_state.state_version,
                payload=StableAgentCheckpoint(
                    semantic_state=confirmed_semantic_state,
                ).model_dump(mode="json"),
            ).model_dump(mode="json")
            published_state = TripState.model_validate(published_payload)
        return CommandExecutionResult(
            next_state=published_state,
            events=(
                EventDraft(
                    EventType.GENERATION_STATUS,
                    GenerationStatusPayload(status=GenerationStatus.COMPLETED),
                ),
            ),
        )

    async def _release_generation(self, generation_id: UUID) -> None:
        async with self._cancellation_lock:
            self._cancellations.pop(generation_id, None)
            self._cancelled_before_start.discard(generation_id)

    async def _enrich_exploration_response(
        self,
        context: CommandExecutionContext,
        result: AgentGraphResult,
        *,
        previous_state: SemanticTripState,
        cancellation: ModelCancellation,
    ) -> AgentGraphResult:
        generation_id = context.generation_id
        if generation_id is None:
            return result
        if result.response.action is ResponseActionKind.CRITICAL_QUESTION:
            return result
        command = context.command
        if not isinstance(command, (UserMessageCommand, AttachmentAnswerCommand)):
            return result
        source_message_id = (
            command.payload.message_id
            if isinstance(command, UserMessageCommand)
            else command.payload.source_message_id
        )
        direct_recommendation = (
            result.classification is not None
            and result.classification.primary_intent.code.value == "B2_recommendation_request"
        )
        target_domains = (
            {domain.value for domain in result.classification.primary_intent.target_domains}
            if result.classification is not None
            else set()
        )
        dining_changed = _dining_preference_fingerprint(
            previous_state
        ) != _dining_preference_fingerprint(result.state)
        answered_dining_direction = False
        answered_city_theme = False
        if isinstance(command, AttachmentAnswerCommand):
            attachment = _find_attachment(context.state, command)
            answered_city_theme = (
                isinstance(attachment, TextMultiChoiceAttachment)
                and attachment.interaction_domain == "city_theme"
            )
            answered_dining_direction = (
                isinstance(attachment, TextMultiChoiceAttachment)
                and attachment.interaction_domain == "dining_direction"
            )
        direct_dining_recommendation = direct_recommendation and bool(
            target_domains & {"dining_direction", "restaurant"}
        )
        dining_exploration = self._dining_exploration
        if dining_exploration is not None and (
            dining_changed or answered_dining_direction or direct_dining_recommendation
        ):
            recommendation = await dining_exploration.recommendation_response(
                result.state,
                request_id=command.request_id,
                source_message_id=source_message_id,
                generation_id=generation_id,
                cancellation=cancellation,
            )
            if recommendation is not None:
                return result.model_copy(update={"response": recommendation})

        exploration = self._attraction_exploration
        if exploration is None:
            return result
        if answered_city_theme or (direct_recommendation and not direct_dining_recommendation):
            recommendation = await exploration.recommendation_response(
                result.state,
                request_id=command.request_id,
                source_message_id=source_message_id,
                generation_id=generation_id,
                cancellation=cancellation,
            )
            if recommendation is not None:
                return result.model_copy(update={"response": recommendation})
        if isinstance(command, UserMessageCommand) and not _has_city_theme_attachment(
            context.state.conversation_messages,
            result.state,
        ):
            themes = exploration.city_theme_response(
                result.state,
                request_id=command.request_id,
                source_message_id=source_message_id,
                generation_id=generation_id,
            )
            if themes is not None:
                return result.model_copy(update={"response": themes})
        return result

    async def _complete_turn(
        self,
        context: CommandExecutionContext,
        checkpoint: StableAgentCheckpoint,
        result: AgentGraphResult,
        *,
        pending_modification: PendingPlanModification | None = None,
    ) -> CommandExecutionResult:
        next_version = context.state.state_version + 1
        created_at = self._clock()
        assistant_message = _assistant_message(
            context,
            result,
            next_version=next_version,
            created_at=created_at,
            pending_modification=pending_modification,
        )
        if assistant_message.text is not None and not await context.emit_progress(
            EventDraft(EventType.STREAM_TOKEN, StreamTokenPayload(token=assistant_message.text))
        ):
            return CommandExecutionResult()
        if assistant_message.attachments and not await context.emit_progress(
            EventDraft(
                EventType.GESTURE_READY,
                GestureReadyPayload(
                    gesture_id=f"agent:{assistant_message.message_id}",
                    kind=GestureKind.CONVERSATION,
                    title=assistant_message.attachments[0].root.prompt,
                    conversation_message=assistant_message,
                ),
            )
        ):
            return CommandExecutionResult()

        candidate = context.state.model_dump(mode="json")
        candidate["state_version"] = next_version
        messages = [
            ConversationMessage.model_validate(item)
            for item in candidate.get("conversation_messages", [])
        ]
        command = context.command
        if isinstance(command, UserMessageCommand):
            messages.append(
                ConversationMessage(
                    message_id=command.payload.message_id,
                    role="user",
                    text=command.payload.text,
                    created_at=created_at,
                    state_version=next_version,
                )
            )
        else:
            assert isinstance(command, AttachmentAnswerCommand)
            messages = _record_attachment_answer(
                messages,
                command,
                next_version=next_version,
                answered_at=created_at,
            )
        messages.append(assistant_message)
        candidate["conversation_messages"] = [
            message.model_dump(mode="json") for message in messages
        ]
        candidate["followup_question_count"] = min(
            3,
            result.state.readiness.critical_questions_asked,
        )
        if result.task_book_result is not None:
            semantic_task_book = result.task_book_result.task_book
            public_task_book = _public_task_book(semantic_task_book)
            day_count = (
                semantic_task_book.date_range.end_date - semantic_task_book.date_range.start_date
            ).days + 1
            candidate.update(
                {
                    "phase": TripPhase.TASK_REFLECTION.value,
                    "city": None,
                    "city_id": semantic_task_book.destination.city_id,
                    "date_range": TripDateRange(
                        start_date=semantic_task_book.date_range.start_date,
                        end_date=semantic_task_book.date_range.end_date,
                    ).model_dump(mode="json"),
                    "day_count": day_count,
                    "night_count": day_count - 1,
                    "task_book": public_task_book.model_dump(mode="json"),
                }
            )
        if pending_modification is not None:
            candidate["phase"] = "revising"
            candidate["pending_plan_modification"] = pending_modification.model_dump(mode="json")
            if context.state.phase.value == "confirmed":
                candidate["base_confirmed_version_id"] = str(context.state.current_plan_version_id)
        stable_checkpoint = StableAgentCheckpoint(
            semantic_state=result.state,
            attachment_bindings=checkpoint.attachment_bindings,
        )
        candidate["agent_checkpoint"] = AgentCheckpointEnvelope(
            trip_id=context.trip_id,
            trip_state_version=next_version,
            payload=stable_checkpoint.model_dump(mode="json"),
        ).model_dump(mode="json")
        return CommandExecutionResult(
            next_state=TripState.model_validate(candidate),
            events=(
                EventDraft(
                    EventType.GENERATION_STATUS,
                    GenerationStatusPayload(status=GenerationStatus.COMPLETED),
                ),
            ),
        )


class AgentJourneyInputError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


def _restore_checkpoint(state: TripState) -> StableAgentCheckpoint:
    envelope = state.agent_checkpoint
    if envelope is None:
        return StableAgentCheckpoint(semantic_state=initial_semantic_state(state.trip_id))
    if envelope.trip_id != state.trip_id or envelope.trip_state_version != state.state_version:
        raise AgentJourneyInputError(
            "agent_checkpoint_conflict",
            "Agent 状态与当前旅行版本不一致，请刷新后重试。",
        )
    try:
        checkpoint = StableAgentCheckpoint.model_validate(
            envelope.payload,
            context={"restore_historical_semantic_state": True},
        )
    except ValueError as exc:
        raise AgentJourneyInputError(
            "agent_checkpoint_invalid",
            "Agent 状态无法恢复，请重新开始本次对话。",
        ) from exc
    if checkpoint.semantic_state.trip_id != state.trip_id:
        raise AgentJourneyInputError(
            "agent_checkpoint_trip_conflict",
            "Agent 状态不属于当前旅行。",
        )
    return checkpoint


def _build_graph_request(
    context: CommandExecutionContext,
    checkpoint: StableAgentCheckpoint,
    *,
    business_date: date,
    timezone: str,
) -> AgentGraphRequest:
    command = context.command
    state = checkpoint.semantic_state
    if isinstance(command, UserMessageCommand):
        return AgentGraphRequest(
            initial_state=state,
            business_date=business_date,
            expected_state_version=state.state_version,
            semantic_input=SemanticInput(
                trip_id=context.trip_id,
                source_message_id=command.payload.message_id,
                user_text=command.payload.text,
                business_date=business_date,
                timezone=timezone,
                resolved_destination_candidates=tuple(
                    {
                        "city_id": city.city_id,
                        "display_name": city.display_name,
                    }
                    for city in default_city_registry().mentioned_in(command.payload.text)
                ),
                current_state_summary=_semantic_state_summary(
                    state,
                    context.state.conversation_messages,
                ),
                available_attraction_candidates=_active_attraction_candidates(
                    context.state.conversation_messages
                ),
                available_restaurant_candidates=_active_restaurant_candidates(
                    context.state.conversation_messages
                ),
            ),
        )
    if not isinstance(command, AttachmentAnswerCommand):
        raise AgentJourneyInputError("agent_command_unsupported", "当前命令不属于对话 Agent。")
    source_message = next(
        (
            message
            for message in context.state.conversation_messages
            if message.message_id == command.payload.source_message_id
        ),
        None,
    )
    if source_message is None:
        raise AgentJourneyInputError("attachment_not_found", "这条交互已经不存在，请刷新后重试。")
    try:
        normalize_attachment_answer(
            command,
            current_trip_id=context.trip_id,
            attachment_trip_id=context.state.trip_id,
            source_message=source_message,
        )
    except AttachmentAnswerConflict as exc:
        raise AgentJourneyInputError(exc.code.value, str(exc)) from exc
    attachment = next(
        (
            item.root
            for item in source_message.attachments
            if item.root.attachment_id == command.payload.attachment_id
        ),
        None,
    )
    if (
        isinstance(attachment, RecommendationSetAttachment)
        and attachment.recommendation_domain == "attraction"
        and isinstance(command.payload.answer, RecommendationFeedbackAnswer)
    ):
        if state.readiness.pending_critical_question is not None:
            raise AgentJourneyInputError(
                "recommendation_answer_does_not_resolve_question",
                "请先回答当前关键问题，再继续调整景点倾向。",
            )
        batch = recommendation_feedback_operations(
            trip_id=context.trip_id,
            request_id=command.request_id,
            source_message_id=command.payload.source_message_id,
            attachment=attachment,
            answer=command.payload.answer,
            business_date=business_date,
        )
        return AgentGraphRequest(
            initial_state=state,
            business_date=business_date,
            expected_state_version=state.state_version,
            attachment_operations=batch,
        )
    if (
        isinstance(attachment, TextMultiChoiceAttachment)
        and attachment.interaction_domain == "city_theme"
        and isinstance(command.payload.answer, MultiChoiceAnswer)
    ):
        if state.readiness.pending_critical_question is not None:
            raise AgentJourneyInputError(
                "city_theme_answer_does_not_resolve_question",
                "请先回答当前关键问题，再继续选择体验方向。",
            )
        batch = city_theme_answer_operations(
            trip_id=context.trip_id,
            request_id=command.request_id,
            source_message_id=command.payload.source_message_id,
            attachment=attachment,
            answer=command.payload.answer,
            business_date=business_date,
        )
        return AgentGraphRequest(
            initial_state=state,
            business_date=business_date,
            expected_state_version=state.state_version,
            attachment_operations=batch,
        )
    if (
        isinstance(attachment, TextMultiChoiceAttachment)
        and attachment.interaction_domain == "dining_direction"
        and isinstance(command.payload.answer, MultiChoiceAnswer)
    ):
        if state.readiness.pending_critical_question is not None:
            raise AgentJourneyInputError(
                "dining_answer_does_not_resolve_question",
                "请先回答当前关键问题，再继续调整餐饮倾向。",
            )
        batch = dining_direction_answer_operations(
            trip_id=context.trip_id,
            request_id=command.request_id,
            source_message_id=command.payload.source_message_id,
            attachment=attachment,
            answer=command.payload.answer,
            business_date=business_date,
        )
        return AgentGraphRequest(
            initial_state=state,
            business_date=business_date,
            expected_state_version=state.state_version,
            attachment_operations=batch,
        )
    if (
        isinstance(attachment, RecommendationSetAttachment)
        and attachment.recommendation_domain == "restaurant"
        and isinstance(command.payload.answer, RecommendationFeedbackAnswer)
    ):
        if state.readiness.pending_critical_question is not None:
            raise AgentJourneyInputError(
                "restaurant_answer_does_not_resolve_question",
                "请先回答当前关键问题，再继续调整餐厅倾向。",
            )
        batch = restaurant_feedback_operations(
            trip_id=context.trip_id,
            request_id=command.request_id,
            source_message_id=command.payload.source_message_id,
            attachment=attachment,
            answer=command.payload.answer,
            business_date=business_date,
        )
        return AgentGraphRequest(
            initial_state=state,
            business_date=business_date,
            expected_state_version=state.state_version,
            attachment_operations=batch,
        )

    answer_fingerprint = _answer_fingerprint(command.payload.answer)
    binding = next(
        (
            item
            for item in checkpoint.attachment_bindings
            if item.attachment_id == command.payload.attachment_id
            and item.source_message_id == command.payload.source_message_id
            and _answer_fingerprint(item.answer) == answer_fingerprint
        ),
        None,
    )
    if binding is None:
        raise AgentJourneyInputError(
            "attachment_semantic_binding_missing",
            "这项选择暂时无法映射到旅行需求，请直接用文字告诉我。",
        )
    batch = SemanticOperationBatch.model_validate(
        {
            "trip_id": str(context.trip_id),
            "operations": [operation.model_dump(mode="json") for operation in binding.operations],
        },
        context={"today": business_date},
    )
    return AgentGraphRequest(
        initial_state=state,
        business_date=business_date,
        expected_state_version=state.state_version,
        attachment_operations=batch,
        attachment_question_resolution=binding.question_resolution,
    )


def _find_attachment(
    state: TripState,
    command: AttachmentAnswerCommand,
) -> ConversationAttachmentValue | None:
    source = next(
        (
            message
            for message in state.conversation_messages
            if message.message_id == command.payload.source_message_id
        ),
        None,
    )
    if source is None:
        return None
    return next(
        (
            item.root
            for item in source.attachments
            if item.root.attachment_id == command.payload.attachment_id
        ),
        None,
    )


def _has_city_theme_attachment(
    messages: list[ConversationMessage],
    state: SemanticTripState,
) -> bool:
    destination_entry = next(
        (
            entry
            for entry in state.entries
            if isinstance(entry.operation, DestinationOperation)
            and entry.operation.value is not None
        ),
        None,
    )
    if destination_entry is None:
        return False
    if any(
        isinstance(entry.operation, ExperiencePreferenceOperation)
        and entry.operation.value.kind is ExperiencePreferenceKind.CITY_THEME
        for entry in state.entries
    ):
        return True

    destination_operation = destination_entry.operation
    assert isinstance(destination_operation, DestinationOperation)
    destination = destination_operation.value
    assert destination is not None
    destination_message_id = destination_operation.evidence.source_message_id
    destination_message_index = next(
        (
            index
            for index, message in enumerate(messages)
            if message.message_id == destination_message_id
        ),
        None,
    )
    relevant_messages = (
        messages[destination_message_index + 1 :]
        if destination_message_index is not None
        else messages
    )
    return any(
        isinstance(item.root, TextMultiChoiceAttachment)
        and item.root.interaction_domain == "city_theme"
        and item.root.context_label == destination.display_name
        for message in relevant_messages
        for item in message.attachments
    )


def _dining_preference_fingerprint(state: SemanticTripState) -> tuple[str, ...]:
    return tuple(
        sorted(
            stable_effect_json(entry.operation)
            for entry in state.entries
            if isinstance(entry.operation, DiningPreferenceOperation)
        )
    )


def _assistant_message(
    context: CommandExecutionContext,
    result: AgentGraphResult,
    *,
    next_version: int,
    created_at: datetime,
    pending_modification: PendingPlanModification | None = None,
) -> ConversationMessage:
    generation_id = context.generation_id
    assert generation_id is not None
    message_id = uuid5(NAMESPACE_URL, f"iter:v2-agent:{context.command.request_id}")
    text: str | None = None
    attachments: list[ConversationAttachment] = []
    response = result.response
    if response.action is ResponseActionKind.ORDINARY_REPLY:
        text = response.message
    elif response.action is ResponseActionKind.CRITICAL_QUESTION:
        assert response.question is not None
        text = _join_copy(response.acknowledgement, response.question.prompt)
    elif response.action is ResponseActionKind.STRUCTURED_ATTACHMENT:
        assert response.attachment is not None
        text = response.acknowledgement
        attachments = [
            _rebind_attachment(
                response.attachment,
                message_id=message_id,
                generation_id=generation_id,
                next_version=next_version,
                created_at=created_at,
            )
        ]
    elif response.action is ResponseActionKind.CANDIDATE_RECOMMENDATION:
        assert response.recommendation is not None
        attachments = [
            _rebind_attachment(
                ConversationAttachment(root=response.recommendation),
                message_id=message_id,
                generation_id=generation_id,
                next_version=next_version,
                created_at=created_at,
            )
        ]
    else:
        task_book = result.state.task_book
        if task_book is None:
            raise AgentJourneyInputError(
                "agent_task_book_missing",
                "旅行任务书尚未形成，请继续补充信息。",
            )
        text = "旅行任务书已经整理好了，你可以先查看再确认。"
        attachments = [
            ConversationAttachment(
                root=TaskBookReferenceAttachment(
                    kind="task_book_reference",
                    attachment_id=uuid5(
                        NAMESPACE_URL,
                        f"iter:v2-agent:task-book-attachment:{task_book.task_book_id}",
                    ),
                    source_message_id=message_id,
                    created_at=created_at,
                    state_version=next_version,
                    generation_id=generation_id,
                    prompt="查看并确认这次旅行任务书",
                    task_book_id=task_book.task_book_id,
                    label="旅行任务书",
                    editable=False,
                )
            )
        ]
    if pending_modification is not None:
        modification_copy = " ".join(
            (
                pending_modification.changed_summary,
                pending_modification.preserved_summary,
                pending_modification.expanded_scope_reason or pending_modification.scope_reason,
            )
        )
        text = f"{modification_copy} {text}" if text else modification_copy
    return ConversationMessage(
        message_id=message_id,
        role="assistant",
        text=text,
        attachments=attachments,
        created_at=created_at,
        state_version=next_version,
        generation_id=generation_id,
        external_facts=(
            list(response.recommendation_external_facts)
            if response.action is ResponseActionKind.CANDIDATE_RECOMMENDATION
            else (
                list(response.attachment_external_facts)
                if response.action is ResponseActionKind.STRUCTURED_ATTACHMENT
                else []
            )
        ),
    )


def _source_message_id(command: UserMessageCommand | AttachmentAnswerCommand) -> UUID:
    if isinstance(command, UserMessageCommand):
        return command.payload.message_id
    return command.payload.source_message_id


def _task_book_reference_source(
    state: TripState,
    task_book_id: UUID,
) -> tuple[UUID | None, UUID | None]:
    for message in reversed(state.conversation_messages):
        for attachment in message.attachments:
            value = attachment.root
            if (
                isinstance(value, TaskBookReferenceAttachment)
                and value.task_book_id == task_book_id
            ):
                return message.message_id, value.attachment_id
    return None, None


def _public_task_book(semantic_task_book: SemanticTaskBook) -> TaskBook:
    preference_summary = "；".join(item.summary for item in semantic_task_book.preferences)
    return TaskBook(
        city_id=semantic_task_book.destination.city_id,
        start_date=semantic_task_book.date_range.start_date,
        end_date=semantic_task_book.date_range.end_date,
        preference_summary=preference_summary or "按本次已确认的信息规划",
        strong_attraction_ids=[
            item.place_id
            for item in semantic_task_book.attraction_intents
            if item.intent.value in {"must", "want"}
        ],
        important_restaurant_ids=[
            item.place_id
            for item in semantic_task_book.important_restaurants
            if item.place_id is not None
        ],
        key_constraints=[item.description for item in semantic_task_book.key_constraints],
        tradeoffs=list(semantic_task_book.tradeoffs),
        omitted_strong_desires=list(semantic_task_book.omitted_strong_desires),
        assumptions=[
            Assumption(
                assumption_id=str(item.assumption_id),
                kind=AssumptionKind.SYSTEM_DEFAULT,
                description=item.description,
                source=EvidenceSource.AGENT_INFERENCE,
            )
            for item in semantic_task_book.assumptions
        ],
        status=ConfirmationStatus.PENDING,
    )


def _rebind_attachment(
    attachment: ConversationAttachment,
    *,
    message_id: UUID,
    generation_id: UUID,
    next_version: int,
    created_at: datetime,
) -> ConversationAttachment:
    value = attachment.root.model_copy(
        update={
            "source_message_id": message_id,
            "created_at": created_at,
            "state_version": next_version,
            "generation_id": generation_id,
            "status": "active",
        },
        deep=True,
    )
    return ConversationAttachment(root=value)


def _record_attachment_answer(
    messages: list[ConversationMessage],
    command: AttachmentAnswerCommand,
    *,
    next_version: int,
    answered_at: datetime,
) -> list[ConversationMessage]:
    updated: list[ConversationMessage] = []
    for message in messages:
        if message.message_id != command.payload.source_message_id:
            updated.append(message)
            continue
        attachments = [
            ConversationAttachment(
                root=attachment.root.model_copy(
                    update={
                        "status": (
                            "completed"
                            if attachment.root.attachment_id == command.payload.attachment_id
                            else attachment.root.status
                        ),
                        "state_version": next_version,
                    },
                    deep=True,
                )
            )
            for attachment in message.attachments
        ]
        answers = [
            answer.model_copy(update={"state_version": next_version}, deep=True)
            for answer in message.attachment_answers
            if answer.attachment_id != command.payload.attachment_id
        ]
        answers.append(
            AttachmentAnswerRecord(
                attachment_id=command.payload.attachment_id,
                answer=command.payload.answer,
                answered_at=answered_at,
                state_version=next_version,
            )
        )
        updated.append(
            message.model_copy(
                update={
                    "attachments": attachments,
                    "attachment_answers": answers,
                    "state_version": next_version,
                },
                deep=True,
            )
        )
    return updated


def _semantic_state_summary(
    state: SemanticTripState,
    messages: list[ConversationMessage],
) -> str:
    summary = {
        "state_version": state.state_version,
        "entries": [
            {
                "target": entry.operation.target.value,
                "value": getattr(entry.operation, "value", None),
                "impact_scope": entry.operation.impact_scope.model_dump(mode="json"),
            }
            for entry in state.entries
        ],
        "conflicts": [conflict.reason for conflict in state.conflicts],
        "pending_question": (
            state.readiness.pending_critical_question.resolution_goal
            if state.readiness.pending_critical_question is not None
            else None
        ),
        "available_attraction_candidates": [
            {"place_id": candidate["place_id"], "name": candidate["name"]}
            for candidate in _active_attraction_candidates(messages)
        ],
        "available_restaurant_candidates": [
            {"place_id": candidate["place_id"], "name": candidate["name"]}
            for candidate in _active_restaurant_candidates(messages)
        ],
    }
    return json.dumps(summary, ensure_ascii=False, sort_keys=True, default=str)


def _active_attraction_candidates(
    messages: list[ConversationMessage],
) -> tuple[dict[str, str], ...]:
    for message in reversed(messages):
        for attachment in reversed(message.attachments):
            value = attachment.root
            if (
                isinstance(value, RecommendationSetAttachment)
                and value.recommendation_domain == "attraction"
                and value.status != "superseded"
            ):
                return tuple(
                    {"place_id": str(item.place_id), "name": item.title}
                    for item in value.items
                    if item.place_id is not None
                )
    return ()


def _active_restaurant_candidates(
    messages: list[ConversationMessage],
) -> tuple[dict[str, str], ...]:
    for message in reversed(messages):
        for attachment in reversed(message.attachments):
            value = attachment.root
            if (
                isinstance(value, RecommendationSetAttachment)
                and value.recommendation_domain == "restaurant"
                and value.status != "superseded"
            ):
                return tuple(
                    {"place_id": str(item.place_id), "name": item.title}
                    for item in value.items
                    if item.place_id is not None
                )
    return ()


def _answer_fingerprint(answer: AttachmentAnswerValue) -> str:
    payload = json.dumps(answer.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _join_copy(first: str | None, second: str) -> str:
    return second if first is None else f"{first} {second}"


def _unsupported_command(*, generation_id: UUID | None) -> CommandExecutionResult:
    result = _agent_failure("agent_command_unsupported", "当前命令尚未接入对话 Agent。")
    if generation_id is not None:
        return result
    return CommandExecutionResult(events=result.events[:1])


def _agent_failure(
    code: str,
    message: str,
    *,
    snapshot_required: bool = True,
    retryable: bool = False,
) -> CommandExecutionResult:
    return CommandExecutionResult(
        events=(
            EventDraft(
                EventType.ERROR,
                ErrorEventPayload(
                    code=code,
                    message=message,
                    retryable=retryable,
                    snapshot_required=snapshot_required,
                ),
            ),
            EventDraft(
                EventType.GENERATION_STATUS,
                GenerationStatusPayload(status=GenerationStatus.FAILED),
            ),
        )
    )


def _model_failure(error: ModelGatewayError) -> CommandExecutionResult:
    messages = {
        ModelFailureCode.TIMEOUT: "模型这次没有及时完成理解，内容尚未写入旅行，请重试一次。",
        ModelFailureCode.RATE_LIMITED: "模型服务当前请求较多，内容尚未写入旅行，请稍后重试。",
        ModelFailureCode.UNAVAILABLE: "模型服务暂时不可用，内容尚未写入旅行，请稍后重试。",
        ModelFailureCode.UPSTREAM_ERROR: "模型服务返回了异常结果，内容尚未写入旅行，请重试一次。",
        ModelFailureCode.MALFORMED_RESPONSE: (
            "模型没有把这段话可靠整理成旅行需求，内容尚未写入，请重新发送一次。"
        ),
        ModelFailureCode.AUTHENTICATION_FAILED: "模型服务配置无效，暂时无法继续对话。",
        ModelFailureCode.PERMISSION_DENIED: "当前模型没有调用权限，暂时无法继续对话。",
        ModelFailureCode.DISABLED: "模型服务尚未启用，暂时无法继续对话。",
        ModelFailureCode.CANCELLED: "这次回复已经停止。",
    }
    return _agent_failure(
        f"model_{error.code.value}",
        messages[error.code],
        snapshot_required=False,
        retryable=error.retryable,
    )


def _planning_failure_message(error: PlanningGraphError) -> str:
    messages = {
        "candidate_recall": "".join(
            ("没有取得足够的真实同城候选，正式方案尚未生成。", "任务书和最近稳定方案已保留。")
        ),
        "daily_scheduling": "".join(
            (
                "现有地点、营业时间和用餐安排暂时无法组成合理日程，正式方案尚未生成。",
                "任务书和最近稳定方案已保留。",
            )
        ),
        "itinerary_repair": "".join(
            (
                "规划仍存在无法安全修复的时间、营业或路线冲突，正式方案尚未生成。",
                "任务书和最近稳定方案已保留。",
            )
        ),
        "plan_publication": "最终方案在发布前校验失败，没有覆盖最近稳定方案。",
    }
    return messages.get(
        error.stage,
        "这次规划没有完成，任务书和最近稳定方案已保留。",
    )
