"""WebSocket command application boundary with generation isolation."""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any, Protocol
from uuid import UUID, uuid4

from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from backend.contracts.base import ContractModel
from backend.contracts.commands import (
    CancelGenerationCommand,
    ClientCommand,
    ClientCommandValue,
)
from backend.contracts.enums import CommandType, EventType, GenerationStatus, OwnerType
from backend.contracts.events import (
    ErrorEventPayload,
    GenerationStatusPayload,
    JsonPatchOperation,
    ServerEvent,
    ServerEventValue,
    StatePatchPayload,
)
from backend.contracts.state import TripState
from backend.contracts.versions import (
    CURRENT_PROTOCOL_VERSION,
    CURRENT_SCHEMA_VERSION,
    ContractVersionDisposition,
    classify_contract_versions,
)
from backend.domain.authorization import RequestActor
from backend.domain.command_policy import (
    CommandAdmissionDecision,
    CommandAdmissionOutcome,
    evaluate_command_admission,
)
from backend.persistence.redis_temporary import RedisTemporaryStore
from backend.persistence.trip_repository import (
    ConcurrentStateVersionError,
    InvalidStableWriteError,
    PlanningRunWrite,
    PlanVersionWrite,
    TripNotFoundError,
    TripRepository,
)


class RealtimeApplicationError(RuntimeError):
    """Base error for realtime state and execution boundaries."""


class RealtimeTripNotFoundError(RealtimeApplicationError):
    pass


class RealtimeStateConflictError(RealtimeApplicationError):
    pass


class RealtimeCommandUseCase(Protocol):
    async def execute(self, context: CommandExecutionContext) -> CommandExecutionResult: ...


class RealtimeEventSink(Protocol):
    async def __call__(self, emission: EventEmission) -> bool: ...


class RealtimeTemporaryStore(Protocol):
    async def get_idempotency(
        self, scope: str, owner_id: str, idempotency_key: str
    ) -> dict[str, Any] | None: ...

    async def claim_idempotency(
        self,
        scope: str,
        owner_id: str,
        idempotency_key: str,
        result: Mapping[str, Any],
        ttl_seconds: int,
    ) -> bool: ...

    async def save_idempotency_result(
        self,
        scope: str,
        owner_id: str,
        idempotency_key: str,
        result: Mapping[str, Any],
        ttl_seconds: int,
    ) -> bool: ...

    async def clear_idempotency(self, scope: str, owner_id: str, idempotency_key: str) -> None: ...

    async def acquire_trip_lock(
        self, trip_id: UUID | str, token: str, ttl_seconds: int
    ) -> bool: ...

    async def release_trip_lock(self, trip_id: UUID | str, token: str) -> bool: ...

    async def replace_active_generation(
        self,
        trip_id: UUID | str,
        generation_id: UUID | str,
        ttl_seconds: int,
        *,
        anonymous_session_id: str | None = None,
    ) -> str | None: ...

    async def get_active_generation(self, trip_id: UUID | str) -> str | None: ...

    async def clear_active_generation_if_matches(
        self, trip_id: UUID | str, generation_id: UUID | str
    ) -> bool: ...

    async def next_generation_sequence(
        self, generation_id: UUID | str, ttl_seconds: int
    ) -> int: ...


class RealtimeStateStore(Protocol):
    async def load(self, actor: RequestActor, trip_id: UUID) -> TripState: ...

    async def commit(
        self,
        actor: RequestActor,
        state: TripState,
        *,
        expected_state_version: int,
    ) -> TripState: ...


@dataclass(frozen=True)
class EventDraft:
    type: EventType
    payload: ContractModel


@dataclass(frozen=True)
class EventEmission:
    event: ServerEventValue
    require_active_generation: bool


@dataclass(frozen=True)
class CommandExecutionContext:
    actor: RequestActor
    trip_id: UUID
    command: ClientCommandValue
    state: TripState
    generation_id: UUID | None
    emit_progress: Callable[[EventDraft], Awaitable[bool]]


@dataclass(frozen=True)
class CommandExecutionResult:
    next_state: TripState | None = None
    events: tuple[EventDraft, ...] = ()


class PendingRealtimeCommandUseCase:
    """Explicit M0 boundary until deterministic and model use cases replace it."""

    async def execute(self, context: CommandExecutionContext) -> CommandExecutionResult:
        events = [
            EventDraft(
                EventType.ERROR,
                ErrorEventPayload(
                    code="command_use_case_pending",
                    message=(
                        "The command transport is ready, but this business use case is pending."
                    ),
                    retryable=False,
                ),
            )
        ]
        if context.generation_id is not None:
            events.append(
                EventDraft(
                    EventType.GENERATION_STATUS,
                    GenerationStatusPayload(
                        status=GenerationStatus.FAILED,
                        message="The generation use case is not connected yet.",
                    ),
                )
            )
        return CommandExecutionResult(events=tuple(events))


class PersistentRealtimeStateStore:
    """Owner-checked state gateway for anonymous Redis and durable Postgres trips."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        redis: RedisTemporaryStore,
        *,
        anonymous_ttl_seconds: int,
    ) -> None:
        self._trips = TripRepository(session_factory)
        self._redis = redis
        self._anonymous_ttl_seconds = anonymous_ttl_seconds

    async def load(self, actor: RequestActor, trip_id: UUID) -> TripState:
        if actor.owner_type is OwnerType.ANONYMOUS:
            raw = await self._redis.load_anonymous_state(actor.owner_id, trip_id)
            if raw is None:
                raise RealtimeTripNotFoundError("trip was not found")
            state = TripState.model_validate(raw)
            if state.owner_type is not actor.owner_type or state.owner_id != actor.owner_id:
                raise RealtimeTripNotFoundError("trip was not found")
            return state
        try:
            return await self._trips.load_user_trip(UUID(actor.owner_id), trip_id)
        except (ValueError, TripNotFoundError) as exc:
            raise RealtimeTripNotFoundError("trip was not found") from exc

    async def commit(
        self,
        actor: RequestActor,
        state: TripState,
        *,
        expected_state_version: int,
    ) -> TripState:
        if state.owner_type is not actor.owner_type:
            raise RealtimeStateConflictError("state ownership changed during command execution")
        if state.owner_id != actor.owner_id:
            raise RealtimeStateConflictError("state ownership changed during command execution")
        if state.state_version != expected_state_version + 1:
            raise RealtimeStateConflictError("state must advance by exactly one version")
        if actor.owner_type is OwnerType.ANONYMOUS:
            result = await self._redis.compare_and_set_anonymous_state(
                actor.owner_id,
                state.trip_id,
                expected_state_version=expected_state_version,
                state=state.model_dump(mode="json"),
                ttl_seconds=self._anonymous_ttl_seconds,
            )
            if result == "missing":
                raise RealtimeTripNotFoundError("trip was not found")
            if result == "conflict":
                raise RealtimeStateConflictError("trip state version no longer matches")
            return state
        try:
            plan_version = None
            confirmed_version_id = None
            if state.phase.value == "draft_ready" and state.current_plan_version_id is not None:
                plan_version = PlanVersionWrite(
                    version_id=state.current_plan_version_id,
                    status="draft",
                    parent_version_id=state.base_confirmed_version_id,
                )
            elif state.phase.value == "confirmed":
                confirmed_version_id = state.current_plan_version_id
            if state.published_plan is not None and state.phase.value == "draft_ready":
                published = state.published_plan
                publication_commit = await self._trips.publish_plan_version(
                    UUID(actor.owner_id),
                    state,
                    expected_state_version=expected_state_version,
                    plan_version=PlanVersionWrite(
                        version_id=published.plan_version_id,
                        status="draft",
                        parent_version_id=published.parent_version_id,
                        publication_key=published.publication_key,
                        generation_id=published.generation_id,
                    ),
                    planning_run=PlanningRunWrite(
                        generation_id=published.generation_id,
                        status="succeeded",
                        input_summary={
                            "state_version": published.input_state_version,
                            "publication_key": published.publication_key,
                        },
                        result_summary={
                            "state_version": state.state_version,
                            "plan_version_id": str(published.plan_version_id),
                            "validation_status": published.validation.status.value,
                        },
                        config_versions={
                            "schema": state.schema_version,
                            "schedule": published.schedule.algorithm_version,
                            "cost": published.cost_estimate.algorithm_version,
                            "validation": published.validation.algorithm_version,
                        },
                        finished_at=published.published_at,
                    ),
                )
                return publication_commit.state
            return await self._trips.commit_stable_state(
                UUID(actor.owner_id),
                state,
                expected_state_version=expected_state_version,
                plan_version=plan_version,
                confirmed_version_id=confirmed_version_id,
            )
        except TripNotFoundError as exc:
            raise RealtimeTripNotFoundError("trip was not found") from exc
        except (ConcurrentStateVersionError, InvalidStableWriteError) as exc:
            raise RealtimeStateConflictError("trip state version no longer matches") from exc


class ServerEventPublisher:
    """Creates public events and allocates generation sequences centrally."""

    def __init__(
        self,
        temporary: RealtimeTemporaryStore,
        *,
        generation_ttl_seconds: int = 3_600,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._temporary = temporary
        self._generation_ttl_seconds = generation_ttl_seconds
        self._clock = clock

    async def publish(
        self,
        *,
        trip_id: UUID,
        request_id: UUID,
        idempotency_key: str,
        generation_id: UUID | None,
        base_state_version: int,
        state_version: int,
        draft: EventDraft,
        require_active_generation: bool = True,
    ) -> ServerEventValue | None:
        if generation_id is None:
            sequence = 0
        else:
            if require_active_generation:
                active = await self._temporary.get_active_generation(trip_id)
                if active != str(generation_id):
                    return None
            sequence = await self._temporary.next_generation_sequence(
                generation_id, self._generation_ttl_seconds
            )
        event = ServerEvent.model_validate(
            {
                "protocol_version": CURRENT_PROTOCOL_VERSION,
                "schema_version": CURRENT_SCHEMA_VERSION,
                "event_id": uuid4(),
                "request_id": request_id,
                "generation_id": generation_id,
                "sequence": sequence,
                "base_state_version": base_state_version,
                "state_version": state_version,
                "idempotency_key": idempotency_key,
                "timestamp": self._clock(),
                "type": draft.type,
                "payload": draft.payload.model_dump(mode="json", by_alias=True),
            }
        )
        return event.root


class RealtimeCommandService:
    """Validates, admits, executes, persists, and publishes one client command."""

    _generation_start_commands = frozenset(
        {
            CommandType.USER_MESSAGE,
            CommandType.ATTACHMENT_ANSWER,
            CommandType.TASK_BOOK_CONFIRM,
        }
    )

    def __init__(
        self,
        state_store: RealtimeStateStore,
        temporary: RealtimeTemporaryStore,
        use_cases: RealtimeCommandUseCase,
        publisher: ServerEventPublisher,
        *,
        idempotency_ttl_seconds: int = 86_400,
        generation_ttl_seconds: int = 3_600,
        trip_lock_ttl_seconds: int = 15,
        today: Callable[[], date] = date.today,
    ) -> None:
        self._state_store = state_store
        self._temporary = temporary
        self._use_cases = use_cases
        self._publisher = publisher
        self._idempotency_ttl_seconds = idempotency_ttl_seconds
        self._generation_ttl_seconds = generation_ttl_seconds
        self._trip_lock_ttl_seconds = trip_lock_ttl_seconds
        self._today = today

    async def handle(
        self,
        actor: RequestActor,
        trip_id: UUID,
        raw_command: Any,
    ) -> list[ServerEventValue]:
        delivered: list[ServerEventValue] = []

        async def collect(emission: EventEmission) -> bool:
            delivered.append(emission.event)
            return True

        await self.handle_stream(actor, trip_id, raw_command, collect)
        return delivered

    async def handle_stream(
        self,
        actor: RequestActor,
        trip_id: UUID,
        raw_command: Any,
        emit: RealtimeEventSink,
        *,
        on_admitted: Callable[[], None] | None = None,
    ) -> list[ServerEventValue]:
        admitted = False

        def notify_admitted() -> None:
            nonlocal admitted
            if not admitted:
                admitted = True
                if on_admitted is not None:
                    on_admitted()

        try:
            state = await self._state_store.load(actor, trip_id)
        except RealtimeTripNotFoundError:
            events = await self._unbound_error(raw_command, "trip_not_found", False)
            notify_admitted()
            return await self._deliver_many(events, emit, require_active_generation=False)

        version_disposition = classify_contract_versions(
            raw_command.get("protocol_version") if isinstance(raw_command, Mapping) else None,
            raw_command.get("schema_version") if isinstance(raw_command, Mapping) else None,
        )
        if version_disposition is not ContractVersionDisposition.CURRENT:
            code = (
                "contract_upgrade_required"
                if version_disposition is ContractVersionDisposition.UPGRADE_REQUIRED
                else "contract_version_unsupported"
            )
            event = await self._error_event(
                state,
                raw_command,
                code=code,
                retryable=False,
            )
            notify_admitted()
            return await self._deliver_many([event], emit, require_active_generation=False)

        try:
            command = ClientCommand.model_validate(
                raw_command, context={"today": self._today()}
            ).root
        except ValidationError:
            event = await self._error_event(
                state,
                raw_command,
                code="command_validation_failed",
                retryable=False,
            )
            notify_admitted()
            return await self._deliver_many([event], emit, require_active_generation=False)

        scope = f"websocket-command:{trip_id}"
        fingerprint = self._fingerprint({"trip_id": trip_id, "command": raw_command})
        existing = await self._temporary.get_idempotency(
            scope, actor.owner_id, command.idempotency_key
        )
        if existing is not None:
            if existing.get("request_fingerprint") != fingerprint:
                event = await self._error_event(
                    state,
                    command,
                    code="idempotency_key_reused",
                    retryable=False,
                )
                notify_admitted()
                return await self._deliver_many([event], emit, require_active_generation=False)
            if existing.get("status") == "complete":
                events = [ServerEvent.model_validate(item).root for item in existing["events"]]
                notify_admitted()
                return await self._deliver_many_cached(events, emit)
            event = await self._error_event(
                state,
                command,
                code="command_in_progress",
                retryable=True,
            )
            notify_admitted()
            return await self._deliver_many([event], emit, require_active_generation=False)

        active_text = await self._temporary.get_active_generation(trip_id)
        active_generation_id = UUID(active_text) if active_text else None
        admission = evaluate_command_admission(
            command,
            current_state_version=state.state_version,
            seen_idempotency_keys=(),
            active_generation_id=active_generation_id,
            current_phase=state.phase,
        )
        if admission.outcome is not CommandAdmissionOutcome.ACCEPTED:
            code = {
                CommandAdmissionOutcome.STATE_CONFLICT: "state_version_conflict",
                CommandAdmissionOutcome.PHASE_CONFLICT: "command_phase_conflict",
                CommandAdmissionOutcome.GENERATION_CONFLICT: "generation_conflict",
                CommandAdmissionOutcome.DUPLICATE: "duplicate_command",
            }[admission.outcome]
            event = await self._error_event(
                state,
                command,
                code=code,
                retryable=admission.outcome is CommandAdmissionOutcome.STATE_CONFLICT,
                snapshot_required=admission.outcome is CommandAdmissionOutcome.STATE_CONFLICT,
            )
            notify_admitted()
            return await self._deliver_many([event], emit, require_active_generation=False)

        claimed = await self._temporary.claim_idempotency(
            scope,
            actor.owner_id,
            command.idempotency_key,
            {"status": "processing", "request_fingerprint": fingerprint},
            self._idempotency_ttl_seconds,
        )
        if not claimed:
            event = await self._error_event(
                state,
                command,
                code="command_in_progress",
                retryable=True,
            )
            notify_admitted()
            return await self._deliver_many([event], emit, require_active_generation=False)

        try:
            events = await self._execute_admitted(
                actor,
                trip_id,
                state,
                command,
                admission,
                emit,
                notify_admitted,
            )
        except Exception:
            await self._temporary.clear_idempotency(scope, actor.owner_id, command.idempotency_key)
            notify_admitted()
            raise
        saved = await self._temporary.save_idempotency_result(
            scope,
            actor.owner_id,
            command.idempotency_key,
            {
                "status": "complete",
                "request_fingerprint": fingerprint,
                "events": [event.model_dump(mode="json", by_alias=True) for event in events],
            },
            self._idempotency_ttl_seconds,
        )
        if not saved:
            raise RealtimeStateConflictError("idempotency record expired before completion")
        return events

    async def _execute_admitted(
        self,
        actor: RequestActor,
        trip_id: UUID,
        state: TripState,
        command: ClientCommandValue,
        admission: CommandAdmissionDecision,
        emit: RealtimeEventSink,
        notify_admitted: Callable[[], None],
    ) -> list[ServerEventValue]:
        if isinstance(command, CancelGenerationCommand):
            cancelled_generation_id = command.payload.generation_id
            async with self._trip_lock(trip_id):
                cancelled = await self._temporary.clear_active_generation_if_matches(
                    trip_id, cancelled_generation_id
                )
            if not cancelled:
                cancel_error = await self._error_event(
                    state,
                    command,
                    code="generation_conflict",
                    retryable=False,
                )
                notify_admitted()
                return await self._deliver_many(
                    [cancel_error], emit, require_active_generation=False
                )
            await self._cancel_use_case_generation(cancelled_generation_id)
            cancel_event = await self._publisher.publish(
                trip_id=trip_id,
                request_id=command.request_id,
                idempotency_key=command.idempotency_key,
                generation_id=cancelled_generation_id,
                base_state_version=state.state_version,
                state_version=state.state_version,
                draft=EventDraft(
                    EventType.GENERATION_STATUS,
                    GenerationStatusPayload(status=GenerationStatus.CANCELLED),
                ),
                require_active_generation=False,
            )
            notify_admitted()
            if cancel_event is None:
                return []
            return await self._deliver_many([cancel_event], emit, require_active_generation=False)

        generation_id: UUID | None = None
        if command.type in self._generation_start_commands:
            generation_id = uuid4()
            async with self._trip_lock(trip_id):
                previous_generation = await self._temporary.replace_active_generation(
                    trip_id,
                    generation_id,
                    self._generation_ttl_seconds,
                    anonymous_session_id=(
                        actor.owner_id if actor.owner_type is OwnerType.ANONYMOUS else None
                    ),
                )
            if previous_generation is not None:
                await self._cancel_use_case_generation(UUID(previous_generation))
        elif admission.invalidate_active_generation and admission.generation_id_to_invalidate:
            async with self._trip_lock(trip_id):
                await self._temporary.clear_active_generation_if_matches(
                    trip_id, admission.generation_id_to_invalidate
                )

        published: list[ServerEventValue] = []
        if generation_id is not None:
            started = await self._publisher.publish(
                trip_id=trip_id,
                request_id=command.request_id,
                idempotency_key=command.idempotency_key,
                generation_id=generation_id,
                base_state_version=state.state_version,
                state_version=state.state_version,
                draft=EventDraft(
                    EventType.GENERATION_STATUS,
                    GenerationStatusPayload(status=GenerationStatus.STARTED),
                ),
            )
            if started is not None:
                await self._deliver(
                    started,
                    emit,
                    published,
                    require_active_generation=True,
                )

        notify_admitted()

        async def emit_progress(draft: EventDraft) -> bool:
            if draft.type is EventType.STATE_PATCH:
                raise RealtimeStateConflictError(
                    "use cases cannot publish uncommitted state patches"
                )
            progress_event = await self._publisher.publish(
                trip_id=trip_id,
                request_id=command.request_id,
                idempotency_key=command.idempotency_key,
                generation_id=generation_id,
                base_state_version=state.state_version,
                state_version=state.state_version,
                draft=draft,
            )
            if progress_event is None:
                return False
            return await self._deliver(
                progress_event,
                emit,
                published,
                require_active_generation=generation_id is not None,
            )

        try:
            result = await self._use_cases.execute(
                CommandExecutionContext(
                    actor,
                    trip_id,
                    command,
                    state,
                    generation_id,
                    emit_progress,
                )
            )
        except Exception:
            return await self._append_execution_failure(
                published,
                state,
                command,
                generation_id,
                emit,
                code="command_execution_failed",
                retryable=True,
            )
        current_state = state
        if result.next_state is not None:
            if result.next_state.trip_id != trip_id:
                raise RealtimeStateConflictError(
                    "command use cases cannot change the trip identifier"
                )
            try:
                async with self._trip_lock(trip_id):
                    if generation_id is not None and not await self._generation_is_active(
                        trip_id, generation_id
                    ):
                        return published
                    current_state = await self._state_store.commit(
                        actor,
                        result.next_state,
                        expected_state_version=state.state_version,
                    )
                    patch_event = await self._publisher.publish(
                        trip_id=trip_id,
                        request_id=command.request_id,
                        idempotency_key=command.idempotency_key,
                        generation_id=generation_id,
                        base_state_version=state.state_version,
                        state_version=current_state.state_version,
                        draft=EventDraft(
                            EventType.STATE_PATCH,
                            StatePatchPayload(patch=_state_patch(state, current_state)),
                        ),
                    )
                    if patch_event is not None:
                        await self._deliver(
                            patch_event,
                            emit,
                            published,
                            require_active_generation=generation_id is not None,
                        )
            except RealtimeStateConflictError:
                latest_state = await self._state_store.load(actor, trip_id)
                return await self._append_execution_failure(
                    published,
                    latest_state,
                    command,
                    generation_id,
                    emit,
                    code="state_version_conflict",
                    retryable=True,
                    snapshot_required=True,
                )

        for draft in result.events:
            if draft.type is EventType.STATE_PATCH:
                raise RealtimeStateConflictError(
                    "use cases cannot publish uncommitted state patches"
                )
            generated_event = await self._publisher.publish(
                trip_id=trip_id,
                request_id=command.request_id,
                idempotency_key=command.idempotency_key,
                generation_id=generation_id,
                base_state_version=current_state.state_version,
                state_version=current_state.state_version,
                draft=draft,
            )
            if generated_event is None:
                break
            sent = await self._deliver(
                generated_event,
                emit,
                published,
                require_active_generation=generation_id is not None,
            )
            if (
                draft.type is EventType.GENERATION_STATUS
                and isinstance(draft.payload, GenerationStatusPayload)
                and draft.payload.status
                in {
                    GenerationStatus.COMPLETED,
                    GenerationStatus.CANCELLED,
                    GenerationStatus.FAILED,
                }
                and generation_id is not None
            ):
                await self._temporary.clear_active_generation_if_matches(trip_id, generation_id)
            if not sent:
                break
        return published

    async def _append_execution_failure(
        self,
        published: list[ServerEventValue],
        state: TripState,
        command: ClientCommandValue,
        generation_id: UUID | None,
        emit: RealtimeEventSink,
        *,
        code: str,
        retryable: bool,
        snapshot_required: bool = False,
    ) -> list[ServerEventValue]:
        if generation_id is not None and not await self._generation_is_active(
            state.trip_id, generation_id
        ):
            return published
        failure = await self._error_event(
            state,
            command,
            code=code,
            retryable=retryable,
            snapshot_required=snapshot_required,
        )
        await self._deliver(
            failure,
            emit,
            published,
            require_active_generation=False,
        )
        if generation_id is not None and await self._generation_is_active(
            state.trip_id, generation_id
        ):
            failed = await self._publisher.publish(
                trip_id=state.trip_id,
                request_id=command.request_id,
                idempotency_key=command.idempotency_key,
                generation_id=generation_id,
                base_state_version=state.state_version,
                state_version=state.state_version,
                draft=EventDraft(
                    EventType.GENERATION_STATUS,
                    GenerationStatusPayload(status=GenerationStatus.FAILED),
                ),
            )
            if failed is not None:
                await self._deliver(
                    failed,
                    emit,
                    published,
                    require_active_generation=True,
                )
            await self._temporary.clear_active_generation_if_matches(state.trip_id, generation_id)
        return published

    async def event_is_deliverable(
        self,
        trip_id: UUID,
        emission: EventEmission,
    ) -> bool:
        if not emission.require_active_generation or emission.event.generation_id is None:
            return True
        return await self._generation_is_active(trip_id, emission.event.generation_id)

    @staticmethod
    async def _deliver(
        event: ServerEventValue,
        emit: RealtimeEventSink,
        delivered: list[ServerEventValue],
        *,
        require_active_generation: bool,
    ) -> bool:
        sent = await emit(
            EventEmission(
                event=event,
                require_active_generation=require_active_generation,
            )
        )
        if sent:
            delivered.append(event)
        return sent

    async def _deliver_many(
        self,
        events: list[ServerEventValue],
        emit: RealtimeEventSink,
        *,
        require_active_generation: bool,
    ) -> list[ServerEventValue]:
        delivered: list[ServerEventValue] = []
        for event in events:
            await self._deliver(
                event,
                emit,
                delivered,
                require_active_generation=require_active_generation,
            )
        return delivered

    async def _deliver_many_cached(
        self,
        events: list[ServerEventValue],
        emit: RealtimeEventSink,
    ) -> list[ServerEventValue]:
        """Replay a completed command result without treating it as live generation output.

        The active-generation guard protects newly emitted work from stale workers. A cached
        idempotent result is already final and its generation has normally been cleared, so
        applying that guard here would turn a successful retry into an empty response.
        """
        delivered: list[ServerEventValue] = []
        for event in events:
            await self._deliver(
                event,
                emit,
                delivered,
                require_active_generation=False,
            )
        return delivered

    async def _generation_is_active(self, trip_id: UUID, generation_id: UUID) -> bool:
        return await self._temporary.get_active_generation(trip_id) == str(generation_id)

    async def _cancel_use_case_generation(self, generation_id: UUID) -> None:
        cancel = getattr(self._use_cases, "cancel_generation", None)
        if cancel is None:
            return
        result = cancel(generation_id)
        if inspect.isawaitable(result):
            await result

    async def _error_event(
        self,
        state: TripState,
        command_or_raw: ClientCommandValue | Any,
        *,
        code: str,
        retryable: bool,
        snapshot_required: bool = False,
    ) -> ServerEventValue:
        request_id, idempotency_key = _correlation(command_or_raw)
        event = await self._publisher.publish(
            trip_id=state.trip_id,
            request_id=request_id,
            idempotency_key=idempotency_key,
            generation_id=None,
            base_state_version=state.state_version,
            state_version=state.state_version,
            draft=EventDraft(
                EventType.ERROR,
                ErrorEventPayload(
                    code=code,
                    message=_error_message(code),
                    retryable=retryable,
                    snapshot_required=snapshot_required,
                ),
            ),
        )
        if event is None:  # pragma: no cover - non-generation events are always published
            raise AssertionError("error event was unexpectedly dropped")
        return event

    async def _unbound_error(
        self, raw_command: Any, code: str, retryable: bool
    ) -> list[ServerEventValue]:
        request_id, idempotency_key = _correlation(raw_command)
        event = await self._publisher.publish(
            trip_id=uuid4(),
            request_id=request_id,
            idempotency_key=idempotency_key,
            generation_id=None,
            base_state_version=0,
            state_version=0,
            draft=EventDraft(
                EventType.ERROR,
                ErrorEventPayload(
                    code=code,
                    message=_error_message(code),
                    retryable=retryable,
                ),
            ),
        )
        return [event] if event is not None else []

    def _trip_lock(self, trip_id: UUID) -> _TripLock:
        return _TripLock(
            self._temporary,
            trip_id,
            ttl_seconds=self._trip_lock_ttl_seconds,
        )

    @staticmethod
    def _fingerprint(value: Any) -> str:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class _TripLock:
    def __init__(
        self,
        temporary: RealtimeTemporaryStore,
        trip_id: UUID,
        *,
        ttl_seconds: int,
    ) -> None:
        self._temporary = temporary
        self._trip_id = trip_id
        self._ttl_seconds = ttl_seconds
        self._token = uuid4().hex

    async def __aenter__(self) -> None:
        acquired = await self._temporary.acquire_trip_lock(
            self._trip_id, self._token, self._ttl_seconds
        )
        if not acquired:
            raise RealtimeStateConflictError("trip is already processing another state mutation")

    async def __aexit__(self, *_args: Any) -> None:
        await self._temporary.release_trip_lock(self._trip_id, self._token)


def _state_patch(before: TripState, after: TripState) -> list[JsonPatchOperation]:
    old = before.model_dump(mode="json")
    new = after.model_dump(mode="json")
    operations: list[JsonPatchOperation] = []
    for field in sorted(old.keys() | new.keys()):
        path = "/" + field.replace("~", "~0").replace("/", "~1")
        if field not in new:
            operations.append(JsonPatchOperation(op="remove", path=path))
        elif field not in old:
            operations.append(JsonPatchOperation(op="add", path=path, value=new[field]))
        elif old[field] != new[field]:
            operations.append(JsonPatchOperation(op="replace", path=path, value=new[field]))
    if not operations:
        raise RealtimeStateConflictError("a stable state update cannot be empty")
    return operations


def _correlation(command_or_raw: ClientCommandValue | Any) -> tuple[UUID, str]:
    if isinstance(command_or_raw, Mapping):
        raw_request_id = command_or_raw.get("request_id")
        raw_key = command_or_raw.get("idempotency_key")
    else:
        raw_request_id = getattr(command_or_raw, "request_id", None)
        raw_key = getattr(command_or_raw, "idempotency_key", None)
    try:
        request_id = UUID(str(raw_request_id))
    except (TypeError, ValueError):
        request_id = uuid4()
    key = str(raw_key).strip() if raw_key is not None else ""
    if not (8 <= len(key) <= 128):
        key = f"invalid-frame:{request_id.hex}"
    return request_id, key


def _error_message(code: str) -> str:
    messages = {
        "trip_not_found": "The trip was not found for the current session.",
        "command_validation_failed": "The command did not match the public contract.",
        "contract_upgrade_required": "This client contract is obsolete; upgrade before retrying.",
        "contract_version_unsupported": "The client contract version is not supported.",
        "idempotency_key_reused": "The idempotency key belongs to another command.",
        "command_in_progress": "The same command is still being processed.",
        "state_version_conflict": "The trip changed; fetch the latest snapshot and retry.",
        "command_phase_conflict": "The command is not valid in the current trip phase.",
        "generation_conflict": "The named generation is no longer active.",
        "duplicate_command": "The command was already processed.",
        "command_execution_failed": "The command use case failed before a stable update.",
    }
    return messages.get(code, "The command could not be completed.")
