"""V4 turn admission and one-shot authoritative result transactions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from pydantic import Field, JsonValue, ValidationError, model_validator
from sqlalchemy import case, func, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import defer

from backend.contracts.cold_start import ColdStartSubmission
from backend.contracts.v4.base import Digest, DisplayText, Identifier, V4ContractModel
from backend.contracts.v4.conversation import (
    AssistantCompletedEvent,
    ConversationEventV4,
    ConversationHistoryPage,
    ConversationHistoryWindow,
    ConversationMessageV4,
    ConversationSnapshotV4,
    ConversationView,
    V4Attachment,
)
from backend.contracts.v4.planner_publication import PlannerPublishedPlan
from backend.contracts.v4.prepare import ToolObservation
from backend.contracts.v4.semantic_operations import SemanticOperationProposal
from backend.contracts.v4.state import (
    ColdStartProfileSnapshot,
    PendingInteraction,
    V4TripStateEnvelope,
)
from backend.contracts.versions import V4_PROTOCOL_VERSION, V4_SCHEMA_VERSION
from backend.persistence.legacy_snapshot_adapter import (
    PersistedTripProjection,
    SnapshotCompatibilityError,
    decode_persisted_snapshot,
)
from backend.persistence.models import (
    AgentTurn,
    DiscoveryRuntimeStateVersion,
    Message,
    OutboxEvent,
    PendingInteractionRecord,
    SemanticOperationRecord,
    ToolObservationRecord,
    Trip,
    TripSemanticStateVersion,
    TripSnapshot,
    TripVersion,
    UserPreference,
)
from backend.persistence.outbox_repository import (
    canonical_json_hash,
    prepare_assistant_bundle,
    stable_event_id,
    terminal_sequence_for_bundle,
)


class TurnPersistenceError(RuntimeError):
    """Base class for V4 turn persistence failures."""


class TurnNotFoundError(TurnPersistenceError):
    pass


class TurnIdempotencyConflictError(TurnPersistenceError):
    pass


class TurnStateVersionConflictError(TurnPersistenceError):
    pass


class InvalidTurnWriteError(TurnPersistenceError):
    pass


FaultInjector = Callable[[str], None]


class UserMessageWrite(V4ContractModel):
    message_id: UUID
    client_message_id: UUID
    text: DisplayText
    attachments: list[V4Attachment] = Field(default_factory=list)
    message_metadata: dict[str, JsonValue] = Field(default_factory=dict)
    protocol_version: Literal["v4"] = "v4"
    schema_version: Literal["4.0.0"] = "4.0.0"


@dataclass(frozen=True)
class AcceptedTurnRecord:
    turn_id: UUID
    trip_id: UUID
    request_id: UUID
    user_message_id: UUID
    generation_id: UUID | None
    base_state_version: int
    status: str
    idempotent_replay: bool


class SemanticOperationWrite(V4ContractModel):
    operation_id: UUID
    proposal: SemanticOperationProposal
    status: Literal["accepted"] = "accepted"
    supersedes_operation_id: UUID | None = None


class ToolObservationWrite(V4ContractModel):
    observation_id: UUID
    observation: ToolObservation
    provider: Identifier
    request_hash: Digest
    expires_at: datetime | None = None


class PendingInteractionWrite(V4ContractModel):
    interaction_id: UUID
    source_message_id: UUID
    interaction: PendingInteraction

    @model_validator(mode="after")
    def interaction_id_matches_payload(self) -> PendingInteractionWrite:
        if str(self.interaction_id) != self.interaction.interaction_id:
            raise ValueError("pending interaction ID must match its payload")
        return self


class AssistantMessageWrite(V4ContractModel):
    message_id: UUID
    text: DisplayText
    attachments: list[V4Attachment] = Field(default_factory=list)
    message_metadata: dict[str, JsonValue] = Field(default_factory=dict)
    generation_id: UUID
    message_type: Literal["text", "card", "task_book", "plan", "status"] = "text"


class TurnResultWrite(V4ContractModel):
    state: V4TripStateEnvelope
    phase: Identifier
    assistant_message: AssistantMessageWrite
    outbox_id: UUID
    outbox_cursor: Identifier
    publication_key: Identifier
    generation_mode: Literal["qwen", "fallback"]
    semantic_operations: list[SemanticOperationWrite] = Field(default_factory=list)
    tool_observations: list[ToolObservationWrite] = Field(default_factory=list)
    invalidated_interaction_ids: list[Identifier] = Field(default_factory=list)
    pending_interaction: PendingInteractionWrite | None = None
    answered_interaction_id: UUID | None = None
    interaction_answer_id: UUID | None = None
    failure_code: Identifier | None = None
    decision_audit: dict[str, JsonValue] = Field(default_factory=dict)
    plan_version_id: UUID | None = None
    event_type: Literal["assistant.message"] = "assistant.message"
    outcome: Literal[
        "answered",
        "card_ready",
        "task_book_ready",
        "plan_ready",
        "awaiting_user",
    ] = "answered"
    chunk_size: int = Field(default=64, ge=1, le=2_000, strict=True)

    @model_validator(mode="after")
    def answered_interaction_fields_are_paired(self) -> TurnResultWrite:
        if (self.answered_interaction_id is None) != (self.interaction_answer_id is None):
            raise ValueError("answered interaction ID and answer ID must appear together")
        return self


@dataclass(frozen=True)
class CommittedTurnRecord:
    turn_id: UUID
    trip_id: UUID
    state_version: int
    assistant_message_id: UUID
    outbox_id: UUID
    outbox_cursor: str
    content_hash: str
    idempotent_replay: bool


class TurnRepository:
    """Owns the two database transactions in the V4 turn protocol."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def find_admitted_turn(
        self,
        owner_user_id: UUID,
        trip_id: UUID,
        *,
        request_id: UUID,
        idempotency_key: str,
        request_fingerprint: str,
    ) -> AcceptedTurnRecord | None:
        async with self._session_factory() as session:
            turn = await self._find_existing_turn(
                session,
                owner_user_id,
                trip_id,
                request_id=request_id,
                idempotency_key=idempotency_key,
            )
            if turn is None:
                return None
            return await self._accepted_replay(
                session,
                turn,
                idempotency_key=idempotency_key,
                request_fingerprint=request_fingerprint,
            )

    async def load_trip_phase(self, owner_user_id: UUID, trip_id: UUID) -> str:
        async with self._session_factory() as session:
            phase = await session.scalar(
                select(Trip.phase).where(
                    Trip.id == trip_id,
                    Trip.owner_user_id == owner_user_id,
                )
            )
        if phase is None:
            raise TurnNotFoundError("trip was not found")
        return str(phase)

    async def load_plan_version_number(
        self,
        owner_user_id: UUID,
        trip_id: UUID,
        plan_version_id: UUID,
    ) -> int:
        async with self._session_factory() as session:
            version = await session.scalar(
                select(TripVersion.version_number)
                .join(Trip, Trip.id == TripVersion.trip_id)
                .where(
                    TripVersion.id == plan_version_id,
                    TripVersion.trip_id == trip_id,
                    Trip.owner_user_id == owner_user_id,
                    Trip.current_plan_version_id == plan_version_id,
                )
            )
        if version is None:
            raise TurnNotFoundError("current plan version was not found")
        return int(version)

    async def load_initial_cold_start_profile(
        self, owner_user_id: UUID, trip_id: UUID
    ) -> ColdStartProfileSnapshot | None:
        """Load this owner's real saved defaults only at the initial V4 boundary."""

        async with self._session_factory() as session:
            owned = await session.scalar(
                select(Trip.id).where(Trip.id == trip_id, Trip.owner_user_id == owner_user_id)
            )
            if owned is None:
                raise TurnNotFoundError("trip was not found")
            preference = await session.scalar(
                select(UserPreference)
                .where(
                    UserPreference.user_id == owner_user_id,
                    UserPreference.preference_key == "cold_start",
                    UserPreference.source == "cold_start",
                    UserPreference.active.is_(True),
                )
                .order_by(UserPreference.updated_at.desc(), UserPreference.id)
                .limit(1)
            )
            if preference is None:
                return None
            return ColdStartProfileSnapshot(
                profile_version=1,
                captured_at=_aware(preference.last_confirmed_at or preference.updated_at),
                preferences=ColdStartSubmission.model_validate(preference.value),
                source_evidence_refs=[f"preference:{preference.id}"],
            )

    async def load_committed_turn(
        self,
        owner_user_id: UUID,
        turn_id: UUID,
    ) -> CommittedTurnRecord:
        """Resolve one already committed idempotent turn for stable event replay."""

        async with self._session_factory() as session:
            turn = await session.scalar(
                select(AgentTurn)
                .join(Trip, Trip.id == AgentTurn.trip_id)
                .where(
                    AgentTurn.id == turn_id,
                    Trip.owner_user_id == owner_user_id,
                )
            )
            if turn is None:
                raise TurnNotFoundError("turn was not found")
            if turn.status != "committed":
                raise InvalidTurnWriteError("turn has no committed result to replay")
            return await self._committed_replay(session, turn)

    async def load_latest_snapshot(
        self,
        owner_user_id: UUID,
        trip_id: UUID,
    ) -> PersistedTripProjection:
        """Restore either a lossless legacy row or a complete V4 dual-state commit."""

        async with self._session_factory() as session:
            return await self._read_projection(session, owner_user_id, trip_id)

    async def _read_projection(
        self, session: AsyncSession, owner_user_id: UUID, trip_id: UUID
    ) -> PersistedTripProjection:
        row = (
            await session.execute(
                select(TripSnapshot, TripSemanticStateVersion, DiscoveryRuntimeStateVersion)
                .join(Trip, Trip.id == TripSnapshot.trip_id)
                .outerjoin(
                    TripSemanticStateVersion,
                    (TripSemanticStateVersion.trip_id == TripSnapshot.trip_id)
                    & (TripSemanticStateVersion.state_version == TripSnapshot.state_version),
                )
                .outerjoin(
                    DiscoveryRuntimeStateVersion,
                    (DiscoveryRuntimeStateVersion.trip_id == TripSnapshot.trip_id)
                    & (DiscoveryRuntimeStateVersion.state_version == TripSnapshot.state_version),
                )
                .where(TripSnapshot.trip_id == trip_id, Trip.owner_user_id == owner_user_id)
                .order_by(TripSnapshot.state_version.desc())
                .limit(1)
            )
        ).one_or_none()
        if row is None:
            raise TurnNotFoundError("trip was not found")
        snapshot, semantic, runtime = row
        return replace(
            decode_persisted_snapshot(
                snapshot, semantic_state=semantic, discovery_runtime_state=runtime
            ),
            owner_user_id=owner_user_id,
        )

    async def append_setup_feedback(
        self,
        owner_user_id: UUID,
        trip_id: UUID,
        turn_id: UUID,
        generation_id: UUID,
        *,
        ordinal: int,
        text: str,
        created_at: datetime,
    ) -> ConversationMessageV4 | None:
        """Commit program feedback independently of the still-running model turn."""
        if ordinal not in (0, 1):
            raise InvalidTurnWriteError("setup feedback ordinal must be zero or one")
        message_id = uuid5(NAMESPACE_URL, f"trip-setup-feedback:{generation_id}:{ordinal}")
        async with self._session_factory() as session, session.begin():
            turn = await session.scalar(
                select(AgentTurn)
                .join(Trip, Trip.id == AgentTurn.trip_id)
                .where(
                    AgentTurn.id == turn_id,
                    AgentTurn.trip_id == trip_id,
                    Trip.owner_user_id == owner_user_id,
                )
                .with_for_update()
            )
            if turn is None or turn.generation_id != generation_id:
                raise TurnNotFoundError("setup feedback turn was not found")
            existing = await session.get(Message, message_id)
            if existing is not None:
                return _conversation_message_from_row(existing)
            if turn.status not in {"accepted", "running"}:
                return None
            row = Message(
                id=message_id,
                trip_id=trip_id,
                turn_id=turn_id,
                generation_id=generation_id,
                request_id=turn.request_id,
                state_version=turn.base_state_version,
                ordinal=ordinal,
                status="committed",
                role="system",
                message_type="status",
                text=text,
                attachments=[],
                message_metadata={"command_type": "trip_setup"},
                protocol_version=V4_PROTOCOL_VERSION,
                schema_version=V4_SCHEMA_VERSION,
                generation_mode="system",
                content_hash=canonical_json_hash({"text": text}),
                created_at=created_at,
            )
            session.add(row)
            await session.flush()
            return _conversation_message_from_row(row)

    async def load_setup_feedback(
        self,
        owner_user_id: UUID,
        trip_id: UUID,
    ) -> list[ConversationMessageV4]:
        # A first turn can still have a legacy/empty trip snapshot while its
        # program feedback is already committed. Restore those messages too.
        async with self._session_factory() as session:
            rows = (
                await session.scalars(
                    select(Message)
                    .join(Trip, Trip.id == Message.trip_id)
                    .where(
                        Message.trip_id == trip_id,
                        Trip.owner_user_id == owner_user_id,
                        Message.role == "system",
                        Message.message_type == "status",
                        Message.status == "committed",
                        Message.protocol_version == V4_PROTOCOL_VERSION,
                    )
                    .order_by(Message.created_at, Message.ordinal)
                )
            ).all()
            return [_conversation_message_from_row(row) for row in rows]

    async def load_conversation_snapshot(
        self,
        owner_user_id: UUID,
        trip_id: UUID,
        *,
        projection: PersistedTripProjection | None = None,
    ) -> ConversationSnapshotV4:
        """Rebuild the complete committed V4 conversation after a process restart."""
        return (
            await self._load_conversation(owner_user_id, trip_id, projection=projection)
        ).snapshot

    async def load_conversation_view(
        self,
        owner_user_id: UUID,
        trip_id: UUID,
        *,
        projection: PersistedTripProjection | None = None,
    ) -> ConversationView:
        return await self._load_conversation(
            owner_user_id, trip_id, history_limit=8, projection=projection
        )

    async def _load_conversation(
        self,
        owner_user_id: UUID,
        trip_id: UUID,
        *,
        history_limit: int | None = None,
        projection: PersistedTripProjection | None = None,
    ) -> ConversationView:

        async with self._session_factory() as session:
            if projection is None:
                try:
                    projection = await self._read_projection(session, owner_user_id, trip_id)
                except (SnapshotCompatibilityError, ValidationError) as error:
                    raise InvalidTurnWriteError(
                        "persisted V4 snapshot failed contract validation"
                    ) from error
            if projection.trip_id != trip_id or projection.owner_user_id != owner_user_id:
                raise TurnNotFoundError("snapshot does not belong to this owner and trip")
            if projection.kind != "v4" or projection.typed_state is None:
                raise InvalidTurnWriteError("legacy snapshot is not a V4 conversation snapshot")
            trip_state = projection.typed_state
            terminal_event = (
                ConversationEventV4.model_validate(projection.terminal_event)
                if projection.terminal_event is not None
                else None
            )
            if terminal_event is None or projection.committed_at is None:
                raise InvalidTurnWriteError("committed V4 snapshot is missing its terminal event")

            history = ConversationHistoryWindow(through_state_version=projection.state_version)
            if history_limit is not None:
                retain_ids: set[UUID] = set()
                # Keep the last published plan visible during a revision, even
                # when its original message predates the recent-message window.
                if trip_state.published_plan is None:
                    latest_plan_id = await session.scalar(
                        select(Message.id)
                        .where(
                            Message.trip_id == trip_id,
                            Message.status == "committed",
                            Message.protocol_version == V4_PROTOCOL_VERSION,
                            Message.message_type == "plan",
                            Message.state_version <= projection.state_version,
                        )
                        .order_by(Message.state_version.desc(), Message.created_at.desc())
                        .limit(1)
                    )
                    if latest_plan_id is not None:
                        retain_ids.add(latest_plan_id)
                pending = trip_state.discovery_runtime_state.pending_interaction
                if pending is not None:
                    source_id = await session.scalar(
                        select(PendingInteractionRecord.source_message_id).where(
                            PendingInteractionRecord.trip_id == trip_id,
                            PendingInteractionRecord.id == UUID(pending.interaction_id),
                        )
                    )
                    if source_id is not None:
                        retain_ids.add(source_id)
                page = await _conversation_history_page(
                    session,
                    trip_id,
                    through_state_version=projection.state_version,
                    limit=history_limit,
                    retain_ids=retain_ids,
                )
                messages, history = page.messages, page.history
            else:
                rows = (
                    await session.scalars(
                        select(Message)
                        .where(
                            Message.trip_id == trip_id,
                            Message.status == "committed",
                            Message.protocol_version == V4_PROTOCOL_VERSION,
                            Message.state_version <= projection.state_version,
                        )
                        .order_by(
                            Message.state_version,
                            case((Message.role == "user", 0), else_=1),
                            Message.created_at,
                            Message.ordinal,
                            Message.id,
                        )
                    )
                ).all()
                messages = [
                    _conversation_message_from_row(row) for row in rows if _is_visible_message(row)
                ]
            try:
                return ConversationView(
                    snapshot=ConversationSnapshotV4(
                        trip_state=trip_state,
                        messages=messages,
                        pending_interaction=trip_state.discovery_runtime_state.pending_interaction,
                        terminal_event=terminal_event,
                        last_outbox_cursor=projection.outbox_cursor,
                        snapshot_at=_aware(projection.committed_at),
                    ),
                    history=history,
                )
            except ValidationError as error:
                raise InvalidTurnWriteError(
                    "persisted conversation failed V4 snapshot validation"
                ) from error

    async def load_conversation_history(
        self,
        owner_user_id: UUID,
        trip_id: UUID,
        *,
        before_state_version: int,
        through_state_version: int,
    ) -> ConversationHistoryPage:
        async with self._session_factory() as session:
            version = await session.scalar(
                select(Trip.state_version).where(
                    Trip.id == trip_id,
                    Trip.owner_user_id == owner_user_id,
                )
            )
            if version is None:
                raise TurnNotFoundError("trip was not found")
            return await _conversation_history_page(
                session,
                trip_id,
                through_state_version=min(version, through_state_version),
                before_state_version=before_state_version,
                limit=4,
            )

    async def load_conversation_message(
        self,
        owner_user_id: UUID,
        trip_id: UUID,
        message_id: UUID,
    ) -> ConversationMessageV4:
        async with self._session_factory() as session:
            row = await session.scalar(
                select(Message)
                .join(Trip, Trip.id == Message.trip_id)
                .where(
                    Trip.owner_user_id == owner_user_id,
                    Message.trip_id == trip_id,
                    Message.id == message_id,
                    Message.status == "committed",
                    Message.protocol_version == V4_PROTOCOL_VERSION,
                )
            )
            if row is None:
                raise TurnNotFoundError("message was not found")
            return _conversation_message_from_row(row)

    async def accept_turn(
        self,
        owner_user_id: UUID,
        trip_id: UUID,
        *,
        request_id: UUID,
        idempotency_key: str,
        request_fingerprint: str,
        base_state_version: int,
        user_message: UserMessageWrite,
        generation_id: UUID | None = None,
    ) -> AcceptedTurnRecord:
        user_message = _revalidate_user_message(user_message)
        if not idempotency_key or not request_fingerprint:
            raise InvalidTurnWriteError("idempotency key and fingerprint are required")
        if base_state_version < 0:
            raise InvalidTurnWriteError("base_state_version must not be negative")

        try:
            async with self._session_factory() as session, session.begin():
                existing = await self._find_existing_turn(
                    session,
                    owner_user_id,
                    trip_id,
                    request_id=request_id,
                    idempotency_key=idempotency_key,
                )
                if existing is not None:
                    return await self._accepted_replay(
                        session,
                        existing,
                        idempotency_key=idempotency_key,
                        request_fingerprint=request_fingerprint,
                    )

                trip = await session.scalar(
                    select(Trip)
                    .where(Trip.id == trip_id, Trip.owner_user_id == owner_user_id)
                    .with_for_update()
                )
                if trip is None:
                    raise TurnNotFoundError("trip was not found")
                if trip.state_version != base_state_version:
                    raise TurnStateVersionConflictError("trip state version no longer matches")

                turn_id = uuid4()
                session.add(
                    AgentTurn(
                        id=turn_id,
                        trip_id=trip_id,
                        request_id=request_id,
                        idempotency_key=idempotency_key,
                        request_fingerprint=request_fingerprint,
                        base_state_version=base_state_version,
                        generation_id=generation_id,
                        status="accepted",
                    )
                )
                session.add(
                    Message(
                        id=user_message.message_id,
                        trip_id=trip_id,
                        client_message_id=user_message.client_message_id,
                        turn_id=turn_id,
                        state_version=base_state_version,
                        ordinal=0,
                        status="accepted",
                        role="user",
                        message_type="text",
                        text=user_message.text,
                        attachments=_dump_attachments(user_message.attachments),
                        message_metadata=dict(user_message.message_metadata),
                        request_id=request_id,
                        generation_id=generation_id,
                        protocol_version=user_message.protocol_version,
                        schema_version=user_message.schema_version,
                        content_hash=canonical_json_hash(
                            {
                                "text": user_message.text,
                                "attachments": _dump_attachments(user_message.attachments),
                                "metadata": user_message.message_metadata,
                            }
                        ),
                    )
                )
                await session.flush()
                return AcceptedTurnRecord(
                    turn_id=turn_id,
                    trip_id=trip_id,
                    request_id=request_id,
                    user_message_id=user_message.message_id,
                    generation_id=generation_id,
                    base_state_version=base_state_version,
                    status="accepted",
                    idempotent_replay=False,
                )
        except IntegrityError:
            existing = await self._load_existing_after_collision(
                owner_user_id,
                trip_id,
                request_id=request_id,
                idempotency_key=idempotency_key,
            )
            if existing is None:
                raise
            async with self._session_factory() as session:
                attached = await session.get(AgentTurn, existing.id)
                assert attached is not None
                return await self._accepted_replay(
                    session,
                    attached,
                    idempotency_key=idempotency_key,
                    request_fingerprint=request_fingerprint,
                )

    async def commit_turn_result(
        self,
        owner_user_id: UUID,
        turn_id: UUID,
        result: TurnResultWrite,
        *,
        fault_injector: FaultInjector | None = None,
    ) -> CommittedTurnRecord:
        result = _revalidate_turn_result(result)
        attachment_payloads = _dump_attachments(result.assistant_message.attachments)
        prepared = prepare_assistant_bundle(
            message_id=result.assistant_message.message_id,
            text=result.assistant_message.text,
            attachments=attachment_payloads,
            generation_mode=result.generation_mode,
            failure_code=result.failure_code,
            chunk_size=result.chunk_size,
            invalidated_interaction_ids=result.invalidated_interaction_ids,
        )
        result_fingerprint = _turn_result_fingerprint(result, prepared.content_hash)
        now = datetime.now(UTC)

        async with self._session_factory() as session, session.begin():
            turn = await session.scalar(
                select(AgentTurn)
                .join(Trip, Trip.id == AgentTurn.trip_id)
                .where(AgentTurn.id == turn_id, Trip.owner_user_id == owner_user_id)
                .with_for_update()
            )
            if turn is None:
                raise TurnNotFoundError("turn was not found")
            if turn.status == "committed":
                if turn.result_fingerprint != result_fingerprint:
                    raise TurnIdempotencyConflictError(
                        "committed turn was retried with a different result"
                    )
                return await self._committed_replay(session, turn)
            if turn.status not in {"accepted", "running"}:
                raise InvalidTurnWriteError("terminal turn cannot commit a result")
            if (
                turn.generation_id is not None
                and turn.generation_id != result.assistant_message.generation_id
            ):
                raise InvalidTurnWriteError(
                    "authoritative result generation does not match the admitted turn"
                )

            next_state_version = turn.base_state_version + 1
            _validate_result_versions(
                result,
                trip_id=turn.trip_id,
                state_version=next_state_version,
            )
            previous_plan_version_id = await session.scalar(
                select(Trip.current_plan_version_id).where(Trip.id == turn.trip_id)
            )
            publication_plan = result.state.published_plan
            if result.plan_version_id is not None and publication_plan is not None:
                change = publication_plan.change_request
                if previous_plan_version_id is not None and change is None:
                    raise InvalidTurnWriteError(
                        "a child formal plan requires its PlanChangeRequest"
                    )
                if change is not None:
                    previous_version_number = await session.scalar(
                        select(TripVersion.version_number).where(
                            TripVersion.id == previous_plan_version_id,
                            TripVersion.trip_id == turn.trip_id,
                        )
                    )
                    if (
                        previous_plan_version_id is None
                        or change.base_plan_id != str(previous_plan_version_id)
                        or previous_version_number is None
                        or change.base_plan_version != int(previous_version_number)
                    ):
                        raise InvalidTurnWriteError(
                            "PlanChangeRequest does not match the current plan version"
                        )
            state_payload = result.state.model_dump(mode="json")
            semantic_payload = result.state.semantic_state.model_dump(mode="json")
            runtime_payload = result.state.discovery_runtime_state.model_dump(mode="json")
            terminal_event = AssistantCompletedEvent(
                event_id=str(
                    stable_event_id(
                        result.outbox_id,
                        event_type="assistant.completed",
                        content_hash=prepared.content_hash,
                    )
                ),
                event_type="assistant.completed",
                trip_id=str(turn.trip_id),
                turn_id=str(turn.id),
                generation_id=str(result.assistant_message.generation_id),
                sequence=terminal_sequence_for_bundle(prepared),
                emitted_at=now,
                message_id=str(result.assistant_message.message_id),
                state_version=next_state_version,
                content_hash=prepared.content_hash,
                generation_mode=result.generation_mode,
                outbox_cursor=result.outbox_cursor,
                outcome=result.outcome,
            )
            terminal_event_payload = terminal_event.model_dump(mode="json")
            claimed = await session.execute(
                update(Trip)
                .where(
                    Trip.id == turn.trip_id,
                    Trip.owner_user_id == owner_user_id,
                    Trip.state_version == turn.base_state_version,
                )
                .values(
                    phase=result.phase,
                    state_version=next_state_version,
                    schema_version=V4_SCHEMA_VERSION,
                    current_plan_version_id=result.state.current_plan_version_id,
                )
            )
            if getattr(claimed, "rowcount", 0) != 1:
                raise TurnStateVersionConflictError("trip state version no longer matches")
            _inject(fault_injector, "after_trip_cas")

            if result.plan_version_id is not None:
                plan = result.state.published_plan
                assert plan is not None
                plan_is_confirmed = result.phase == "confirmed"
                current_version_number = await session.scalar(
                    select(func.max(TripVersion.version_number)).where(
                        TripVersion.trip_id == turn.trip_id
                    )
                )
                session.add(
                    TripVersion(
                        id=result.plan_version_id,
                        trip_id=turn.trip_id,
                        parent_version_id=previous_plan_version_id,
                        publication_key=result.publication_key,
                        generation_id=result.assistant_message.generation_id,
                        version_number=int(current_version_number or 0) + 1,
                        state_version=next_state_version,
                        status="confirmed" if plan_is_confirmed else "draft",
                        snapshot=plan.model_dump(mode="json"),
                        schema_version=V4_SCHEMA_VERSION,
                        confirmed_at=now if plan_is_confirmed else None,
                    )
                )
            _inject(fault_injector, "after_plan_version")

            session.add(
                TripSnapshot(
                    id=uuid4(),
                    trip_id=turn.trip_id,
                    plan_version_id=result.state.current_plan_version_id,
                    state_version=next_state_version,
                    schema_version=V4_SCHEMA_VERSION,
                    snapshot=state_payload,
                    snapshot_kind="v4",
                    protocol_version=V4_PROTOCOL_VERSION,
                    turn_id=turn.id,
                    terminal_status="committed",
                    terminal_event=terminal_event_payload,
                    message_cursor=result.assistant_message.message_id,
                    outbox_cursor=result.outbox_cursor,
                )
            )
            session.add(
                TripSemanticStateVersion(
                    id=uuid4(),
                    trip_id=turn.trip_id,
                    state_version=next_state_version,
                    schema_version=V4_SCHEMA_VERSION,
                    payload=semantic_payload,
                    content_hash=canonical_json_hash(semantic_payload),
                )
            )
            session.add(
                DiscoveryRuntimeStateVersion(
                    id=uuid4(),
                    trip_id=turn.trip_id,
                    state_version=next_state_version,
                    schema_version=V4_SCHEMA_VERSION,
                    payload=runtime_payload,
                    content_hash=canonical_json_hash(runtime_payload),
                )
            )
            _inject(fault_injector, "after_state_snapshots")

            for operation in result.semantic_operations:
                proposal = operation.proposal.root
                payload = operation.proposal.model_dump(mode="json")
                evidence = {
                    "source_refs": list(proposal.source_refs),
                    "confidence": proposal.confidence.value,
                }
                session.add(
                    SemanticOperationRecord(
                        operation_id=operation.operation_id,
                        trip_id=turn.trip_id,
                        turn_id=turn.id,
                        state_version=next_state_version,
                        target=proposal.target.value,
                        operation_kind=proposal.operation_type,
                        status=operation.status,
                        payload=payload,
                        evidence=evidence,
                        supersedes_operation_id=operation.supersedes_operation_id,
                        content_hash=canonical_json_hash(
                            {"payload": payload, "evidence": evidence}
                        ),
                    )
                )
            _inject(fault_injector, "after_semantic_operations")

            for observation in result.tool_observations:
                observation_payload = observation.observation.model_dump(mode="json")
                session.add(
                    ToolObservationRecord(
                        observation_id=observation.observation_id,
                        trip_id=turn.trip_id,
                        turn_id=turn.id,
                        tool_name=observation.observation.capability.value,
                        provider=observation.provider,
                        request_hash=observation.request_hash,
                        status=observation.observation.status.value,
                        safe_payload=observation_payload,
                        source_refs=list(observation.observation.source_refs),
                        content_hash=canonical_json_hash(
                            {
                                "safe_payload": observation_payload,
                                "source_refs": observation.observation.source_refs,
                            }
                        ),
                        observed_at=observation.observation.observed_at,
                        expires_at=observation.expires_at,
                    )
                )
            _inject(fault_injector, "after_tool_observations")

            if result.answered_interaction_id is not None:
                answered = await session.execute(
                    update(PendingInteractionRecord)
                    .where(
                        PendingInteractionRecord.id == result.answered_interaction_id,
                        PendingInteractionRecord.trip_id == turn.trip_id,
                        PendingInteractionRecord.status == "active",
                    )
                    .values(
                        status="answered",
                        answer_id=result.interaction_answer_id,
                        closed_state_version=next_state_version,
                        closed_at=now,
                    )
                )
                if getattr(answered, "rowcount", 0) != 1:
                    raise InvalidTurnWriteError("answered interaction is not active")

            active_pending_id = await session.scalar(
                select(PendingInteractionRecord.id).where(
                    PendingInteractionRecord.trip_id == turn.trip_id,
                    PendingInteractionRecord.status == "active",
                )
            )
            incoming_pending_id = (
                result.pending_interaction.interaction_id
                if result.pending_interaction is not None
                else None
            )
            preserves_active = (
                active_pending_id is not None and active_pending_id == incoming_pending_id
            )
            if not preserves_active:
                await session.execute(
                    update(PendingInteractionRecord)
                    .where(
                        PendingInteractionRecord.trip_id == turn.trip_id,
                        PendingInteractionRecord.status == "active",
                    )
                    .values(
                        status="superseded",
                        closed_state_version=next_state_version,
                        closed_at=now,
                    )
                )
            if result.pending_interaction is not None and not preserves_active:
                pending = result.pending_interaction
                pending_payload = pending.interaction.model_dump(mode="json")
                session.add(
                    PendingInteractionRecord(
                        id=pending.interaction_id,
                        trip_id=turn.trip_id,
                        turn_id=turn.id,
                        source_message_id=pending.source_message_id,
                        section=pending.interaction.section.value,
                        interaction_kind=pending.interaction.kind.value,
                        status="active",
                        issued_state_version=next_state_version,
                        payload=pending_payload,
                        content_hash=canonical_json_hash(pending_payload),
                    )
                )
            _inject(fault_injector, "after_pending_interaction")

            user_message = await session.execute(
                update(Message)
                .where(
                    Message.turn_id == turn.id,
                    Message.trip_id == turn.trip_id,
                    Message.role == "user",
                    Message.status == "accepted",
                )
                .values(
                    status="committed",
                    state_version=next_state_version,
                    generation_id=result.assistant_message.generation_id,
                )
            )
            if getattr(user_message, "rowcount", 0) != 1:
                raise InvalidTurnWriteError("accepted turn must own exactly one user message")
            session.add(
                Message(
                    id=result.assistant_message.message_id,
                    trip_id=turn.trip_id,
                    turn_id=turn.id,
                    state_version=next_state_version,
                    ordinal=0,
                    status="committed",
                    role="assistant",
                    message_type=result.assistant_message.message_type,
                    text=result.assistant_message.text,
                    attachments=attachment_payloads,
                    message_metadata=dict(result.assistant_message.message_metadata),
                    request_id=turn.request_id,
                    generation_id=result.assistant_message.generation_id,
                    protocol_version=V4_PROTOCOL_VERSION,
                    schema_version=V4_SCHEMA_VERSION,
                    generation_mode=result.generation_mode,
                    failure_code=result.failure_code,
                    content_hash=prepared.content_hash,
                )
            )
            _inject(fault_injector, "after_messages")

            session.add(
                OutboxEvent(
                    id=result.outbox_id,
                    cursor=result.outbox_cursor,
                    publication_key=result.publication_key,
                    trip_id=turn.trip_id,
                    turn_id=turn.id,
                    generation_id=result.assistant_message.generation_id,
                    message_id=result.assistant_message.message_id,
                    committed_state_version=next_state_version,
                    plan_version_id=result.plan_version_id,
                    event_type=result.event_type,
                    payload=prepared.payload,
                    chunks=prepared.chunks,
                    terminal_event=terminal_event_payload,
                    content_hash=prepared.content_hash,
                    chunking_algorithm_version=prepared.chunking_algorithm_version,
                    delivery_status="pending",
                    delivered_sequence=0,
                    retry_count=0,
                    next_attempt_at=now,
                )
            )
            _inject(fault_injector, "after_outbox")

            turn.status = "committed"
            turn.committed_state_version = next_state_version
            turn.result_fingerprint = result_fingerprint
            turn.generation_mode = result.generation_mode
            turn.failure_code = result.failure_code
            turn.decision_audit = dict(result.decision_audit)
            turn.committed_at = now
            if turn.generation_id is None:
                turn.generation_id = result.assistant_message.generation_id
            _inject(fault_injector, "before_turn_commit")
            await session.flush()

            return CommittedTurnRecord(
                turn_id=turn.id,
                trip_id=turn.trip_id,
                state_version=next_state_version,
                assistant_message_id=result.assistant_message.message_id,
                outbox_id=result.outbox_id,
                outbox_cursor=result.outbox_cursor,
                content_hash=prepared.content_hash,
                idempotent_replay=False,
            )

    async def mark_turn_failed(
        self,
        owner_user_id: UUID,
        turn_id: UUID,
        *,
        failure_code: str,
        cancelled: bool = False,
    ) -> None:
        async with self._session_factory() as session, session.begin():
            turn = await session.scalar(
                select(AgentTurn)
                .join(Trip, Trip.id == AgentTurn.trip_id)
                .where(
                    AgentTurn.id == turn_id,
                    Trip.owner_user_id == owner_user_id,
                    AgentTurn.status.in_(("accepted", "running")),
                )
                .with_for_update()
            )
            if turn is None:
                raise TurnNotFoundError("active turn was not found")
            turn.status = "cancelled" if cancelled else "failed"
            turn.failure_code = failure_code

    async def _find_existing_turn(
        self,
        session: AsyncSession,
        owner_user_id: UUID,
        trip_id: UUID,
        *,
        request_id: UUID,
        idempotency_key: str,
    ) -> AgentTurn | None:
        existing: AgentTurn | None = await session.scalar(
            select(AgentTurn)
            .join(Trip, Trip.id == AgentTurn.trip_id)
            .where(
                AgentTurn.trip_id == trip_id,
                Trip.owner_user_id == owner_user_id,
                or_(
                    AgentTurn.idempotency_key == idempotency_key,
                    AgentTurn.request_id == request_id,
                ),
            )
        )
        return existing

    async def _load_existing_after_collision(
        self,
        owner_user_id: UUID,
        trip_id: UUID,
        *,
        request_id: UUID,
        idempotency_key: str,
    ) -> AgentTurn | None:
        async with self._session_factory() as session:
            return await self._find_existing_turn(
                session,
                owner_user_id,
                trip_id,
                request_id=request_id,
                idempotency_key=idempotency_key,
            )

    async def _accepted_replay(
        self,
        session: AsyncSession,
        turn: AgentTurn,
        *,
        idempotency_key: str,
        request_fingerprint: str,
    ) -> AcceptedTurnRecord:
        if (
            turn.idempotency_key != idempotency_key
            or turn.request_fingerprint != request_fingerprint
        ):
            raise TurnIdempotencyConflictError(
                "idempotency key or request ID was already used for different content"
            )
        message_id = await session.scalar(
            select(Message.id).where(
                Message.turn_id == turn.id,
                Message.trip_id == turn.trip_id,
                Message.role == "user",
                Message.ordinal == 0,
            )
        )
        if message_id is None:
            raise InvalidTurnWriteError("turn is missing its admitted user message")
        return AcceptedTurnRecord(
            turn_id=turn.id,
            trip_id=turn.trip_id,
            request_id=turn.request_id,
            user_message_id=message_id,
            generation_id=turn.generation_id,
            base_state_version=turn.base_state_version,
            status=turn.status,
            idempotent_replay=True,
        )

    async def _committed_replay(
        self,
        session: AsyncSession,
        turn: AgentTurn,
    ) -> CommittedTurnRecord:
        message = await session.scalar(
            select(Message).where(
                Message.turn_id == turn.id,
                Message.trip_id == turn.trip_id,
                Message.role == "assistant",
                Message.ordinal == 0,
                Message.status == "committed",
            )
        )
        outbox = await session.scalar(
            select(OutboxEvent).where(
                OutboxEvent.turn_id == turn.id,
                OutboxEvent.trip_id == turn.trip_id,
            )
        )
        if message is None or outbox is None or turn.committed_state_version is None:
            raise InvalidTurnWriteError("committed turn is missing its durable result")
        return CommittedTurnRecord(
            turn_id=turn.id,
            trip_id=turn.trip_id,
            state_version=turn.committed_state_version,
            assistant_message_id=message.id,
            outbox_id=outbox.id,
            outbox_cursor=outbox.cursor,
            content_hash=outbox.content_hash,
            idempotent_replay=True,
        )


async def _conversation_history_page(
    session: AsyncSession,
    trip_id: UUID,
    *,
    through_state_version: int,
    limit: int,
    before_state_version: int | None = None,
    retain_ids: set[UUID] | None = None,
) -> ConversationHistoryPage:
    """Bound at SQL level; never read old plan JSON just to throw it away."""
    retain_ids = retain_ids or set()
    filters = [
        Message.trip_id == trip_id,
        Message.status == "committed",
        Message.protocol_version == V4_PROTOCOL_VERSION,
        Message.state_version <= through_state_version,
    ]
    window_filters = list(filters)
    if before_state_version is not None:
        window_filters.append(Message.state_version < before_state_version)
    versions = (
        await session.scalars(
            select(Message.state_version)
            .where(*window_filters)
            .distinct()
            .order_by(Message.state_version.desc())
            .limit(limit + 1)
        )
    ).all()
    selected_versions = versions[:limit]
    rows = (
        await session.scalars(
            select(Message)
            .options(defer(Message.attachments))
            .where(
                *filters,
                or_(Message.state_version.in_(selected_versions), Message.id.in_(retain_ids)),
            )
            .order_by(
                Message.state_version,
                case((Message.role == "user", 0), else_=1),
                Message.created_at,
                Message.ordinal,
                Message.id,
            )
        )
    ).all()
    deferred_ids = {
        row.id for row in rows if row.message_type == "plan" and row.id not in retain_ids
    }
    attachment_ids = [row.id for row in rows if row.id not in deferred_ids]
    attachments: dict[UUID, list[dict[str, Any]]] = (
        {
            message_id: values
            for message_id, values in (
                await session.execute(
                    select(Message.id, Message.attachments).where(Message.id.in_(attachment_ids))
                )
            )
            .tuples()
            .all()
        }
        if attachment_ids
        else {}
    )
    return ConversationHistoryPage(
        trip_id=str(trip_id),
        messages=[
            _conversation_message_from_row(row, attachments=attachments.get(row.id, []))
            for row in rows
            if _is_visible_message(row)
        ],
        history=ConversationHistoryWindow(
            through_state_version=through_state_version,
            before_state_version=selected_versions[-1] if len(versions) > limit else None,
            deferred_attachment_message_ids=[str(row.id) for row in rows if row.id in deferred_ids],
        ),
    )


def _is_visible_message(row: Message) -> bool:
    return not (row.role == "user" and row.message_metadata.get("command_type") == "trip_setup")


def _conversation_message_from_row(
    row: Message,
    *,
    attachments: list[Any] | None = None,
) -> ConversationMessageV4:
    if (
        row.turn_id is None
        or row.generation_id is None
        or row.content_hash is None
        or row.text is None
        or row.state_version is None
    ):
        raise InvalidTurnWriteError("committed V4 message is missing authoritative fields")
    generation_mode = "user" if row.role == "user" else row.generation_mode
    if generation_mode is None:
        raise InvalidTurnWriteError("committed V4 message is missing generation mode")
    try:
        return ConversationMessageV4.model_validate(
            {
                "message_id": str(row.id),
                "trip_id": str(row.trip_id),
                "turn_id": str(row.turn_id),
                "generation_id": str(row.generation_id),
                "role": row.role,
                "message_type": row.message_type,
                "text": row.text,
                "state_version": row.state_version,
                "generation_mode": generation_mode,
                "content_hash": row.content_hash,
                "attachments": row.attachments if attachments is None else attachments,
                "status": "committed",
                "created_at": _aware(row.created_at),
            }
        )
    except ValidationError as error:
        raise InvalidTurnWriteError("committed message failed V4 contract validation") from error


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _validate_result_versions(
    result: TurnResultWrite,
    *,
    trip_id: UUID,
    state_version: int,
) -> None:
    if result.state.protocol_version != V4_PROTOCOL_VERSION:
        raise InvalidTurnWriteError("authoritative result requires protocol_version=v4")
    if result.state.schema_version != V4_SCHEMA_VERSION:
        raise InvalidTurnWriteError("authoritative result requires schema_version=4.0.0")
    if result.state.semantic_state.state_version != state_version:
        raise InvalidTurnWriteError("V4 state must use the committed state version")
    if result.state.semantic_state.trip_id != str(trip_id):
        raise InvalidTurnWriteError("V4 state belongs to another trip")
    if result.pending_interaction is not None and (
        result.pending_interaction.source_message_id != result.assistant_message.message_id
    ):
        raise InvalidTurnWriteError("pending interaction must belong to the assistant message")
    runtime_pending = result.state.discovery_runtime_state.pending_interaction
    committed_pending = (
        result.pending_interaction.interaction if result.pending_interaction is not None else None
    )
    if runtime_pending != committed_pending:
        raise InvalidTurnWriteError("pending interaction write must match DiscoveryRuntimeState")
    attached_plans = [
        attachment.root
        for attachment in result.assistant_message.attachments
        if isinstance(attachment.root, PlannerPublishedPlan)
    ]
    if result.plan_version_id is None:
        if result.outcome == "plan_ready" or result.assistant_message.message_type == "plan":
            raise InvalidTurnWriteError("plan-ready result requires a plan version")
        if attached_plans:
            raise InvalidTurnWriteError("formal plan attachment requires a plan version")
        return
    plan = result.state.published_plan
    if (
        plan is None
        or result.state.current_plan_version_id != result.plan_version_id
        or plan.plan_version_id != result.plan_version_id
        or plan.publication_key != result.publication_key
        or plan.generation_id != result.assistant_message.generation_id
        or plan.trip_id != trip_id
        or plan.based_on_state_version != state_version - 1
    ):
        raise InvalidTurnWriteError("formal plan does not match the authoritative turn result")
    if result.outcome != "plan_ready" or result.assistant_message.message_type != "plan":
        raise InvalidTurnWriteError("formal plan requires plan-ready message semantics")
    if len(attached_plans) != 1 or attached_plans[0] != plan:
        raise InvalidTurnWriteError("assistant message must attach the authoritative formal plan")


def _turn_result_fingerprint(result: TurnResultWrite, content_hash: str) -> str:
    payload = result.model_dump(mode="json", warnings=False)
    payload["assistant_content_hash"] = content_hash
    return canonical_json_hash(payload)


def _revalidate_user_message(message: UserMessageWrite) -> UserMessageWrite:
    try:
        payload = message.model_dump(mode="json", warnings=False)
        return UserMessageWrite.model_validate(payload)
    except (AttributeError, TypeError, ValidationError) as error:
        raise InvalidTurnWriteError("user message does not satisfy the V4 contract") from error


def _revalidate_turn_result(result: TurnResultWrite) -> TurnResultWrite:
    try:
        payload = result.model_dump(mode="json", warnings=False)
        return TurnResultWrite.model_validate(
            payload,
            context={"restore_historical_semantic_state": True},
        )
    except (AttributeError, TypeError, ValidationError) as error:
        raise InvalidTurnWriteError("turn result does not satisfy the V4 contract") from error


def _dump_attachments(attachments: list[V4Attachment]) -> list[dict[str, Any]]:
    return [attachment.model_dump(mode="json") for attachment in attachments]


def _inject(injector: FaultInjector | None, stage: str) -> None:
    if injector is not None:
        injector(stage)
