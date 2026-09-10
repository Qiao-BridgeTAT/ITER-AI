"""Trip-and-turn scoped checkpoints and recoverable planner workspaces."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import ValidationError
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from backend.persistence.models import (
    AgentCheckpoint,
    AgentTurn,
    PlannerWorkspace,
    Trip,
    TripSnapshot,
)
from backend.persistence.outbox_repository import canonical_json_hash

_FORBIDDEN_RUNTIME_KEYS = frozenset(
    {
        "runtime_context",
        "model_gateway",
        "provider_registry",
        "cancellation_token",
        "database_session",
        "db_session",
        "business_date",
        "timezone",
    }
)

V4_PLANNER_CHECKPOINT_VERSION = "v4-planner-1"


class CheckpointPersistenceError(RuntimeError):
    """Base class for invalid or stale checkpoint persistence."""


class CheckpointNotFoundError(CheckpointPersistenceError):
    pass


class CheckpointStaleError(CheckpointPersistenceError):
    pass


class CheckpointPayloadError(CheckpointPersistenceError):
    pass


class PlannerWorkspaceConflictError(CheckpointPersistenceError):
    pass


@dataclass(frozen=True)
class CheckpointRecord:
    checkpoint_row_id: UUID
    trip_id: UUID
    turn_id: UUID
    checkpoint_namespace: str
    checkpoint_id: str
    checkpoint_version: str
    base_state_version: int
    payload: dict[str, Any]
    content_hash: str
    status: str
    expires_at: datetime | None


@dataclass(frozen=True)
class PlannerWorkspaceWrite:
    workspace_id: UUID
    trip_id: UUID
    turn_id: UUID
    generation_id: UUID
    confirmed_task_book_id: UUID
    confirmed_task_book_version: int
    confirmed_task_book_hash: str
    base_state_version: int
    revision: int
    checkpoint_version: str
    payload: dict[str, Any]
    status: str = "working"


@dataclass(frozen=True)
class PlannerWorkspaceRecord:
    workspace_id: UUID
    trip_id: UUID
    turn_id: UUID
    generation_id: UUID
    confirmed_task_book_id: UUID
    confirmed_task_book_version: int
    confirmed_task_book_hash: str
    base_state_version: int
    revision: int
    checkpoint_version: str
    payload: dict[str, Any]
    content_hash: str
    status: str


class CheckpointRepository:
    """Stores execution recovery data without treating it as committed domain state."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def save_checkpoint(
        self,
        owner_user_id: UUID,
        trip_id: UUID,
        turn_id: UUID,
        *,
        checkpoint_namespace: str,
        checkpoint_id: str,
        checkpoint_version: str,
        base_state_version: int,
        payload: dict[str, Any],
        expires_at: datetime | None = None,
    ) -> CheckpointRecord:
        _validate_checkpoint_payload(payload)
        content_hash = canonical_json_hash(payload)
        async with self._session_factory() as session, session.begin():
            await _require_live_turn(
                session,
                owner_user_id,
                trip_id,
                turn_id,
                base_state_version=base_state_version,
            )
            row = await session.scalar(
                select(AgentCheckpoint).where(
                    AgentCheckpoint.trip_id == trip_id,
                    AgentCheckpoint.turn_id == turn_id,
                    AgentCheckpoint.checkpoint_namespace == checkpoint_namespace,
                    AgentCheckpoint.checkpoint_id == checkpoint_id,
                )
            )
            if row is None:
                row = AgentCheckpoint(
                    id=uuid4(),
                    trip_id=trip_id,
                    turn_id=turn_id,
                    checkpoint_namespace=checkpoint_namespace,
                    checkpoint_id=checkpoint_id,
                    checkpoint_version=checkpoint_version,
                    base_state_version=base_state_version,
                    payload=dict(payload),
                    content_hash=content_hash,
                    status="active",
                    expires_at=expires_at,
                )
                session.add(row)
            else:
                if row.base_state_version != base_state_version:
                    raise CheckpointStaleError("checkpoint base state version changed")
                row.checkpoint_version = checkpoint_version
                row.payload = dict(payload)
                row.content_hash = content_hash
                row.status = "active"
                row.expires_at = expires_at
            await session.flush()
            return _checkpoint_record(row)

    async def load_checkpoint(
        self,
        owner_user_id: UUID,
        trip_id: UUID,
        turn_id: UUID,
        *,
        checkpoint_namespace: str,
        checkpoint_id: str,
        now: datetime | None = None,
    ) -> CheckpointRecord | None:
        async with self._session_factory() as session:
            row = await session.scalar(
                select(AgentCheckpoint)
                .join(Trip, Trip.id == AgentCheckpoint.trip_id)
                .join(AgentTurn, AgentTurn.id == AgentCheckpoint.turn_id)
                .where(
                    AgentCheckpoint.trip_id == trip_id,
                    AgentCheckpoint.turn_id == turn_id,
                    AgentCheckpoint.checkpoint_namespace == checkpoint_namespace,
                    AgentCheckpoint.checkpoint_id == checkpoint_id,
                    Trip.owner_user_id == owner_user_id,
                )
            )
            if row is None:
                return None
            trip_version = await session.scalar(
                select(Trip.state_version).where(Trip.id == trip_id)
            )
            turn_status = await session.scalar(
                select(AgentTurn.status).where(AgentTurn.id == turn_id)
            )
            if trip_version != row.base_state_version or turn_status not in {"accepted", "running"}:
                raise CheckpointStaleError("checkpoint no longer matches the active trip turn")
            if row.status != "active":
                raise CheckpointStaleError("checkpoint is no longer active")
            if now is not None and row.expires_at is not None and row.expires_at <= now:
                raise CheckpointStaleError("checkpoint has expired")
            _validate_checkpoint_payload(row.payload)
            return _checkpoint_record(row)

    async def complete_turn_checkpoints(
        self,
        owner_user_id: UUID,
        trip_id: UUID,
        turn_id: UUID,
    ) -> int:
        async with self._session_factory() as session, session.begin():
            owned = await session.scalar(
                select(AgentTurn.id)
                .join(Trip, Trip.id == AgentTurn.trip_id)
                .where(
                    AgentTurn.id == turn_id,
                    AgentTurn.trip_id == trip_id,
                    Trip.owner_user_id == owner_user_id,
                )
            )
            if owned is None:
                raise CheckpointNotFoundError("turn was not found")
            result = await session.execute(
                update(AgentCheckpoint)
                .where(
                    AgentCheckpoint.trip_id == trip_id,
                    AgentCheckpoint.turn_id == turn_id,
                    AgentCheckpoint.status == "active",
                )
                .values(status="completed")
            )
            return int(getattr(result, "rowcount", 0))

    async def save_planner_workspace(
        self,
        owner_user_id: UUID,
        write: PlannerWorkspaceWrite,
        *,
        expected_previous_revision: int | None = None,
    ) -> PlannerWorkspaceRecord:
        if write.confirmed_task_book_version < 1 or write.revision < 0:
            raise PlannerWorkspaceConflictError("workspace versions are invalid")
        _validate_checkpoint_payload(write.payload)
        _validate_v4_workspace_write(write)
        content_hash = canonical_json_hash(write.payload)
        async with self._session_factory() as session, session.begin():
            turn = await _require_live_turn(
                session,
                owner_user_id,
                write.trip_id,
                write.turn_id,
                base_state_version=write.base_state_version,
            )
            if write.checkpoint_version == V4_PLANNER_CHECKPOINT_VERSION:
                if turn.generation_id != write.generation_id:
                    raise PlannerWorkspaceConflictError(
                        "workspace generation differs from admitted turn"
                    )
                await _validate_current_task_book(session, write)
            latest = await session.scalar(
                select(PlannerWorkspace)
                .where(
                    PlannerWorkspace.trip_id == write.trip_id,
                    PlannerWorkspace.generation_id == write.generation_id,
                )
                .order_by(PlannerWorkspace.revision.desc())
                .limit(1)
                .with_for_update()
            )
            if latest is not None:
                existing_task_book = (
                    latest.confirmed_task_book_id,
                    latest.confirmed_task_book_version,
                    latest.confirmed_task_book_hash,
                )
                requested_task_book = (
                    write.confirmed_task_book_id,
                    write.confirmed_task_book_version,
                    write.confirmed_task_book_hash,
                )
                if existing_task_book != requested_task_book:
                    raise PlannerWorkspaceConflictError(
                        "planner workspace cannot modify its confirmed task book"
                    )
                if write.revision <= latest.revision:
                    if write.revision == latest.revision and latest.content_hash == content_hash:
                        return _workspace_record(latest)
                    raise PlannerWorkspaceConflictError("planner workspace revision is stale")
            if expected_previous_revision is not None and (
                (latest.revision if latest is not None else -1) != expected_previous_revision
            ):
                raise PlannerWorkspaceConflictError("planner workspace compare-and-swap failed")
            row = PlannerWorkspace(
                id=write.workspace_id,
                trip_id=write.trip_id,
                turn_id=write.turn_id,
                generation_id=write.generation_id,
                confirmed_task_book_id=write.confirmed_task_book_id,
                confirmed_task_book_version=write.confirmed_task_book_version,
                confirmed_task_book_hash=write.confirmed_task_book_hash,
                base_state_version=write.base_state_version,
                revision=write.revision,
                checkpoint_version=write.checkpoint_version,
                payload=dict(write.payload),
                content_hash=content_hash,
                status=write.status,
            )
            session.add(row)
            await session.flush()
            return _workspace_record(row)

    async def load_latest_planner_workspace(
        self,
        owner_user_id: UUID,
        trip_id: UUID,
        generation_id: UUID | None = None,
    ) -> PlannerWorkspaceRecord | None:
        async with self._session_factory() as session:
            query = (
                select(PlannerWorkspace)
                .join(Trip, Trip.id == PlannerWorkspace.trip_id)
                .where(
                    PlannerWorkspace.trip_id == trip_id,
                    Trip.owner_user_id == owner_user_id,
                )
            )
            if generation_id is not None:
                query = query.where(PlannerWorkspace.generation_id == generation_id)
            else:
                query = query.where(
                    PlannerWorkspace.checkpoint_version == V4_PLANNER_CHECKPOINT_VERSION
                )
            row = await session.scalar(
                query.order_by(
                    PlannerWorkspace.created_at.desc(), PlannerWorkspace.revision.desc()
                ).limit(1)
            )
            if row is not None:
                if canonical_json_hash(row.payload) != row.content_hash:
                    raise CheckpointPayloadError(
                        "workspace content hash does not match persisted content"
                    )
                _validate_v4_workspace_write(
                    PlannerWorkspaceWrite(
                        workspace_id=row.id,
                        trip_id=row.trip_id,
                        turn_id=row.turn_id,
                        generation_id=row.generation_id,
                        confirmed_task_book_id=row.confirmed_task_book_id,
                        confirmed_task_book_version=row.confirmed_task_book_version,
                        confirmed_task_book_hash=row.confirmed_task_book_hash,
                        base_state_version=row.base_state_version,
                        revision=row.revision,
                        checkpoint_version=row.checkpoint_version,
                        payload=row.payload,
                        status=row.status,
                    )
                )
            return _workspace_record(row) if row is not None else None

    async def planner_turn_status(self, owner_user_id: UUID, trip_id: UUID, turn_id: UUID) -> str:
        async with self._session_factory() as session:
            status = await session.scalar(
                select(AgentTurn.status)
                .join(Trip, Trip.id == AgentTurn.trip_id)
                .where(
                    AgentTurn.id == turn_id,
                    AgentTurn.trip_id == trip_id,
                    Trip.owner_user_id == owner_user_id,
                )
            )
            if status is None:
                raise CheckpointNotFoundError("Planner turn was not found")
            return str(status)

    async def cancel_v4_planner_workspace(
        self, owner_user_id: UUID, trip_id: UUID, generation_id: UUID
    ) -> PlannerWorkspaceRecord:
        from backend.agent.planner.workspace import advance
        from backend.contracts.v4.enums import InteractionStatus, PlannerStatus
        from backend.contracts.v4.planner_workspace import PlannerWorkspaceState

        async with self._session_factory() as session, session.begin():
            query = (
                select(PlannerWorkspace)
                .where(
                    PlannerWorkspace.trip_id == trip_id,
                    PlannerWorkspace.generation_id == generation_id,
                    PlannerWorkspace.checkpoint_version == V4_PLANNER_CHECKPOINT_VERSION,
                )
                .order_by(PlannerWorkspace.revision.desc())
                .limit(1)
            )
            initial = await session.scalar(query)
            if initial is None:
                raise CheckpointNotFoundError("Planner workspace not found")
            turn = await _require_live_turn(
                session,
                owner_user_id,
                trip_id,
                initial.turn_id,
                base_state_version=initial.base_state_version,
            )
            latest = await session.scalar(query.with_for_update())
            assert latest is not None
            workspace = PlannerWorkspaceState.model_validate(latest.payload)
            stopped = advance(
                workspace,
                status=PlannerStatus.CANCELLED,
                active_interaction=workspace.active_interaction.model_copy(
                    update={"status": InteractionStatus.SUPERSEDED}
                )
                if workspace.active_interaction
                else None,
            )
            row = PlannerWorkspace(
                id=uuid4(),
                trip_id=trip_id,
                turn_id=turn.id,
                generation_id=generation_id,
                confirmed_task_book_id=latest.confirmed_task_book_id,
                confirmed_task_book_version=latest.confirmed_task_book_version,
                confirmed_task_book_hash=latest.confirmed_task_book_hash,
                base_state_version=latest.base_state_version,
                revision=stopped.workspace_revision,
                checkpoint_version=V4_PLANNER_CHECKPOINT_VERSION,
                payload=stopped.model_dump(mode="json"),
                content_hash=canonical_json_hash(stopped.model_dump(mode="json")),
                status="stale",
            )
            session.add(row)
            turn.status = "cancelled"
            turn.failure_code = "cancelled_by_user"
            await session.flush()
            return _workspace_record(row)


def _validate_v4_workspace_write(write: PlannerWorkspaceWrite) -> None:
    from backend.contracts.v4.planner_workspace import PlannerWorkspaceState

    is_v4 = write.checkpoint_version.startswith("v4") or "generation_id" in write.payload
    if not is_v4:
        return  # Historical workspace-1 is not served by the V4 runtime.
    if write.checkpoint_version != V4_PLANNER_CHECKPOINT_VERSION:
        raise CheckpointPayloadError("V4 Planner requires its frozen checkpoint version")
    try:
        workspace = PlannerWorkspaceState.model_validate(write.payload)
    except ValidationError as error:
        raise CheckpointPayloadError("V4 Planner workspace failed contract validation") from error
    if (
        workspace.trip_id != str(write.trip_id)
        or workspace.generation_id != str(write.generation_id)
        or workspace.based_on_task_book_id != str(write.confirmed_task_book_id)
        or workspace.based_on_task_book_version != write.confirmed_task_book_version
        or workspace.workspace_revision != write.revision
    ):
        raise CheckpointPayloadError("V4 Planner workspace metadata does not match typed content")
    expected_status = {
        "planning": "working",
        "awaiting_user": "awaiting_user",
        "draft_ready": "completed",
        "ready_to_publish": "completed",
        "failed": "stale",
        "cancelled": "stale",
        "stale": "stale",
    }[workspace.status.value]
    if write.status != expected_status:
        raise CheckpointPayloadError("V4 Planner status does not match typed content")


async def _validate_current_task_book(session: AsyncSession, write: PlannerWorkspaceWrite) -> None:
    from backend.agent.planner.workspace import confirmed_task_book
    from backend.contracts.v4.state import V4TripStateEnvelope

    snapshot = await session.scalar(
        select(TripSnapshot).where(
            TripSnapshot.trip_id == write.trip_id,
            TripSnapshot.state_version == write.base_state_version,
            TripSnapshot.snapshot_kind == "v4",
        )
    )
    try:
        if snapshot is None:
            raise ValueError("no committed V4 state")
        state = V4TripStateEnvelope.model_validate(
            snapshot.snapshot, context={"restore_historical_semantic_state": True}
        )
        book = confirmed_task_book(state)
        if (
            book.task_book_id != str(write.confirmed_task_book_id)
            or book.version != write.confirmed_task_book_version
            or canonical_json_hash(book.model_dump(mode="json")) != write.confirmed_task_book_hash
            or book.based_on_state_version
            != write.payload["candidate_pool"]["scope"]["task_book_state_version"]
        ):
            raise ValueError("confirmed task book changed")
    except (ValueError, KeyError) as error:
        raise CheckpointStaleError(
            "Planner requires the current immutable confirmed task book"
        ) from error


async def _require_live_turn(
    session: AsyncSession,
    owner_user_id: UUID,
    trip_id: UUID,
    turn_id: UUID,
    *,
    base_state_version: int,
) -> AgentTurn:
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
    if turn is None:
        raise CheckpointNotFoundError("turn was not found")
    trip_version = await session.scalar(select(Trip.state_version).where(Trip.id == trip_id))
    if (
        turn.status not in {"accepted", "running"}
        or turn.base_state_version != base_state_version
        or trip_version != base_state_version
    ):
        raise CheckpointStaleError("turn no longer matches the current trip version")
    return turn


def _validate_checkpoint_payload(payload: dict[str, Any]) -> None:
    def visit(value: Any) -> None:
        if isinstance(value, dict):
            forbidden = _FORBIDDEN_RUNTIME_KEYS.intersection(value)
            if forbidden:
                names = ", ".join(sorted(forbidden))
                raise CheckpointPayloadError(
                    f"runtime dependencies cannot be checkpointed: {names}"
                )
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list | tuple):
            for nested in value:
                visit(nested)

    visit(payload)
    try:
        canonical_json_hash(payload)
    except (TypeError, ValueError) as exc:
        raise CheckpointPayloadError("checkpoint payload must be JSON serializable") from exc


def _checkpoint_record(row: AgentCheckpoint) -> CheckpointRecord:
    return CheckpointRecord(
        checkpoint_row_id=row.id,
        trip_id=row.trip_id,
        turn_id=row.turn_id,
        checkpoint_namespace=row.checkpoint_namespace,
        checkpoint_id=row.checkpoint_id,
        checkpoint_version=row.checkpoint_version,
        base_state_version=row.base_state_version,
        payload=dict(row.payload),
        content_hash=row.content_hash,
        status=row.status,
        expires_at=row.expires_at,
    )


def _workspace_record(row: PlannerWorkspace) -> PlannerWorkspaceRecord:
    return PlannerWorkspaceRecord(
        workspace_id=row.id,
        trip_id=row.trip_id,
        turn_id=row.turn_id,
        generation_id=row.generation_id,
        confirmed_task_book_id=row.confirmed_task_book_id,
        confirmed_task_book_version=row.confirmed_task_book_version,
        confirmed_task_book_hash=row.confirmed_task_book_hash,
        base_state_version=row.base_state_version,
        revision=row.revision,
        checkpoint_version=row.checkpoint_version,
        payload=dict(row.payload),
        content_hash=row.content_hash,
        status=row.status,
    )
