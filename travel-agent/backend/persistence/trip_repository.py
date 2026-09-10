"""Transactional trip persistence with optimistic state-version control."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import UUID, uuid4

from sqlalchemy import delete, func, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from backend.contracts.enums import OwnerType
from backend.contracts.state import TripState
from backend.persistence.models import PlanningRun, Trip, TripSnapshot, TripVersion


class TripPersistenceError(RuntimeError):
    """Base class for stable-trip persistence failures."""


class TripNotFoundError(TripPersistenceError):
    pass


class ConcurrentStateVersionError(TripPersistenceError):
    pass


class InvalidStableWriteError(TripPersistenceError):
    pass


@dataclass(frozen=True)
class PlanVersionWrite:
    version_id: UUID
    status: Literal["draft", "confirmed"]
    parent_version_id: UUID | None = None
    confirmed_at: datetime | None = None
    publication_key: str | None = None
    generation_id: UUID | None = None


@dataclass(frozen=True)
class PlanningRunWrite:
    generation_id: UUID
    status: Literal["queued", "running", "succeeded", "failed", "cancelled"]
    input_summary: dict[str, Any]
    result_summary: dict[str, Any] | None
    config_versions: dict[str, Any]
    failure_code: str | None = None
    finished_at: datetime | None = None


@dataclass(frozen=True)
class TripSummaryRecord:
    trip_id: UUID
    phase: str
    title: str
    updated_at: datetime


@dataclass(frozen=True)
class PlanPublicationCommit:
    state: TripState
    idempotent_replay: bool


class TripRepository:
    """Owns complete transactions for durable user trips."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def create_user_trip(
        self,
        owner_user_id: UUID,
        state: TripState,
        *,
        title: str = "未命名旅行",
    ) -> TripState:
        self._validate_user_state(owner_user_id, state)
        if state.state_version != 0:
            raise InvalidStableWriteError("new trips must start at state version zero")
        if state.current_plan_version_id is not None or state.base_confirmed_version_id is not None:
            raise InvalidStableWriteError("new trips cannot reference plan versions")

        snapshot = state.model_dump(mode="json")
        async with self._session_factory() as session, session.begin():
            session.add(
                Trip(
                    id=state.trip_id,
                    owner_user_id=owner_user_id,
                    phase=state.phase.value,
                    state_version=state.state_version,
                    schema_version=state.schema_version,
                    title=title,
                )
            )
            session.add(
                TripSnapshot(
                    id=uuid4(),
                    trip_id=state.trip_id,
                    state_version=state.state_version,
                    schema_version=state.schema_version,
                    snapshot=snapshot,
                )
            )
        return state

    async def load_user_trip(self, owner_user_id: UUID, trip_id: UUID) -> TripState:
        async with self._session_factory() as session:
            snapshot = await session.scalar(
                select(TripSnapshot)
                .join(Trip, Trip.id == TripSnapshot.trip_id)
                .where(Trip.id == trip_id, Trip.owner_user_id == owner_user_id)
                .order_by(TripSnapshot.state_version.desc())
                .limit(1)
            )
        if snapshot is None:
            raise TripNotFoundError("trip was not found")
        return TripState.model_validate(snapshot.snapshot)

    async def list_user_trips(self, owner_user_id: UUID) -> list[TripSummaryRecord]:
        async with self._session_factory() as session:
            trips = (
                await session.scalars(
                    select(Trip)
                    .where(Trip.owner_user_id == owner_user_id)
                    .order_by(Trip.updated_at.desc(), Trip.id)
                )
            ).all()
        return [
            TripSummaryRecord(
                trip_id=trip.id,
                phase=trip.phase,
                title=trip.title or "未命名旅行",
                updated_at=trip.updated_at,
            )
            for trip in trips
        ]

    async def delete_user_trip(self, owner_user_id: UUID, trip_id: UUID) -> bool:
        async with self._session_factory() as session, session.begin():
            exists = await session.scalar(
                select(Trip.id).where(Trip.id == trip_id, Trip.owner_user_id == owner_user_id)
            )
            if exists is None:
                return False
            await session.execute(
                update(Trip)
                .where(Trip.id == trip_id)
                .values(current_plan_version_id=None, base_confirmed_version_id=None)
            )
            await session.execute(delete(Trip).where(Trip.id == trip_id))
        return True

    async def commit_stable_state(
        self,
        owner_user_id: UUID,
        state: TripState,
        *,
        expected_state_version: int,
        plan_version: PlanVersionWrite | None = None,
        confirmed_version_id: UUID | None = None,
        planning_run: PlanningRunWrite | None = None,
    ) -> TripState:
        self._validate_stable_write(
            owner_user_id,
            state,
            expected_state_version=expected_state_version,
            plan_version=plan_version,
        )
        if confirmed_version_id is not None and (
            state.current_plan_version_id != confirmed_version_id
            or state.phase.value != "confirmed"
        ):
            raise InvalidStableWriteError(
                "confirmed version must be the current version of a confirmed state"
            )
        snapshot_payload = state.model_dump(mode="json")

        async with self._session_factory() as session, session.begin():
            claimed = await session.execute(
                update(Trip)
                .where(
                    Trip.id == state.trip_id,
                    Trip.owner_user_id == owner_user_id,
                    Trip.state_version == expected_state_version,
                )
                .values(
                    phase=state.phase.value,
                    state_version=state.state_version,
                    schema_version=state.schema_version,
                    current_plan_version_id=state.current_plan_version_id,
                    base_confirmed_version_id=state.base_confirmed_version_id,
                )
            )
            if getattr(claimed, "rowcount", 0) != 1:
                exists = await session.scalar(
                    select(Trip.id).where(
                        Trip.id == state.trip_id,
                        Trip.owner_user_id == owner_user_id,
                    )
                )
                if exists is None:
                    raise TripNotFoundError("trip was not found")
                raise ConcurrentStateVersionError("trip state version no longer matches")

            if plan_version is not None:
                latest_number = await session.scalar(
                    select(func.max(TripVersion.version_number)).where(
                        TripVersion.trip_id == state.trip_id
                    )
                )
                session.add(
                    TripVersion(
                        id=plan_version.version_id,
                        trip_id=state.trip_id,
                        parent_version_id=plan_version.parent_version_id,
                        publication_key=plan_version.publication_key,
                        generation_id=plan_version.generation_id,
                        version_number=(latest_number or 0) + 1,
                        state_version=state.state_version,
                        status=plan_version.status,
                        snapshot=snapshot_payload,
                        schema_version=state.schema_version,
                        confirmed_at=plan_version.confirmed_at,
                    )
                )

            if confirmed_version_id is not None:
                confirmed = await session.execute(
                    update(TripVersion)
                    .where(
                        TripVersion.id == confirmed_version_id,
                        TripVersion.trip_id == state.trip_id,
                        TripVersion.status == "draft",
                    )
                    .values(status="confirmed", confirmed_at=datetime.now(UTC))
                )
                if getattr(confirmed, "rowcount", 0) != 1:
                    raise InvalidStableWriteError(
                        "confirmed plan version does not belong to the trip"
                    )

            session.add(
                TripSnapshot(
                    id=uuid4(),
                    trip_id=state.trip_id,
                    plan_version_id=state.current_plan_version_id,
                    state_version=state.state_version,
                    schema_version=state.schema_version,
                    snapshot=snapshot_payload,
                )
            )

            if planning_run is not None:
                session.add(
                    PlanningRun(
                        id=uuid4(),
                        trip_id=state.trip_id,
                        plan_version_id=(
                            plan_version.version_id if plan_version is not None else None
                        ),
                        generation_id=planning_run.generation_id,
                        status=planning_run.status,
                        input_summary=planning_run.input_summary,
                        result_summary=planning_run.result_summary,
                        config_versions=planning_run.config_versions,
                        failure_code=planning_run.failure_code,
                        finished_at=planning_run.finished_at,
                    )
                )
            await session.flush()
        return state

    async def publish_plan_version(
        self,
        owner_user_id: UUID,
        state: TripState,
        *,
        expected_state_version: int,
        plan_version: PlanVersionWrite,
        planning_run: PlanningRunWrite,
    ) -> PlanPublicationCommit:
        """Persist one complete plan and its current TripState pointer atomically.

        The publication key and generation ID are durable idempotency identities. A retry
        returns the already committed snapshot; a collision with different identities is
        rejected instead of silently selecting either plan.
        """

        if (
            plan_version.status != "draft"
            or plan_version.publication_key is None
            or plan_version.generation_id is None
            or plan_version.generation_id != planning_run.generation_id
        ):
            raise InvalidStableWriteError(
                "atomic publication requires one draft publication key and generation"
            )
        existing = await self._load_idempotent_publication(
            owner_user_id,
            state.trip_id,
            plan_version,
        )
        if existing is not None:
            return PlanPublicationCommit(state=existing, idempotent_replay=True)
        try:
            committed = await self.commit_stable_state(
                owner_user_id,
                state,
                expected_state_version=expected_state_version,
                plan_version=plan_version,
                planning_run=planning_run,
            )
        except (ConcurrentStateVersionError, IntegrityError):
            existing = await self._load_idempotent_publication(
                owner_user_id,
                state.trip_id,
                plan_version,
            )
            if existing is None:
                raise
            return PlanPublicationCommit(state=existing, idempotent_replay=True)
        return PlanPublicationCommit(state=committed, idempotent_replay=False)

    async def _load_idempotent_publication(
        self,
        owner_user_id: UUID,
        trip_id: UUID,
        plan_version: PlanVersionWrite,
    ) -> TripState | None:
        async with self._session_factory() as session:
            version = await session.scalar(
                select(TripVersion)
                .join(Trip, Trip.id == TripVersion.trip_id)
                .where(
                    Trip.owner_user_id == owner_user_id,
                    TripVersion.trip_id == trip_id,
                    or_(
                        TripVersion.publication_key == plan_version.publication_key,
                        TripVersion.generation_id == plan_version.generation_id,
                    ),
                )
            )
        if version is None:
            return None
        if (
            version.id != plan_version.version_id
            or version.publication_key != plan_version.publication_key
            or version.generation_id != plan_version.generation_id
        ):
            raise InvalidStableWriteError("plan publication idempotency identity collided")
        return TripState.model_validate(version.snapshot)

    @staticmethod
    def _validate_user_state(owner_user_id: UUID, state: TripState) -> None:
        if state.owner_type is not OwnerType.USER or state.owner_id != str(owner_user_id):
            raise InvalidStableWriteError("durable trip state must match its user owner")

    @classmethod
    def _validate_stable_write(
        cls,
        owner_user_id: UUID,
        state: TripState,
        *,
        expected_state_version: int,
        plan_version: PlanVersionWrite | None,
    ) -> None:
        cls._validate_user_state(owner_user_id, state)
        if state.state_version != expected_state_version + 1:
            raise InvalidStableWriteError("stable writes must advance state_version by exactly one")
        if plan_version is not None and state.current_plan_version_id != plan_version.version_id:
            raise InvalidStableWriteError("visible version must equal current_plan_version_id")
        if plan_version is not None:
            has_publication_key = plan_version.publication_key is not None
            has_generation_id = plan_version.generation_id is not None
            if has_publication_key != has_generation_id:
                raise InvalidStableWriteError(
                    "publication key and generation ID must be written together"
                )
            if plan_version.status == "confirmed" and plan_version.confirmed_at is None:
                raise InvalidStableWriteError("confirmed versions require confirmed_at")
            if plan_version.status == "draft" and plan_version.confirmed_at is not None:
                raise InvalidStableWriteError("draft versions cannot have confirmed_at")
