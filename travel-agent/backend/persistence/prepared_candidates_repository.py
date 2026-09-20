"""Durable, owner-scoped Prepare candidate pools in the observation store."""

from datetime import UTC, datetime, timedelta
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from backend.contracts.prepared_candidates import PreparedPlanningPool
from backend.persistence.models import AgentTurn, ToolObservationRecord, Trip
from backend.persistence.outbox_repository import canonical_json_hash

SEED_TOOL = "prepare_planning_seed_v1"
POOL_TOOL = "prepare_planning_pool_v1"


class PreparedCandidatesRepository:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self.sessions = sessions

    async def source_turn(self, owner: UUID, trip: UUID, version: int) -> tuple[UUID, int]:
        async with self.sessions() as session:
            row = await session.scalar(
                select(AgentTurn)
                .join(Trip, Trip.id == AgentTurn.trip_id)
                .where(
                    Trip.id == trip,
                    Trip.owner_user_id == owner,
                    AgentTurn.status == "committed",
                    AgentTurn.committed_state_version <= version,
                )
                .order_by(AgentTurn.committed_state_version.desc())
                .limit(1)
            )
            if row is None:
                raise ValueError("candidate pool requires a committed owned source turn")
            assert row.committed_state_version is not None
            return row.id, row.committed_state_version

    async def save(self, owner: UUID, value: PreparedPlanningPool) -> None:
        tool = SEED_TOOL if value.status == "seed" else POOL_TOOL
        fingerprint = (
            value.context_fingerprint if tool == SEED_TOOL else value.selection_fingerprint
        )
        identity = uuid5(NAMESPACE_URL, f"{tool}:{value.trip_id}:{value.domain}:{fingerprint}")
        payload = value.model_dump(mode="json")
        async with self.sessions() as session, session.begin():
            turn = await session.scalar(
                select(AgentTurn)
                .join(Trip, Trip.id == AgentTurn.trip_id)
                .where(
                    Trip.id == value.trip_id,
                    Trip.owner_user_id == owner,
                    AgentTurn.id == value.source_turn_id,
                    AgentTurn.status == "committed",
                    AgentTurn.committed_state_version == value.source_state_version,
                )
                .with_for_update()
            )
            if turn is None:
                raise ValueError("candidate pool source is not committed for this owner")
            row = await session.get(ToolObservationRecord, identity)
            if row is not None:
                old_version = row.safe_payload.get("source_state_version", -1)
                if old_version > value.source_state_version:
                    return
                row.turn_id = value.source_turn_id
                row.safe_payload, row.content_hash = payload, canonical_json_hash(payload)
                row.status, row.observed_at = value.status, value.updated_at
                row.expires_at = value.updated_at + timedelta(hours=24)
            else:
                session.add(
                    ToolObservationRecord(
                        observation_id=identity,
                        trip_id=value.trip_id,
                        turn_id=value.source_turn_id,
                        tool_name=tool,
                        provider="amap+qwen",
                        request_hash=fingerprint,
                        status=value.status,
                        safe_payload=payload,
                        source_refs=[f"turn:{value.source_turn_id}"],
                        content_hash=canonical_json_hash(payload),
                        observed_at=value.updated_at,
                        expires_at=value.updated_at + timedelta(hours=24),
                    )
                )

    async def load(
        self,
        owner: UUID,
        trip: UUID,
        domain: str,
        context_fingerprint: str,
        *,
        based_on_state_version: int,
        now: datetime,
        seed: bool = False,
        selection_fingerprint: str | None = None,
    ) -> PreparedPlanningPool | None:
        async with self.sessions() as session:
            rows = (
                await session.scalars(
                    select(ToolObservationRecord)
                    .join(Trip, Trip.id == ToolObservationRecord.trip_id)
                    .join(AgentTurn, AgentTurn.id == ToolObservationRecord.turn_id)
                    .where(
                        Trip.id == trip,
                        Trip.owner_user_id == owner,
                        ToolObservationRecord.tool_name == (SEED_TOOL if seed else POOL_TOOL),
                        AgentTurn.status == "committed",
                        AgentTurn.committed_state_version <= based_on_state_version,
                    )
                    .order_by(
                        AgentTurn.committed_state_version.desc(),
                        ToolObservationRecord.observed_at.desc(),
                    )
                )
            ).all()
        for row in rows:
            if canonical_json_hash(row.safe_payload) != row.content_hash:
                continue
            try:
                value = PreparedPlanningPool.model_validate(row.safe_payload)
            except ValidationError:
                continue
            if (
                value.trip_id != trip
                or value.domain != domain
                or value.context_fingerprint != context_fingerprint
                or value.source_turn_id != row.turn_id
                or value.source_state_version > based_on_state_version
                or (
                    selection_fingerprint is not None
                    and value.selection_fingerprint != selection_fingerprint
                )
                or not timedelta(0) <= now.astimezone(UTC) - value.updated_at < timedelta(hours=24)
            ):
                continue
            return value
        return None
