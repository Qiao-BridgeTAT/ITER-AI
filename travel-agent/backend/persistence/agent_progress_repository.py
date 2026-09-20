"""Append-before-emit public progress with owner and live-turn barriers."""

from datetime import UTC, datetime
from typing import Literal, cast
from uuid import UUID, uuid4

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from backend.contracts.v4.planner_react import AgentProgressEntry
from backend.persistence.checkpoint_repository import CheckpointStaleError, _require_live_turn
from backend.persistence.models import AgentProgress, Trip


class AgentProgressRepository:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self.sessions = session_factory

    async def append(
        self,
        owner: UUID,
        trip_id: UUID,
        turn_id: UUID,
        generation_id: UUID,
        base_version: int,
        source: Literal["planner", "reviewer", "tool", "runtime"],
        text: str,
    ) -> AgentProgressEntry:
        async with self.sessions() as session, session.begin():
            # Match checkpoint lock ordering. The live-turn join locks both turn and trip.
            turn = await _require_live_turn(
                session, owner, trip_id, turn_id, base_state_version=base_version
            )
            if turn.generation_id != generation_id:
                raise CheckpointStaleError("progress generation is no longer current")
            index = await session.scalar(
                select(func.max(AgentProgress.progress_index)).where(
                    AgentProgress.generation_id == generation_id
                )
            )
            entry = AgentProgressEntry(
                event_id=str(uuid4()),
                trip_id=str(trip_id),
                turn_id=str(turn_id),
                generation_id=str(generation_id),
                progress_index=(index or 0) + 1,
                source=source,
                text=text,
                emitted_at=datetime.now(UTC),
            )
            session.add(
                AgentProgress(
                    id=UUID(entry.event_id),
                    trip_id=trip_id,
                    turn_id=turn_id,
                    generation_id=generation_id,
                    progress_index=entry.progress_index,
                    source=source,
                    text=text,
                    emitted_at=entry.emitted_at,
                )
            )
        return entry

    async def list(self, owner: UUID, trip_id: UUID) -> tuple[AgentProgressEntry, ...]:
        async with self.sessions() as session:
            rows = (
                await session.scalars(
                    select(AgentProgress)
                    .join(Trip, Trip.id == AgentProgress.trip_id)
                    .where(Trip.id == trip_id, Trip.owner_user_id == owner)
                    .order_by(AgentProgress.emitted_at, AgentProgress.progress_index)
                )
            ).all()
            return tuple(
                AgentProgressEntry(
                    event_id=str(r.id),
                    trip_id=str(r.trip_id),
                    turn_id=str(r.turn_id),
                    generation_id=str(r.generation_id),
                    progress_index=r.progress_index,
                    source=cast(Literal["planner", "reviewer", "tool", "runtime"], r.source),
                    text=r.text,
                    emitted_at=r.emitted_at.replace(tzinfo=UTC)
                    if r.emitted_at.tzinfo is None
                    else r.emitted_at,
                )
                for r in rows
            )
