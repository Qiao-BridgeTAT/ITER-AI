"""Private Prepare facts reuse the existing durable observation table."""

from datetime import UTC, datetime, timedelta
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from backend.contracts.prepared_evidence import PreparedCandidateEvidence
from backend.contracts.v4.planner_evidence import PlannerCandidateOrigin
from backend.persistence.models import AgentTurn, ToolObservationRecord, Trip
from backend.persistence.outbox_repository import canonical_json_hash

PREPARED_FACT_TOOL = "prepare_candidate_facts_v1"


class PreparedEvidenceRepository:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self.sessions = sessions

    async def save(self, turn_id: UUID, value: PreparedCandidateEvidence) -> None:
        payload = value.model_dump(mode="json")
        identity = uuid5(
            NAMESPACE_URL,
            f"{PREPARED_FACT_TOOL}:{value.trip_id}:{value.source_attachment_id}:"
            f"{value.place.canonical_entity_id}",
        )
        async with self.sessions() as session, session.begin():
            turn = await session.scalar(
                select(AgentTurn)
                .where(AgentTurn.id == turn_id, AgentTurn.trip_id == value.trip_id)
                .with_for_update()
            )
            if turn is None or turn.status not in {"accepted", "running", "committed"}:
                return
            row = await session.get(ToolObservationRecord, identity)
            if row is None:
                row = ToolObservationRecord(
                    observation_id=identity,
                    trip_id=value.trip_id,
                    turn_id=turn_id,
                    tool_name=PREPARED_FACT_TOOL,
                    provider="amap+flyai",
                    request_hash=canonical_json_hash(
                        {
                            "attachment": str(value.source_attachment_id),
                            "entity": value.place.canonical_entity_id,
                            "dates": [d.isoformat() for d in value.service_dates],
                        }
                    ),
                    status="partial",
                    source_refs=[f"provider:amap:{value.place.provider_entity_id}"],
                    safe_payload=payload,
                    content_hash=canonical_json_hash(payload),
                    observed_at=value.place.observed_at,
                    expires_at=value.place.observed_at + timedelta(hours=24),
                )
                session.add(row)
            else:
                row.safe_payload = payload
                row.content_hash = canonical_json_hash(payload)

    async def load(
        self,
        owner_id: UUID,
        trip_id: UUID,
        *,
        origins: tuple[PlannerCandidateOrigin, ...],
        based_on_state_version: int,
        now: datetime,
    ) -> tuple[PreparedCandidateEvidence, ...]:
        wanted = {
            (
                o.canonical_entity_id,
                o.provider_entity_id,
                o.source_attachment_id,
                o.dependency_fingerprint,
            )
            for o in origins
            if o.source_attachment_id and o.dependency_fingerprint
        }
        if not wanted:
            return ()
        async with self.sessions() as session:
            rows = (
                await session.scalars(
                    select(ToolObservationRecord)
                    .join(Trip, Trip.id == ToolObservationRecord.trip_id)
                    .join(AgentTurn, AgentTurn.id == ToolObservationRecord.turn_id)
                    .where(
                        Trip.id == trip_id,
                        Trip.owner_user_id == owner_id,
                        ToolObservationRecord.tool_name == PREPARED_FACT_TOOL,
                        AgentTurn.status == "committed",
                        AgentTurn.committed_state_version <= based_on_state_version,
                    )
                )
            ).all()
        values = []
        for row in rows:
            if canonical_json_hash(row.safe_payload) != row.content_hash:
                continue
            try:
                value = PreparedCandidateEvidence.model_validate(row.safe_payload)
            except ValidationError:
                continue
            place = value.place
            if (
                value.trip_id != trip_id
                or (
                    place.canonical_entity_id,
                    place.provider_entity_id,
                    str(value.source_attachment_id),
                    value.dependency_fingerprint,
                )
                not in wanted
                or not timedelta(0) <= now.astimezone(UTC) - place.observed_at < timedelta(hours=24)
            ):
                continue
            values.append(value)
        return tuple(values)
