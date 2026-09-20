"""Private, source-bound candidate facts; never a discovery-card payload."""

from datetime import date
from typing import Literal
from uuid import UUID

from pydantic import model_validator

from backend.contracts.v4.base import V4ContractModel
from backend.contracts.v4.planner_evidence import (
    PlannerHoursEvidence,
    PlannerPlaceEvidence,
    PlannerTicketEvidence,
)


class PreparedCandidateEvidence(V4ContractModel):
    version: Literal[1] = 1
    trip_id: UUID
    source_attachment_id: str
    dependency_fingerprint: str
    service_dates: tuple[date, ...]
    place: PlannerPlaceEvidence
    hours: PlannerHoursEvidence | None = None
    tickets: tuple[PlannerTicketEvidence, ...] = ()

    @model_validator(mode="after")
    def evidence_keeps_identity_and_dates(self) -> "PreparedCandidateEvidence":
        if self.hours and (
            self.hours.canonical_entity_id != self.place.canonical_entity_id
            or self.hours.provider_entity_id != self.place.provider_entity_id
            or not {d.service_date for d in self.hours.days} <= set(self.service_dates)
        ):
            raise ValueError("prepared hours must bind the same entity and dates")
        if any(
            t.canonical_entity_id != self.place.canonical_entity_id
            or t.service_date not in self.service_dates
            for t in self.tickets
        ):
            raise ValueError("prepared tickets must bind the same entity and dates")
        return self
