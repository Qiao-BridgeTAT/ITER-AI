"""Private candidate handoff, separate from user intent and displayed cards."""

from typing import Literal
from uuid import UUID

from pydantic import AwareDatetime, Field, model_validator

from backend.contracts.v4.base import V4ContractModel
from backend.contracts.v4.planner_evidence import PlannerCandidateOrigin, PlannerPlaceEvidence


class PreparedPlanningCandidate(V4ContractModel):
    origin: PlannerCandidateOrigin
    place: PlannerPlaceEvidence

    @model_validator(mode="after")
    def identity_matches(self) -> "PreparedPlanningCandidate":
        if (
            self.origin.canonical_entity_id != self.place.canonical_entity_id
            or self.origin.provider_entity_id != self.place.provider_entity_id
            or self.origin.entity_kind != self.place.entity_kind
        ):
            raise ValueError("candidate source must bind its verified entity and kind")
        return self


class PreparedPlanningPool(V4ContractModel):
    version: Literal[1] = 1
    trip_id: UUID
    domain: Literal["attraction", "dining"]
    context_fingerprint: str
    selection_fingerprint: str
    source_turn_id: UUID
    source_state_version: int = Field(ge=0)
    target_count: int = Field(ge=3, le=15)
    candidates: tuple[PreparedPlanningCandidate, ...] = ()
    status: Literal["seed", "building", "ready", "partial"] = "seed"
    model_rounds: int = Field(default=0, ge=0)
    failures: tuple[str, ...] = ()
    updated_at: AwareDatetime

    @property
    def missing_count(self) -> int:
        return max(0, self.target_count - len(self.candidates))

    @model_validator(mode="after")
    def distinct_verified_candidates(self) -> "PreparedPlanningPool":
        ids = [c.place.canonical_entity_id for c in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("prepared candidates must have distinct identities")
        kind = "attraction" if self.domain == "attraction" else "restaurant"
        if any(c.place.entity_kind.value != kind for c in self.candidates):
            raise ValueError("prepared pool cannot mix domains")
        if self.status == "ready" and self.missing_count:
            raise ValueError("ready pool cannot have a candidate deficit")
        return self
