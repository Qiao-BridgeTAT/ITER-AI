"""Normalized dining recall and durable admission state; no raw Provider payloads."""

from typing import Annotated, Literal

from pydantic import Field, model_validator

from backend.contracts.v4.base import DisplayText, Identifier, V4ContractModel, require_unique
from backend.contracts.v4.planner_evidence import PlannerPlaceEvidence


class ModelDiningSearchPlan(V4ContractModel):
    city_representative_queries: tuple[DisplayText, ...] = Field(min_length=4, max_length=4)
    exploration_queries: tuple[DisplayText, ...] = Field(min_length=4, max_length=4)


class ModelDiningReview(V4ContractModel):
    candidate_keys: tuple[Identifier, ...] = Field(max_length=20)


class ModelDiningSlotChoice(V4ContractModel):
    candidate_key: Annotated[Identifier, Field(pattern=r"^c[1-9][0-9]*$")] | None


class DiningSearchDirection(V4ContractModel):
    query_key: Identifier
    query: DisplayText
    kind: Literal["representative", "personalized"]
    limit: int = Field(ge=2, le=5, strict=True)

    @model_validator(mode="after")
    def quota_matches_kind(self) -> "DiningSearchDirection":
        if self.limit != (2 if self.kind == "representative" else 5):
            raise ValueError("dining direction quota must be 2 or 5 for its kind")
        return self


class DiningQueryHit(V4ContractModel):
    query_key: Identifier
    page: int = Field(ge=1, le=2, strict=True)
    provider_rank: int = Field(ge=1, le=20, strict=True)
    provider_entity_id: Identifier
    fact_reference_id: Identifier


class DiningRecallCandidate(V4ContractModel):
    candidate_key: Identifier
    allocation_query_key: Identifier | None = None
    place: PlannerPlaceEvidence
    query_hits: tuple[DiningQueryHit, ...] = Field(min_length=1)
    source_evidence: tuple[PlannerPlaceEvidence, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def facts_are_bound_to_sources(self) -> "DiningRecallCandidate":
        if self.place.entity_kind != "restaurant" or any(
            source.entity_kind != "restaurant" for source in self.source_evidence
        ):
            raise ValueError("dining candidates must be restaurants")
        sources = {(s.provider_entity_id, s.fact_reference_id) for s in self.source_evidence}
        if (self.place.provider_entity_id, self.place.fact_reference_id) not in sources or any(
            (hit.provider_entity_id, hit.fact_reference_id) not in sources
            for hit in self.query_hits
        ):
            raise ValueError("dining query provenance must bind its exact normalized source")
        return self


class DiningSearchAttempt(V4ContractModel):
    query_key: Identifier
    page: int = Field(ge=1, le=2, strict=True)
    status: Literal["complete", "partial", "empty", "unavailable", "budget_exhausted"]
    returned_count: int = Field(default=0, ge=0, strict=True)
    admitted_count: int = Field(default=0, ge=0, strict=True)
    failure_code: Identifier | None = None


class PlannerDiningState(V4ContractModel):
    status: Literal["pending", "skipped", "recalled", "reviewed", "partial", "unavailable"] = (
        "pending"
    )
    initial_threshold: int = Field(default=0, ge=0, strict=True)
    inherited_canonical_ids: tuple[Identifier, ...] = ()
    admitted_canonical_ids: tuple[Identifier, ...] = ()
    queries: tuple[DiningSearchDirection, ...] = Field(default=(), max_length=8)
    candidates: tuple[DiningRecallCandidate, ...] = Field(default=(), max_length=28)
    inherited_hits: tuple[DiningRecallCandidate, ...] = ()
    search_attempts: tuple[DiningSearchAttempt, ...] = Field(default=(), max_length=12)
    search_plan_attempts: int = Field(default=0, ge=0, le=2, strict=True)
    review_attempts: int = Field(default=0, ge=0, le=2, strict=True)
    review_candidate_keys: tuple[Identifier, ...] = Field(default=(), max_length=20)
    failure_codes: tuple[Identifier, ...] = ()

    @model_validator(mode="after")
    def identities_and_budgets_are_consistent(self) -> "PlannerDiningState":
        for values, label in (
            (self.inherited_canonical_ids, "inherited dining identities"),
            (self.admitted_canonical_ids, "admitted dining identities"),
            (self.review_candidate_keys, "review dining keys"),
            ((q.query_key for q in self.queries), "dining query keys"),
            ((c.candidate_key for c in self.candidates), "dining candidate keys"),
            ((c.place.canonical_entity_id for c in self.candidates), "dining candidates"),
            (((a.query_key, a.page) for a in self.search_attempts), "dining query pages"),
        ):
            require_unique(values, label)
        keys = {c.candidate_key for c in self.candidates}
        if not set(self.review_candidate_keys) <= keys:
            raise ValueError("review can only select recalled dining keys")
        return self


V4_PLANNER_DINING_CONTRACTS = (
    ModelDiningSearchPlan,
    ModelDiningReview,
    ModelDiningSlotChoice,
    DiningSearchDirection,
    DiningQueryHit,
    DiningRecallCandidate,
    DiningSearchAttempt,
    PlannerDiningState,
)
