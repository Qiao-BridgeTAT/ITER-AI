"""V3-31 contracts for deterministic candidate deduplication and ranking."""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import AwareDatetime, ConfigDict, Field, model_validator

from backend.contracts.base import ContractModel
from backend.contracts.candidate_recall import (
    CandidateRecallRequest,
    CandidateRecallResult,
    RecalledCandidate,
)
from backend.contracts.common import NonEmptyText, ShortText
from backend.contracts.enums import Confidence, DataAvailability


class ImmutableRankingModel(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RankingFactor(StrEnum):
    STRONG_DESIRE = "strong_desire"
    PREFERENCE_MATCH = "preference_match"
    SPATIAL_COST = "spatial_cost"
    TIME_COST = "time_cost"
    FACT_COMPLETENESS = "fact_completeness"
    SOURCE_TRUST = "source_trust"


class RankingSelectionStatus(StrEnum):
    RECOMMENDED = "recommended"
    RESERVE = "reserve"


class CandidateHardConflictCode(StrEnum):
    EXPLICITLY_EXCLUDED = "explicitly_excluded"
    CLOSED_ON_TRIP_DATE = "closed_on_trip_date"
    FIXED_SCHEDULE_CONFLICT = "fixed_schedule_conflict"
    SPECIAL_CONSTRAINT_VIOLATION = "special_constraint_violation"
    WRONG_CITY = "wrong_city"
    WRONG_DOMAIN = "wrong_domain"


class CandidateRankingPreferences(ImmutableRankingModel):
    pace_level: int = Field(default=3, ge=1, le=5, strict=True)
    classic_niche_level: int = Field(default=3, ge=1, le=5, strict=True)
    transit_taxi_level: int = Field(default=3, ge=1, le=5, strict=True)
    selected_theme_ids: tuple[NonEmptyText, ...] = ()

    @model_validator(mode="after")
    def theme_ids_are_unique(self) -> CandidateRankingPreferences:
        _unique(self.selected_theme_ids, "ranking preference theme IDs")
        return self


class CandidateCostSignal(ImmutableRankingModel):
    candidate_id: UUID
    distance_from_anchor_m: int | None = Field(default=None, ge=0, strict=True)
    travel_time_minutes: int | None = Field(default=None, ge=0, le=1_440, strict=True)
    visit_duration_minutes: int | None = Field(default=None, ge=1, le=1_440, strict=True)
    source_reference_ids: tuple[NonEmptyText, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def contains_at_least_one_cost(self) -> CandidateCostSignal:
        if (
            self.distance_from_anchor_m is None
            and self.travel_time_minutes is None
            and self.visit_duration_minutes is None
        ):
            raise ValueError("candidate cost signal requires at least one cost value")
        _unique(self.source_reference_ids, "candidate cost source references")
        return self


class CandidateHardConflict(ImmutableRankingModel):
    candidate_id: UUID
    code: CandidateHardConflictCode
    reason: ShortText
    source_reference_ids: tuple[NonEmptyText, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def source_references_are_unique(self) -> CandidateHardConflict:
        _unique(self.source_reference_ids, "hard-conflict source references")
        return self


class CandidateRankingRequest(ImmutableRankingModel):
    ranking_request_id: UUID
    recall_request: CandidateRecallRequest
    recall_result: CandidateRecallResult
    preferences: CandidateRankingPreferences = Field(default_factory=CandidateRankingPreferences)
    cost_signals: tuple[CandidateCostSignal, ...] = ()
    hard_conflicts: tuple[CandidateHardConflict, ...] = ()
    verified_fact_ids: tuple[NonEmptyText, ...] = ()
    recommendation_limit: int | None = Field(default=None, ge=1, le=14, strict=True)

    @model_validator(mode="after")
    def inputs_have_one_owner_and_known_candidate_references(self) -> CandidateRankingRequest:
        request = self.recall_request
        result = self.recall_result
        if (
            request.request_id != result.request_id
            or request.trip_id != result.trip_id
            or request.semantic_state_version != result.semantic_state_version
            or request.task_book_id != result.task_book_id
            or request.task_book_revision != result.task_book_revision
            or request.city_id != result.city_id
        ):
            raise ValueError("ranking recall request and result must share one planning boundary")
        candidate_ids = {candidate.candidate_id for candidate in result.candidates}
        budgeted_domains = {item.domain for item in request.budget.domains}
        if {candidate.domain for candidate in result.candidates} - budgeted_domains:
            raise ValueError("ranking candidates must belong to a budgeted recall domain")
        selected_theme_ids = set(self.preferences.selected_theme_ids)
        known_theme_ids = {theme.theme_id for theme in request.themes}
        if not selected_theme_ids <= known_theme_ids:
            raise ValueError("ranking preferences reference an unknown recall theme")
        _unique(self.verified_fact_ids, "verified ranking fact IDs")
        verified_fact_ids = set(self.verified_fact_ids)
        signal_ids = [signal.candidate_id for signal in self.cost_signals]
        if len(set(signal_ids)) != len(signal_ids):
            raise ValueError("candidate cost signals must reference each candidate at most once")
        if not set(signal_ids) <= candidate_ids:
            raise ValueError("candidate cost signal references an unknown recalled candidate")
        signal_fact_ids = {
            reference_id
            for signal in self.cost_signals
            for reference_id in signal.source_reference_ids
        }
        if not signal_fact_ids <= verified_fact_ids:
            raise ValueError("candidate cost signal references an unverified fact")
        conflict_keys = [(item.candidate_id, item.code) for item in self.hard_conflicts]
        if len(set(conflict_keys)) != len(conflict_keys):
            raise ValueError("candidate hard conflicts must be unique per candidate and code")
        if not {item.candidate_id for item in self.hard_conflicts} <= candidate_ids:
            raise ValueError("candidate hard conflict references an unknown recalled candidate")
        conflict_fact_ids = {
            reference_id
            for conflict in self.hard_conflicts
            for reference_id in conflict.source_reference_ids
        }
        if not conflict_fact_ids <= verified_fact_ids:
            raise ValueError("candidate hard conflict references an unverified fact")
        return self

    @property
    def effective_recommendation_limit(self) -> int:
        if self.recommendation_limit is not None:
            return self.recommendation_limit
        return {1: 5, 2: 6, 3: 7, 4: 10, 5: 12}[self.recall_request.day_count]


class RankingScoreComponent(ImmutableRankingModel):
    factor: RankingFactor
    raw_score: int = Field(ge=0, le=100, strict=True)
    weight_percent: int = Field(ge=0, le=100, strict=True)
    weighted_score: int = Field(ge=0, le=100, strict=True)

    @model_validator(mode="after")
    def weighted_score_is_reproducible(self) -> RankingScoreComponent:
        expected = (self.raw_score * self.weight_percent + 50) // 100
        if self.weighted_score != expected:
            raise ValueError("ranking weighted score must match raw score and weight")
        return self


class RankedCandidate(ImmutableRankingModel):
    rank: int = Field(ge=1, strict=True)
    selection_status: RankingSelectionStatus
    recommendation_origin: Literal["system_recommendation"] = "system_recommendation"
    candidate: RecalledCandidate
    merged_candidate_ids: tuple[UUID, ...] = Field(min_length=1)
    components: tuple[RankingScoreComponent, ...] = Field(min_length=6, max_length=6)
    total_score: int = Field(ge=0, le=100, strict=True)
    confidence: Confidence
    explanations: tuple[ShortText, ...] = Field(min_length=1)
    soft_risks: tuple[ShortText, ...] = ()

    @model_validator(mode="after")
    def score_and_merge_evidence_are_consistent(self) -> RankedCandidate:
        _unique(self.merged_candidate_ids, "merged candidate IDs")
        if self.candidate.candidate_id not in self.merged_candidate_ids:
            raise ValueError("ranked candidate must retain its canonical source candidate ID")
        factors = [component.factor for component in self.components]
        if set(factors) != set(RankingFactor) or len(factors) != len(set(factors)):
            raise ValueError("ranked candidate requires exactly one score for every factor")
        if sum(component.weight_percent for component in self.components) != 100:
            raise ValueError("ranking component weights must total 100 percent")
        if self.total_score != sum(component.weighted_score for component in self.components):
            raise ValueError("candidate total score must equal its weighted components")
        _unique(self.explanations, "ranking explanations")
        _unique(self.soft_risks, "ranking soft risks")
        if self.candidate.availability is DataAvailability.PARTIAL and not self.soft_risks:
            raise ValueError("partial ranked candidate requires an explicit soft risk")
        return self


class FilteredCandidate(ImmutableRankingModel):
    candidate: RecalledCandidate
    merged_candidate_ids: tuple[UUID, ...] = Field(min_length=1)
    conflicts: tuple[CandidateHardConflict, ...] = Field(min_length=1)
    was_strong_desire: bool = False

    @model_validator(mode="after")
    def filtered_candidate_has_unique_evidence(self) -> FilteredCandidate:
        _unique(self.merged_candidate_ids, "filtered merged candidate IDs")
        if self.candidate.candidate_id not in self.merged_candidate_ids:
            raise ValueError("filtered candidate must retain its canonical source candidate ID")
        keys = [(item.candidate_id, item.code) for item in self.conflicts]
        _unique(keys, "filtered candidate conflicts")
        if not {item.candidate_id for item in self.conflicts} <= set(self.merged_candidate_ids):
            raise ValueError("filtered candidate conflicts must reference its merged sources")
        return self


class CandidateRankingResult(ImmutableRankingModel):
    ranking_version: str = Field(pattern=r"^[1-9]\d*\.\d+\.\d+$")
    config_id: NonEmptyText
    ranking_request_id: UUID
    recall_request_id: UUID
    trip_id: UUID
    semantic_state_version: int = Field(default=0, ge=0, strict=True)
    task_book_id: UUID | None = None
    task_book_revision: int | None = Field(default=None, ge=1, strict=True)
    city_id: NonEmptyText
    status: DataAvailability
    input_candidate_count: int = Field(ge=0, le=40, strict=True)
    deduplicated_candidate_count: int = Field(ge=0, le=40, strict=True)
    recommendation_limit: int = Field(ge=1, le=14, strict=True)
    candidates: tuple[RankedCandidate, ...] = ()
    filtered_candidates: tuple[FilteredCandidate, ...] = ()
    degradation_reasons: tuple[ShortText, ...] = ()
    generated_at: AwareDatetime

    @model_validator(mode="after")
    def result_is_a_complete_partition_of_recalled_candidates(self) -> CandidateRankingResult:
        if self.deduplicated_candidate_count != len(self.candidates) + len(
            self.filtered_candidates
        ):
            raise ValueError("deduplicated count must equal ranked and filtered groups")
        all_groups = [
            *(candidate.merged_candidate_ids for candidate in self.candidates),
            *(candidate.merged_candidate_ids for candidate in self.filtered_candidates),
        ]
        all_input_ids = [candidate_id for group in all_groups for candidate_id in group]
        _unique(all_input_ids, "ranking input candidate IDs")
        if len(all_input_ids) != self.input_candidate_count:
            raise ValueError("ranking result must account for every recalled candidate")
        expected_ranks = list(range(1, len(self.candidates) + 1))
        if [candidate.rank for candidate in self.candidates] != expected_ranks:
            raise ValueError("ranked candidates must have contiguous ranks")
        if any(candidate.candidate.place.city_id != self.city_id for candidate in self.candidates):
            raise ValueError("ranked candidates cannot cross cities")
        if any(
            candidate.candidate.place.city_id != self.city_id
            for candidate in self.filtered_candidates
        ):
            raise ValueError("filtered candidates cannot cross cities")
        expected_recommended = min(self.recommendation_limit, len(self.candidates))
        actual_recommended = sum(
            candidate.selection_status is RankingSelectionStatus.RECOMMENDED
            for candidate in self.candidates
        )
        if actual_recommended != expected_recommended:
            raise ValueError("ranking result has an invalid recommended candidate count")
        if any(
            candidate.selection_status is RankingSelectionStatus.RESERVE
            for candidate in self.candidates[:expected_recommended]
        ) or any(
            candidate.selection_status is RankingSelectionStatus.RECOMMENDED
            for candidate in self.candidates[expected_recommended:]
        ):
            raise ValueError("recommended candidates must precede reserve candidates")
        _unique(self.degradation_reasons, "ranking degradation reasons")
        if self.status is DataAvailability.MISSING:
            if self.candidates or not self.degradation_reasons:
                raise ValueError("missing ranking result requires no candidates and a reason")
        elif not self.candidates:
            raise ValueError("available or partial ranking result requires candidates")
        if self.status is DataAvailability.AVAILABLE and (
            self.filtered_candidates
            or self.degradation_reasons
            or any(candidate.soft_risks for candidate in self.candidates)
        ):
            raise ValueError("available ranking result cannot contain degraded candidate data")
        return self


def _unique(values: Sequence[object], label: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must be unique")


V3_CANDIDATE_RANKING_CONTRACTS: tuple[type[ContractModel], ...] = (
    CandidateRankingRequest,
    CandidateRankingResult,
)
