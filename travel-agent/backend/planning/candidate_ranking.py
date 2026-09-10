"""Deterministic V3-31 place deduplication, hard filtering and candidate ranking."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from difflib import SequenceMatcher
from pathlib import Path
from statistics import median_low
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.contracts.candidate_ranking import (
    CandidateCostSignal,
    CandidateHardConflict,
    CandidateHardConflictCode,
    CandidateRankingRequest,
    CandidateRankingResult,
    FilteredCandidate,
    RankedCandidate,
    RankingFactor,
    RankingScoreComponent,
    RankingSelectionStatus,
)
from backend.contracts.candidate_recall import (
    CandidateDomain,
    CandidateSourceReference,
    NamedPlacePriority,
    RecallChannel,
    RecalledCandidate,
    RecalledPlace,
    RecallSourceKind,
)
from backend.contracts.enums import Confidence, DataAvailability
from backend.contracts.places import Gcj02Coordinates

DEFAULT_RANKING_CONFIG = Path(__file__).resolve().parent / "config" / "candidate-ranking.v1.json"


class DeduplicationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    attraction_max_distance_m: int = Field(ge=1, le=2_000, strict=True)
    branch_max_distance_m: int = Field(ge=1, le=1_000, strict=True)
    name_similarity_minimum: float = Field(ge=0.5, le=1.0)
    coordinate_contradiction_m: int = Field(ge=500, le=10_000, strict=True)


class DiversityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    maximum_same_theme_share_percent: int = Field(ge=25, le=100, strict=True)


class CandidateRankingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    config_id: str = Field(min_length=1)
    ranking_version: str = Field(pattern=r"^[1-9]\d*\.\d+\.\d+$")
    weights: dict[RankingFactor, int]
    source_trust: dict[RecallSourceKind, int]
    deduplication: DeduplicationConfig
    diversity: DiversityConfig

    @model_validator(mode="after")
    def scoring_dimensions_are_complete(self) -> CandidateRankingConfig:
        if set(self.weights) != set(RankingFactor):
            raise ValueError("ranking config requires exactly the supported score factors")
        if sum(self.weights.values()) != 100 or any(
            value < 0 or value > 100 for value in self.weights.values()
        ):
            raise ValueError("ranking weights must be percentages totaling 100")
        if set(self.source_trust) != set(RecallSourceKind) or any(
            value < 0 or value > 100 for value in self.source_trust.values()
        ):
            raise ValueError("ranking config requires bounded trust for every source kind")
        return self


@dataclass(frozen=True)
class _CandidateGroup:
    candidate: RecalledCandidate
    original_candidate_ids: tuple[UUID, ...]


@dataclass(frozen=True)
class _GroupCosts:
    distance_m: int | None
    travel_minutes: int | None
    visit_minutes: int | None


@dataclass(frozen=True)
class _ScoredCandidate:
    group: _CandidateGroup
    components: tuple[RankingScoreComponent, ...]
    total_score: int
    confidence: Confidence
    explanations: tuple[str, ...]
    soft_risks: tuple[str, ...]
    strong_priority: int


class CandidateRankingService:
    def __init__(
        self,
        *,
        config: CandidateRankingConfig | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._config = config or load_candidate_ranking_config()
        self._clock = clock

    def rank(self, request: CandidateRankingRequest) -> CandidateRankingResult:
        request = CandidateRankingRequest.model_validate(request.model_dump(mode="json"))
        groups = _deduplicate_candidates(
            request.recall_result.candidates,
            self._config.deduplication,
        )
        costs_by_id = {signal.candidate_id: signal for signal in request.cost_signals}
        explicit_conflicts = tuple(request.hard_conflicts)
        executable: list[_ScoredCandidate] = []
        filtered: list[FilteredCandidate] = []

        for group in groups:
            strong_priority = _strong_priority(request, group)
            conflicts = _hard_conflicts(request, group, explicit_conflicts)
            if conflicts:
                filtered.append(
                    FilteredCandidate(
                        candidate=group.candidate,
                        merged_candidate_ids=group.original_candidate_ids,
                        conflicts=conflicts,
                        was_strong_desire=strong_priority >= 80,
                    )
                )
                continue
            costs = _group_costs(request, group, costs_by_id)
            executable.append(
                _score_candidate(
                    request,
                    group,
                    costs,
                    strong_priority,
                    self._config,
                )
            )

        recommendation_limit = request.effective_recommendation_limit
        ordered = _assemble_diverse_order(
            executable,
            request,
            recommendation_limit,
            self._config.diversity,
        )
        ranked = tuple(
            RankedCandidate(
                rank=index,
                selection_status=(
                    RankingSelectionStatus.RECOMMENDED
                    if index <= min(recommendation_limit, len(ordered))
                    else RankingSelectionStatus.RESERVE
                ),
                candidate=item.group.candidate,
                merged_candidate_ids=item.group.original_candidate_ids,
                components=item.components,
                total_score=item.total_score,
                confidence=item.confidence,
                explanations=item.explanations,
                soft_risks=item.soft_risks,
            )
            for index, item in enumerate(ordered, start=1)
        )
        degradation = list(request.recall_result.degradation_reasons)
        if filtered:
            degradation.append("one or more candidates were removed by hard constraints")
        if any(item.soft_risks for item in ranked):
            degradation.append("some candidates have incomplete ranking inputs")
        if not ranked:
            degradation.append("no executable candidates remain after hard filtering")
        degradation = list(dict.fromkeys(degradation))
        status = _result_status(ranked, filtered, degradation)
        generated_at = self._clock()
        if generated_at.tzinfo is None or generated_at.utcoffset() is None:
            raise ValueError("candidate ranking clock must return an aware datetime")
        return CandidateRankingResult(
            ranking_version=self._config.ranking_version,
            config_id=self._config.config_id,
            ranking_request_id=request.ranking_request_id,
            recall_request_id=request.recall_request.request_id,
            trip_id=request.recall_request.trip_id,
            semantic_state_version=request.recall_request.semantic_state_version,
            task_book_id=request.recall_request.task_book_id,
            task_book_revision=request.recall_request.task_book_revision,
            city_id=request.recall_request.city_id,
            status=status,
            input_candidate_count=len(request.recall_result.candidates),
            deduplicated_candidate_count=len(groups),
            recommendation_limit=recommendation_limit,
            candidates=ranked,
            filtered_candidates=tuple(filtered),
            degradation_reasons=tuple(degradation),
            generated_at=generated_at.astimezone(UTC),
        )


def load_candidate_ranking_config(
    path: Path = DEFAULT_RANKING_CONFIG,
) -> CandidateRankingConfig:
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    return CandidateRankingConfig.model_validate(payload)


def _deduplicate_candidates(
    candidates: tuple[RecalledCandidate, ...],
    config: DeduplicationConfig,
) -> list[_CandidateGroup]:
    ordered = sorted(candidates, key=_stable_candidate_key)
    parents = list(range(len(ordered)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[max(left_root, right_root)] = min(left_root, right_root)

    for left in range(len(ordered)):
        for right in range(left + 1, len(ordered)):
            if _same_place(ordered[left], ordered[right], config):
                union(left, right)

    grouped: dict[int, list[RecalledCandidate]] = {}
    for index, candidate in enumerate(ordered):
        grouped.setdefault(find(index), []).append(candidate)
    return [_merge_candidate_group(grouped[key]) for key in sorted(grouped)]


def _same_place(
    left: RecalledCandidate,
    right: RecalledCandidate,
    config: DeduplicationConfig,
) -> bool:
    if left.place.city_id != right.place.city_id or left.domain is not right.domain:
        return False
    if left.place.category is not right.place.category:
        return False
    if left.place.place_id == right.place.place_id or _sources_overlap(left, right):
        return True
    similarity = _name_similarity(left.place.name, right.place.name)
    if similarity < config.name_similarity_minimum:
        return False
    address_match = _addresses_match(left.place.address, right.place.address)
    distance = _coordinate_distance(left.place.coordinates, right.place.coordinates)
    if distance is not None and distance > config.coordinate_contradiction_m:
        return False
    if left.domain in {CandidateDomain.RESTAURANT, CandidateDomain.HOTEL}:
        return address_match and (distance is None or distance <= config.branch_max_distance_m)
    if address_match:
        return distance is None or distance <= config.coordinate_contradiction_m
    if distance is not None and distance <= config.attraction_max_distance_m:
        return True
    return (
        similarity >= 0.9
        and _is_reviewed_content_provider_pair(left, right)
        and _one_side_lacks_identity_details(left, right)
    )


def _merge_candidate_group(candidates: list[RecalledCandidate]) -> _CandidateGroup:
    ordered = sorted(candidates, key=_canonical_candidate_key)
    best = ordered[0]
    reviewed = next(
        (
            candidate
            for candidate in ordered
            if any(source.kind is RecallSourceKind.CITY_CONTENT for source in candidate.sources)
        ),
        None,
    )
    canonical_name = reviewed.place.name if reviewed is not None else best.place.name
    # Fuzzy duplicates may be different branches or neighbouring venues. Business
    # facts can only follow the chosen canonical identity, never the fuzzy group.
    same_entity = [
        candidate
        for candidate in ordered
        if candidate.place.place_id == best.place.place_id
        and (candidate is best or _provider_identities(candidate) & _provider_identities(best))
    ]
    address = next(
        (candidate.place.address for candidate in ordered if candidate.place.address is not None),
        None,
    )
    coordinates = next(
        (
            candidate.place.coordinates
            for candidate in ordered
            if candidate.place.coordinates is not None
        ),
        None,
    )
    sources = _unique_sources(source for candidate in ordered for source in candidate.sources)
    has_available_source = any(
        candidate.availability is DataAvailability.AVAILABLE for candidate in ordered
    )
    if has_available_source and coordinates is not None:
        availability = DataAvailability.AVAILABLE
        missing_fields: tuple[str, ...] = ()
        missing_reason = None
    else:
        availability = DataAvailability.PARTIAL
        unresolved = list(best.missing_fields)
        if address is not None:
            unresolved = [field for field in unresolved if field != "address"]
        if coordinates is not None:
            unresolved = [field for field in unresolved if field != "coordinates"]
        if any(source.kind is RecallSourceKind.PROVIDER for source in sources):
            unresolved = [field for field in unresolved if field != "provider_place_id"]
        if not unresolved:
            unresolved = ["source_details"]
        missing_fields = tuple(dict.fromkeys(unresolved))
        missing_reason = "merged candidate still lacks one or more ranking facts"
    merged = RecalledCandidate(
        candidate_id=best.candidate_id,
        domain=best.domain,
        place=RecalledPlace(
            place_id=best.place.place_id,
            city_id=best.place.city_id,
            category=best.place.category,
            provider_typecode=best.place.provider_typecode,
            provider_parent_place_id=best.place.provider_parent_place_id,
            name=canonical_name,
            address=address,
            short_description=next(
                (
                    item.place.short_description
                    for item in same_entity
                    if item.place.short_description is not None
                ),
                None,
            ),
            coordinates=coordinates,
            rating=next(
                (item.place.rating for item in same_entity if item.place.rating is not None), None
            ),
            cuisine=next(
                (item.place.cuisine for item in same_entity if item.place.cuisine is not None), None
            ),
            average_cost=next(
                (
                    item.place.average_cost
                    for item in same_entity
                    if item.place.average_cost is not None
                ),
                None,
            ),
            image_url=next(
                (
                    candidate.place.image_url
                    for candidate in ordered
                    if candidate.place.image_url is not None
                ),
                None,
            ),
        ),
        channels=_unique_enum_values(candidate.channels for candidate in ordered),
        # Fuzzy deduplication is not identity verification. Only an observation
        # of the final displayed entity can retain this exact-query privilege.
        representative_identity_verified=any(
            candidate.representative_identity_verified
            and candidate.place.place_id == best.place.place_id
            and candidate.place.name == canonical_name
            and _provider_identities(candidate) & _provider_identities(best)
            for candidate in ordered
        ),
        theme_ids=_unique_text_values(candidate.theme_ids for candidate in ordered),
        reasons=_unique_text_values(candidate.reasons for candidate in ordered),
        named_evidence_ids=tuple(
            sorted(
                {
                    evidence_id
                    for candidate in ordered
                    for evidence_id in candidate.named_evidence_ids
                },
                key=str,
            )
        ),
        sources=sources,
        availability=availability,
        missing_fields=missing_fields,
        missing_reason=missing_reason,
    )
    return _CandidateGroup(
        candidate=merged,
        original_candidate_ids=tuple(
            sorted((candidate.candidate_id for candidate in candidates), key=str)
        ),
    )


def _hard_conflicts(
    request: CandidateRankingRequest,
    group: _CandidateGroup,
    explicit: tuple[CandidateHardConflict, ...],
) -> tuple[CandidateHardConflict, ...]:
    candidate = group.candidate
    conflicts = [
        conflict for conflict in explicit if conflict.candidate_id in group.original_candidate_ids
    ]
    for clue in request.recall_request.excluded_places:
        if clue.domain not in {None, candidate.domain}:
            continue
        matches = (
            clue.known_place_id == candidate.place.place_id
            if clue.known_place_id is not None
            else _normalize_text(clue.name) == _normalize_text(candidate.place.name)
        )
        if matches:
            conflicts.append(
                CandidateHardConflict(
                    candidate_id=candidate.candidate_id,
                    code=CandidateHardConflictCode.EXPLICITLY_EXCLUDED,
                    reason="candidate matches an explicit user exclusion",
                    source_reference_ids=(str(clue.source_message_id),),
                )
            )
    return tuple(
        sorted(
            _deduplicate_conflicts(conflicts),
            key=lambda item: (item.code.value, str(item.candidate_id)),
        )
    )


def _score_candidate(
    request: CandidateRankingRequest,
    group: _CandidateGroup,
    costs: _GroupCosts,
    strong_priority: int,
    config: CandidateRankingConfig,
) -> _ScoredCandidate:
    preference_score, preference_explanation = _preference_score(request, group.candidate)
    spatial_score, spatial_explanation, spatial_risk = _spatial_score(request, costs)
    time_score, time_explanation, time_risk = _time_score(request, costs)
    completeness_score = _fact_completeness_score(group.candidate)
    source_score = _source_trust_score(group.candidate, config)
    raw_scores = {
        RankingFactor.STRONG_DESIRE: strong_priority,
        RankingFactor.PREFERENCE_MATCH: preference_score,
        RankingFactor.SPATIAL_COST: spatial_score,
        RankingFactor.TIME_COST: time_score,
        RankingFactor.FACT_COMPLETENESS: completeness_score,
        RankingFactor.SOURCE_TRUST: source_score,
    }
    components = tuple(
        RankingScoreComponent(
            factor=factor,
            raw_score=raw_scores[factor],
            weight_percent=config.weights[factor],
            weighted_score=(raw_scores[factor] * config.weights[factor] + 50) // 100,
        )
        for factor in RankingFactor
    )
    soft_risks: list[str] = []
    if group.candidate.availability is DataAvailability.PARTIAL:
        soft_risks.append(group.candidate.missing_reason or "candidate facts are incomplete")
    if spatial_risk is not None:
        soft_risks.append(spatial_risk)
    if time_risk is not None:
        soft_risks.append(time_risk)
    explanations = [
        _strong_desire_explanation(strong_priority),
        preference_explanation,
        spatial_explanation,
        time_explanation,
        (
            "地点事实完整度较高"
            if completeness_score >= 90
            else f"地点事实完整度得分 {completeness_score}/100"
        ),
        f"由 {len(group.candidate.sources)} 个可追溯来源支持",
    ]
    confidence = _ranking_confidence(group.candidate, costs)
    return _ScoredCandidate(
        group=group,
        components=components,
        total_score=sum(component.weighted_score for component in components),
        confidence=confidence,
        explanations=tuple(dict.fromkeys(explanations)),
        soft_risks=tuple(dict.fromkeys(soft_risks)),
        strong_priority=strong_priority,
    )


def _strong_priority(request: CandidateRankingRequest, group: _CandidateGroup) -> int:
    priorities = {clue.clue_id: clue.priority for clue in request.recall_request.named_places}
    values = [priorities[item] for item in group.candidate.named_evidence_ids if item in priorities]
    if NamedPlacePriority.MUST in values:
        return 100
    if NamedPlacePriority.WANT in values:
        return 85
    if values or RecallChannel.USER_NAMED in group.candidate.channels:
        return 65
    return 50


def _preference_score(
    request: CandidateRankingRequest,
    candidate: RecalledCandidate,
) -> tuple[int, str]:
    selected = set(request.preferences.selected_theme_ids) or {
        theme.theme_id for theme in request.recall_request.themes
    }
    matches = selected & set(candidate.theme_ids)
    if selected:
        theme_score = 100 if matches else 40
    elif request.recall_request.theme_mode.value == "open_to_any":
        theme_score = 65
    else:
        theme_score = 55
    target_level = 3
    if RecallChannel.CITY_LANDMARK in candidate.channels:
        target_level = 1
    elif RecallChannel.EXPLORATION in candidate.channels:
        target_level = 5
    classic_score = max(
        20,
        100 - 20 * abs(request.preferences.classic_niche_level - target_level),
    )
    score = (theme_score * 70 + classic_score * 30 + 50) // 100
    if matches:
        explanation = f"匹配本次选择的 {len(matches)} 个体验方向"
    elif RecallChannel.CITY_LANDMARK in candidate.channels:
        explanation = "作为城市代表地点参与经典与小众取舍"
    elif RecallChannel.EXPLORATION in candidate.channels:
        explanation = "作为少量探索候选参与经典与小众取舍"
    else:
        explanation = "按当前体验倾向进行中性匹配"
    return score, explanation


def _spatial_score(
    request: CandidateRankingRequest,
    costs: _GroupCosts,
) -> tuple[int, str, str | None]:
    if costs.distance_m is None:
        return 50, "尚无可用于空间比较的距离数据", "spatial distance is unavailable"
    comfortable = (3_000, 4_000, 5_500, 7_000, 9_000)[request.preferences.transit_taxi_level - 1]
    penalty = min(90, costs.distance_m * 60 // comfortable)
    return (
        max(10, 100 - penalty),
        f"距当前空间锚点约 {costs.distance_m} 米",
        None,
    )


def _time_score(
    request: CandidateRankingRequest,
    costs: _GroupCosts,
) -> tuple[int, str, str | None]:
    if costs.travel_minutes is None and costs.visit_minutes is None:
        return 50, "尚无可用于比较的交通或游玩时长", "time cost is unavailable"
    pace = request.preferences.pace_level
    travel = costs.travel_minutes or 0
    visit = costs.visit_minutes or 0
    travel_multiplier = (10, 11, 12, 13, 14)[pace - 1]
    visit_multiplier = (30, 24, 18, 12, 6)[pace - 1]
    penalty_tenths = travel * travel_multiplier + max(0, visit - 60) * visit_multiplier // 10
    score = max(10, 100 - min(90, penalty_tenths // 10))
    parts = []
    if costs.travel_minutes is not None:
        parts.append(f"交通约 {costs.travel_minutes} 分钟")
    if costs.visit_minutes is not None:
        parts.append(f"建议停留约 {costs.visit_minutes} 分钟")
    risk = None
    if costs.travel_minutes is None or costs.visit_minutes is None:
        risk = "one time-cost dimension is unavailable"
    return score, "，".join(parts), risk


def _fact_completeness_score(candidate: RecalledCandidate) -> int:
    score = 100
    if candidate.place.coordinates is None:
        score -= 35
    if candidate.place.address is None:
        score -= 15
    score -= min(30, len(candidate.missing_fields) * 10)
    if candidate.availability is DataAvailability.PARTIAL:
        score -= 10
    return max(10, score)


def _source_trust_score(
    candidate: RecalledCandidate,
    config: CandidateRankingConfig,
) -> int:
    strongest = max(config.source_trust[source.kind] for source in candidate.sources)
    return min(100, strongest + min(10, (len(candidate.sources) - 1) * 5))


def _group_costs(
    request: CandidateRankingRequest,
    group: _CandidateGroup,
    signals_by_id: dict[UUID, CandidateCostSignal],
) -> _GroupCosts:
    signals = [
        signals_by_id[candidate_id]
        for candidate_id in group.original_candidate_ids
        if candidate_id in signals_by_id
    ]
    distances = [
        signal.distance_from_anchor_m
        for signal in signals
        if signal.distance_from_anchor_m is not None
    ]
    if not distances and group.candidate.place.coordinates is not None:
        distances = [
            round(_distance_meters(group.candidate.place.coordinates, anchor.coordinates))
            for anchor in request.recall_request.anchors
        ]
    travel = [
        signal.travel_time_minutes for signal in signals if signal.travel_time_minutes is not None
    ]
    visit = [
        signal.visit_duration_minutes
        for signal in signals
        if signal.visit_duration_minutes is not None
    ]
    return _GroupCosts(
        distance_m=min(distances) if distances else None,
        travel_minutes=int(median_low(travel)) if travel else None,
        visit_minutes=int(median_low(visit)) if visit else None,
    )


def _assemble_diverse_order(
    candidates: list[_ScoredCandidate],
    request: CandidateRankingRequest,
    limit: int,
    config: DiversityConfig,
) -> list[_ScoredCandidate]:
    score_order = sorted(candidates, key=_score_order_key)
    selected: list[_ScoredCandidate] = []

    def add(item: _ScoredCandidate | None) -> None:
        if item is not None and item not in selected and len(selected) < limit:
            selected.append(item)

    for item in score_order:
        if item.strong_priority == 100:
            add(item)
    add(score_order[0] if score_order else None)
    selected_themes = request.preferences.selected_theme_ids or tuple(
        theme.theme_id for theme in request.recall_request.themes
    )
    for theme_id in selected_themes:
        add(
            next((item for item in score_order if theme_id in item.group.candidate.theme_ids), None)
        )
    if request.recall_request.landmark_policy.value != "exclude":
        add(
            next(
                (
                    item
                    for item in score_order
                    if RecallChannel.CITY_LANDMARK in item.group.candidate.channels
                ),
                None,
            )
        )
    add(
        next(
            (
                item
                for item in score_order
                if RecallChannel.CITY_FEATURE in item.group.candidate.channels
            ),
            None,
        )
    )
    add(
        next(
            (
                item
                for item in score_order
                if RecallChannel.EXPLORATION in item.group.candidate.channels
            ),
            None,
        )
    )

    theme_cap = max(
        1,
        math.ceil(limit * config.maximum_same_theme_share_percent / 100),
    )
    for item in score_order:
        if len(selected) >= limit:
            break
        primary_theme = _primary_theme(item.group.candidate)
        if (
            primary_theme is not None
            and sum(
                _primary_theme(selected_item.group.candidate) == primary_theme
                for selected_item in selected
            )
            >= theme_cap
        ):
            continue
        add(item)
    for item in score_order:
        add(item)
    reserves = [item for item in score_order if item not in selected]
    return [*selected, *reserves]


def _result_status(
    ranked: tuple[RankedCandidate, ...],
    filtered: list[FilteredCandidate],
    degradation: list[str],
) -> DataAvailability:
    if not ranked:
        return DataAvailability.MISSING
    if filtered or degradation or any(item.soft_risks for item in ranked):
        return DataAvailability.PARTIAL
    return DataAvailability.AVAILABLE


def _ranking_confidence(
    candidate: RecalledCandidate,
    costs: _GroupCosts,
) -> Confidence:
    if candidate.availability is DataAvailability.PARTIAL:
        return Confidence.LOW
    if costs.distance_m is not None and costs.travel_minutes is not None:
        return Confidence.HIGH
    return Confidence.MEDIUM


def _strong_desire_explanation(score: int) -> str:
    if score == 100:
        return "用户明确标记为必去"
    if score == 85:
        return "用户明确表达想去"
    if score == 65:
        return "用户主动点名了这个地点"
    return "用户尚未对这个地点作出选择"


def _score_order_key(item: _ScoredCandidate) -> tuple[int, int, str, str]:
    return (
        -item.total_score,
        -item.strong_priority,
        _normalize_text(item.group.candidate.place.name),
        str(item.group.candidate.place.place_id),
    )


def _stable_candidate_key(candidate: RecalledCandidate) -> tuple[str, str, str, str, str]:
    return (
        candidate.place.city_id,
        candidate.domain.value,
        _normalize_text(candidate.place.name),
        str(candidate.place.place_id),
        str(candidate.candidate_id),
    )


def _canonical_candidate_key(candidate: RecalledCandidate) -> tuple[int, str]:
    quality = (
        (40 if candidate.availability is DataAvailability.AVAILABLE else 0)
        + (25 if candidate.place.coordinates is not None else 0)
        + (15 if candidate.place.address is not None else 0)
        + min(10, len(candidate.sources) * 5)
        + min(10, len(candidate.named_evidence_ids) * 5)
    )
    return -quality, str(candidate.candidate_id)


def _unique_sources(
    sources: Iterable[CandidateSourceReference],
) -> tuple[CandidateSourceReference, ...]:
    unique: dict[tuple[str, ...], CandidateSourceReference] = {}
    for source in sources:
        key: tuple[str, ...]
        if source.kind is RecallSourceKind.PROVIDER:
            key = (
                source.kind.value,
                source.provider.value if source.provider is not None else "",
                source.source_place_id or "",
            )
        elif source.kind is RecallSourceKind.CITY_CONTENT:
            key = (
                source.kind.value,
                source.content_version or "",
                str(source.source_url or ""),
            )
        else:
            key = (source.kind.value, source.source_record_id)
        unique.setdefault(key, source)
    return tuple(
        sorted(
            unique.values(),
            key=lambda item: (
                item.kind.value,
                item.provider.value if item.provider is not None else "",
                item.source_place_id or "",
                item.source_record_id,
            ),
        )
    )


def _unique_enum_values(values: Iterable[tuple[RecallChannel, ...]]) -> tuple[RecallChannel, ...]:
    return tuple(sorted({item for group in values for item in group}, key=lambda item: item.value))


def _unique_text_values(values: Iterable[tuple[str, ...]]) -> tuple[str, ...]:
    return tuple(sorted({item for group in values for item in group}))


def _deduplicate_conflicts(
    conflicts: Iterable[CandidateHardConflict],
) -> list[CandidateHardConflict]:
    unique: dict[tuple[UUID, CandidateHardConflictCode], CandidateHardConflict] = {}
    for conflict in conflicts:
        unique.setdefault((conflict.candidate_id, conflict.code), conflict)
    return list(unique.values())


def _provider_identities(candidate: RecalledCandidate) -> set[tuple[str, str]]:
    return {
        (source.provider.value, source.source_place_id)
        for source in candidate.sources
        if source.kind is RecallSourceKind.PROVIDER
        and source.provider is not None
        and source.source_place_id is not None
    }


def _sources_overlap(left: RecalledCandidate, right: RecalledCandidate) -> bool:
    left_keys = {
        (source.kind, source.provider, source.source_place_id)
        for source in left.sources
        if source.source_place_id is not None
    }
    right_keys = {
        (source.kind, source.provider, source.source_place_id)
        for source in right.sources
        if source.source_place_id is not None
    }
    return bool(left_keys & right_keys)


def _is_reviewed_content_provider_pair(
    left: RecalledCandidate,
    right: RecalledCandidate,
) -> bool:
    kinds = {source.kind for candidate in (left, right) for source in candidate.sources}
    return {RecallSourceKind.CITY_CONTENT, RecallSourceKind.PROVIDER} <= kinds


def _one_side_lacks_identity_details(
    left: RecalledCandidate,
    right: RecalledCandidate,
) -> bool:
    return any(
        candidate.place.address is None and candidate.place.coordinates is None
        for candidate in (left, right)
    )


def _name_similarity(left: str, right: str) -> float:
    normalized_left = _normalize_text(left)
    normalized_right = _normalize_text(right)
    if normalized_left == normalized_right:
        return 1.0
    if normalized_left in normalized_right or normalized_right in normalized_left:
        return 0.9
    return SequenceMatcher(None, normalized_left, normalized_right).ratio()


def _addresses_match(left: str | None, right: str | None) -> bool:
    if left is None or right is None:
        return False
    normalized_left = _normalize_text(left)
    normalized_right = _normalize_text(right)
    if min(len(normalized_left), len(normalized_right)) < 5:
        return False
    return normalized_left in normalized_right or normalized_right in normalized_left


def _coordinate_distance(
    left: Gcj02Coordinates | None,
    right: Gcj02Coordinates | None,
) -> int | None:
    if left is None or right is None:
        return None
    return round(_distance_meters(left, right))


def _distance_meters(left: Gcj02Coordinates, right: Gcj02Coordinates) -> float:
    radius = 6_371_000.0
    left_latitude = math.radians(left.latitude)
    right_latitude = math.radians(right.latitude)
    latitude_delta = right_latitude - left_latitude
    longitude_delta = math.radians(right.longitude - left.longitude)
    haversine = (
        math.sin(latitude_delta / 2) ** 2
        + math.cos(left_latitude) * math.cos(right_latitude) * math.sin(longitude_delta / 2) ** 2
    )
    return 2 * radius * math.asin(math.sqrt(haversine))


def _normalize_text(value: str) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]", "", value.casefold())


def _primary_theme(candidate: RecalledCandidate) -> str | None:
    return min(candidate.theme_ids) if candidate.theme_ids else None
