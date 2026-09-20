"""Deterministic V3-35 selection of one route-aware lodging base."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from math import ceil
from statistics import fmean
from uuid import NAMESPACE_URL, UUID, uuid5

from backend.contracts.common import CnyAmountRange
from backend.contracts.enums import DataAvailability, MobilityTolerance
from backend.contracts.lodging_strategy import (
    LodgingBaseCandidate,
    LodgingBaseKind,
    LodgingBaseStrategy,
    LodgingClusterAccess,
    LodgingCommuteSummary,
    LodgingStrategyPreferences,
    LodgingStrategyRequest,
    LodgingStrategyResult,
    LodgingStrategyScore,
)
from backend.contracts.spatial_planning import SpatialAnchorStrength

LODGING_STRATEGY_ALGORITHM_VERSION = "1.0.0"
_WALKING_METERS = {
    MobilityTolerance.NEVER: 0,
    MobilityTolerance.WITHIN_5: 400,
    MobilityTolerance.AROUND_10: 800,
    MobilityTolerance.FIFTEEN_PLUS: 1_200,
}


class LodgingStrategyService:
    """Compare area/node bases using route facts, then select one base only."""

    def __init__(
        self,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._clock = clock

    def build(self, request: LodgingStrategyRequest) -> LodgingStrategyResult:
        request = LodgingStrategyRequest.model_validate(request.model_dump(mode="json"))
        generated_at = self._clock()
        if generated_at.tzinfo is None or generated_at.utcoffset() is None:
            raise ValueError("lodging strategy clock must return an aware datetime")
        if request.night_count == 0:
            return LodgingStrategyResult(
                algorithm_version=LODGING_STRATEGY_ALGORITHM_VERSION,
                request_id=request.request_id,
                trip_id=request.trip_id,
                input_state_version=request.input_state_version,
                city_id=request.city_id,
                night_count=0,
                status=DataAvailability.AVAILABLE,
                generated_at=generated_at.astimezone(UTC),
            )
        if request.fixed_hotel_node_id is not None:
            strategy = _fixed_hotel_strategy(
                request,
                cluster_weights=_cluster_weights(request),
            )
            degradation = _degradation_reasons(
                request.fixed_hotel_accesses,
                strategy_count=1,
                fixed=True,
            )
            return LodgingStrategyResult(
                algorithm_version=LODGING_STRATEGY_ALGORITHM_VERSION,
                request_id=request.request_id,
                trip_id=request.trip_id,
                input_state_version=request.input_state_version,
                city_id=request.city_id,
                night_count=request.night_count,
                status=(DataAvailability.PARTIAL if degradation else DataAvailability.AVAILABLE),
                strategies=(strategy,),
                selected_strategy_id=strategy.strategy_id,
                degradation_reasons=degradation,
                generated_at=generated_at.astimezone(UTC),
            )

        usable = [
            candidate
            for candidate in request.candidates
            if _has_known_access(candidate, request.preferences)
        ]
        if not usable:
            raise ValueError("no lodging candidate has a usable route to an activity cluster")
        cluster_weights = _cluster_weights(request)
        fastest = min(
            _blended_typical_minutes(item, request.preferences, cluster_weights) for item in usable
        )
        price_bounds = _price_bounds(usable)
        projected = [
            _project_candidate(
                request,
                candidate,
                fastest_minutes=fastest,
                price_bounds=price_bounds,
                cluster_weights=cluster_weights,
            )
            for candidate in usable
        ]
        selected = sorted(
            projected,
            key=lambda item: (
                -item.score.total,
                item.commute.typical_minutes or 10_000,
                item.label,
            ),
        )[:3]
        ranked = tuple(
            item.model_copy(update={"rank": index}) for index, item in enumerate(selected, 1)
        )
        degradation = _degradation_reasons(
            tuple(access for item in usable for access in item.cluster_accesses),
            strategy_count=len(ranked),
            fixed=False,
        )
        return LodgingStrategyResult(
            algorithm_version=LODGING_STRATEGY_ALGORITHM_VERSION,
            request_id=request.request_id,
            trip_id=request.trip_id,
            input_state_version=request.input_state_version,
            city_id=request.city_id,
            night_count=request.night_count,
            status=DataAvailability.PARTIAL if degradation else DataAvailability.AVAILABLE,
            strategies=ranked,
            selected_strategy_id=ranked[0].strategy_id,
            degradation_reasons=degradation,
            generated_at=generated_at.astimezone(UTC),
        )


def _project_candidate(
    request: LodgingStrategyRequest,
    candidate: LodgingBaseCandidate,
    *,
    fastest_minutes: int,
    price_bounds: tuple[int, int] | None,
    cluster_weights: dict[UUID, int],
) -> LodgingBaseStrategy:
    preferences = request.preferences
    summary = _commute_summary(
        candidate.cluster_accesses,
        preferences,
        fastest_minutes=fastest_minutes,
        cluster_weights=cluster_weights,
    )
    score = _score_candidate(
        candidate,
        summary,
        preferences,
        price_bounds=price_bounds,
        cluster_weights=cluster_weights,
    )
    additional = summary.additional_minutes_vs_fastest or 0
    value_tradeoff = (
        f"相较通勤最快方案，典型单程约增加 {additional} 分钟。"
        if additional > 0
        else "通勤时间没有因控制住宿价格而额外增加。"
    )
    advantages = [_transport_advantage(candidate, summary, preferences)]
    if candidate.typical_nightly_price is not None:
        advantages.append(_price_advantage(candidate.typical_nightly_price))
    if candidate.quality_level is not None:
        advantages.append(f"区域住宿品质信号约为 {candidate.quality_level}/5。")
    tradeoffs = [value_tradeoff]
    if summary.typical_last_mile_walk_m is not None:
        tradeoffs.append(f"公共交通末端步行通常约 {summary.typical_last_mile_walk_m} 米。")
    if summary.typical_taxi_cost is not None:
        tradeoffs.append(f"典型单程打车约 {_format_price(summary.typical_taxi_cost)}。")
    conditions = [_condition(preferences)]
    if candidate.kind is LodgingBaseKind.TRANSIT_NODE:
        conditions.append("适合愿意围绕公共交通节点组织每日出发和返回的旅行。")
    return LodgingBaseStrategy(
        strategy_id=_strategy_id(request.trip_id, candidate.candidate_id),
        rank=1,
        candidate_id=candidate.candidate_id,
        kind=candidate.kind,
        label=candidate.label,
        center=candidate.center,
        covered_cluster_ids=tuple(access.cluster_id for access in candidate.cluster_accesses),
        covered_anchor_ids=tuple(anchor.node_id for anchor in request.spatial_result.anchors),
        commute=summary,
        score=score,
        typical_nightly_price=candidate.typical_nightly_price,
        advantages=tuple(advantages[:4]),
        tradeoffs=tuple(tradeoffs[:4]),
        applicable_conditions=tuple(conditions[:4]),
        explanation=_explanation(candidate, summary, score, preferences),
        source_reference_ids=tuple(
            sorted(
                {
                    *candidate.source_reference_ids,
                    *(
                        source
                        for access in candidate.cluster_accesses
                        for source in access.source_reference_ids
                    ),
                }
            )
        ),
    )


def _fixed_hotel_strategy(
    request: LodgingStrategyRequest,
    *,
    cluster_weights: dict[UUID, int],
) -> LodgingBaseStrategy:
    node_id = request.fixed_hotel_node_id
    if node_id is None:  # protected by caller and request validation.
        raise ValueError("fixed lodging strategy is missing its hotel anchor")
    anchor = next(item for item in request.spatial_result.anchors if item.node_id == node_id)
    summary = _commute_summary(
        request.fixed_hotel_accesses,
        request.preferences,
        fastest_minutes=None,
        cluster_weights=cluster_weights,
    )
    known_ratio = summary.known_cluster_count / summary.assessed_cluster_count
    score = LodgingStrategyScore(
        coverage=round(known_ratio * 100),
        transport_fit=50,
        walking_fit=50,
        route_simplicity=50,
        comfort_fit=50,
        quality_fit=50,
        value_fit=50,
        total=round(50 + known_ratio * 20),
    )
    return LodgingBaseStrategy(
        strategy_id=_strategy_id(request.trip_id, node_id),
        rank=1,
        kind=LodgingBaseKind.FIXED_HOTEL,
        label=f"已订酒店 · {anchor.name}",
        center=anchor.coordinates,
        fixed_hotel_node_id=node_id,
        covered_cluster_ids=tuple(item.cluster_id for item in request.fixed_hotel_accesses),
        covered_anchor_ids=tuple(item.node_id for item in request.spatial_result.anchors),
        commute=summary,
        score=score,
        advantages=("尊重已完成的住宿预订，不再让其他区域参与竞争。",),
        tradeoffs=("其他活动将围绕这个固定住宿基地衡量往返成本。",),
        applicable_conditions=("适用于用户已明确预订且本次不换酒店的行程。",),
        explanation="已订酒店直接成为本次唯一住宿基地，评分仅用于说明覆盖，不用于换店。",
        source_reference_ids=tuple(
            sorted(
                {
                    *anchor.source_reference_ids,
                    *(
                        source
                        for access in request.fixed_hotel_accesses
                        for source in access.source_reference_ids
                    ),
                }
            )
        ),
    )


def _commute_summary(
    accesses: Iterable[LodgingClusterAccess],
    preferences: LodgingStrategyPreferences,
    *,
    fastest_minutes: int | None,
    cluster_weights: dict[UUID, int],
) -> LodgingCommuteSummary:
    items = tuple(accesses)
    known = [item for item in items if _access_minutes(item, preferences) is not None]
    if not known:
        return LodgingCommuteSummary(
            assessed_cluster_count=len(items),
            known_cluster_count=0,
        )
    durations = [_access_minutes(item, preferences) or 0 for item in known]
    known_weights = [cluster_weights[item.cluster_id] for item in known]
    transit = [item for item in known if item.transit_minutes is not None]
    taxi_costs = [item.taxi_cost for item in known if item.taxi_cost is not None]
    typical = ceil(_weighted_mean(durations, known_weights))
    return LodgingCommuteSummary(
        assessed_cluster_count=len(items),
        known_cluster_count=len(known),
        typical_minutes=typical,
        maximum_minutes=max(durations),
        typical_transfers=(
            round(
                _weighted_mean(
                    [item.transit_transfer_count or 0 for item in transit],
                    [cluster_weights[item.cluster_id] for item in transit],
                ),
                1,
            )
            if transit
            else None
        ),
        typical_last_mile_walk_m=(
            ceil(
                _weighted_mean(
                    [item.transit_last_mile_walk_m or 0 for item in transit],
                    [cluster_weights[item.cluster_id] for item in transit],
                )
            )
            if transit
            else None
        ),
        typical_taxi_cost=(
            CnyAmountRange(
                minimum_fen=ceil(fmean(item.minimum_fen for item in taxi_costs)),
                maximum_fen=ceil(fmean(item.maximum_fen for item in taxi_costs)),
            )
            if taxi_costs
            else None
        ),
        additional_minutes_vs_fastest=(
            max(0, typical - fastest_minutes) if fastest_minutes is not None else None
        ),
    )


def _score_candidate(
    candidate: LodgingBaseCandidate,
    summary: LodgingCommuteSummary,
    preferences: LodgingStrategyPreferences,
    *,
    price_bounds: tuple[int, int] | None,
    cluster_weights: dict[UUID, int],
) -> LodgingStrategyScore:
    known_cluster_ids = {
        item.cluster_id
        for item in candidate.cluster_accesses
        if _access_minutes(item, preferences) is not None
    }
    coverage = round(
        sum(cluster_weights[item] for item in known_cluster_ids)
        / sum(cluster_weights.values())
        * 100
    )
    transport_scores = [
        value
        for item in candidate.cluster_accesses
        if (value := _access_transport_score(item, preferences)) is not None
    ]
    transport = round(fmean(transport_scores)) if transport_scores else 0
    walking = _walking_score(summary.typical_last_mile_walk_m, preferences.walking_tolerance)
    cycling_scores = [
        _cycling_score(item.cycling_minutes, preferences.cycling_tolerance)
        for item in candidate.cluster_accesses
        if item.cycling_minutes is not None
        and _cycling_is_acceptable(item.cycling_minutes, preferences.cycling_tolerance)
    ]
    if cycling_scores and preferences.cycling_tolerance is not MobilityTolerance.NEVER:
        walking = max(walking, round(fmean(cycling_scores)))
    transfers = summary.typical_transfers or 0
    simplicity = _bounded(round(100 - transfers * 22))
    typical = summary.typical_minutes or 180
    maximum = summary.maximum_minutes or 360
    comfort = _bounded(round(115 - typical * 1.2 - max(0, maximum - typical) * 0.8))
    quality = _quality_score(candidate.quality_level, preferences.quality_level)
    value = _value_score(candidate, price_bounds)

    quality_weight = 5 + preferences.quality_level**2 * 2
    value_weight = 5 + preferences.value_priority_level**2 * 3
    comfort_weight = 5 + preferences.pace_level**2 * 2
    weights = {
        "coverage": 18,
        "transport": 24,
        "walking": 8,
        "simplicity": 8,
        "comfort": comfort_weight,
        "quality": quality_weight,
        "value": value_weight,
    }
    weighted = (
        coverage * weights["coverage"]
        + transport * weights["transport"]
        + walking * weights["walking"]
        + simplicity * weights["simplicity"]
        + comfort * weights["comfort"]
        + quality * weights["quality"]
        + value * weights["value"]
    )
    total = round(weighted / sum(weights.values()))
    return LodgingStrategyScore(
        coverage=coverage,
        transport_fit=transport,
        walking_fit=walking,
        route_simplicity=simplicity,
        comfort_fit=comfort,
        quality_fit=quality,
        value_fit=value,
        total=_bounded(total),
    )


def _access_minutes(
    access: LodgingClusterAccess,
    preferences: LodgingStrategyPreferences,
) -> int | None:
    transit = access.transit_minutes
    taxi = access.taxi_minutes
    motorized: int | None
    if transit is None:
        motorized = taxi
    elif taxi is None:
        motorized = transit
    else:
        taxi_weight = (preferences.transit_taxi_level - 1) / 4
        motorized = ceil(transit * (1 - taxi_weight) + taxi * taxi_weight)
    cycling = access.cycling_minutes
    if cycling is None or not _cycling_is_acceptable(
        cycling,
        preferences.cycling_tolerance,
    ):
        return motorized
    return cycling if motorized is None else min(motorized, cycling)


def _access_transport_score(
    access: LodgingClusterAccess,
    preferences: LodgingStrategyPreferences,
) -> int | None:
    transit_score = _transit_score(access)
    taxi_score = _taxi_score(access)
    motorized: float | None
    if transit_score is None:
        motorized = taxi_score
    elif taxi_score is None:
        motorized = transit_score
    else:
        taxi_weight = (preferences.transit_taxi_level - 1) / 4
        motorized = transit_score * (1 - taxi_weight) + taxi_score * taxi_weight
    cycling_score = (
        _cycling_score(access.cycling_minutes, preferences.cycling_tolerance)
        if access.cycling_minutes is not None
        and _cycling_is_acceptable(access.cycling_minutes, preferences.cycling_tolerance)
        else None
    )
    usable = [value for value in (motorized, cycling_score) if value is not None]
    return round(max(usable)) if usable else None


def _transit_score(access: LodgingClusterAccess) -> int | None:
    if access.transit_minutes is None:
        return None
    return _bounded(
        round(
            110
            - access.transit_minutes * 1.35
            - (access.transit_transfer_count or 0) * 12
            - (access.transit_last_mile_walk_m or 0) / 45
        )
    )


def _taxi_score(access: LodgingClusterAccess) -> int | None:
    if access.taxi_minutes is None:
        return None
    cost_yuan = (
        ((access.taxi_cost.minimum_fen + access.taxi_cost.maximum_fen) / 200)
        if access.taxi_cost is not None
        else 30
    )
    return _bounded(round(110 - access.taxi_minutes * 1.6 - cost_yuan * 0.7))


def _walking_score(distance_m: int | None, tolerance: MobilityTolerance) -> int:
    if distance_m is None:
        return 50
    limit = _WALKING_METERS[tolerance]
    if limit == 0:
        return 100 if distance_m == 0 else _bounded(55 - round(distance_m / 20))
    return _bounded(round(105 - max(0, distance_m - limit) / 10 - distance_m / limit * 15))


def _cycling_score(minutes: int, tolerance: MobilityTolerance) -> int:
    limit = {
        MobilityTolerance.NEVER: 0,
        MobilityTolerance.WITHIN_5: 5,
        MobilityTolerance.AROUND_10: 10,
        MobilityTolerance.FIFTEEN_PLUS: 15,
    }[tolerance]
    if limit == 0:
        return 0
    return _bounded(round(100 - max(0, minutes - limit) * 7 - minutes / limit * 10))


def _cycling_is_acceptable(minutes: int, tolerance: MobilityTolerance) -> bool:
    if tolerance is MobilityTolerance.NEVER:
        return False
    if tolerance is MobilityTolerance.WITHIN_5:
        return minutes <= 5
    if tolerance is MobilityTolerance.AROUND_10:
        return minutes <= 10
    return True


def _quality_score(candidate_level: int | None, preferred_level: int) -> int:
    if candidate_level is None:
        return 50
    if candidate_level >= preferred_level:
        return _bounded(80 + (candidate_level - preferred_level) * 5)
    return _bounded(80 - (preferred_level - candidate_level) * 25)


def _value_score(
    candidate: LodgingBaseCandidate,
    bounds: tuple[int, int] | None,
) -> int:
    if candidate.typical_nightly_price is None or bounds is None:
        return 50
    low, high = bounds
    midpoint = (
        candidate.typical_nightly_price.minimum_fen + candidate.typical_nightly_price.maximum_fen
    ) / 2
    if high == low:
        return 75
    return _bounded(round(100 - (midpoint - low) / (high - low) * 70))


def _price_bounds(candidates: Iterable[LodgingBaseCandidate]) -> tuple[int, int] | None:
    values = [
        (item.typical_nightly_price.minimum_fen + item.typical_nightly_price.maximum_fen) // 2
        for item in candidates
        if item.typical_nightly_price is not None
    ]
    return (min(values), max(values)) if values else None


def _blended_typical_minutes(
    candidate: LodgingBaseCandidate,
    preferences: LodgingStrategyPreferences,
    cluster_weights: dict[UUID, int],
) -> int:
    values = [
        (value, cluster_weights[access.cluster_id])
        for access in candidate.cluster_accesses
        if (value := _access_minutes(access, preferences)) is not None
    ]
    return (
        ceil(_weighted_mean([value for value, _ in values], [weight for _, weight in values]))
        if values
        else 10_000
    )


def _has_known_access(
    candidate: LodgingBaseCandidate,
    preferences: LodgingStrategyPreferences,
) -> bool:
    return any(
        _access_minutes(access, preferences) is not None for access in candidate.cluster_accesses
    )


def _transport_advantage(
    candidate: LodgingBaseCandidate,
    summary: LodgingCommuteSummary,
    preferences: LodgingStrategyPreferences,
) -> str:
    typical = summary.typical_minutes
    has_motorized = any(
        access.transit_minutes is not None or access.taxi_minutes is not None
        for access in candidate.cluster_accesses
    )
    has_acceptable_cycling = any(
        access.cycling_minutes is not None
        and _cycling_is_acceptable(access.cycling_minutes, preferences.cycling_tolerance)
        for access in candidate.cluster_accesses
    )
    if has_acceptable_cycling and not has_motorized:
        return f"用户允许的骑行路线可达各活动簇，典型单程约 {typical} 分钟。"
    if preferences.transit_taxi_level <= 2:
        return (
            f"公共交通优先时，前往各活动簇典型单程约 {typical} 分钟，"
            f"平均换乘 {summary.typical_transfers or 0:g} 次。"
        )
    if preferences.transit_taxi_level >= 4:
        return (
            f"打车优先时，前往各活动簇门到门典型单程约 {typical} 分钟，"
            f"费用约 {_format_price(summary.typical_taxi_cost)}。"
        )
    return f"公共交通与打车混合时，前往各活动簇典型单程约 {typical} 分钟。"


def _price_advantage(value: CnyAmountRange) -> str:
    return f"区域典型每晚住宿约 {_format_price(value)}，可与实际酒店报价继续核对。"


def _condition(preferences: LodgingStrategyPreferences) -> str:
    if preferences.value_priority_level >= 4:
        return "适合愿意用可量化的额外通勤换取住宿性价比的旅行。"
    if preferences.pace_level >= 4 or preferences.quality_level >= 4:
        return "适合更重视少折返、回住方便和住宿品质的旅行。"
    return "适合在通勤、舒适和住宿价格之间保持平衡的旅行。"


def _explanation(
    candidate: LodgingBaseCandidate,
    summary: LodgingCommuteSummary,
    score: LodgingStrategyScore,
    preferences: LodgingStrategyPreferences,
) -> str:
    mode = (
        "公共交通"
        if preferences.transit_taxi_level <= 2
        else "打车"
        if preferences.transit_taxi_level >= 4
        else "公交与打车混合"
    )
    return (
        f"{candidate.label} 覆盖 {summary.known_cluster_count}/{summary.assessed_cluster_count} "
        f"个活动簇；按{mode}偏好、步行容忍、节奏、品质与价格综合评分 {score.total}。"
    )


def _degradation_reasons(
    accesses: Iterable[LodgingClusterAccess],
    *,
    strategy_count: int,
    fixed: bool,
) -> tuple[str, ...]:
    reasons: list[str] = []
    items = tuple(accesses)
    if any(item.status is not DataAvailability.AVAILABLE for item in items):
        reasons.append("one or more lodging-to-cluster route facts are partial or missing")
    if not fixed and strategy_count < 2:
        reasons.append("fewer than two usable lodging areas were available for comparison")
    return tuple(reasons)


def _strategy_id(trip_id: UUID, source_id: UUID) -> UUID:
    return uuid5(
        NAMESPACE_URL,
        f"iter:lodging:{LODGING_STRATEGY_ALGORITHM_VERSION}:{trip_id}:{source_id}",
    )


def _cluster_weights(request: LodgingStrategyRequest) -> dict[UUID, int]:
    anchors = {anchor.node_id: anchor for anchor in request.spatial_result.anchors}
    return {
        cluster.cluster_id: sum(
            2 if anchors[node_id].strength is SpatialAnchorStrength.STRONG else 1
            for node_id in cluster.member_node_ids
        )
        for cluster in request.spatial_result.clusters
    }


def _weighted_mean(values: list[int], weights: list[int]) -> float:
    return sum(value * weight for value, weight in zip(values, weights, strict=True)) / sum(weights)


def _format_price(value: CnyAmountRange | None) -> str:
    if value is None:
        return "待查询"
    return f"¥{value.minimum_fen / 100:.0f}–{value.maximum_fen / 100:.0f}"


def _bounded(value: int) -> int:
    return max(0, min(100, value))
