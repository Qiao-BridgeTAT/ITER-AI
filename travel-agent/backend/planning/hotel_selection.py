"""Deterministic V3-36 hotel shortlist and explicitly authorized selection."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from math import asin, cos, radians, sin, sqrt
from uuid import NAMESPACE_URL, UUID, uuid5

from backend.contracts.city_registry import CityProviderCapability
from backend.contracts.common import CnyAmountRange
from backend.contracts.enums import DataAvailability, ProviderCode
from backend.contracts.hotel_selection import (
    HotelDecisionMode,
    HotelDecisionStatus,
    HotelPriceBand,
    HotelRouteBaseline,
    HotelSelectionCandidate,
    HotelSelectionRequest,
    HotelSelectionResult,
    HotelSelectionScore,
    HotelSelectionSource,
)
from backend.contracts.lodging_strategy import LodgingBaseStrategy
from backend.contracts.places import Gcj02Coordinates
from backend.planning.city_registry import CityRegistry
from backend.providers.contracts import (
    HotelSearchRequest,
    ProviderError,
    ProviderFailureCode,
    ProviderHotelOffer,
    ProviderResponse,
    ProviderResultStatus,
)
from backend.providers.interfaces import TravelProductProvider
from backend.providers.place_matching import coordinates_to_gcj02

HOTEL_SELECTION_ALGORITHM_VERSION = "1.0.0"


class HotelSelectionService:
    """Build seven diverse options, then select only with user authority."""

    def __init__(
        self,
        *,
        registry: CityRegistry,
        products: TravelProductProvider,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._registry = registry
        self._products = products
        self._clock = clock

    async def select(self, request: HotelSelectionRequest) -> HotelSelectionResult:
        request = HotelSelectionRequest.model_validate(request.model_dump(mode="json"))
        generated_at = self._aware_now()
        if request.night_count == 0:
            return _day_trip_result(request, generated_at)
        strategy = _selected_strategy(request)
        if not self._registry.supports(
            request.city_id,
            CityProviderCapability.FLYAI_PRODUCTS,
        ):
            if request.fixed_hotel is not None:
                return _fixed_without_offer(
                    request,
                    strategy,
                    generated_at,
                    reason="this city has no registered FlyAI product capability",
                )
            return _missing_result(
                request,
                strategy,
                generated_at,
                reason="this city has no registered FlyAI product capability",
                failure_code=ProviderFailureCode.UNAVAILABLE,
            )
        city_scope = self._registry.provider_scope(request.city_id, ProviderCode.FLYAI)
        search = HotelSearchRequest(
            city=city_scope,
            check_in=request.check_in,
            check_out=request.check_out,
            anchor=strategy.center,
            anchor_name=(
                request.fixed_hotel.name if request.fixed_hotel is not None else strategy.label
            ),
            query=(request.fixed_hotel.name if request.fixed_hotel is not None else strategy.label),
        )
        try:
            response = await self._products.search_hotels(search)
        except ProviderError as error:
            if request.fixed_hotel is not None:
                return _fixed_without_offer(
                    request,
                    strategy,
                    generated_at,
                    reason=f"FlyAI hotel search failed: {error.code.value}",
                    failure_code=error.code,
                )
            return _missing_result(
                request,
                strategy,
                generated_at,
                reason=f"FlyAI hotel search failed: {error.code.value}",
                failure_code=error.code,
            )
        if request.fixed_hotel is not None:
            return _select_fixed_hotel(request, strategy, response, generated_at)
        if response.status is ProviderResultStatus.EMPTY or not response.items:
            return _missing_result(
                request,
                strategy,
                generated_at,
                reason="FlyAI returned no hotels for the selected lodging area and dates",
            )
        candidates = _project_and_rank(request, strategy, response)
        if not candidates:
            return _missing_result(
                request,
                strategy,
                generated_at,
                reason="FlyAI returned no usable hotel records",
            )
        selected = _decide_hotel(request, candidates)
        selection_source = _selection_source(request)
        degradation = _result_degradation(response, selected, candidates, request.candidate_count)
        return HotelSelectionResult(
            algorithm_version=HOTEL_SELECTION_ALGORITHM_VERSION,
            request_id=request.request_id,
            trip_id=request.trip_id,
            input_state_version=request.input_state_version,
            city_id=request.city_id,
            check_in=request.check_in,
            check_out=request.check_out,
            night_count=request.night_count,
            selected_strategy_id=strategy.strategy_id,
            status=DataAvailability.PARTIAL if degradation else DataAvailability.AVAILABLE,
            decision_status=(
                HotelDecisionStatus.FINAL
                if selected is not None
                else HotelDecisionStatus.AWAITING_USER
            ),
            candidates=candidates,
            selected_hotel_place_id=(selected.hotel_place_id if selected is not None else None),
            selection_source=selection_source,
            route_baseline=(_route_baseline(selected) if selected is not None else None),
            degradation_reasons=degradation,
            generated_at=generated_at,
        )

    def _aware_now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("hotel selection clock must return an aware datetime")
        return value.astimezone(UTC)


def _project_and_rank(
    request: HotelSelectionRequest,
    strategy: LodgingBaseStrategy,
    response: ProviderResponse[ProviderHotelOffer],
) -> tuple[HotelSelectionCandidate, ...]:
    price_bounds = _price_bounds(response.items)
    projected = [
        _project_offer(
            request,
            strategy,
            offer,
            price_bounds=price_bounds,
            preferred=(
                _hotel_place_id(request.city_id, offer.source_hotel_id)
                == request.chosen_hotel_place_id
            ),
        )
        for offer in response.items
    ]
    ranked = sorted(projected, key=_candidate_rank_key)
    return _diverse_representatives(ranked, request.candidate_count)


def _candidate_rank_key(candidate: HotelSelectionCandidate) -> tuple[object, ...]:
    return (
        -candidate.score.preference_boost,
        -candidate.score.total,
        candidate.distance_to_strategy_center_m or 10_000_000,
        candidate.name,
        str(candidate.hotel_place_id),
    )


def _diverse_representatives(
    ranked: list[HotelSelectionCandidate], target_count: int
) -> tuple[HotelSelectionCandidate, ...]:
    """Cover known price bands and provider hotel types before score-only fill."""

    selected_ids: set[UUID] = set()
    selected: list[HotelSelectionCandidate] = []

    def add(candidate: HotelSelectionCandidate) -> None:
        if candidate.hotel_place_id not in selected_ids and len(selected) < target_count:
            selected.append(candidate)
            selected_ids.add(candidate.hotel_place_id)

    seen_bands: set[HotelPriceBand] = set()
    for candidate in ranked:
        if candidate.price_band is not None and candidate.price_band not in seen_bands:
            add(candidate)
            seen_bands.add(candidate.price_band)

    seen_types: set[str] = set()
    for candidate in ranked:
        if candidate.hotel_type is None:
            continue
        normalized_type = _normalize_name(candidate.hotel_type)
        if normalized_type not in seen_types:
            add(candidate)
            seen_types.add(normalized_type)

    for candidate in ranked:
        add(candidate)

    return tuple(sorted(selected, key=_candidate_rank_key))


def _decide_hotel(
    request: HotelSelectionRequest,
    candidates: tuple[HotelSelectionCandidate, ...],
) -> HotelSelectionCandidate | None:
    if request.decision_mode is HotelDecisionMode.AWAIT_USER:
        return None
    if request.decision_mode is HotelDecisionMode.AGENT_DELEGATED:
        return candidates[0]
    if request.decision_mode is HotelDecisionMode.USER_CHOICE:
        selected = next(
            (item for item in candidates if item.hotel_place_id == request.chosen_hotel_place_id),
            None,
        )
        if selected is None:
            raise ValueError("chosen hotel is not in the current seven-hotel candidate set")
        return selected
    raise ValueError("non-fixed hotel search received an invalid decision mode")


def _selection_source(request: HotelSelectionRequest) -> HotelSelectionSource | None:
    if request.decision_mode is HotelDecisionMode.USER_CHOICE:
        return HotelSelectionSource.USER_CHOICE
    if request.decision_mode is HotelDecisionMode.AGENT_DELEGATED:
        return HotelSelectionSource.AGENT_DELEGATED
    return None


def _project_offer(
    request: HotelSelectionRequest,
    strategy: LodgingBaseStrategy,
    offer: ProviderHotelOffer,
    *,
    price_bounds: tuple[int, int] | None,
    preferred: bool,
) -> HotelSelectionCandidate:
    coordinates = (
        coordinates_to_gcj02(offer.raw_coordinates) if offer.raw_coordinates is not None else None
    )
    distance_m = _distance_m(strategy.center, coordinates) if coordinates is not None else None
    missing = set(offer.missing_fields)
    if offer.address is None:
        missing.add("address")
    if coordinates is None:
        missing.add("coordinates")
    if offer.room_price is None:
        missing.add("room_price")
        missing.add("price_band")
    if offer.hotel_type is None:
        missing.add("hotel_type")
    if offer.rating is None:
        missing.add("rating")
    if not offer.image_urls:
        missing.add("image_urls")
    if offer.detail_url is None:
        missing.add("detail_url")
    availability = DataAvailability.PARTIAL if missing else DataAvailability.AVAILABLE
    score = _score_offer(
        request,
        strategy,
        offer,
        distance_m=distance_m,
        missing_count=len(missing),
        price_bounds=price_bounds,
        preferred=preferred,
    )
    source_reference = _provider_source_reference(offer)
    return HotelSelectionCandidate(
        hotel_place_id=_hotel_place_id(request.city_id, offer.source_hotel_id),
        strategy_id=strategy.strategy_id,
        provider=offer.provider,
        source_hotel_id=offer.source_hotel_id,
        source_offer_id=offer.source_offer_id,
        name=offer.name,
        hotel_type=offer.hotel_type,
        brand_name=offer.brand_name,
        price_band=_price_band(offer.room_price),
        address=offer.address,
        coordinates=coordinates,
        check_in=request.check_in,
        check_out=request.check_out,
        night_count=request.night_count,
        availability=availability,
        room_price=offer.room_price,
        rating=offer.rating,
        image_urls=tuple(offer.image_urls),
        detail_url=offer.detail_url,
        distance_to_strategy_center_m=distance_m,
        missing_fields=tuple(sorted(missing)),
        missing_reason=("FlyAI did not return all hotel fields" if missing else None),
        score=score,
        fit_reason=_fit_reason(strategy, offer, distance_m, preferred),
        source_reference_ids=(source_reference,),
        fetched_at=offer.fetched_at,
    )


def _score_offer(
    request: HotelSelectionRequest,
    strategy: LodgingBaseStrategy,
    offer: ProviderHotelOffer,
    *,
    distance_m: int | None,
    missing_count: int,
    price_bounds: tuple[int, int] | None,
    preferred: bool,
) -> HotelSelectionScore:
    strategy_match = _strategy_match(distance_m)
    commute_fit = round((strategy.score.transport_fit * 2 + strategy_match) / 3)
    price_fit = _price_fit(offer.room_price, price_bounds)
    rating_fit = round(offer.rating * 20) if offer.rating is not None else 40
    completeness = max(0, 100 - missing_count * 16)
    preference_boost = 100 if preferred else 0
    quality_weight = 10 + request.preferences.quality_level**2 * 2
    value_weight = 10 + request.preferences.value_priority_level**2 * 3
    weights = {
        "strategy": 24,
        "commute": 18,
        "price": value_weight,
        "rating": quality_weight,
        "completeness": 16,
        "preference": 30 if preferred else 0,
    }
    weighted = (
        strategy_match * weights["strategy"]
        + commute_fit * weights["commute"]
        + price_fit * weights["price"]
        + rating_fit * weights["rating"]
        + completeness * weights["completeness"]
        + preference_boost * weights["preference"]
    )
    return HotelSelectionScore(
        strategy_match=strategy_match,
        commute_fit=commute_fit,
        price_fit=price_fit,
        rating_fit=rating_fit,
        fact_completeness=completeness,
        preference_boost=preference_boost,
        total=round(weighted / sum(weights.values())),
    )


def _price_band(value: CnyAmountRange | None) -> HotelPriceBand | None:
    if value is None:
        return None
    midpoint_fen = (value.minimum_fen + value.maximum_fen) // 2
    if midpoint_fen < 40_000:
        return HotelPriceBand.BUDGET
    if midpoint_fen < 70_000:
        return HotelPriceBand.MID_RANGE
    if midpoint_fen < 120_000:
        return HotelPriceBand.UPSCALE
    return HotelPriceBand.LUXURY


def _select_fixed_hotel(
    request: HotelSelectionRequest,
    strategy: LodgingBaseStrategy,
    response: ProviderResponse[ProviderHotelOffer],
    generated_at: datetime,
) -> HotelSelectionResult:
    fixed = request.fixed_hotel
    if fixed is None:
        raise ValueError("fixed hotel selection is missing booked hotel identity")
    expected = _normalize_name(fixed.name)
    matched = next(
        (offer for offer in response.items if _normalize_name(offer.name) == expected),
        None,
    )
    if matched is None:
        return _fixed_without_offer(
            request,
            strategy,
            generated_at,
            reason="FlyAI did not return the already booked hotel; no replacement was selected",
        )
    projected = _project_offer(
        request,
        strategy,
        matched,
        price_bounds=_price_bounds((matched,)),
        preferred=False,
    )
    missing = tuple(field for field in projected.missing_fields if field != "coordinates")
    candidate = HotelSelectionCandidate.model_validate(
        {
            **projected.model_dump(mode="python"),
            "hotel_place_id": fixed.place_id,
            "coordinates": fixed.coordinates,
            "distance_to_strategy_center_m": 0,
            "source_reference_ids": tuple(
                sorted({*fixed.source_reference_ids, *projected.source_reference_ids})
            ),
            "fit_reason": "这是用户已经预订的酒店；FlyAI 仅补充本次日期的可用资料。",
            "missing_fields": missing,
            "availability": DataAvailability.PARTIAL if missing else DataAvailability.AVAILABLE,
            "missing_reason": "FlyAI did not return all hotel fields" if missing else None,
        }
    )
    degradation = _result_degradation(response, candidate, (candidate,), None)
    return HotelSelectionResult(
        algorithm_version=HOTEL_SELECTION_ALGORITHM_VERSION,
        request_id=request.request_id,
        trip_id=request.trip_id,
        input_state_version=request.input_state_version,
        city_id=request.city_id,
        check_in=request.check_in,
        check_out=request.check_out,
        night_count=request.night_count,
        selected_strategy_id=strategy.strategy_id,
        status=DataAvailability.PARTIAL if degradation else DataAvailability.AVAILABLE,
        decision_status=HotelDecisionStatus.FINAL,
        candidates=(candidate,),
        selected_hotel_place_id=candidate.hotel_place_id,
        selection_source=HotelSelectionSource.PREBOOKED,
        route_baseline=_route_baseline(candidate),
        degradation_reasons=degradation,
        generated_at=generated_at,
    )


def _fixed_without_offer(
    request: HotelSelectionRequest,
    strategy: LodgingBaseStrategy,
    generated_at: datetime,
    *,
    reason: str,
    failure_code: ProviderFailureCode | None = None,
) -> HotelSelectionResult:
    fixed = request.fixed_hotel
    if fixed is None:
        raise ValueError("fixed hotel fallback is missing booked hotel identity")
    candidate = HotelSelectionCandidate(
        hotel_place_id=fixed.place_id,
        strategy_id=strategy.strategy_id,
        provider=ProviderCode.MANUAL,
        source_hotel_id=f"fixed:{fixed.node_id}",
        source_offer_id=f"fixed:{fixed.node_id}",
        name=fixed.name,
        coordinates=fixed.coordinates,
        check_in=request.check_in,
        check_out=request.check_out,
        night_count=request.night_count,
        availability=DataAvailability.PARTIAL,
        missing_fields=(
            "hotel_type",
            "price_band",
            "address",
            "room_price",
            "rating",
            "image_urls",
            "detail_url",
        ),
        missing_reason=reason,
        score=HotelSelectionScore(
            strategy_match=100,
            commute_fit=strategy.score.transport_fit,
            price_fit=0,
            rating_fit=0,
            fact_completeness=17,
            preference_boost=100,
            total=60,
        ),
        fit_reason="这是用户已订酒店，商品资料缺失不会触发换店。",
        source_reference_ids=fixed.source_reference_ids,
        fetched_at=generated_at,
    )
    return HotelSelectionResult(
        algorithm_version=HOTEL_SELECTION_ALGORITHM_VERSION,
        request_id=request.request_id,
        trip_id=request.trip_id,
        input_state_version=request.input_state_version,
        city_id=request.city_id,
        check_in=request.check_in,
        check_out=request.check_out,
        night_count=request.night_count,
        selected_strategy_id=strategy.strategy_id,
        status=DataAvailability.PARTIAL,
        decision_status=HotelDecisionStatus.FINAL,
        candidates=(candidate,),
        selected_hotel_place_id=candidate.hotel_place_id,
        selection_source=HotelSelectionSource.PREBOOKED,
        route_baseline=_route_baseline(candidate),
        provider_failure_code=failure_code,
        degradation_reasons=(reason,),
        generated_at=generated_at,
    )


def _missing_result(
    request: HotelSelectionRequest,
    strategy: LodgingBaseStrategy,
    generated_at: datetime,
    *,
    reason: str,
    failure_code: ProviderFailureCode | None = None,
) -> HotelSelectionResult:
    return HotelSelectionResult(
        algorithm_version=HOTEL_SELECTION_ALGORITHM_VERSION,
        request_id=request.request_id,
        trip_id=request.trip_id,
        input_state_version=request.input_state_version,
        city_id=request.city_id,
        check_in=request.check_in,
        check_out=request.check_out,
        night_count=request.night_count,
        selected_strategy_id=strategy.strategy_id,
        status=DataAvailability.MISSING,
        decision_status=HotelDecisionStatus.AWAITING_USER,
        provider_failure_code=failure_code,
        missing_reason=reason,
        generated_at=generated_at,
    )


def _day_trip_result(
    request: HotelSelectionRequest, generated_at: datetime
) -> HotelSelectionResult:
    return HotelSelectionResult(
        algorithm_version=HOTEL_SELECTION_ALGORITHM_VERSION,
        request_id=request.request_id,
        trip_id=request.trip_id,
        input_state_version=request.input_state_version,
        city_id=request.city_id,
        check_in=request.check_in,
        check_out=request.check_out,
        night_count=0,
        status=DataAvailability.AVAILABLE,
        decision_status=HotelDecisionStatus.NOT_REQUIRED,
        generated_at=generated_at,
    )


def _selected_strategy(request: HotelSelectionRequest) -> LodgingBaseStrategy:
    strategy_id = request.lodging_result.selected_strategy_id
    strategy = next(
        (item for item in request.lodging_result.strategies if item.strategy_id == strategy_id),
        None,
    )
    if strategy is None:
        raise ValueError("selected lodging strategy is missing from its result")
    return strategy


def _route_baseline(candidate: HotelSelectionCandidate) -> HotelRouteBaseline:
    return HotelRouteBaseline(
        hotel_place_id=candidate.hotel_place_id,
        strategy_id=candidate.strategy_id,
        availability=(
            DataAvailability.AVAILABLE
            if candidate.coordinates is not None
            else DataAvailability.MISSING
        ),
        coordinates=candidate.coordinates,
        source_reference_ids=candidate.source_reference_ids,
        missing_reason=(
            None
            if candidate.coordinates is not None
            else "selected hotel has no GCJ-02 coordinates"
        ),
    )


def _result_degradation(
    response: ProviderResponse[ProviderHotelOffer],
    selected: HotelSelectionCandidate | None,
    candidates: Iterable[HotelSelectionCandidate],
    target_count: int | None,
) -> tuple[str, ...]:
    candidate_list = tuple(candidates)
    reasons: list[str] = []
    if response.status is ProviderResultStatus.PARTIAL:
        reasons.append(response.provider_notice or "FlyAI returned partial hotel data")
    if selected is not None and selected.availability is not DataAvailability.AVAILABLE:
        reasons.append(selected.missing_reason or "selected hotel facts are partial")
    if selected is not None and selected.coordinates is None:
        reasons.append("selected hotel has no GCJ-02 route baseline")
    if any(item.availability is not DataAvailability.AVAILABLE for item in candidate_list):
        reasons.append("one or more hotel candidates have item-specific missing fields")
    if target_count is not None and len(candidate_list) < target_count:
        reasons.append(
            f"FlyAI returned only {len(candidate_list)} usable hotels; target is {target_count}"
        )
    return tuple(dict.fromkeys(reasons))


def _provider_source_reference(offer: ProviderHotelOffer) -> str:
    return (
        f"provider:{offer.provider.value}:hotel:{offer.source_hotel_id}:"
        f"{offer.fetched_at.astimezone(UTC).isoformat()}"
    )


def _hotel_place_id(city_id: str, source_hotel_id: str) -> UUID:
    return uuid5(NAMESPACE_URL, f"iter:hotel:flyai:{city_id}:{source_hotel_id}")


def _price_bounds(offers: Iterable[ProviderHotelOffer]) -> tuple[int, int] | None:
    midpoints = [
        (item.room_price.minimum_fen + item.room_price.maximum_fen) // 2
        for item in offers
        if item.room_price is not None
    ]
    return (min(midpoints), max(midpoints)) if midpoints else None


def _price_fit(value: CnyAmountRange | None, bounds: tuple[int, int] | None) -> int:
    if value is None or bounds is None:
        return 30
    low, high = bounds
    midpoint = (value.minimum_fen + value.maximum_fen) / 2
    if high == low:
        return 75
    return max(0, min(100, round(100 - (midpoint - low) / (high - low) * 70)))


def _strategy_match(distance_m: int | None) -> int:
    if distance_m is None:
        return 45
    return max(0, min(100, round(105 - distance_m / 45)))


def _fit_reason(
    strategy: LodgingBaseStrategy,
    offer: ProviderHotelOffer,
    distance_m: int | None,
    preferred: bool,
) -> str:
    distance = f"距离策略中心约 {distance_m} 米" if distance_m is not None else "位置距离待补充"
    rating = f"评分 {offer.rating:.1f}" if offer.rating is not None else "评分缺失"
    prefix = "用户在候选中表达了偏好；" if preferred else ""
    return f"{prefix}{distance}，{rating}，符合“{strategy.label}”住宿基地。"


def _distance_m(left: Gcj02Coordinates, right: Gcj02Coordinates) -> int:
    latitude_delta = radians(right.latitude - left.latitude)
    longitude_delta = radians(right.longitude - left.longitude)
    left_latitude = radians(left.latitude)
    right_latitude = radians(right.latitude)
    haversine = (
        sin(latitude_delta / 2) ** 2
        + cos(left_latitude) * cos(right_latitude) * sin(longitude_delta / 2) ** 2
    )
    return round(6_371_000 * 2 * asin(sqrt(haversine)))


def _normalize_name(value: str) -> str:
    return re.sub(r"[\W_]+", "", value, flags=re.UNICODE).casefold()
