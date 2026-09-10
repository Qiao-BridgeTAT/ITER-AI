"""Prepare Agent capability for materializing one signed V4 discovery card."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TypeAlias
from uuid import NAMESPACE_URL, UUID, uuid5

from backend.agent.model_gateway import ModelCancellation
from backend.contracts.common import CnyAmountRange
from backend.contracts.enums import PlaceCategory, ProviderCode
from backend.contracts.v4.cards import (
    AttractionPreferenceCard,
    CardControlAction,
    CardEntityRef,
    CardGenerationMetadata,
    CardOption,
    CardSemanticValue,
    DiningPreferenceCard,
    DirectionSemanticValue,
    LodgingAreaPreferenceCard,
    LodgingClassPreferenceCard,
    SpecificCandidateCard,
)
from backend.contracts.v4.enums import (
    CardDomain,
    CardKind,
    CardStatus,
    CompositionRole,
    DiscoverySection,
)
from backend.contracts.v4.state import TripSemanticState
from backend.discovery.cards.candidate_composition import (
    CandidateCompositionService,
    CardGenerationError,
    specific_dependency_fingerprint,
)
from backend.discovery.cards.preference_generation import (
    DirectionDraft,
    LodgingAreaDirectionDraft,
    PreferenceDirectionGenerator,
)
from backend.persistence.outbox_repository import canonical_json_hash
from backend.planning.city_registry import CityRegistry
from backend.providers.contracts import (
    HotelSearchRequest,
    KeywordPlaceSearchRequest,
    ProviderCityScope,
    ProviderError,
    ProviderHotelOffer,
    ProviderPlace,
    ProviderResponse,
)
from backend.providers.interfaces import PlaceProvider, TravelProductProvider
from backend.providers.place_taxonomy import category_from_original_typecodes, original_typecodes

CardAttachment: TypeAlias = (
    AttractionPreferenceCard
    | DiningPreferenceCard
    | LodgingAreaPreferenceCard
    | LodgingClassPreferenceCard
    | SpecificCandidateCard
)

_LODGING_AREA_OTHER_TYPE_PREFIXES = ("1901", "1201", "1203")
_LODGING_AREA_NAMED_TYPES = {"190301", "190700"}
_LODGING_AREA_MALL_TYPE_PREFIXES = ("060101", "060102")
_LODGING_AREA_COMMERCIAL_STREET_TYPES = {"061000", "061001"}
_LODGING_TRANSIT_WHOLE_TYPES = {"150100", "150200", "150500", "150700"}
_LODGING_AREA_FORBIDDEN_NAME_TOKENS = (
    "酒店",
    "宾馆",
    "饭店",
    "客栈",
    "民宿",
    "公寓",
    "hotel",
    "餐厅",
    "餐馆",
    "小吃",
    "珠宝",
    "黄金",
    "汽车",
    "4s店",
    "旗舰店",
    "专卖店",
    "便利店",
    "超市",
    "公司",
    "写字楼",
    "大厦",
)
_TRANSIT_ANCHOR_TOKENS = ("地铁站", "公交站", "车站", "火车站", "高铁站", "机场", "枢纽")
_LODGING_AREA_TARGET_COUNT = 6
_LODGING_AREA_MINIMUM_COUNT = 3
_LODGING_AREA_PLAN_ATTEMPTS = 2


class PrepareCardService:
    """Generate content dynamically, then bind it to real facts and signed IDs."""

    def __init__(
        self,
        *,
        directions: PreferenceDirectionGenerator,
        candidates: CandidateCompositionService,
        registry: CityRegistry,
        places: PlaceProvider,
        products: TravelProductProvider | None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        lodging_observer: Callable[[dict[str, object]], None] | None = None,
    ) -> None:
        self._directions = directions
        self._candidates = candidates
        self._registry = registry
        self._places = places
        self._products = products
        self._clock = clock
        self._lodging_observer = lodging_observer

    async def generate(
        self,
        state: TripSemanticState,
        *,
        section: DiscoverySection,
        turn_id: UUID,
        cancellation: ModelCancellation | None = None,
    ) -> CardAttachment:
        interaction_id = uuid5(
            NAMESPACE_URL,
            f"v4-card-interaction:{turn_id}:{section.value}:{state.state_version}",
        )
        attachment_id = uuid5(
            NAMESPACE_URL,
            f"v4-card-attachment:{interaction_id}",
        )
        if section is DiscoverySection.ATTRACTION_PREFERENCE:
            return await self._attraction_preference(
                state,
                interaction_id,
                attachment_id,
                cancellation,
            )
        if section is DiscoverySection.ATTRACTION_SPECIFIC:
            result = await self._candidates.compose(
                state,
                domain=CardDomain.ATTRACTION,
                interaction_id=interaction_id,
                attachment_id=attachment_id,
                cancellation=cancellation,
            )
            return result.card.model_copy(
                update={"control_actions": _specific_controls(interaction_id, "attraction")},
                deep=True,
            )
        if section is DiscoverySection.DINING_PREFERENCE:
            return await self._dining_preference(
                state,
                interaction_id,
                attachment_id,
                cancellation,
            )
        if section is DiscoverySection.DINING_SPECIFIC:
            result = await self._candidates.compose(
                state,
                domain=CardDomain.DINING,
                interaction_id=interaction_id,
                attachment_id=attachment_id,
                cancellation=cancellation,
            )
            return result.card.model_copy(
                update={"control_actions": _specific_controls(interaction_id, "dining")},
                deep=True,
            )
        if section is DiscoverySection.LODGING_AREA_PREFERENCE:
            return await self._lodging_area(
                state,
                interaction_id,
                attachment_id,
                cancellation,
            )
        if section is DiscoverySection.LODGING_CLASS_PREFERENCE:
            return await self._lodging_class(
                state,
                interaction_id,
                attachment_id,
                cancellation,
            )
        raise CardGenerationError(f"section {section.value} does not own a card")

    def dependency_fingerprint(
        self,
        state: TripSemanticState,
        section: DiscoverySection,
    ) -> str:
        if section is DiscoverySection.ATTRACTION_SPECIFIC:
            return specific_dependency_fingerprint(state, CardDomain.ATTRACTION)
        if section is DiscoverySection.DINING_SPECIFIC:
            return specific_dependency_fingerprint(state, CardDomain.DINING)
        domains = {
            DiscoverySection.ATTRACTION_PREFERENCE: "attraction",
            DiscoverySection.DINING_PREFERENCE: "dining",
            DiscoverySection.LODGING_AREA_PREFERENCE: "lodging_area",
            DiscoverySection.LODGING_CLASS_PREFERENCE: "lodging_class",
        }
        try:
            return _preference_fingerprint(state, domains[section])
        except KeyError as error:
            raise CardGenerationError("section does not own a card fingerprint") from error

    async def _attraction_preference(
        self,
        state: TripSemanticState,
        interaction_id: UUID,
        attachment_id: UUID,
        cancellation: ModelCancellation | None,
    ) -> AttractionPreferenceCard:
        plan, mode = await self._directions.generate_attraction(
            state,
            cancellation=cancellation,
        )
        now = self._now()
        plan_ref = str(uuid5(NAMESPACE_URL, f"v4-direction-plan:{interaction_id}"))
        return AttractionPreferenceCard(
            attachment_id=str(attachment_id),
            interaction_id=str(interaction_id),
            kind=CardKind.PREFERENCE_CARD,
            domain=CardDomain.ATTRACTION,
            section=DiscoverySection.ATTRACTION_PREFERENCE,
            based_on_state_version=state.state_version,
            dependency_fingerprint=_preference_fingerprint(state, "attraction"),
            status=CardStatus.ACTIVE,
            options=[
                _direction_option(item, interaction_id=interaction_id, source_refs=[plan_ref])
                for item in plan.directions
            ],
            control_actions=_preference_controls(interaction_id, "attraction"),
            generation_metadata=CardGenerationMetadata(
                generated_at=now,
                source_refs=[plan_ref],
                model_plan_id=plan_ref,
                generation_mode=mode,
                strategy_version="attraction-v2",
            ),
            prompt="喜欢的方向可以多选，不感兴趣的也可以排除。",
        )

    async def _dining_preference(
        self,
        state: TripSemanticState,
        interaction_id: UUID,
        attachment_id: UUID,
        cancellation: ModelCancellation | None,
    ) -> DiningPreferenceCard:
        plan, mode = await self._directions.generate_dining(
            state,
            cancellation=cancellation,
        )
        now = self._now()
        plan_ref = str(uuid5(NAMESPACE_URL, f"v4-direction-plan:{interaction_id}"))
        return DiningPreferenceCard(
            attachment_id=str(attachment_id),
            interaction_id=str(interaction_id),
            kind=CardKind.PREFERENCE_CARD,
            domain=CardDomain.DINING,
            section=DiscoverySection.DINING_PREFERENCE,
            based_on_state_version=state.state_version,
            dependency_fingerprint=_preference_fingerprint(state, "dining"),
            status=CardStatus.ACTIVE,
            options=[
                _direction_option(item, interaction_id=interaction_id, source_refs=[plan_ref])
                for item in plan.directions
            ],
            control_actions=_preference_controls(interaction_id, "dining"),
            generation_metadata=CardGenerationMetadata(
                generated_at=now,
                source_refs=[plan_ref],
                model_plan_id=plan_ref,
                generation_mode=mode,
            ),
            prompt="挑挑你喜欢的口味；有忌口或过敏，也记得告诉我。",
        )

    async def _lodging_area(
        self,
        state: TripSemanticState,
        interaction_id: UUID,
        attachment_id: UUID,
        cancellation: ModelCancellation | None,
    ) -> LodgingAreaPreferenceCard:
        if state.lodging.not_applicable or state.lodging.existing_bookings:
            raise CardGenerationError("lodging area card is bypassed by current lodging state")
        plan, mode = await self._directions.generate_lodging_area(
            state,
            cancellation=cancellation,
        )
        basics = state.trip_basics
        if basics.destination_canonical_id is None:
            raise CardGenerationError("lodging area card requires a canonical destination")
        scope = self._registry.provider_scope(
            basics.destination_canonical_id,
            ProviderCode.AMAP,
        )
        options: list[CardOption] = []
        used_places: set[str] = set()
        used_names: set[str] = set()
        source_refs: list[str] = []
        observed: list[datetime] = []
        resolved: list[dict[str, object]] = []
        failed: list[dict[str, object]] = []
        attempted_queries: list[str] = []
        attempted_query_identities: set[str] = set()
        used_safe_seed_fallback = mode == "safe_seed_fallback"
        for attempt in range(_LODGING_AREA_PLAN_ATTEMPTS):
            new_directions: list[LodgingAreaDirectionDraft] = []
            for direction in plan.directions:
                query_identity = _lodging_area_identity(
                    direction.search_query,
                    scope.display_name,
                )
                if query_identity in attempted_query_identities:
                    failed.append(
                        {
                            "query": direction.search_query,
                            "label": direction.label,
                            "reason": "duplicate_or_already_attempted_query",
                            "observations": [],
                        }
                    )
                    continue
                attempted_query_identities.add(query_identity)
                attempted_queries.append(direction.search_query)
                new_directions.append(direction)
            results = await asyncio.gather(
                *(
                    self._search_lodging_area_with_retry(scope, direction.search_query)
                    for direction in new_directions
                ),
                return_exceptions=True,
            )
            for direction, response in zip(new_directions, results, strict=True):
                if len(options) >= _LODGING_AREA_TARGET_COUNT:
                    break
                items = [] if isinstance(response, BaseException) else response.items
                place = _first_unused_lodging_area(
                    items,
                    used_places,
                    used_names=used_names,
                    query=direction.search_query,
                    scope=scope,
                )
                if place is None:
                    failed.append(
                        {
                            "query": direction.search_query,
                            "label": direction.label,
                            "reason": (
                                "provider_unavailable"
                                if isinstance(response, BaseException)
                                else "no_distinct_matching_area"
                            ),
                            "observations": [
                                {
                                    "name": item.name,
                                    "original_type": item.provider_typecode,
                                    "area_kind_valid": _is_lodging_area_place(item),
                                    "query_identity_matches": _lodging_area_identity(
                                        item.name, scope.display_name
                                    )
                                    == _lodging_area_identity(
                                        direction.search_query, scope.display_name
                                    ),
                                }
                                for item in items[:12]
                            ],
                        }
                    )
                    continue
                used_places.add(place.source_place_id)
                used_names.add(_lodging_area_identity(place.name, scope.display_name))
                option = _lodging_area_option(direction, place, interaction_id)
                options.append(option)
                source_refs.extend(option.source_refs)
                observed.append(place.fetched_at)
                resolved.append(
                    {
                        "direction": direction.model_dump(mode="json"),
                        "name": place.name,
                        "original_type": place.provider_typecode,
                        "source_ref": option.source_refs[0],
                    }
                )
            feedback: dict[str, object] = {
                "target_option_count": _LODGING_AREA_TARGET_COUNT,
                "missing_option_count": max(
                    0,
                    _LODGING_AREA_TARGET_COUNT - len(options),
                ),
                "resolved_directions": resolved,
                "failed_queries": failed,
                "attempted_queries": attempted_queries,
            }
            if self._lodging_observer is not None:
                self._lodging_observer(
                    {"city": basics.destination_name, "attempt": attempt + 1, **feedback}
                )
            if (
                len(options) >= _LODGING_AREA_TARGET_COUNT
                or attempt == _LODGING_AREA_PLAN_ATTEMPTS - 1
            ):
                break
            plan, mode = await self._directions.generate_lodging_area(
                state,
                provider_feedback=feedback,
                cancellation=cancellation,
            )
            used_safe_seed_fallback = used_safe_seed_fallback or mode == "safe_seed_fallback"
        if len(options) < _LODGING_AREA_MINIMUM_COUNT:
            raise CardGenerationError(
                "fewer than three distinct real lodging areas were resolved",
                code="lodging_areas_insufficient",
                recoverable=True,
            )
        options = options[:_LODGING_AREA_TARGET_COUNT]
        now = max(observed) if observed else self._now()
        plan_ref = str(uuid5(NAMESPACE_URL, f"v4-direction-plan:{interaction_id}"))
        return LodgingAreaPreferenceCard(
            attachment_id=str(attachment_id),
            interaction_id=str(interaction_id),
            kind=CardKind.PREFERENCE_CARD,
            domain=CardDomain.LODGING_AREA,
            section=DiscoverySection.LODGING_AREA_PREFERENCE,
            based_on_state_version=state.state_version,
            dependency_fingerprint=_preference_fingerprint(state, "lodging_area"),
            status=(
                CardStatus.ACTIVE
                if len(options) == _LODGING_AREA_TARGET_COUNT
                else CardStatus.PARTIAL_AVAILABILITY
            ),
            options=options,
            control_actions=_lodging_controls(interaction_id, "lodging_area"),
            generation_metadata=CardGenerationMetadata(
                generated_at=now,
                source_refs=list(dict.fromkeys([plan_ref, *source_refs])),
                model_plan_id=plan_ref,
                generation_mode=(
                    "safe_seed_fallback" if used_safe_seed_fallback else "provider_composed"
                ),
            ),
            prompt="先看看想住在哪一带，具体酒店等路线定好后再选。",
        )

    async def _search_lodging_area(
        self,
        scope: ProviderCityScope,
        query: str,
    ) -> ProviderResponse[ProviderPlace]:
        return await self._places.search_places(
            KeywordPlaceSearchRequest(
                city=scope,
                query=query,
                category_hint=PlaceCategory.OTHER,
                # Coarse types may return unrelated businesses ahead of this
                # keyword. Validate the original source type after retrieval.
                typecodes=[],
                page_size=12,
            )
        )

    async def _search_lodging_area_with_retry(
        self,
        scope: ProviderCityScope,
        query: str,
    ) -> ProviderResponse[ProviderPlace]:
        try:
            return await self._search_lodging_area(scope, query)
        except ProviderError as error:
            if not error.retryable:
                raise
            await asyncio.sleep(0.2)
            return await self._search_lodging_area(scope, query)

    async def _lodging_class(
        self,
        state: TripSemanticState,
        interaction_id: UUID,
        attachment_id: UUID,
        cancellation: ModelCancellation | None,
    ) -> LodgingClassPreferenceCard:
        if state.lodging.not_applicable or state.lodging.existing_bookings:
            raise CardGenerationError("lodging class card is bypassed by current lodging state")
        plan, mode = await self._directions.generate_lodging_class(
            state,
            cancellation=cancellation,
        )
        ranges, price_ref, observed_at = await self._hotel_price_buckets(state)
        plan_ref = str(uuid5(NAMESPACE_URL, f"v4-direction-plan:{interaction_id}"))
        options: list[CardOption] = []
        for item in plan.directions:
            budget = ranges.get(item.direction_id)
            quality_tier = item.direction_id if item.direction_id != "boutique_resort" else None
            description = item.description
            source_refs = [plan_ref]
            if budget is not None and observed_at is not None and price_ref is not None:
                description += (
                    f" 参考 ¥{budget.minimum_fen // 100}–¥{budget.maximum_fen // 100}/晚。"
                )
                source_refs.append(price_ref)
            elif item.direction_id != "boutique_resort":
                description += " 参考价暂缺。"
            option_id = str(
                uuid5(
                    NAMESPACE_URL,
                    f"v4-card-option:{interaction_id}:{item.direction_id}",
                )
            )
            options.append(
                CardOption(
                    option_id=option_id,
                    label=item.label,
                    description=description,
                    semantic_value=CardSemanticValue(
                        root=DirectionSemanticValue(
                            kind="direction",
                            direction_id=item.direction_id,
                            direction_kind="hotel_class",
                            hotel_quality_tier=quality_tier,
                            property_type=(
                                "boutique_or_resort"
                                if item.direction_id == "boutique_resort"
                                else None
                            ),
                            nightly_budget_minimum_minor=(
                                budget.minimum_fen if budget is not None else None
                            ),
                            nightly_budget_maximum_minor=(
                                budget.maximum_fen if budget is not None else None
                            ),
                        )
                    ),
                    signed_operation_ref=str(
                        uuid5(
                            NAMESPACE_URL,
                            f"v4-card-operation:{interaction_id}:{option_id}",
                        )
                    ),
                    source_refs=source_refs,
                    observed_at=observed_at,
                    composition_role=CompositionRole.PERSONALIZED_TOP,
                )
            )
        return LodgingClassPreferenceCard(
            attachment_id=str(attachment_id),
            interaction_id=str(interaction_id),
            kind=CardKind.PREFERENCE_CARD,
            domain=CardDomain.LODGING_CLASS,
            section=DiscoverySection.LODGING_CLASS_PREFERENCE,
            based_on_state_version=state.state_version,
            dependency_fingerprint=_preference_fingerprint(state, "lodging_class"),
            status=CardStatus.ACTIVE,
            options=options,
            control_actions=_lodging_controls(interaction_id, "lodging_class"),
            generation_metadata=CardGenerationMetadata(
                generated_at=self._now(),
                source_refs=list(
                    dict.fromkeys([plan_ref, *([price_ref] if price_ref is not None else [])])
                ),
                model_plan_id=plan_ref,
                generation_mode=(
                    "provider_composed" if mode == "qwen" and price_ref is not None else mode
                ),
            ),
            prompt="挑一种住着舒服的类型吧。有每晚预算，也可以直接告诉我。",
        )

    async def _hotel_price_buckets(
        self,
        state: TripSemanticState,
    ) -> tuple[dict[str, CnyAmountRange], str | None, datetime | None]:
        basics = state.trip_basics
        if (
            self._products is None
            or basics.destination_canonical_id is None
            or basics.start_date is None
            or basics.end_date is None
            or basics.end_date <= basics.start_date
        ):
            return {}, None, None
        try:
            response = await self._products.search_hotels(
                HotelSearchRequest(
                    city=self._registry.provider_scope(
                        basics.destination_canonical_id,
                        ProviderCode.FLYAI,
                    ),
                    check_in=basics.start_date,
                    check_out=basics.end_date,
                )
            )
        except (ProviderError, ValueError):
            return {}, None, None
        priced_offers = [item for item in response.items if item.room_price is not None]
        if not priced_offers:
            return {}, None, response.fetched_at
        buckets = _price_buckets(priced_offers)
        if not buckets:
            return {}, None, response.fetched_at
        source_key = (
            response.source_request_id
            or canonical_json_hash(
                {
                    "provider": response.provider.value,
                    "city": basics.destination_canonical_id,
                    "check_in": basics.start_date.isoformat(),
                    "check_out": basics.end_date.isoformat(),
                    "observed_at": response.fetched_at.isoformat(),
                    "sample_count": len(priced_offers),
                    "observed_tiers": sorted(buckets),
                }
            )[:24]
        )
        return (
            buckets,
            f"provider:{response.provider.value}:hotel-price-sample:{source_key}",
            response.fetched_at,
        )

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("card clock must be timezone-aware")
        return value.astimezone(UTC)


def _direction_option(
    item: DirectionDraft,
    *,
    interaction_id: UUID,
    source_refs: list[str],
) -> CardOption:
    option_id = str(uuid5(NAMESPACE_URL, f"v4-card-option:{interaction_id}:{item.direction_id}"))
    return CardOption(
        option_id=option_id,
        label=item.label,
        description=item.description,
        semantic_value=CardSemanticValue(
            root=DirectionSemanticValue(
                kind="direction",
                direction_id=item.direction_id,
                direction_kind=(
                    "local_representative"
                    if item.composition_role is CompositionRole.REPRESENTATIVE_EXTRA
                    else "personalized"
                ),
                tags=item.tags,
                search_query=item.search_query,
            )
        ),
        signed_operation_ref=str(
            uuid5(NAMESPACE_URL, f"v4-card-operation:{interaction_id}:{option_id}")
        ),
        source_refs=source_refs,
        composition_role=item.composition_role,
    )


def _lodging_area_option(
    direction: LodgingAreaDirectionDraft,
    place: ProviderPlace,
    interaction_id: UUID,
) -> CardOption:
    canonical_id = str(uuid5(NAMESPACE_URL, f"{place.provider.value}:{place.source_place_id}"))
    source_ref = f"provider:{place.provider.value}:{place.source_place_id}"
    option_id = str(uuid5(NAMESPACE_URL, f"v4-card-option:{interaction_id}:{canonical_id}"))
    is_transit = place.category is PlaceCategory.TRANSPORT
    return CardOption(
        option_id=option_id,
        label=f"{direction.label} · {place.name}",
        description=direction.description,
        entity_ref=CardEntityRef(
            canonical_entity_id=canonical_id,
            entity_kind="transit_node" if is_transit else "area",
            provider_entity_refs=[source_ref],
        ),
        semantic_value=CardSemanticValue(
            root=DirectionSemanticValue(
                kind="direction",
                direction_id=direction.direction_id,
                direction_kind="area_strategy",
                tags=[canonical_id],
            )
        ),
        signed_operation_ref=str(
            uuid5(NAMESPACE_URL, f"v4-card-operation:{interaction_id}:{option_id}")
        ),
        source_refs=[source_ref],
        observed_at=place.fetched_at,
        composition_role=CompositionRole.PERSONALIZED_TOP,
    )


def _is_lodging_area_place(place: ProviderPlace) -> bool:
    codes = original_typecodes(place.provider_typecode)
    if not codes or place.category is not category_from_original_typecodes(place.provider_typecode):
        return False
    normalized = "".join(place.name.casefold().split())
    if any(token in normalized for token in _LODGING_AREA_FORBIDDEN_NAME_TOKENS):
        return False
    if re.search(r"[（(][^）)]*店[）)]|(?:门店|商店|出入口|入口|出口)$", normalized):
        return False
    if re.search(r"[a-z0-9][东南西北]*口|进站口|出站口|检票口|停车场|候车室", normalized):
        return False
    if re.search(r"暂停营业|暂停开放|已停业|已关闭|歇业|已搬迁", normalized):
        return False
    if place.category is PlaceCategory.TRANSPORT:
        # Whole stations only: a 150501 exit or 150202 ticket gate is not an area.
        return (
            any(code in _LODGING_TRANSIT_WHOLE_TYPES for code in codes)
            and not any(
                code.startswith("15") and code not in _LODGING_TRANSIT_WHOLE_TYPES for code in codes
            )
        ) and (
            normalized.endswith("站")
            or any(token in normalized for token in _TRANSIT_ANCHOR_TOKENS)
        )
    if place.category is not PlaceCategory.OTHER:
        return False
    if any(
        code.startswith(("05", "10", "11", "15")) or code[:4] in {"1401", "1404", "1406", "1407"}
        for code in codes
    ):
        return False
    if any(code.startswith("1901") or code in _LODGING_AREA_NAMED_TYPES for code in codes):
        return True
    if any(code in _LODGING_AREA_COMMERCIAL_STREET_TYPES for code in codes):
        return bool(re.search(r"(?:街|街区|商圈|步行区)$", normalized))
    if any(code.startswith(_LODGING_AREA_OTHER_TYPE_PREFIXES) for code in codes):
        return any(code.startswith("1901") for code in codes) or any(
            token in normalized for token in ("园区", "片区", "社区", "小区", "住宅区")
        )
    # An actual mall name need not contain 商场/广场 (e.g. a named commercial
    # complex). Trust its specific original mall type, never a category hint.
    return any(code.startswith(_LODGING_AREA_MALL_TYPE_PREFIXES) for code in codes) and not (
        normalized.endswith("店")
    )


def _first_unused_lodging_area(
    places: list[ProviderPlace],
    used_places: set[str],
    *,
    used_names: set[str],
    query: str,
    scope: ProviderCityScope,
) -> ProviderPlace | None:
    expected = _lodging_area_identity(query, scope.display_name)
    if len(expected) < 2:
        return None
    return next(
        (
            item
            for item in places
            if item.source_place_id not in used_places
            and item.city_id == scope.city_id
            and _is_lodging_area_place(item)
            and _lodging_area_identity(item.name, scope.display_name) == expected
            and expected not in used_names
        ),
        None,
    )


def _lodging_area_identity(value: str, city_name: str | None) -> str:
    """Only normalize city prefixes and area/station descriptors, not fuzzy substrings."""

    normalized = "".join(value.casefold().split())
    normalized = re.sub(r"[（(](?:地铁站|公交站|火车站)[）)]", "", normalized)
    city = (city_name or "").removesuffix("市")
    if city and normalized.startswith(city):
        normalized = normalized[len(city) :].removeprefix("市")
    return re.sub(r"(?:地铁站|公交站|片区|商圈|街道)$", "", normalized)


def _price_buckets(offers: list[ProviderHotelOffer]) -> dict[str, CnyAmountRange]:
    """Use the Provider's actual grade; price quantiles are never hotel grades."""

    observed_grades = {
        "经济型": "economy",
        "经济": "economy",
        "舒适型": "comfort",
        "舒适": "comfort",
        "高档型": "upscale",
        "高档": "upscale",
        "豪华型": "luxury",
        "豪华": "luxury",
        "economy": "economy",
        "comfort": "comfort",
        "upscale": "upscale",
        "luxury": "luxury",
    }
    grouped: dict[str, list[CnyAmountRange]] = {}
    for offer in offers:
        grade = observed_grades.get((offer.hotel_type or "").strip().casefold())
        if grade is not None and offer.room_price is not None:
            grouped.setdefault(grade, []).append(offer.room_price)
    return {
        grade: CnyAmountRange(
            minimum_fen=min(price.minimum_fen for price in prices),
            maximum_fen=max(price.maximum_fen for price in prices),
        )
        for grade, prices in grouped.items()
    }


def _preference_fingerprint(state: TripSemanticState, domain: str) -> str:
    base: dict[str, object] = {
        "destination": state.trip_basics.destination_canonical_id,
        "dates": [
            (
                state.trip_basics.start_date.isoformat()
                if state.trip_basics.start_date is not None
                else None
            ),
            (
                state.trip_basics.end_date.isoformat()
                if state.trip_basics.end_date is not None
                else None
            ),
        ],
        "travelers": state.trip_basics.travelers,
        "pace": state.transport_and_pace.pace_preferences,
        "constraints": state.constraints,
    }
    if domain == "dining":
        base["dining_requirements"] = [
            *state.dining.requirements,
            *state.dining.allergies,
            *state.dining.avoidances,
        ]
    elif domain == "lodging_area":
        base["attraction_anchors"] = [
            item.model_dump(mode="json")
            for item in state.attractions.concrete_intents
            if item.disposition in {"must", "want"}
        ]
        base["dining_anchors"] = [
            item.model_dump(mode="json")
            for item in state.dining.concrete_restaurant_intents
            if item.disposition == "destination"
        ]
    elif domain == "lodging_class":
        base["area_preferences"] = [
            item.model_dump(mode="json") for item in state.lodging.area_preferences
        ]
    return canonical_json_hash(base)


def _control(
    interaction_id: UUID,
    domain: str,
    kind: str,
    label: str,
    *,
    signed: bool = True,
) -> CardControlAction:
    control_id = f"{domain}:{kind}"
    return CardControlAction(
        control_id=control_id,
        kind=kind,  # type: ignore[arg-type]
        label=label,
        signed_operation_ref=(
            str(uuid5(NAMESPACE_URL, f"v4-card-control:{interaction_id}:{control_id}"))
            if signed
            else None
        ),
    )


def _preference_controls(interaction_id: UUID, domain: str) -> list[CardControlAction]:
    return [
        _control(interaction_id, domain, "no_preference", "没有特别偏好"),
        _control(interaction_id, domain, "delegate", "交给 Agent 决定"),
        _control(interaction_id, domain, "refresh", "换一批", signed=False),
        _control(interaction_id, domain, "free_text", "我自己补充", signed=False),
    ]


def _specific_controls(interaction_id: UUID, domain: str) -> list[CardControlAction]:
    return [
        _control(interaction_id, domain, "no_preference", "没有具体要求"),
        _control(interaction_id, domain, "delegate", "按路线帮我选"),
        _control(interaction_id, domain, "refresh", "换一批", signed=False),
        _control(interaction_id, domain, "free_text", "我想去别的地方", signed=False),
    ]


def _lodging_controls(interaction_id: UUID, domain: str) -> list[CardControlAction]:
    return [
        _control(interaction_id, domain, "delegate", "交给 Planner 权衡"),
        _control(interaction_id, domain, "existing_booking", "我已经订好了"),
        _control(interaction_id, domain, "not_applicable", "这次不需要住宿"),
        _control(interaction_id, domain, "free_text", "我自己补充", signed=False),
    ]


__all__ = ["CardAttachment", "PrepareCardService"]
