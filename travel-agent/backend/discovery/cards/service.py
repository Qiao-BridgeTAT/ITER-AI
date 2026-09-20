"""Prepare Agent capability for materializing one signed V4 discovery card."""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Literal, TypeAlias, cast
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
from backend.contracts.v4.lodging_preferences import HotelQualityTier
from backend.contracts.v4.state import TripSemanticState
from backend.discovery.cards.candidate_composition import (
    CandidateCompositionService,
    CardGenerationError,
    specific_dependency_fingerprint,
)
from backend.discovery.cards.preference_generation import (
    AttractionDirectionDraft,
    DiningDirectionDraft,
    DiningPreferencePlan,
    DirectionDraft,
    LodgingAreaDirectionDraft,
    PreferenceDirectionGenerator,
)
from backend.discovery.prepared_evidence import PreparedEvidenceCollector
from backend.persistence.outbox_repository import canonical_json_hash
from backend.planning.city_registry import CityRegistry
from backend.providers.contracts import (
    HotelSearchRequest,
    ProviderCityScope,
    ProviderError,
    ProviderHotelOffer,
    ProviderPlace,
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
        evidence_collector: PreparedEvidenceCollector | None = None,
    ) -> None:
        self._directions = directions
        self._candidates = candidates
        self._registry = registry
        self._places = places
        self._products = products
        self._clock = clock
        self._lodging_observer = lodging_observer
        self._evidence_collector = evidence_collector

    def start_lodging_prefetch(self, state: TripSemanticState) -> None:
        from backend.discovery.lodging_search import hotel_requests, start_prefetch

        lodging, basics = state.lodging, state.trip_basics
        if (
            self._products is None
            or lodging.not_applicable
            or lodging.existing_bookings
            or not lodging.class_preference_source_operation_refs
            or not basics.destination_canonical_id
            or not basics.start_date
            or not basics.end_date
            or basics.end_date <= basics.start_date
        ):
            return
        requests = hotel_requests(
            city=self._registry.provider_scope(basics.destination_canonical_id, ProviderCode.FLYAI),
            check_in=basics.start_date,
            check_out=basics.end_date,
            examples=[
                example
                for area in lodging.area_preferences
                if area.selected
                for example in area.lodging_examples
            ],
            tiers=lodging.hotel_quality_tiers
            or ([lodging.hotel_quality_tier] if lodging.hotel_quality_tier else []),
            types=lodging.property_type_preferences,
        )
        start_prefetch(self._products, requests)

    def start_attraction_prefetch(self, trip_id: str, city_id: str) -> None:
        self._candidates.start_attraction_prefetch(trip_id, city_id)
        self._candidates.start_dining_prefetch(trip_id, city_id)

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
            if state.trip_basics.destination_canonical_id:
                self._candidates.start_dining_prefetch(
                    state.trip_id, state.trip_basics.destination_canonical_id, fixed=True
                )
            result = await self._candidates.compose(
                state,
                domain=CardDomain.ATTRACTION,
                interaction_id=interaction_id,
                attachment_id=attachment_id,
                cancellation=cancellation,
            )
            if self._evidence_collector:
                await self._evidence_collector.remember(
                    state, turn_id, result.card, result.selected_provider_candidates, cancellation
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
            if self._evidence_collector:
                await self._evidence_collector.remember(
                    state, turn_id, result.card, result.selected_provider_candidates, cancellation
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
        if state.trip_basics.destination_canonical_id is not None:
            self.start_attraction_prefetch(
                state.trip_id, state.trip_basics.destination_canonical_id
            )
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
                strategy_version="dining-v3"
                if isinstance(plan, DiningPreferencePlan)
                else "legacy",
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
        plan, mode = await self._directions.generate_lodging_area(state, cancellation=cancellation)
        options = []
        for key, label in (
            ("transit", "交通枢纽附近"),
            ("attraction", "核心景点附近"),
            ("commercial", "商圈附近"),
        ):
            copy = getattr(plan, key)
            option_id = str(uuid5(NAMESPACE_URL, f"v4-card-option:{interaction_id}:{key}"))
            options.append(
                CardOption(
                    option_id=option_id,
                    label=label,
                    description=copy.advantage + " " + copy.tradeoff,
                    lodging_area_copy=copy,
                    semantic_value=CardSemanticValue(
                        root=DirectionSemanticValue(
                            kind="direction",
                            direction_id=key,
                            direction_kind="area_strategy",
                            lodging_examples=copy.examples,
                        )
                    ),
                    signed_operation_ref=str(
                        uuid5(NAMESPACE_URL, f"v4-card-operation:{interaction_id}:{option_id}")
                    ),
                )
            )
        return LodgingAreaPreferenceCard(
            attachment_id=str(attachment_id),
            interaction_id=str(interaction_id),
            based_on_state_version=state.state_version,
            dependency_fingerprint=_preference_fingerprint(state, "lodging_area"),
            status=CardStatus.ACTIVE,
            options=options,
            control_actions=_lodging_controls(interaction_id, "lodging_area"),
            generation_metadata=CardGenerationMetadata(
                generated_at=self._now(),
                generation_mode=mode,
                strategy_version="lodging-v2",
            ),
            prompt="你更倾向住在哪类区域？可以多选。",
        )

    async def _lodging_class(
        self,
        state: TripSemanticState,
        interaction_id: UUID,
        attachment_id: UUID,
        cancellation: ModelCancellation | None,
    ) -> LodgingClassPreferenceCard:
        if state.lodging.not_applicable or state.lodging.existing_bookings:
            raise CardGenerationError("lodging class card is bypassed by current lodging state")
        options = []
        for key, label, group, tier, property_type in (
            ("economy", "二星/经济", "quality", "economy", None),
            ("comfort", "三星/舒适", "quality", "comfort", None),
            ("upscale", "四星/高档", "quality", "upscale", None),
            ("luxury", "五星/豪华", "quality", "luxury", None),
            ("hotel", "酒店", "property_type", None, "酒店"),
            ("homestay", "民宿", "property_type", None, "民宿"),
        ):
            option_id = str(uuid5(NAMESPACE_URL, f"v4-card-option:{interaction_id}:{key}"))
            options.append(
                CardOption(
                    option_id=option_id,
                    label=label,
                    description="可选择符合心意的住宿档次与类型。",
                    selection_group=cast(Literal["quality", "property_type"], group),
                    semantic_value=CardSemanticValue(
                        root=DirectionSemanticValue(
                            kind="direction",
                            direction_id=key,
                            direction_kind="hotel_class",
                            hotel_quality_tier=cast(HotelQualityTier | None, tier),
                            property_type=property_type,
                        )
                    ),
                    signed_operation_ref=str(
                        uuid5(NAMESPACE_URL, f"v4-card-operation:{interaction_id}:{option_id}")
                    ),
                )
            )
        return LodgingClassPreferenceCard(
            attachment_id=str(attachment_id),
            interaction_id=str(interaction_id),
            based_on_state_version=state.state_version,
            dependency_fingerprint=_preference_fingerprint(state, "lodging_class"),
            status=CardStatus.ACTIVE,
            options=options,
            control_actions=_lodging_controls(interaction_id, "lodging_class"),
            generation_metadata=CardGenerationMetadata(
                generated_at=self._now(),
                generation_mode="static",
                strategy_version="lodging-v2",
            ),
            prompt="你对住宿品质有什么要求？住宿档次和住宿类型均可多选。",
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
    item: DirectionDraft | AttractionDirectionDraft,
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
                dining_search_hints=item.dining_search_hints
                if isinstance(item, DiningDirectionDraft)
                else None,
                tags=item.tags,
                search_query=item.search_query,
                attraction_search_hints=(
                    item.attraction_search_hints
                    if isinstance(item, AttractionDirectionDraft)
                    else None
                ),
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
