"""Bounded live media/weather view; no model calls, plan writes or pricing changes."""

from __future__ import annotations

import asyncio
import re
from collections import OrderedDict
from time import monotonic
from uuid import NAMESPACE_URL, UUID, uuid5

from backend.application.published_plan_view import visible_published_plan
from backend.contracts.enums import ProviderCode
from backend.contracts.v4.cards import SpecificCandidateCard
from backend.contracts.v4.conversation import ConversationSnapshotV4
from backend.contracts.v4.plan_preview import PlannerPlacePreview, PlannerPlanPreview
from backend.contracts.v4.planner_evidence import PlannerWeatherEvidence
from backend.contracts.v4.planner_publication import PlannerPublishedPlan
from backend.discovery.cards.attraction_media import AttractionMediaResolver
from backend.planning.city_registry import CityRegistryError, default_city_registry
from backend.providers.contracts import (
    PlaceDetailRequest,
    ProviderError,
    WeatherRequest,
)
from backend.providers.interfaces import PlaceProvider, TravelProductProvider, WeatherProvider
from backend.providers.place_copy import provider_dining_facts, provider_place_intro


class PlanPreviewService:
    def __init__(
        self,
        places: PlaceProvider,
        products: TravelProductProvider | None = None,
        weather: WeatherProvider | None = None,
    ) -> None:
        self._places = places
        self._weather = weather
        self._registry = default_city_registry()
        self._media = AttractionMediaResolver(
            registry=self._registry, places=places, products=products
        )
        self._cache: OrderedDict[UUID, tuple[float, PlannerPlanPreview]] = OrderedDict()
        self._locks = tuple(asyncio.Lock() for _ in range(64))
        self._semaphore = asyncio.Semaphore(3)

    async def get_preview(self, snapshot: ConversationSnapshotV4) -> PlannerPlanPreview:
        """Call only AFTER the snapshot's owner has been authenticated by the journey service."""
        plan = visible_published_plan(snapshot)
        if plan is None:
            raise ValueError("plan preview requires a published plan")
        # Also bounds concurrent cache misses and prevents duplicate detail requests
        # when multiple browser tabs open the same plan simultaneously.
        async with self._locks[plan.plan_version_id.int % len(self._locks)]:
            cached = self._cache.get(plan.plan_version_id)
            if cached is not None and cached[0] > monotonic():
                self._cache.move_to_end(plan.plan_version_id)
                return cached[1]
            seeds: dict[str, PlannerPlacePreview] = {}
            for message in snapshot.messages:
                for attachment in message.attachments:
                    card = attachment.root
                    if not isinstance(card, SpecificCandidateCard):
                        continue
                    for option in card.options:
                        if option.entity_ref is None:
                            continue
                        place_id = option.entity_ref.canonical_entity_id
                        try:
                            canonical_id = UUID(place_id)
                        except ValueError:
                            continue
                        previous = seeds.get(place_id)
                        # Older card descriptions contain ranking machinery, not
                        # a visitor-facing introduction. Only the v2 experience is copy.
                        description = (
                            _short_intro(option.description)
                            if card.generation_metadata.strategy_version
                            in {"attraction-v2", "dining-v2"}
                            else None
                        )
                        seeds[place_id] = PlannerPlacePreview(
                            place_id=canonical_id,
                            image_url=option.image_url
                            or (previous.image_url if previous else None),
                            image_source_ref=(
                                option.image_source_ref
                                if option.image_url
                                else previous.image_source_ref
                                if previous
                                else None
                            ),
                            description=description or (previous.description if previous else None),
                            dining_details=option.dining_details
                            or (previous.dining_details if previous else None),
                        )

            targets = {
                str(activity.place_id): activity.kind
                for day in plan.materialized_schedule.days
                for activity in day.activities
            }
            evidence = {
                item.canonical_entity_id: item
                for item in plan.place_evidence
                if item.canonical_entity_id in targets
            }
            # The same physical hotel is represented by a plan boundary UUID.
            baseline = plan.working_itinerary.lodging_baseline
            property_id = (
                baseline.selected_offer_ref.property_id
                if baseline.selected_offer_ref
                else baseline.fixed_commitment_ref.commitment_id
                if baseline.fixed_commitment_ref
                else None
            )
            prefix = "hotel-property" if baseline.selected_offer_ref else "fixed-hotel"
            hotels = {
                str(uuid5(NAMESPACE_URL, f"iter:v4-planner:{prefix}:{hotel.property_id}")): hotel
                for hotel in plan.hotel_location_evidence
                if hotel.property_id == property_id
            }

            async def resolve(place_id: str) -> PlannerPlacePreview:
                seed = seeds.get(place_id, PlannerPlacePreview(place_id=UUID(place_id)))
                if (
                    seed.image_url
                    and seed.description
                    and (targets.get(place_id) != "restaurant" or seed.dining_details is not None)
                ):
                    return seed
                item = evidence.get(place_id)
                hotel = hotels.get(place_id)
                if item is None and hotel is None:
                    return seed
                source_id = item.provider_entity_id if item else hotel.provider_entity_id  # type: ignore[union-attr]
                image, source, description = seed.image_url, seed.image_source_ref, seed.description
                dining_details = seed.dining_details
                try:
                    # Includes queue time: many missing photos cannot delay this
                    # optional endpoint by ten seconds per batch of three.
                    async with asyncio.timeout(12):
                        async with self._semaphore:
                            detail = await self._places.get_place(
                                PlaceDetailRequest(
                                    city=self._registry.provider_scope(
                                        plan.materialized_schedule.city_id, ProviderCode.AMAP
                                    ),
                                    source_place_id=source_id,
                                )
                            )
                            match = next(
                                (
                                    place
                                    for place in detail.items
                                    if (
                                        place.source_place_id == source_id
                                        and place.city_id == plan.materialized_schedule.city_id
                                    )
                                ),
                                None,
                            )
                            if match:
                                if not image and match.image_url:
                                    image, source = match.image_url, f"provider:amap:{source_id}"
                                description = description or provider_place_intro(match)
                                dining_details = provider_dining_facts(match) or dining_details
                            if image is None and targets.get(place_id) == "attraction" and item:
                                media = await self._media.resolve_place(
                                    place_id=source_id,
                                    name=item.display_name,
                                    city_id=item.city_id,
                                    detail_checked=True,
                                )
                                if media:
                                    image, source = media.image_url, media.source_ref
                except (ProviderError, CityRegistryError, TimeoutError):
                    pass
                return PlannerPlacePreview(
                    place_id=UUID(place_id),
                    image_url=image,
                    image_source_ref=source,
                    description=description,
                    dining_details=dining_details,
                )

            place_previews, weather_evidence = await asyncio.gather(
                asyncio.gather(
                    *(resolve(place_id) for place_id in dict.fromkeys([*targets, *hotels]))
                ),
                self._get_weather(plan),
            )
            result = PlannerPlanPreview(
                trip_id=plan.trip_id,
                plan_version_id=plan.plan_version_id,
                places=tuple(place_previews),
                weather_evidence=weather_evidence,
            )
            self._cache[plan.plan_version_id] = (monotonic() + 1800, result)
            while len(self._cache) > 64:
                self._cache.popitem(last=False)
            return result

    async def _get_weather(self, plan: PlannerPublishedPlan) -> tuple[PlannerWeatherEvidence, ...]:
        if self._weather is None:
            return ()
        dates = {day.service_date for day in plan.materialized_schedule.days}
        if not dates:
            return ()
        try:
            async with asyncio.timeout(13):
                result = await self._weather.get_forecast(
                    WeatherRequest(
                        city=self._registry.provider_scope(
                            plan.materialized_schedule.city_id, ProviderCode.WEATHER
                        ),
                        start_date=min(dates),
                        end_date=max(dates),
                    )
                )
        except (ProviderError, CityRegistryError, TimeoutError):
            return ()
        return tuple(
            PlannerWeatherEvidence(
                service_date=item.forecast_date,
                condition_day=item.condition_day,
                condition_night=item.condition_night,
                high_celsius=item.high_celsius,
                low_celsius=item.low_celsius,
                source_name=item.source_name,
                forecast_kind=item.forecast_kind,
                observed_at=item.fetched_at,
                fact_reference_id=str(
                    uuid5(
                        NAMESPACE_URL,
                        f"weather:{plan.materialized_schedule.city_id}:{item.source_name}:"
                        f"{item.forecast_date}:{item.fetched_at.isoformat()}",
                    )
                ),
            )
            for item in result.items
            if item.forecast_date in dates
        )


def _short_intro(value: str | None) -> str | None:
    if not value or re.search(r"建议范围|调整为|用户尚未|匹配本次|候选|补位|source_ref", value):
        return None
    if re.search(r"^走进.+感受这里的风景与人文|^以.+风味为主，可以体验这一菜系", value):
        return None
    first = re.split(r"[。！？\n]", value.strip())[0].strip()
    if len(first) > 64:
        first = re.split(r"[，；]", first)[0][:64]
    return f"{first}。" if first else None
