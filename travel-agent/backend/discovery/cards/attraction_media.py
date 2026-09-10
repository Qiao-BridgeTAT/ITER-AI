"""Optional, bounded, same-entity photos. Missing media never removes a place."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass

from pydantic import HttpUrl

from backend.agent.model_gateway import ModelCancellation
from backend.contracts.enums import ProviderCode
from backend.contracts.v4.cards import CardOption
from backend.planning.city_registry import CityRegistry, CityRegistryError
from backend.providers.contracts import PlaceDetailRequest, ProductSearchRequest, ProviderError
from backend.providers.interfaces import PlaceProvider, TravelProductProvider


def same_photo_subject(left: str, right: str, city: str) -> bool:
    """Conservative whole-name match; substring matches can be internal exhibits."""

    def key(value: str) -> str:
        value = re.sub(r"[\s·•()（）\-—]", "", value).casefold()
        for prefix in (city, city.removesuffix("市")):
            if prefix and value.startswith(prefix.casefold()):
                value = value[len(prefix) :]
                break
        return value

    return bool(key(left)) and key(left) == key(right)


@dataclass(frozen=True)
class ResolvedPlaceMedia:
    image_url: HttpUrl
    source_ref: str


class AttractionMediaResolver:
    def __init__(
        self,
        *,
        registry: CityRegistry,
        places: PlaceProvider,
        products: TravelProductProvider | None = None,
    ) -> None:
        self._registry = registry
        self._places = places
        self._products = products

    async def enrich(
        self,
        options: list[CardOption],
        city_id: str,
        *,
        cancellation: ModelCancellation | None = None,
    ) -> list[CardOption]:
        semaphore = asyncio.Semaphore(3)

        async def resolve(option: CardOption) -> CardOption:
            if option.image_url is not None:
                return option
            async with semaphore:
                if cancellation:
                    cancellation.raise_if_cancelled("attraction_media")
                try:
                    async with asyncio.timeout(10):
                        return await self._resolve(option, city_id)
                except (ProviderError, CityRegistryError, TimeoutError):
                    return option

        resolved = list(await asyncio.gather(*(resolve(item) for item in options)))
        if cancellation:
            cancellation.raise_if_cancelled("attraction_media")
        return resolved

    async def _resolve(self, option: CardOption, city_id: str) -> CardOption:
        amap_ref = next(
            (ref for ref in option.source_refs if ref.startswith("provider:amap:")), None
        )
        media = await self.resolve_place(
            place_id=amap_ref.removeprefix("provider:amap:") if amap_ref else None,
            name=option.label,
            city_id=city_id,
        )
        if media is None:
            return option
        return option.model_copy(
            update={
                "image_url": media.image_url,
                "image_source_ref": media.source_ref,
                "source_refs": list(dict.fromkeys([*option.source_refs, media.source_ref])),
            }
        )

    async def resolve_place(
        self,
        *,
        place_id: str | None,
        name: str,
        city_id: str,
        detail_checked: bool = False,
    ) -> ResolvedPlaceMedia | None:
        """Share exact-entity image lookup with the published-plan preview path."""
        scope = self._registry.provider_scope(city_id, ProviderCode.AMAP)
        if place_id and not detail_checked:
            try:
                detail = await self._places.get_place(
                    PlaceDetailRequest(
                        city=scope,
                        source_place_id=place_id,
                    )
                )
                match = next(
                    (
                        item
                        for item in detail.items
                        if (
                            item.source_place_id == place_id
                            and item.city_id == city_id
                            and item.image_url
                        )
                    ),
                    None,
                )
                if match:
                    assert match.image_url is not None
                    return ResolvedPlaceMedia(match.image_url, f"provider:amap:{place_id}")
            except ProviderError:
                pass
        if self._products is None:
            return None
        products = await self._products.search_place_products(
            ProductSearchRequest(
                city=self._registry.provider_scope(city_id, ProviderCode.FLYAI),
                query=name,
            )
        )
        product_match = next(
            (
                item
                for item in products.items
                if (
                    item.image_url
                    and item.place_name
                    and same_photo_subject(name, item.place_name, scope.display_name or "")
                )
            ),
            None,
        )
        if not product_match:
            return None
        assert product_match.image_url is not None
        image_ref = f"provider:flyai:{product_match.source_place_id}"
        return ResolvedPlaceMedia(product_match.image_url, image_ref)
