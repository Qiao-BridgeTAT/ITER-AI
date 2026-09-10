"""Short public introductions from existing Provider facts, never ranking rationale."""

from __future__ import annotations

import re

from backend.contracts.v4.dining_display import DiningDisplayFacts
from backend.providers.contracts import ProviderPlace


def provider_place_cuisine(place: ProviderPlace) -> str | None:
    """Only the same POI's provider taxonomy, never a search keyword or place name."""
    if place.category != "restaurant":
        return None
    label = _type_label(place)
    if label is None:
        return None
    if label in place.name and not re.search(r"菜|餐|小吃|咖啡|糕饼|火锅|甜品|茶艺", label):
        # Some vendor taxonomies end in a chain name, not a cuisine. Use the
        # actual parent category instead of repeating the restaurant title.
        label = _type_label(place, skip_label=label)
        if label is None:
            return None
    return label.replace("菜品餐厅", "菜").replace("中餐厅", "中式餐饮")


def provider_dining_facts(place: ProviderPlace) -> DiningDisplayFacts | None:
    if place.category != "restaurant":
        return None
    cuisine = provider_place_cuisine(place)
    rating = place.rating if place.rating and place.rating > 0 else None
    if not (cuisine or rating or place.average_cost):
        return None
    return DiningDisplayFacts(
        cuisine=cuisine,
        rating=rating,
        average_cost=place.average_cost,
        source_ref=f"provider:{place.provider.value}:{place.source_place_id}",
        source_name="高德" if place.provider.value == "amap" else place.provider.value,
        observed_at=place.fetched_at,
    )


def provider_place_intro(place: ProviderPlace) -> str | None:
    """Taxonomy is a fact, not prose. Natural copy has its own optional view."""
    return None


def _type_label(place: ProviderPlace, *, skip_label: str | None = None) -> str | None:
    value = place.raw_payload.get("type")
    if not isinstance(value, str):
        return None
    labels = [part.strip() for part in re.split(r"[;|]", value) if part.strip()]
    generic = {"餐饮服务", "风景名胜", "住宿服务", "餐饮相关场所", "风景名胜相关", "住宿服务相关"}
    label = next(
        (part for part in reversed(labels) if part not in generic and part != skip_label), None
    )
    if label is None or len(label) > 24 or not re.fullmatch(r"[\w\u4e00-\u9fff（）()· /-]+", label):
        return None
    return label
