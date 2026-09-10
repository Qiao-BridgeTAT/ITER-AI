"""Deterministic cross-provider place identity evidence and coordinate normalization."""

from __future__ import annotations

import math
import re
from difflib import SequenceMatcher

from backend.contracts.enums import CoordinateSystem, PlaceCategory
from backend.contracts.places import (
    CanonicalPlace,
    Coordinates,
    Gcj02Coordinates,
    PlaceIdentityEvidence,
    PlaceMatchProposal,
    PlaceSourceMapping,
)

DEFAULT_PLACE_MATCH_DISTANCE_METERS = 300
MAX_ADDRESS_COORDINATE_CONTRADICTION_METERS = 2_000


def matches_place_search_identity(
    query: str,
    name: str,
    *,
    city_name: str,
    category: PlaceCategory,
) -> bool:
    """Conservatively bind a named search hypothesis to a real returned entity.

    A cuisine/category search is not identity evidence. Only harmless current
    city prefixes, branch parentheses and institution suffixes are normalized.
    """

    def identity(value: str) -> str:
        value = re.sub(r"[（(][^）)]*[）)]", "", value)
        if category is PlaceCategory.RESTAURANT:
            value = re.split(r"[·•・]", value, maxsplit=1)[0]
        normalized = _normalize_text(value)
        city = _normalize_text(city_name).removesuffix("市")
        if city and normalized.startswith(city):
            normalized = normalized[len(city) :].removeprefix("市")
        suffix = (
            r"(?:旗舰店|总店|分店|菜馆|酒楼|餐厅|餐馆|饭店|店)$"
            if category is PlaceCategory.RESTAURANT
            else r"(?:风景名胜区|风景区|景区|公园|遗址|博物院|博物馆)$"
        )
        shortened = re.sub(suffix, "", normalized)
        return shortened if len(shortened) >= 2 else normalized

    expected = identity(query)
    names = [name]
    if category is PlaceCategory.ATTRACTION:
        names.extend(re.findall(r"[（(]([^）)]+)[）)]", name))
    actual_names = [identity(value) for value in names]
    # Generic queries are not identity evidence, but a city-qualified official
    # name (e.g. a municipal museum) must not become generic merely because the
    # same city prefix is normalized from both sides of an exact comparison.
    if _normalize_text(query) in {
        "经典景点",
        "景点",
        "公园",
        "博物馆",
        "本地美食",
        "当地美食",
        "特色餐厅",
        "盐水鸭",
        "烤鸭",
        "小吃",
        "老字号",
    }:
        return False
    return len(expected) >= 2 and any(
        actual == expected or (len(expected) >= 3 and actual.endswith(expected))
        for actual in actual_names
    )


def propose_place_match(
    existing: CanonicalPlace,
    *,
    existing_city_id: str,
    incoming_city_id: str,
    incoming: PlaceSourceMapping,
    incoming_category: PlaceCategory | None = None,
    maximum_distance_meters: int = DEFAULT_PLACE_MATCH_DISTANCE_METERS,
) -> PlaceMatchProposal | None:
    """Propose, but never silently apply, a source-to-canonical place match."""

    if existing_city_id != incoming_city_id:
        return None
    if (
        incoming_category is not None
        and incoming_category is not PlaceCategory.OTHER
        and existing.category is not PlaceCategory.OTHER
        and incoming_category is not existing.category
    ):
        return None
    name_similarity = _name_similarity(existing.name, incoming.raw_name)
    if name_similarity < 0.72:
        return None
    address_match = _addresses_match(existing.address, incoming.raw_address)
    coordinate_distance: int | None = None
    coordinate_match = False
    if incoming.raw_coordinates is not None:
        normalized = coordinates_to_gcj02(incoming.raw_coordinates)
        coordinate_distance = round(_distance_meters(existing.coordinates, normalized))
        coordinate_match = coordinate_distance <= maximum_distance_meters
        if address_match and coordinate_distance > MAX_ADDRESS_COORDINATE_CONTRADICTION_METERS:
            return None
    if not address_match and not coordinate_match:
        return None
    return PlaceMatchProposal(
        existing_place_id=existing.place_id,
        incoming=incoming,
        evidence=PlaceIdentityEvidence(
            city_match=True,
            name_similarity=name_similarity,
            normalized_address_match=address_match,
            coordinate_proximity_match=coordinate_match,
            coordinate_distance_m=(coordinate_distance if coordinate_match else None),
        ),
    )


def coordinates_to_gcj02(coordinates: Coordinates) -> Gcj02Coordinates:
    """Normalize known source coordinate systems while callers retain the raw value."""

    if coordinates.coord_system is CoordinateSystem.GCJ_02:
        return Gcj02Coordinates(
            latitude=coordinates.latitude,
            longitude=coordinates.longitude,
            coord_system=CoordinateSystem.GCJ_02,
        )
    if coordinates.coord_system is CoordinateSystem.BD_09:
        x = coordinates.longitude - 0.0065
        y = coordinates.latitude - 0.006
        z = math.sqrt(x * x + y * y) - 0.00002 * math.sin(y * math.pi)
        theta = math.atan2(y, x) - 0.000003 * math.cos(x * math.pi)
        return Gcj02Coordinates(
            latitude=z * math.sin(theta),
            longitude=z * math.cos(theta),
            coord_system=CoordinateSystem.GCJ_02,
        )
    longitude, latitude = _wgs84_to_gcj02(
        coordinates.longitude,
        coordinates.latitude,
    )
    return Gcj02Coordinates(
        latitude=latitude,
        longitude=longitude,
        coord_system=CoordinateSystem.GCJ_02,
    )


def coordinates_to_wgs84(coordinates: Gcj02Coordinates) -> Coordinates:
    """Invert the existing GCJ-02 transform at the external weather boundary."""
    longitude, latitude = coordinates.longitude, coordinates.latitude
    for _ in range(4):
        projected_lon, projected_lat = _wgs84_to_gcj02(longitude, latitude)
        longitude -= projected_lon - coordinates.longitude
        latitude -= projected_lat - coordinates.latitude
    return Coordinates(longitude=longitude, latitude=latitude, coord_system=CoordinateSystem.WGS_84)


def _name_similarity(left: str, right: str) -> float:
    normalized_left = _normalize_text(left)
    normalized_right = _normalize_text(right)
    if not normalized_left or not normalized_right:
        return 0.0
    if normalized_left == normalized_right:
        return 1.0
    if normalized_left in normalized_right or normalized_right in normalized_left:
        return 0.9
    return round(SequenceMatcher(None, normalized_left, normalized_right).ratio(), 6)


def _addresses_match(left: str | None, right: str | None) -> bool:
    if left is None or right is None:
        return False
    normalized_left = _normalize_text(left)
    normalized_right = _normalize_text(right)
    if min(len(normalized_left), len(normalized_right)) < 5:
        return False
    return normalized_left in normalized_right or normalized_right in normalized_left


def _normalize_text(value: str) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]", "", value.casefold())


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


def _wgs84_to_gcj02(longitude: float, latitude: float) -> tuple[float, float]:
    if _outside_china(longitude, latitude):
        return longitude, latitude
    semi_major_axis = 6_378_245.0
    eccentricity_squared = 0.006693421622965943
    latitude_delta = _transform_latitude(longitude - 105.0, latitude - 35.0)
    longitude_delta = _transform_longitude(longitude - 105.0, latitude - 35.0)
    radian_latitude = latitude / 180.0 * math.pi
    magic = math.sin(radian_latitude)
    magic = 1 - eccentricity_squared * magic * magic
    sqrt_magic = math.sqrt(magic)
    latitude_delta = (
        latitude_delta
        * 180.0
        / ((semi_major_axis * (1 - eccentricity_squared)) / (magic * sqrt_magic) * math.pi)
    )
    longitude_delta = (
        longitude_delta
        * 180.0
        / (semi_major_axis / sqrt_magic * math.cos(radian_latitude) * math.pi)
    )
    return longitude + longitude_delta, latitude + latitude_delta


def _outside_china(longitude: float, latitude: float) -> bool:
    return not (72.004 <= longitude <= 137.8347 and 0.8293 <= latitude <= 55.8271)


def _transform_latitude(x: float, y: float) -> float:
    value = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y
    value += 0.2 * math.sqrt(abs(x))
    value += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2 / 3
    value += (20.0 * math.sin(y * math.pi) + 40.0 * math.sin(y / 3 * math.pi)) * 2 / 3
    value += (160.0 * math.sin(y / 12 * math.pi) + 320 * math.sin(y * math.pi / 30)) * 2 / 3
    return value


def _transform_longitude(x: float, y: float) -> float:
    value = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y
    value += 0.1 * math.sqrt(abs(x))
    value += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2 / 3
    value += (20.0 * math.sin(x * math.pi) + 40.0 * math.sin(x / 3 * math.pi)) * 2 / 3
    value += (150.0 * math.sin(x / 12 * math.pi) + 300.0 * math.sin(x / 30 * math.pi)) * 2 / 3
    return value
