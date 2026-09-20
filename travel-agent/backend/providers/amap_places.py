"""AMap Web Service place search (v3/v5) and geocoding adapter."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, cast
from uuid import UUID

import httpx
from pydantic import HttpUrl, TypeAdapter

from backend.config.settings import AmapPolygonProxySettings, AmapSearchProxySettings
from backend.contracts.common import CnyAmountRange
from backend.contracts.enums import CityCode, CoordinateSystem, PlaceCategory, ProviderCode
from backend.contracts.places import CanonicalPlace, Gcj02Coordinates, PlaceSourceMapping
from backend.providers.amap_http import AMAP_BASE_URL, amap_malformed, request_amap_json
from backend.providers.contracts import (
    GeocodedAddress,
    GeocodeRequest,
    KeywordPlaceSearchRequest,
    NearbyPlaceSearchRequest,
    PlaceDetailRequest,
    PolygonPlaceSearchRequest,
    ProviderError,
    ProviderPlace,
    ProviderResponse,
    ProviderResultStatus,
)
from backend.providers.place_taxonomy import category_from_original_typecodes

AMAP_TEXT_PATH = "/v3/place/text"
AMAP_AROUND_PATH = "/v3/place/around"
AMAP_POLYGON_PATH = "/v5/place/polygon"
AMAP_DETAIL_PATH = "/v3/place/detail"
AMAP_GEOCODE_PATH = "/v3/geocode/geo"


class AmapPlaceProvider:
    """Translate AMap responses into vendor-neutral provider contracts."""

    def __init__(
        self,
        api_key: str,
        *,
        client: httpx.AsyncClient | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        timeout_seconds: float = 5.0,
        search_proxy: AmapSearchProxySettings | None = None,
        polygon_proxy: AmapPolygonProxySettings | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("AMAP_WEB_SERVICE_KEY must not be empty")
        self._api_key = api_key
        self._search_proxy = search_proxy
        self._polygon_proxy = polygon_proxy
        self._client = client or httpx.AsyncClient(
            base_url=AMAP_BASE_URL,
            timeout=timeout_seconds,
        )
        self._owns_client = client is None
        self._clock = clock

    async def search_places(
        self, request: KeywordPlaceSearchRequest
    ) -> ProviderResponse[ProviderPlace]:
        parameters = {
            "keywords": request.query,
            "city": request.city.provider_city_code,
            "citylimit": "true",
            "offset": str(request.page_size),
            "page": str(request.page),
            "extensions": "all",
            # AMap otherwise flattens entrances/exhibits beside their parent POI.
            # Keep restaurant searches flat: shops inside a mall are independent.
            **({"children": "1"} if request.category_hint is PlaceCategory.ATTRACTION else {}),
            **({"types": "|".join(request.typecodes)} if request.typecodes else {}),
        }
        payload = await request_amap_json(
            self._client,
            self._api_key,
            AMAP_TEXT_PATH,
            parameters,
            "place_search",
            search_proxy=self._search_proxy,
        )
        return self._parse_places(payload, request.city.city_id, request.category_hint)

    async def search_nearby(
        self, request: NearbyPlaceSearchRequest
    ) -> ProviderResponse[ProviderPlace]:
        parameters = {
            "location": _format_location(request.center),
            "city": request.city.provider_city_code,
            "citylimit": "true",
            "radius": str(request.radius_m),
            "sortrule": "distance",
            "offset": str(request.page_size),
            "page": str(request.page),
            "extensions": "all",
            **({"keywords": request.query} if request.query is not None else {}),
            **({"types": "|".join(request.typecodes)} if request.typecodes else {}),
        }
        payload = await request_amap_json(
            self._client,
            self._api_key,
            AMAP_AROUND_PATH,
            parameters,
            "place_nearby",
            search_proxy=self._search_proxy,
        )
        return self._parse_places(payload, request.city.city_id, request.category_hint)

    async def search_polygon(
        self, request: PolygonPlaceSearchRequest
    ) -> ProviderResponse[ProviderPlace]:
        payload = await request_amap_json(
            self._client,
            self._api_key,
            AMAP_POLYGON_PATH,
            {
                "polygon": "|".join(_format_location(point) for point in request.polygon),
                "types": "|".join(request.typecodes),
                **({"keywords": request.query} if request.query else {}),
                "page_num": str(request.page),
                "page_size": str(request.page_size),
                "show_fields": "business,navi,photos",
            },
            "place_polygon",
            search_proxy=self._search_proxy,
            polygon_proxy=self._polygon_proxy,
        )
        # This endpoint has no city or sortrule parameter. Its native order is
        # the search service's composite ranking, not a rating sort.
        return self._parse_places(payload, request.city.city_id, request.category_hint)

    async def get_place(self, request: PlaceDetailRequest) -> ProviderResponse[ProviderPlace]:
        payload = await request_amap_json(
            self._client,
            self._api_key,
            AMAP_DETAIL_PATH,
            {"id": request.source_place_id, "extensions": "all"},
            "place_detail",
            search_proxy=self._search_proxy,
        )
        return self._parse_places(payload, request.city.city_id, request.category_hint)

    async def geocode(self, request: GeocodeRequest) -> ProviderResponse[GeocodedAddress]:
        payload = await request_amap_json(
            self._client,
            self._api_key,
            AMAP_GEOCODE_PATH,
            {"address": request.address, "city": request.city.provider_city_code},
            "geocode",
        )
        fetched_at = self._fetched_at()
        records = payload.get("geocodes")
        if not isinstance(records, list):
            raise _malformed("geocode")
        if not records:
            return ProviderResponse[GeocodedAddress](
                provider=ProviderCode.AMAP,
                status=ProviderResultStatus.EMPTY,
                fetched_at=fetched_at,
            )

        items: list[GeocodedAddress] = []
        missing_fields: set[str] = set()
        for raw_record in records:
            if not isinstance(raw_record, dict):
                raise _malformed("geocode")
            record = cast(dict[str, Any], raw_record)
            formatted_address = _required_text(record.get("formatted_address"), "geocode")
            coordinates = _parse_location(record.get("location"), "geocode")
            provider_city_code = _optional_text(record.get("adcode"), "geocode")
            match_level = _optional_text(record.get("level"), "geocode")
            if provider_city_code is None:
                missing_fields.add("items.provider_city_code")
            if match_level is None:
                missing_fields.add("items.match_level")
            items.append(
                GeocodedAddress(
                    provider=ProviderCode.AMAP,
                    city_id=request.city.city_id,
                    provider_city_code=provider_city_code,
                    formatted_address=formatted_address,
                    coordinates=coordinates,
                    match_level=match_level,
                    fetched_at=fetched_at,
                    raw_payload=record,
                )
            )
        return _response(items, fetched_at, missing_fields)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def _parse_places(
        self,
        payload: dict[str, Any],
        city_id: str,
        category_hint: PlaceCategory | None,
    ) -> ProviderResponse[ProviderPlace]:
        fetched_at = self._fetched_at()
        records = payload.get("pois")
        if not isinstance(records, list):
            raise _malformed("place_parse")
        if not records:
            return ProviderResponse[ProviderPlace](
                provider=ProviderCode.AMAP,
                status=ProviderResultStatus.EMPTY,
                fetched_at=fetched_at,
            )

        items: list[ProviderPlace] = []
        missing_fields: set[str] = set()
        for raw_record in records:
            if not isinstance(raw_record, dict):
                raise _malformed("place_parse")
            record = cast(dict[str, Any], raw_record)
            source_place_id = _required_text(record.get("id"), "place_parse")
            name = _required_text(record.get("name"), "place_parse")
            coordinates = _parse_location(record.get("location"), "place_parse")
            address = _optional_text(record.get("address"), "place_parse")
            provider_city_code = _optional_text(record.get("adcode"), "place_parse")
            typecode = _optional_text(record.get("typecode"), "place_parse")
            parent_place_id = _optional_text(record.get("parent"), "place_parse")
            image_url = _first_photo_url(record.get("photos"))
            business = record.get("business")
            if not isinstance(business, dict):
                business = record.get("biz_ext")
            business = business if isinstance(business, dict) else {}
            navigation = record.get("navi")
            navigation = navigation if isinstance(navigation, dict) else {}
            rating = _business_number(business.get("rating"))
            cost = _business_number(business.get("cost"))
            if address is None:
                missing_fields.add("items.address")
            if provider_city_code is None:
                missing_fields.add("items.provider_city_code")
            if typecode is None:
                missing_fields.add("items.provider_typecode")
            items.append(
                ProviderPlace(
                    provider=ProviderCode.AMAP,
                    source_place_id=source_place_id,
                    city_id=city_id,
                    provider_city_code=provider_city_code,
                    name=name,
                    category=_resolved_category(typecode, category_hint),
                    address=address,
                    coordinates=coordinates,
                    entrance_coordinates=_optional_location(navigation.get("entr_location")),
                    exit_coordinates=_optional_location(navigation.get("exit_location")),
                    provider_typecode=typecode,
                    provider_parent_place_id=parent_place_id,
                    image_url=image_url,
                    rating=float(rating) if rating is not None and 0 < rating <= 5 else None,
                    average_cost=(
                        CnyAmountRange(
                            currency="CNY",
                            minimum_fen=int(cost * 100),
                            maximum_fen=int(cost * 100),
                        )
                        if cost is not None and cost > 0
                        else None
                    ),
                    fetched_at=fetched_at,
                    raw_payload=record,
                )
            )
        return _response(items, fetched_at, missing_fields)

    def _fetched_at(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("provider clock must return an aware datetime")
        return value.astimezone(UTC)


def _business_number(value: object) -> Decimal | None:
    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        return None
    try:
        number = Decimal(str(value))
    except InvalidOperation:
        return None
    return number if number.is_finite() and number >= 0 else None


def amap_place_to_canonical(
    place: ProviderPlace,
    *,
    place_id: UUID,
    city: CityCode,
) -> CanonicalPlace:
    """Create a stage-0 canonical place without inventing facts absent from AMap."""

    if place.provider is not ProviderCode.AMAP:
        raise ValueError("only AMap places can use the AMap canonicalization helper")
    return CanonicalPlace(
        place_id=place_id,
        city=city,
        category=place.category,
        name=place.name,
        address=place.address,
        coordinates=place.coordinates,
        source_mappings=[
            PlaceSourceMapping(
                provider=ProviderCode.AMAP,
                source_place_id=place.source_place_id,
                raw_name=place.name,
                raw_address=place.address,
                raw_coordinates=place.coordinates,
                fetched_at=place.fetched_at,
            )
        ],
    )


def _response(
    items: list[Any],
    fetched_at: datetime,
    missing_fields: set[str],
) -> ProviderResponse[Any]:
    return ProviderResponse[Any](
        provider=ProviderCode.AMAP,
        status=(ProviderResultStatus.PARTIAL if missing_fields else ProviderResultStatus.SUCCESS),
        items=items,
        fetched_at=fetched_at,
        missing_fields=sorted(missing_fields),
    )


def _malformed(operation: str) -> ProviderError:
    return amap_malformed(operation)


def _required_text(value: Any, operation: str) -> str:
    parsed = _optional_text(value, operation)
    if parsed is None:
        raise _malformed(operation)
    return parsed


def _first_photo_url(value: Any) -> HttpUrl | None:
    """Return one normalized public photo URL without exposing AMap's raw shape."""

    if not isinstance(value, list):
        return None
    for item in value:
        if not isinstance(item, dict):
            continue
        url = _optional_text(item.get("url"), "place_parse")
        if url is not None and url.startswith(("https://", "http://")):
            return TypeAdapter(HttpUrl).validate_python(url)
    return None


def _optional_text(value: Any, operation: str) -> str | None:
    if value is None or value == "" or value == []:
        return None
    if not isinstance(value, str):
        raise _malformed(operation)
    parsed = value.strip()
    return parsed or None


def _parse_location(value: Any, operation: str) -> Gcj02Coordinates:
    text = _required_text(value, operation)
    parts = text.split(",")
    if len(parts) != 2:
        raise _malformed(operation)
    try:
        longitude, latitude = (float(part) for part in parts)
        return Gcj02Coordinates(
            latitude=latitude,
            longitude=longitude,
            coord_system=CoordinateSystem.GCJ_02,
        )
    except (TypeError, ValueError):
        raise _malformed(operation) from None


def _format_location(coordinates: Gcj02Coordinates) -> str:
    return f"{coordinates.longitude:.6f},{coordinates.latitude:.6f}"


def _optional_location(value: Any) -> Gcj02Coordinates | None:
    if not value:
        return None
    try:
        return _parse_location(value, "place_parse")
    except ProviderError:
        # Optional navigation must not replace or invalidate the primary POI.
        return None


def _category_from_typecode(typecode: str | None) -> PlaceCategory:
    return category_from_original_typecodes(typecode)


def _resolved_category(
    typecode: str | None,
    category_hint: PlaceCategory | None,
) -> PlaceCategory:
    """Project only the Provider's original type; a query hint is not evidence."""

    del category_hint
    return _category_from_typecode(typecode)
