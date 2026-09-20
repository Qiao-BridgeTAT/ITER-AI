"""Baidu Place v3 adapter for regular opening hours only."""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, cast
from uuid import UUID

import httpx

from backend.contracts.enums import CoordinateSystem, DataAvailability, ProviderCode
from backend.contracts.places import Gcj02Coordinates, RegularHours
from backend.providers.baidu_http import BAIDU_BASE_URL, baidu_malformed, request_baidu_json
from backend.providers.contracts import (
    HoursRequest,
    ProviderRegularHours,
    ProviderResponse,
    ProviderResultStatus,
)

BAIDU_AROUND_PATH = "/place/v3/around"
BAIDU_DETAIL_PATH = "/place/v3/detail"
BAIDU_MATCH_RADIUS_METERS = 1_000
BAIDU_SAFE_COORDINATE_DISTANCE_METERS = 300


class BaiduHoursProvider:
    """Read regular hours without inferring temporary business status."""

    def __init__(
        self,
        api_key: str,
        *,
        client: httpx.AsyncClient | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        timeout_seconds: float = 5.0,
    ) -> None:
        if not api_key.strip():
            raise ValueError("BAIDU_MAP_AK must not be empty")
        self._api_key = api_key
        self._client = client or httpx.AsyncClient(
            base_url=BAIDU_BASE_URL,
            timeout=timeout_seconds,
        )
        self._owns_client = client is None
        self._clock = clock

    async def get_regular_hours(
        self, request: HoursRequest
    ) -> ProviderResponse[ProviderRegularHours]:
        source_place_id = request.source_place_ids.get(ProviderCode.BAIDU)
        if source_place_id is None:
            source_place_id = await self._match_source_place_id(request)
        if source_place_id is None:
            return ProviderResponse[ProviderRegularHours](
                provider=ProviderCode.BAIDU,
                status=ProviderResultStatus.EMPTY,
                fetched_at=self._fetched_at(),
            )

        payload = await request_baidu_json(
            self._client,
            self._api_key,
            BAIDU_DETAIL_PATH,
            {
                "uid": source_place_id,
                "scope": "2",
                "output": "json",
                "ret_coordtype": "gcj02ll",
            },
            "regular_hours_detail",
        )
        record = _detail_record(payload, source_place_id)
        if record is None:
            return ProviderResponse[ProviderRegularHours](
                provider=ProviderCode.BAIDU,
                status=ProviderResultStatus.EMPTY,
                fetched_at=self._fetched_at(),
            )
        detail_info = record.get("detail_info")
        if detail_info in (None, "", []):
            detail_data: dict[str, Any] = {}
        elif isinstance(detail_info, dict):
            detail_data = cast(dict[str, Any], detail_info)
        else:
            raise baidu_malformed("regular_hours_detail")

        fetched_at = self._fetched_at()
        shop_hours = _optional_text(detail_data.get("shop_hours"))
        if shop_hours is None:
            item = ProviderRegularHours(
                provider=ProviderCode.BAIDU,
                source_place_id=source_place_id,
                fetched_at=fetched_at,
                missing_reason="Baidu did not return regular opening hours",
                raw_payload=record,
            )
            return ProviderResponse[ProviderRegularHours](
                provider=ProviderCode.BAIDU,
                status=ProviderResultStatus.PARTIAL,
                items=[item],
                fetched_at=fetched_at,
                missing_fields=["items.display_text"],
            )

        item = ProviderRegularHours(
            provider=ProviderCode.BAIDU,
            source_place_id=source_place_id,
            display_text=shop_hours,
            fetched_at=fetched_at,
            raw_payload=record,
        )
        return ProviderResponse[ProviderRegularHours](
            provider=ProviderCode.BAIDU,
            status=ProviderResultStatus.SUCCESS,
            items=[item],
            fetched_at=fetched_at,
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _match_source_place_id(self, request: HoursRequest) -> str | None:
        payload = await request_baidu_json(
            self._client,
            self._api_key,
            BAIDU_AROUND_PATH,
            {
                "query": request.name,
                "location": _format_location(request.coordinates),
                "radius": str(BAIDU_MATCH_RADIUS_METERS),
                "radius_limit": "true",
                "scope": "1",
                "coord_type": "2",
                "ret_coordtype": "gcj02ll",
                "page_size": "10",
                "output": "json",
            },
            "regular_hours_match",
        )
        results = payload.get("results")
        if not isinstance(results, list):
            raise baidu_malformed("regular_hours_match")
        for raw_candidate in results:
            if not isinstance(raw_candidate, dict):
                raise baidu_malformed("regular_hours_match")
            candidate = cast(dict[str, Any], raw_candidate)
            if _is_safe_match(request, candidate):
                return _required_text(candidate.get("uid"), "regular_hours_match")
        return None

    def _fetched_at(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("provider clock must return an aware datetime")
        return value.astimezone(UTC)


def baidu_hours_to_canonical(
    hours: ProviderRegularHours,
    *,
    place_id: UUID,
) -> RegularHours:
    """Convert one provider result without adding unsupported weekly periods."""

    if hours.provider is not ProviderCode.BAIDU:
        raise ValueError("only Baidu hours can use the Baidu canonicalization helper")
    availability = (
        DataAvailability.MISSING if hours.missing_reason is not None else DataAvailability.AVAILABLE
    )
    return RegularHours(
        place_id=place_id,
        provider=ProviderCode.BAIDU,
        availability=availability,
        display_text=hours.display_text,
        weekly_periods=hours.weekly_periods,
        source_record_id=hours.source_place_id,
        fetched_at=hours.fetched_at,
        missing_reason=hours.missing_reason,
    )


def _is_safe_match(request: HoursRequest, candidate: dict[str, Any]) -> bool:
    name = _optional_text(candidate.get("name"))
    if name is None or _normalize_text(name) != _normalize_text(request.name):
        return False
    if not _same_city(request, candidate):
        return False
    address = _optional_text(candidate.get("address"))
    address_matches = (
        request.address is not None
        and address is not None
        and _texts_overlap(request.address, address)
    )
    candidate_coordinates = _candidate_coordinates(candidate.get("location"))
    coordinates_match = (
        candidate_coordinates is not None
        and _distance_meters(request.coordinates, candidate_coordinates)
        <= BAIDU_SAFE_COORDINATE_DISTANCE_METERS
    )
    return address_matches or coordinates_match


def _detail_record(
    payload: dict[str, Any],
    source_place_id: str,
) -> dict[str, Any] | None:
    """Read current Place v3 `results` while retaining legacy recording compatibility."""

    singular = payload.get("result")
    if singular not in (None, "", []):
        if not isinstance(singular, dict):
            raise baidu_malformed("regular_hours_detail")
        return cast(dict[str, Any], singular)

    plural = payload.get("results")
    if plural in (None, "", []):
        return None
    if not isinstance(plural, list) or any(not isinstance(item, dict) for item in plural):
        raise baidu_malformed("regular_hours_detail")
    matches = [cast(dict[str, Any], item) for item in plural if item.get("uid") == source_place_id]
    if len(matches) != 1:
        raise baidu_malformed("regular_hours_detail")
    return matches[0]


def _same_city(request: HoursRequest, candidate: dict[str, Any]) -> bool:
    candidate_codes = {
        str(value).strip()
        for key in ("city_id", "adcode")
        if (value := candidate.get(key)) not in (None, "", [])
    }
    if request.city.provider_city_code in candidate_codes:
        return True
    candidate_city = _optional_text(candidate.get("city"))
    display_name = request.city.display_name
    if candidate_city is None or display_name is None:
        return False
    return _normalize_city(candidate_city) == _normalize_city(display_name)


def _candidate_coordinates(value: Any) -> Gcj02Coordinates | None:
    if not isinstance(value, dict):
        return None
    latitude = value.get("lat")
    longitude = value.get("lng")
    if (
        latitude is None
        or longitude is None
        or isinstance(latitude, bool)
        or isinstance(longitude, bool)
    ):
        return None
    try:
        return Gcj02Coordinates(
            latitude=float(latitude),
            longitude=float(longitude),
            coord_system=CoordinateSystem.GCJ_02,
        )
    except (TypeError, ValueError):
        return None


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


def _format_location(coordinates: Gcj02Coordinates) -> str:
    return f"{coordinates.latitude:.6f},{coordinates.longitude:.6f}"


def _normalize_text(value: str) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]", "", value.casefold())


def _normalize_city(value: str) -> str:
    normalized = _normalize_text(value)
    return normalized.removesuffix("市")


def _texts_overlap(left: str, right: str) -> bool:
    normalized_left = _normalize_text(left)
    normalized_right = _normalize_text(right)
    return bool(normalized_left and normalized_right) and (
        normalized_left in normalized_right or normalized_right in normalized_left
    )


def _required_text(value: Any, operation: str) -> str:
    parsed = _optional_text(value)
    if parsed is None:
        raise baidu_malformed(operation)
    return parsed


def _optional_text(value: Any) -> str | None:
    if value is None or value == "" or value == []:
        return None
    if not isinstance(value, str):
        return None
    parsed = value.strip()
    return parsed or None
