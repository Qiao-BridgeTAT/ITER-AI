"""AMap Web Service v5 route adapter with authorization-gated caching."""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, cast

import httpx
from pydantic import ValidationError

from backend.contracts.common import CnyAmountRange
from backend.contracts.enums import CoordinateSystem, ProviderCode
from backend.contracts.places import Gcj02Coordinates
from backend.providers.amap_http import AMAP_BASE_URL, amap_malformed, request_amap_json
from backend.providers.contracts import (
    ProviderError,
    ProviderFailureDetail,
    ProviderResponse,
    ProviderResultStatus,
    ProviderRoute,
    RouteMode,
    RouteRequest,
)
from backend.providers.interfaces import ProviderCache

AMAP_ROUTE_PATHS = {
    RouteMode.WALKING: "/v5/direction/walking",
    RouteMode.CYCLING: "/v5/direction/bicycling",
    RouteMode.TRANSIT: "/v5/direction/transit/integrated",
    RouteMode.DRIVING: "/v5/direction/driving",
}
AMAP_ROUTE_CACHE_VERSION = "amap-route-v5-normalized-v2-polyline"
AMAP_ROUTE_CACHE_TTL_SECONDS = 0
_LOGGER = logging.getLogger(__name__)


class AmapRouteProvider:
    """Normalize comparable walking, cycling, transit, and driving routes."""

    def __init__(
        self,
        api_key: str,
        *,
        client: httpx.AsyncClient | None = None,
        cache: ProviderCache | None = None,
        cache_ttl_seconds: int = AMAP_ROUTE_CACHE_TTL_SECONDS,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        timeout_seconds: float = 5.0,
    ) -> None:
        if not api_key.strip():
            raise ValueError("AMAP_WEB_SERVICE_KEY must not be empty")
        self._api_key = api_key
        self._client = client or httpx.AsyncClient(
            base_url=AMAP_BASE_URL,
            timeout=timeout_seconds,
        )
        self._owns_client = client is None
        self._cache = cache
        if cache_ttl_seconds < 0:
            raise ValueError("AMap route cache TTL cannot be negative")
        self._cache_ttl_seconds = cache_ttl_seconds
        self._clock = clock

    async def get_routes(self, request: RouteRequest) -> ProviderResponse[ProviderRoute]:
        results: list[ProviderRoute] = []
        missing_fields: set[str] = set()
        failures: dict[str, ProviderFailureDetail] = {}
        first_error: ProviderError | None = None
        fetched_at: datetime | None = None

        for mode in request.modes:
            try:
                response = await self._get_mode_routes(request, mode)
            except ProviderError as error:
                first_error = first_error or error
                failures[mode.value] = ProviderFailureDetail.from_error(error)
                missing_fields.add(f"modes.{mode.value}")
                continue
            fetched_at = max(fetched_at, response.fetched_at) if fetched_at else response.fetched_at
            results.extend(response.items)
            missing_fields.update(response.missing_fields)
            if response.status is ProviderResultStatus.EMPTY:
                missing_fields.add(f"modes.{mode.value}")

        response_time = fetched_at or self._fetched_at()
        if not results:
            if first_error is not None:
                first_error.failures = failures
                raise first_error
            return ProviderResponse[ProviderRoute](
                provider=ProviderCode.AMAP,
                status=ProviderResultStatus.EMPTY,
                fetched_at=response_time,
            )
        return ProviderResponse[ProviderRoute](
            provider=ProviderCode.AMAP,
            status=(
                ProviderResultStatus.PARTIAL if missing_fields else ProviderResultStatus.SUCCESS
            ),
            items=results,
            fetched_at=response_time,
            missing_fields=sorted(missing_fields),
            failures=failures,
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _get_mode_routes(
        self,
        request: RouteRequest,
        mode: RouteMode,
    ) -> ProviderResponse[ProviderRoute]:
        cache_parameters = _cache_parameters(request, mode)
        cached = await self._read_cache(cache_parameters) if self._cache_ttl_seconds > 0 else None
        if cached is not None:
            return cached

        parameters = {
            "origin": _format_location(request.origin),
            "destination": _format_location(request.destination),
            "show_fields": "cost,polyline",
        }
        if mode in {RouteMode.WALKING, RouteMode.CYCLING}:
            parameters["alternative_route"] = "3"
        elif mode is RouteMode.TRANSIT:
            transit_code = request.city.provider_transit_city_code
            if transit_code is None:  # guarded by RouteRequest; protects model_construct callers.
                raise amap_malformed("route_transit")
            parameters.update(
                {
                    "city1": transit_code,
                    "city2": transit_code,
                    "AlternativeRoute": "3",
                }
            )

        operation = f"route_{mode.value}"
        payload = await request_amap_json(
            self._client,
            self._api_key,
            AMAP_ROUTE_PATHS[mode],
            parameters,
            operation,
        )
        response = self._parse_mode(payload, mode, operation)
        if self._cache is not None and self._cache_ttl_seconds > 0:
            try:
                await self._cache.put_cache(
                    ProviderCode.AMAP.value,
                    AMAP_ROUTE_CACHE_VERSION,
                    cache_parameters,
                    response.model_dump(mode="json"),
                    self._cache_ttl_seconds,
                )
            except Exception:
                _LOGGER.warning("AMap route cache write failed; returning live result")
        return response

    async def _read_cache(
        self, parameters: dict[str, str]
    ) -> ProviderResponse[ProviderRoute] | None:
        if self._cache is None:
            return None
        try:
            cached = await self._cache.get_cache(
                ProviderCode.AMAP.value,
                AMAP_ROUTE_CACHE_VERSION,
                parameters,
            )
        except Exception:
            _LOGGER.warning("AMap route cache read failed; continuing without cache")
            return None
        if cached is None:
            return None
        try:
            return ProviderResponse[ProviderRoute].model_validate(cached)
        except ValidationError:
            return None

    def _parse_mode(
        self,
        payload: dict[str, Any],
        mode: RouteMode,
        operation: str,
    ) -> ProviderResponse[ProviderRoute]:
        fetched_at = self._fetched_at()
        route = payload.get("route")
        if not isinstance(route, dict):
            raise amap_malformed(operation)
        route_data = cast(dict[str, Any], route)
        collection_name = "transits" if mode is RouteMode.TRANSIT else "paths"
        records = route_data.get(collection_name)
        if not isinstance(records, list):
            raise amap_malformed(operation)
        if not records:
            return ProviderResponse[ProviderRoute](
                provider=ProviderCode.AMAP,
                status=ProviderResultStatus.EMPTY,
                fetched_at=fetched_at,
            )

        items: list[ProviderRoute] = []
        missing_fields: set[str] = set()
        for index, raw_record in enumerate(records):
            if not isinstance(raw_record, dict):
                raise amap_malformed(operation)
            record = cast(dict[str, Any], raw_record)
            cost = record.get("cost")
            if not isinstance(cost, dict):
                raise amap_malformed(operation)
            cost_data = cast(dict[str, Any], cost)
            distance = _required_nonnegative_int(record.get("distance"), operation)
            duration = _required_nonnegative_int(cost_data.get("duration"), operation)

            walking_distance: int | None = None
            transfer_count: int | None = None
            fare: CnyAmountRange | None = None
            if mode is RouteMode.WALKING:
                walking_distance = distance
            elif mode is RouteMode.TRANSIT:
                walking_distance = _transit_walking_distance(record, operation)
                transfer_count = _transit_transfer_count(record, operation)
                fare = _optional_yuan_amount(cost_data.get("transit_fee"), operation)
                if walking_distance is None:
                    missing_fields.add("items.walking_distance_m")
                if transfer_count is None:
                    missing_fields.add("items.transfer_count")
                if fare is None:
                    missing_fields.add("items.fare")
            elif mode is RouteMode.DRIVING:
                fare = _optional_yuan_amount(route_data.get("taxi_cost"), operation)
                if fare is None:
                    missing_fields.add("items.fare")

            polyline = _route_polyline(record)

            items.append(
                ProviderRoute(
                    provider=ProviderCode.AMAP,
                    mode=mode,
                    source_route_index=index,
                    distance_m=distance,
                    duration_seconds=duration,
                    walking_distance_m=walking_distance,
                    transfer_count=transfer_count,
                    fare=fare,
                    polyline=polyline,
                    fetched_at=fetched_at,
                    raw_payload=record,
                )
            )

        return ProviderResponse[ProviderRoute](
            provider=ProviderCode.AMAP,
            status=(
                ProviderResultStatus.PARTIAL if missing_fields else ProviderResultStatus.SUCCESS
            ),
            items=items,
            fetched_at=fetched_at,
            missing_fields=sorted(missing_fields),
        )

    def _fetched_at(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("provider clock must return an aware datetime")
        return value.astimezone(UTC)


def _route_polyline(record: dict[str, Any]) -> list[Gcj02Coordinates]:
    """Collect normalized GCJ-02 geometry without exposing the raw AMap shape."""

    points: list[Gcj02Coordinates] = []

    def collect(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "polyline" and isinstance(child, str):
                    for pair in child.split(";"):
                        values = pair.split(",")
                        if len(values) != 2:
                            continue
                        try:
                            point = Gcj02Coordinates(
                                latitude=float(values[1]),
                                longitude=float(values[0]),
                                coord_system=CoordinateSystem.GCJ_02,
                            )
                        except (ValueError, ValidationError):
                            continue
                        if not points or point != points[-1]:
                            points.append(point)
                else:
                    collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    collect(record)
    return points if len(points) >= 2 else []


def _cache_parameters(request: RouteRequest, mode: RouteMode) -> dict[str, str]:
    return {
        "city_id": request.city.city_id,
        "provider_city_code": request.city.provider_city_code,
        "provider_transit_city_code": request.city.provider_transit_city_code or "",
        "mode": mode.value,
        "origin": _format_location(request.origin),
        "destination": _format_location(request.destination),
    }


def _format_location(coordinates: Gcj02Coordinates) -> str:
    return f"{coordinates.longitude:.6f},{coordinates.latitude:.6f}"


def _required_nonnegative_int(value: Any, operation: str) -> int:
    if isinstance(value, bool):
        raise amap_malformed(operation)
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise amap_malformed(operation) from None
    if not parsed.is_finite() or parsed < 0 or parsed != parsed.to_integral_value():
        raise amap_malformed(operation)
    return int(parsed)


def _optional_yuan_amount(value: Any, operation: str) -> CnyAmountRange | None:
    if value is None or value == "" or value == []:
        return None
    if isinstance(value, bool):
        raise amap_malformed(operation)
    try:
        yuan = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise amap_malformed(operation) from None
    fen = yuan * 100
    if not yuan.is_finite() or yuan < 0 or fen != fen.to_integral_value():
        raise amap_malformed(operation)
    amount = int(fen)
    return CnyAmountRange(minimum_fen=amount, maximum_fen=amount)


def _transit_walking_distance(record: dict[str, Any], operation: str) -> int | None:
    segments = record.get("segments")
    if not isinstance(segments, list):
        raise amap_malformed(operation)
    total = 0
    found = False
    for raw_segment in segments:
        if not isinstance(raw_segment, dict):
            raise amap_malformed(operation)
        walking = raw_segment.get("walking")
        if walking in (None, "", []):
            continue
        if not isinstance(walking, dict):
            raise amap_malformed(operation)
        distance = walking.get("distance")
        if distance in (None, "", []):
            continue
        total += _required_nonnegative_int(distance, operation)
        found = True
    return total if found else None


def _transit_transfer_count(record: dict[str, Any], operation: str) -> int | None:
    segments = record.get("segments")
    if not isinstance(segments, list):
        raise amap_malformed(operation)
    ride_legs = 0
    recognized = False
    for raw_segment in segments:
        if not isinstance(raw_segment, dict):
            raise amap_malformed(operation)
        segment = cast(dict[str, Any], raw_segment)
        bus = segment.get("bus")
        if isinstance(bus, dict):
            buslines = bus.get("buslines")
            if isinstance(buslines, list):
                recognized = True
                if buslines:
                    ride_legs += 1
            elif buslines not in (None, "", []):
                raise amap_malformed(operation)
        for field in ("railway", "taxi"):
            ride = segment.get(field)
            if isinstance(ride, dict) and ride:
                recognized = True
                ride_legs += 1
            elif ride not in (None, "", []):
                raise amap_malformed(operation)
    return max(0, ride_legs - 1) if recognized else None
