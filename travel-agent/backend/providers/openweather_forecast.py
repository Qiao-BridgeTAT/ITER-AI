"""OpenWeather One Call 4.0 daily timeline, normalized at the Provider boundary."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime, time, timedelta
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any
from zoneinfo import ZoneInfo

import httpx
from pydantic import ValidationError

from backend.contracts.enums import ProviderCode
from backend.planning.city_registry import CityRegistry, CityRegistryError, default_city_registry
from backend.providers.contracts import (
    GeocodeRequest,
    ProviderError,
    ProviderFailureCode,
    ProviderForecastDay,
    ProviderResponse,
    ProviderResultStatus,
    WeatherRequest,
)
from backend.providers.interfaces import PlaceProvider
from backend.providers.place_matching import coordinates_to_wgs84

OPENWEATHER_DAILY_URL = "https://api.openweathermap.org/data/4.0/onecall/timeline/1day"
OPENWEATHER_SOURCE = "OpenWeather One Call 4.0"
# Product presentation policy, not an API permission/range restriction.
SHORT_RANGE_DAYS = 8


class OpenWeatherForecastProvider:
    def __init__(
        self,
        api_key: str,
        *,
        places: PlaceProvider,
        registry: CityRegistry | None = None,
        client: httpx.AsyncClient | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        if not api_key.strip():
            raise ValueError("OPENWEATHER_API_KEY must not be empty")
        self._api_key = api_key.strip()
        self._places = places
        self._registry = registry or default_city_registry()
        self._client = client or httpx.AsyncClient(timeout=5.0, follow_redirects=False)
        self._owns_client = client is None
        self._clock = clock

    async def get_forecast(self, request: WeatherRequest) -> ProviderResponse[ProviderForecastDay]:
        try:
            city = self._registry.resolve(request.city.city_id)
            scope = self._registry.provider_scope(city.city_id, ProviderCode.AMAP)
        except CityRegistryError:
            raise _error(ProviderFailureCode.INVALID_REQUEST) from None
        # Resolve the registered administrative city, never the first phonetic name
        # match (OpenWeather's "Suzhou" also matches 宿州 and 肃州).
        located = await self._places.geocode(
            GeocodeRequest(city=scope, address=f"{city.province_name}{city.display_name}")
        )
        matches = {
            (item.coordinates.longitude, item.coordinates.latitude): item.coordinates
            for item in located.items
            if item.city_id == city.city_id and item.provider_city_code == scope.provider_city_code
        }
        if len(matches) != 1:
            raise _error(ProviderFailureCode.INVALID_REQUEST)
        coordinates = coordinates_to_wgs84(next(iter(matches.values())))
        timezone = ZoneInfo(city.timezone)
        # Daily timeline buckets are not aligned to local midnight. Local noon
        # avoids selecting the preceding UTC day for the supported China cities.
        start = datetime.combine(request.start_date, time(12), timezone)
        payload = await self._request(
            {
                "lat": str(coordinates.latitude),
                "lon": str(coordinates.longitude),
                "start": str(int(start.timestamp())),
                "cnt": str((request.end_date - request.start_date).days + 2),
                "units": "metric",
                "lang": "zh_cn",
            }
        )
        return self._parse(payload, request, timezone)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _request(self, parameters: dict[str, str]) -> dict[str, Any]:
        try:
            response = await self._client.get(
                OPENWEATHER_DAILY_URL, params={"appid": self._api_key, **parameters}
            )
        except httpx.TimeoutException:
            raise _error(ProviderFailureCode.TIMEOUT) from None
        except httpx.HTTPError:
            raise _error(ProviderFailureCode.UNAVAILABLE) from None
        status = response.status_code
        if status >= 400:
            code = (
                ProviderFailureCode.AUTHENTICATION_FAILED
                if status == 401
                else ProviderFailureCode.PERMISSION_DENIED
                if status == 403
                else ProviderFailureCode.RATE_LIMITED
                if status == 429
                else ProviderFailureCode.UNAVAILABLE
                if status >= 500
                else ProviderFailureCode.INVALID_REQUEST
            )
            raise _error(code)
        try:
            payload: Any = response.json()
        except ValueError:
            raise _error(ProviderFailureCode.MALFORMED_RESPONSE) from None
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            raise _error(ProviderFailureCode.MALFORMED_RESPONSE)
        return payload

    def _parse(
        self, payload: dict[str, Any], request: WeatherRequest, timezone: ZoneInfo
    ) -> ProviderResponse[ProviderForecastDay]:
        fetched_at = self._clock().astimezone(UTC)
        today = fetched_at.astimezone(timezone).date()
        dates = {
            request.start_date + timedelta(days=offset)
            for offset in range((request.end_date - request.start_date).days + 1)
        }
        items: dict[date, ProviderForecastDay] = {}
        for record in payload["data"]:
            if not isinstance(record, dict):
                continue
            try:
                timestamp = record.get("dt")
                if not isinstance(timestamp, (int, float)) or isinstance(timestamp, bool):
                    continue
                service_date = datetime.fromtimestamp(timestamp, timezone).date()
                if service_date not in dates or service_date in items:
                    continue
                temp = record.get("temp")
                temp = temp if isinstance(temp, dict) else {}
                conditions = record.get("weather")
                conditions = conditions if isinstance(conditions, list) else []
                text = next(
                    (
                        value["description"].strip()
                        for value in conditions
                        if isinstance(value, dict)
                        and isinstance(value.get("description"), str)
                        and value["description"].strip()
                    ),
                    None,
                )
                items[service_date] = ProviderForecastDay(
                    provider=ProviderCode.WEATHER,
                    forecast_date=service_date,
                    condition_day=text,
                    # Daily summaries do not provide a separate night condition.
                    condition_night=None,
                    low_celsius=_temperature(temp.get("min")),
                    high_celsius=_temperature(temp.get("max")),
                    source_name=OPENWEATHER_SOURCE,
                    forecast_kind=(
                        "outlook" if (service_date - today).days >= SHORT_RANGE_DAYS else "forecast"
                    ),
                    fetched_at=fetched_at,
                    # Never retain pagination URLs (they contain appid) or raw payloads.
                    raw_payload={},
                )
            except (ValueError, OverflowError, OSError, ValidationError):
                # A malformed day must not discard other valid travel dates.
                continue
        missing = {f"dates.{value.isoformat()}" for value in dates - items.keys()}
        for item in items.values():
            for field in ("condition_day", "condition_night", "low_celsius", "high_celsius"):
                if getattr(item, field) is None:
                    missing.add(f"items.{field}")
        return ProviderResponse[ProviderForecastDay](
            provider=ProviderCode.WEATHER,
            status=(ProviderResultStatus.PARTIAL if missing else ProviderResultStatus.SUCCESS)
            if items
            else ProviderResultStatus.EMPTY,
            items=[items[value] for value in sorted(items)],
            fetched_at=fetched_at,
            missing_fields=sorted(missing) if items else [],
            provider_notice=OPENWEATHER_SOURCE,
        )


def _temperature(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = Decimal(str(value))
        if not number.is_finite() or not -100 <= number <= 70:
            return None
        return int(number.quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    except (InvalidOperation, ValueError):
        return None


def _error(code: ProviderFailureCode) -> ProviderError:
    return ProviderError(
        ProviderCode.WEATHER,
        code,
        "forecast",
        retryable=code in {ProviderFailureCode.TIMEOUT, ProviderFailureCode.UNAVAILABLE},
    )
