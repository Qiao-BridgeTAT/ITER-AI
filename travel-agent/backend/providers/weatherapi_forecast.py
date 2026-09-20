"""WeatherAPI.com daily forecast adapter for the stage-0 travel date window."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any, cast

import httpx
from pydantic import ValidationError

from backend.contracts.enums import ProviderCode
from backend.providers.contracts import (
    ProviderError,
    ProviderFailureCode,
    ProviderForecastDay,
    ProviderResponse,
    ProviderResultStatus,
    WeatherRequest,
)
from backend.providers.request_budget import budgeted_external_request

WEATHERAPI_BASE_URL = "https://api.weatherapi.com"
WEATHERAPI_FORECAST_PATH = "/v1/forecast.json"
WEATHERAPI_MAX_FORECAST_DAYS = 14


class WeatherApiForecastProvider:
    """Return only dated forecast facts actually supplied by WeatherAPI.com."""

    def __init__(
        self,
        api_key: str,
        *,
        client: httpx.AsyncClient | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        today: Callable[[], date] = date.today,
        timeout_seconds: float = 5.0,
    ) -> None:
        if not api_key.strip():
            raise ValueError("WEATHER_API_KEY must not be empty")
        self._api_key = api_key
        self._client = client or httpx.AsyncClient(
            base_url=WEATHERAPI_BASE_URL,
            timeout=timeout_seconds,
        )
        self._owns_client = client is None
        self._clock = clock
        self._today = today

    async def get_forecast(self, request: WeatherRequest) -> ProviderResponse[ProviderForecastDay]:
        today = self._today()
        available_end = today + timedelta(days=WEATHERAPI_MAX_FORECAST_DAYS - 1)
        query_start = max(request.start_date, today)
        query_end = min(request.end_date, available_end)
        if query_start > query_end:
            return ProviderResponse[ProviderForecastDay](
                provider=ProviderCode.WEATHER,
                status=ProviderResultStatus.EMPTY,
                fetched_at=self._fetched_at(),
            )
        city_name = request.city.display_name
        if city_name is None:
            raise ProviderError(
                ProviderCode.WEATHER,
                ProviderFailureCode.INVALID_REQUEST,
                "forecast",
                retryable=False,
            )
        days = (query_end - today).days + 1
        payload = await self._request(
            {
                "q": city_name,
                "days": str(days),
                "aqi": "no",
                "alerts": "no",
                "lang": "zh",
            }
        )
        return self._parse(payload, request, query_start, query_end)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    @budgeted_external_request
    async def _request(self, parameters: dict[str, str]) -> dict[str, Any]:
        operation = "forecast"
        try:
            response = await self._client.get(
                f"{WEATHERAPI_BASE_URL}{WEATHERAPI_FORECAST_PATH}",
                params={"key": self._api_key, **parameters},
            )
        except httpx.TimeoutException:
            raise ProviderError(
                ProviderCode.WEATHER,
                ProviderFailureCode.TIMEOUT,
                operation,
                retryable=True,
            ) from None
        except httpx.HTTPError:
            raise ProviderError(
                ProviderCode.WEATHER,
                ProviderFailureCode.UNAVAILABLE,
                operation,
                retryable=True,
            ) from None
        try:
            payload = response.json()
        except ValueError:
            raise _weather_malformed(operation) from None
        if not isinstance(payload, dict):
            raise _weather_malformed(operation)
        parsed = cast(dict[str, Any], payload)
        if response.is_error or "error" in parsed:
            raise _weather_payload_error(parsed, response.status_code, operation)
        return parsed

    def _parse(
        self,
        payload: dict[str, Any],
        request: WeatherRequest,
        query_start: date,
        query_end: date,
    ) -> ProviderResponse[ProviderForecastDay]:
        forecast = payload.get("forecast")
        if not isinstance(forecast, dict):
            raise _weather_malformed("forecast")
        records = forecast.get("forecastday")
        if not isinstance(records, list):
            raise _weather_malformed("forecast")
        fetched_at = self._fetched_at()
        items: list[ProviderForecastDay] = []
        missing_fields: set[str] = set()
        seen_dates: set[date] = set()
        for raw_record in records:
            if not isinstance(raw_record, dict):
                raise _weather_malformed("forecast")
            record = cast(dict[str, Any], raw_record)
            forecast_date = _required_date(record.get("date"), "forecast")
            if forecast_date < query_start or forecast_date > query_end:
                continue
            if forecast_date in seen_dates:
                raise _weather_malformed("forecast")
            day = record.get("day")
            if not isinstance(day, dict):
                raise _weather_malformed("forecast")
            day_data = cast(dict[str, Any], day)
            condition_day = _condition_text(day_data.get("condition"))
            condition_night = _night_condition(record.get("hour"))
            low = _optional_temperature(day_data.get("mintemp_c"))
            high = _optional_temperature(day_data.get("maxtemp_c"))
            values = {
                "condition_day": condition_day,
                "condition_night": condition_night,
                "low_celsius": low,
                "high_celsius": high,
            }
            for field, value in values.items():
                if value is None:
                    missing_fields.add(f"items.{field}")
            if all(value is None for value in values.values()):
                continue
            seen_dates.add(forecast_date)
            try:
                item = ProviderForecastDay(
                    provider=ProviderCode.WEATHER,
                    forecast_date=forecast_date,
                    condition_day=condition_day,
                    condition_night=condition_night,
                    low_celsius=low,
                    high_celsius=high,
                    source_name="WeatherAPI",
                    fetched_at=fetched_at,
                    raw_payload=record,
                )
            except ValidationError:
                raise _weather_malformed("forecast") from None
            items.append(item)

        requested_dates = {
            request.start_date + timedelta(days=offset)
            for offset in range((request.end_date - request.start_date).days + 1)
        }
        for missing_date in requested_dates - seen_dates:
            missing_fields.add(f"dates.{missing_date.isoformat()}")
        if not items:
            return ProviderResponse[ProviderForecastDay](
                provider=ProviderCode.WEATHER,
                status=ProviderResultStatus.EMPTY,
                fetched_at=fetched_at,
            )
        return ProviderResponse[ProviderForecastDay](
            provider=ProviderCode.WEATHER,
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


def _weather_payload_error(
    payload: dict[str, Any], status_code: int, operation: str
) -> ProviderError:
    error = payload.get("error")
    error_code: int | None = None
    if isinstance(error, dict):
        value = error.get("code")
        if isinstance(value, int) and not isinstance(value, bool):
            error_code = value
    if error_code == 2007:
        code = ProviderFailureCode.RATE_LIMITED
    elif error_code in {1002, 2006}:
        code = ProviderFailureCode.AUTHENTICATION_FAILED
    elif error_code in {2008, 2009}:
        code = ProviderFailureCode.PERMISSION_DENIED
    elif error_code in {1003, 1005, 1006}:
        code = ProviderFailureCode.INVALID_REQUEST
    elif error_code == 9999:
        code = ProviderFailureCode.UNAVAILABLE
    elif status_code == 401:
        code = ProviderFailureCode.AUTHENTICATION_FAILED
    elif status_code == 403:
        code = ProviderFailureCode.PERMISSION_DENIED
    elif status_code == 429:
        code = ProviderFailureCode.RATE_LIMITED
    elif status_code in {400, 404, 422}:
        code = ProviderFailureCode.INVALID_REQUEST
    elif status_code >= 500:
        code = ProviderFailureCode.UNAVAILABLE
    else:
        code = ProviderFailureCode.UPSTREAM_ERROR
    return ProviderError(
        ProviderCode.WEATHER,
        code,
        operation,
        retryable=code
        in {
            ProviderFailureCode.RATE_LIMITED,
            ProviderFailureCode.UNAVAILABLE,
            ProviderFailureCode.UPSTREAM_ERROR,
        },
    )


def _weather_malformed(operation: str) -> ProviderError:
    return ProviderError(
        ProviderCode.WEATHER,
        ProviderFailureCode.MALFORMED_RESPONSE,
        operation,
        retryable=False,
    )


def _required_date(value: Any, operation: str) -> date:
    if not isinstance(value, str):
        raise _weather_malformed(operation)
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise _weather_malformed(operation) from None


def _condition_text(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    text = value.get("text")
    if not isinstance(text, str):
        return None
    parsed = text.strip()
    return parsed or None


def _night_condition(value: Any) -> str | None:
    if not isinstance(value, list):
        return None
    candidates: list[tuple[int, str]] = []
    for raw_hour in value:
        if not isinstance(raw_hour, dict):
            continue
        if raw_hour.get("is_day") != 0:
            continue
        time_text = raw_hour.get("time")
        condition = _condition_text(raw_hour.get("condition"))
        if not isinstance(time_text, str) or condition is None:
            continue
        try:
            hour = int(time_text.rsplit(" ", 1)[-1].split(":", 1)[0])
        except ValueError:
            continue
        candidates.append((hour, condition))
    if not candidates:
        return None
    return min(candidates, key=lambda candidate: abs(candidate[0] - 21))[1]


def _optional_temperature(value: Any) -> int | None:
    if value in (None, "", []):
        return None
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    if not parsed.is_finite():
        return None
    return int(parsed.quantize(Decimal("1"), rounding=ROUND_HALF_UP))
