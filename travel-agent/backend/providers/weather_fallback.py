"""Try a secondary weather source only for absent days/essential daily values."""

from datetime import timedelta

from backend.contracts.enums import ProviderCode
from backend.providers.contracts import (
    ProviderError,
    ProviderForecastDay,
    ProviderResponse,
    ProviderResultStatus,
    WeatherRequest,
)
from backend.providers.interfaces import WeatherProvider


class FallbackWeatherProvider:
    def __init__(self, primary: WeatherProvider, fallback: WeatherProvider) -> None:
        self._primary = primary
        self._fallback = fallback

    async def get_forecast(self, request: WeatherRequest) -> ProviderResponse[ProviderForecastDay]:
        primary_error = None
        primary = None
        try:
            primary = await self._primary.get_forecast(request)
        except ProviderError as error:
            primary_error = error
        dates = {
            request.start_date + timedelta(days=offset)
            for offset in range((request.end_date - request.start_date).days + 1)
        }
        items = {item.forecast_date: item for item in primary.items} if primary else {}
        missing = dates - {day for day, item in items.items() if _complete(item)}
        if not missing:
            assert primary is not None
            return primary
        try:
            fallback = await self._fallback.get_forecast(
                WeatherRequest(city=request.city, start_date=min(missing), end_date=max(missing))
            )
        except ProviderError:
            if primary is not None and primary.items:
                return primary
            if primary_error is not None:
                raise primary_error from None
            raise
        if not fallback.items:
            if primary_error is not None:
                raise primary_error
            assert primary is not None
            return primary
        for item in fallback.items:
            previous = items.get(item.forecast_date)
            if (
                item.forecast_date in dates
                and (previous is None or _complete(item))
                and (previous is None or not _complete(previous))
            ):
                items[item.forecast_date] = item
        missing_fields = {f"dates.{day.isoformat()}" for day in dates - items.keys()}
        for item in items.values():
            for field in ("condition_day", "condition_night", "low_celsius", "high_celsius"):
                if getattr(item, field) is None:
                    missing_fields.add(f"items.{field}")
        return ProviderResponse[ProviderForecastDay](
            provider=ProviderCode.WEATHER,
            status=ProviderResultStatus.PARTIAL if missing_fields else ProviderResultStatus.SUCCESS,
            items=[items[day] for day in sorted(items)],
            fetched_at=max(item.fetched_at for item in items.values()),
            missing_fields=sorted(missing_fields),
        )


def _complete(item: ProviderForecastDay) -> bool:
    # A daily weather product lacking a distinct night condition is still useful;
    # do not double vendor calls for a field that the product does not supply.
    return (
        item.condition_day is not None
        and item.low_celsius is not None
        and item.high_celsius is not None
    )
