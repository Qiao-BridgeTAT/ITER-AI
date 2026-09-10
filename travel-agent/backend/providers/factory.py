"""Runtime composition for live versus record/replay provider integration."""

from __future__ import annotations

from dataclasses import dataclass, replace


from backend.config import ConfigurationError, Settings
from backend.contracts.enums import ProviderCode
from backend.persistence.redis_temporary import RedisTemporaryStore
from backend.providers.amap_hours import AmapHoursProvider
from backend.providers.amap_http import AMAP_PROXY_SEARCH_TIMEOUT_SECONDS
from backend.providers.amap_places import AmapPlaceProvider
from backend.providers.amap_routes import AmapRouteProvider
from backend.providers.flyai_products import (
    FlyAiCliTransport,
    FlyAiProductProvider,
)
from backend.providers.interfaces import (
    HoursProvider,
    PlaceProvider,
    RouteProvider,
    TravelProductProvider,
    WeatherProvider,
)
from backend.providers.managed import (
    ManagedHoursProvider,
    ManagedPlaceProvider,
    ManagedRouteProvider,
    ManagedTravelProductProvider,
    ManagedWeatherProvider,
)
from backend.providers.openweather_forecast import OpenWeatherForecastProvider
from backend.providers.resilience import DEFAULT_PROVIDER_POLICIES, ProviderRuntime
from backend.providers.weather_fallback import FallbackWeatherProvider
from backend.providers.weatherapi_forecast import WeatherApiForecastProvider


@dataclass
class ProviderGatewayRuntime:
    closeables: tuple[object, ...] = ()
    places: PlaceProvider | None = None
    routes: RouteProvider | None = None
    hours: HoursProvider | None = None
    products: TravelProductProvider | None = None
    weather: WeatherProvider | None = None

    async def close(self) -> None:
        for provider in self.closeables:
            close = getattr(provider, "aclose", None)
            if close is not None:
                await close()


def build_provider_gateway(
    settings: Settings, temporary: RedisTemporaryStore
) -> ProviderGatewayRuntime:
    if settings.provider_mode != "live":
        raise ConfigurationError("This distribution requires live Providers")

    required = ("AMAP_WEB_SERVICE_KEY", "FLYAI_API_KEY")
    missing = [name for name in required if not settings.values.get(name, "").strip()]
    if missing:
        raise ConfigurationError(
            "Live provider mode requires server configuration: " + ", ".join(sorted(missing))
        )

    amap_places = AmapPlaceProvider(
        settings.values["AMAP_WEB_SERVICE_KEY"], search_proxy=settings.amap_search_proxy
    )
    amap_routes = AmapRouteProvider(settings.values["AMAP_WEB_SERVICE_KEY"])
    amap_hours = AmapHoursProvider(
        settings.values["AMAP_WEB_SERVICE_KEY"], search_proxy=settings.amap_search_proxy
    )
    flyai_products = FlyAiProductProvider(FlyAiCliTransport(settings.values["FLYAI_API_KEY"]))
    weather_key = settings.values.get("WEATHER_API_KEY", "").strip()
    weather = WeatherApiForecastProvider(weather_key) if weather_key else None

    amap_policy = DEFAULT_PROVIDER_POLICIES[ProviderCode.AMAP]
    if settings.v4_planner_enabled:
        amap_policy = replace(
            amap_policy,
            max_concurrency=3,
            rate_limit=3,
            rate_wait_seconds=3.0,
            retry_delay_seconds=1.0,
        )
    amap_runtime = ProviderRuntime(
        ProviderCode.AMAP,
        amap_policy,
        cache=temporary,
        rate_limiter=temporary,
        operation_timeout_seconds={
            operation: AMAP_PROXY_SEARCH_TIMEOUT_SECONDS
            for operation in ("search_places", "search_nearby", "get_place", "get_regular_hours")
        }
        if settings.amap_search_proxy is not None
        else None,
    )
    flyai_runtime = ProviderRuntime(
        ProviderCode.FLYAI,
        DEFAULT_PROVIDER_POLICIES[ProviderCode.FLYAI],
        cache=temporary,
        rate_limiter=temporary,
    )
    managed_places = ManagedPlaceProvider(amap_places, amap_runtime)
    managed_weather: WeatherProvider | None = None
    if weather is not None:
        weather_runtime = ProviderRuntime(
            ProviderCode.WEATHER,
            replace(DEFAULT_PROVIDER_POLICIES[ProviderCode.WEATHER], retry_count=0),
            cache=temporary,
            rate_limiter=temporary,
        )
        managed_weather = ManagedWeatherProvider(weather, weather_runtime)

    openweather_key = settings.values.get("OPENWEATHER_API_KEY", "").strip()
    openweather = None
    if openweather_key:
        openweather = OpenWeatherForecastProvider(openweather_key, places=managed_places)
        primary = ManagedWeatherProvider(
            openweather,
            ProviderRuntime(
                ProviderCode.WEATHER,
                replace(
                    DEFAULT_PROVIDER_POLICIES[ProviderCode.WEATHER],
                    timeout_seconds=8,
                    retry_count=0,
                ),
                cache=temporary,
                rate_limiter=temporary,
            ),
            semantic_version="openweather-onecall4-daily-v1",
        )
        managed_weather = (
            FallbackWeatherProvider(primary, managed_weather) if managed_weather else primary
        )
    managed_routes = ManagedRouteProvider(amap_routes, amap_runtime)
    managed_hours = ManagedHoursProvider(amap_hours, amap_runtime)
    managed_products = ManagedTravelProductProvider(flyai_products, flyai_runtime)
    closeables = tuple(
        provider
        for provider in (amap_places, amap_routes, amap_hours, weather, openweather)
        if provider is not None
    )
    return ProviderGatewayRuntime(
        closeables=closeables,
        places=managed_places,
        routes=managed_routes,
        hours=managed_hours,
        products=managed_products,
        weather=managed_weather,
    )
