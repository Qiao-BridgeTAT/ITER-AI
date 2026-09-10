"""Adapters and vendor-neutral boundaries for external travel data."""

from backend.providers.contracts import ProviderError, ProviderFailureCode
from backend.providers.interfaces import (
    HoursProvider,
    PlaceProvider,
    ProviderCache,
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
from backend.providers.resilience import (
    DEFAULT_PROVIDER_POLICIES,
    ProviderRuntime,
    ProviderRuntimePolicy,
)

__all__ = [
    "HoursProvider",
    "ManagedHoursProvider",
    "ManagedPlaceProvider",
    "ManagedRouteProvider",
    "ManagedTravelProductProvider",
    "ManagedWeatherProvider",
    "PlaceProvider",
    "ProviderCache",
    "ProviderError",
    "ProviderFailureCode",
    "ProviderRuntime",
    "ProviderRuntimePolicy",
    "RouteProvider",
    "TravelProductProvider",
    "WeatherProvider",
    "DEFAULT_PROVIDER_POLICIES",
]
