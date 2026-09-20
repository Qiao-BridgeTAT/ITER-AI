"""Dependency-inversion protocols for all stage-0 external travel data."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from backend.providers.contracts import (
    GeocodedAddress,
    GeocodeRequest,
    HotelSearchRequest,
    HoursRequest,
    KeywordPlaceSearchRequest,
    NearbyPlaceSearchRequest,
    PlaceDetailRequest,
    PolygonPlaceSearchRequest,
    ProductSearchRequest,
    ProviderForecastDay,
    ProviderHotelOffer,
    ProviderPlace,
    ProviderRegularHours,
    ProviderResponse,
    ProviderRoute,
    ProviderTicketOffer,
    RouteRequest,
    WeatherRequest,
)


class PlaceProvider(Protocol):
    async def search_places(
        self, request: KeywordPlaceSearchRequest
    ) -> ProviderResponse[ProviderPlace]: ...

    async def search_nearby(
        self, request: NearbyPlaceSearchRequest
    ) -> ProviderResponse[ProviderPlace]: ...

    async def search_polygon(
        self, request: PolygonPlaceSearchRequest
    ) -> ProviderResponse[ProviderPlace]: ...

    async def get_place(self, request: PlaceDetailRequest) -> ProviderResponse[ProviderPlace]: ...

    async def geocode(self, request: GeocodeRequest) -> ProviderResponse[GeocodedAddress]: ...


class HoursProvider(Protocol):
    async def get_regular_hours(
        self, request: HoursRequest
    ) -> ProviderResponse[ProviderRegularHours]: ...


class RouteProvider(Protocol):
    async def get_routes(self, request: RouteRequest) -> ProviderResponse[ProviderRoute]: ...


class TravelProductProvider(Protocol):
    async def search_hotels(
        self, request: HotelSearchRequest
    ) -> ProviderResponse[ProviderHotelOffer]: ...

    async def search_place_products(
        self, request: ProductSearchRequest
    ) -> ProviderResponse[ProviderTicketOffer]: ...


class WeatherProvider(Protocol):
    async def get_forecast(
        self, request: WeatherRequest
    ) -> ProviderResponse[ProviderForecastDay]: ...


class ProviderCache(Protocol):
    """Cache boundary implemented by RedisTemporaryStore in live runtime."""

    async def get_cache(
        self,
        provider: str,
        semantic_version: str,
        parameters: Mapping[str, Any],
    ) -> Any | None: ...

    async def put_cache(
        self,
        provider: str,
        semantic_version: str,
        parameters: Mapping[str, Any],
        value: Any,
        ttl_seconds: int,
    ) -> str: ...
