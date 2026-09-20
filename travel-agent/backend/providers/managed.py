"""Resilient wrappers that keep Provider policies outside vendor adapters."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo

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
    ProviderError,
    ProviderFailureDetail,
    ProviderForecastDay,
    ProviderHotelOffer,
    ProviderPlace,
    ProviderRegularHours,
    ProviderResponse,
    ProviderResultStatus,
    ProviderRoute,
    ProviderTicketOffer,
    RouteMode,
    RouteRequest,
    WeatherRequest,
)
from backend.providers.hours_rules import PARSER_VERSION
from backend.providers.interfaces import (
    HoursProvider,
    PlaceProvider,
    RouteProvider,
    TravelProductProvider,
    WeatherProvider,
)
from backend.providers.resilience import (
    HOTEL_PRICE_CACHE_TTL_SECONDS,
    HOURS_CACHE_TTL_SECONDS,
    POI_CACHE_TTL_SECONDS,
    ROUTE_CACHE_TTL_SECONDS,
    TICKET_PRICE_CACHE_TTL_SECONDS,
    WEATHER_CACHE_TTL_SECONDS,
    ProviderRuntime,
)

_LOGGER = logging.getLogger(__name__)


def _parameters(request: Any) -> dict[str, Any]:
    return dict(request.model_dump(mode="json", exclude_none=False))


class ManagedPlaceProvider:
    def __init__(self, delegate: PlaceProvider, runtime: ProviderRuntime) -> None:
        self._delegate = delegate
        self._runtime = runtime

    async def search_places(
        self, request: KeywordPlaceSearchRequest
    ) -> ProviderResponse[ProviderPlace]:
        return await self._runtime.execute(
            operation="search_places",
            semantic_version="place-search-normalized-v2-business-facts",
            parameters=_parameters(request),
            cache_ttl_seconds=POI_CACHE_TTL_SECONDS,
            decode=ProviderResponse[ProviderPlace].model_validate,
            call=lambda: self._delegate.search_places(request),
        )

    async def search_nearby(
        self, request: NearbyPlaceSearchRequest
    ) -> ProviderResponse[ProviderPlace]:
        return await self._runtime.execute(
            operation="search_nearby",
            semantic_version="place-nearby-normalized-v2-business-facts",
            parameters=_parameters(request),
            cache_ttl_seconds=POI_CACHE_TTL_SECONDS,
            decode=ProviderResponse[ProviderPlace].model_validate,
            call=lambda: self._delegate.search_nearby(request),
        )

    async def search_polygon(
        self, request: PolygonPlaceSearchRequest
    ) -> ProviderResponse[ProviderPlace]:
        return await self._runtime.execute(
            operation="search_polygon",
            semantic_version="place-polygon-normalized-v1-business-navi",
            parameters=_parameters(request),
            cache_ttl_seconds=POI_CACHE_TTL_SECONDS,
            decode=ProviderResponse[ProviderPlace].model_validate,
            call=lambda: self._delegate.search_polygon(request),
        )

    async def get_place(self, request: PlaceDetailRequest) -> ProviderResponse[ProviderPlace]:
        return await self._runtime.execute(
            operation="get_place",
            semantic_version="place-detail-normalized-v2-business-facts",
            parameters=_parameters(request),
            cache_ttl_seconds=POI_CACHE_TTL_SECONDS,
            decode=ProviderResponse[ProviderPlace].model_validate,
            call=lambda: self._delegate.get_place(request),
        )

    async def geocode(self, request: GeocodeRequest) -> ProviderResponse[GeocodedAddress]:
        return await self._runtime.execute(
            operation="geocode",
            semantic_version="geocode-normalized-v1",
            parameters=_parameters(request),
            cache_ttl_seconds=POI_CACHE_TTL_SECONDS,
            decode=ProviderResponse[GeocodedAddress].model_validate,
            call=lambda: self._delegate.geocode(request),
        )


class ManagedHoursProvider:
    def __init__(self, delegate: HoursProvider, runtime: ProviderRuntime) -> None:
        self._delegate = delegate
        self._runtime = runtime

    async def get_regular_hours(
        self, request: HoursRequest
    ) -> ProviderResponse[ProviderRegularHours]:
        return await self._runtime.execute(
            operation="get_regular_hours",
            semantic_version=PARSER_VERSION,
            parameters={
                **_parameters(request),
                "observation_local_date": datetime.now(ZoneInfo("Asia/Shanghai"))
                .date()
                .isoformat(),
            },
            cache_ttl_seconds=HOURS_CACHE_TTL_SECONDS,
            decode=ProviderResponse[ProviderRegularHours].model_validate,
            call=lambda: self._delegate.get_regular_hours(request),
        )


class ManagedRouteProvider:
    def __init__(self, delegate: RouteProvider, runtime: ProviderRuntime) -> None:
        self._delegate = delegate
        self._runtime = runtime

    async def get_routes(self, request: RouteRequest) -> ProviderResponse[ProviderRoute]:
        # One route mode is one outbound request and one independent retry/timeout
        # budget. A successful walk must not hide a failed bus or driving request.
        async def query(mode: RouteMode) -> ProviderResponse[ProviderRoute] | ProviderError:
            single = request.model_copy(update={"modes": [mode]})
            try:
                return await self._runtime.execute(
                    operation=f"route_{mode.value}",
                    semantic_version="routes-normalized-v3-mode-recovery",
                    parameters=_parameters(single),
                    cache_ttl_seconds=ROUTE_CACHE_TTL_SECONDS,
                    decode=ProviderResponse[ProviderRoute].model_validate,
                    call=lambda: self._delegate.get_routes(single),
                )
            except ProviderError as error:
                _LOGGER.warning(
                    "route mode query failed after bounded recovery",
                    extra={
                        "provider": error.provider.value,
                        "route_mode": mode.value,
                        "failure_code": error.code.value,
                        "upstream_code": error.upstream_code,
                        "attempts": error.attempts,
                    },
                )
                return error

        outcomes = await asyncio.gather(*(query(mode) for mode in request.modes))
        items: list[ProviderRoute] = []
        missing: set[str] = set()
        failures: dict[str, ProviderFailureDetail] = {}
        responses: list[ProviderResponse[ProviderRoute]] = []
        first_error: ProviderError | None = None
        for mode, outcome in zip(request.modes, outcomes, strict=True):
            if isinstance(outcome, ProviderError):
                first_error = first_error or outcome
                failures[mode.value] = ProviderFailureDetail.from_error(outcome)
                missing.add(f"modes.{mode.value}")
                continue
            responses.append(outcome)
            items.extend(outcome.items)
            missing.update(outcome.missing_fields)
            failures.update(outcome.failures)
            if not outcome.items:
                missing.add(f"modes.{mode.value}")
        if not items and first_error is not None:
            first_error.failures = failures
            raise first_error
        return ProviderResponse[ProviderRoute](
            provider=responses[0].provider if responses else self._runtime.provider,
            status=(
                ProviderResultStatus.EMPTY
                if not items
                else ProviderResultStatus.PARTIAL
                if missing
                else ProviderResultStatus.SUCCESS
            ),
            items=items,
            missing_fields=sorted(missing) if items else [],
            failures=failures,
            fetched_at=max((value.fetched_at for value in responses), default=datetime.now(UTC)),
        )


class ManagedTravelProductProvider:
    def __init__(self, delegate: TravelProductProvider, runtime: ProviderRuntime) -> None:
        self._delegate = delegate
        self._runtime = runtime

    async def search_hotels(
        self, request: HotelSearchRequest
    ) -> ProviderResponse[ProviderHotelOffer]:
        return await self._runtime.execute(
            operation="search_hotels",
            semantic_version="hotel-offers-normalized-v2-structured-filters",
            parameters=_parameters(request),
            cache_ttl_seconds=HOTEL_PRICE_CACHE_TTL_SECONDS,
            decode=ProviderResponse[ProviderHotelOffer].model_validate,
            call=lambda: self._delegate.search_hotels(request),
        )

    async def search_place_products(
        self, request: ProductSearchRequest
    ) -> ProviderResponse[ProviderTicketOffer]:
        return await self._runtime.execute(
            operation="search_place_products",
            semantic_version="ticket-offers-normalized-v3-admission-status",
            parameters=_parameters(request),
            cache_ttl_seconds=TICKET_PRICE_CACHE_TTL_SECONDS,
            decode=ProviderResponse[ProviderTicketOffer].model_validate,
            call=lambda: self._delegate.search_place_products(request),
        )


class ManagedWeatherProvider:
    def __init__(
        self,
        delegate: WeatherProvider,
        runtime: ProviderRuntime,
        *,
        semantic_version: str = "weather-forecast-normalized-v2-source",
    ) -> None:
        self._delegate = delegate
        self._runtime = runtime
        self._semantic_version = semantic_version

    async def get_forecast(self, request: WeatherRequest) -> ProviderResponse[ProviderForecastDay]:
        return await self._runtime.execute(
            operation="get_forecast",
            semantic_version=self._semantic_version,
            parameters=_parameters(request),
            cache_ttl_seconds=WEATHER_CACHE_TTL_SECONDS,
            decode=ProviderResponse[ProviderForecastDay].model_validate,
            call=lambda: self._delegate.get_forecast(request),
        )
