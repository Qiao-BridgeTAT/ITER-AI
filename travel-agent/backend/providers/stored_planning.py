"""Replay/fake Provider adapters backed by one normalized stable trip snapshot.

These adapters do not invent travel facts. They make the same formal planning graph
usable in replay and model-disabled runtimes by exposing facts that are already stored
in ``TripState`` through the ordinary Provider protocols.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from math import hypot
from uuid import UUID

from backend.contracts.common import CnyAmountRange
from backend.contracts.enums import ProviderCode, TransportMode
from backend.contracts.places import Coordinates, Gcj02Coordinates
from backend.contracts.provider_display import ProviderDisplayPlace
from backend.contracts.state import TripState
from backend.planning.city_registry import default_city_registry
from backend.planning.dining_geometry import point_in_polygon
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
    ProviderResultStatus,
    ProviderRoute,
    ProviderTicketOffer,
    RouteMode,
    RouteRequest,
    WeatherRequest,
)


def _now(clock: Callable[[], datetime]) -> datetime:
    value = clock()
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("stored Provider clock must return an aware datetime")
    return value.astimezone(UTC)


def _empty(item_type: type[object], clock: Callable[[], datetime]) -> ProviderResponse[object]:
    del item_type
    return ProviderResponse[object](
        provider=ProviderCode.MANUAL,
        status=ProviderResultStatus.EMPTY,
        fetched_at=_now(clock),
    )


def _route_mode(mode: TransportMode) -> RouteMode:
    return {
        TransportMode.WALK: RouteMode.WALKING,
        TransportMode.BICYCLE: RouteMode.CYCLING,
        TransportMode.PUBLIC_TRANSIT: RouteMode.TRANSIT,
        TransportMode.TAXI: RouteMode.DRIVING,
    }[mode]


class StoredPlaceProvider:
    def __init__(self, state: TripState, *, clock: Callable[[], datetime]) -> None:
        self._state = state
        self._clock = clock

    async def search_places(
        self, request: KeywordPlaceSearchRequest
    ) -> ProviderResponse[ProviderPlace]:
        query = request.query.casefold()
        values = [
            place
            for place in self._places()
            if (request.category_hint is None or place.category is request.category_hint)
            and (query in place.name.casefold() or _generic_query(query))
        ][: request.page_size]
        return self._response(values, f"stored:search:{request.city.city_id}:{request.query}")

    async def search_nearby(
        self, request: NearbyPlaceSearchRequest
    ) -> ProviderResponse[ProviderPlace]:
        query = request.query.casefold() if request.query else ""
        values = sorted(
            (
                place
                for place in self._places()
                if (request.category_hint is None or place.category is request.category_hint)
                and (not query or query in place.name.casefold() or _generic_query(query))
                and _distance_hint(place.coordinates, request.center) <= request.radius_m
            ),
            key=lambda place: _distance_hint(place.coordinates, request.center),
        )[: request.page_size]
        return self._response(values, f"stored:nearby:{request.city.city_id}")

    async def search_polygon(
        self, request: PolygonPlaceSearchRequest
    ) -> ProviderResponse[ProviderPlace]:
        values = [
            place
            for place in self._places()
            if place.category is request.category_hint
            and place.city_id == request.city.city_id
            and point_in_polygon(place.coordinates, request.polygon)
        ]
        start = (request.page - 1) * request.page_size
        return self._response(
            values[start : start + request.page_size],
            f"stored:polygon:{request.city.city_id}:{request.page}",
        )

    async def get_place(self, request: PlaceDetailRequest) -> ProviderResponse[ProviderPlace]:
        values = [
            place for place in self._places() if place.source_place_id == request.source_place_id
        ]
        return self._response(values, f"stored:detail:{request.source_place_id}")

    async def geocode(self, request: GeocodeRequest) -> ProviderResponse[GeocodedAddress]:
        del request
        return ProviderResponse[GeocodedAddress](
            provider=ProviderCode.MANUAL,
            status=ProviderResultStatus.EMPTY,
            fetched_at=_now(self._clock),
        )

    def _places(self) -> list[ProviderPlace]:
        city_reference = self._state.city_id or (
            self._state.city.value if self._state.city is not None else "unknown"
        )
        city_id = default_city_registry().resolve(city_reference).city_id
        values: dict[UUID, ProviderPlace] = {}
        for canonical in self._state.candidate_places:
            source = next(
                (item for item in canonical.source_mappings if item.provider is ProviderCode.AMAP),
                None,
            )
            values[canonical.place_id] = ProviderPlace(
                provider=ProviderCode.MANUAL,
                source_place_id=(
                    source.source_place_id if source else f"stored-{canonical.place_id}"
                ),
                city_id=city_id,
                name=canonical.name,
                category=canonical.category,
                address=canonical.address,
                coordinates=canonical.coordinates,
                fetched_at=_now(self._clock),
                raw_payload={"source": "stable_trip_state"},
            )
        for projected in self._state.provider_display.places:
            values.setdefault(
                projected.place_id,
                ProviderPlace(
                    provider=ProviderCode.MANUAL,
                    source_place_id=f"stored-{projected.place_id}",
                    city_id=city_id,
                    name=projected.name,
                    category=projected.category,
                    address=projected.address,
                    coordinates=projected.coordinates,
                    fetched_at=_now(self._clock),
                    raw_payload={"source": "provider_display"},
                ),
            )
        return list(values.values())

    def _response(
        self, values: list[ProviderPlace], request_id: str
    ) -> ProviderResponse[ProviderPlace]:
        return ProviderResponse[ProviderPlace](
            provider=ProviderCode.MANUAL,
            status=(ProviderResultStatus.SUCCESS if values else ProviderResultStatus.EMPTY),
            items=values,
            fetched_at=_now(self._clock),
            source_request_id=request_id,
        )


class StoredRouteProvider:
    def __init__(self, state: TripState, *, clock: Callable[[], datetime]) -> None:
        self._state = state
        self._clock = clock

    async def get_routes(self, request: RouteRequest) -> ProviderResponse[ProviderRoute]:
        coordinates = {
            place.place_id: place.coordinates for place in self._state.provider_display.places
        }
        origin = _matching_place(coordinates, request.origin)
        destination = _matching_place(coordinates, request.destination)
        values: list[ProviderRoute] = []
        if origin is not None and destination is not None:
            for index, route in enumerate(self._state.provider_display.routes):
                endpoints_match = {
                    route.from_place_id,
                    route.to_place_id,
                } == {origin, destination}
                mode = _route_mode(route.mode)
                if (
                    endpoints_match
                    and mode in request.modes
                    and route.distance_m is not None
                    and route.duration_minutes is not None
                ):
                    values.append(
                        ProviderRoute(
                            provider=ProviderCode.MANUAL,
                            mode=mode,
                            source_route_index=index,
                            distance_m=route.distance_m,
                            duration_seconds=route.duration_minutes * 60,
                            walking_distance_m=route.walking_m,
                            fare=route.fare,
                            polyline=route.polyline,
                            fetched_at=_now(self._clock),
                            raw_payload={"source": "provider_display", "route_id": route.route_id},
                        )
                    )
        return ProviderResponse[ProviderRoute](
            provider=ProviderCode.MANUAL,
            status=(ProviderResultStatus.SUCCESS if values else ProviderResultStatus.EMPTY),
            items=values,
            fetched_at=_now(self._clock),
            source_request_id="stored:routes",
        )


class StoredHoursProvider:
    def __init__(self, state: TripState, *, clock: Callable[[], datetime]) -> None:
        self._state = state
        self._clock = clock

    async def get_regular_hours(
        self, request: HoursRequest
    ) -> ProviderResponse[ProviderRegularHours]:
        fact = next(
            (
                item
                for item in self._state.provider_display.facts
                if item.kind == "regular_hours" and item.place_id == request.place_id
            ),
            None,
        )
        item = ProviderRegularHours(
            provider=ProviderCode.MANUAL,
            source_place_id=f"stored-{request.place_id}",
            display_text=fact.display_text if fact is not None else None,
            fetched_at=(fact.fetched_at if fact and fact.fetched_at else _now(self._clock)),
            missing_reason=(
                fact.missing_reason if fact is not None else "稳定状态没有营业时间资料。"
            ),
            raw_payload={"source": "provider_display"},
        )
        if item.display_text is not None:
            item = item.model_copy(update={"missing_reason": None})
        return ProviderResponse[ProviderRegularHours](
            provider=ProviderCode.MANUAL,
            status=ProviderResultStatus.SUCCESS,
            items=[item],
            fetched_at=_now(self._clock),
            source_request_id=f"stored:hours:{request.place_id}",
        )


class StoredProductProvider:
    def __init__(self, state: TripState, *, clock: Callable[[], datetime]) -> None:
        self._state = state
        self._clock = clock

    async def search_hotels(
        self, request: HotelSearchRequest
    ) -> ProviderResponse[ProviderHotelOffer]:
        selected = self._state.selected_hotel
        if selected is None:
            return ProviderResponse[ProviderHotelOffer](
                provider=ProviderCode.MANUAL,
                status=ProviderResultStatus.EMPTY,
                fetched_at=_now(self._clock),
            )
        place = _place(self._state, selected.place_id)
        if place is None:
            return ProviderResponse[ProviderHotelOffer](
                provider=ProviderCode.MANUAL,
                status=ProviderResultStatus.EMPTY,
                fetched_at=_now(self._clock),
            )
        fact = _price_fact(self._state, "hotel_price", selected.place_id)
        offer = ProviderHotelOffer(
            provider=ProviderCode.MANUAL,
            source_offer_id=f"stored-hotel-{selected.place_id}",
            source_hotel_id=f"stored-{selected.place_id}",
            name=place.name,
            address=place.address,
            raw_coordinates=Coordinates.model_validate(place.coordinates.model_dump()),
            room_price=fact,
            missing_fields=([] if fact is not None else ["room_price"]),
            fetched_at=_now(self._clock),
            raw_payload={"source": "stable_trip_state"},
        )
        return ProviderResponse[ProviderHotelOffer](
            provider=ProviderCode.MANUAL,
            status=ProviderResultStatus.SUCCESS,
            items=[offer],
            fetched_at=_now(self._clock),
            source_request_id=f"stored:hotel:{selected.place_id}",
        )

    async def search_place_products(
        self, request: ProductSearchRequest
    ) -> ProviderResponse[ProviderTicketOffer]:
        place = next(
            (
                item
                for item in self._state.provider_display.places
                if request.query is not None and item.name == request.query
            ),
            None,
        )
        if place is None:
            return ProviderResponse[ProviderTicketOffer](
                provider=ProviderCode.MANUAL,
                status=ProviderResultStatus.EMPTY,
                fetched_at=_now(self._clock),
            )
        price = _price_fact(self._state, "ticket_price", place.place_id)
        offer = ProviderTicketOffer(
            provider=ProviderCode.MANUAL,
            source_offer_id=f"stored-ticket-{place.place_id}",
            source_place_id=f"stored-{place.place_id}",
            name=place.name,
            price_date=request.visit_date,
            price=price,
            missing_fields=([] if price is not None else ["price"]),
            fetched_at=_now(self._clock),
            raw_payload={"source": "provider_display"},
        )
        return ProviderResponse[ProviderTicketOffer](
            provider=ProviderCode.MANUAL,
            status=ProviderResultStatus.SUCCESS,
            items=[offer],
            fetched_at=_now(self._clock),
            source_request_id=f"stored:ticket:{place.place_id}",
        )


class StoredWeatherProvider:
    def __init__(self, state: TripState, *, clock: Callable[[], datetime]) -> None:
        self._state = state
        self._clock = clock

    async def get_forecast(self, request: WeatherRequest) -> ProviderResponse[ProviderForecastDay]:
        values = [
            ProviderForecastDay(
                provider=ProviderCode.MANUAL,
                forecast_date=fact.forecast_date,
                condition_day=fact.display_text,
                condition_night=fact.display_text,
                low_celsius=(
                    int(fact.minimum_celsius) if fact.minimum_celsius is not None else None
                ),
                high_celsius=(
                    int(fact.maximum_celsius) if fact.maximum_celsius is not None else None
                ),
                fetched_at=fact.fetched_at or _now(self._clock),
                raw_payload={"source": "provider_display"},
            )
            for fact in self._state.provider_display.facts
            if fact.kind == "weather"
            and fact.forecast_date is not None
            and request.start_date <= fact.forecast_date <= request.end_date
            and (
                fact.display_text is not None
                or fact.minimum_celsius is not None
                or fact.maximum_celsius is not None
            )
        ]
        return ProviderResponse[ProviderForecastDay](
            provider=ProviderCode.MANUAL,
            status=(ProviderResultStatus.SUCCESS if values else ProviderResultStatus.EMPTY),
            items=values,
            fetched_at=_now(self._clock),
            source_request_id="stored:weather",
        )


def _place(state: TripState, place_id: UUID) -> ProviderDisplayPlace | None:
    return next((item for item in state.provider_display.places if item.place_id == place_id), None)


def _price_fact(state: TripState, kind: str, place_id: UUID) -> CnyAmountRange | None:
    fact = next(
        (
            item
            for item in state.provider_display.facts
            if item.kind == kind and item.place_id == place_id
        ),
        None,
    )
    return fact.amount if fact is not None else None


def _matching_place(
    coordinates: Mapping[UUID, Gcj02Coordinates], target: Gcj02Coordinates
) -> UUID | None:
    return next(
        (
            place_id
            for place_id, value in coordinates.items()
            if abs(value.latitude - target.latitude) < 0.000001
            and abs(value.longitude - target.longitude) < 0.000001
        ),
        None,
    )


def _distance_hint(left: Gcj02Coordinates, right: Gcj02Coordinates) -> int:
    return int(hypot(left.latitude - right.latitude, left.longitude - right.longitude) * 100_000)


def _generic_query(value: str) -> bool:
    return any(token in value for token in ("景点", "餐厅", "酒店", "博物馆", "历史"))
