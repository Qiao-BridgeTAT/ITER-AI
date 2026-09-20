"""Build the immutable whole-trip map projection from a strict schedule."""

from __future__ import annotations

from backend.contracts.daily_scheduling import DailyScheduleResult, DailySchedulingRequest
from backend.contracts.enums import ItineraryEntryKind, PlaceCategory, TransportMode
from backend.contracts.events import MapMarker, MapRouteSegment, MapUpdatePayload
from backend.providers.contracts import RouteMode


class MapProjectionError(ValueError):
    """Raised when a published schedule cannot be projected without inventing geography."""


class MapProjectionService:
    """Create one whole-trip GCJ-02 projection that clients can filter by schedule day."""

    def build(
        self,
        request: DailySchedulingRequest,
        schedule: DailyScheduleResult,
    ) -> MapUpdatePayload:
        if (
            request.trip_id != schedule.trip_id
            or request.input_state_version != schedule.input_state_version
            or request.request_id != schedule.request_id
        ):
            raise MapProjectionError("map projection requires the current schedule request")

        place_by_id = {item.place_id: item for item in request.places}
        marker_by_id: dict[object, MapMarker] = {}
        for day in schedule.days:
            for place_id in (
                day.start_place_id,
                *(activity.place_id for activity in day.activities),
                day.end_place_id,
            ):
                if place_id in marker_by_id:
                    continue
                place = place_by_id.get(place_id)
                if place is not None:
                    marker_by_id[place_id] = MapMarker(
                        place_id=place_id,
                        label=place.name,
                        kind=_entry_kind(place.category),
                        coordinates=place.coordinates,
                    )
                    continue
                hotel = request.hotel_result
                baseline = hotel.route_baseline
                selected = next(
                    (
                        candidate
                        for candidate in hotel.candidates
                        if candidate.hotel_place_id == place_id
                    ),
                    None,
                )
                if (
                    place_id == hotel.selected_hotel_place_id
                    and baseline is not None
                    and baseline.coordinates is not None
                    and selected is not None
                ):
                    marker_by_id[place_id] = MapMarker(
                        place_id=place_id,
                        label=selected.name,
                        kind=ItineraryEntryKind.HOTEL,
                        coordinates=baseline.coordinates,
                    )
                    continue
                raise MapProjectionError("every scheduled map place requires GCJ-02 coordinates")

        routes: list[MapRouteSegment] = []
        seen_routes: set[tuple[object, object, TransportMode]] = set()
        route_fact_by_key = {
            (route.origin_place_id, route.destination_place_id, _transport_mode(route.mode)): route
            for route in request.routes
            if route.availability.value != "missing" and route.polyline
        }
        for day in schedule.days:
            for leg in day.transport_legs:
                mode = _transport_mode(leg.mode)
                key = (leg.origin_place_id, leg.destination_place_id, mode)
                if key in seen_routes:
                    continue
                route_fact = route_fact_by_key.get(key)
                if route_fact is None:
                    continue
                seen_routes.add(key)
                routes.append(
                    MapRouteSegment(
                        from_place_id=leg.origin_place_id,
                        to_place_id=leg.destination_place_id,
                        mode=mode,
                        polyline=list(route_fact.polyline),
                    )
                )

        return MapUpdatePayload(
            selected_day_index=0,
            markers=list(marker_by_id.values()),
            routes=routes,
        )


def _entry_kind(category: PlaceCategory) -> ItineraryEntryKind:
    if category is PlaceCategory.ATTRACTION:
        return ItineraryEntryKind.ATTRACTION
    if category is PlaceCategory.RESTAURANT:
        return ItineraryEntryKind.RESTAURANT
    if category is PlaceCategory.HOTEL:
        return ItineraryEntryKind.HOTEL
    return ItineraryEntryKind.ACTIVITY


def _transport_mode(mode: RouteMode) -> TransportMode:
    return {
        RouteMode.WALKING: TransportMode.WALK,
        RouteMode.CYCLING: TransportMode.BICYCLE,
        RouteMode.TRANSIT: TransportMode.PUBLIC_TRANSIT,
        RouteMode.DRIVING: TransportMode.TAXI,
    }[mode]
