"""Build the V4 formal-plan map from normalized Planner evidence only."""

from __future__ import annotations

from uuid import UUID

from backend.agent.planner.workspace import PlannerGuardError, server_id
from backend.contracts.daily_scheduling import ScheduleActivityKind
from backend.contracts.enums import ItineraryEntryKind, TransportMode
from backend.contracts.events import MapMarker, MapRouteSegment, MapUpdatePayload
from backend.contracts.places import Gcj02Coordinates
from backend.contracts.v4.planner_observations import SpatialRouteEdge, SpatialRouteEndpoint
from backend.contracts.v4.planner_publication import (
    scheduled_map_place_ids,
    verified_lodging_map_place_ids,
)
from backend.contracts.v4.planner_workspace import PlannerWorkspaceState
from backend.contracts.v4.task_book import TaskBookV4
from backend.providers.contracts import RouteMode


def build_planner_map_projection(
    workspace: PlannerWorkspaceState,
    book: TaskBookV4,
) -> MapUpdatePayload:
    """Project exact scheduled places; never synthesize coordinates or route geometry."""

    schedule = workspace.materialized_schedule
    if schedule is None or not schedule.days:
        raise PlannerGuardError("planner_map_schedule_missing")

    marker_facts: dict[UUID, tuple[str, Gcj02Coordinates, ItineraryEntryKind]] = {}
    for place_fact in workspace.place_evidence:
        place_id = _canonical_place_id(place_fact.canonical_entity_id)
        marker_facts[place_id] = (
            place_fact.display_name,
            place_fact.coordinates,
            _candidate_kind(place_fact.entity_kind.value),
        )

    hotel_names = {
        offer.offer_ref.property_id: offer.property_name
        for offer in (workspace.hotel_observation.offers if workspace.hotel_observation else ())
    }
    fixed = book.lodging_direction.existing_booking
    for hotel_fact in workspace.hotel_location_evidence:
        if fixed is not None and hotel_fact.property_id == fixed.booking_id:
            marker_facts[UUID(server_id("fixed-hotel", fixed.booking_id))] = (
                fixed.user_description,
                hotel_fact.coordinates,
                ItineraryEntryKind.HOTEL,
            )
        if hotel_fact.property_id in hotel_names:
            marker_facts[UUID(server_id("hotel-property", hotel_fact.property_id))] = (
                hotel_names[hotel_fact.property_id],
                hotel_fact.coordinates,
                ItineraryEntryKind.HOTEL,
            )

    draft = workspace.working_itinerary
    if draft is None:
        raise PlannerGuardError("planner_map_working_itinerary_missing")
    # Some Schedule UUIDs represent no-place rest boundaries or fixed timeline
    # events. The publication contract owns the same projection rule, so map
    # generation omits only those IDs without weakening real place checks.
    scheduled_place_ids = scheduled_map_place_ids(
        draft,
        schedule,
        verified_lodging_place_ids=verified_lodging_map_place_ids(
            draft,
            workspace.hotel_location_evidence,
        ),
    )
    activity_kinds = {
        activity.place_id: _activity_kind(activity.kind)
        for day in schedule.days
        for activity in day.activities
    }
    markers = []
    for place_id in sorted(scheduled_place_ids, key=str):
        marker_fact = marker_facts.get(place_id)
        if marker_fact is None:
            raise PlannerGuardError(f"planner_map_place_fact_missing:{place_id}")
        label, coordinates, default_kind = marker_fact
        markers.append(
            MapMarker(
                place_id=place_id,
                label=label,
                kind=activity_kinds.get(place_id, default_kind),
                coordinates=coordinates,
            )
        )

    candidate_places = {
        item.candidate_ref.candidate_id: _canonical_place_id(item.candidate_ref.canonical_entity_id)
        for item in workspace.candidate_pool.candidates
    }
    hotel_places = {
        offer.offer_ref.offer_id: UUID(server_id("hotel-property", offer.offer_ref.property_id))
        for offer in (workspace.hotel_observation.offers if workspace.hotel_observation else ())
    }
    edges = {
        edge.route_edge_id: edge
        for edge in (
            *(workspace.spatial_observation.route_edges if workspace.spatial_observation else ()),
            *workspace.route_evidence,
        )
    }
    route_by_key: dict[tuple[UUID, UUID, TransportMode], MapRouteSegment] = {}
    for edge in edges.values():
        if edge.status == "missing" or len(edge.polyline) < 2:
            continue
        origin = _endpoint_place_id(edge.origin, candidate_places, hotel_places, workspace)
        destination = _endpoint_place_id(
            edge.destination,
            candidate_places,
            hotel_places,
            workspace,
        )
        if origin is None or destination is None:
            continue
        mode = _edge_transport_mode(edge)
        key = (origin, destination, mode)
        if origin in scheduled_place_ids and destination in scheduled_place_ids:
            route_by_key[key] = MapRouteSegment(
                from_place_id=origin,
                to_place_id=destination,
                mode=mode,
                polyline=list(edge.polyline),
            )

    scheduled_route_keys = {
        (leg.origin_place_id, leg.destination_place_id, _schedule_transport_mode(leg.mode))
        for day in schedule.days
        for leg in day.transport_legs
    }
    routes = [
        route_by_key[key] for key in sorted(scheduled_route_keys, key=str) if key in route_by_key
    ]
    return MapUpdatePayload(selected_day_index=0, markers=markers, routes=routes)


def _canonical_place_id(value: str) -> UUID:
    try:
        return UUID(value)
    except ValueError:
        return UUID(server_id("candidate-place", value))


def _endpoint_place_id(
    endpoint: SpatialRouteEndpoint,
    candidate_places: dict[str, UUID],
    hotel_places: dict[str, UUID],
    workspace: PlannerWorkspaceState,
) -> UUID | None:
    if endpoint.kind == "candidate":
        return candidate_places.get(endpoint.reference_id)
    if endpoint.kind == "hotel_offer":
        return hotel_places.get(endpoint.reference_id)
    if endpoint.kind == "fixed_commitment":
        baseline = (
            workspace.working_itinerary.lodging_baseline if workspace.working_itinerary else None
        )
        if (
            baseline is not None
            and baseline.fixed_commitment_ref is not None
            and baseline.fixed_commitment_ref.commitment_id == endpoint.reference_id
        ):
            return UUID(server_id("fixed-hotel", endpoint.reference_id))
        return UUID(server_id("fixed-place", endpoint.reference_id))
    return None


def _candidate_kind(value: str) -> ItineraryEntryKind:
    if value == "attraction":
        return ItineraryEntryKind.ATTRACTION
    if value == "restaurant":
        return ItineraryEntryKind.RESTAURANT
    return ItineraryEntryKind.ACTIVITY


def _activity_kind(value: ScheduleActivityKind) -> ItineraryEntryKind:
    if value is ScheduleActivityKind.ATTRACTION:
        return ItineraryEntryKind.ATTRACTION
    if value is ScheduleActivityKind.RESTAURANT:
        return ItineraryEntryKind.RESTAURANT
    return ItineraryEntryKind.ACTIVITY


def _edge_transport_mode(edge: SpatialRouteEdge) -> TransportMode:
    return {
        "walking": TransportMode.WALK,
        "public_transit": TransportMode.PUBLIC_TRANSIT,
        "taxi": TransportMode.TAXI,
        "driving": TransportMode.TAXI,
    }[edge.transport_mode]


def _schedule_transport_mode(value: RouteMode) -> TransportMode:
    return {
        RouteMode.WALKING: TransportMode.WALK,
        RouteMode.CYCLING: TransportMode.BICYCLE,
        RouteMode.TRANSIT: TransportMode.PUBLIC_TRANSIT,
        RouteMode.DRIVING: TransportMode.TAXI,
    }[value]
