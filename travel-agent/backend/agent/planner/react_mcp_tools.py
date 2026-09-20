"""Expose discovered MCP definitions directly; ingest verified POIs after calls."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from math import ceil
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from backend.agent.model_gateway import ModelToolDefinition
from backend.agent.planner.dining_context import dining_place_blocked, mark_dining_admitted
from backend.agent.planner.evidence import _place_evidence
from backend.agent.planner.observation_router import EvidenceUpdate
from backend.agent.planner.react_runtime import AgentSession, AgentTool, ToolOutcome
from backend.agent.planner.workspace import PlannerGuardError, server_id
from backend.contracts.enums import PlaceCategory, ProviderCode
from backend.contracts.v4.enums import CandidateEntityKind, PlannerCapability
from backend.contracts.v4.planner_evidence import PlannerCapabilityObservation
from backend.contracts.v4.planner_observations import SpatialRouteEdge, SpatialRouteEndpoint
from backend.contracts.v4.planner_workspace import PlannerWorkspaceState, VerifiedFactSummary
from backend.providers.amap_mcp import SEARCH_TOOLS, AmapMcpRouter
from backend.providers.amap_mcp_adapters import AmapMcpPlaceProvider, AmapMcpRouteProvider
from backend.providers.contracts import RouteMode

if TYPE_CHECKING:
    from backend.agent.planner.react_tools import PlannerToolBindings


def compact_data(value: Any) -> Any:
    """Remove display geometry/photos, not decision facts or missing statuses."""
    if isinstance(value, dict):
        return {
            k: compact_data(v)
            for k, v in value.items()
            if k not in {"polyline", "photos", "raw_payload", "photo_url", "steps"}
        }
    if isinstance(value, list):
        return [compact_data(v) for v in value[:20]]
    return value


async def discovered_mcp_tools(
    mcp: AmapMcpRouter, bindings: PlannerToolBindings
) -> tuple[AgentTool, ...]:
    definitions = await mcp.tools()
    adapter = AmapMcpPlaceProvider(mcp)

    async def call(name: str, args: dict[str, Any], session: AgentSession) -> ToolOutcome:
        # Reject invalid coordinate order before an external request. Keep the
        # model's choice; never swap or round coordinates on its behalf.
        for field in ("location", "origin", "destination"):
            if field not in args:
                continue
            try:
                longitude, latitude = map(float, args[field].split(","))
                if not (-180 <= longitude <= 180 and -90 <= latitude <= 90):
                    raise ValueError
            except (ValueError, TypeError, AttributeError):
                raise PlannerGuardError(
                    f"mcp_coordinate_invalid:parameter={field}:"
                    "expected=longitude,latitude:use=candidates.mcp_location"
                ) from None
        for receipt in reversed(session.state.receipts):
            if (
                receipt.status == "completed"
                and receipt.result
                and receipt.call.function.name == name
                and json.loads(receipt.call.function.arguments) == args
            ):
                previous = json.loads(receipt.result)
                observed = datetime.fromisoformat(previous["observed_at"])
                if 0 <= (datetime.now(UTC) - observed).total_seconds() < 600:
                    return ToolOutcome({**previous, "cached": True})
        raw = await mcp.call(name, args)
        result = {
            "tool": name,
            "arguments": args,
            "source": mcp.source(name),
            "observed_at": datetime.now(UTC).isoformat(),
            "data": compact_data(raw),
        }
        if name in {
            "maps_direction_walking",
            "maps_direction_driving",
            "maps_direction_transit_integrated",
            "maps_direction_bicycling",
        }:
            return _route_outcome(mcp, bindings, name, args, raw, result, session)
        if name not in SEARCH_TOOLS:
            return ToolOutcome(result)
        city = bindings.evidence.registry.provider_scope(
            session.context.book.destination_and_dates.destination_name, ProviderCode.AMAP
        )
        response = await adapter._normalize(
            raw, city.city_id, None, hydrate=name != "maps_search_detail"
        )
        places = []
        for place in response.items:
            if place.category not in {PlaceCategory.ATTRACTION, PlaceCategory.RESTAURANT}:
                continue
            kind = CandidateEntityKind(place.category.value)
            evidence = _place_evidence(place, kind)
            if kind is CandidateEntityKind.RESTAURANT and dining_place_blocked(
                evidence, session.context.book
            ):
                continue
            places.append(evidence)
        verified = tuple({p.canonical_entity_id: p for p in places}.values())
        # Hotel POIs are facts for matching branches, never priced hotel products.
        result.update(
            data={"places": [compact_data(p.model_dump(mode="json")) for p in response.items]},
            admitted_places=[p.model_dump(mode="json") for p in verified],
            missing_fields=response.missing_fields,
            status=response.status.value,
        )

        async def apply(w: PlannerWorkspaceState) -> PlannerWorkspaceState:
            if not verified:
                return w
            w = mark_dining_admitted(
                w,
                (
                    p.canonical_entity_id
                    for p in verified
                    if p.entity_kind is CandidateEntityKind.RESTAURANT
                ),
            )
            update = EvidenceUpdate(
                places=verified,
                observation=PlannerCapabilityObservation(
                    observation_id=str(uuid4()),
                    scope=w.current_scope,
                    request_id=str(uuid4()),
                    capability=PlannerCapability.PLACE_FACTS,
                    status="complete" if response.status.value == "success" else "partial",
                    fact_reference_ids=tuple(p.fact_reference_id for p in verified),
                    reason_summary=f"{mcp.source(name)} {name}：实体、城市和类型已核验。",
                    observed_at=response.fetched_at,
                ),
            )
            return await bindings.merge(w, update, session)

        return ToolOutcome(result, apply)

    tools = []
    for name, definition in definitions.items():

        async def handler(
            args: dict[str, Any], session: AgentSession, name: str = name
        ) -> ToolOutcome:
            return await call(name, args, session)

        tools.append(
            AgentTool(
                ModelToolDefinition(
                    name=name,
                    description=definition.get("description", name),
                    parameters=definition["inputSchema"],
                ),
                handler,
                read_only=True,
                reviewer_allowed=True,
            )
        )
    return tuple(tools)


def _route_outcome(
    mcp: AmapMcpRouter,
    bindings: PlannerToolBindings,
    name: str,
    args: dict[str, Any],
    raw: dict[str, Any],
    result: dict[str, Any],
    session: AgentSession,
) -> ToolOutcome:
    mode = {
        "maps_direction_walking": RouteMode.WALKING,
        "maps_direction_driving": RouteMode.DRIVING,
        "maps_direction_transit_integrated": RouteMode.TRANSIT,
        "maps_direction_bicycling": RouteMode.CYCLING,
    }[name]
    response = AmapMcpRouteProvider(mcp).normalize_result(raw, mode)
    result["normalized_routes"] = [
        compact_data(route.model_dump(mode="json")) for route in response.items
    ]
    result["status"] = response.status.value
    endpoints = []
    workspace = session.workspace
    places = {place.canonical_entity_id: place for place in workspace.place_evidence}
    for entry in workspace.candidate_pool.candidates:
        place = places.get(entry.candidate_ref.canonical_entity_id)
        if place is not None:
            endpoints.append(
                (
                    SpatialRouteEndpoint(
                        kind="candidate", reference_id=entry.candidate_ref.candidate_id
                    ),
                    place.coordinates,
                )
            )
    hotels = {place.property_id: place for place in workspace.hotel_location_evidence}
    if workspace.hotel_observation:
        for offer in workspace.hotel_observation.offers:
            hotel_place = hotels.get(offer.offer_ref.property_id)
            if hotel_place is not None:
                endpoints.append(
                    (
                        SpatialRouteEndpoint(
                            kind="hotel_offer", reference_id=offer.offer_ref.offer_id
                        ),
                        hotel_place.coordinates,
                    )
                )
        fixed = workspace.hotel_observation.fixed_booking
        if fixed and fixed.property_id in hotels:
            endpoints.append(
                (
                    SpatialRouteEndpoint(
                        kind="fixed_commitment", reference_id=fixed.commitment_ref.commitment_id
                    ),
                    hotels[fixed.property_id].coordinates,
                )
            )

    def bind(value: str) -> SpatialRouteEndpoint | None:
        try:
            longitude, latitude = map(float, value.split(","))
        except (ValueError, AttributeError):
            return None
        matches = {
            (endpoint.kind, endpoint.reference_id): endpoint
            for endpoint, location in endpoints
            if abs(location.longitude - longitude) <= 0.00001
            and abs(location.latitude - latitude) <= 0.00001
        }
        return next(iter(matches.values())) if len(matches) == 1 else None

    origin, destination = bind(args.get("origin", "")), bind(args.get("destination", ""))
    if origin is None or destination is None or mode is RouteMode.CYCLING:
        result["itinerary_binding"] = "unbound"
        return ToolOutcome(result)
    result["itinerary_binding"] = "verified_endpoints"
    route = min(response.items, key=lambda item: item.duration_seconds) if response.items else None
    edges, facts = [], []
    for transport in {
        RouteMode.WALKING: ("walking",),
        RouteMode.DRIVING: ("driving", "taxi"),
        RouteMode.TRANSIT: ("public_transit",),
    }[mode]:
        edge_id = server_id(
            workspace.generation_id,
            origin.kind,
            origin.reference_id,
            destination.kind,
            destination.reference_id,
            transport,
        )
        fact_id = server_id(edge_id, response.fetched_at.isoformat())
        if route is not None:
            facts.append(
                VerifiedFactSummary(
                    fact_reference_id=fact_id,
                    fact_kind="route_cost",
                    safe_summary=(
                        f"{name}：已核实端点间的当前路线，"
                        f"预计{ceil(route.duration_seconds / 60)}分钟。"
                    ),
                    observed_at=response.fetched_at,
                    expires_at=response.fetched_at + timedelta(hours=2),
                    source_reference_ids=(f"provider:amap:route:{edge_id}",),
                )
            )
        edges.append(
            SpatialRouteEdge.model_validate(
                {
                    "route_edge_id": edge_id,
                    "origin": origin,
                    "destination": destination,
                    "transport_mode": transport,
                    "status": "available" if route else "missing",
                    "duration_minutes": ceil(route.duration_seconds / 60) if route else None,
                    "distance_meters": route.distance_m if route else None,
                    "transfer_count": route.transfer_count if route else None,
                    "fare": route.fare if route else None,
                    "polyline": tuple(route.polyline) if route else (),
                    "fact_reference_ids": (fact_id,) if route else (),
                    "missing_reason": None
                    if route
                    else "MCP 查询完成但未返回该方式路线；不能认定没有路线。",
                }
            )
        )

    async def apply(current: PlannerWorkspaceState) -> PlannerWorkspaceState:
        return await bindings.merge(
            current,
            EvidenceUpdate(
                route_edges=tuple(edges),
                facts=tuple(facts),
                observation=PlannerCapabilityObservation(
                    observation_id=str(uuid4()),
                    scope=current.current_scope,
                    request_id=str(uuid4()),
                    capability=PlannerCapability.SPATIAL_ROUTES,
                    status="complete" if route else "unavailable",
                    fact_reference_ids=tuple(f.fact_reference_id for f in facts),
                    reason_summary="Agent 自主请求的 MCP 路线已按核实端点归入当前证据。",
                    observed_at=response.fetched_at,
                ),
            ),
            session,
        )

    return ToolOutcome(result, apply)
