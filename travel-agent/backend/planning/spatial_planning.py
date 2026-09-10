"""Deterministic V3-34 route-aware spatial skeleton construction."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from itertools import combinations
from math import ceil
from typing import TypeVar
from uuid import NAMESPACE_URL, UUID, uuid5

from backend.agent.model_gateway import ModelCancellation
from backend.contracts.city_registry import CityProviderCapability
from backend.contracts.enums import CoordinateSystem, DataAvailability, ProviderCode
from backend.contracts.places import Gcj02Coordinates
from backend.contracts.spatial_planning import (
    ClusterCostSummary,
    InterClusterCost,
    SpatialActivityCluster,
    SpatialAnchor,
    SpatialAnchorStrength,
    SpatialIssueCode,
    SpatialPlanningIssue,
    SpatialPlanningRequest,
    SpatialPlanningResult,
    SpatialRouteEdge,
    SpatialRouteOption,
)
from backend.planning.city_registry import CityProviderUnavailableError, CityRegistry
from backend.providers.contracts import (
    ProviderCityScope,
    ProviderError,
    ProviderResponse,
    ProviderResultStatus,
    ProviderRoute,
    RouteMode,
    RouteRequest,
)
from backend.providers.interfaces import RouteProvider

SPATIAL_ALGORITHM_VERSION = "1.0.0"
SpatialMember = TypeVar("SpatialMember")


def group_by_route_costs(
    members: tuple[SpatialMember, ...],
    *,
    available_dates: Callable[[SpatialMember], set[object]],
    duration_minutes: Callable[[SpatialMember, SpatialMember], int | None],
    threshold_minutes: int,
) -> tuple[tuple[SpatialMember, ...], ...]:
    """Shared complete-link clustering; unknown routes never imply proximity.

    No day-count argument or itinerary output: this produces evidence, not a plan.
    """

    groups: list[list[SpatialMember]] = []
    for member in members:
        choices: list[tuple[int, int]] = []
        for index, group in enumerate(groups):
            shared_dates = available_dates(member)
            durations = []
            for other in group:
                shared_dates &= available_dates(other)
                duration = duration_minutes(member, other)
                if duration is None or duration > threshold_minutes:
                    break
                durations.append(duration)
            else:
                if shared_dates:
                    choices.append((max(durations, default=0), index))
        if choices:
            groups[min(choices)[1]].append(member)
        else:
            groups.append([member])
    return tuple(tuple(group) for group in groups)


class SpatialPlanningService:
    """Build anchors, pairwise route costs, clusters and explicit outlier issues."""

    def __init__(
        self,
        *,
        registry: CityRegistry,
        routes: RouteProvider,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._registry = registry
        self._routes = routes
        self._clock = clock

    async def build(
        self,
        request: SpatialPlanningRequest,
        *,
        cancellation: ModelCancellation | None = None,
    ) -> SpatialPlanningResult:
        request = SpatialPlanningRequest.model_validate(request.model_dump(mode="json"))
        if not self._registry.supports(request.city_id, CityProviderCapability.AMAP_ROUTES):
            raise CityProviderUnavailableError(
                f"city {request.city_id} has no registered route capability"
            )
        city_scope = self._registry.provider_scope(request.city_id, ProviderCode.AMAP)
        places_by_id = {place.place_id: place for place in request.places}
        anchors = tuple(
            SpatialAnchor(
                node_id=node.node_id,
                place_id=node.place_id,
                candidate_id=node.candidate_id,
                role=node.role,
                strength=node.strength,
                name=places_by_id[node.place_id].name,
                coordinates=_coordinates(places_by_id[node.place_id].coordinates),
                available_dates=node.available_dates,
                fixed_time_window=node.fixed_time_window,
                source_reference_ids=node.source_reference_ids,
            )
            for node in request.nodes
        )
        ordered_anchors = tuple(sorted(anchors, key=_anchor_sort_key))
        edges: list[SpatialRouteEdge] = []
        for origin, destination in combinations(ordered_anchors, 2):
            if cancellation is not None:
                cancellation.raise_if_cancelled("spatial_route_matrix")
            edges.append(
                await self._route_edge(
                    request,
                    city_scope=city_scope,
                    origin=origin,
                    destination=destination,
                )
            )
        route_edges = tuple(edges)
        edge_lookup = _edge_lookup(route_edges)
        member_groups = _cluster_members(
            ordered_anchors,
            edge_lookup,
            threshold_minutes=request.cluster_threshold_minutes,
        )
        clusters = tuple(
            _project_cluster(group, edge_lookup, index=index)
            for index, group in enumerate(member_groups, start=1)
        )
        inter_cluster_costs = tuple(
            _inter_cluster_cost(left, right, edge_lookup)
            for left, right in combinations(clusters, 2)
        )
        issues = _outlier_issues(
            ordered_anchors,
            edge_lookup,
            threshold_minutes=request.outlier_threshold_minutes,
        )
        degradation_reasons: list[str] = []
        if any(edge.status is not DataAvailability.AVAILABLE for edge in route_edges):
            degradation_reasons.append("one or more route costs are partial or unknown")
        if issues:
            degradation_reasons.append("one or more strong anchors are spatial outliers")
        generated_at = self._clock()
        if generated_at.tzinfo is None or generated_at.utcoffset() is None:
            raise ValueError("spatial planning clock must return an aware datetime")
        return SpatialPlanningResult(
            algorithm_version=SPATIAL_ALGORITHM_VERSION,
            request_id=request.request_id,
            trip_id=request.trip_id,
            input_state_version=request.input_state_version,
            task_book_id=request.task_book_id,
            task_book_revision=request.task_book_revision,
            city_id=request.city_id,
            status=(
                DataAvailability.PARTIAL if degradation_reasons else DataAvailability.AVAILABLE
            ),
            anchors=ordered_anchors,
            route_edges=route_edges,
            clusters=clusters,
            inter_cluster_costs=inter_cluster_costs,
            issues=issues,
            degradation_reasons=tuple(degradation_reasons),
            generated_at=generated_at.astimezone(UTC),
        )

    async def _route_edge(
        self,
        request: SpatialPlanningRequest,
        *,
        city_scope: ProviderCityScope,
        origin: SpatialAnchor,
        destination: SpatialAnchor,
    ) -> SpatialRouteEdge:
        edge_id = _edge_id(origin.node_id, destination.node_id)
        supported_modes = tuple(
            mode
            for mode in request.route_modes
            if mode is not RouteMode.TRANSIT or city_scope.provider_transit_city_code is not None
        )
        missing_fields = {
            "modes.transit"
            for mode in request.route_modes
            if mode is RouteMode.TRANSIT and city_scope.provider_transit_city_code is None
        }
        if not supported_modes:
            return _missing_edge(
                edge_id,
                origin.node_id,
                destination.node_id,
                missing_fields or {"routes"},
                "the registered city has no usable route mode for this pair",
            )
        try:
            response = await self._routes.get_routes(
                RouteRequest(
                    city=city_scope,
                    origin=origin.coordinates,
                    destination=destination.coordinates,
                    modes=list(supported_modes),
                )
            )
        except ProviderError as error:
            return _missing_edge(
                edge_id,
                origin.node_id,
                destination.node_id,
                {*(f"modes.{mode.value}" for mode in supported_modes), *missing_fields},
                f"{error.provider.value} route {error.code.value}",
            )
        return _edge_from_response(
            edge_id,
            origin.node_id,
            destination.node_id,
            request.route_modes,
            response,
            missing_fields,
        )


def _edge_from_response(
    edge_id: UUID,
    origin_node_id: UUID,
    destination_node_id: UUID,
    requested_modes: tuple[RouteMode, ...],
    response: ProviderResponse[ProviderRoute],
    initial_missing_fields: set[str],
) -> SpatialRouteEdge:
    best_by_mode: dict[RouteMode, ProviderRoute] = {}
    for route in response.items:
        current = best_by_mode.get(route.mode)
        if current is None or _route_sort_key(route) < _route_sort_key(current):
            best_by_mode[route.mode] = route
    routes = tuple(
        SpatialRouteOption(
            provider=route.provider,
            mode=route.mode,
            distance_m=route.distance_m,
            duration_seconds=route.duration_seconds,
            walking_distance_m=route.walking_distance_m,
            transfer_count=route.transfer_count,
            source_route_index=route.source_route_index,
            polyline=tuple(route.polyline),
            fetched_at=route.fetched_at,
        )
        for route in sorted(best_by_mode.values(), key=lambda item: item.mode.value)
    )
    missing_fields = set(initial_missing_fields) | set(response.missing_fields)
    missing_fields.update(
        f"modes.{mode.value}" for mode in requested_modes if mode not in best_by_mode
    )
    if not routes:
        return _missing_edge(
            edge_id,
            origin_node_id,
            destination_node_id,
            missing_fields or {"routes"},
            response.provider_notice or "route provider returned no usable route",
        )
    if response.status is ProviderResultStatus.PARTIAL or missing_fields:
        return SpatialRouteEdge(
            edge_id=edge_id,
            origin_node_id=origin_node_id,
            destination_node_id=destination_node_id,
            status=DataAvailability.PARTIAL,
            routes=routes,
            missing_fields=tuple(sorted(missing_fields)),
            missing_reason=(
                response.provider_notice or "one or more requested route modes are unavailable"
            ),
        )
    return SpatialRouteEdge(
        edge_id=edge_id,
        origin_node_id=origin_node_id,
        destination_node_id=destination_node_id,
        status=DataAvailability.AVAILABLE,
        routes=routes,
    )


def _missing_edge(
    edge_id: UUID,
    origin_node_id: UUID,
    destination_node_id: UUID,
    missing_fields: set[str],
    reason: str,
) -> SpatialRouteEdge:
    return SpatialRouteEdge(
        edge_id=edge_id,
        origin_node_id=origin_node_id,
        destination_node_id=destination_node_id,
        status=DataAvailability.MISSING,
        missing_fields=tuple(sorted(missing_fields)),
        missing_reason=reason,
    )


def _cluster_members(
    anchors: tuple[SpatialAnchor, ...],
    edges: dict[frozenset[UUID], SpatialRouteEdge],
    *,
    threshold_minutes: int,
) -> tuple[tuple[SpatialAnchor, ...], ...]:
    groups = group_by_route_costs(
        anchors,
        available_dates=lambda anchor: set(anchor.available_dates),
        duration_minutes=lambda left, right: _effective_minutes(
            edges[frozenset((left.node_id, right.node_id))]
        ),
        threshold_minutes=threshold_minutes,
    )
    normalized = [tuple(sorted(group, key=_anchor_sort_key)) for group in groups]
    return tuple(sorted(normalized, key=lambda group: _anchor_sort_key(group[0])))


def _project_cluster(
    members: tuple[SpatialAnchor, ...],
    edges: dict[frozenset[UUID], SpatialRouteEdge],
    *,
    index: int,
) -> SpatialActivityCluster:
    member_ids = tuple(member.node_id for member in members)
    shared_dates = set(members[0].available_dates)
    for member in members[1:]:
        shared_dates &= set(member.available_dates)
    weights = [2 if member.strength is SpatialAnchorStrength.STRONG else 1 for member in members]
    weight_total = sum(weights)
    center = Gcj02Coordinates(
        latitude=sum(
            member.coordinates.latitude * weight
            for member, weight in zip(members, weights, strict=True)
        )
        / weight_total,
        longitude=sum(
            member.coordinates.longitude * weight
            for member, weight in zip(members, weights, strict=True)
        )
        / weight_total,
        coord_system=CoordinateSystem.GCJ_02,
    )
    cluster_id = uuid5(
        NAMESPACE_URL,
        f"iter:spatial:{SPATIAL_ALGORITHM_VERSION}:"
        + ":".join(str(node_id) for node_id in sorted(member_ids, key=str)),
    )
    names = [member.name for member in members]
    label = " / ".join(names[:2]) + (f" 等{len(names)}处" if len(names) > 2 else "")
    return SpatialActivityCluster(
        cluster_id=cluster_id,
        label=f"活动簇 {index} · {label}",
        member_node_ids=member_ids,
        center=center,
        suitable_dates=tuple(sorted(shared_dates)),
        intra_cluster_cost=_cluster_cost(members, edges),
    )


def _cluster_cost(
    members: tuple[SpatialAnchor, ...],
    edges: dict[frozenset[UUID], SpatialRouteEdge],
) -> ClusterCostSummary:
    pair_edges = [
        edges[frozenset((left.node_id, right.node_id))] for left, right in combinations(members, 2)
    ]
    minutes = [value for edge in pair_edges if (value := _effective_minutes(edge)) is not None]
    return ClusterCostSummary(
        pair_count=len(pair_edges),
        known_pair_count=len(minutes),
        partial_pair_count=sum(
            edge.status is DataAvailability.PARTIAL and bool(edge.routes) for edge in pair_edges
        ),
        unknown_pair_count=len(pair_edges) - len(minutes),
        minimum_minutes=min(minutes) if minutes else None,
        average_minutes=ceil(sum(minutes) / len(minutes)) if minutes else None,
        maximum_minutes=max(minutes) if minutes else None,
    )


def _inter_cluster_cost(
    left: SpatialActivityCluster,
    right: SpatialActivityCluster,
    edges: dict[frozenset[UUID], SpatialRouteEdge],
) -> InterClusterCost:
    pair_edges = [
        edges[frozenset((left_id, right_id))]
        for left_id in left.member_node_ids
        for right_id in right.member_node_ids
    ]
    known = [edge for edge in pair_edges if _effective_minutes(edge) is not None]
    if not known:
        return InterClusterCost(
            origin_cluster_id=left.cluster_id,
            destination_cluster_id=right.cluster_id,
            status=DataAvailability.MISSING,
            missing_reason="no known route cost connects these activity clusters",
        )
    best = min(known, key=lambda edge: (_effective_minutes(edge) or 0, str(edge.edge_id)))
    route = min(best.routes, key=_route_option_sort_key)
    degraded = any(edge.status is not DataAvailability.AVAILABLE for edge in pair_edges)
    return InterClusterCost(
        origin_cluster_id=left.cluster_id,
        destination_cluster_id=right.cluster_id,
        status=DataAvailability.PARTIAL if degraded else DataAvailability.AVAILABLE,
        best_edge_id=best.edge_id,
        duration_minutes=_minutes(route.duration_seconds),
        distance_m=route.distance_m,
        missing_reason=(
            "some routes between these activity clusters are partial or unknown"
            if degraded
            else None
        ),
    )


def _outlier_issues(
    anchors: tuple[SpatialAnchor, ...],
    edges: dict[frozenset[UUID], SpatialRouteEdge],
    *,
    threshold_minutes: int,
) -> tuple[SpatialPlanningIssue, ...]:
    issues: list[SpatialPlanningIssue] = []
    for anchor in anchors:
        if anchor.strength is not SpatialAnchorStrength.STRONG:
            continue
        costs = [
            (other, _effective_minutes(edges[frozenset((anchor.node_id, other.node_id))]))
            for other in anchors
            if other.node_id != anchor.node_id
        ]
        known = [(other, minutes) for other, minutes in costs if minutes is not None]
        if not known:
            continue
        closest, minimum_minutes = min(known, key=lambda item: (item[1], str(item[0].node_id)))
        if minimum_minutes <= threshold_minutes:
            continue
        issues.append(
            SpatialPlanningIssue(
                code=SpatialIssueCode.OUTLIER_STRONG_ANCHOR,
                anchor_node_id=anchor.node_id,
                closest_node_id=closest.node_id,
                minimum_known_minutes=minimum_minutes,
                reason=(
                    f"{anchor.name} 与最近的其他节点仍需约 {minimum_minutes} 分钟，"
                    "会显著增加往返成本。"
                ),
                question=f"{anchor.name} 是必须保留，还是可以为减少奔波调整？",
            )
        )
    return tuple(issues)


def _edge_lookup(edges: Iterable[SpatialRouteEdge]) -> dict[frozenset[UUID], SpatialRouteEdge]:
    return {frozenset((edge.origin_node_id, edge.destination_node_id)): edge for edge in edges}


def _edge_id(left: UUID, right: UUID) -> UUID:
    ordered = sorted((str(left), str(right)))
    return uuid5(
        NAMESPACE_URL,
        f"iter:spatial-edge:{SPATIAL_ALGORITHM_VERSION}:{ordered[0]}:{ordered[1]}",
    )


def _effective_minutes(edge: SpatialRouteEdge) -> int | None:
    if not edge.routes:
        return None
    return min(_minutes(route.duration_seconds) for route in edge.routes)


def _minutes(duration_seconds: int) -> int:
    return ceil(duration_seconds / 60)


def _route_sort_key(route: ProviderRoute) -> tuple[int, int, int]:
    return (route.duration_seconds, route.distance_m, route.source_route_index)


def _route_option_sort_key(route: SpatialRouteOption) -> tuple[int, int, int]:
    return (route.duration_seconds, route.distance_m, route.source_route_index)


def _anchor_sort_key(anchor: SpatialAnchor) -> tuple[int, int, str, str]:
    return (
        0 if anchor.fixed_time_window is not None else 1,
        0 if anchor.strength is SpatialAnchorStrength.STRONG else 1,
        anchor.name,
        str(anchor.node_id),
    )


def _coordinates(value: Gcj02Coordinates | None) -> Gcj02Coordinates:
    if value is None:  # guarded by SpatialPlanningRequest; protects model_construct callers.
        raise ValueError("spatial anchor is missing GCJ-02 coordinates")
    return value
