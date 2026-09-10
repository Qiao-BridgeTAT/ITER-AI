"""Live route observations using the shared V3 complete-link spatial calculation."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from itertools import combinations, permutations
from math import ceil, cos, radians
from typing import Literal
from uuid import uuid4

from backend.agent.model_gateway import ModelCancellation
from backend.agent.planner.route_diagnostics import missing_route_reason
from backend.agent.planner.workspace import advance, server_id, service_dates
from backend.contracts.v4.enums import CommitmentLevel
from backend.contracts.v4.planner_evidence import PlannerPlaceEvidence
from backend.contracts.v4.planner_observations import (
    InterClusterCost,
    MissingRouteFact,
    SpatialCluster,
    SpatialObservation,
    SpatialOutlier,
    SpatialRouteEdge,
    SpatialRouteEndpoint,
)
from backend.contracts.v4.planner_strategy import CandidatePoolEntry
from backend.contracts.v4.planner_workspace import PlannerWorkspaceState, VerifiedFactSummary
from backend.contracts.v4.task_book import TaskBookV4
from backend.persistence.outbox_repository import canonical_json_hash
from backend.planning.spatial_planning import group_by_route_costs
from backend.providers.contracts import ProviderCityScope, ProviderError, RouteMode, RouteRequest
from backend.providers.interfaces import RouteProvider

MAX_INITIAL_DIRECTED_ROUTE_CALLS = 24
MAX_PARALLEL_INITIAL_ROUTE_CALLS = 6


async def build_spatial_observation(
    workspace: PlannerWorkspaceState,
    book: TaskBookV4,
    *,
    routes: RouteProvider,
    city: ProviderCityScope,
    cancellation: ModelCancellation,
    now: datetime,
) -> PlannerWorkspaceState:
    entries = tuple(
        sorted(
            (
                entry
                for entry in workspace.candidate_pool.candidates
                if entry.commitment_level is not CommitmentLevel.FORBIDDEN
            ),
            key=lambda entry: (
                entry.commitment_level is not CommitmentLevel.STRONG,
                entry.candidate_ref.canonical_entity_id,
            ),
        )
    )
    places = {place.canonical_entity_id: place for place in workspace.place_evidence}
    semaphore = asyncio.Semaphore(MAX_PARALLEL_INITIAL_ROUTE_CALLS)

    async def query(
        left: CandidatePoolEntry, right: CandidatePoolEntry
    ) -> tuple[tuple[SpatialRouteEdge, ...], tuple[VerifiedFactSummary, ...]]:
        async with semaphore:
            cancellation.raise_if_cancelled("planner_spatial_routes")
            origin = SpatialRouteEndpoint(
                kind="candidate", reference_id=left.candidate_ref.candidate_id
            )
            destination = SpatialRouteEndpoint(
                kind="candidate", reference_id=right.candidate_ref.candidate_id
            )
            failure: ProviderError | None = None
            try:
                response = await routes.get_routes(
                    RouteRequest(
                        city=city,
                        origin=places[left.candidate_ref.canonical_entity_id].coordinates,
                        destination=places[right.candidate_ref.canonical_entity_id].coordinates,
                        modes=[RouteMode.DRIVING, RouteMode.WALKING],
                    )
                )
            except ProviderError as error:
                response = None
                failure = error
            cancellation.raise_if_cancelled("planner_spatial_observe")
            results: list[SpatialRouteEdge] = []
            facts: list[VerifiedFactSummary] = []
            modes: tuple[tuple[Literal["taxi", "walking"], RouteMode], ...] = (
                ("taxi", RouteMode.DRIVING),
                ("walking", RouteMode.WALKING),
            )
            for mode, provider_mode in modes:
                edge_id = server_id(
                    workspace.generation_id,
                    origin.kind,
                    origin.reference_id,
                    destination.kind,
                    destination.reference_id,
                    mode,
                )
                options = (
                    [item for item in response.items if item.mode is provider_mode]
                    if response is not None
                    else []
                )
                if not options:
                    results.append(
                        SpatialRouteEdge(
                            route_edge_id=edge_id,
                            origin=origin,
                            destination=destination,
                            transport_mode=mode,
                            status="missing",
                            missing_reason=missing_route_reason(provider_mode, response, failure),
                        )
                    )
                    continue
                route = min(options, key=lambda item: item.duration_seconds)
                fact_id = server_id(edge_id, route.fetched_at.isoformat())
                facts.append(
                    VerifiedFactSummary(
                        fact_reference_id=fact_id,
                        fact_kind="route_cost",
                        safe_summary=(
                            f"{left.display_name}至{right.display_name}，{mode}，"
                            f"查询时估算{ceil(route.duration_seconds / 60)}分钟、"
                            f"{route.distance_m}米；不是未来出发时刻的交通保证。"
                        ),
                        observed_at=route.fetched_at,
                        expires_at=route.fetched_at + timedelta(hours=2),
                        source_reference_ids=(f"provider:amap:route:{edge_id}",),
                    )
                )
                results.append(
                    SpatialRouteEdge(
                        route_edge_id=edge_id,
                        origin=origin,
                        destination=destination,
                        transport_mode=mode,
                        status="available",
                        duration_minutes=ceil(route.duration_seconds / 60),
                        distance_meters=route.distance_m,
                        transfer_count=route.transfer_count,
                        fare=route.fare,
                        polyline=tuple(route.polyline),
                        fact_reference_ids=(fact_id,),
                    )
                )
            return tuple(results), tuple(facts)

    # Full directed all-pairs grows quadratically (nine candidates used to issue
    # 72 Provider calls). Coordinates only choose a bounded set of nearby pairs
    # to *query*; they never become route duration evidence. Clustering below
    # remains conservative and only joins candidates backed by observed routes.
    pairs = _initial_directed_route_pairs(entries, places)
    batches = await asyncio.gather(*(query(left, right) for left, right in pairs))
    edges = tuple(edge for batch, _ in batches for edge in batch)
    facts = tuple(fact for _, batch_facts in batches for fact in batch_facts)
    usable: dict[tuple[str, str], SpatialRouteEdge] = {}
    for edge in edges:
        if edge.duration_minutes is None:
            continue
        key = (edge.origin.reference_id, edge.destination.reference_id)
        if key not in usable or edge.duration_minutes < (usable[key].duration_minutes or 0):
            usable[key] = edge

    def symmetric_cost(left: CandidatePoolEntry, right: CandidatePoolEntry) -> int | None:
        outbound = usable.get((left.candidate_ref.candidate_id, right.candidate_ref.candidate_id))
        inbound = usable.get((right.candidate_ref.candidate_id, left.candidate_ref.candidate_id))
        if outbound is None or inbound is None:
            return None
        return max(outbound.duration_minutes or 0, inbound.duration_minutes or 0)

    # Geography is independent of open dates. Calendar eligibility is separately
    # retained in the pool and enforced when the model assigns actual dates.
    groups = group_by_route_costs(
        entries,
        available_dates=lambda _: set(service_dates(book)),
        duration_minutes=symmetric_cost,
        threshold_minutes=35,
    )
    clusters = tuple(
        SpatialCluster(
            cluster_id=server_id(
                workspace.generation_id,
                "cluster",
                *(entry.candidate_ref.candidate_id for entry in group),
            ),
            candidate_refs=tuple(entry.candidate_ref for entry in group),
        )
        for group in groups
    )
    costs = []
    for left, right in permutations(clusters, 2):
        choices = [
            usable[(a.candidate_id, b.candidate_id)]
            for a in left.candidate_refs
            for b in right.candidate_refs
            if (a.candidate_id, b.candidate_id) in usable
        ]
        if choices:
            edge = min(choices, key=lambda item: item.duration_minutes or 0)
            costs.append(
                InterClusterCost(
                    from_cluster_id=left.cluster_id,
                    to_cluster_id=right.cluster_id,
                    route_edge_ids=(edge.route_edge_id,),
                    duration_minutes=edge.duration_minutes or 0,
                    distance_meters=edge.distance_meters,
                )
            )
    outliers = _strong_spatial_outliers(entries, clusters, edges)
    observation = SpatialObservation(
        observation_id=str(uuid4()),
        scope=workspace.current_scope.model_copy(
            update={"workspace_revision": workspace.workspace_revision + 1}
        ),
        clusters=clusters,
        route_edges=edges,
        inter_cluster_costs=tuple(costs),
        outliers=tuple(outliers),
        missing_route_facts=tuple(
            MissingRouteFact(
                origin=edge.origin,
                destination=edge.destination,
                transport_modes=(edge.transport_mode,),
                reason_summary=edge.missing_reason or "路线缺失",
            )
            for edge in edges
            if edge.status == "missing"
        ),
        source_reference_ids=tuple(dict.fromkeys(fact.fact_reference_id for fact in facts))
        or tuple(place.fact_reference_id for place in places.values())
        or workspace.candidate_pool.source_reference_ids,
    )
    cluster_by_candidate = {
        reference.candidate_id: cluster.cluster_id
        for cluster in clusters
        for reference in cluster.candidate_refs
    }
    pool = workspace.candidate_pool.model_copy(
        update={
            "spatial_observation_id": observation.observation_id,
            "candidates": tuple(
                entry.model_copy(
                    update={
                        "cluster_ids": (cluster_by_candidate[entry.candidate_ref.candidate_id],)
                        if entry.candidate_ref.candidate_id in cluster_by_candidate
                        else ()
                    }
                )
                for entry in workspace.candidate_pool.candidates
            ),
        }
    )
    pool = pool.model_copy(
        update={
            "pool_fingerprint": canonical_json_hash(
                pool.model_dump(mode="json", exclude={"pool_fingerprint"})
            )
        }
    )
    unique_facts = {fact.fact_reference_id: fact for fact in (*workspace.verified_facts, *facts)}
    return advance(
        workspace,
        candidate_pool=pool,
        spatial_observation=observation,
        route_evidence=edges,
        verified_facts=tuple(unique_facts.values()),
    )


def _initial_directed_route_pairs(
    entries: tuple[CandidatePoolEntry, ...],
    places: dict[str, PlannerPlaceEvidence],
) -> tuple[tuple[CandidatePoolEntry, CandidatePoolEntry], ...]:
    """Select a deterministic sparse route graph with bounded candidate coverage.

    The coordinate distance is only a query-prioritization heuristic. Every
    returned cost, cluster merge, outlier, and inter-cluster value still comes
    from a real directed Provider observation.
    """

    if len(entries) < 2:
        return ()

    def coordinate(entry: CandidatePoolEntry) -> PlannerPlaceEvidence:
        return places[entry.candidate_ref.canonical_entity_id]

    def pair_key(
        pair: tuple[CandidatePoolEntry, CandidatePoolEntry],
    ) -> tuple[float, str, str]:
        left, right = pair
        left_coordinates = coordinate(left).coordinates
        right_coordinates = coordinate(right).coordinates
        mean_latitude = (left_coordinates.latitude + right_coordinates.latitude) / 2
        longitude_meters = (
            (left_coordinates.longitude - right_coordinates.longitude)
            * cos(radians(mean_latitude))
            * 111_320
        )
        latitude_meters = (left_coordinates.latitude - right_coordinates.latitude) * 110_540
        left_id, right_id = sorted(
            (left.candidate_ref.candidate_id, right.candidate_ref.candidate_id)
        )
        return longitude_meters**2 + latitude_meters**2, left_id, right_id

    ranked = tuple(sorted(combinations(entries, 2), key=pair_key))
    limit = min(len(ranked), MAX_INITIAL_DIRECTED_ROUTE_CALLS // 2)
    selected: list[tuple[CandidatePoolEntry, CandidatePoolEntry]] = []
    selected_ids: set[frozenset[str]] = set()
    uncovered = {entry.candidate_ref.candidate_id for entry in entries}

    # Build a small deterministic edge cover first, so optional/filler entries
    # are not silently denied all spatial evidence when strong entries exist.
    while len(uncovered) >= 2 and len(selected) < limit:
        pair = next(
            candidate
            for candidate in ranked
            if candidate[0].candidate_ref.candidate_id in uncovered
            and candidate[1].candidate_ref.candidate_id in uncovered
        )
        selected.append(pair)
        selected_ids.add(
            frozenset((pair[0].candidate_ref.candidate_id, pair[1].candidate_ref.candidate_id))
        )
        uncovered.difference_update(
            (pair[0].candidate_ref.candidate_id, pair[1].candidate_ref.candidate_id)
        )
    if uncovered and len(selected) < limit:
        remaining = next(iter(uncovered))
        pair = next(
            candidate
            for candidate in ranked
            if remaining
            in (
                candidate[0].candidate_ref.candidate_id,
                candidate[1].candidate_ref.candidate_id,
            )
        )
        selected.append(pair)
        selected_ids.add(
            frozenset((pair[0].candidate_ref.candidate_id, pair[1].candidate_ref.candidate_id))
        )

    for pair in ranked:
        identity = frozenset(
            (pair[0].candidate_ref.candidate_id, pair[1].candidate_ref.candidate_id)
        )
        if identity in selected_ids:
            continue
        selected.append(pair)
        selected_ids.add(identity)
        if len(selected) >= limit:
            break

    directed: list[tuple[CandidatePoolEntry, CandidatePoolEntry]] = []
    for left, right in sorted(selected, key=pair_key):
        first, second = sorted((left, right), key=lambda item: item.candidate_ref.candidate_id)
        directed.extend(((first, second), (second, first)))
    return tuple(directed)


def _strong_spatial_outliers(
    entries: tuple[CandidatePoolEntry, ...],
    clusters: tuple[SpatialCluster, ...],
    edges: tuple[SpatialRouteEdge, ...],
) -> tuple[SpatialOutlier, ...]:
    """Observe remote strong anchors, without deciding which day contains them."""
    cluster_by_candidate = {
        ref.candidate_id: cluster.cluster_id
        for cluster in clusters
        for ref in cluster.candidate_refs
    }
    shortest: dict[tuple[str, str], SpatialRouteEdge] = {}
    for edge in edges:
        if (
            edge.origin.kind != "candidate"
            or edge.destination.kind != "candidate"
            or edge.duration_minutes is None
        ):
            continue
        pair = (edge.origin.reference_id, edge.destination.reference_id)
        previous = shortest.get(pair)
        if previous is None or edge.duration_minutes < (previous.duration_minutes or 0):
            shortest[pair] = edge
    outliers = []
    for entry in entries:
        if entry.commitment_level is not CommitmentLevel.STRONG:
            continue
        key = entry.candidate_ref.candidate_id
        own_cluster = cluster_by_candidate.get(key)
        if own_cluster is None:
            continue
        outside = [
            item.candidate_ref.candidate_id
            for item in entries
            if cluster_by_candidate.get(item.candidate_ref.candidate_id) != own_cluster
        ]
        required = [
            (left, right) for other in outside for left, right in ((key, other), (other, key))
        ]
        # An adjacent restaurant in the same remote cluster must not erase the
        # anchor's remoteness. Missing destinations or reverse routes are unknown,
        # not evidence that every route is long.
        if not required or any(pair not in shortest for pair in required):
            continue
        observed = [shortest[pair] for pair in required]
        if any((edge.duration_minutes or 0) <= 75 for edge in observed):
            continue
        outliers.append(
            SpatialOutlier(
                candidate_ref=entry.candidate_ref,
                reason_summary=(
                    "这项强意愿到本活动簇以外各候选的双向最快已观测路线均超过75分钟；"
                    "同簇邻近点不抵消远郊距离，需要模型结合实际路线决定分天。"
                ),
                route_edge_ids=tuple(edge.route_edge_id for edge in observed),
            )
        )
    return tuple(outliers)
