"""Typed capability results and deterministic workspace observation merge."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4

from backend.agent.planner.dependencies import (
    prepare_workspace_for_evidence_refresh,
    rebind_semantic_artifacts_after_evidence,
)
from backend.agent.planner.workspace import advance, refresh_pool
from backend.contracts.v4.planner_evidence import (
    PlannerCapabilityObservation,
    PlannerHotelLocationEvidence,
    PlannerHoursEvidence,
    PlannerPlaceEvidence,
    PlannerRouteComparisonObservation,
    PlannerTicketEvidence,
    PlannerWeatherEvidence,
)
from backend.contracts.v4.planner_observations import HotelObservation, SpatialRouteEdge
from backend.contracts.v4.planner_workspace import PlannerWorkspaceState, VerifiedFactSummary
from backend.contracts.v4.task_book import TaskBookV4
from backend.persistence.outbox_repository import canonical_json_hash


@dataclass(frozen=True)
class EvidenceUpdate:
    observation: PlannerCapabilityObservation
    places: tuple[PlannerPlaceEvidence, ...] = ()
    hours: tuple[PlannerHoursEvidence, ...] = ()
    weather: tuple[PlannerWeatherEvidence, ...] = ()
    tickets: tuple[PlannerTicketEvidence, ...] = ()
    hotel: HotelObservation | None = None
    hotel_locations: tuple[PlannerHotelLocationEvidence, ...] = ()
    route_edges: tuple[SpatialRouteEdge, ...] = ()
    route_comparisons: tuple[PlannerRouteComparisonObservation, ...] = ()
    facts: tuple[VerifiedFactSummary, ...] = ()


def observe_capability_results(
    workspace: PlannerWorkspaceState,
    book: TaskBookV4,
    updates: tuple[EvidenceUpdate, ...],
    now: datetime,
) -> PlannerWorkspaceState:
    semantic_workspace = workspace if workspace.working_itinerary is not None else None
    workspace = prepare_workspace_for_evidence_refresh(workspace)
    places = {item.canonical_entity_id: item for item in workspace.place_evidence}
    hours = {item.canonical_entity_id: item for item in workspace.hours_evidence}
    weather = {item.service_date: item for item in workspace.weather_evidence}
    tickets = {
        (item.canonical_entity_id, item.service_date): item for item in workspace.ticket_evidence
    }
    locations = {item.property_id: item for item in workspace.hotel_location_evidence}
    facts = {item.fact_reference_id: item for item in workspace.verified_facts}
    route_edges = {edge.route_edge_id: edge for edge in workspace.route_evidence}
    comparisons = {item.observation_id: item for item in workspace.route_comparisons}
    observations = list(workspace.capability_observations)
    hotel = workspace.hotel_observation
    pool_changed = False
    for update in updates:
        pool_changed |= bool(update.places or update.hours)
        places.update({item.canonical_entity_id: item for item in update.places})
        hours.update({item.canonical_entity_id: item for item in update.hours})
        weather.update({item.service_date: item for item in update.weather})
        tickets.update(
            {(item.canonical_entity_id, item.service_date): item for item in update.tickets}
        )
        locations.update({item.property_id: item for item in update.hotel_locations})
        route_edges.update({item.route_edge_id: item for item in update.route_edges})
        comparisons.update({item.observation_id: item for item in update.route_comparisons})
        facts.update({item.fact_reference_id: item for item in update.facts})
        observations.append(update.observation)
        hotel = update.hotel if update.hotel is not None else hotel
        values: tuple[
            PlannerPlaceEvidence
            | PlannerHoursEvidence
            | PlannerWeatherEvidence
            | PlannerTicketEvidence,
            ...,
        ] = (*update.places, *update.hours, *update.weather, *update.tickets)
        for value in values:
            provider = (
                "amap"
                if update.places or update.hours
                else "weather"
                if update.weather
                else "flyai"
            )
            facts[value.fact_reference_id] = VerifiedFactSummary(
                fact_reference_id=value.fact_reference_id,
                fact_kind=update.observation.capability.value,
                safe_summary=(
                    f"已绑定本次{update.observation.capability.value}规范化证据；"
                    "具体值见工作区对应 evidence 字段，缺失值不表示可用。"
                ),
                observed_at=value.observed_at,
                expires_at=value.expires_at if isinstance(value, PlannerHoursEvidence) else None,
                source_reference_ids=(f"provider:{provider}:{value.fact_reference_id}",),
            )
    result = advance(
        workspace,
        place_evidence=tuple(places.values()),
        hours_evidence=tuple(hours.values()),
        weather_evidence=tuple(weather.values()),
        ticket_evidence=tuple(tickets.values()),
        hotel_location_evidence=tuple(locations.values()),
        route_evidence=tuple(route_edges.values()),
        route_comparisons=tuple(comparisons.values()),
        hotel_observation=hotel,
        verified_facts=tuple(facts.values()),
        capability_observations=tuple(observations),
    )
    if pool_changed:
        refreshed = refresh_pool(result, book, now)
        if not any(update.places or update.route_edges for update in updates):
            # Calendar evidence does not change geometry. Rebind current pool
            # references without discarding fresh routes and randomly regrouping
            # when the same query later has a transient Provider failure.
            result = _rebind_calendar_only_spatial(workspace, refreshed, now)
            return _restore_semantic_workspace(semantic_workspace, result)
        return _restore_semantic_workspace(semantic_workspace, refreshed)
    if any(update.route_edges for update in updates) and result.spatial_observation is not None:
        spatial = result.spatial_observation.model_copy(
            update={
                "observation_id": str(uuid4()),
                "route_edges": tuple(route_edges.values()),
                "source_reference_ids": tuple(
                    dict.fromkeys(
                        (
                            *result.spatial_observation.source_reference_ids,
                            *(
                                fact.fact_reference_id
                                for update in updates
                                for fact in update.facts
                            ),
                        )
                    )
                ),
            }
        )
        pool = result.candidate_pool.model_copy(
            update={"spatial_observation_id": spatial.observation_id}
        )
        result = advance(result, spatial_observation=spatial, candidate_pool=pool)
    return _restore_semantic_workspace(semantic_workspace, result)


def _restore_semantic_workspace(
    previous: PlannerWorkspaceState | None,
    current: PlannerWorkspaceState,
) -> PlannerWorkspaceState:
    if previous is None:
        return current
    return rebind_semantic_artifacts_after_evidence(previous, current)


def _rebind_calendar_only_spatial(
    previous: PlannerWorkspaceState, current: PlannerWorkspaceState, now: datetime
) -> PlannerWorkspaceState:
    spatial = previous.spatial_observation
    if spatial is None:
        return current
    facts = {item.fact_reference_id: item for item in previous.verified_facts}
    fresh_facts = {
        key for key, fact in facts.items() if fact.expires_at is not None and fact.expires_at > now
    }
    available_edges = [edge for edge in spatial.route_edges if edge.status == "available"]
    # ReAct may check a draft's hours before querying any routes. Geometry is
    # still valid in that case: a calendar update must not erase its clusters.
    if any(ref not in fresh_facts for edge in available_edges for ref in edge.fact_reference_ids):
        return current
    entries = current.candidate_pool.candidate_by_id()
    if entries.keys() != previous.candidate_pool.candidate_by_id().keys():
        return current
    clusters = tuple(
        cluster.model_copy(
            update={
                "candidate_refs": tuple(
                    entries[ref.candidate_id].candidate_ref for ref in cluster.candidate_refs
                )
            }
        )
        for cluster in spatial.clusters
    )
    spatial = spatial.model_copy(
        update={
            "observation_id": str(uuid4()),
            "scope": current.current_scope.model_copy(
                update={"workspace_revision": current.workspace_revision + 1}
            ),
            "clusters": clusters,
            "outliers": tuple(
                item.model_copy(
                    update={"candidate_ref": entries[item.candidate_ref.candidate_id].candidate_ref}
                )
                for item in spatial.outliers
            ),
        }
    )
    old_entries = previous.candidate_pool.candidate_by_id()
    pool = current.candidate_pool.model_copy(
        update={
            "spatial_observation_id": spatial.observation_id,
            "candidates": tuple(
                item.model_copy(
                    update={"cluster_ids": old_entries[item.candidate_ref.candidate_id].cluster_ids}
                )
                for item in current.candidate_pool.candidates
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
    return advance(current, spatial_observation=spatial, candidate_pool=pool)
