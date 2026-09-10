"""Compare model-proposed alternatives using exact, sourced route legs only."""

from __future__ import annotations

from datetime import datetime

from backend.agent.planner.workspace import PlannerGuardError, server_id
from backend.contracts.v4.planner_draft import WorkingItineraryDraft
from backend.contracts.v4.planner_evidence import PlannerRouteComparisonObservation
from backend.contracts.v4.planner_observations import RouteComparisonInput, SpatialRouteEdge
from backend.contracts.v4.planner_refs import CandidateRef, PlannerScope
from backend.contracts.v4.planner_workspace import PlannerWorkspaceState, VerifiedFactSummary


def compare_observed_routes(
    comparison: RouteComparisonInput,
    edges: tuple[SpatialRouteEdge, ...],
    facts: tuple[VerifiedFactSummary, ...],
    *,
    scope: PlannerScope,
    pool_revision: int,
    request_id: str,
    now: datetime,
) -> PlannerRouteComparisonObservation | None:
    pairs = {
        (left, right)
        for day in (*comparison.baseline_days, *comparison.proposed_days)
        for left, right in zip(day.ordered_endpoints, day.ordered_endpoints[1:], strict=False)
    }
    selected = {
        (edge.origin, edge.destination): edge
        for edge in edges
        if edge.transport_mode == comparison.transport_mode
        and (edge.origin, edge.destination) in pairs
    }
    if (
        set(selected) != pairs
        or not pairs
        or any(
            edge.status != "available" or edge.duration_minutes is None
            for edge in selected.values()
        )
    ):
        return None
    referenced = {ref for edge in selected.values() for ref in edge.fact_reference_ids}
    sources = {fact.fact_reference_id: fact for fact in facts}
    if not referenced or not referenced <= sources.keys():
        return None
    expires = [sources[ref].expires_at for ref in referenced]
    if any(value is None or value <= now for value in expires):
        return None
    totals = tuple(
        sum(
            selected[pair].duration_minutes or 0
            for day in days
            for pair in zip(day.ordered_endpoints, day.ordered_endpoints[1:], strict=False)
        )
        for days in (comparison.baseline_days, comparison.proposed_days)
    )
    return PlannerRouteComparisonObservation(
        observation_id=server_id(request_id, "route-comparison"),
        scope=scope,
        candidate_pool_revision=pool_revision,
        comparison=comparison,
        route_edges=tuple(selected.values()),
        baseline_duration_minutes=totals[0],
        proposed_duration_minutes=totals[1],
        observed_at=now,
        expires_at=min(value for value in expires if value is not None),
    )


def guard_global_route_comparison(
    observation_id: str | None,
    draft: WorkingItineraryDraft,
    workspace: PlannerWorkspaceState,
    now: datetime,
) -> None:
    comparison = next(
        (item for item in workspace.route_comparisons if item.observation_id == observation_id),
        None,
    )
    if comparison is None:
        raise PlannerGuardError("planner_global_route_comparison_not_verified")
    if (
        comparison.candidate_pool_revision != workspace.candidate_pool.revision
        or comparison.expires_at <= now
        or comparison.proposed_duration_minutes >= comparison.baseline_duration_minutes
    ):
        raise PlannerGuardError("planner_route_comparison_stale_or_not_improved")
    current_edges = {edge.route_edge_id: edge for edge in workspace.route_evidence}
    if any(current_edges.get(edge.route_edge_id) != edge for edge in comparison.route_edges):
        raise PlannerGuardError("planner_route_comparison_legs_not_current")
    baseline = draft.lodging_baseline
    lodging = (
        ("hotel_offer", baseline.selected_offer_ref.offer_id)
        if baseline.selected_offer_ref
        else ("fixed_commitment", baseline.fixed_commitment_ref.commitment_id)
        if baseline.fixed_commitment_ref
        else None
    )
    expected = []
    for day in draft.days:
        path = tuple(
            ("candidate", item.object_ref.candidate_id)
            if isinstance(item.object_ref, CandidateRef)
            else ("fixed_commitment", item.object_ref.commitment_id)
            for item in day.ordered_items
        )
        if path and lodging:
            path = (lodging, *path, lodging)
        expected.append((day.service_date, path))
        if comparison.comparison.transport_mode not in day.transport_preferences:
            raise PlannerGuardError("planner_route_comparison_mode_not_in_draft")
    actual = [
        (day.service_date, tuple((item.kind, item.reference_id) for item in day.ordered_endpoints))
        for day in comparison.comparison.proposed_days
    ]
    if expected != actual:
        raise PlannerGuardError("planner_route_comparison_does_not_match_full_draft")
