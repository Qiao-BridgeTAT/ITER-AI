"""Apply an explicit route choice to the same published daily semantics."""

from backend.agent.planner.dependencies import invalidate_materialized_dependencies
from backend.agent.planner.workspace import PlannerGuardError, advance
from backend.contracts.v4.commands import V4PlanTransportSelectionCommand
from backend.contracts.v4.enums import PlannerStatus
from backend.contracts.v4.planner_draft import (
    DraftRouteModeSelection,
    WorkingItineraryDraft,
    canonical_planning_projection,
    planning_projection_digest,
)
from backend.contracts.v4.planner_publication import PlannerPublishedPlan
from backend.contracts.v4.planner_workspace import PlannerWorkspaceState


def apply_transport_selection(
    workspace: PlannerWorkspaceState,
    plan: PlannerPublishedPlan,
    command: V4PlanTransportSelectionCommand,
) -> PlannerWorkspaceState:
    if command.payload.plan_version_id != plan.plan_version_id:
        raise PlannerGuardError("planner_plan_version_conflict")
    draft = workspace.working_itinerary
    if draft is None:
        raise PlannerGuardError("planner_transport_workspace_not_current")
    if draft.content_digest != plan.working_itinerary.content_digest:
        # A rejected route choice must not strand the menu on an unpublished
        # draft. Recover only route-choice differences; unrelated edits still
        # require explicit plan recovery and are never overwritten here.
        current_semantics = canonical_planning_projection(draft)
        published_semantics = canonical_planning_projection(plan.working_itinerary)
        for projection in (current_semantics, published_semantics):
            for day in projection["days"]:
                day.pop("route_mode_selections", None)
        if workspace.status is not PlannerStatus.FAILED or current_semantics != published_semantics:
            raise PlannerGuardError("planner_transport_workspace_not_current")
        published_days = {day.service_date: day for day in plan.working_itinerary.days}
        draft = draft.model_copy(
            update={
                "days": tuple(
                    day.model_copy(
                        update={
                            "route_mode_selections": published_days[
                                day.service_date
                            ].route_mode_selections
                        }
                    )
                    for day in draft.days
                )
            }
        )
    selected = next(
        (
            (day, leg)
            for day in plan.materialized_schedule.days
            for leg in day.transport_legs
            if leg.leg_id == command.payload.leg_id
        ),
        None,
    )
    if selected is None:
        raise PlannerGuardError("planner_unknown_transport_leg")
    scheduled_day, leg = selected
    edges = (
        *workspace.route_evidence,
        *(workspace.spatial_observation.route_edges if workspace.spatial_observation else ()),
    )
    current_edge = next(
        (edge for edge in edges if set(edge.fact_reference_ids) & set(leg.source_reference_ids)),
        None,
    )
    if current_edge is None:
        raise PlannerGuardError("planner_transport_route_evidence_missing")
    alternative = next(
        (
            edge
            for edge in edges
            if edge.origin == current_edge.origin
            and edge.destination == current_edge.destination
            and edge.transport_mode == command.payload.transport_mode
            and edge.status != "missing"
            and edge.duration_minutes is not None
        ),
        None,
    )
    if alternative is None:
        raise PlannerGuardError("planner_transport_alternative_unavailable")
    choice = DraftRouteModeSelection(
        origin=alternative.origin,
        destination=alternative.destination,
        transport_mode=command.payload.transport_mode,
    )
    days = tuple(
        day.model_copy(
            update={
                "route_mode_selections": (
                    *(
                        item
                        for item in day.route_mode_selections
                        if (item.origin, item.destination) != (choice.origin, choice.destination)
                    ),
                    choice,
                )
            }
        )
        if day.service_date == scheduled_day.service_date
        else day
        for day in draft.days
    )
    provisional = draft.model_copy(
        update={
            "days": days,
            "draft_revision": draft.draft_revision + 1,
            "scope": workspace.current_scope,
        }
    )
    revised = WorkingItineraryDraft.model_validate(
        {
            **provisional.model_dump(mode="json"),
            "content_digest": planning_projection_digest(provisional),
        }
    )
    return advance(
        workspace,
        working_itinerary=revised,
        status=PlannerStatus.PLANNING,
        **invalidate_materialized_dependencies(
            workspace, affected_dates=(scheduled_day.service_date,)
        ),
    )
