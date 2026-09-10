"""Bind a validated Planner workspace to one immutable V4 plan version."""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import UTC, datetime
from uuid import UUID

from backend.agent.planner.daily_repair import measure_repair_issues
from backend.agent.planner.map_projection import build_planner_map_projection
from backend.agent.planner.plan_intent_compiler import preferred_transport_modes
from backend.agent.planner.workspace import PlannerGuardError
from backend.contracts.v4.enums import PlannerStatus
from backend.contracts.v4.plan_change import PlanChangeRequest
from backend.contracts.v4.planner_decision import ProposeFinalizePayload
from backend.contracts.v4.planner_evidence import PlannerHotelLocationEvidence
from backend.contracts.v4.planner_observations import HotelObservation
from backend.contracts.v4.planner_publication import (
    PlannerPublishedPlan,
    planner_publication_digest,
)
from backend.contracts.v4.planner_workspace import PlannerWorkspaceState
from backend.contracts.v4.task_book import TaskBookV4


def build_planner_published_plan(
    workspace: PlannerWorkspaceState,
    book: TaskBookV4,
    *,
    plan_version_id: UUID,
    publication_key: str,
    based_on_state_version: int,
    change_request: PlanChangeRequest | None = None,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> PlannerPublishedPlan:
    """Create a formal plan only after the server-owned finalize decision."""

    if workspace.status is not PlannerStatus.READY_TO_PUBLISH:
        raise PlannerGuardError("planner_publication_workspace_not_ready")
    if not workspace.decision_trace or not isinstance(
        workspace.decision_trace[-1].payload,
        ProposeFinalizePayload,
    ):
        raise PlannerGuardError("planner_publication_finalize_decision_missing")
    draft = workspace.working_itinerary
    schedule = workspace.materialized_schedule
    cost = workspace.cost_draft
    report = workspace.validation_report
    observation = workspace.validation_observation
    if any(value is None for value in (draft, schedule, cost, report, observation)):
        raise PlannerGuardError("planner_publication_artifacts_incomplete")
    assert draft is not None
    assert schedule is not None
    assert cost is not None
    assert report is not None
    assert observation is not None
    now = clock()
    if now.tzinfo is None or now.utcoffset() is None:
        raise PlannerGuardError("planner_publication_clock_not_aware")
    map_projection = build_planner_map_projection(workspace, book)
    provisional = PlannerPublishedPlan.model_construct(
        plan_version_id=plan_version_id,
        publication_key=publication_key,
        generation_id=UUID(workspace.generation_id),
        trip_id=UUID(workspace.trip_id),
        based_on_state_version=based_on_state_version,
        based_on_task_book_id=book.task_book_id,
        based_on_task_book_version=book.version,
        travel_style_summary=_travel_style_summary(book),
        best_effort_reasons=workspace.best_effort_reasons,
        schedule_quality_status="partial"
        if any(issue.kind != "evening" for issue in measure_repair_issues(workspace, book))
        else "complete",
        working_itinerary=draft,
        materialized_schedule=schedule,
        cost_draft=cost,
        validation_report=report,
        validation_observation=observation,
        map_projection=map_projection,
        hotel_observation=_public_hotel_observation(workspace),
        selected_hotel=workspace.selected_hotel,
        hotel_recommendations=workspace.hotel_recommendations,
        hotel_location_evidence=_public_hotel_location_evidence(workspace),
        place_evidence=workspace.place_evidence,
        hours_evidence=workspace.hours_evidence,
        weather_evidence=workspace.weather_evidence,
        ticket_evidence=workspace.ticket_evidence,
        route_evidence=tuple(
            {
                edge.route_edge_id: edge
                for edge in (
                    *(
                        workspace.spatial_observation.route_edges
                        if workspace.spatial_observation
                        else ()
                    ),
                    *workspace.route_evidence,
                )
            }.values()
        ),
        change_request=change_request,
        published_at=now.astimezone(UTC),
        content_digest="0" * 64,
    )
    payload = provisional.model_dump(mode="json")
    return PlannerPublishedPlan.model_validate(
        {**payload, "content_digest": planner_publication_digest(provisional)}
    )


def _travel_style_summary(book: TaskBookV4) -> str:
    pace = " ".join(item.value for item in book.pace_and_transport.pace_preferences)
    mode = {
        "public_transit": "以公共交通为主",
        "driving": "以自驾为主",
        "walking": "以步行为主",
        "taxi": "以打车为主",
    }[preferred_transport_modes(book)[0]]
    rhythm = (
        "少一些项目，多一些深度游览"
        if re.search(r"松|慢|舒适|悠闲|少走", pace)
        else "游览较紧凑，尽量多体验不同地点"
        if re.search(r"紧凑|充实|多逛|高效", pace)
        else "游览深度与沿途体验兼顾"
    )
    return f"{mode}，{rhythm}。"


def _public_hotel_observation(workspace: PlannerWorkspaceState) -> HotelObservation | None:
    """Keep replacement candidates private while publishing the selected offer's facts."""

    observation = workspace.hotel_observation
    selection = workspace.selected_hotel
    if observation is None or selection is None:
        # Legacy 1+2 publications still need all three bound offers. Their
        # historical shape and digest remain readable by PlannerPublishedPlan.
        return observation
    selected_offer = next(
        (offer for offer in observation.offers if offer.offer_ref == selection.hotel_offer_ref),
        None,
    )
    if selected_offer is None:
        raise PlannerGuardError("planner_publication_selected_hotel_offer_missing")
    return observation.model_copy(update={"offers": (selected_offer,)})


def _public_hotel_location_evidence(
    workspace: PlannerWorkspaceState,
) -> tuple[PlannerHotelLocationEvidence, ...]:
    """Publish location evidence only for the active lodging baseline."""

    draft = workspace.working_itinerary
    if draft is None:
        return ()
    selected = draft.lodging_baseline.selected_offer_ref
    fixed = draft.lodging_baseline.fixed_commitment_ref
    property_id = (
        selected.property_id
        if selected is not None
        else fixed.commitment_id
        if fixed is not None
        else None
    )
    if property_id is None:
        return ()
    return tuple(
        item for item in workspace.hotel_location_evidence if item.property_id == property_id
    )
