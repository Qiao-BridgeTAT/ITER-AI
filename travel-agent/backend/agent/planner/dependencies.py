"""Minimum dependency invalidation and safe reference rebinding for V4 Planner."""

from __future__ import annotations

from collections import Counter
from datetime import date
from typing import Any
from uuid import UUID

from backend.agent.planner.workspace import PlannerGuardError, server_id
from backend.contracts.v4.enums import PlannerCapability, PlannerStatus
from backend.contracts.v4.plan_change import (
    FormalHotelRecommendationSet,
    SelectedHotelRecommendation,
)
from backend.contracts.v4.planner_draft import (
    WorkingItineraryDraft,
    planning_projection_digest,
)
from backend.contracts.v4.planner_strategy import PlanningStrategy
from backend.contracts.v4.planner_workspace import PlannerWorkspaceState


def invalidate_materialized_dependencies(
    workspace: PlannerWorkspaceState,
    *,
    affected_dates: tuple[date, ...],
) -> dict[str, object]:
    """Return the formal artifacts invalidated by any semantic itinerary change."""

    # Schedule/cost contracts are whole-trip envelopes today, so even a local day
    # replacement invalidates their envelope and validation fingerprint. The
    # materializer itself still receives the exact affected dates for future
    # partial persistence and does not reselect unaffected semantic days.
    del affected_dates
    return {
        "materialized_schedule": None,
        "cost_draft": None,
        "validation_report": None,
        "validation_observation": None,
    }


def rebase_workspace_for_plan_change(
    workspace: PlannerWorkspaceState,
    *,
    generation_id: UUID,
) -> PlannerWorkspaceState:
    """Clone immutable published-plan inputs into a fresh turn-owned workspace."""

    def rebase(value: object) -> object:
        if isinstance(value, dict):
            result = {key: rebase(child) for key, child in value.items()}
            scope_keys = {
                "trip_id",
                "generation_id",
                "task_book_id",
                "task_book_version",
                "task_book_state_version",
                "workspace_revision",
            }
            if scope_keys <= set(result):
                result["generation_id"] = str(generation_id)
            return result
        if isinstance(value, (list, tuple)):
            return [rebase(child) for child in value]
        return value

    payload = rebase(workspace.model_dump(mode="json"))
    assert isinstance(payload, dict)
    if str(generation_id) != workspace.generation_id:
        # These receipts also represent the per-generation nearby-search budget.
        # Rebinding their scope would incorrectly spend the new run's budget,
        # while full evidence initialization rebuilds its pool without the old
        # supplemental choices. Keep facts and ordinary evidence receipts; the
        # original execution receipts remain in the published checkpoint/audit.
        legacy_recall_id = server_id(workspace.generation_id, "schedule-nearby-choices")
        payload["capability_observations"] = [
            rebase(item.model_dump(mode="json"))
            for item in workspace.capability_observations
            if not (
                item.capability is PlannerCapability.OPENING_HOURS
                or (
                    item.capability is PlannerCapability.CANDIDATE_RECALL
                    and (
                        item.request_id == legacy_recall_id
                        or item.reason_summary.startswith("排程补查：")
                    )
                )
            )
        ]
        # An unavailable lookup from a previous run is not a permanent fact.
        # Fresh known facts still avoid a request; each unknown identity/date
        # receives at most one lookup in this new generation.
        payload["hours_evidence"] = [
            item.model_dump(mode="json")
            for item in workspace.hours_evidence
            if all(day.status in {"open", "closed"} for day in item.days)
        ]
    payload.update(
        {
            "generation_id": str(generation_id),
            "workspace_revision": workspace.workspace_revision + 1,
            "guard_observations": [],
            "interaction_answers": [],
            "unresolved_decisions": [],
            # Accepted repairs carry the authority for unassigned soft intents.
            # They remain immutable input history, with ownership rebased above.
            "revision_round": 0,
            "user_interrupt_count": 0,
            "active_interaction": None,
            # New user turns acquire their own authority. Historical patch
            # receipts stay in the trace but cannot scope the new request.
            "plan_change_request": None,
            "segment_attempt_count": 0,
            "segment_evidence_count": 0,
            "status": PlannerStatus.PLANNING,
        }
    )
    return PlannerWorkspaceState.model_validate(payload)


def prepare_workspace_for_evidence_refresh(
    workspace: PlannerWorkspaceState,
    *,
    invalidate_hotel: bool = False,
    reset_initial_evidence: bool = False,
) -> PlannerWorkspaceState:
    """Detach stale semantic artifacts while Provider observations are merged."""

    if workspace.working_itinerary is None:
        return workspace
    return PlannerWorkspaceState.model_validate(
        {
            **workspace.model_dump(mode="json"),
            "planning_strategy": None,
            "hotel_observation": None if invalidate_hotel else workspace.hotel_observation,
            "selected_hotel": None,
            "hotel_recommendations": None,
            "working_itinerary": None,
            "materialized_schedule": None,
            "cost_draft": None,
            "validation_report": None,
            "validation_observation": None,
            "revision_round": 0,
            "initial_evidence_ready": (
                False if reset_initial_evidence else workspace.initial_evidence_ready
            ),
            "status": PlannerStatus.PLANNING,
        }
    )


def rebind_semantic_artifacts_after_evidence(
    previous: PlannerWorkspaceState,
    current: PlannerWorkspaceState,
    *,
    refresh_clusters: bool = False,
) -> PlannerWorkspaceState:
    """Rebind unchanged semantic choices when evidence only minted new formal refs."""

    strategy = previous.planning_strategy
    draft = previous.working_itinerary
    if strategy is None or draft is None:
        return current
    current_entries = current.candidate_pool.candidate_by_id()
    previous_ids = previous.candidate_pool.candidate_by_id().keys()
    if not set(previous_ids) <= set(current_entries):
        raise PlannerGuardError("planner_evidence_changed_selected_candidate_identity")
    replacements = {
        candidate_id: entry.candidate_ref.model_dump(mode="python")
        for candidate_id, entry in current_entries.items()
    }
    output_scope = current.current_scope
    strategy_payload = _replace_candidate_refs(
        strategy.model_dump(mode="python"),
        replacements,
    )
    assert isinstance(strategy_payload, dict)
    strategy_payload.update(
        {
            "scope": output_scope.model_dump(mode="python"),
            "candidate_pool_revision": current.candidate_pool.revision,
        }
    )
    rebound_strategy = PlanningStrategy.model_validate(strategy_payload)

    draft_payload = _replace_candidate_refs(
        draft.model_dump(mode="python"),
        replacements,
    )
    draft_payload = _replace_hotel_offer_refs(draft_payload, current)
    assert isinstance(draft_payload, dict)
    if refresh_clusters:
        # New nearby choices may merge the spatial groups. Retag only formal
        # cluster fields; never change selected identities/order/time preferences.
        for day in draft_payload["days"]:
            # These optional legacy explanations bind old clusters and route
            # evidence. Rebuild formal cluster tags from current observations,
            # then let the ordinary adjacent-route query verify the new path.
            # Never ask Qwen to reconstruct obsolete spatial bookkeeping.
            day["cross_cluster_segments"] = []
            clusters = []
            for item in day["ordered_items"]:
                reference = item["object_ref"]
                if reference["kind"] != "candidate":
                    continue
                entry = current_entries[reference["candidate_id"]]
                item["cluster_id"] = entry.cluster_ids[0] if entry.cluster_ids else None
                if item["cluster_id"] is not None:
                    clusters.append(item["cluster_id"])
            day["primary_cluster_id"] = Counter(clusters).most_common(1)[0][0] if clusters else None
            day["day_kind"] = (
                "active" if clusters else "arrival_departure" if day["ordered_items"] else "rest"
            )
    draft_payload.update(
        {
            "scope": output_scope.model_dump(mode="python"),
            "candidate_pool_revision": current.candidate_pool.revision,
            "spatial_observation_id": current.spatial_observation.observation_id
            if current.spatial_observation is not None
            else draft.spatial_observation_id,
            "hotel_observation_id": (
                current.hotel_observation.hotel_observation_id
                if draft.hotel_observation_id is not None and current.hotel_observation is not None
                else draft.hotel_observation_id
            ),
            "content_digest": "0" * 64,
        }
    )
    provisional = WorkingItineraryDraft.model_validate(draft_payload)
    draft_payload["content_digest"] = planning_projection_digest(provisional)
    rebound_draft = WorkingItineraryDraft.model_validate(draft_payload)

    recommendations = _rebind_recommendations(
        previous.hotel_recommendations,
        current,
        output_scope.model_dump(mode="python"),
    )
    selected_hotel = _rebind_selected_hotel(
        previous.selected_hotel,
        current,
        output_scope.model_dump(mode="python"),
    )
    try:
        return PlannerWorkspaceState.model_validate(
            {
                **current.model_dump(mode="json"),
                "planning_strategy": rebound_strategy,
                "working_itinerary": rebound_draft,
                "selected_hotel": selected_hotel,
                "hotel_recommendations": recommendations,
                # The draft and its omission receipts must advance together.
                # Observations stay immutable historical evidence.
                "recovery_omissions": _replace_candidate_refs(
                    [item.model_dump(mode="python") for item in current.recovery_omissions],
                    replacements,
                ),
                "materialized_schedule": None,
                "cost_draft": None,
                "validation_report": None,
                "validation_observation": None,
                "revision_round": previous.revision_round,
                "status": PlannerStatus.PLANNING,
            }
        )
    except ValueError as error:
        raise PlannerGuardError("planner_evidence_rebind_failed") from error


def _replace_candidate_refs(value: object, replacements: dict[str, dict[str, Any]]) -> object:
    if isinstance(value, dict):
        if value.get("kind") == "candidate" and isinstance(value.get("candidate_id"), str):
            candidate_id = value["candidate_id"]
            replacement = replacements.get(candidate_id)
            if replacement is None:
                raise PlannerGuardError("planner_evidence_candidate_reference_missing")
            if any(
                value.get(key) != replacement.get(key)
                for key in (
                    "candidate_pool_id",
                    "candidate_id",
                    "canonical_entity_id",
                    "entity_kind",
                )
            ):
                raise PlannerGuardError("planner_evidence_changed_selected_candidate_identity")
            return dict(replacement)
        return {key: _replace_candidate_refs(child, replacements) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_replace_candidate_refs(child, replacements) for child in value]
    return value


def _rebind_recommendations(
    recommendations: FormalHotelRecommendationSet | None,
    workspace: PlannerWorkspaceState,
    scope: dict[str, Any],
) -> FormalHotelRecommendationSet | None:
    if recommendations is None:
        return None
    payload = _replace_hotel_offer_refs(
        recommendations.model_dump(mode="python"),
        workspace,
    )
    assert isinstance(payload, dict)
    if workspace.hotel_observation is None:
        raise PlannerGuardError("planner_evidence_hotel_observation_missing")
    return FormalHotelRecommendationSet.model_validate(
        {
            **payload,
            "scope": scope,
            "hotel_observation_id": workspace.hotel_observation.hotel_observation_id,
        }
    )


def _rebind_selected_hotel(
    selection: SelectedHotelRecommendation | None,
    workspace: PlannerWorkspaceState,
    scope: dict[str, Any],
) -> SelectedHotelRecommendation | None:
    if selection is None:
        return None
    payload = _replace_hotel_offer_refs(
        selection.model_dump(mode="python"),
        workspace,
    )
    assert isinstance(payload, dict)
    if workspace.hotel_observation is None:
        raise PlannerGuardError("planner_evidence_hotel_observation_missing")
    return SelectedHotelRecommendation.model_validate(
        {
            **payload,
            "scope": scope,
            "hotel_observation_id": workspace.hotel_observation.hotel_observation_id,
        }
    )


def _replace_hotel_offer_refs(value: object, workspace: PlannerWorkspaceState) -> object:
    observation = workspace.hotel_observation
    if isinstance(value, dict):
        if value.get("kind") == "hotel_offer" and isinstance(value.get("property_id"), str):
            if observation is None:
                raise PlannerGuardError("planner_evidence_hotel_observation_missing")
            property_id = value["property_id"]
            replacement = next(
                (
                    offer.offer_ref.model_dump(mode="python")
                    for offer in observation.offers
                    if offer.offer_ref.property_id == property_id
                    and offer.availability_status != "unavailable"
                ),
                None,
            )
            if replacement is None:
                raise PlannerGuardError("planner_evidence_selected_hotel_no_longer_available")
            return replacement
        return {key: _replace_hotel_offer_refs(child, workspace) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_replace_hotel_offer_refs(child, workspace) for child in value]
    return value
