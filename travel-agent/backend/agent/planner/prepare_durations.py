"""Reuse Prepare's estimates after re-verifying the selected place identities."""

from backend.agent.planner.workspace import advance
from backend.contracts.v4.enums import CandidateEntityKind
from backend.contracts.v4.planner_evidence import PlannerVisitDurationEstimate
from backend.contracts.v4.planner_workspace import PlannerWorkspaceState
from backend.persistence.outbox_repository import canonical_json_hash


def inherit_prepare_visit_durations(workspace: PlannerWorkspaceState) -> PlannerWorkspaceState:
    estimates = {item.canonical_entity_id: item for item in workspace.visit_duration_estimates}
    admitted = {
        item.candidate_ref.canonical_entity_id for item in workspace.candidate_pool.candidates
    }
    verified = {
        (place.canonical_entity_id, place.provider_entity_id)
        for place in workspace.place_evidence
        if place.entity_kind is CandidateEntityKind.ATTRACTION
    }
    additions = []
    for origin in workspace.candidate_origins:
        duration = origin.suggested_visit_duration
        if (
            duration is None
            or origin.canonical_entity_id in estimates
            or origin.canonical_entity_id not in admitted
            or (origin.canonical_entity_id, origin.provider_entity_id) not in verified
        ):
            continue
        additions.append(
            PlannerVisitDurationEstimate(
                canonical_entity_id=origin.canonical_entity_id,
                minimum_minutes=duration.minimum_minutes,
                maximum_minutes=duration.maximum_minutes,
                source="llm_estimate",
                source_reference_ids=(
                    f"message:{origin.source_message_id}",
                    f"card_option:{origin.source_option_id}",
                ),
                context_fingerprint=canonical_json_hash(origin.model_dump(mode="json")),
            )
        )
    if not additions:
        return workspace
    return advance(workspace, visit_duration_estimates=(*estimates.values(), *additions))
