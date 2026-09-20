"""Review approval binds exact semantic, evidence and calculation versions."""

import json
from typing import Any

from backend.agent.planner.workspace import PlannerGuardError
from backend.contracts.v4.planner_workspace import PlannerWorkspaceState
from backend.persistence.outbox_repository import canonical_json_hash


def evidence_digest(workspace: PlannerWorkspaceState) -> str:
    fields = (
        "place_evidence",
        "hours_evidence",
        "weather_evidence",
        "ticket_evidence",
        "hotel_observation",
        "hotel_location_evidence",
        "route_evidence",
        "verified_facts",
    )
    facts = workspace.model_dump(mode="json", include=set(fields))
    if workspace.visit_duration_estimates:
        facts["visit_duration_estimates"] = [
            item.model_dump(mode="json") for item in workspace.visit_duration_estimates
        ]
    if workspace.react_state is not None:
        facts["mcp_observations"] = mcp_observations(workspace)
    return canonical_json_hash(facts)


def has_current_review(workspace: PlannerWorkspaceState) -> bool:
    """A rejection is a reusable verdict too; freshness does not imply approval."""
    review = workspace.react_state.review if workspace.react_state else None
    draft, validation = workspace.working_itinerary, workspace.validation_observation
    return bool(
        review
        and draft
        and validation
        and review.draft_revision == draft.draft_revision
        and review.draft_digest == draft.content_digest
        and review.evidence_digest == evidence_digest(workspace)
        and review.validation_fingerprint == validation.validation_fingerprint
    )


def require_current_review(workspace: PlannerWorkspaceState) -> None:
    if workspace.react_state is None:
        return
    review = workspace.react_state.review
    draft = workspace.working_itinerary
    validation = workspace.validation_observation
    if review is None or draft is None or validation is None:
        raise PlannerGuardError("planner_current_review_required")
    if not has_current_review(workspace):
        raise PlannerGuardError("planner_review_stale")
    if not review.verdict.accepted:
        raise PlannerGuardError("planner_review_not_accepted")


def mcp_observations(workspace: PlannerWorkspaceState) -> list[dict[str, Any]]:
    """Shared facts only; cached repeats do not invalidate an otherwise current review."""
    evidence = {}
    if workspace.react_state is None:
        return []
    for receipt in workspace.react_state.receipts:
        if (
            receipt.status != "completed"
            or not receipt.result
            or not receipt.call.function.name.startswith(("maps_", "tavily_"))
        ):
            continue
        result = json.loads(receipt.result)
        result.pop("cached", None)
        args = json.loads(receipt.call.function.arguments)
        key = canonical_json_hash({"name": receipt.call.function.name, "arguments": args})
        evidence[key] = {"tool": receipt.call.function.name, "arguments": args, "result": result}
    return [evidence[key] for key in sorted(evidence)]
