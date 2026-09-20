"""Deliver saved Agent choices with their actual checks, including unresolved issues."""

from typing import Literal

from backend.agent.model_gateway import ModelCancellation
from backend.agent.planner.finalization import build_finalize_decision
from backend.agent.planner.materializer import PlannerDraftMaterializer
from backend.agent.planner.react_review import evidence_digest
from backend.agent.planner.validator import PlannerDraftValidator
from backend.agent.planner.workspace import advance
from backend.contracts.v4.enums import InteractionStatus, PlannerStatus
from backend.contracts.v4.planner_workspace import PlannerWorkspaceState
from backend.contracts.v4.task_book import TaskBookV4


def delivery_assessment(
    workspace: PlannerWorkspaceState,
) -> tuple[Literal["verified", "with_issues", "not_reviewed"], tuple[str, ...]]:
    """A rejected/stale review must never turn into an approval during delivery."""
    draft, validation = workspace.working_itinerary, workspace.validation_observation
    assert draft is not None and validation is not None
    review = workspace.react_state.review if workspace.react_state else None
    current_review = bool(
        review
        and review.draft_revision == draft.draft_revision
        and review.draft_digest == draft.content_digest
        and review.evidence_digest == evidence_digest(workspace)
        and review.validation_fingerprint == validation.validation_fingerprint
    )
    notes: list[str] = []
    if review and current_review:
        notes.extend(
            f"{issue.target}：{issue.description} 建议：{issue.suggestion}"
            for issue in review.verdict.issues
        )
    elif workspace.react_state:
        notes.append("本版行程未完成独立复核，以下时间和费用按已获得的资料安排。")
    notes.extend(issue.message_summary for issue in validation.issues)
    if draft.unassigned_intents:
        notes.append("部分旅行意愿未安排，请结合逐日安排查看。")
        candidates = workspace.candidate_pool.candidate_by_id()
        for intent in draft.unassigned_intents:
            entry = candidates.get(intent.candidate_ref.candidate_id)
            if intent.planner_reason and entry:
                notes.append(f"本次未安排{entry.display_name}：{intent.planner_reason}")
    if workspace.unresolved_decisions:
        notes.append("部分要求仍存在取舍，当前行程按原任务书保留安排，未改变已确认要求。")
    status: Literal["verified", "with_issues", "not_reviewed"] = "verified"
    if validation.result != "passed" or (current_review and review and not review.verdict.accepted):
        status = "with_issues"
    elif workspace.react_state and not current_review:
        status = "not_reviewed"
    return status, tuple(dict.fromkeys(notes))


async def prepare_result_delivery(
    workspace: PlannerWorkspaceState,
    book: TaskBookV4,
    cancellation: ModelCancellation,
    *,
    materializer: PlannerDraftMaterializer,
    validator: PlannerDraftValidator,
    input_state_version: int | None,
) -> PlannerWorkspaceState:
    """Local closing work only: no model, no Provider, no new semantic choices."""
    if workspace.working_itinerary is None or workspace.status in {
        PlannerStatus.AWAITING_USER,
        PlannerStatus.CANCELLED,
        PlannerStatus.STALE,
    }:
        return workspace
    cancellation.raise_if_cancelled("planner_result_delivery")
    materialized = await materializer.materialize(
        workspace,
        book,
        cancellation,
        input_state_version=input_state_version,
        allow_time_savings=False,
    )
    workspace = advance(
        workspace,
        status=PlannerStatus.PLANNING,
        active_interaction=(
            workspace.active_interaction.model_copy(update={"status": InteractionStatus.SUPERSEDED})
            if workspace.active_interaction
            else None
        ),
        materialized_schedule=materialized.schedule,
        cost_draft=materialized.cost,
        validation_observation=None,
        validation_report=None,
    )
    checked = await validator.validate(workspace, book, cancellation)
    workspace = advance(
        workspace,
        validation_observation=checked.observation,
        validation_report=checked.legacy_report,
    )
    decision = build_finalize_decision(workspace, allow_unresolved=True)
    # Repeating closing work after recovery binds the same artifacts; do not
    # append a duplicate deterministic decision ID.
    trace = tuple(d for d in workspace.decision_trace if d.decision_id != decision.decision_id)
    return advance(
        workspace,
        status=PlannerStatus.READY_TO_PUBLISH,
        decision_trace=(*trace, decision),
    )
