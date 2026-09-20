"""Server-owned finalize audit action for a fully validated V4 Planner workspace."""

from __future__ import annotations

from backend.agent.planner.workspace import PlannerGuardError, server_id
from backend.contracts.v4.enums import InteractionStatus
from backend.contracts.v4.planner_decision import (
    FinalizeArtifactRefs,
    PlannerCompletionAssessment,
    PlannerDecision,
    PlannerInputRefs,
    ProposeFinalizePayload,
    RequestEvidencePayload,
)
from backend.contracts.v4.planner_workspace import PlannerWorkspaceState


def build_finalize_decision(
    workspace: PlannerWorkspaceState, *, allow_unresolved: bool = False
) -> PlannerDecision:
    """Bind current formal artifacts without asking the model to echo fixed fields."""

    from backend.agent.planner.react_review import require_current_review

    if not allow_unresolved:
        require_current_review(workspace)
    strategy = workspace.planning_strategy
    draft = workspace.working_itinerary
    schedule = workspace.materialized_schedule
    cost = workspace.cost_draft
    report = workspace.validation_report
    observation = workspace.validation_observation
    if any(value is None for value in (strategy, draft, schedule, cost, report, observation)):
        raise PlannerGuardError("planner_finalize_artifacts_incomplete")
    assert strategy is not None
    assert draft is not None
    assert schedule is not None
    assert cost is not None
    assert observation is not None
    if not allow_unresolved and observation.result != "passed":
        raise PlannerGuardError("planner_finalize_validation_not_passed")
    if (
        observation.draft_id != draft.draft_id
        or observation.draft_revision != draft.draft_revision
        or observation.materialized_schedule_id != str(schedule.request_id)
        or observation.materialized_schedule_revision != draft.draft_revision
        or observation.cost_draft_id != str(cost.request_id)
        or observation.cost_draft_revision != draft.draft_revision
    ):
        raise PlannerGuardError("planner_finalize_validation_stale")
    if not allow_unresolved and workspace.unresolved_decisions:
        raise PlannerGuardError("planner_finalize_has_unresolved_decisions")
    if (
        not allow_unresolved
        and workspace.active_interaction is not None
        and workspace.active_interaction.status is InteractionStatus.ACTIVE
    ):
        raise PlannerGuardError("planner_finalize_active_interaction")
    if (
        not allow_unresolved
        and workspace.decision_trace
        and isinstance(
            workspace.decision_trace[-1].payload,
            RequestEvidencePayload,
        )
    ):
        requested = {
            request.request_id
            for request in workspace.decision_trace[-1].payload.capability_requests
        }
        observed = {item.request_id for item in workspace.capability_observations}
        if not requested <= observed:
            raise PlannerGuardError("planner_finalize_pending_evidence_request")

    final_refs = FinalizeArtifactRefs(
        draft_id=draft.draft_id,
        draft_revision=draft.draft_revision,
        draft_content_digest=draft.content_digest,
        materialized_schedule_id=str(schedule.request_id),
        materialized_schedule_revision=draft.draft_revision,
        cost_draft_id=str(cost.request_id),
        cost_draft_revision=draft.draft_revision,
        validation_observation_id=observation.observation_id,
        validation_fingerprint=observation.validation_fingerprint,
    )
    incomplete_quality = any(
        any(ref.startswith("internal_schedule_quality:") for ref in issue.violated_constraint_refs)
        for issue in observation.issues
    )
    return PlannerDecision(
        decision_id=server_id(
            workspace.generation_id,
            observation.validation_fingerprint,
            "finalize",
        ),
        scope=workspace.current_scope,
        action="propose_finalize",
        current_goal="输出当前完整行程，保留检查问题和未知信息。"
        if allow_unresolved
        else "发布已核验的尽力完成版。"
        if incomplete_quality
        else "发布已经完成确定性校验的正式行程。",
        reason_summary=(
            "行程输出与检查结论分别记录，未通过的检查不阻止交付已有安排。"
            if allow_unresolved
            else "硬校验通过，但仍有逐日游览或正餐缺口；保留可用安排，不将安全通过等同于质量完成。"
            if incomplete_quality
            else "所有正式引用由服务端从当前草稿、时间轴、费用和校验结果绑定。"
        ),
        input_refs=PlannerInputRefs(
            strategy_revision=strategy.strategy_revision,
            candidate_pool_revision=workspace.candidate_pool.revision,
            draft_revision=draft.draft_revision,
            validation_observation_id=observation.observation_id,
        ),
        completion_assessment=PlannerCompletionAssessment(
            ready_to_finalize=True,
            blocking_issue_ids=(),
        ),
        payload=ProposeFinalizePayload(final_refs=final_refs),
    )
