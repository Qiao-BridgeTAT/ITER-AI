"""Explicit user choices and delegated capacity tradeoffs retain original preferences."""

from backend.agent.planner.workspace import PlannerGuardError, advance, server_id
from backend.contracts.v4.enums import (
    AskUserReasonCode,
    CandidateEntityKind,
    CommitmentLevel,
    InteractionStatus,
    PlannerStatus,
)
from backend.contracts.v4.planner_draft import UnassignedIntent
from backend.contracts.v4.planner_evidence import PlannerInteractionAnswer
from backend.contracts.v4.planner_observations import PlannerInteraction, PlannerInteractionOption
from backend.contracts.v4.planner_refs import CandidateRef
from backend.contracts.v4.planner_workspace import PlannerWorkspaceState


def candidate_answer(
    workspace: PlannerWorkspaceState, entity_id: str
) -> PlannerInteractionAnswer | None:
    return next(
        (
            answer
            for answer in reversed(workspace.interaction_answers)
            if answer.semantic_action in {"keep_required_candidate", "omit_required_candidate"}
            and answer.affected_refs == (entity_id,)
        ),
        None,
    )


def authorized_omission(workspace: PlannerWorkspaceState, intent: UnassignedIntent) -> bool:
    answer = candidate_answer(workspace, intent.candidate_ref.canonical_entity_id)
    if (
        answer
        and answer.semantic_action == "omit_required_candidate"
        and intent.reason_code == "user_requested"
        and intent.supporting_observation_refs == (answer.answer_id,)
        and not intent.requires_user_resolution
    ):
        return True
    entry = workspace.candidate_pool.candidate_by_id().get(intent.candidate_ref.candidate_id)
    return bool(
        workspace.react_state
        and workspace.planning_strategy
        and entry
        and entry.candidate_ref == intent.candidate_ref
        and entry.entity_kind in {CandidateEntityKind.ATTRACTION, CandidateEntityKind.RESTAURANT}
        and entry.commitment_level in {CommitmentLevel.STRONG, CommitmentLevel.SOFT}
        and entry.selection_permission != "forbidden"
        and intent.commitment_level == entry.commitment_level.value
        and intent.reason_code == "capacity_conflict"
        and intent.planner_reason
        and not intent.requires_user_resolution
        and intent.supporting_observation_refs == (workspace.planning_strategy.strategy_id,)
        and not (answer and answer.semantic_action == "keep_required_candidate")
    )


def capacity_omission(
    workspace: PlannerWorkspaceState, ref: CandidateRef, reason: str
) -> UnassignedIntent:
    """Bind an Agent's explicit choice to the current policy, never a fake user answer."""
    entry = workspace.candidate_pool.candidate_by_id().get(ref.candidate_id)
    if (
        workspace.react_state is None
        or workspace.planning_strategy is None
        or entry is None
        or entry.candidate_ref != ref
        or entry.entity_kind not in {CandidateEntityKind.ATTRACTION, CandidateEntityKind.RESTAURANT}
        or entry.commitment_level not in {CommitmentLevel.STRONG, CommitmentLevel.SOFT}
        or entry.selection_permission == "forbidden"
    ):
        raise PlannerGuardError("capacity_tradeoff_requires_current_place_intent")
    answer = candidate_answer(workspace, ref.canonical_entity_id)
    if answer and answer.semantic_action == "keep_required_candidate":
        raise PlannerGuardError("capacity_tradeoff_conflicts_with_explicit_keep")
    return UnassignedIntent(
        candidate_ref=ref,
        commitment_level="strong" if entry.commitment_level is CommitmentLevel.STRONG else "soft",
        reason_code="capacity_conflict",
        supporting_observation_refs=(workspace.planning_strategy.strategy_id,),
        requires_user_resolution=False,
        planner_reason=reason,
    )


def ask_required_candidate(
    workspace: PlannerWorkspaceState, ref: CandidateRef, reason: str
) -> PlannerWorkspaceState:
    entry = workspace.candidate_pool.candidate_by_id().get(ref.candidate_id)
    if entry is None or entry.commitment_level is not CommitmentLevel.STRONG:
        raise PlannerGuardError("ask_user_requires_required_candidate:想去和顺路去可自行取舍")
    if candidate_answer(workspace, ref.canonical_entity_id) is not None:
        raise PlannerGuardError("candidate_tradeoff_already_answered:请按用户已作出的选择继续规划")
    revision = workspace.workspace_revision + 1
    identifier = server_id(workspace.generation_id, ref.canonical_entity_id, revision, "tradeoff")
    scope = workspace.current_scope.model_copy(update={"workspace_revision": revision})
    question = f"{entry.display_name[:60]}：{reason} 这次仍要安排吗？"
    interaction = PlannerInteraction(
        interaction_id=identifier,
        scope=scope,
        reason_code=AskUserReasonCode.MATERIAL_TRADEOFF_OUTSIDE_DELEGATION,
        question=question,
        issue_ids=(identifier,),
        decision_scope="item",
        option_contracts=tuple(
            PlannerInteractionOption(
                option_id=server_id(identifier, action),
                semantic_action=action,
                affected_refs=(ref.canonical_entity_id,),
                verified_impact_summary=label,
            )
            for action, label in (
                ("keep_required_candidate", "仍要去，调整其他安排"),
                ("omit_required_candidate", "这次不去，选择更合适的地点"),
            )
        ),
        allow_free_text=False,
        based_on_workspace_revision=revision,
        resume_token=server_id(identifier, "resume"),
        status=InteractionStatus.ACTIVE,
    )
    return advance(
        workspace,
        active_interaction=interaction,
        unresolved_decisions=(identifier,),
        user_interrupt_count=workspace.user_interrupt_count + 1,
        status=PlannerStatus.AWAITING_USER,
    )
