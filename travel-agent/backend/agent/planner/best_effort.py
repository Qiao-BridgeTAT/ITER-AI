"""Bounded safety recovery of an existing real Planner blueprint, not a new planner.

No model or Provider facts are manufactured here. Unusable candidate visits can
be left unassigned with a receipt; fixed commitments and non-time safety errors
are never removed or downgraded. Every resulting artifact is revalidated.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from backend.agent.model_gateway import ModelCancellation
from backend.agent.planner.daily_repair import execution_remaining, refresh_repair_state
from backend.agent.planner.finalization import build_finalize_decision
from backend.agent.planner.guards import validate_working_draft
from backend.agent.planner.materializer import PlannerDraftMaterializer
from backend.agent.planner.timing_quality import (
    missing_concrete_meals,
    schedule_coverage_issues,
    schedule_quality_gaps,
)
from backend.agent.planner.validator import PlannerDraftValidator
from backend.agent.planner.workspace import PlannerGuardError, advance
from backend.contracts.v4.enums import InteractionStatus, PlannerStatus
from backend.contracts.v4.planner_draft import (
    UnassignedIntent,
    WorkingItineraryDraft,
    planning_projection_digest,
)
from backend.contracts.v4.planner_refs import CandidateRef
from backend.contracts.v4.planner_workspace import PlannerWorkspaceState
from backend.contracts.v4.task_book import TaskBookV4


async def recover_best_effort(
    workspace: PlannerWorkspaceState,
    book: TaskBookV4,
    cancellation: ModelCancellation,
    *,
    materializer: PlannerDraftMaterializer,
    validator: PlannerDraftValidator,
    checkpoint: Callable[[PlannerWorkspaceState], Awaitable[None]],
    input_state_version: int,
    refresh_evidence: Callable[[PlannerWorkspaceState], Awaitable[PlannerWorkspaceState]]
    | None = None,
    repair_schedule: Callable[[PlannerWorkspaceState], Awaitable[PlannerWorkspaceState]]
    | None = None,
) -> PlannerWorkspaceState:
    original = workspace
    draft = workspace.working_itinerary
    if (
        draft is None
        or not workspace.place_evidence
        or workspace.status in {PlannerStatus.CANCELLED, PlannerStatus.STALE}
        or (
            workspace.readiness_observation
            and any(
                issue.user_authority_required and issue.code != "strong_opening_conflict"
                for issue in workspace.readiness_observation.issues
            )
        )
    ):
        return original
    reasons = list(workspace.best_effort_reasons)
    reasons.append("本版保留已完成的真实规划，部分要求或优化尚未完成。")
    workspace = advance(
        workspace,
        status=PlannerStatus.PLANNING,
        best_effort_reasons=tuple(dict.fromkeys(reasons)),
        timing_optimization_pending=False,
        unresolved_decisions=(),
        active_interaction=workspace.active_interaction.model_copy(
            update={"status": InteractionStatus.SUPERSEDED}
        )
        if workspace.active_interaction
        else None,
        materialized_schedule=None,
        cost_draft=None,
        validation_report=None,
        validation_observation=None,
    )
    # Each failed pass must remove a concrete unusable candidate. No retry of
    # an unchanged semantic output and no fresh unbounded model/tool loop.
    repaired_digests: set[str] = set()
    for _ in range(min(17, sum(len(day.ordered_items) for day in draft.days) + 2)):
        cancellation.raise_if_cancelled("planner_best_effort")
        if refresh_evidence is not None:
            # Removal changes adjacency, including hotel->first and last->hotel.
            # Every new pair must be queried, even if the old plan had no errors.
            workspace = await refresh_evidence(workspace)
        materialized = await materializer.materialize(
            workspace, book, cancellation, input_state_version=input_state_version
        )
        workspace = advance(
            workspace, materialized_schedule=materialized.schedule, cost_draft=materialized.cost
        )
        validated = await validator.validate(workspace, book, cancellation)
        workspace = advance(
            workspace,
            validation_report=validated.legacy_report,
            validation_observation=validated.observation,
        )
        errors = [i for i in validated.observation.issues if i.severity != "warning"]
        if not errors:
            assert workspace.working_itinerary is not None
            digest = workspace.working_itinerary.content_digest
            if (
                repair_schedule is not None
                and digest not in repaired_digests
                and execution_remaining(workspace, calls=True) > 8
            ):
                repaired_digests.add(digest)
                workspace = await repair_schedule(workspace)
                if workspace.working_itinerary is not None:
                    repaired_digests.add(workspace.working_itinerary.content_digest)
                # The callback may select different items. Always re-query the
                # resulting adjacency and run materialization/validation again.
                continue
            assert workspace.working_itinerary is not None
            if not any(day.activities for day in materialized.schedule.days):
                return original
            names = workspace.candidate_pool.candidate_by_id()
            unmet = [
                f"未安排：{names[i.candidate_ref.candidate_id].display_name}；原要求保留。"
                for i in workspace.working_itinerary.unassigned_intents
            ]
            unmet.extend(
                f"{gap['date']} {gap['start']}–{gap['end']}仍有约{gap['minutes']}分钟未安排。"
                for gap in schedule_quality_gaps(workspace, book)
            )
            unmet.extend(
                f"{issue['date']}的{'上午' if issue['period'] == 'morning' else '下午'}"
                "游览尚未补齐。"
                for issue in schedule_coverage_issues(workspace, book)
            )
            unmet.extend(
                f"{meal['date']}的{'午餐' if meal['meal'] == 'lunch' else '晚餐'}"
                "尚未核验到合适的具体餐厅。"
                for meal in missing_concrete_meals(workspace)
            )
            workspace = advance(
                workspace,
                best_effort_reasons=tuple(dict.fromkeys((*workspace.best_effort_reasons, *unmet))),
            )
            try:
                assert workspace.working_itinerary is not None
                validate_working_draft(
                    workspace.working_itinerary, workspace, book, validated.observation.checked_at
                )
            except PlannerGuardError:
                return original
            workspace = refresh_repair_state(workspace, book, stop_reason="publication")
            decision = build_finalize_decision(workspace)
            workspace = advance(
                workspace,
                decision_trace=(*workspace.decision_trace, decision),
                status=PlannerStatus.READY_TO_PUBLISH,
            )
            await checkpoint(workspace)
            return workspace
        if any(
            i.code not in {"opening_conflict", "meal_constraint_violation", "time_overlap"}
            for i in errors
        ):
            return original
        assert workspace.working_itinerary is not None
        draft = workspace.working_itinerary
        items = {i.draft_item_id: i for d in draft.days for i in d.ordered_items}
        remove_ids: set[str] = set()
        # Specific item failures identify the cause. A day-wide overflow also
        # references all that day's items for context, not permission to delete
        # the whole day. Apply the smallest causal removal and remeasure first.
        for issue in sorted(errors, key=lambda value: value.scope_kind == "day"):
            targets = [items[key] for key in issue.draft_item_ids if key in items]
            if issue.code == "time_overlap" and issue.scope_kind == "day":
                if remove_ids.intersection(issue.draft_item_ids):
                    continue
                targets = [
                    item
                    for item in reversed(targets)
                    if isinstance(item.object_ref, CandidateRef)
                    and item.commitment_level != "immutable"
                    and item.item_kind == "visit"
                ][:1]
            elif issue.code == "meal_constraint_violation":
                # Keep the real meal and remove only a preceding flexible visit
                # causing its late arrival, rather than hiding a missing meal.
                targets = [
                    candidate
                    for day in draft.days
                    if day.service_date in issue.affected_dates
                    for candidate in reversed(day.ordered_items)
                    if candidate.item_kind == "visit"
                    and (not targets or candidate.position < targets[0].position)
                ][:1]
            if not targets or any(
                not isinstance(i.object_ref, CandidateRef) or i.commitment_level == "immutable"
                for i in targets
            ):
                return original
            remove_ids.update(i.draft_item_id for i in targets)
        if not remove_ids:
            return original
        unassigned = list(draft.unassigned_intents)
        recovered_omissions = list(workspace.recovery_omissions)
        for key in remove_ids:
            item = items[key]
            assert isinstance(item.object_ref, CandidateRef)
            names = workspace.candidate_pool.candidate_by_id()
            reasons.append(
                f"未安排：{names[item.object_ref.candidate_id].display_name}；当前时间无法安全容纳。"
            )
            if item.commitment_level in {"strong", "soft"}:
                omission = UnassignedIntent(
                    candidate_ref=item.object_ref,
                    commitment_level="strong" if item.commitment_level == "strong" else "soft",
                    reason_code="awaiting_user",
                    supporting_observation_refs=(validated.observation.observation_id,),
                    requires_user_resolution=item.commitment_level == "strong",
                )
                unassigned.append(omission)
                recovered_omissions.append(omission)
        revised = draft.model_copy(
            update={
                "draft_revision": draft.draft_revision + 1,
                "days": tuple(
                    day.model_copy(
                        update={
                            # Compatibility metadata only; never add a REST
                            # activity when the last unsafe item was removed.
                            "day_kind": day.day_kind
                            if any(i.draft_item_id not in remove_ids for i in day.ordered_items)
                            else "rest",
                            "primary_cluster_id": day.primary_cluster_id
                            if any(i.draft_item_id not in remove_ids for i in day.ordered_items)
                            else None,
                            "ordered_items": tuple(
                                item.model_copy(update={"position": position})
                                for position, item in enumerate(
                                    i
                                    for i in day.ordered_items
                                    if i.draft_item_id not in remove_ids
                                )
                            ),
                            "cross_cluster_segments": (),
                        }
                    )
                    for day in draft.days
                ),
                "unassigned_intents": tuple(unassigned),
                "discardable_objects": tuple(
                    i for i in draft.discardable_objects if i.draft_item_id not in remove_ids
                ),
            }
        )
        revised = WorkingItineraryDraft.model_validate(
            {
                **revised.model_dump(mode="json"),
                "content_digest": planning_projection_digest(revised),
            }
        )
        workspace = advance(
            workspace,
            working_itinerary=revised,
            recovery_observations=(*workspace.recovery_observations, validated.observation),
            recovery_omissions=tuple(recovered_omissions),
            best_effort_reasons=tuple(dict.fromkeys(reasons)),
            materialized_schedule=None,
            cost_draft=None,
            validation_report=None,
            validation_observation=None,
        )
    return original
