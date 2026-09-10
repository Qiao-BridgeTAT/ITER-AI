"""Compile a thin model repair intent into one authoritative Planner decision."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal, cast

from backend.agent.planner.decision_contracts import (
    ModelAskUserRepairChoice,
    ModelChangeHotelRepairChoice,
    ModelChangeTransportRepairChoice,
    ModelChangeWindowRepairChoice,
    ModelMoveRepairChoice,
    ModelOmitSoftRepairChoice,
    ModelPlanChangePatchIntent,
    ModelReorderRepairChoice,
    ModelRepairIntent,
    ModelReplaceRepairChoice,
    ModelRequestEvidenceRepairChoice,
)
from backend.agent.planner.proposals import PlannerReferenceCatalog
from backend.agent.planner.workspace import PlannerGuardError, server_id
from backend.contracts.v4.enums import AskUserReasonCode, InteractionStatus
from backend.contracts.v4.plan_change import (
    PlanChangeRequest,
    SelectedHotelRecommendation,
    validate_selected_hotel_against_observation,
)
from backend.contracts.v4.planner_decision import (
    AskUserPayload,
    PlannerCompletionAssessment,
    PlannerDecision,
    PlannerInputRefs,
    PlannerResumeContract,
    RequestEvidencePayload,
    ReviseDraftPayload,
)
from backend.contracts.v4.planner_draft import DraftItem, LodgingBaseline, UnassignedIntent
from backend.contracts.v4.planner_observations import (
    PlannerCapabilityRequest,
    PlannerInteraction,
    PlannerInteractionOption,
    PlannerValidationIssue,
)
from backend.contracts.v4.planner_patch import (
    ItineraryPatch,
    ItineraryPatchOperation,
    MoveItemOperation,
    PlanChangeRequestPatchAuthority,
    RemoveItemOperation,
    ReorderItemOperation,
    ReplaceItemOperation,
    RouteEndpointRef,
    SetHotelBaselineOperation,
    SetPreferredWindowOperation,
    SetTransportPreferenceOperation,
    ValidationIssuePatchAuthority,
    atomic_apply_itinerary_patch,
)
from backend.contracts.v4.planner_refs import CandidateRef, PlannerScope, planner_object_ref_key
from backend.contracts.v4.planner_workspace import PlannerWorkspaceState
from backend.persistence.outbox_repository import canonical_json_hash


@dataclass(frozen=True)
class DraftItemLocation:
    key: str
    service_date: date
    item: DraftItem


def compile_model_repair_intent(
    intent: ModelRepairIntent,
    workspace: PlannerWorkspaceState,
) -> PlannerDecision:
    """Bind semantic repair choices to the current validation authority."""

    draft = workspace.working_itinerary
    observation = workspace.validation_observation
    strategy = workspace.planning_strategy
    if draft is None or observation is None or strategy is None:
        raise PlannerGuardError("planner_repair_prerequisites_missing")
    if observation.draft_id != draft.draft_id or observation.draft_revision != draft.draft_revision:
        raise PlannerGuardError("planner_repair_validation_stale")
    if observation.result == "passed":
        raise PlannerGuardError("planner_repair_not_allowed_after_passed_validation")

    catalog = PlannerReferenceCatalog(workspace)
    issue_catalog = catalog.validation_issues
    try:
        issues = tuple(issue_catalog[key] for key in intent.issue_keys)
    except KeyError as error:
        raise PlannerGuardError("planner_unknown_validation_issue_key") from error
    choice = intent.choice
    required_action = _required_issue_action(choice.operation)
    if any(required_action not in issue.allowed_actions for issue in issues):
        raise PlannerGuardError("planner_repair_operation_not_authorized")

    if isinstance(choice, ModelRequestEvidenceRepairChoice):
        return _compile_evidence_decision(intent, choice, workspace, issues, catalog)
    if isinstance(choice, ModelAskUserRepairChoice):
        return _compile_ask_user_decision(intent, workspace, issues)

    issue_ids = tuple(issue.issue_id for issue in issues)
    locations = _draft_locations(workspace, catalog)
    target_key = getattr(choice, "target_key", None)
    if isinstance(target_key, str):
        target = _location(target_key, locations)
        affected_items = {item_id for issue in issues for item_id in issue.draft_item_ids}
        if affected_items and target.item.draft_item_id not in affected_items:
            allowed = tuple(
                key
                for key, location in locations.items()
                if location.item.draft_item_id in affected_items
            )
            raise PlannerGuardError(
                "planner_repair_target_outside_current_issue:path=choice.target_key"
                f":allowed_values={','.join(allowed)}"
            )
    operation = _compile_patch_operation(
        choice,
        workspace=workspace,
        catalog=catalog,
        locations=locations,
        authority_refs=issue_ids,
        issues=issues,
    )
    affected_dates = _affected_dates(choice, locations, workspace)
    selected_hotel = (
        _compile_selected_hotel(
            workspace,
            output_scope=workspace.current_scope.model_copy(
                update={"workspace_revision": workspace.workspace_revision + 1}
            ),
            hotel_offer_key=choice.hotel_offer_key,
            reason_summary=intent.reason_summary,
        )
        if isinstance(choice, ModelChangeHotelRepairChoice)
        else None
    )
    patch = ItineraryPatch(
        patch_id=server_id(
            workspace.generation_id,
            "patch",
            draft.draft_revision,
            canonical_json_hash(intent.model_dump(mode="json")),
        ),
        scope=workspace.current_scope,
        base_draft_id=draft.draft_id,
        base_draft_revision=draft.draft_revision,
        base_content_digest=draft.content_digest,
        authority=ValidationIssuePatchAuthority(
            reference_id=observation.observation_id,
            issue_ids=issue_ids,
        ),
        operations=(operation,),
        declared_affected_dates=affected_dates,
        reason_summary=intent.reason_summary,
    )
    try:
        atomic_apply_itinerary_patch(
            draft,
            patch,
            candidate_pool=workspace.candidate_pool,
            allowed_authority_refs=frozenset(issue_ids),
        )
    except ValueError as error:
        raise PlannerGuardError(_safe_patch_error(error)) from error

    return PlannerDecision(
        decision_id=server_id(workspace.generation_id, "decision", patch.patch_id),
        scope=workspace.current_scope,
        action="revise_draft",
        current_goal="修复当前行程校验问题。",
        reason_summary=intent.reason_summary,
        input_refs=PlannerInputRefs(
            strategy_revision=strategy.strategy_revision,
            candidate_pool_revision=workspace.candidate_pool.revision,
            draft_revision=draft.draft_revision,
            validation_observation_id=observation.observation_id,
        ),
        completion_assessment=PlannerCompletionAssessment(
            ready_to_finalize=False,
            blocking_issue_ids=issue_ids,
        ),
        payload=ReviseDraftPayload(
            itinerary_patch=patch,
            selected_hotel=selected_hotel,
            declared_affected_dates=affected_dates,
        ),
    )


def compile_model_plan_change_intent(
    intent: ModelPlanChangePatchIntent,
    request: PlanChangeRequest,
    workspace: PlannerWorkspaceState,
) -> PlannerDecision:
    """Compile thin user-edit choices into one request-authorized atomic Patch."""

    draft = workspace.working_itinerary
    strategy = workspace.planning_strategy
    if draft is None or strategy is None:
        raise PlannerGuardError("planner_plan_change_prerequisites_missing")
    if request.requested_scope not in {"local_replan", "full_replan"}:
        raise PlannerGuardError("planner_plan_change_scope_not_patchable")
    if request.trip_id != workspace.trip_id:
        raise PlannerGuardError("planner_plan_change_trip_mismatch")
    authority_refs = tuple(request.semantic_operation_ids)
    if not authority_refs:
        raise PlannerGuardError("planner_plan_change_authority_missing")

    catalog = PlannerReferenceCatalog(workspace)
    locations = _draft_locations(workspace, catalog)
    operations = tuple(
        _compile_patch_operation(
            choice,
            workspace=workspace,
            catalog=catalog,
            locations=locations,
            authority_refs=authority_refs,
            issues=(),
            user_authority_ref=request.plan_change_request_id,
        )
        for choice in intent.choices
    )
    affected_dates = tuple(
        sorted(
            {
                service_date
                for choice in intent.choices
                for service_date in _affected_dates(choice, locations, workspace)
            }
        )
    )
    hotel_choices = tuple(
        choice for choice in intent.choices if isinstance(choice, ModelChangeHotelRepairChoice)
    )
    if len(hotel_choices) > 1:
        raise PlannerGuardError("planner_plan_change_allows_one_hotel_choice")
    selected_hotel = (
        _compile_selected_hotel(
            workspace,
            output_scope=workspace.current_scope.model_copy(
                update={"workspace_revision": workspace.workspace_revision + 1}
            ),
            hotel_offer_key=hotel_choices[0].hotel_offer_key,
            reason_summary=intent.reason_summary,
        )
        if hotel_choices
        else None
    )
    patch = ItineraryPatch(
        patch_id=server_id(
            workspace.generation_id,
            request.plan_change_request_id,
            canonical_json_hash(intent.model_dump(mode="json")),
        ),
        scope=workspace.current_scope,
        base_draft_id=draft.draft_id,
        base_draft_revision=draft.draft_revision,
        base_content_digest=draft.content_digest,
        authority=PlanChangeRequestPatchAuthority(
            reference_id=request.plan_change_request_id,
            user_message_or_answer_ref=request.user_message_id,
        ),
        operations=operations,
        declared_affected_dates=affected_dates,
        reason_summary=intent.reason_summary,
    )
    try:
        atomic_apply_itinerary_patch(
            draft,
            patch,
            candidate_pool=workspace.candidate_pool,
            allowed_authority_refs=frozenset(authority_refs),
        )
    except ValueError as error:
        raise PlannerGuardError(_safe_patch_error(error)) from error

    return PlannerDecision(
        decision_id=server_id(workspace.generation_id, "decision", patch.patch_id),
        scope=workspace.current_scope,
        action="revise_draft",
        current_goal="按用户要求修改当前正式行程。",
        reason_summary=intent.reason_summary,
        input_refs=PlannerInputRefs(
            strategy_revision=strategy.strategy_revision,
            candidate_pool_revision=workspace.candidate_pool.revision,
            draft_revision=draft.draft_revision,
            plan_change_request_id=request.plan_change_request_id,
        ),
        completion_assessment=PlannerCompletionAssessment(
            ready_to_finalize=False,
            blocking_issue_ids=(),
        ),
        payload=ReviseDraftPayload(
            itinerary_patch=patch,
            selected_hotel=selected_hotel,
            declared_affected_dates=affected_dates,
        ),
    )


def _compile_selected_hotel(
    workspace: PlannerWorkspaceState,
    *,
    output_scope: PlannerScope,
    hotel_offer_key: str,
    reason_summary: str,
) -> SelectedHotelRecommendation:
    """Bind one semantic h key to the current hotel observation."""

    observation = workspace.hotel_observation
    if observation is None or observation.mode != "search":
        raise PlannerGuardError("planner_repair_hotel_observation_missing")
    catalog = PlannerReferenceCatalog(workspace)
    offer = catalog.alternative_hotel(hotel_offer_key)
    if offer.availability_status == "unavailable":
        raise PlannerGuardError("planner_repair_hotel_offer_unavailable")
    durations = [
        commute.duration_minutes
        for commute in offer.commute_to_clusters
        if commute.duration_minutes is not None
    ]
    average_duration = round(sum(durations) / len(durations)) if durations else None
    route_fit = (
        f"已核验 {len(durations)} 个活动区域，平均通勤约 {average_duration} 分钟。"
        if durations
        else "当前通勤时长仍有缺口，将按主要活动区域安排并保留核验提示。"
    )
    if offer.availability_status == "unknown":
        tradeoff = "当前房型库存未知，作为规划住宿使用，预订前仍需核验实时房态。"
    elif offer.availability_status == "limited":
        tradeoff = "当前库存有限，预订前需刷新房型与价格。"
    elif offer.total_price is None:
        tradeoff = "当前可预订总价未知，预算中保留住宿价格缺口。"
    else:
        tradeoff = "最终体验仍取决于实际房型与到店情况。"
    selection = SelectedHotelRecommendation.model_validate(
        {
            "selection_id": server_id(
                workspace.generation_id,
                observation.hotel_observation_id,
                offer.offer_ref.offer_id,
                "selected-hotel",
            ),
            "scope": output_scope,
            "hotel_observation_id": observation.hotel_observation_id,
            "hotel_offer_ref": offer.offer_ref,
            "area_reason": "该酒店所在区域适合作为本次行程的唯一住宿基线。",
            "route_fit": route_fit,
            "selection_reason": reason_summary,
            "main_tradeoff": tradeoff,
        }
    )
    try:
        validate_selected_hotel_against_observation(selection, observation)
    except ValueError as error:
        raise PlannerGuardError("planner_repair_hotel_selection_invalid") from error
    return selection


def compile_validation_interaction(
    workspace: PlannerWorkspaceState,
) -> PlannerDecision:
    """Create the only legal user interrupt when Validator already owns the choice boundary."""

    observation = workspace.validation_observation
    if observation is None:
        raise PlannerGuardError("planner_repair_validation_missing")
    issues = tuple(
        issue
        for issue in observation.issues
        if issue.user_authority_required and "ask_user" in issue.allowed_actions
    )[:4]
    if not issues:
        raise PlannerGuardError("planner_repair_no_user_authority_issue")
    intent = ModelRepairIntent(
        issue_keys=tuple(f"v{index}" for index, _ in enumerate(issues, 1)),
        choice=ModelAskUserRepairChoice(),
        reason_summary="当前冲突会改变已确认的强意愿或固定安排，需要由用户选择。",
    )
    return _compile_ask_user_decision(intent, workspace, issues)


def _required_issue_action(operation: str) -> str:
    return {
        "move": "move_item",
        "reorder": "reorder_item",
        "replace": "replace_item",
        "omit_soft": "remove_item",
        "change_window": "change_window",
        "change_transport": "change_transport",
        "change_hotel": "change_hotel",
        "request_evidence": "request_evidence",
        "ask_user": "ask_user",
    }[operation]


def _draft_locations(
    workspace: PlannerWorkspaceState,
    catalog: PlannerReferenceCatalog,
) -> dict[str, DraftItemLocation]:
    assert workspace.working_itinerary is not None
    keys_by_ref = {
        planner_object_ref_key(item.candidate_ref): key for key, item in catalog.candidates.items()
    }
    keys_by_ref.update(
        {planner_object_ref_key(reference): key for key, reference in catalog.fixed.items()}
    )
    result: dict[str, DraftItemLocation] = {}
    for day in workspace.working_itinerary.days:
        for item in day.ordered_items:
            key = keys_by_ref.get(planner_object_ref_key(item.object_ref))
            if key is not None:
                result[key] = DraftItemLocation(key=key, service_date=day.service_date, item=item)
    return result


def _location(
    key: str,
    locations: dict[str, DraftItemLocation],
    *,
    field: str = "choice.target_key",
) -> DraftItemLocation:
    try:
        return locations[key]
    except KeyError as error:
        raise PlannerGuardError(
            f"planner_repair_target_not_in_current_draft:path={field}"
            f":allowed_values={','.join(locations)}"
        ) from error


def _placement(
    choice: ModelMoveRepairChoice | ModelReorderRepairChoice,
    *,
    target_date: date,
    target_item_id: str,
    locations: dict[str, DraftItemLocation],
) -> dict[str, object]:
    if choice.placement == "at_end":
        return {"at_end": True}
    assert choice.relative_to_key is not None
    anchor = _location(choice.relative_to_key, locations, field="choice.relative_to_key")
    if anchor.service_date != target_date or anchor.item.draft_item_id == target_item_id:
        raise PlannerGuardError("planner_repair_placement_anchor_invalid")
    return {
        "before_item_id" if choice.placement == "before" else "after_item_id": (
            anchor.item.draft_item_id
        )
    }


def _compile_patch_operation(
    choice: object,
    *,
    workspace: PlannerWorkspaceState,
    catalog: PlannerReferenceCatalog,
    locations: dict[str, DraftItemLocation],
    authority_refs: tuple[str, ...],
    issues: tuple[PlannerValidationIssue, ...],
    user_authority_ref: str | None = None,
) -> ItineraryPatchOperation:
    draft = workspace.working_itinerary
    assert draft is not None
    if isinstance(choice, ModelMoveRepairChoice):
        target = _location(choice.target_key, locations)
        if target.service_date == choice.destination_date:
            raise PlannerGuardError("planner_repair_same_day_move_requires_reorder")
        if target.item.commitment_level == "immutable":
            raise PlannerGuardError("planner_repair_cannot_move_immutable_item")
        destination_day = next(
            (day for day in draft.days if day.service_date == choice.destination_date), None
        )
        if destination_day is None:
            raise PlannerGuardError("planner_repair_destination_date_invalid")
        return MoveItemOperation.model_validate(
            {
                "draft_item_id": target.item.draft_item_id,
                "from_date": target.service_date,
                "to_date": choice.destination_date,
                "authority_item_refs": authority_refs,
                **_placement(
                    choice,
                    target_date=choice.destination_date,
                    target_item_id=target.item.draft_item_id,
                    locations=locations,
                ),
            }
        )
    if isinstance(choice, ModelReorderRepairChoice):
        target = _location(choice.target_key, locations)
        return ReorderItemOperation.model_validate(
            {
                "service_date": target.service_date,
                "draft_item_id": target.item.draft_item_id,
                "authority_item_refs": authority_refs,
                **_placement(
                    choice,
                    target_date=target.service_date,
                    target_item_id=target.item.draft_item_id,
                    locations=locations,
                ),
            }
        )
    if isinstance(choice, ModelReplaceRepairChoice):
        target = _location(choice.target_key, locations)
        if not isinstance(target.item.object_ref, CandidateRef):
            raise PlannerGuardError("planner_repair_cannot_replace_immutable_item")
        replacement = catalog.candidate(
            choice.replacement_key,
            field="repair.choice.replacement_key",
        )
        if choice.replacement_key in locations:
            raise PlannerGuardError("planner_repair_replacement_already_scheduled")
        if any(item.candidate_ref == replacement for item in draft.unassigned_intents):
            raise PlannerGuardError("planner_repair_replacement_is_unassigned_intent")
        if replacement.entity_kind != target.item.object_ref.entity_kind:
            raise PlannerGuardError("planner_repair_replacement_kind_mismatch")
        return ReplaceItemOperation(
            draft_item_id=target.item.draft_item_id,
            expected_old_ref=target.item.object_ref,
            new_candidate_ref=replacement,
            authority_item_refs=authority_refs,
            unscheduled_intent_record=_unassigned_record(
                target,
                workspace,
                issues,
                user_authority_ref=user_authority_ref,
            ),
        )
    if isinstance(choice, ModelOmitSoftRepairChoice):
        target = _location(choice.target_key, locations)
        if target.item.commitment_level not in {"soft", "filler", "neutral"} or not isinstance(
            target.item.object_ref, CandidateRef
        ):
            raise PlannerGuardError("planner_repair_omit_requires_optional_candidate")
        return RemoveItemOperation(
            draft_item_id=target.item.draft_item_id,
            expected_object_ref=target.item.object_ref,
            authority_item_refs=authority_refs,
            removal_reason_code=(
                "user_requested" if user_authority_ref else _removal_reason(issues)
            ),
            unscheduled_intent_record=_unassigned_record(
                target,
                workspace,
                issues,
                user_authority_ref=user_authority_ref,
            ),
        )
    if isinstance(choice, ModelChangeWindowRepairChoice):
        target = _location(choice.target_key, locations)
        return SetPreferredWindowOperation(
            draft_item_id=target.item.draft_item_id,
            expected_window=choice.preferred_window,
            authority_item_refs=authority_refs,
        )
    if isinstance(choice, ModelChangeTransportRepairChoice):
        origin = _location(choice.from_key, locations, field="choice.from_key")
        destination_item = _location(choice.to_key, locations, field="choice.to_key")
        if (
            origin.service_date != choice.service_date
            or destination_item.service_date != choice.service_date
            or origin.item.draft_item_id == destination_item.item.draft_item_id
        ):
            raise PlannerGuardError("planner_repair_transport_endpoints_invalid")
        allowed = set(workspace.planning_strategy.spatial_policy.preferred_transport_modes)  # type: ignore[union-attr]
        if not set(choice.transport_preferences) <= allowed:
            raise PlannerGuardError("planner_repair_transport_not_in_strategy")
        return SetTransportPreferenceOperation(
            service_date=choice.service_date,
            from_endpoint=RouteEndpointRef(
                kind="draft_item", reference_id=origin.item.draft_item_id
            ),
            to_endpoint=RouteEndpointRef(
                kind="draft_item", reference_id=destination_item.item.draft_item_id
            ),
            allowed_transport_modes=choice.transport_preferences,
            authority_item_refs=authority_refs,
        )
    if isinstance(choice, ModelChangeHotelRepairChoice):
        offer = catalog.alternative_hotel(choice.hotel_offer_key)
        if offer.availability_status == "unavailable":
            raise PlannerGuardError("planner_repair_hotel_offer_unavailable")
        baseline = draft.lodging_baseline
        if baseline.mode != "selected_offer":
            raise PlannerGuardError("planner_repair_hotel_baseline_not_replaceable")
        return SetHotelBaselineOperation(
            expected_old_baseline=baseline,
            new_baseline=LodgingBaseline(mode="selected_offer", selected_offer_ref=offer.offer_ref),
            authority_item_refs=authority_refs,
        )
    raise PlannerGuardError("planner_repair_choice_not_patchable")


def _affected_dates(
    choice: object,
    locations: dict[str, DraftItemLocation],
    workspace: PlannerWorkspaceState,
) -> tuple[date, ...]:
    draft = workspace.working_itinerary
    assert draft is not None
    if isinstance(choice, ModelMoveRepairChoice):
        return tuple(
            sorted({_location(choice.target_key, locations).service_date, choice.destination_date})
        )
    if isinstance(choice, ModelChangeHotelRepairChoice):
        return tuple(day.service_date for day in draft.days)
    if isinstance(choice, ModelChangeTransportRepairChoice):
        return (choice.service_date,)
    target_key = cast(
        ModelReorderRepairChoice
        | ModelReplaceRepairChoice
        | ModelOmitSoftRepairChoice
        | ModelChangeWindowRepairChoice,
        choice,
    ).target_key
    return (_location(target_key, locations).service_date,)


def _unassigned_record(
    target: DraftItemLocation,
    workspace: PlannerWorkspaceState,
    issues: tuple[PlannerValidationIssue, ...],
    *,
    user_authority_ref: str | None = None,
) -> UnassignedIntent | None:
    if target.item.commitment_level != "soft":
        return None
    if not isinstance(target.item.object_ref, CandidateRef):
        raise PlannerGuardError("planner_repair_soft_target_reference_invalid")
    if user_authority_ref is None and workspace.validation_observation is None:
        raise PlannerGuardError("planner_repair_validation_missing")
    return UnassignedIntent(
        candidate_ref=target.item.object_ref,
        commitment_level="soft",
        reason_code=("user_requested" if user_authority_ref else _removal_reason(issues)),
        supporting_observation_refs=(
            (user_authority_ref,)
            if user_authority_ref is not None
            else (workspace.validation_observation.observation_id,)  # type: ignore[union-attr]
        ),
        requires_user_resolution=False,
    )


def _removal_reason(
    issues: tuple[PlannerValidationIssue, ...],
) -> Literal[
    "capacity_conflict",
    "route_conflict",
    "opening_conflict",
    "budget_conflict",
    "duplicate_experience",
]:
    codes = {item.code for item in issues}
    if codes & {"pace_limit_exceeded", "time_overlap", "meal_gap", "meal_constraint_violation"}:
        return "capacity_conflict"
    if codes & {"route_unavailable", "route_cost_exceeded", "walking_limit_exceeded"}:
        return "route_conflict"
    if "opening_conflict" in codes:
        return "opening_conflict"
    if codes & {"budget_exceeded", "missing_price", "hotel_constraint_conflict"}:
        return "budget_conflict"
    return "duplicate_experience"


def _compile_evidence_decision(
    intent: ModelRepairIntent,
    choice: ModelRequestEvidenceRepairChoice,
    workspace: PlannerWorkspaceState,
    issues: tuple[PlannerValidationIssue, ...],
    catalog: PlannerReferenceCatalog,
) -> PlannerDecision:
    issue_ids = tuple(item.issue_id for item in issues)
    decision_id = server_id(
        workspace.generation_id,
        "decision",
        canonical_json_hash(intent.model_dump(mode="json")),
    )
    requests = []
    for index, proposed in enumerate(choice.requests):
        if proposed.purpose != "resolve_validation_issue" or tuple(
            proposed.based_on_issue_keys
        ) != tuple(intent.issue_keys):
            raise PlannerGuardError("planner_repair_evidence_issue_keys_mismatch")
        data = proposed.arguments.model_dump(mode="json")
        capability = data.pop("capability")
        _resolve_argument_keys(data, catalog)
        requests.append(
            PlannerCapabilityRequest.model_validate(
                {
                    "request_id": server_id(decision_id, proposed.local_key, index),
                    "scope": workspace.current_scope,
                    "capability": capability,
                    "purpose": "resolve_validation_issue",
                    "blocking": proposed.blocking,
                    "based_on_issue_ids": issue_ids,
                    "service_dates": data.get("service_dates", ()),
                    "arguments": data,
                }
            )
        )
    return PlannerDecision(
        decision_id=decision_id,
        scope=workspace.current_scope,
        action="request_evidence",
        current_goal="补齐当前校验问题缺少的真实证据。",
        reason_summary=intent.reason_summary,
        input_refs=_repair_input_refs(workspace),
        completion_assessment=PlannerCompletionAssessment(
            ready_to_finalize=False, blocking_issue_ids=issue_ids
        ),
        payload=RequestEvidencePayload(
            capability_requests=tuple(requests),
            resume_goal="获得新 Observation 后重新物化并完整校验。",
        ),
    )


def _resolve_argument_keys(data: dict[str, object], catalog: PlannerReferenceCatalog) -> None:
    for field, target in (
        ("candidate_keys", "candidate_refs"),
        ("nearby_candidate_keys", "nearby_candidate_refs"),
    ):
        if field in data:
            values = cast(list[str], data.pop(field))
            data[target] = [
                catalog.candidate(key, field=f"repair.requests.arguments.{field}").model_dump(
                    mode="json"
                )
                for key in values
            ]
    for field, target in (
        ("nearby_cluster_keys", "nearby_cluster_refs"),
        ("activity_cluster_keys", "activity_cluster_refs"),
    ):
        if field in data:
            values = cast(list[str], data.pop(field))
            data[target] = [catalog.cluster_id(key) for key in values]
    if "offer_key" in data:
        offer = catalog.hotel(
            cast(str, data.pop("offer_key")), field="repair.requests.arguments.offer_key"
        )
        data["offer_ref"] = offer.offer_ref.model_dump(mode="json")
    if "endpoint_pairs" in data:
        pairs = cast(list[dict[str, object]], data["endpoint_pairs"])
        data["endpoint_pairs"] = [
            {
                "origin": catalog.endpoint(cast(str, pair["origin_key"])).model_dump(mode="json"),
                "destination": catalog.endpoint(cast(str, pair["destination_key"])).model_dump(
                    mode="json"
                ),
            }
            for pair in pairs
        ]
    comparison = data.get("comparison")
    if isinstance(comparison, dict):
        for field in ("baseline_days", "proposed_days"):
            days = cast(list[dict[str, object]], comparison[field])
            for day in days:
                keys = cast(list[str], day.pop("ordered_endpoint_keys"))
                day["ordered_endpoints"] = [
                    catalog.endpoint(key).model_dump(mode="json") for key in keys
                ]


def _compile_ask_user_decision(
    intent: ModelRepairIntent,
    workspace: PlannerWorkspaceState,
    issues: tuple[PlannerValidationIssue, ...],
) -> PlannerDecision:
    if any(not issue.user_authority_required for issue in issues):
        raise PlannerGuardError("planner_repair_ask_user_not_authorized")
    assert workspace.validation_observation is not None
    issue_ids = tuple(issue.issue_id for issue in issues)
    output_scope = workspace.current_scope.model_copy(
        update={"workspace_revision": workspace.workspace_revision + 1}
    )
    interaction_id = server_id(
        workspace.generation_id,
        "interaction",
        workspace.validation_observation.observation_id,
        *issue_ids,
    )
    interaction = PlannerInteraction(
        interaction_id=interaction_id,
        scope=output_scope,
        reason_code=_ask_reason(issues),
        issue_ids=issue_ids,
        decision_scope="global",
        affected_dates=tuple(sorted({day for issue in issues for day in issue.affected_dates})),
        option_contracts=(
            PlannerInteractionOption(
                option_id=server_id(interaction_id, "keep"),
                semantic_action="keep_task_book",
                affected_refs=issue_ids,
                verified_impact_summary="保留已确认的强承诺，由系统继续调整其余可修改安排。",
            ),
            PlannerInteractionOption(
                option_id=server_id(interaction_id, "revise"),
                semantic_action="revise_task_book",
                affected_refs=issue_ids,
                verified_impact_summary="修改相关强承诺或硬限制，并重新确认任务书。",
            ),
        ),
        allow_free_text=True,
        based_on_workspace_revision=output_scope.workspace_revision,
        resume_token=server_id(interaction_id, "resume"),
        status=InteractionStatus.ACTIVE,
    )
    return PlannerDecision(
        decision_id=server_id(interaction_id, "decision"),
        scope=workspace.current_scope,
        action="ask_user",
        current_goal="请求用户处理不能由系统擅自改变的强承诺冲突。",
        reason_summary=intent.reason_summary,
        input_refs=_repair_input_refs(workspace),
        completion_assessment=PlannerCompletionAssessment(
            ready_to_finalize=False, blocking_issue_ids=issue_ids
        ),
        payload=AskUserPayload(
            user_decision_request=interaction,
            blocking_issue_ids=issue_ids,
            resume_contract=PlannerResumeContract(
                resume_token=interaction.resume_token,
                resume_goal="应用用户选择后重新物化并校验。",
                expected_next_actions=("revise_draft", "request_evidence"),
            ),
        ),
    )


def _repair_input_refs(workspace: PlannerWorkspaceState) -> PlannerInputRefs:
    assert workspace.planning_strategy is not None
    assert workspace.working_itinerary is not None
    assert workspace.validation_observation is not None
    return PlannerInputRefs(
        strategy_revision=workspace.planning_strategy.strategy_revision,
        candidate_pool_revision=workspace.candidate_pool.revision,
        draft_revision=workspace.working_itinerary.draft_revision,
        validation_observation_id=workspace.validation_observation.observation_id,
    )


def _ask_reason(issues: tuple[PlannerValidationIssue, ...]) -> AskUserReasonCode:
    codes = {item.code for item in issues}
    if "reservation_conflict" in codes:
        return AskUserReasonCode.FIXED_BOOKING_CONFLICT
    if "unscheduled_strong_intent" in codes or "required_missing" in codes:
        return AskUserReasonCode.INCOMPATIBLE_STRONG_COMMITMENTS
    if codes & {"budget_exceeded", "hotel_constraint_conflict"}:
        return AskUserReasonCode.PERMISSION_TO_RELAX_CONSTRAINT
    return AskUserReasonCode.MATERIAL_TRADEOFF_OUTSIDE_DELEGATION


def _safe_patch_error(error: ValueError) -> str:
    message = str(error)
    known = {
        "ItineraryPatch is a no-op under the canonical planning projection": (
            "planner_repair_no_effective_change"
        ),
        "removing or replacing a soft/want item requires an unassigned record": (
            "planner_repair_soft_unassigned_record_required"
        ),
        "immutable and strong items cannot be removed by Planner Patch": (
            "planner_repair_strong_removal_forbidden"
        ),
        "immutable and strong items cannot be replaced by Planner Patch": (
            "planner_repair_strong_replacement_forbidden"
        ),
    }
    return known.get(message, "planner_repair_patch_invalid")
