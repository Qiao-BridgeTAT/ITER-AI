"""Contextual Planner guards: validate authority and evidence, never choose an itinerary."""

from __future__ import annotations

from datetime import datetime

from backend.agent.planner.proposals import PlannerReferenceCatalog
from backend.agent.planner.route_comparison import guard_global_route_comparison
from backend.agent.planner.workspace import (
    PlannerGuardError,
    fixed_commitments,
    service_dates,
    task_book_references,
)
from backend.contracts.v4.content_quality import visible_text_quality_issue
from backend.contracts.v4.enums import CommitmentLevel, CrossClusterReasonCode
from backend.contracts.v4.planner_decision import (
    AskUserPayload,
    PlannerDecision,
    ReviseDraftPayload,
)
from backend.contracts.v4.planner_draft import (
    DraftItem,
    UnassignedIntent,
    WorkingItineraryDraft,
    planning_projection_digest,
    validate_draft_against_pool,
)
from backend.contracts.v4.planner_patch import (
    PlanChangeRequestPatchAuthority,
    RemoveItemOperation,
    ReplaceItemOperation,
    ValidationIssuePatchAuthority,
)
from backend.contracts.v4.planner_refs import CandidateRef
from backend.contracts.v4.planner_strategy import PlanningStrategy, validate_strategy_against_pool
from backend.contracts.v4.planner_workspace import PlannerWorkspaceState
from backend.contracts.v4.task_book import TaskBookV4


def validate_strategy(
    strategy: PlanningStrategy, workspace: PlannerWorkspaceState, book: TaskBookV4
) -> None:
    _guard_visible_text(strategy.core_experience_summary, "strategy.core_experience_summary", 6)
    _guard_visible_text(strategy.reason_summary, "strategy.reason_summary", 6)
    validate_strategy_against_pool(strategy, workspace.candidate_pool)
    references = task_book_references(book)
    reference_fields = {
        "daily_capacity_policy.source_constraint_refs": (
            strategy.daily_capacity_policy.source_constraint_refs
        ),
        "daily_capacity_policy.rest_policy.hard_requirement_refs": (
            strategy.daily_capacity_policy.rest_policy.hard_requirement_refs
        ),
        "daily_capacity_policy.walking_policy.hard_limit_ref": (
            (strategy.daily_capacity_policy.walking_policy.hard_limit_ref,)
            if strategy.daily_capacity_policy.walking_policy.hard_limit_ref
            else ()
        ),
        "lodging_policy.preferred_area_refs": strategy.lodging_policy.preferred_area_refs,
        "lodging_policy.facility_constraint_refs": strategy.lodging_policy.facility_constraint_refs,
        "lodging_policy.budget_constraint_ref": (
            (strategy.lodging_policy.budget_constraint_ref,)
            if strategy.lodging_policy.budget_constraint_ref
            else ()
        ),
        "dining_policy.dietary_constraint_refs": strategy.dining_policy.dietary_constraint_refs,
        "conflict_policy.hard_guard_refs": strategy.conflict_policy.hard_guard_refs,
    }
    for field, values in reference_fields.items():
        if not set(values) <= references.keys():
            # Return a static field path, never echo the rejected model value.
            raise PlannerGuardError(f"planner_strategy_unknown_task_book_reference:{field}")
    # Keep this authority set identical to the short keys advertised in
    # planner_context. The proposal resolver expands those keys to these values.
    assumption_sources = {
        *references,
        *PlannerReferenceCatalog(workspace).evidence.values(),
    }
    invalid_assumption_indices = [
        str(index)
        for index, item in enumerate(strategy.non_blocking_assumptions)
        if item.source_ref not in assumption_sources
    ]
    if invalid_assumption_indices:
        # Indices are safe program-generated repair coordinates. Never echo the
        # rejected model value into an authoritative observation.
        raise PlannerGuardError(
            "planner_strategy_unknown_assumption_source:invalid_indices="
            + ",".join(invalid_assumption_indices)
        )
    required_hard = {key for key in references if key.startswith(("hard:", "dietary:"))}
    missing_hard = sorted(required_hard - set(strategy.conflict_policy.hard_guard_refs))
    if missing_hard:
        raise PlannerGuardError(
            "planner_strategy_hard_constraints_not_covered:missing=" + ",".join(missing_hard)
        )
    required_diet = {key for key in references if key.startswith("dietary:")}
    missing_diet = sorted(required_diet - set(strategy.dining_policy.dietary_constraint_refs))
    if missing_diet:
        raise PlannerGuardError(
            "planner_strategy_dietary_constraints_not_covered:missing=" + ",".join(missing_diet)
        )
    mode = (
        "not_applicable"
        if book.lodging_direction.not_applicable
        else "fixed"
        if book.lodging_direction.existing_booking
        else "search"
    )
    if strategy.lodging_policy.mode != mode:
        raise PlannerGuardError("planner_strategy_lodging_mode_changed")
    if mode == "fixed" and strategy.lodging_policy.fixed_commitment_ref not in fixed_commitments(
        book
    ):
        raise PlannerGuardError("planner_strategy_fixed_hotel_changed")
    if (
        book.lodging_direction.nightly_budget is not None
        and strategy.lodging_policy.budget_constraint_ref != "lodging_budget"
    ):
        raise PlannerGuardError("planner_strategy_hotel_budget_dropped")
    strong_dining = {
        entry.candidate_ref
        for entry in workspace.candidate_pool.candidates
        if entry.entity_kind.value == "restaurant"
        and entry.commitment_level is CommitmentLevel.STRONG
    }
    if set(strategy.dining_policy.destination_restaurant_refs) != strong_dining:
        raise PlannerGuardError("planner_strategy_destination_dining_dropped")
    dates = set(service_dates(book))
    if any(
        item.service_date not in dates for item in strategy.daily_capacity_policy.per_date_overrides
    ) or any(
        item.service_date not in dates for item in strategy.dining_policy.required_meal_windows
    ):
        raise PlannerGuardError("planner_strategy_date_outside_trip")
    if not strategy.dining_policy.required_meal_windows:
        raise PlannerGuardError("planner_strategy_meal_goals_missing")


def validate_working_draft(
    draft: WorkingItineraryDraft, workspace: PlannerWorkspaceState, book: TaskBookV4, now: datetime
) -> None:
    _guard_visible_text(draft.reason_summary, "draft.reason_summary", 6)
    for index, day in enumerate(draft.days):
        _guard_visible_text(day.day_theme, f"draft.days.{index}.day_theme", 2)
    validate_draft_against_pool(
        draft, workspace.candidate_pool, expected_service_dates=service_dates(book)
    )
    strategy = workspace.planning_strategy
    spatial = workspace.spatial_observation
    if strategy is None or spatial is None:
        raise PlannerGuardError("planner_draft_evidence_missing")
    # Freshness and missing-provider facts are validation warnings unless they
    # prove a concrete closure or unavailability. The Validator keeps those
    # caveats attached to the published plan; the draft Guard stays structural.
    if (
        draft.based_on_strategy_revision != strategy.strategy_revision
        or draft.spatial_observation_id != spatial.observation_id
    ):
        raise PlannerGuardError("planner_draft_stale_strategy_or_spatial")
    if draft.content_digest != planning_projection_digest(draft):
        raise PlannerGuardError("planner_draft_digest_invalid")
    # Provider identity gaps cannot be repaired by asking the model to invent an
    # entity.  They remain attached to the workspace and are emitted by the
    # Validator as explicit publication warnings.  Only conflicts that truly
    # require user authority stay blocking here.
    if workspace.readiness_observation and any(
        issue.user_authority_required
        and not (workspace.best_effort_reasons and issue.code == "strong_opening_conflict")
        for issue in workspace.readiness_observation.issues
    ):
        raise PlannerGuardError("planner_readiness_blocker_unresolved")
    if any(
        (intent.commitment_level == "strong" or intent.requires_user_resolution)
        and not workspace.best_effort_reasons
        for intent in draft.unassigned_intents
    ):
        raise PlannerGuardError("planner_strong_or_user_tradeoff_requires_ask_user")
    if book.lodging_direction.not_applicable:
        if draft.lodging_baseline.mode != "not_applicable":
            raise PlannerGuardError("planner_day_trip_cannot_select_hotel")
    elif book.lodging_direction.existing_booking:
        fixed = book.lodging_direction.existing_booking
        if (
            draft.lodging_baseline.mode != "fixed"
            or draft.lodging_baseline.fixed_commitment_ref is None
            or draft.lodging_baseline.fixed_commitment_ref.commitment_id != fixed.booking_id
        ):
            raise PlannerGuardError("planner_fixed_hotel_changed")
        observation = workspace.hotel_observation
        if observation is not None and (
            observation.mode != "fixed_booking_verification"
            or observation.fixed_booking is None
            or observation.fixed_booking.commitment_ref
            != draft.lodging_baseline.fixed_commitment_ref
        ):
            raise PlannerGuardError("planner_fixed_hotel_not_verified")
    else:
        observation = workspace.hotel_observation
        if draft.lodging_baseline.mode == "unresolved":
            selectable = (
                tuple(
                    offer
                    for offer in observation.offers
                    if offer.availability_status != "unavailable"
                )
                if observation is not None
                else ()
            )
            if selectable:
                raise PlannerGuardError("planner_unresolved_hotel_has_selectable_offer")
        elif (
            draft.lodging_baseline.mode != "selected_offer"
            or observation is None
            or observation.hotel_observation_id != draft.hotel_observation_id
        ):
            raise PlannerGuardError("planner_draft_hotel_observation_missing")
        else:
            selected = next(
                (
                    offer
                    for offer in observation.offers
                    if offer.offer_ref == draft.lodging_baseline.selected_offer_ref
                ),
                None,
            )
            if selected is None or selected.availability_status == "unavailable":
                raise PlannerGuardError("planner_hotel_offer_unknown_or_unavailable")
    assigned_fixed = {
        item.object_ref
        for day in draft.days
        for item in day.ordered_items
        if not isinstance(item.object_ref, CandidateRef)
    }
    if draft.lodging_baseline.fixed_commitment_ref:
        assigned_fixed.add(draft.lodging_baseline.fixed_commitment_ref)
    if assigned_fixed != set(workspace.candidate_pool.fixed_commitments):
        raise PlannerGuardError("planner_immutable_commitment_missing")
    bookings = {booking.booking_id: booking for booking in book.existing_bookings}
    clusters = {cluster.cluster_id for cluster in spatial.clusters}
    edges = {edge.route_edge_id: edge for edge in spatial.route_edges}
    pool = workspace.candidate_pool.candidate_by_id()
    known_evidence = {
        workspace.candidate_pool.candidate_pool_id,
        strategy.strategy_id,
        spatial.observation_id,
        *(fact.fact_reference_id for fact in workspace.verified_facts),
        *(obs.observation_id for obs in workspace.capability_observations),
    }
    if workspace.hotel_observation is not None:
        known_evidence.add(workspace.hotel_observation.hotel_observation_id)
    for entry in pool.values():
        known_evidence.update(entry.fact_reference_ids)
        known_evidence.update(entry.source_intent_refs)
    for day_index, day in enumerate(draft.days):
        if day.primary_cluster_id is not None and day.primary_cluster_id not in clusters:
            raise PlannerGuardError("planner_day_primary_cluster_unknown")
        if day.day_kind == "active" and not day.ordered_items:
            raise PlannerGuardError("planner_active_day_cannot_be_empty")
        allowed_transport_modes = strategy.spatial_policy.preferred_transport_modes
        if not set(day.transport_preferences) <= set(allowed_transport_modes):
            raise PlannerGuardError(
                f"planner_day_transport_not_in_strategy:day_index={day_index}:allowed="
                + ",".join(allowed_transport_modes)
            )
        visits = sum(item.item_kind == "visit" for item in day.ordered_items)
        if visits > strategy.daily_capacity_policy.major_activity_target.maximum:
            raise PlannerGuardError(
                f"planner_day_exceeds_declared_capacity:day_index={day_index}:"
                f"visits={visits}:maximum="
                f"{strategy.daily_capacity_policy.major_activity_target.maximum}"
            )
        expected_meals = {
            meal.meal
            for meal in strategy.dining_policy.required_meal_windows
            if meal.service_date == day.service_date
        }
        if not expected_meals <= set(day.dining_goals):
            raise PlannerGuardError("planner_day_meal_goal_dropped")
        for item in day.ordered_items:
            if not isinstance(item.object_ref, CandidateRef):
                booking = bookings.get(item.object_ref.commitment_id)
                if booking and booking.start_date and booking.start_date != day.service_date:
                    raise PlannerGuardError("planner_fixed_commitment_moved_date")
        for segment_index, segment in enumerate(day.cross_cluster_segments):
            if segment.reason_code not in strategy.spatial_policy.allowed_cross_cluster_reasons:
                raise PlannerGuardError("planner_cross_cluster_reason_not_allowed")
            if not set(segment.supporting_intent_or_fact_refs) <= known_evidence:
                raise PlannerGuardError("planner_cross_cluster_evidence_unknown")
            covered = [
                item for item in day.ordered_items if item.draft_item_id in segment.covered_item_ids
            ]
            if segment.reason_code is CrossClusterReasonCode.STRONG_USER_INTENT and not any(
                item.commitment_level in {"strong", "immutable"} for item in covered
            ):
                raise PlannerGuardError(
                    "planner_cross_cluster_has_no_strong_intent:"
                    f"day_index={day_index}:segment_index={segment_index}:"
                    "covered_commitments="
                    + ",".join(sorted({item.commitment_level for item in covered}))
                )
            if segment.reason_code is CrossClusterReasonCode.DATE_SPECIFIC_AVAILABILITY and not any(
                isinstance(item.object_ref, CandidateRef)
                and pool[item.object_ref.candidate_id].feasible_dates == (day.service_date,)
                for item in covered
            ):
                raise PlannerGuardError("planner_cross_cluster_has_no_date_specific_fact")
            if (
                segment.reason_code is CrossClusterReasonCode.RESERVATION_OR_FIXED_COMMITMENT
                and not any(item.commitment_level == "immutable" for item in covered)
            ):
                raise PlannerGuardError("planner_cross_cluster_has_no_fixed_commitment")
            if (
                segment.reason_code is CrossClusterReasonCode.LODGING_OR_TRANSPORT_ANCHOR
                and draft.lodging_baseline.mode == "not_applicable"
                and not workspace.candidate_pool.fixed_commitments
            ):
                raise PlannerGuardError("planner_cross_cluster_anchor_missing")
            if segment.reason_code is CrossClusterReasonCode.VERIFIED_GLOBAL_ROUTE_IMPROVEMENT:
                guard_global_route_comparison(
                    segment.comparison_observation_ref, draft, workspace, now
                )
            covered_ids = set(segment.covered_item_ids)
            # Cluster-to-cluster estimates cannot be inherited by different final
            # entities. Bind every cited edge to an actual ordered boundary pair.
            actual_pairs = {
                (_endpoint_key(left), _endpoint_key(right))
                for left, right in zip(day.ordered_items, day.ordered_items[1:], strict=False)
                if {left.cluster_id, right.cluster_id}
                == {segment.from_cluster_id, segment.to_cluster_id}
                and {left.draft_item_id, right.draft_item_id} & covered_ids
            }
            cited_pairs = set()
            for edge_id in segment.route_edge_ids:
                edge = edges.get(edge_id)
                if (
                    edge is None
                    or edge.duration_minutes is None
                    or edge.transport_mode not in day.transport_preferences
                ):
                    raise PlannerGuardError("planner_cross_cluster_edge_unusable")
                pair = (
                    (edge.origin.kind, edge.origin.reference_id),
                    (edge.destination.kind, edge.destination.reference_id),
                )
                if pair not in actual_pairs:
                    raise PlannerGuardError(
                        "planner_cross_cluster_edge_wrong_endpoints:"
                        f"day_index={day_index}:segment_index={segment_index}"
                    )
                if pair in cited_pairs:
                    raise PlannerGuardError("planner_cross_cluster_duplicate_route_alternative")
                cited_pairs.add(pair)
            if cited_pairs != actual_pairs:
                raise PlannerGuardError(
                    "planner_cross_cluster_boundary_route_missing:"
                    f"day_index={day_index}:segment_index={segment_index}"
                )
            actual_cost = sum(edges[key].duration_minutes or 0 for key in segment.route_edge_ids)
            if segment.expected_route_cost.duration_minutes != actual_cost:
                raise PlannerGuardError("planner_cross_cluster_route_cost_changed")
    for intent in draft.unassigned_intents:
        if _has_accepted_omission_receipt(intent, draft, workspace):
            # A server-compiled, already-applied validation repair is not a new
            # model claim. Keep its exact receipt across route-only revisions;
            # the revised schedule still goes through the full Validator.
            continue
        if not set(intent.supporting_observation_refs) <= known_evidence:
            raise PlannerGuardError("planner_unassigned_intent_evidence_unknown")
        entry = pool[intent.candidate_ref.candidate_id]
        cited = set(intent.supporting_observation_refs)
        if cited <= set(entry.source_intent_refs):
            raise PlannerGuardError("planner_unassigned_intent_requires_observation_not_intent")
        if intent.reason_code in {"opening_conflict", "infeasible_date"} and (
            set(entry.infeasible_dates) != set(service_dates(book))
            or not cited & set(entry.fact_reference_ids)
        ):
            raise PlannerGuardError("planner_unassigned_opening_conflict_not_verified")
        if intent.reason_code == "capacity_conflict":
            feasible_days = [
                day for day in draft.days if day.service_date not in entry.infeasible_dates
            ]
            if (
                strategy.strategy_id not in cited
                or not feasible_days
                or any(
                    day.day_kind != "rest"
                    and sum(item.item_kind == "visit" for item in day.ordered_items)
                    < strategy.daily_capacity_policy.major_activity_target.maximum
                    for day in feasible_days
                )
            ):
                candidate_key = next(
                    (
                        f"c{index}"
                        for index, candidate in enumerate(workspace.candidate_pool.candidates, 1)
                        if candidate.candidate_ref.candidate_id == intent.candidate_ref.candidate_id
                    ),
                    "unknown",
                )
                raise PlannerGuardError(
                    "planner_unassigned_capacity_conflict_not_supported:"
                    f"candidate_key={candidate_key}"
                )
        if intent.reason_code == "route_conflict":
            related_routes = [
                edge
                for edge in spatial.route_edges
                if intent.candidate_ref.candidate_id
                in {edge.origin.reference_id, edge.destination.reference_id}
                and edge.duration_minutes is not None
            ]
            if spatial.observation_id not in cited or not related_routes:
                raise PlannerGuardError("planner_unassigned_route_conflict_not_observed")
        if intent.reason_code in {"budget_conflict", "duplicate_experience"}:
            # Costs and equivalence have no validated producer in V4-04. A label
            # on an arbitrary fact cannot authorize dropping a wanted experience.
            raise PlannerGuardError("planner_unassigned_reason_evidence_not_available_in_stage")
    for discard in draft.discardable_objects:
        if discard.authorization_ref not in known_evidence:
            raise PlannerGuardError("planner_discard_authority_unknown")


def _has_accepted_omission_receipt(
    intent: UnassignedIntent,
    draft: WorkingItineraryDraft,
    workspace: PlannerWorkspaceState,
) -> bool:
    if (
        workspace.best_effort_reasons
        and intent.reason_code == "awaiting_user"
        and any(
            intent.model_copy(update={"candidate_ref": receipt.candidate_ref}) == receipt
            and all(
                getattr(intent.candidate_ref, field) == getattr(receipt.candidate_ref, field)
                for field in (
                    "candidate_pool_id",
                    "candidate_id",
                    "canonical_entity_id",
                    "entity_kind",
                )
            )
            for receipt in workspace.recovery_omissions
        )
    ):
        for observation in workspace.recovery_observations:
            if (
                intent.supporting_observation_refs == (observation.observation_id,)
                and observation.draft_id == draft.draft_id
                and observation.draft_revision < draft.draft_revision
                and any(
                    (
                        issue.severity != "warning"
                        and issue.code
                        in {"opening_conflict", "meal_constraint_violation", "time_overlap"}
                    )
                    or (
                        issue.code == "route_cost_exceeded"
                        and "internal_schedule_quality:dining_detour"
                        in issue.violated_constraint_refs
                        and any(
                            ref.canonical_entity_id == intent.candidate_ref.canonical_entity_id
                            for ref in issue.candidate_refs
                        )
                    )
                    for issue in observation.issues
                )
            ):
                return True
    if intent.commitment_level != "soft" or intent.requires_user_resolution:
        return False
    for decision in workspace.decision_trace:
        if not isinstance(decision.payload, ReviseDraftPayload):
            continue
        patch = decision.payload.itinerary_patch
        if (
            not isinstance(
                patch.authority, (ValidationIssuePatchAuthority, PlanChangeRequestPatchAuthority)
            )
            or patch.base_draft_id != draft.draft_id
            or patch.base_draft_revision >= draft.draft_revision
            or intent.supporting_observation_refs != (patch.authority.reference_id,)
        ):
            continue
        if isinstance(patch.authority, PlanChangeRequestPatchAuthority) and (
            decision.input_refs.plan_change_request_id != patch.authority.reference_id
            or intent.reason_code != "user_requested"
        ):
            continue
        if any(
            isinstance(operation, (RemoveItemOperation, ReplaceItemOperation))
            and (receipt := operation.unscheduled_intent_record) is not None
            and intent.model_copy(update={"candidate_ref": receipt.candidate_ref}) == receipt
            and all(
                getattr(intent.candidate_ref, field) == getattr(receipt.candidate_ref, field)
                for field in (
                    "candidate_pool_id",
                    "candidate_id",
                    "canonical_entity_id",
                    "entity_kind",
                )
            )
            for operation in patch.operations
        ):
            return True
    return False


def _endpoint_key(item: DraftItem) -> tuple[str, str]:
    reference = item.object_ref
    if isinstance(reference, CandidateRef):
        return "candidate", reference.candidate_id
    return "fixed_commitment", reference.commitment_id


def _guard_visible_text(value: str, field: str, minimum: int) -> None:
    issue = visible_text_quality_issue(value, minimum_units=minimum)
    if issue is not None:
        raise PlannerGuardError(f"planner_visible_text_invalid:{field}:{issue}")


def guard_ask_user(decision: PlannerDecision, workspace: PlannerWorkspaceState) -> None:
    payload = decision.payload
    if not isinstance(payload, AskUserPayload):
        raise PlannerGuardError("planner_ask_user_blocking_source_missing")
    if decision.input_refs.validation_observation_id is not None:
        validation_observation = workspace.validation_observation
        if (
            validation_observation is None
            or decision.input_refs.validation_observation_id
            != validation_observation.observation_id
        ):
            raise PlannerGuardError("planner_ask_user_source_stale")
        issues = {issue.issue_id: issue for issue in validation_observation.issues}
        selected = [issues.get(key) for key in payload.blocking_issue_ids]
        if not selected or any(
            issue is None
            or not issue.user_authority_required
            or "ask_user" not in issue.allowed_actions
            for issue in selected
        ):
            raise PlannerGuardError("planner_ask_user_not_authorized")
        if not payload.user_decision_request.option_contracts or any(
            not set(option.affected_refs) <= set(payload.blocking_issue_ids)
            or option.semantic_action not in {"keep_task_book", "revise_task_book"}
            for option in payload.user_decision_request.option_contracts
        ):
            raise PlannerGuardError("planner_ask_user_option_not_verified")
        return

    readiness_observation = workspace.readiness_observation
    if (
        readiness_observation is None
        or decision.input_refs.readiness_observation_id != readiness_observation.observation_id
    ):
        raise PlannerGuardError("planner_ask_user_source_stale")
    readiness_issues = {issue.issue_id: issue for issue in readiness_observation.issues}
    readiness_selected = [readiness_issues.get(key) for key in payload.blocking_issue_ids]
    if not readiness_selected or any(
        issue is None or not issue.user_authority_required for issue in readiness_selected
    ):
        raise PlannerGuardError("planner_ask_user_not_authorized")
    allowed = {
        option.option_id: option
        for issue in readiness_selected
        if issue is not None
        for option in issue.option_contracts
    }
    if not payload.user_decision_request.option_contracts or any(
        allowed.get(option.option_id) != option
        for option in payload.user_decision_request.option_contracts
    ):
        raise PlannerGuardError("planner_ask_user_option_not_verified")
