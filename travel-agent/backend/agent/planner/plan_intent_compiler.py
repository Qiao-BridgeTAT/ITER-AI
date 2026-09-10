"""Compile a compact model-owned plan intent into authoritative Planner artifacts."""

from __future__ import annotations

from collections import Counter
from datetime import time
from typing import Literal, cast
from uuid import uuid4

from backend.agent.planner.calendar_alignment import align_days_with_opening_evidence
from backend.agent.planner.decision_contracts import ModelPlanIntent, ModelPlanStop
from backend.agent.planner.guards import validate_strategy
from backend.agent.planner.proposals import PlannerReferenceCatalog
from backend.agent.planner.visit_identity import overlapping_visits
from backend.agent.planner.workspace import (
    PlannerGuardError,
    fixed_commitments,
    server_id,
    service_dates,
    task_book_references,
)
from backend.contracts.v4.enums import (
    CandidateEntityKind,
    CommitmentLevel,
    CrossClusterReasonCode,
    PlannerCapability,
)
from backend.contracts.v4.plan_change import SelectedHotelRecommendation
from backend.contracts.v4.planner_decision import (
    BuildOrUpdateStrategyPayload,
    MaterializeDraftPayload,
    PlannerCompletionAssessment,
    PlannerDecision,
    PlannerInputRefs,
)
from backend.contracts.v4.planner_draft import (
    DraftItem,
    ExpectedWindow,
    LodgingBaseline,
    UnassignedIntent,
    WorkingItineraryDay,
    WorkingItineraryDraft,
    planning_projection_digest,
)
from backend.contracts.v4.planner_observations import (
    HotelOfferObservation,
    HotelSearchArguments,
    PlannerCapabilityRequest,
)
from backend.contracts.v4.planner_refs import PlannerScope
from backend.contracts.v4.planner_strategy import (
    MAX_DAILY_MAJOR_ACTIVITIES,
    ActivityTargetRange,
    ConflictPolicy,
    DailyCapacityPolicy,
    DiningPolicy,
    LodgingPolicy,
    PlanningStrategy,
    PreferredTimeWindow,
    RequiredMealWindow,
    RestPolicy,
    SpatialPolicy,
    StrategyAnchorPolicy,
    WalkingPolicy,
)
from backend.contracts.v4.planner_workspace import PlannerWorkspaceState
from backend.contracts.v4.task_book import BookingReference, TaskBookV4
from backend.providers.contracts import HoursDayStatus


def build_automatic_hotel_request(
    workspace: PlannerWorkspaceState,
    book: TaskBookV4,
) -> PlannerCapabilityRequest | None:
    """Return the one deterministic lodging lookup required before plan selection."""

    if book.lodging_direction.not_applicable or workspace.hotel_observation is not None:
        return None
    spatial = workspace.spatial_observation
    if spatial is None or not spatial.clusters:
        return None
    references = task_book_references(book)
    return PlannerCapabilityRequest(
        request_id=server_id(workspace.generation_id, "automatic-hotel-search"),
        scope=workspace.current_scope,
        capability=PlannerCapability.HOTEL_SEARCH,
        purpose="complete_initial_evidence",
        service_dates=(),
        blocking=False,
        arguments=HotelSearchArguments(
            check_in_date=book.destination_and_dates.start_date,
            check_out_date=book.destination_and_dates.end_date,
            party_size_ref="party",
            lodging_preference_refs=tuple(
                key for key in references if key == "lodging" or key.startswith("area:")
            ),
            budget_constraint_ref=("lodging_budget" if "lodging_budget" in references else None),
            facility_constraint_refs=tuple(
                key for key in references if key.startswith("facility:")
            ),
            activity_cluster_refs=tuple(cluster.cluster_id for cluster in spatial.clusters),
        ),
    )


def compile_default_strategy_decision(
    workspace: PlannerWorkspaceState,
    book: TaskBookV4,
) -> PlannerDecision:
    """Compile policy and all formal references from the confirmed task book."""

    if workspace.planning_strategy is not None:
        raise PlannerGuardError("planner_compact_strategy_already_exists")
    output_scope = workspace.current_scope.model_copy(
        update={"workspace_revision": workspace.workspace_revision + 1}
    )
    references = task_book_references(book)
    candidate_entries = tuple(workspace.candidate_pool.candidates)
    strong_count = sum(
        entry.commitment_level is CommitmentLevel.STRONG
        and entry.entity_kind is not CandidateEntityKind.RESTAURANT
        for entry in candidate_entries
    )
    pace_profile = _pace_profile(book)
    # Pace controls time, travel and rests, not a lower implicit POI ceiling.
    # Four is capacity for a feasible plan, never a target to fill every day.
    maximum = MAX_DAILY_MAJOR_ACTIVITIES
    minimum = 0 if strong_count == 0 else 1
    modes = preferred_transport_modes(book)
    lodging = _lodging_policy(book, references)
    required_meals: tuple[Literal["lunch", "dinner"], ...] = ("lunch", "dinner")
    strategy = PlanningStrategy(
        strategy_id=server_id(workspace.generation_id, "strategy"),
        strategy_revision=1,
        scope=output_scope,
        candidate_pool_revision=workspace.candidate_pool.revision,
        core_experience_summary=_core_experience_summary(book),
        anchor_policy=StrategyAnchorPolicy(
            immutable_refs=tuple(workspace.candidate_pool.fixed_commitments),
            strong_candidate_refs=tuple(
                entry.candidate_ref
                for entry in candidate_entries
                if entry.commitment_level is CommitmentLevel.STRONG
            ),
            soft_candidate_refs=tuple(
                entry.candidate_ref
                for entry in candidate_entries
                if entry.commitment_level is CommitmentLevel.SOFT
            ),
            filler_candidate_refs=tuple(
                entry.candidate_ref
                for entry in candidate_entries
                if entry.commitment_level in {CommitmentLevel.FILLER, CommitmentLevel.NEUTRAL}
            ),
        ),
        daily_capacity_policy=DailyCapacityPolicy(
            pace_profile=pace_profile,
            preferred_start_window=PreferredTimeWindow(earliest=time(9), latest=time(10)),
            preferred_end_window=PreferredTimeWindow(earliest=time(18), latest=time(20)),
            major_activity_target=ActivityTargetRange(minimum=minimum, maximum=maximum),
            walking_policy=WalkingPolicy(
                goal="minimize" if pace_profile == "relaxed" else "balanced"
            ),
            # Compatibility field only: pace means visit depth, never forced downtime.
            rest_policy=RestPolicy(mode="minimal"),
            source_constraint_refs=tuple(key for key in references if key.startswith("pace:")),
        ),
        spatial_policy=SpatialPolicy(
            preferred_transport_modes=modes,
            prefer_single_primary_cluster=True,
            split_large_cluster=True,
            merge_adjacent_clusters=True,
            allowed_cross_cluster_reasons=tuple(CrossClusterReasonCode),
        ),
        lodging_policy=lodging,
        dining_policy=DiningPolicy(
            destination_restaurant_refs=tuple(
                entry.candidate_ref
                for entry in candidate_entries
                if entry.entity_kind is CandidateEntityKind.RESTAURANT
                and entry.commitment_level is CommitmentLevel.STRONG
            ),
            required_meal_windows=tuple(
                RequiredMealWindow(service_date=day, meal=meal)
                for day in service_dates(book)
                for meal in required_meals
            ),
            flexible_meal_placement="near_route",
            dietary_constraint_refs=tuple(key for key in references if key.startswith("dietary:")),
        ),
        conflict_policy=ConflictPolicy(
            hard_guard_refs=tuple(
                key for key in references if key.startswith(("hard:", "dietary:"))
            ),
            flexible_tradeoff_order=(
                "retain_soft_attractions",
                "retain_destination_meals",
                "minimize_walking",
                "minimize_cross_cluster",
                "minimize_total_cost",
            ),
        ),
        reason_summary="已由程序根据确认任务书编译容量、交通、住宿和硬约束策略。",
    )
    validate_strategy(strategy, workspace, book)
    payload = BuildOrUpdateStrategyPayload(mode="initialize", proposed_strategy=strategy)
    return PlannerDecision(
        decision_id=str(uuid4()),
        scope=workspace.current_scope,
        action="build_or_update_strategy",
        current_goal="编译正式行程的确定性规划策略",
        reason_summary="模型只负责每日语义选择，固定政策与引用由程序编译。",
        input_refs=PlannerInputRefs(candidate_pool_revision=workspace.candidate_pool.revision),
        completion_assessment=PlannerCompletionAssessment(
            ready_to_finalize=False,
            blocking_issue_ids=("plan_intent_pending", "validation_pending"),
        ),
        payload=payload,
    )


def compile_plan_intent_decision(
    intent: ModelPlanIntent,
    workspace: PlannerWorkspaceState,
    book: TaskBookV4,
    *,
    base_draft: WorkingItineraryDraft | None = None,
) -> PlannerDecision:
    """Resolve local c/h keys and deterministically fill the formal draft shape."""

    strategy = workspace.planning_strategy
    spatial = workspace.spatial_observation
    if strategy is None or spatial is None or workspace.working_itinerary is not None:
        raise PlannerGuardError("planner_compact_plan_prerequisites_missing")
    catalog = PlannerReferenceCatalog(workspace)
    current_refs = {
        entry.candidate_ref.canonical_entity_id: entry.candidate_ref
        for entry in catalog.candidates.values()
    }
    accepted_omissions = {}
    for item in workspace.recovery_omissions:
        current_ref = current_refs.get(item.candidate_ref.canonical_entity_id)
        if current_ref is None or any(
            getattr(current_ref, field) != getattr(item.candidate_ref, field)
            for field in ("candidate_pool_id", "candidate_id", "entity_kind")
        ):
            raise PlannerGuardError("planner_recovery_omission_identity_changed")
        # Also repair already persisted workspaces from before receipt rebinding.
        accepted_omissions[current_ref.canonical_entity_id] = item.model_copy(
            update={"candidate_ref": current_ref}
        )
    dates = service_dates(book)
    submitted_days: dict[int, object] = {}
    for model_day in intent.days:
        if model_day.day_index > len(dates):
            raise PlannerGuardError(
                f"planner_plan_day_outside_trip:path=days[].day_index:allowed_values=1-{len(dates)}"
            )
        if model_day.day_index in submitted_days:
            raise PlannerGuardError(
                f"planner_plan_duplicate_day:path=days[].day_index:value={model_day.day_index}"
            )
        submitted_days[model_day.day_index] = model_day

    stops_by_day: dict[int, list[ModelPlanStop]] = {
        index: _without_unrequested_extra_meal(
            list(getattr(submitted_days.get(index), "stops", ())), catalog, book
        )
        for index in range(1, len(dates) + 1)
    }
    for index, stops in stops_by_day.items():
        slots: list[str] = [
            stop.meal_slot
            for stop in stops
            if stop.meal_slot is not None
            and (entry := catalog.candidates.get(stop.candidate_key)) is not None
            and entry.entity_kind is CandidateEntityKind.RESTAURANT
        ]
        if any(slots.count(slot) > 1 for slot in ("lunch", "dinner")):
            raise PlannerGuardError(
                f"planner_plan_duplicate_main_meal:path=days[{index - 1}].stops:"
                "repair=每天仅午餐和晚餐各一次；保留最适合路线的餐厅，不添加加餐"
            )
        if any(
            stop.meal_slot == "lunch"
            and stop.candidate_key in catalog.candidates
            and catalog.candidates[stop.candidate_key].entity_kind is CandidateEntityKind.RESTAURANT
            for stop in stops
        ):
            # An explicitly selected restaurant meal owns lunch. The optional
            # onsite flag must not compile a second meal over the same period.
            # Preserve all selected places/order; actual feasibility is still
            # materialized and validated, including the long visit's duration.
            stops_by_day[index] = [
                stop.model_copy(update={"onsite_lunch": False}) if stop.onsite_lunch else stop
                for stop in stops
            ]
    stops_by_day = align_days_with_opening_evidence(stops_by_day, dates, catalog, workspace, book)
    seen: dict[str, tuple[int, int]] = {}
    for day_index, stops in stops_by_day.items():
        unique_stops = []
        for stop_index, stop in enumerate(stops):
            if stop.candidate_key in seen:
                # Exact duplicate references need no second whole-plan model
                # call. Keep the first calendar-aligned placement; the common
                # day/meal repair ledger measures and repairs the resulting gap.
                # Different identities at the same venue still need validation.
                continue
            unique_stops.append(stop)
            seen[stop.candidate_key] = (day_index, stop_index)
            entry = catalog.candidates.get(stop.candidate_key)
            if entry is None or entry.selection_permission == "forbidden":
                raise PlannerGuardError(
                    "planner_plan_candidate_key_invalid:"
                    f"path=days[{day_index - 1}].stops[{stop_index}].candidate_key:"
                    "allowed_values=" + ",".join(_allowed_candidate_keys(catalog))
                )
            if stop.onsite_lunch:
                estimate = next(
                    (
                        x
                        for x in workspace.visit_duration_estimates
                        if x.canonical_entity_id == entry.candidate_ref.canonical_entity_id
                    ),
                    None,
                )
                maximum_duration = (
                    estimate.maximum_minutes
                    if estimate
                    else (entry.advisory_features.typical_duration_minutes or 0)
                )
                if entry.entity_kind is CandidateEntityKind.RESTAURANT or maximum_duration < 240:
                    raise PlannerGuardError(
                        "planner_onsite_lunch_requires_long_visit:"
                        f"path=days[{day_index - 1}].stops[{stop_index}].onsite_lunch:"
                        "repair=只有半天或全天景区可跨午餐；普通短景点改为沿途具体餐厅"
                    )
            service_date = dates[day_index - 1]
            if service_date in entry.infeasible_dates or (
                entry.feasible_dates and service_date not in entry.feasible_dates
            ):
                allowed_dates = tuple(
                    value
                    for value in dates
                    if value not in entry.infeasible_dates
                    and (not entry.feasible_dates or value in entry.feasible_dates)
                )
                raise PlannerGuardError(
                    "planner_candidate_date_infeasible:"
                    f"path=days[{day_index - 1}].stops[{stop_index}].candidate_key:"
                    f"candidate_key={stop.candidate_key}:allowed_values="
                    + ",".join(value.isoformat() for value in allowed_dates)
                )
        stops_by_day[day_index] = unique_stops

    # Strong commitments are authoritative. If the model omits one, place it on
    # the least-loaded feasible day instead of asking it to reproduce policy fields.
    for key, entry in catalog.candidates.items():
        if entry.commitment_level is not CommitmentLevel.STRONG or key in seen:
            continue
        if entry.candidate_ref.canonical_entity_id in accepted_omissions:
            continue
        if entry.entity_kind is CandidateEntityKind.RESTAURANT:
            # Choosing its date/meal is semantic work. Appending a default
            # anytime stop silently turns a required dinner into a late snack.
            raise PlannerGuardError(
                "planner_plan_missing_required_restaurant:"
                f"path=days[].stops:candidate_key={key}:"
                "repair=保留该必吃餐厅并明确选择合适日期的lunch或dinner；"
                "可以替换普通餐厅，不得补到晚餐后作为额外下午茶"
            )
        feasible_indices = [
            index
            for index, value in enumerate(dates, 1)
            if value not in entry.infeasible_dates
            and (not entry.feasible_dates or value in entry.feasible_dates)
        ]
        if not feasible_indices:
            if workspace.best_effort_reasons:
                # Preserve the must-visit requirement as an explicit unmet
                # intent; never send the visitor into a known closed venue.
                continue
            raise PlannerGuardError(
                "planner_strong_candidate_has_no_feasible_date:"
                f"path=candidates.{key}.feasible_dates"
            )
        verified_open = {
            day.service_date
            for evidence in workspace.hours_evidence
            if evidence.canonical_entity_id == entry.candidate_ref.canonical_entity_id
            for day in evidence.days
            if day.status is HoursDayStatus.OPEN and day.intervals
        }
        target = min(
            feasible_indices,
            key=lambda value: (
                bool(verified_open) and dates[value - 1] not in verified_open,
                len(stops_by_day[value]),
                value,
            ),
        )
        stops_by_day[target].append(ModelPlanStop(candidate_key=key))
        seen[key] = (target, len(stops_by_day[target]) - 1)

    allowed_candidate_keys = _allowed_candidate_keys(catalog)
    if not seen and allowed_candidate_keys:
        raise PlannerGuardError(
            "planner_plan_requires_activity:path=days[].stops:allowed_values="
            + ",".join(allowed_candidate_keys)
        )

    maximum_visits = strategy.daily_capacity_policy.major_activity_target.maximum
    overlaps = overlapping_visits(workspace.place_evidence)
    selected_visits: list[str] = []
    for key in seen:
        entry = catalog.candidates[key]
        if entry.entity_kind is CandidateEntityKind.RESTAURANT:
            continue
        previous = next(
            (
                selected
                for selected in selected_visits
                if frozenset(
                    (
                        entry.candidate_ref.canonical_entity_id,
                        catalog.candidates[selected].candidate_ref.canonical_entity_id,
                    )
                )
                in overlaps
            ),
            None,
        )
        if previous is not None and not (
            entry.commitment_level is CommitmentLevel.STRONG
            and catalog.candidates[previous].commitment_level is CommitmentLevel.STRONG
        ):
            raise PlannerGuardError(
                f"planner_plan_same_venue_duplicate:path=days[].stops:candidate_keys={previous},{key}:"
                "repair=同一景区和内部子地点不能重复算两个景点；保留必去地点，另选独立的沿途景点"
            )
        selected_visits.append(key)
    for day_index, stops in stops_by_day.items():
        visit_count = sum(
            catalog.candidates[stop.candidate_key].entity_kind is not CandidateEntityKind.RESTAURANT
            for stop in stops
        )
        if visit_count > maximum_visits:
            raise PlannerGuardError(
                "planner_plan_day_capacity_exceeded:"
                f"path=days[{day_index - 1}].stops:"
                f"maximum={maximum_visits}:actual={visit_count}"
            )

    decision_id = str(uuid4())
    item_ids = {key: server_id(decision_id, "item", key) for key in (*seen, *catalog.fixed)}
    bookings = _bookings(book)
    fixed_before_by_day: dict[int, list[str]] = {index: [] for index in stops_by_day}
    fixed_after_by_day: dict[int, list[str]] = {index: [] for index in stops_by_day}
    for key, reference in catalog.fixed.items():
        if reference.commitment_kind == "named_hotel":
            continue
        booking = bookings.get(reference.commitment_id)
        target_date = (
            booking.start_date
            if booking is not None and booking.start_date in dates
            else dates[-1]
            if reference.commitment_kind == "departure"
            else dates[0]
        )
        target_day = dates.index(target_date) + 1
        target_collection = (
            fixed_after_by_day if reference.commitment_kind == "departure" else fixed_before_by_day
        )
        target_collection[target_day].append(key)

    days: list[WorkingItineraryDay] = []
    for day_index, service_date in enumerate(dates, 1):
        resolved_items: list[DraftItem] = []
        for key in fixed_before_by_day[day_index]:
            reference = catalog.fixed[key]
            resolved_items.append(
                DraftItem(
                    draft_item_id=item_ids[key],
                    position=len(resolved_items),
                    item_kind=_fixed_item_kind(reference.commitment_kind),
                    object_ref=reference,
                    expected_window=ExpectedWindow(part_of_day="anytime"),
                    commitment_level="immutable",
                )
            )
        for stop in stops_by_day[day_index]:
            entry = catalog.candidates[stop.candidate_key]
            is_restaurant = entry.entity_kind is CandidateEntityKind.RESTAURANT
            resolved_items.append(
                DraftItem(
                    draft_item_id=item_ids[stop.candidate_key],
                    position=len(resolved_items),
                    item_kind="dining" if is_restaurant else "visit",
                    object_ref=entry.candidate_ref,
                    cluster_id=entry.cluster_ids[0] if entry.cluster_ids else None,
                    expected_window=ExpectedWindow(part_of_day=stop.part_of_day),
                    meal_slot=(
                        stop.meal_slot or _meal_slot(stop.part_of_day) if is_restaurant else None
                    ),
                    duration_preference=stop.duration_preference,
                    onsite_lunch=stop.onsite_lunch,
                    commitment_level=cast(
                        Literal["immutable", "strong", "soft", "filler", "neutral"],
                        entry.commitment_level.value,
                    ),
                )
            )
        for key in fixed_after_by_day[day_index]:
            reference = catalog.fixed[key]
            resolved_items.append(
                DraftItem(
                    draft_item_id=item_ids[key],
                    position=len(resolved_items),
                    item_kind="departure",
                    object_ref=reference,
                    expected_window=ExpectedWindow(part_of_day="anytime"),
                    commitment_level="immutable",
                )
            )
        candidate_clusters = [
            item.cluster_id for item in resolved_items if item.cluster_id is not None
        ]
        primary_cluster = (
            Counter(candidate_clusters).most_common(1)[0][0] if candidate_clusters else None
        )
        day_kind: Literal["active", "arrival_departure", "rest"] = (
            "active" if candidate_clusters else "arrival_departure" if resolved_items else "rest"
        )
        days.append(
            WorkingItineraryDay(
                service_date=service_date,
                day_kind=day_kind,
                # A model theme can mention a place it never selected. The public
                # day heading must describe executable choices, not that promise.
                day_theme=" · ".join(
                    catalog.candidates[stop.candidate_key].display_name
                    for stop in stops_by_day[day_index]
                    if catalog.candidates[stop.candidate_key].entity_kind
                    is not CandidateEntityKind.RESTAURANT
                )
                or ("抵达与返程" if day_kind == "arrival_departure" else "休息与用餐"),
                primary_cluster_id=primary_cluster,
                ordered_items=tuple(resolved_items),
                cross_cluster_segments=(),
                dining_goals=tuple(
                    item.meal
                    for item in strategy.dining_policy.required_meal_windows
                    if item.service_date == service_date
                ),
                transport_preferences=strategy.spatial_policy.preferred_transport_modes,
            )
        )

    output_scope = workspace.current_scope.model_copy(
        update={"workspace_revision": workspace.workspace_revision + 1}
    )
    lodging, selected_hotel, hotel_observation_id = _compile_lodging(
        intent, workspace, catalog, output_scope
    )
    unassigned = tuple(
        accepted_omissions.get(entry.candidate_ref.canonical_entity_id)
        or UnassignedIntent(
            candidate_ref=entry.candidate_ref,
            commitment_level="strong"
            if entry.commitment_level is CommitmentLevel.STRONG
            else "soft",
            reason_code=(
                "opening_conflict"
                if entry.commitment_level is CommitmentLevel.STRONG
                else "planner_tradeoff"
            ),
            supporting_observation_refs=(
                entry.fact_reference_ids
                if entry.commitment_level is CommitmentLevel.STRONG
                else (strategy.strategy_id,)
            ),
            requires_user_resolution=entry.commitment_level is CommitmentLevel.STRONG,
        )
        for key, entry in catalog.candidates.items()
        if entry.commitment_level in {CommitmentLevel.SOFT, CommitmentLevel.STRONG}
        and key not in seen
    )
    draft = WorkingItineraryDraft(
        draft_id=server_id(workspace.generation_id, "draft"),
        draft_revision=base_draft.draft_revision + 1 if base_draft else 1,
        content_digest="0" * 64,
        scope=output_scope,
        based_on_strategy_revision=strategy.strategy_revision,
        candidate_pool_revision=workspace.candidate_pool.revision,
        spatial_observation_id=spatial.observation_id,
        hotel_observation_id=hotel_observation_id,
        lodging_baseline=lodging,
        days=tuple(days),
        discardable_objects=(),
        unassigned_intents=unassigned,
        reason_summary=intent.overall_rationale,
    )
    draft = draft.model_copy(update={"content_digest": planning_projection_digest(draft)})
    payload = MaterializeDraftPayload(
        proposed_working_draft=draft,
        selected_hotel=selected_hotel,
        declared_affected_dates=dates,
    )
    return PlannerDecision(
        decision_id=decision_id,
        scope=workspace.current_scope,
        action="materialize_draft",
        current_goal="形成最终正式行程",
        reason_summary=intent.overall_rationale,
        input_refs=PlannerInputRefs(
            strategy_revision=strategy.strategy_revision,
            candidate_pool_revision=workspace.candidate_pool.revision,
        ),
        completion_assessment=PlannerCompletionAssessment(
            ready_to_finalize=False,
            blocking_issue_ids=("materialization_pending", "validation_pending"),
        ),
        payload=payload,
    )


def _compile_lodging(
    intent: ModelPlanIntent,
    workspace: PlannerWorkspaceState,
    catalog: PlannerReferenceCatalog,
    output_scope: PlannerScope,
) -> tuple[LodgingBaseline, SelectedHotelRecommendation | None, str | None]:
    strategy = workspace.planning_strategy
    assert strategy is not None
    if strategy.lodging_policy.mode == "not_applicable":
        if intent.selected_hotel_key is not None:
            raise PlannerGuardError(
                "planner_plan_hotel_selection_forbidden:path=selected_hotel_key:allowed_values=null"
            )
        return LodgingBaseline(mode="not_applicable"), None, None
    if strategy.lodging_policy.mode == "fixed":
        if intent.selected_hotel_key is not None:
            raise PlannerGuardError(
                "planner_plan_hotel_selection_forbidden:path=selected_hotel_key:allowed_values=null"
            )
        observation_id = (
            workspace.hotel_observation.hotel_observation_id
            if workspace.hotel_observation is not None
            else None
        )
        return (
            LodgingBaseline(
                mode="fixed",
                fixed_commitment_ref=strategy.lodging_policy.fixed_commitment_ref,
            ),
            None,
            observation_id,
        )
    observation = workspace.hotel_observation
    selectable = {
        key: offer
        for key, offer in catalog.hotels.items()
        if offer.availability_status != "unavailable"
    }
    if observation is None or not selectable:
        if intent.selected_hotel_key is not None:
            raise PlannerGuardError(
                "planner_plan_hotel_selection_forbidden:path=selected_hotel_key:allowed_values=null"
            )
        provider_unavailable = observation is None or (
            observation.status == "unavailable"
            or "provider_city_binding" in observation.missing_fact_kinds
        )
        return (
            LodgingBaseline(
                mode="unresolved",
                unresolved_reason=(
                    "provider_unavailable" if provider_unavailable else "no_verified_hotel"
                ),
            ),
            None,
            observation.hotel_observation_id if observation is not None else None,
        )
    allowed = ",".join(selectable)
    if intent.selected_hotel_key is None:
        raise PlannerGuardError(
            "planner_plan_hotel_selection_required:"
            f"path=selected_hotel_key:allowed_values={allowed}"
        )
    offer = selectable.get(intent.selected_hotel_key)
    if offer is None:
        raise PlannerGuardError(
            f"planner_unknown_hotel_key:path=selected_hotel_key:allowed_values={allowed}"
        )
    location_verified = any(
        item.property_id == offer.offer_ref.property_id
        for item in workspace.hotel_location_evidence
    )
    selected = SelectedHotelRecommendation(
        selection_id=server_id(
            workspace.generation_id,
            observation.hotel_observation_id,
            offer.offer_ref.offer_id,
            "selected-hotel",
        ),
        scope=output_scope,
        hotel_observation_id=observation.hotel_observation_id,
        hotel_offer_ref=offer.offer_ref,
        area_reason=(
            "酒店实体与同城位置已核验，并作为本次行程唯一住宿基点。"
            if location_verified
            else "已作为本次行程唯一住宿基点；同城位置证据暂缺，入住前仍需核验。"
        ),
        route_fit=_route_fit(offer),
        selection_reason=_hotel_selection_reason(offer),
        main_tradeoff=_hotel_tradeoff(offer),
    )
    return (
        LodgingBaseline(mode="selected_offer", selected_offer_ref=offer.offer_ref),
        selected,
        observation.hotel_observation_id,
    )


def _lodging_policy(book: TaskBookV4, references: dict[str, str]) -> LodgingPolicy:
    if book.lodging_direction.not_applicable:
        return LodgingPolicy(mode="not_applicable")
    if book.lodging_direction.existing_booking is not None:
        fixed = next(
            reference
            for reference in fixed_commitments(book)
            if reference.commitment_id == book.lodging_direction.existing_booking.booking_id
        )
        return LodgingPolicy(mode="fixed", fixed_commitment_ref=fixed)
    return LodgingPolicy(
        mode="search",
        preferred_area_refs=tuple(key for key in references if key.startswith("area:")),
        selection_objectives=(
            "minimize_total_commute",
            "transit_convenience",
            "better_value",
        ),
        budget_constraint_ref=("lodging_budget" if "lodging_budget" in references else None),
        facility_constraint_refs=tuple(key for key in references if key.startswith("facility:")),
    )


def _pace_profile(book: TaskBookV4) -> Literal["relaxed", "balanced", "intensive"]:
    text = " ".join(item.value for item in book.pace_and_transport.pace_preferences)
    relaxed = ("轻松", "松弛", "慢", "长辈", "老人", "少走", "亲子")
    intensive = ("紧凑", "特种兵", "尽量多", "高强度", "充实")
    if not any(token in text for token in (*relaxed, *intensive, "平衡", "适中")):
        text = _cold_start_default(book, "步调")
    if any(token in text for token in relaxed):
        return "relaxed"
    if any(token in text for token in intensive):
        return "intensive"
    return "balanced"


def _cold_start_default(book: TaskBookV4, topic: str) -> str:
    return " ".join(
        item.value
        for item in book.tradeoffs_and_assumptions
        if item.value.startswith("长期默认") and f"｜{topic}：" in item.value
    )


def preferred_transport_modes(
    book: TaskBookV4,
) -> tuple[Literal["public_transit", "taxi", "walking", "driving"], ...]:
    # The combined transport_and_pace semantic operation historically stores
    # its grounded text in pace_preferences. Do not silently discard its mode.
    text = " ".join(
        item.value
        for item in (
            *book.pace_and_transport.transport_preferences,
            *book.pace_and_transport.pace_preferences,
        )
    )
    if not any(
        token in text
        for token in (
            "自驾",
            "开车",
            "租车",
            "地铁",
            "公交",
            "公共交通",
            "打车",
            "出租车",
            "网约车",
            "步行为主",
            "主要步行",
        )
    ):
        text = _cold_start_default(book, "出行")
    if any(token in text for token in ("自驾", "开车", "租车")):
        return "driving", "taxi", "walking"
    if any(token in text for token in ("地铁", "公交", "公共交通")):
        return "public_transit", "walking", "taxi"
    if any(token in text for token in ("打车", "出租车", "网约车")):
        return "taxi", "walking"
    if any(token in text for token in ("步行为主", "主要步行")):
        return "walking", "public_transit", "taxi"
    return "taxi", "walking", "public_transit"


def _core_experience_summary(book: TaskBookV4) -> str:
    goals = "、".join(item.value for item in book.travelers_and_trip_goal.trip_goals[:3])
    destination = book.destination_and_dates.destination_name
    return (
        f"围绕{destination}的{goals}安排每日体验，并优先减少无效往返。"
        if goals
        else f"围绕{destination}的已确认偏好安排每日体验，并优先减少无效往返。"
    )


def _allowed_candidate_keys(catalog: PlannerReferenceCatalog) -> tuple[str, ...]:
    return tuple(
        key
        for key, entry in catalog.candidates.items()
        if entry.selection_permission != "forbidden"
        and entry.eligibility not in {"unavailable", "excluded"}
    )


def _bookings(book: TaskBookV4) -> dict[str, BookingReference]:
    values = {booking.booking_id: booking for booking in book.existing_bookings}
    if book.lodging_direction.existing_booking is not None:
        values[book.lodging_direction.existing_booking.booking_id] = (
            book.lodging_direction.existing_booking
        )
    return values


def _fixed_item_kind(
    commitment_kind: str,
) -> Literal["fixed_event", "arrival", "departure"]:
    if commitment_kind == "arrival":
        return "arrival"
    if commitment_kind == "departure":
        return "departure"
    return "fixed_event"


def _without_unrequested_extra_meal(
    stops: list[ModelPlanStop], catalog: PlannerReferenceCatalog, book: TaskBookV4
) -> list[ModelPlanStop]:
    """Compile missing mechanical labels, never invent a third meal from a POI."""
    occupied = {
        stop.meal_slot
        for stop in stops
        if stop.meal_slot
        and stop.candidate_key in catalog.candidates
        and catalog.candidates[stop.candidate_key].entity_kind is CandidateEntityKind.RESTAURANT
    }
    output = []
    for stop in stops:
        entry = catalog.candidates.get(stop.candidate_key)
        if (
            entry is None
            or entry.entity_kind is not CandidateEntityKind.RESTAURANT
            or stop.meal_slot
        ):
            output.append(stop)
            continue
        main_slots: tuple[Literal["lunch", "dinner"], ...] = ("lunch", "dinner")
        available = [slot for slot in main_slots if slot not in occupied]
        if not available:
            if entry.commitment_level in {CommitmentLevel.STRONG, CommitmentLevel.IMMUTABLE}:
                raise PlannerGuardError(
                    "planner_plan_restaurant_requires_main_meal:path=days[].stops:"
                    f"candidate_key={stop.candidate_key}:repair=只能替换某天午餐或晚餐，不能增加第三餐"
                )
            continue
        slot = _meal_slot(stop.part_of_day)
        slot = slot if slot in available else available[0]
        occupied.add(slot)
        output.append(stop.model_copy(update={"meal_slot": slot}))
    return output


def _meal_slot(
    part_of_day: str,
) -> Literal["lunch", "dinner"]:
    return "dinner" if part_of_day in {"afternoon", "evening"} else "lunch"


def _route_fit(offer: HotelOfferObservation) -> str:
    durations = [
        item.duration_minutes
        for item in offer.commute_to_clusters
        if item.duration_minutes is not None
    ]
    if not durations:
        return "与主要活动簇的实时通勤耗时待补充。"
    return f"到已核验活动簇的单程通勤约 {min(durations)}–{max(durations)} 分钟。"


def _hotel_selection_reason(offer: HotelOfferObservation) -> str:
    evidence = ["已核验的同城酒店实体与位置"]
    if any(item.duration_minutes is not None for item in offer.commute_to_clusters):
        evidence.append("活动簇通勤数据")
    if offer.reference_price is not None:
        evidence.append("每间夜列表参考价")
    return (
        "依据本轮" + "、".join(evidence) + "确定这一家住宿；"
        "未核验的档次、房型、设施、实时房态与入住总价不作为已满足事实。"
    )


def _hotel_tradeoff(offer: HotelOfferObservation) -> str:
    if offer.availability_status == "unknown":
        return "酒店身份和位置已核验，但具体房型库存与可订状态仍需实时确认。"
    if offer.total_price is None:
        return "当前入住总价仍待确认，已有参考价不代表可订含税总价。"
    return "最终房型、取消规则和到店体验仍以预订页实时信息为准。"
