"""Deterministic, issue-oriented validation for materialized V4 Planner drafts."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from functools import partial
from typing import Literal
from uuid import UUID

from backend.agent.model_gateway import ModelCancellation
from backend.agent.planner.timing_quality import (
    MEAL_START_WINDOWS,
    clock_overflow_minutes,
    dining_commute_issues,
    explicit_end_deadline,
    meal_time_violations,
    missing_concrete_meals,
    schedule_coverage_issues,
    schedule_quality_gaps,
)
from backend.agent.planner.workspace import PlannerGuardError, server_id
from backend.contracts.daily_scheduling import SchedulePauseKind
from backend.contracts.itinerary_draft import (
    CostValidationDraft,
    DraftScheduledActivity,
    DraftScheduledDay,
    ScheduleValidationDraft,
)
from backend.contracts.itinerary_validation import (
    ItineraryValidationResult,
    RepairAction,
    ValidationIssue,
    ValidationIssueCode,
    ValidationSeverity,
    ValidationStatus,
    ValidationTargetKind,
)
from backend.contracts.v4.plan_change import (
    validate_hotel_recommendations_against_observation,
    validate_selected_hotel_against_observation,
)
from backend.contracts.v4.planner_draft import DraftItem, WorkingItineraryDay
from backend.contracts.v4.planner_observations import (
    PlannerValidationIssue,
    PlannerValidationObservation,
    SpatialRouteEndpoint,
)
from backend.contracts.v4.planner_refs import CandidateRef
from backend.contracts.v4.planner_workspace import PlannerWorkspaceState
from backend.contracts.v4.task_book import EvidenceBackedText, TaskBookV4
from backend.persistence.outbox_repository import canonical_json_hash
from backend.providers.contracts import HoursDayStatus, HoursInterval

VALIDATOR_VERSION = "v4-validator-3-meal-clock"
MAX_PARALLEL_VALIDATORS = 4

AllowedAction = Literal[
    "request_evidence",
    "move_item",
    "reorder_item",
    "replace_item",
    "remove_item",
    "change_window",
    "change_transport",
    "change_hotel",
    "ask_user",
]


@dataclass(frozen=True)
class PlannerValidationBundle:
    observation: PlannerValidationObservation
    legacy_report: ItineraryValidationResult


@dataclass(frozen=True)
class _DraftActivity:
    day: WorkingItineraryDay
    item: DraftItem
    materialized: DraftScheduledActivity | None


class PlannerDraftValidator:
    """Validate independent domains concurrently, then merge in stable order."""

    def __init__(
        self,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        max_parallel_validators: int = MAX_PARALLEL_VALIDATORS,
    ) -> None:
        if max_parallel_validators < 1:
            raise ValueError("max_parallel_validators must be positive")
        self._clock = clock
        self._max_parallel_validators = max_parallel_validators

    async def validate(
        self,
        workspace: PlannerWorkspaceState,
        book: TaskBookV4,
        cancellation: ModelCancellation,
    ) -> PlannerValidationBundle:
        draft = workspace.working_itinerary
        schedule = workspace.materialized_schedule
        cost = workspace.cost_draft
        if draft is None or schedule is None or cost is None:
            raise PlannerGuardError("planner_validation_prerequisites_missing")
        _validate_artifact_boundary(workspace, book, schedule, cost)
        now = self._aware_now()
        semaphore = asyncio.Semaphore(self._max_parallel_validators)

        async def run(
            function: Callable[[], tuple[PlannerValidationIssue, ...]],
        ) -> tuple[PlannerValidationIssue, ...]:
            async with semaphore:
                cancellation.raise_if_cancelled("planner_validate_domain")
                return await asyncio.to_thread(function)

        tasks = [run(partial(_validate_day, day, workspace, book, now)) for day in draft.days]
        tasks.extend(
            (
                run(lambda: _validate_hotel(workspace, book, now)),
                run(lambda: _validate_cost(workspace, book)),
                run(lambda: _validate_unassigned(workspace)),
                run(lambda: _validate_internal_readiness_gaps(workspace)),
                run(lambda: _validate_unverified_task_book_constraints(workspace, book)),
                run(lambda: _validate_schedule_quality(workspace, book)),
            )
        )
        groups = await asyncio.gather(*tasks)
        cancellation.raise_if_cancelled("planner_validate_merge")
        issues = tuple(sorted((issue for group in groups for issue in group), key=_issue_sort_key))
        result = _validation_result(issues)
        next_scope = workspace.current_scope.model_copy(
            update={"workspace_revision": workspace.workspace_revision + 1}
        )
        projection = {
            "draft_id": draft.draft_id,
            "draft_revision": draft.draft_revision,
            "draft_digest": draft.content_digest,
            "schedule_request_id": str(schedule.request_id),
            "cost_request_id": str(cost.request_id),
            "issues": [issue.model_dump(mode="json") for issue in issues],
        }
        observation = PlannerValidationObservation(
            observation_id=server_id(
                workspace.generation_id,
                draft.draft_id,
                draft.draft_revision,
                "validation",
            ),
            scope=next_scope,
            draft_id=draft.draft_id,
            draft_revision=draft.draft_revision,
            materialized_schedule_id=str(schedule.request_id),
            materialized_schedule_revision=draft.draft_revision,
            cost_draft_id=str(cost.request_id),
            cost_draft_revision=draft.draft_revision,
            validator_version=VALIDATOR_VERSION,
            validation_fingerprint=canonical_json_hash(projection),
            result=result,
            issues=issues,
            globally_affected_dates=tuple(
                sorted(
                    {
                        service_date
                        for issue in issues
                        if issue.scope_kind == "global"
                        for service_date in issue.affected_dates
                    }
                )
            ),
            checked_at=now,
        )
        legacy = _legacy_report(observation, schedule, cost, now)
        return PlannerValidationBundle(observation=observation, legacy_report=legacy)

    def _aware_now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Planner validator clock must return an aware datetime")
        return value.astimezone(UTC)


def _validate_artifact_boundary(
    workspace: PlannerWorkspaceState,
    book: TaskBookV4,
    schedule: ScheduleValidationDraft,
    cost: CostValidationDraft,
) -> None:
    draft = workspace.working_itinerary
    assert draft is not None
    expected = (
        UUID(workspace.trip_id),
        UUID(book.task_book_id),
        book.version,
        book.destination_and_dates.start_date,
        book.destination_and_dates.end_date,
    )
    schedule_boundary = (
        schedule.trip_id,
        schedule.task_book_id,
        schedule.task_book_revision,
        schedule.start_date,
        schedule.end_date,
    )
    cost_boundary = (
        cost.trip_id,
        cost.task_book_id,
        cost.task_book_revision,
        cost.start_date,
        cost.end_date,
    )
    if schedule_boundary != expected or cost_boundary != expected:
        raise PlannerGuardError("planner_materialized_artifact_boundary_stale")
    if cost.schedule_request_id != schedule.request_id:
        raise PlannerGuardError("planner_cost_schedule_reference_stale")
    if tuple(day.service_date for day in schedule.days) != tuple(
        day.service_date for day in draft.days
    ):
        raise PlannerGuardError("planner_materialized_dates_do_not_match_draft")


def _validate_day(
    day: WorkingItineraryDay,
    workspace: PlannerWorkspaceState,
    book: TaskBookV4,
    now: datetime,
) -> tuple[PlannerValidationIssue, ...]:
    schedule = workspace.materialized_schedule
    assert schedule is not None
    scheduled = next(value for value in schedule.days if value.service_date == day.service_date)
    issues: list[PlannerValidationIssue] = []
    activities = _draft_activities(day, scheduled)
    issues.extend(_validate_opening(activities, workspace, now))
    for activity in activities:
        value = activity.materialized
        if value is not None and (
            value.end_time.hour * 60
            + value.end_time.minute
            - value.start_time.hour * 60
            - value.start_time.minute
            != value.duration_minutes
        ):
            issues.append(
                _issue(
                    workspace,
                    code="time_overlap",
                    severity="error",
                    scope_kind="item",
                    affected_dates=(day.service_date,),
                    draft_item_ids=(activity.item.draft_item_id,),
                    violated_constraint_refs=("schedule:same_day_clock",),
                    message=(
                        f"{value.title} 的时间段不足以容纳 {value.duration_minutes} 分钟参观；"
                        "请调整到其他日期或合法时段，不得截断到午夜。"
                    ),
                    actions=("move_item", "change_window"),
                )
            )
    issues.extend(_validate_routes(day, workspace))
    if clock_overflow_minutes(scheduled):
        issues.append(
            _issue(
                workspace,
                code="time_overlap",
                severity="error",
                scope_kind="day",
                affected_dates=(day.service_date,),
                draft_item_ids=tuple(item.draft_item_id for item in day.ordered_items),
                violated_constraint_refs=("schedule:same_day_clock",),
                message="当天活动、交通或返酒店缓冲跨越午夜，时钟不能截成23:59；请提前出发、改用更快交通或跨日重新分配。",
                actions=("move_item", "change_window", "change_transport"),
            )
        )
    issues.extend(_validate_daily_capacity(day, scheduled, workspace, book))
    issues.extend(_validate_meals(day, scheduled, workspace))
    return tuple(issues)


def _draft_activities(
    day: WorkingItineraryDay,
    scheduled: DraftScheduledDay,
) -> tuple[_DraftActivity, ...]:
    by_activity_id = {str(item.activity_id): item for item in scheduled.activities}

    def envelope(item: DraftItem) -> DraftScheduledActivity | None:
        first = by_activity_id.get(server_id(item.draft_item_id, day.service_date, "activity"))
        if not item.onsite_lunch or first is None:
            return first
        continued = by_activity_id.get(
            server_id(item.draft_item_id, day.service_date, "continued-visit")
        )
        pause = next(
            (
                p
                for p in scheduled.pauses
                if str(p.pause_id)
                == server_id(item.draft_item_id, day.service_date, "onsite-lunch")
            ),
            None,
        )
        if (
            continued is None
            or pause is None
            or continued.node_id != first.node_id
            or continued.place_id != first.place_id
            or first.end_time != pause.start_time
            or pause.end_time != continued.start_time
            or first.duration_minutes <= 0
            or continued.duration_minutes <= 0
        ):
            raise PlannerGuardError("planner_onsite_visit_segments_invalid")
        # Validate one continuous entry/exit envelope, not a fictitious second
        # admission after last-entry time. Published durations remain pure visits.
        elapsed = (
            continued.end_time.hour * 60
            + continued.end_time.minute
            - first.start_time.hour * 60
            - first.start_time.minute
        )
        if elapsed != first.duration_minutes + pause.duration_minutes + continued.duration_minutes:
            raise PlannerGuardError("planner_onsite_visit_duration_invalid")
        return first.model_copy(
            update={"end_time": continued.end_time, "duration_minutes": elapsed}
        )

    return tuple(
        _DraftActivity(
            day=day,
            item=item,
            materialized=envelope(item),
        )
        for item in day.ordered_items
    )


def _validate_opening(
    activities: tuple[_DraftActivity, ...],
    workspace: PlannerWorkspaceState,
    now: datetime,
) -> tuple[PlannerValidationIssue, ...]:
    hours_by_entity = {item.canonical_entity_id: item for item in workspace.hours_evidence}
    result: list[PlannerValidationIssue] = []
    for entry in activities:
        item = entry.item
        if not isinstance(item.object_ref, CandidateRef):
            continue
        candidate = item.object_ref
        evidence = hours_by_entity.get(candidate.canonical_entity_id)
        if evidence is None or evidence.expires_at <= now:
            result.append(
                _issue(
                    workspace,
                    code="stale_evidence",
                    severity="warning",
                    scope_kind="item",
                    affected_dates=(entry.day.service_date,),
                    draft_item_ids=(item.draft_item_id,),
                    candidate_refs=(candidate,),
                    fact_reference_ids=(evidence.fact_reference_id,) if evidence else (),
                    violated_constraint_refs=("opening_hours:freshness",),
                    message="地点营业资料缺失或已经过期，先作为待核验提示。",
                    actions=("request_evidence",),
                )
            )
            continue
        date_hours = next(
            (value for value in evidence.days if value.service_date == entry.day.service_date),
            None,
        )
        if date_hours is None or date_hours.status is HoursDayStatus.UNKNOWN:
            result.append(
                _issue(
                    workspace,
                    code="stale_evidence",
                    severity="warning",
                    scope_kind="item",
                    affected_dates=(entry.day.service_date,),
                    draft_item_ids=(item.draft_item_id,),
                    candidate_refs=(candidate,),
                    fact_reference_ids=(evidence.fact_reference_id,),
                    violated_constraint_refs=("opening_hours:verified_date",),
                    message="该日期的营业资料仍未知，行程可以先发布并保留核验提示。",
                    actions=("request_evidence", "move_item", "replace_item"),
                )
            )
            continue
        if date_hours.status is HoursDayStatus.CONFLICT:
            result.append(
                _issue(
                    workspace,
                    code="opening_conflict",
                    severity="error",
                    scope_kind="item",
                    affected_dates=(entry.day.service_date,),
                    draft_item_ids=(item.draft_item_id,),
                    candidate_refs=(candidate,),
                    fact_reference_ids=(evidence.fact_reference_id,),
                    violated_constraint_refs=("opening_hours:conflicting_sources",),
                    message="该日期的营业资料相互冲突，不能据此确认活动时段。",
                    actions=("request_evidence", "move_item", "replace_item"),
                )
            )
            continue
        materialized = entry.materialized
        if (
            date_hours.status is HoursDayStatus.CLOSED
            or materialized is None
            or not _activity_inside_hours(materialized, date_hours.intervals)
        ):
            actions: tuple[AllowedAction, ...] = (
                "change_window",
                "move_item",
                "reorder_item",
                "replace_item",
            )
            result.append(
                _issue(
                    workspace,
                    code="opening_conflict",
                    severity="error",
                    scope_kind="item",
                    affected_dates=(entry.day.service_date,),
                    draft_item_ids=(item.draft_item_id,),
                    candidate_refs=(candidate,),
                    fact_reference_ids=(evidence.fact_reference_id,),
                    violated_constraint_refs=("opening_hours:date_window",),
                    message=(
                        f"{materialized.title} 计划 {materialized.start_time:%H:%M}–"
                        f"{materialized.end_time:%H:%M}；该日真实开放为"
                        + (
                            "、".join(
                                f"{interval.opens_at:%H:%M}–{interval.closes_at:%H:%M}"
                                + (
                                    f"（{interval.last_entry_at:%H:%M}停止入场）"
                                    if interval.last_entry_at
                                    else ""
                                )
                                for interval in date_hours.intervals
                            )
                            if date_hours.intervals
                            else "关闭"
                        )
                        + "。调整实际时段或更换允许替换的地点；仅交换日期未必能解决营业限制。"
                    )
                    if materialized is not None
                    else "该活动没有可核验的实际时段，需重新安排。",
                    actions=actions,
                )
            )
    return tuple(result)


def _validate_routes(
    day: WorkingItineraryDay,
    workspace: PlannerWorkspaceState,
) -> tuple[PlannerValidationIssue, ...]:
    endpoints = [_item_endpoint(item) for item in day.ordered_items]
    boundary = _hotel_endpoint(workspace)
    if boundary is not None and endpoints:
        endpoints = [boundary, *endpoints, boundary]
    route_edges = (
        *workspace.route_evidence,
        *(workspace.spatial_observation.route_edges if workspace.spatial_observation else ()),
    )
    result: list[PlannerValidationIssue] = []
    for index, (origin, destination) in enumerate(zip(endpoints, endpoints[1:], strict=False)):
        chosen_mode = next(
            (
                item.transport_mode
                for item in day.route_mode_selections
                if item.origin == origin and item.destination == destination
            ),
            None,
        )
        matching = [
            edge
            for edge in route_edges
            if edge.origin == origin
            and edge.destination == destination
            and (
                edge.transport_mode == chosen_mode
                if chosen_mode
                else edge.transport_mode in day.transport_preferences
            )
        ]
        available = next(
            (
                edge
                for edge in matching
                if edge.status == "available" and edge.duration_minutes is not None
            ),
            None,
        )
        if available is not None:
            continue
        edge_ids = tuple(edge.route_edge_id for edge in matching) or (
            server_id(
                "route-gap",
                day.service_date,
                index,
                origin.kind,
                origin.reference_id,
                destination.kind,
                destination.reference_id,
            ),
        )
        facts = tuple(dict.fromkeys(ref for edge in matching for ref in edge.fact_reference_ids))
        allowed_actions: tuple[AllowedAction, ...] = (
            ("request_evidence",)
            if not matching
            else ("request_evidence", "reorder_item", "change_transport")
        )
        result.append(
            _issue(
                workspace,
                code="route_unavailable",
                severity="warning",
                scope_kind="route_edge",
                affected_dates=(day.service_date,),
                route_edge_ids=edge_ids,
                fact_reference_ids=facts,
                violated_constraint_refs=("route:exact_adjacent_pair",),
                message="相邻安排暂缺当前交通方式的 Provider 路线结果，先作为待核验提示。",
                actions=allowed_actions,
            )
        )
    return tuple(result)


def _validate_daily_capacity(
    day: WorkingItineraryDay,
    scheduled: DraftScheduledDay,
    workspace: PlannerWorkspaceState,
    book: TaskBookV4 | None = None,
) -> tuple[PlannerValidationIssue, ...]:
    strategy = workspace.planning_strategy
    assert strategy is not None
    result: list[PlannerValidationIssue] = []
    visits = [item for item in day.ordered_items if item.item_kind == "visit"]
    maximum = strategy.daily_capacity_policy.major_activity_target.maximum
    end_limit = strategy.daily_capacity_policy.preferred_end_window.latest
    deadline = explicit_end_deadline(book, day.service_date) if book is not None else None
    hard_over_time = deadline is not None and scheduled.end_time > deadline[0]
    # A modest overrun of a preferred end is normal, not removal authority.
    preferred_overrun = (
        (
            scheduled.end_time.hour * 60
            + scheduled.end_time.minute
            - (end_limit.hour * 60 + end_limit.minute)
        )
        if end_limit is not None
        else 0
    )
    pace_threshold = {
        "relaxed": 10 * 60,
        "balanced": 11 * 60,
        "intensive": 12 * 60,
        "custom": 10 * 60,
    }[strategy.daily_capacity_policy.pace_profile]
    hard_conflict = len(visits) > maximum or hard_over_time
    if hard_conflict or scheduled.active_minutes > pace_threshold or preferred_overrun > 60:
        late_visits = [
            activity
            for activity in _draft_activities(day, scheduled)
            if activity.item.item_kind == "visit"
            and activity.materialized is not None
            and deadline is not None
            and activity.materialized.end_time > deadline[0]
        ]
        # A late-window visit will still finish late if an unrelated morning
        # stop is deleted. Name the actual overrun before broad day-load targets.
        targets = [activity.item for activity in late_visits] or visits
        soft = [item for item in targets if item.commitment_level in {"soft", "filler", "neutral"}]
        late_summary = "、".join(
            f"{activity.materialized.title} "
            f"{activity.materialized.start_time.strftime('%H:%M')}–"
            f"{activity.materialized.end_time.strftime('%H:%M')}"
            for activity in late_visits
            if activity.materialized is not None
        )
        actions: tuple[AllowedAction, ...] = (
            ("move_item", "remove_item") if soft else ("move_item", "ask_user")
        )
        result.append(
            _issue(
                workspace,
                code="pace_limit_exceeded",
                severity="error" if hard_conflict else "warning",
                scope_kind="day",
                affected_dates=(day.service_date,),
                draft_item_ids=tuple(item.draft_item_id for item in (soft or targets)),
                candidate_refs=tuple(
                    item.object_ref
                    for item in (soft or targets)
                    if isinstance(item.object_ref, CandidateRef)
                ),
                violated_constraint_refs=(deadline[1],)
                if hard_over_time and deadline
                else ("pace:daily_capacity",),
                message=(
                    f"{day.service_date.isoformat()} 的主要景点 {len(visits)} 个"
                    f"（上限 {maximum} 个）；活动与交通合计 {scheduled.active_minutes} 分钟"
                    f"（节奏参考 {pace_threshold} 分钟）；"
                    f"当天结束 {scheduled.end_time.strftime('%H:%M')}"
                    + (
                        f"（用户硬截止 {deadline[0].strftime('%H:%M')}）"
                        if deadline is not None
                        else "（默认收尾为软目标）"
                    )
                    + (f"。直接超时活动：{late_summary}" if late_summary else "")
                    + (
                        "。优先调整时间和顺序；保留必去和固定预订，不要删除无关的上午景点。"
                        if hard_conflict
                        else "。可适度延后收尾；此软提示不授权删除景点。"
                    )
                ),
                actions=actions,
                user_authority_required=hard_conflict and not soft,
            )
        )
    walking_goal = strategy.daily_capacity_policy.walking_policy.goal
    threshold = {"minimize": 6_000, "balanced": 10_000, "no_preference": 15_000}[walking_goal]
    if scheduled.walking_m > threshold:
        result.append(
            _issue(
                workspace,
                code="walking_limit_exceeded",
                severity="error"
                if strategy.daily_capacity_policy.walking_policy.hard_limit_ref
                else "warning",
                scope_kind="day",
                affected_dates=(day.service_date,),
                route_edge_ids=tuple(str(item.leg_id) for item in scheduled.transport_legs),
                violated_constraint_refs=(
                    strategy.daily_capacity_policy.walking_policy.hard_limit_ref
                    or "walking:preference",
                ),
                message="当天步行距离超过当前节奏策略的校验阈值。",
                actions=("change_transport", "reorder_item"),
            )
        )
    return tuple(result)


def _validate_meals(
    day: WorkingItineraryDay,
    scheduled: DraftScheduledDay,
    workspace: PlannerWorkspaceState,
) -> tuple[PlannerValidationIssue, ...]:
    concrete = {
        item.meal_slot
        for item in day.ordered_items
        if item.item_kind == "dining" and item.meal_slot is not None
    }
    pauses = [item for item in scheduled.pauses if item.kind is SchedulePauseKind.MEAL]
    pause_slots = {_meal_slot(item.start_time) for item in pauses}
    result: list[PlannerValidationIssue] = []
    for problem in meal_time_violations(day, scheduled):
        result.append(
            _issue(
                workspace,
                code="meal_constraint_violation",
                severity="error",
                scope_kind="item",
                affected_dates=(day.service_date,),
                draft_item_ids=(str(problem["draft_item_id"]),),
                violated_constraint_refs=(f"meal:{problem['meal']}",),
                message=(
                    f"{problem['path']}：{problem['name']} 的 {problem['meal']} 被排到 "
                    f"{problem['actual_start']}，合理开始窗口为 {problem['start_window']}；"
                    "提前出发、改用省时交通、调整前后景点或替换非指定餐厅，不要推迟正餐。"
                ),
                actions=(
                    "move_item",
                    "reorder_item",
                    "replace_item",
                    "change_transport",
                    "change_window",
                ),
            )
        )
    for meal in day.dining_goals:
        if meal not in concrete and meal not in pause_slots:
            result.append(
                _issue(
                    workspace,
                    code="meal_gap",
                    severity="error",
                    scope_kind="day",
                    affected_dates=(day.service_date,),
                    violated_constraint_refs=(f"meal:{meal}",),
                    message="当天缺少草稿要求的用餐时段。",
                    actions=("reorder_item", "change_window"),
                )
            )
    for pause in pauses:
        slot = _meal_slot(pause.start_time)
        if slot is None:
            result.append(
                _issue(
                    workspace,
                    code="meal_constraint_violation",
                    severity="error",
                    scope_kind="day",
                    affected_dates=(day.service_date,),
                    violated_constraint_refs=("meal:timing",),
                    message="用餐暂停落在通常用餐窗口之外。",
                    actions=("reorder_item", "change_window"),
                )
            )
    return tuple(result)


def _validate_hotel(
    workspace: PlannerWorkspaceState,
    book: TaskBookV4,
    now: datetime,
) -> tuple[PlannerValidationIssue, ...]:
    draft = workspace.working_itinerary
    assert draft is not None
    nights = (book.destination_and_dates.end_date - book.destination_and_dates.start_date).days
    if nights == 0 or book.lodging_direction.not_applicable:
        return ()
    observation = workspace.hotel_observation
    baseline = draft.lodging_baseline
    all_dates = tuple(day.service_date for day in draft.days)
    if baseline.mode == "fixed":
        fixed_ref = baseline.fixed_commitment_ref
        if observation is not None and (
            observation.mode != "fixed_booking_verification"
            or observation.fixed_booking is None
            or observation.fixed_booking.commitment_ref != fixed_ref
        ):
            return (
                _issue(
                    workspace,
                    code="hotel_constraint_conflict",
                    severity="error",
                    scope_kind="hotel",
                    affected_dates=all_dates,
                    violated_constraint_refs=("hotel:fixed_booking_reference",),
                    message="固定酒店的核验结果没有绑定任务书中的同一预订引用。",
                    actions=("request_evidence",),
                    user_authority_required=False,
                ),
            )
        fixed_booking = observation.fixed_booking if observation is not None else None
        if fixed_booking is None or fixed_booking.verification_status != "verified":
            return (
                _issue(
                    workspace,
                    code="hotel_unavailable",
                    severity="warning",
                    scope_kind="hotel",
                    affected_dates=all_dates,
                    violated_constraint_refs=("hotel:identity_verification",),
                    message=(
                        "用户确认的固定酒店仍被保留，但 Provider 暂时无法核验其同城实体；"
                        "这不等于预订无效，发布后仍需补充证据。"
                        if fixed_booking is None
                        or fixed_booking.verification_status == "unavailable"
                        else "用户确认的固定酒店仍被保留，但 Provider 返回多个可能实体；"
                        "这不等于预订无效，发布后仍需补充证据。"
                    ),
                    actions=("request_evidence",),
                    user_authority_required=False,
                ),
            )
        return ()
    if baseline.mode == "unresolved":
        return (
            _issue(
                workspace,
                code="hotel_night_gap",
                severity="warning",
                scope_kind="hotel",
                affected_dates=all_dates,
                violated_constraint_refs=("hotel:overnight_coverage",),
                message=(
                    "酒店 Provider 当前不可用，正式行程保留住宿待核验提示。"
                    if baseline.unresolved_reason == "provider_unavailable"
                    else "当前没有可核验的酒店结果，正式行程保留住宿待补充提示。"
                ),
                actions=("request_evidence",),
                user_authority_required=False,
            ),
        )
    if baseline.mode != "selected_offer":
        return (
            _issue(
                workspace,
                code="hotel_constraint_conflict",
                severity="error",
                scope_kind="hotel",
                affected_dates=all_dates,
                violated_constraint_refs=("hotel:lodging_baseline_mode",),
                message="过夜行程的住宿基线模式与已确认任务书冲突。",
                actions=("change_hotel",),
            ),
        )
    if observation is None:
        return (
            _issue(
                workspace,
                code="hotel_constraint_conflict",
                severity="error",
                scope_kind="hotel",
                affected_dates=all_dates,
                violated_constraint_refs=("hotel:observation_reference",),
                message="已选酒店缺少其引用的 HotelObservation，属于内部引用错误。",
                actions=("request_evidence", "change_hotel"),
            ),
        )
    selected_ref = baseline.selected_offer_ref
    assert selected_ref is not None
    selected = next(
        (offer for offer in observation.offers if offer.offer_ref == selected_ref), None
    )
    if selected is None:
        return (
            _issue(
                workspace,
                code="hotel_constraint_conflict",
                severity="error",
                scope_kind="hotel",
                affected_dates=all_dates,
                hotel_offer_refs=(selected_ref,),
                violated_constraint_refs=("hotel:offer_reference",),
                message="已选酒店键不属于当前 HotelObservation。",
                actions=("change_hotel",),
            ),
        )
    if selected.availability_status == "unavailable":
        return (
            _issue(
                workspace,
                code="hotel_unavailable",
                severity="error",
                scope_kind="hotel",
                affected_dates=all_dates,
                hotel_offer_refs=(selected_ref,),
                violated_constraint_refs=("hotel:explicitly_unavailable",),
                message="已选酒店的当前 Provider 结果明确标记为不可用。",
                actions=("request_evidence", "change_hotel"),
            ),
        )

    result: list[PlannerValidationIssue] = []
    hotel_evidence_stale = (
        observation.expires_at is not None
        and observation.expires_at <= now
        or selected.expires_at is not None
        and selected.expires_at <= now
    )
    if selected.availability_status == "unknown" or hotel_evidence_stale:
        result.append(
            _issue(
                workspace,
                code="hotel_unavailable",
                severity="warning",
                scope_kind="hotel",
                affected_dates=all_dates,
                hotel_offer_refs=(selected_ref,),
                violated_constraint_refs=(
                    "hotel:availability_unknown"
                    if selected.availability_status == "unknown"
                    else "hotel:availability_stale",
                ),
                message=(
                    "已选酒店仅有地点或参考价，实时房态仍未知；这不会阻断行程发布。"
                    if selected.availability_status == "unknown"
                    else "已选酒店证据已经过期；行程可以发布，但入住前需要重新核验。"
                ),
                actions=("request_evidence", "change_hotel"),
            )
        )

    try:
        if (workspace.selected_hotel is None) == (workspace.hotel_recommendations is None):
            raise ValueError("selected lodging requires exactly one hotel representation")
        if workspace.selected_hotel is not None:
            validate_selected_hotel_against_observation(workspace.selected_hotel, observation)
            formal_ref = workspace.selected_hotel.hotel_offer_ref
        else:
            assert workspace.hotel_recommendations is not None
            validate_hotel_recommendations_against_observation(
                workspace.hotel_recommendations,
                observation,
            )
            formal_ref = workspace.hotel_recommendations.recommended_hotel.hotel_offer_ref
        if formal_ref != selected_ref:
            raise ValueError("formal hotel selection differs from lodging baseline")
    except ValueError:
        result.append(
            _issue(
                workspace,
                code="hotel_constraint_conflict",
                severity="error",
                scope_kind="hotel",
                affected_dates=all_dates,
                hotel_offer_refs=(selected_ref,),
                violated_constraint_refs=("hotel:formal_selection",),
                message="酒店正式选择结构不完整，或没有绑定当前 HotelObservation。",
                actions=("change_hotel",),
            )
        )
    covered_nights = sum(segment.nights for segment in observation.stay_segments)
    if covered_nights != nights:
        result.append(
            _issue(
                workspace,
                code="hotel_night_gap",
                severity="error",
                scope_kind="hotel",
                affected_dates=all_dates,
                hotel_offer_refs=(selected_ref,),
                violated_constraint_refs=("hotel:night_count",),
                message="酒店 Observation 覆盖晚数与旅行晚数不一致。",
                actions=("request_evidence", "change_hotel"),
            )
        )
    return tuple(result)


def _validate_cost(
    workspace: PlannerWorkspaceState,
    book: TaskBookV4,
) -> tuple[PlannerValidationIssue, ...]:
    cost = workspace.cost_draft
    draft = workspace.working_itinerary
    assert cost is not None and draft is not None
    result: list[PlannerValidationIssue] = []
    for summary in cost.categories:
        if summary.missing_item_count:
            result.append(
                _issue(
                    workspace,
                    code="missing_price",
                    severity="warning",
                    scope_kind="cost",
                    affected_dates=tuple(day.service_date for day in draft.days),
                    violated_constraint_refs=(f"cost:{summary.category.value}",),
                    message="该费用类别仍有真实价格缺口，缺失项没有按零元计算。",
                    actions=("request_evidence",),
                )
            )
    budget = book.lodging_direction.nightly_budget
    selected_ref = draft.lodging_baseline.selected_offer_ref
    observation = workspace.hotel_observation
    selected = next(
        (
            offer
            for offer in (observation.offers if observation else ())
            if selected_ref is not None and offer.offer_ref == selected_ref
        ),
        None,
    )
    nights = (book.destination_and_dates.end_date - book.destination_and_dates.start_date).days
    if budget is None or budget.maximum_minor is None or selected is None or nights <= 0:
        return tuple(result)

    affected_dates = tuple(day.service_date for day in draft.days)
    maximum_total = budget.maximum_minor * nights
    if selected.total_price is not None:
        if selected.total_price.currency != budget.currency:
            result.append(
                _issue(
                    workspace,
                    code="missing_price",
                    severity="warning",
                    scope_kind="cost",
                    affected_dates=affected_dates,
                    hotel_offer_refs=(selected.offer_ref,),
                    fact_reference_ids=selected.room_and_price_fact_refs,
                    violated_constraint_refs=("lodging:nightly_budget_currency",),
                    message="酒店总价与每晚预算币种不同，不能判定是否满足预算。",
                    actions=("request_evidence", "change_hotel"),
                )
            )
        elif selected.total_price.amount_minor > maximum_total:
            result.append(
                _issue(
                    workspace,
                    code="budget_exceeded",
                    severity="error",
                    scope_kind="cost",
                    affected_dates=affected_dates,
                    hotel_offer_refs=(selected.offer_ref,),
                    fact_reference_ids=selected.room_and_price_fact_refs,
                    violated_constraint_refs=("lodging:nightly_budget",),
                    message="当前酒店总价超过已确认的每晚住宿预算上限乘以入住晚数。",
                    actions=("change_hotel", "ask_user"),
                )
            )
        return tuple(result)

    reference_price = selected.reference_price
    if reference_price is None:
        result.append(
            _issue(
                workspace,
                code="missing_price",
                severity="warning",
                scope_kind="cost",
                affected_dates=affected_dates,
                hotel_offer_refs=(selected.offer_ref,),
                violated_constraint_refs=("lodging:bookable_total_unknown",),
                message="酒店缺少可预订总价和每间夜参考价，不能判定是否满足预算。",
                actions=("request_evidence", "change_hotel"),
            )
        )
    elif reference_price.currency != budget.currency:
        result.append(
            _issue(
                workspace,
                code="missing_price",
                severity="warning",
                scope_kind="cost",
                affected_dates=affected_dates,
                hotel_offer_refs=(selected.offer_ref,),
                fact_reference_ids=selected.reference_price_fact_refs,
                violated_constraint_refs=("lodging:nightly_budget_currency",),
                message="酒店每间夜参考价与每晚预算币种不同，不能判定是否满足预算。",
                actions=("request_evidence", "change_hotel"),
            )
        )
    elif reference_price.minimum_minor * nights > maximum_total:
        result.append(
            _issue(
                workspace,
                code="budget_exceeded",
                severity="error",
                scope_kind="cost",
                affected_dates=affected_dates,
                hotel_offer_refs=(selected.offer_ref,),
                fact_reference_ids=selected.reference_price_fact_refs,
                violated_constraint_refs=("lodging:nightly_budget",),
                message=(
                    "酒店每间夜参考价下限乘以入住晚数，已超过每晚预算上限乘以入住晚数；"
                    "即使总价仍待核验，已知参考区间也明确超限。"
                ),
                actions=("change_hotel", "ask_user"),
            )
        )
    elif reference_price.maximum_minor * nights > maximum_total:
        result.append(
            _issue(
                workspace,
                code="missing_price",
                severity="warning",
                scope_kind="cost",
                affected_dates=affected_dates,
                hotel_offer_refs=(selected.offer_ref,),
                fact_reference_ids=selected.reference_price_fact_refs,
                violated_constraint_refs=("lodging:nightly_budget_reference_overlap",),
                message=(
                    "酒店每间夜参考价区间部分超过每晚预算上限；"
                    "缺少房型库存与可预订总价，不能判定满足预算。"
                ),
                actions=("request_evidence", "change_hotel"),
            )
        )
    else:
        result.append(
            _issue(
                workspace,
                code="missing_price",
                severity="warning",
                scope_kind="cost",
                affected_dates=affected_dates,
                hotel_offer_refs=(selected.offer_ref,),
                fact_reference_ids=selected.reference_price_fact_refs,
                violated_constraint_refs=("lodging:bookable_total_unknown",),
                message=(
                    "酒店每间夜参考价区间未超过预算上限，但它不是可预订总价；"
                    "不能据此宣称实际住宿费用已满足预算。"
                ),
                actions=("request_evidence", "change_hotel"),
            )
        )
    return tuple(result)


def _validate_unassigned(
    workspace: PlannerWorkspaceState,
) -> tuple[PlannerValidationIssue, ...]:
    draft = workspace.working_itinerary
    assert draft is not None
    result: list[PlannerValidationIssue] = []
    for intent in draft.unassigned_intents:
        if intent.commitment_level != "strong":
            continue
        result.append(
            _issue(
                workspace,
                code="unscheduled_strong_intent",
                severity="warning" if workspace.best_effort_reasons else "blocking",
                scope_kind="item",
                affected_dates=tuple(day.service_date for day in draft.days),
                candidate_refs=(intent.candidate_ref,),
                fact_reference_ids=intent.supporting_observation_refs,
                violated_constraint_refs=("commitment:strong",),
                message=(
                    "尽力完成版仍未安排这项强意愿，原任务书要求保留，不能宣称已满足。"
                    if workspace.best_effort_reasons
                    else "强意愿仍未安排，程序不能替用户牺牲。"
                ),
                actions=("move_item", "replace_item", "ask_user"),
                user_authority_required=True,
            )
        )
    return tuple(result)


def _validate_internal_readiness_gaps(
    workspace: PlannerWorkspaceState,
) -> tuple[PlannerValidationIssue, ...]:
    """Keep Provider evidence gaps visible without turning them into fake facts."""

    observation = workspace.readiness_observation
    if observation is None:
        return ()
    affected_dates = (
        tuple(day.service_date for day in workspace.working_itinerary.days)
        if workspace.working_itinerary is not None
        else ()
    )
    return tuple(
        _issue(
            workspace,
            code="stale_evidence",
            severity="warning",
            scope_kind="global",
            affected_dates=issue.affected_dates or affected_dates,
            fact_reference_ids=issue.fact_reference_ids,
            violated_constraint_refs=(f"readiness:{issue.issue_id}",),
            message=(f"{issue.reason_summary} 本版行程未用同名地点替代，并保留为待核验项。"),
            actions=("request_evidence",),
        )
        for issue in observation.issues
        if not issue.user_authority_required
    )


def _validate_unverified_task_book_constraints(
    workspace: PlannerWorkspaceState,
    book: TaskBookV4,
) -> tuple[PlannerValidationIssue, ...]:
    """Expose free-text hard requirements that no deterministic evaluator proves."""

    affected_dates = (
        tuple(day.service_date for day in workspace.working_itinerary.days)
        if workspace.working_itinerary is not None
        else ()
    )
    result: list[PlannerValidationIssue] = []
    groups: tuple[
        tuple[
            str,
            list[EvidenceBackedText],
            Literal["global", "hotel"],
            str,
            str,
        ],
        ...,
    ] = (
        (
            "hard",
            book.hard_constraints,
            "global",
            "硬约束",
            "当前工具与结构化规则尚不能证明这项要求已满足",
        ),
        (
            "dietary",
            book.dining_direction.hard_requirements,
            "global",
            "饮食硬要求",
            "当前餐厅证据尚不能证明这项要求已满足",
        ),
        (
            "facility",
            book.lodging_direction.facility_requirements,
            "hotel",
            "酒店设施要求",
            "当前酒店 Provider 结果尚不能证明这项要求已满足",
        ),
    )
    for prefix, constraints, scope_kind, label, reason in groups:
        for index, constraint in enumerate(constraints):
            result.append(
                _issue(
                    workspace,
                    code="stale_evidence",
                    severity="warning",
                    scope_kind=scope_kind,
                    affected_dates=affected_dates,
                    violated_constraint_refs=(f"{prefix}:{index}",),
                    message=(
                        f"{label}“{constraint.value[:240]}”待核验：{reason}；"
                        "本版行程保留该要求，不能把缺少证据解释为已经满足。"
                    ),
                    actions=("request_evidence",),
                )
            )
    return tuple(result)


def _validate_schedule_quality(
    workspace: PlannerWorkspaceState, book: TaskBookV4
) -> tuple[PlannerValidationIssue, ...]:
    """Quality is executable feedback, not a fabricated hard user constraint."""
    result = []
    for meal in missing_concrete_meals(workspace):
        label = "午餐" if meal["meal"] == "lunch" else "晚餐"
        result.append(
            _issue(
                workspace,
                code="meal_gap",
                severity="warning",
                scope_kind="day",
                affected_dates=(date.fromisoformat(str(meal["date"])),),
                violated_constraint_refs=(
                    f"internal_schedule_quality:concrete_meal:{meal['meal']}",
                ),
                message=f"{meal['date']}{label}未绑定具体餐厅；用餐占位不代表已完成正餐安排。",
                actions=("request_evidence", "replace_item"),
            )
        )
    for issue in schedule_coverage_issues(workspace, book):
        period = "上午" if issue["period"] == "morning" else "下午"
        result.append(
            _issue(
                workspace,
                code="half_day_without_visit",
                severity="warning",
                scope_kind="day",
                affected_dates=(date.fromisoformat(str(issue["date"])),),
                violated_constraint_refs=(f"internal_schedule_quality:{issue['period']}",),
                message=(
                    f"{issue['date']}{period}实际游览仅{issue['visit_minutes']}分钟，尚不充足；"
                    f"{issue['path']}需深游、补入沿途景点或跨日平衡。"
                ),
                actions=("request_evidence", "move_item", "reorder_item"),
            )
        )
    for gap in schedule_quality_gaps(workspace, book):
        result.append(
            _issue(
                workspace,
                code="unfilled_sightseeing_gap",
                severity="warning",
                scope_kind="day",
                affected_dates=(date.fromisoformat(gap["date"]),),
                violated_constraint_refs=(f"internal_schedule_quality:gap:{gap['start']}",),
                message=(
                    f"{gap['date']} {gap['start']}–{gap['end']}仍有{gap['minutes']}分钟空档，"
                    "不能用绕路或标题冒充填补。"
                ),
                actions=("request_evidence", "move_item", "reorder_item"),
            )
        )
    candidates = workspace.candidate_pool.candidate_by_id()
    for detour in dining_commute_issues(workspace):
        entry = candidates[str(detour["candidate_id"])]
        result.append(
            _issue(
                workspace,
                code="route_cost_exceeded",
                severity="warning",
                scope_kind="item",
                affected_dates=(date.fromisoformat(str(detour["date"])),),
                draft_item_ids=(str(detour["draft_item_id"]),),
                candidate_refs=(entry.candidate_ref,),
                fact_reference_ids=tuple(detour["fact_reference_ids"]),
                violated_constraint_refs=("internal_schedule_quality:dining_detour",),
                message=f"该非预约餐厅前后真实交通合计{detour['combined_minutes']}分钟，应比较沿途替换或改餐次，必吃意愿不豁免路线取舍。",
                actions=("replace_item", "move_item", "remove_item"),
            )
        )
    return tuple(result)


def _issue(
    workspace: PlannerWorkspaceState | None,
    *,
    code: str,
    severity: Literal["warning", "error", "blocking"],
    scope_kind: Literal["day", "item", "route_edge", "hotel", "cost", "global"],
    affected_dates: tuple[date, ...] = (),
    draft_item_ids: tuple[str, ...] = (),
    candidate_refs: tuple[CandidateRef, ...] = (),
    route_edge_ids: tuple[str, ...] = (),
    hotel_offer_refs: tuple[object, ...] = (),
    fact_reference_ids: tuple[str, ...] = (),
    violated_constraint_refs: tuple[str, ...] = (),
    message: str,
    actions: tuple[AllowedAction, ...],
    user_authority_required: bool = False,
) -> PlannerValidationIssue:
    seed = {
        "code": code,
        "scope_kind": scope_kind,
        "affected_dates": [value.isoformat() for value in affected_dates],
        "draft_item_ids": draft_item_ids,
        "route_edge_ids": route_edge_ids,
        "candidate_ids": [value.candidate_id for value in candidate_refs],
        "hotel_offer_ids": [getattr(value, "offer_id", "") for value in hotel_offer_refs],
        "constraints": violated_constraint_refs,
    }
    return PlannerValidationIssue.model_validate(
        {
            "issue_id": server_id(
                workspace.generation_id if workspace is not None else "validation",
                canonical_json_hash(seed),
            ),
            "code": code,
            "severity": severity,
            "scope_kind": scope_kind,
            "affected_dates": affected_dates,
            "draft_item_ids": draft_item_ids,
            "candidate_refs": candidate_refs,
            "route_edge_ids": route_edge_ids,
            "hotel_offer_refs": hotel_offer_refs,
            "fact_reference_ids": fact_reference_ids,
            "violated_constraint_refs": violated_constraint_refs,
            "message_summary": message,
            "allowed_actions": actions,
            "user_authority_required": user_authority_required,
        }
    )


def _validation_result(
    issues: tuple[PlannerValidationIssue, ...],
) -> Literal["passed", "repairable", "requires_user", "insufficient_evidence", "fatal"]:
    blocking = tuple(issue for issue in issues if issue.severity in {"error", "blocking"})
    if not blocking:
        return "passed"
    if any(
        issue.user_authority_required and "ask_user" in issue.allowed_actions for issue in blocking
    ):
        return "requires_user"
    repair_actions = {
        "move_item",
        "reorder_item",
        "replace_item",
        "remove_item",
        "change_window",
        "change_transport",
        "change_hotel",
    }
    if any(set(issue.allowed_actions) & repair_actions for issue in blocking):
        return "repairable"
    if any("request_evidence" in issue.allowed_actions for issue in blocking):
        return "insufficient_evidence"
    return "fatal"


def _legacy_report(
    observation: PlannerValidationObservation,
    schedule: ScheduleValidationDraft,
    cost: CostValidationDraft,
    now: datetime,
) -> ItineraryValidationResult:
    issues = tuple(_legacy_issue(issue) for issue in observation.issues)
    status = (
        ValidationStatus.BLOCKED
        if any(issue.severity is ValidationSeverity.HARD_CONFLICT for issue in issues)
        else ValidationStatus.REVIEW
        if issues
        else ValidationStatus.VALID
    )
    return ItineraryValidationResult(
        algorithm_version="4.0.0",
        request_id=UUID(server_id(observation.observation_id, "legacy-report")),
        trip_id=schedule.trip_id,
        input_state_version=schedule.input_state_version,
        task_book_id=schedule.task_book_id,
        task_book_revision=schedule.task_book_revision,
        schedule_request_id=schedule.request_id,
        cost_request_id=cost.request_id,
        status=status,
        issues=issues,
        generated_at=now,
    )


def _legacy_issue(issue: PlannerValidationIssue) -> ValidationIssue:
    code = {
        "opening_conflict": ValidationIssueCode.OPENING_HOURS_CONFLICT,
        "stale_evidence": ValidationIssueCode.OPENING_HOURS_UNKNOWN,
        "route_unavailable": ValidationIssueCode.PARTIAL_ROUTE,
        "walking_limit_exceeded": ValidationIssueCode.WALKING_LIMIT,
        "pace_limit_exceeded": ValidationIssueCode.DAILY_LOAD,
        "half_day_without_visit": ValidationIssueCode.DAILY_LOAD,
        "unfilled_sightseeing_gap": ValidationIssueCode.DAILY_LOAD,
        "meal_constraint_violation": ValidationIssueCode.MEAL_TIMING,
        "meal_gap": ValidationIssueCode.MEAL_TIMING,
        "hotel_night_gap": ValidationIssueCode.LODGING_COVERAGE,
        "hotel_unavailable": ValidationIssueCode.LODGING_COVERAGE,
        "hotel_constraint_conflict": ValidationIssueCode.LODGING_COVERAGE,
        "missing_price": ValidationIssueCode.MISSING_PRICE,
        "budget_exceeded": ValidationIssueCode.COST_INCONSISTENCY,
        "unscheduled_strong_intent": ValidationIssueCode.UNSCHEDULED_STRONG_DESIRE,
    }.get(issue.code, ValidationIssueCode.REFERENCE_CONFLICT)
    severity = (
        ValidationSeverity.HARD_CONFLICT
        if issue.severity in {"error", "blocking"}
        else ValidationSeverity.UNKNOWN_FACT
        if issue.code
        in {
            "missing_price",
            "stale_evidence",
            "route_unavailable",
            "hotel_unavailable",
            "hotel_night_gap",
        }
        else ValidationSeverity.SOFT_RISK
    )
    action = _legacy_action(issue.allowed_actions)
    target_kind = {
        "day": ValidationTargetKind.DAY,
        "item": ValidationTargetKind.TRIP,
        "route_edge": ValidationTargetKind.TRIP,
        "hotel": ValidationTargetKind.TRIP,
        "cost": ValidationTargetKind.TRIP,
        "global": ValidationTargetKind.TRIP,
    }[issue.scope_kind]
    return ValidationIssue(
        issue_id=UUID(issue.issue_id),
        code=code,
        severity=severity,
        target_kind=target_kind,
        message=issue.message_summary,
        service_date=issue.affected_dates[0] if issue.affected_dates else None,
        repairable=action is not RepairAction.NONE,
        repair_action=action,
        source_reference_ids=issue.fact_reference_ids,
    )


def _legacy_action(actions: tuple[AllowedAction, ...]) -> RepairAction:
    priorities = (
        ("move_item", RepairAction.REASSIGN_DAY),
        ("reorder_item", RepairAction.REORDER_DAY),
        ("change_transport", RepairAction.CHANGE_ROUTE),
        ("replace_item", RepairAction.REPLACE_CANDIDATE),
        ("change_window", RepairAction.ADJUST_TIME),
        ("request_evidence", RepairAction.REQUERY_FACT),
    )
    return next((mapped for action, mapped in priorities if action in actions), RepairAction.NONE)


def _activity_inside_hours(
    activity: DraftScheduledActivity,
    intervals: Iterable[HoursInterval],
) -> bool:
    for interval in intervals:
        opens_at = interval.opens_at
        closes_at = interval.closes_at
        last_entry_at = interval.last_entry_at
        if (
            opens_at <= activity.start_time
            and activity.end_time <= closes_at
            and (last_entry_at is None or activity.start_time <= last_entry_at)
        ):
            return True
    return False


def _item_endpoint(item: DraftItem) -> SpatialRouteEndpoint:
    if isinstance(item.object_ref, CandidateRef):
        return SpatialRouteEndpoint(kind="candidate", reference_id=item.object_ref.candidate_id)
    return SpatialRouteEndpoint(kind="fixed_commitment", reference_id=item.object_ref.commitment_id)


def _hotel_endpoint(workspace: PlannerWorkspaceState) -> SpatialRouteEndpoint | None:
    draft = workspace.working_itinerary
    assert draft is not None
    if draft.lodging_baseline.selected_offer_ref is not None:
        return SpatialRouteEndpoint(
            kind="hotel_offer",
            reference_id=draft.lodging_baseline.selected_offer_ref.offer_id,
        )
    if draft.lodging_baseline.fixed_commitment_ref is not None:
        return SpatialRouteEndpoint(
            kind="fixed_commitment",
            reference_id=draft.lodging_baseline.fixed_commitment_ref.commitment_id,
        )
    return None


def _meal_slot(value: time) -> str | None:
    minutes = value.hour * 60 + value.minute
    # Prefer full meals on shared endpoints (14:00 lunch, 17:00 dinner).
    return next(
        (
            meal
            for meal in ("breakfast", "lunch", "dinner", "snack")
            if MEAL_START_WINDOWS[meal][0] <= minutes <= MEAL_START_WINDOWS[meal][1]
        ),
        None,
    )


def _issue_sort_key(issue: PlannerValidationIssue) -> tuple[object, ...]:
    return (
        issue.affected_dates[0] if issue.affected_dates else date.max,
        {"blocking": 0, "error": 1, "warning": 2}[issue.severity],
        issue.scope_kind,
        issue.code,
        issue.issue_id,
    )
