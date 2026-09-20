"""Bounded V3-40 repair loop over V3-39 validation drafts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, date, datetime
from uuid import UUID

from backend.contracts.cost_estimation import CostCoverageStatus, CostSubjectKind
from backend.contracts.daily_scheduling import ScheduleActivityKind
from backend.contracts.enums import AnchorRole, CostCategory, DataAvailability
from backend.contracts.itinerary_draft import (
    CostValidationDraft,
    DraftAmountRange,
    DraftCategoryCostSummary,
    DraftCostEstimateLine,
    DraftDailyCostEstimate,
    DraftScheduledActivity,
    DraftScheduledDay,
    ScheduleValidationDraft,
)
from backend.contracts.itinerary_repair import (
    ItineraryRepairRequest,
    ItineraryRepairResult,
    RepairProposal,
    RepairRoundRecord,
    RepairRoundStatus,
    RepairRunStatus,
)
from backend.contracts.itinerary_validation import (
    ItineraryValidationRequest,
    ItineraryValidationResult,
    RepairAction,
    ValidationIssue,
    ValidationIssueCode,
    ValidationSeverity,
    ValidationStatus,
)
from backend.planning.daily_scheduling import DailySchedulingError, DailySchedulingService
from backend.planning.itinerary_validation import ItineraryValidationService

ITINERARY_REPAIR_ALGORITHM_VERSION = "1.0.0"
MAX_REPAIR_ROUNDS = 2

_PROTECTED_ROLES = {
    AnchorRole.MUST_ATTRACTION,
    AnchorRole.DESTINATION_RESTAURANT,
    AnchorRole.FIXED_EVENT,
}
_AUTOMATIC_SCHEDULE_ACTIONS = {
    RepairAction.REASSIGN_DAY,
    RepairAction.REORDER_DAY,
    RepairAction.CHANGE_ROUTE,
    RepairAction.ADJUST_BUFFER,
    RepairAction.ADJUST_TIME,
    RepairAction.RECALCULATE_SCHEDULE,
}


class ItineraryRepairService:
    """Apply only local, validated repairs and stop after two rounds."""

    def __init__(
        self,
        *,
        validation_service: ItineraryValidationService | None = None,
        scheduling_service: DailySchedulingService | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._validation = validation_service or ItineraryValidationService(clock=clock)
        self._scheduling = scheduling_service or DailySchedulingService(clock=clock)
        self._clock = clock

    def repair(
        self,
        request: ItineraryRepairRequest,
        *,
        is_generation_active: Callable[[UUID], bool] = lambda _generation_id: True,
    ) -> ItineraryRepairResult:
        request = ItineraryRepairRequest.model_validate(
            request.model_dump(mode="json"),
            context={"today": request.validation_request.scheduling_request.business_date},
        )
        current_request = request.validation_request
        current_validation = self._validation.validate(current_request)
        if _validation_signature(current_validation) != _validation_signature(
            request.validation_result
        ):
            raise ValueError("repair requires the current deterministic validation result")

        best_request = current_request
        best_validation = current_validation
        records: list[RepairRoundRecord] = []

        if not is_generation_active(request.generation_id):
            records.append(self._cancelled_record(1, current_request, current_validation))
            return self._result(
                request, RepairRunStatus.CANCELLED, records, best_request, best_validation
            )
        if current_validation.status is ValidationStatus.VALID:
            return self._result(
                request,
                RepairRunStatus.NOT_NEEDED,
                records,
                best_request,
                best_validation,
            )

        proposals = sorted(
            request.proposals,
            key=lambda item: (item.round_number, str(item.issue_id), str(item.proposal_id)),
        )
        for round_number in range(1, MAX_REPAIR_ROUNDS + 1):
            if not is_generation_active(request.generation_id):
                records.append(
                    self._cancelled_record(round_number, current_request, current_validation)
                )
                return self._result(
                    request,
                    RepairRunStatus.CANCELLED,
                    records,
                    best_request,
                    best_validation,
                )

            repairable = tuple(item for item in current_validation.issues if item.repairable)
            if not repairable:
                break
            external = next(
                (
                    item
                    for item in proposals
                    if item.round_number == round_number
                    and any(issue.issue_id == item.issue_id for issue in repairable)
                ),
                None,
            )
            before_fingerprint = _draft_fingerprint(current_request)
            try:
                if external is not None:
                    candidate_request, issue_ids, actions, affected_dates, message = (
                        self._apply_proposal(current_request, current_validation, external)
                    )
                else:
                    automatic = self._automatic_repair(current_request, repairable)
                    if automatic is None:
                        records.append(
                            self._record(
                                round_number=round_number,
                                status=RepairRoundStatus.NO_CANDIDATE,
                                before=current_request,
                                after=current_request,
                                before_validation=current_validation,
                                after_validation=current_validation,
                                issue_ids=tuple(item.issue_id for item in repairable),
                                actions=tuple(
                                    dict.fromkeys(item.repair_action for item in repairable)
                                ),
                                affected_dates=tuple(
                                    dict.fromkeys(
                                        item.service_date
                                        for item in repairable
                                        if item.service_date is not None
                                    )
                                ),
                                message="当前没有满足来源、范围和强意愿保护条件的本地修复候选。",
                            )
                        )
                        break
                    candidate_request, issue_ids, actions, affected_dates, message = automatic
            except (DailySchedulingError, ValueError) as exc:
                records.append(
                    self._record(
                        round_number=round_number,
                        status=RepairRoundStatus.REJECTED,
                        before=current_request,
                        after=current_request,
                        before_validation=current_validation,
                        after_validation=current_validation,
                        issue_ids=(external.issue_id,)
                        if external is not None
                        else tuple(item.issue_id for item in repairable),
                        actions=(external.action,)
                        if external is not None
                        else tuple(dict.fromkeys(item.repair_action for item in repairable)),
                        affected_dates=external.affected_dates if external is not None else (),
                        message=f"修复候选未通过边界检查：{exc}",
                    )
                )
                continue

            if not is_generation_active(request.generation_id):
                records.append(
                    self._cancelled_record(round_number, current_request, current_validation)
                )
                return self._result(
                    request,
                    RepairRunStatus.CANCELLED,
                    records,
                    best_request,
                    best_validation,
                )

            candidate_validation = self._validation.validate(candidate_request)
            improved = _validation_score(candidate_validation) < _validation_score(
                current_validation
            )
            changed = _draft_fingerprint(candidate_request) != before_fingerprint
            if not improved or not changed:
                records.append(
                    self._record(
                        round_number=round_number,
                        status=RepairRoundStatus.REJECTED,
                        before=current_request,
                        after=current_request,
                        before_validation=current_validation,
                        after_validation=current_validation,
                        issue_ids=issue_ids,
                        actions=actions,
                        affected_dates=affected_dates,
                        message="修复没有减少校验风险，已保留上一版稳定候选。",
                    )
                )
                continue

            record = self._record(
                round_number=round_number,
                status=RepairRoundStatus.APPLIED,
                before=current_request,
                after=candidate_request,
                before_validation=current_validation,
                after_validation=candidate_validation,
                issue_ids=issue_ids,
                actions=actions,
                affected_dates=affected_dates,
                message=message,
            )
            records.append(record)
            current_request = candidate_request
            current_validation = candidate_validation
            if _validation_score(current_validation) < _validation_score(best_validation):
                best_request = current_request
                best_validation = current_validation
            if current_validation.status is ValidationStatus.VALID:
                break

        if not is_generation_active(request.generation_id):
            next_round = min(len(records) + 1, MAX_REPAIR_ROUNDS)
            cancellation = self._cancelled_record(next_round, current_request, current_validation)
            if len(records) >= MAX_REPAIR_ROUNDS:
                records[-1] = cancellation
            else:
                records.append(cancellation)
            return self._result(
                request,
                RepairRunStatus.CANCELLED,
                records,
                best_request,
                best_validation,
            )
        status = (
            RepairRunStatus.REPAIRED
            if best_validation.status is ValidationStatus.VALID
            else RepairRunStatus.PARTIAL
            if best_validation.status is ValidationStatus.REVIEW
            else RepairRunStatus.FAILED
        )
        return self._result(request, status, records, best_request, best_validation)

    def _automatic_repair(
        self,
        current: ItineraryValidationRequest,
        issues: tuple[ValidationIssue, ...],
    ) -> (
        tuple[
            ItineraryValidationRequest,
            tuple[UUID, ...],
            tuple[RepairAction, ...],
            tuple[date, ...],
            str,
        ]
        | None
    ):
        schedule_issues = tuple(
            item
            for item in issues
            if item.repair_action in _AUTOMATIC_SCHEDULE_ACTIONS and item.service_date is not None
        )
        cost_issues = tuple(
            item for item in issues if item.repair_action is RepairAction.RECALCULATE_COST
        )
        if not schedule_issues and not cost_issues:
            return None

        schedule = current.schedule_draft
        affected_dates: tuple[date, ...] = tuple(
            dict.fromkeys(item.service_date for item in schedule_issues if item.service_date)
        )
        if affected_dates:
            canonical = ScheduleValidationDraft.from_result(
                self._scheduling.build(current.scheduling_request)
            )
            replacements = {
                item.service_date: item
                for item in canonical.days
                if item.service_date in affected_dates
            }
            if set(replacements) != set(affected_dates):
                raise ValueError("canonical schedule did not cover every affected date")
            schedule = _replace_days(schedule, tuple(replacements.values()), canonical)
            _protect_user_commitments(current.schedule_draft, schedule)

        cost = _recalculate_cost(current.cost_draft) if cost_issues else current.cost_draft
        candidate = current.model_copy(update={"schedule_draft": schedule, "cost_draft": cost})
        _cost_references_follow_schedule(candidate)
        all_issues = (*schedule_issues, *cost_issues)
        return (
            candidate,
            tuple(item.issue_id for item in all_issues),
            tuple(dict.fromkeys(item.repair_action for item in all_issues)),
            affected_dates,
            "已局部重算受影响日期、路线、缓冲或费用，并保留其他日期与用户强意愿。",
        )

    def _apply_proposal(
        self,
        current: ItineraryValidationRequest,
        validation: ItineraryValidationResult,
        proposal: RepairProposal,
    ) -> tuple[
        ItineraryValidationRequest,
        tuple[UUID, ...],
        tuple[RepairAction, ...],
        tuple[date, ...],
        str,
    ]:
        issue = next(item for item in validation.issues if item.issue_id == proposal.issue_id)
        if not issue.repairable or not _action_matches_issue(proposal.action, issue):
            raise ValueError("proposal action is not compatible with the current issue")
        if (
            issue.service_date is not None
            and proposal.affected_dates
            and (issue.service_date not in proposal.affected_dates)
        ):
            raise ValueError("proposal does not include the issue date")

        schedule = current.schedule_draft
        if proposal.replacement_days:
            known_dates = {item.service_date for item in schedule.days}
            if not set(proposal.affected_dates) <= known_dates:
                raise ValueError("proposal cannot modify a date outside the current trip")
            schedule = _replace_days(schedule, proposal.replacement_days, schedule)
            _validate_schedule_change(current, proposal, schedule, issue)
            _protect_user_commitments(current.schedule_draft, schedule)
        cost = proposal.replacement_cost_draft or current.cost_draft
        candidate = current.model_copy(update={"schedule_draft": schedule, "cost_draft": cost})
        _cost_references_follow_schedule(candidate)
        return (
            candidate,
            (issue.issue_id,),
            (proposal.action,),
            proposal.affected_dates,
            proposal.reason,
        )

    def _record(
        self,
        *,
        round_number: int,
        status: RepairRoundStatus,
        before: ItineraryValidationRequest,
        after: ItineraryValidationRequest,
        before_validation: ItineraryValidationResult,
        after_validation: ItineraryValidationResult,
        issue_ids: tuple[UUID, ...],
        actions: tuple[RepairAction, ...],
        affected_dates: tuple[date, ...],
        message: str,
    ) -> RepairRoundRecord:
        changed_activities, changed_legs = _changed_schedule_ids(
            before.schedule_draft, after.schedule_draft
        )
        return RepairRoundRecord(
            round_number=round_number,
            status=status,
            issue_ids=tuple(dict.fromkeys(issue_ids)),
            actions=tuple(dict.fromkeys(actions)),
            affected_dates=tuple(dict.fromkeys(affected_dates)),
            changed_activity_ids=changed_activities,
            changed_transport_leg_ids=changed_legs,
            changed_cost_categories=_changed_cost_categories(before.cost_draft, after.cost_draft),
            before_status=before_validation.status,
            after_status=after_validation.status,
            before_hard_conflict_count=_hard_count(before_validation),
            after_hard_conflict_count=_hard_count(after_validation),
            before_fingerprint=_draft_fingerprint(before),
            after_fingerprint=_draft_fingerprint(after),
            message=message,
        )

    def _cancelled_record(
        self,
        round_number: int,
        current: ItineraryValidationRequest,
        validation: ItineraryValidationResult,
    ) -> RepairRoundRecord:
        repairable = tuple(item for item in validation.issues if item.repairable)
        return self._record(
            round_number=round_number,
            status=RepairRoundStatus.CANCELLED,
            before=current,
            after=current,
            before_validation=validation,
            after_validation=validation,
            issue_ids=tuple(item.issue_id for item in repairable),
            actions=tuple(dict.fromkeys(item.repair_action for item in repairable)),
            affected_dates=tuple(
                dict.fromkeys(item.service_date for item in repairable if item.service_date)
            ),
            message="当前生成已被取消或被更新的生成替代，未提交本轮修复。",
        )

    def _result(
        self,
        request: ItineraryRepairRequest,
        status: RepairRunStatus,
        records: Sequence[RepairRoundRecord],
        best_request: ItineraryValidationRequest,
        best_validation: ItineraryValidationResult,
    ) -> ItineraryRepairResult:
        generated_at = self._clock()
        if generated_at.tzinfo is None or generated_at.utcoffset() is None:
            raise ValueError("itinerary repair clock must return an aware datetime")
        return ItineraryRepairResult(
            algorithm_version=ITINERARY_REPAIR_ALGORITHM_VERSION,
            request_id=request.request_id,
            trip_id=request.trip_id,
            input_state_version=request.input_state_version,
            generation_id=request.generation_id,
            status=status,
            rounds=tuple(records),
            best_schedule_draft=best_request.schedule_draft,
            best_cost_draft=best_request.cost_draft,
            final_validation=best_validation,
            remaining_issue_ids=tuple(item.issue_id for item in best_validation.issues),
            strict_ready=best_validation.status is not ValidationStatus.BLOCKED,
            generated_at=generated_at.astimezone(UTC),
        )


def _replace_days(
    schedule: ScheduleValidationDraft,
    replacements: Sequence[DraftScheduledDay],
    metadata_source: ScheduleValidationDraft,
) -> ScheduleValidationDraft:
    by_date = {item.service_date: item for item in replacements}
    days = tuple(by_date.get(item.service_date, item) for item in schedule.days)
    degradation = list(schedule.degradation_reasons)
    has_partial_fact = any(
        item.availability is DataAvailability.PARTIAL for day in days for item in day.activities
    ) or any(
        item.availability is DataAvailability.PARTIAL for day in days for item in day.transport_legs
    )
    if has_partial_fact and not degradation:
        degradation.append("one or more repaired schedule facts remain partial")
    status = (
        DataAvailability.PARTIAL
        if degradation or schedule.unscheduled_strong_desires
        else (DataAvailability.AVAILABLE)
    )
    return schedule.model_copy(
        update={
            "status": status,
            "days": days,
            "degradation_reasons": tuple(dict.fromkeys(degradation)),
            "provider_fact_ids": tuple(
                dict.fromkeys((*schedule.provider_fact_ids, *metadata_source.provider_fact_ids))
            ),
            "generated_at": metadata_source.generated_at,
        }
    )


def _validate_schedule_change(
    current: ItineraryValidationRequest,
    proposal: RepairProposal,
    replacement: ScheduleValidationDraft,
    issue: ValidationIssue,
) -> None:
    before_days = {item.service_date: item for item in current.schedule_draft.days}
    after_days = {item.service_date: item for item in replacement.days}
    for service_date, before in before_days.items():
        if service_date not in proposal.affected_dates and after_days[service_date] != before:
            raise ValueError("repair proposal changed an unrelated date")
    before_activities = _activities_by_id(current.schedule_draft)
    after_activities = _activities_by_id(replacement)
    removed_ids = set(before_activities) - set(after_activities)
    added_ids = set(after_activities) - set(before_activities)
    if proposal.action is RepairAction.REPLACE_CANDIDATE:
        if len(removed_ids) != 1 or len(added_ids) != 1:
            raise ValueError("candidate repair must replace exactly one activity")
        removed = before_activities[next(iter(removed_ids))]
        added = after_activities[next(iter(added_ids))]
        if removed.role in _PROTECTED_ROLES or removed.kind is ScheduleActivityKind.FIXED_EVENT:
            raise ValueError("must-go and fixed activities cannot be replaced")
        if removed.kind is not added.kind or removed.role is not added.role:
            raise ValueError("candidate repair must use the same activity class and role")
        if issue.activity_id is not None and issue.activity_id != removed.activity_id:
            raise ValueError("candidate repair replaced an activity unrelated to the issue")
        known_place_ids = {item.place_id for item in current.scheduling_request.places}
        if added.place_id not in known_place_ids:
            raise ValueError("replacement candidate requires a known same-city place fact")
    elif proposal.action is not RepairAction.REASSIGN_DAY and (
        set(before_activities) != set(after_activities)
    ):
        raise ValueError("local time or route repair cannot add or remove activities")


def _protect_user_commitments(
    before: ScheduleValidationDraft,
    after: ScheduleValidationDraft,
) -> None:
    before_activities = _activities_by_id(before)
    after_activities = _activities_by_id(after)
    for activity_id, activity in before_activities.items():
        if activity.role not in _PROTECTED_ROLES:
            continue
        repaired = after_activities.get(activity_id)
        if repaired is None:
            raise ValueError("repair cannot silently remove a must-go or fixed activity")
        if activity.kind is ScheduleActivityKind.FIXED_EVENT and repaired != activity:
            raise ValueError("repair cannot move or rewrite a fixed activity")


def _cost_references_follow_schedule(request: ItineraryValidationRequest) -> None:
    activity_ids = {
        item.activity_id for day in request.schedule_draft.days for item in day.activities
    }
    transport_ids = {
        item.leg_id for day in request.schedule_draft.days for item in day.transport_legs
    }
    for line in (item for day in request.cost_draft.days for item in day.lines):
        if line.subject_kind is CostSubjectKind.ACTIVITY and line.subject_id not in activity_ids:
            raise ValueError("cost draft references a removed activity")
        if (
            line.subject_kind is CostSubjectKind.TRANSPORT_LEG
            and line.subject_id not in transport_ids
        ):
            raise ValueError("cost draft references a removed transport leg")


def _recalculate_cost(cost: CostValidationDraft) -> CostValidationDraft:
    days = tuple(
        DraftDailyCostEstimate(
            service_date=day.service_date,
            lines=day.lines,
            categories=tuple(_summarize_cost(category, day.lines) for category in CostCategory),
            known_subtotal_per_person=_sum_ranges(
                line.amount_per_person for line in day.lines if line.amount_per_person is not None
            ),
        )
        for day in cost.days
    )
    all_lines = tuple(line for day in days for line in day.lines)
    return cost.model_copy(
        update={
            "days": days,
            "categories": tuple(_summarize_cost(category, all_lines) for category in CostCategory),
            "known_total_per_person": _sum_ranges(
                day.known_subtotal_per_person
                for day in days
                if day.known_subtotal_per_person is not None
            ),
        }
    )


def _summarize_cost(
    category: CostCategory,
    lines: Iterable[DraftCostEstimateLine],
) -> DraftCategoryCostSummary:
    applicable = tuple(item for item in lines if item.category is category)
    if not applicable:
        return DraftCategoryCostSummary(
            category=category,
            status=CostCoverageStatus.NOT_APPLICABLE,
            priced_item_count=0,
            missing_item_count=0,
            note="本日没有这一类已选项目。",
        )
    priced = tuple(item for item in applicable if item.amount_per_person is not None)
    missing = tuple(
        item
        for item in applicable
        if item.amount_per_person is None or item.availability is DataAvailability.PARTIAL
    )
    status = (
        CostCoverageStatus.MISSING
        if not priced
        else CostCoverageStatus.PARTIAL
        if missing
        else CostCoverageStatus.AVAILABLE
    )
    return DraftCategoryCostSummary(
        category=category,
        status=status,
        amount_per_person=_sum_ranges(
            item.amount_per_person for item in priced if item.amount_per_person is not None
        ),
        priced_item_count=len(priced),
        missing_item_count=len(missing),
        source_reference_ids=tuple(
            dict.fromkeys(source for item in applicable for source in item.source_reference_ids)
        ),
        note=("部分项目价格暂缺。" if missing else None),
    )


def _sum_ranges(values: Iterable[DraftAmountRange]) -> DraftAmountRange | None:
    materialized = tuple(values)
    if not materialized:
        return None
    if any(
        item.currency != "CNY" or item.minimum_fen < 0 or item.maximum_fen < item.minimum_fen
        for item in materialized
    ):
        return DraftAmountRange(currency="INVALID", minimum_fen=-1, maximum_fen=-1)
    return DraftAmountRange(
        currency="CNY",
        minimum_fen=sum(item.minimum_fen for item in materialized),
        maximum_fen=sum(item.maximum_fen for item in materialized),
    )


def _action_matches_issue(action: RepairAction, issue: ValidationIssue) -> bool:
    if action is issue.repair_action:
        return True
    return (
        issue.code is ValidationIssueCode.OPENING_HOURS_CONFLICT
        and action is RepairAction.REPLACE_CANDIDATE
    )


def _activities_by_id(
    schedule: ScheduleValidationDraft,
) -> dict[UUID, DraftScheduledActivity]:
    return {item.activity_id: item for day in schedule.days for item in day.activities}


def _changed_schedule_ids(
    before: ScheduleValidationDraft,
    after: ScheduleValidationDraft,
) -> tuple[tuple[UUID, ...], tuple[UUID, ...]]:
    before_activities = _activities_by_id(before)
    after_activities = _activities_by_id(after)
    changed_activities = tuple(
        sorted(
            (
                key
                for key in set(before_activities) | set(after_activities)
                if before_activities.get(key) != after_activities.get(key)
            ),
            key=str,
        )
    )
    before_legs = {item.leg_id: item for day in before.days for item in day.transport_legs}
    after_legs = {item.leg_id: item for day in after.days for item in day.transport_legs}
    changed_legs = tuple(
        sorted(
            (
                key
                for key in set(before_legs) | set(after_legs)
                if before_legs.get(key) != after_legs.get(key)
            ),
            key=str,
        )
    )
    return changed_activities, changed_legs


def _changed_cost_categories(
    before: CostValidationDraft,
    after: CostValidationDraft,
) -> tuple[CostCategory, ...]:
    if before == after:
        return ()
    changed: list[CostCategory] = []
    for category in CostCategory:
        before_lines = tuple(
            item for day in before.days for item in day.lines if item.category is category
        )
        after_lines = tuple(
            item for day in after.days for item in day.lines if item.category is category
        )
        before_summary = next(
            (item for item in before.categories if item.category is category), None
        )
        after_summary = next((item for item in after.categories if item.category is category), None)
        before_daily = tuple(
            (day.service_date, day.known_subtotal_per_person)
            for day in before.days
            if any(item.category is category for item in day.lines)
        )
        after_daily = tuple(
            (day.service_date, day.known_subtotal_per_person)
            for day in after.days
            if any(item.category is category for item in day.lines)
        )
        if (
            before_lines != after_lines
            or before_summary != after_summary
            or before_daily != after_daily
            or (
                before.known_total_per_person != after.known_total_per_person
                and (before_lines or after_lines)
            )
        ):
            changed.append(category)
    return tuple(changed)


def _validation_signature(result: ItineraryValidationResult) -> tuple[object, ...]:
    return (
        result.request_id,
        result.trip_id,
        result.input_state_version,
        result.status,
        tuple(
            (
                item.issue_id,
                item.code,
                item.severity,
                item.target_kind,
                item.service_date,
                item.activity_id,
                item.unscheduled_node_id,
                item.transport_leg_id,
                item.hotel_place_id,
                item.cost_category,
                item.repairable,
                item.repair_action,
            )
            for item in result.issues
        ),
    )


def _validation_score(result: ItineraryValidationResult) -> tuple[int, int, int, int]:
    hard = _hard_count(result)
    soft = sum(item.severity is ValidationSeverity.SOFT_RISK for item in result.issues)
    unknown = sum(item.severity is ValidationSeverity.UNKNOWN_FACT for item in result.issues)
    return hard, soft + unknown, unknown, len(result.issues)


def _hard_count(result: ItineraryValidationResult) -> int:
    return sum(item.severity is ValidationSeverity.HARD_CONFLICT for item in result.issues)


def _draft_fingerprint(request: ItineraryValidationRequest) -> str:
    payload = {
        "schedule": request.schedule_draft.model_dump(mode="json"),
        "cost": request.cost_draft.model_dump(mode="json"),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


V3_ITINERARY_REPAIR_SERVICE_EXPORTS = (
    ItineraryRepairService,
    ITINERARY_REPAIR_ALGORITHM_VERSION,
    MAX_REPAIR_ROUNDS,
)
