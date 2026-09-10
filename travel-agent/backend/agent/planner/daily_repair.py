"""Measured repair worklist shared by generation, editing and recovery.

Scores compare proposals. Only another measurement can resolve an issue.
All helpers are deterministic and return new checkpoint values.
"""

from datetime import UTC, date, datetime, time, timedelta
from typing import Any, Literal

from backend.agent.planner.timing_quality import (
    afternoon_activity_opportunities,
    dining_commute_issues,
    evening_activity_opportunities,
    minutes,
    missing_concrete_meals,
    schedule_coverage_issues,
    schedule_meal_issues,
    schedule_quality_gaps,
)
from backend.agent.planner.workspace import advance, server_id
from backend.contracts.v4.planner_schedule_repair import (
    PlannerScheduleRepairState,
    ScheduleRepairAttempt,
    ScheduleRepairIssue,
)
from backend.contracts.v4.planner_workspace import PlannerWorkspaceState
from backend.contracts.v4.task_book import TaskBookV4

TOTAL_SECONDS = 300
RECOVERY_SECONDS = 110
FINALIZE_SECONDS = 20


def bind_execution_budget(
    workspace: PlannerWorkspaceState, turn_id: str, started_at: datetime
) -> PlannerWorkspaceState:
    state = workspace.schedule_repair_state or PlannerScheduleRepairState()
    if state.turn_id == turn_id and state.deadline_at is not None:
        return workspace
    # A new explicit request gets a new budget; same-turn recovery never does.
    state = PlannerScheduleRepairState(
        turn_id=turn_id,
        started_at=started_at,
        deadline_at=started_at + timedelta(seconds=TOTAL_SECONDS),
        initial_deadline_at=started_at + timedelta(seconds=TOTAL_SECONDS - RECOVERY_SECONDS),
        call_cutoff_at=started_at + timedelta(seconds=TOTAL_SECONDS - FINALIZE_SECONDS),
    )
    return advance(workspace, schedule_repair_state=state)


def execution_remaining(workspace: PlannerWorkspaceState, *, calls: bool = False) -> float:
    state = workspace.schedule_repair_state
    end = (state.call_cutoff_at if calls else state.deadline_at) if state else None
    return max(0.0, (end - datetime.now(UTC)).total_seconds()) if end else float("inf")


def measure_repair_issues(
    workspace: PlannerWorkspaceState, book: TaskBookV4
) -> tuple[ScheduleRepairIssue, ...]:
    draft, schedule = workspace.working_itinerary, workspace.materialized_schedule
    if draft is None or schedule is None:
        return ()
    issues: dict[str, ScheduleRepairIssue] = {}

    def add(
        kind: Literal["hard_time", "meal", "coverage", "gap", "dining_route", "evening"],
        priority: int,
        value: dict[str, Any],
        anchor: str = "",
    ) -> None:
        service_date = date.fromisoformat(str(value["date"]))
        period = str(value.get("period", value.get("meal", ""))) or None
        start = time.fromisoformat(str(value["start"])) if value.get("start") else None
        end = time.fromisoformat(str(value["end"])) if value.get("end") else None
        actual = next(day for day in schedule.days if day.service_date == service_date)
        previous = next(
            (a for a in reversed(actual.activities) if start and a.end_time <= start), None
        )
        following = next((a for a in actual.activities if end and a.start_time >= end), None)
        code = str(value.get("code", kind))
        identity = server_id(
            workspace.generation_id,
            "daily-repair",
            service_date,
            kind,
            # Half-day identity survives a change from empty to underfilled.
            period or (f"{getattr(previous, 'node_id', '')}/{getattr(following, 'node_id', '')}"),
            anchor,
        )
        issues[identity] = ScheduleRepairIssue(
            issue_id=identity,
            service_date=service_date,
            kind=kind,
            priority=priority,
            code=code,
            field_path=str(value.get("path", f"days[{service_date}].stops")),
            period=period,
            start=start,
            end=end,
            previous_node_id=str(previous.node_id) if previous else None,
            next_node_id=str(following.node_id) if following else None,
            missing_minutes=max(0, int(value.get("minutes", 0)))
            if kind != "coverage"
            else max(0, int(value["minimum_visit_minutes"]) - int(value["visit_minutes"])),
        )

    if workspace.validation_observation:
        for issue in workspace.validation_observation.issues:
            if issue.severity == "warning" or issue.code not in {
                "opening_conflict",
                "time_overlap",
                "meal_constraint_violation",
            }:
                continue
            for affected_date in issue.affected_dates:
                add(
                    "hard_time",
                    0,
                    {"date": affected_date, "code": issue.code},
                    f"{issue.code}:{','.join(issue.draft_item_ids)}",
                )
    for value in schedule_meal_issues(workspace):
        add("hard_time", 0, value, str(value.get("draft_item_id", value.get("meal", ""))))
    for day in draft.days:
        extra = [
            i
            for i in day.ordered_items
            if i.item_kind == "dining" and i.meal_slot not in {"lunch", "dinner"}
        ]
        slots: list[str | None] = [
            i.meal_slot for i in day.ordered_items if i.item_kind == "dining"
        ]
        if extra or any(slots.count(slot) > 1 for slot in ("lunch", "dinner")):
            add("hard_time", 0, {"date": day.service_date, "code": "two_main_meals_only"}, "meals")
    for value in missing_concrete_meals(workspace):
        add("meal", 1, value)
    for value in schedule_coverage_issues(workspace, book):
        add("coverage", 2 if not value["visit_minutes"] else 3, value)
    for gap in schedule_quality_gaps(workspace, book):
        add("gap", 3, dict(gap))
    for gap in afternoon_activity_opportunities(workspace, book):
        add("gap", 3, dict(gap), "afternoon_capacity")
    for detour in dining_commute_issues(workspace):
        add("dining_route", 4, dict(detour), str(detour["draft_item_id"]))
    for gap in evening_activity_opportunities(workspace, book):
        add("evening", 4, dict(gap), "optional_evening")
    return tuple(issues.values())


def refresh_repair_state(
    workspace: PlannerWorkspaceState, book: TaskBookV4, *, stop_reason: str | None = None
) -> PlannerWorkspaceState:
    state = workspace.schedule_repair_state or PlannerScheduleRepairState()
    if workspace.working_itinerary is None or workspace.materialized_schedule is None:
        # No timeline is not a clean timeline. Keep prior unresolved issues
        # while compilation/recovery is rebuilding the materialized artifact.
        return advance(
            workspace,
            schedule_repair_state=state.model_copy(
                update={
                    "status": "partial" if stop_reason else "planning",
                    "stop_reason": stop_reason,
                    "measured_schedule_id": None,
                }
            ),
        )
    old = {i.issue_id: i for i in state.issues}
    measured = measure_repair_issues(workspace, book)
    active = []
    for issue in measured:
        previous = old.pop(issue.issue_id, None)
        if previous:
            issue = issue.model_copy(
                update={
                    "attempts": previous.attempts,
                    "last_batch": previous.last_batch,
                    "status": "pending" if previous.status == "resolved" else previous.status,
                    "stop_reason": previous.stop_reason,
                }
            )
        active.append(issue)
    resolved = [
        i.model_copy(update={"status": "resolved", "stop_reason": None}) for i in old.values()
    ]
    incomplete = any(i.kind != "evening" for i in active)
    if stop_reason == "publication":
        stop_reason = state.stop_reason or ("unresolved_at_publication" if incomplete else None)
    status = "partial" if stop_reason and incomplete else "planning" if incomplete else "complete"
    remaining = execution_remaining(workspace)
    return advance(
        workspace,
        schedule_repair_state=state.model_copy(
            update={
                "issues": tuple((*active, *resolved)),
                "status": status,
                "stop_reason": stop_reason,
                "measured_schedule_id": str(workspace.materialized_schedule.request_id)
                if workspace.materialized_schedule
                else None,
                "remaining_seconds": remaining if remaining != float("inf") else None,
            }
        ),
    )


def next_repair_batch(
    state: PlannerScheduleRepairState, allowed_dates: frozenset[date] | None = None
) -> tuple[ScheduleRepairIssue, ...]:
    eligible = [
        i
        for i in state.issues
        if i.status == "pending"
        and len(i.attempts) < 2
        and (allowed_dates is None or i.service_date in allowed_dates)
    ]
    if not eligible:
        return ()
    last_by_date = {
        day: max(i.last_batch for i in state.issues if i.service_date == day)
        for day in {i.service_date for i in state.issues}
    }
    eligible.sort(
        key=lambda i: (
            i.priority,
            last_by_date[i.service_date],
            i.last_batch,
            i.service_date,
            -i.missing_minutes,
            i.issue_id,
        )
    )
    # One operation family per call; an untouched day wins over a second gap
    # on the same day. Hard day rewrites already cap themselves at two days.
    first = eligible[0]

    def family(issue: ScheduleRepairIssue) -> str:
        return "visit" if issue.kind in {"coverage", "gap", "evening"} else issue.kind

    selected: list[ScheduleRepairIssue] = []
    dates: set[date] = set()
    for issue in eligible:
        if family(issue) != family(first) or issue.priority != first.priority:
            continue
        if issue.service_date in dates:
            continue
        selected.append(issue)
        dates.add(issue.service_date)
        if len(selected) == (
            2 if first.kind == "hard_time" else 1 if first.kind == "dining_route" else 3
        ):
            break
    return tuple(selected)


def begin_repair_batch(
    workspace: PlannerWorkspaceState, batch: tuple[ScheduleRepairIssue, ...], action: str
) -> PlannerWorkspaceState:
    assert workspace.schedule_repair_state is not None
    state = workspace.schedule_repair_state
    number = state.batch_number + 1
    selected = {i.issue_id for i in batch}
    return advance(
        workspace,
        schedule_repair_state=state.model_copy(
            update={
                "batch_number": number,
                "issues": tuple(
                    i.model_copy(
                        update={
                            "last_batch": number,
                            "attempts": (
                                *i.attempts,
                                ScheduleRepairAttempt(
                                    fingerprint=server_id(i.issue_id, number, action), action=action
                                ),
                            ),
                        }
                    )
                    if i.issue_id in selected
                    else i
                    for i in state.issues
                ),
            }
        ),
    )


def finish_repair_batch(
    workspace: PlannerWorkspaceState,
    batch: tuple[ScheduleRepairIssue, ...],
    feedback: list[dict[str, Any]],
    *,
    unavailable: bool = False,
) -> PlannerWorkspaceState:
    assert workspace.schedule_repair_state is not None
    state = workspace.schedule_repair_state
    selected = {i.issue_id for i in batch}
    last = feedback[-1] if feedback else {}
    fingerprint = str(last.get("proposal_fingerprint", ""))
    code = (
        "requires_day_repair"
        if last.get("requires_day_repair")
        else str(last.get("failure_code", "no_feasible_candidate" if unavailable else "unchanged"))
    )
    updated = []
    for issue in state.issues:
        if issue.issue_id not in selected:
            updated.append(issue)
            continue
        attempts = issue.attempts
        if attempts and not unavailable:
            from backend.agent.planner.proposals import PlannerReferenceCatalog

            catalog = PlannerReferenceCatalog(workspace)
            rejected = tuple(
                dict.fromkeys(
                    catalog.candidates[choice["candidate_key"]].candidate_ref.canonical_entity_id
                    for item in feedback
                    for choice in item.get("rejected_gap_choices", [])
                    if choice.get("date") == str(issue.service_date)
                    and choice.get("candidate_key") in catalog.candidates
                )
            )
            attempt = attempts[-1].model_copy(
                update={
                    "fingerprint": fingerprint or attempts[-1].fingerprint,
                    "outcome": "improved" if last.get("status") == "improved" else "rejected",
                    "failure_code": None if last.get("status") == "improved" else code[:300],
                    "rejected_candidate_ids": rejected,
                }
            )
            attempts = (*attempts[:-1], attempt)
        updated.append(
            issue.model_copy(
                update={
                    "attempts": attempts,
                    "status": "resolved"
                    if issue.status == "resolved"
                    else "unavailable"
                    if unavailable
                    else "exhausted"
                    if len(attempts) >= 2
                    else "pending",
                    "stop_reason": "no_feasible_candidate"
                    if unavailable
                    else "attempts_exhausted"
                    if len(attempts) >= 2
                    else None,
                    "last_batch": max(issue.last_batch, state.batch_number),
                }
            )
        )
    return advance(
        workspace, schedule_repair_state=state.model_copy(update={"issues": tuple(updated)})
    )


def gap_matches_issue(gap: dict[str, Any], issue: ScheduleRepairIssue) -> bool:
    if gap["date"] != str(issue.service_date):
        return False
    start, end = minutes(time.fromisoformat(gap["start"])), minutes(time.fromisoformat(gap["end"]))
    if issue.period == "morning":
        return start < 12 * 60 + 30
    if issue.period == "afternoon":
        return end > 13 * 60 and start < 20 * 60
    return (
        issue.start is None
        or issue.end is None
        or (start <= minutes(issue.end) and end >= minutes(issue.start))
    )
