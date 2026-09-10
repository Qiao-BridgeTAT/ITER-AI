"""Prefer verified opening dates without inventing facts or another model call."""

from __future__ import annotations

import re
from datetime import date
from itertools import permutations

from backend.agent.planner.decision_contracts import ModelPlanStop
from backend.agent.planner.proposals import PlannerReferenceCatalog
from backend.contracts.v4.enums import CandidateEntityKind, CommitmentLevel
from backend.contracts.v4.planner_workspace import PlannerWorkspaceState
from backend.contracts.v4.task_book import TaskBookV4
from backend.providers.contracts import HoursDayStatus


def align_days_with_opening_evidence(
    stops: dict[int, list[ModelPlanStop]],
    dates: tuple[date, ...],
    catalog: PlannerReferenceCatalog,
    workspace: PlannerWorkspaceState,
    book: TaskBookV4,
) -> dict[int, list[ModelPlanStop]]:
    """Reassign intact flexible day routes (at most 5! cheap comparisons).

    Dates of fixed commitments or explicit date-specific requirements are never
    moved. Unknown hours remain unknown, but must not be preferred over a verified
    open day just because they allow a conveniently unconstrained start time.
    Full travel, meal, last-entry and duration validation still follows compilation.
    """
    if (
        len(dates) < 2
        or catalog.fixed
        or any(
            re.search(
                r"\d{1,2}月\d{1,2}|\d{4}-\d{2}-\d{2}|第.{1,3}天|周[一二三四五六日天]|星期",
                item.value,
            )
            for item in book.hard_constraints
        )
    ):
        return stops
    if any(stop.candidate_key not in catalog.candidates for day in stops.values() for stop in day):
        return stops  # The normal reference Guard reports the precise invalid key.
    hours = {item.canonical_entity_id: item for item in workspace.hours_evidence}
    indices = tuple(stops)

    def score(order: tuple[int, ...]) -> tuple[int, int, int, int]:
        infeasible = strong_unknown = other_unknown = moved = 0
        for target, source in enumerate(order, 1):
            service_date = dates[target - 1]
            moved += source != target
            for stop in stops[source]:
                entry = catalog.candidates[stop.candidate_key]
                if entry.entity_kind is CandidateEntityKind.RESTAURANT:
                    continue
                if service_date in entry.infeasible_dates or (
                    entry.feasible_dates and service_date not in entry.feasible_dates
                ):
                    infeasible += 1
                evidence = hours.get(entry.candidate_ref.canonical_entity_id)
                if evidence is None:
                    continue
                available = {
                    day.service_date
                    for day in evidence.days
                    if day.status is HoursDayStatus.OPEN and day.intervals
                }
                day = next((day for day in evidence.days if day.service_date == service_date), None)
                if day is not None and day.status is HoursDayStatus.CLOSED:
                    infeasible += 1
                elif available.intersection(dates) and service_date not in available:
                    if entry.commitment_level is CommitmentLevel.STRONG:
                        strong_unknown += 1
                    else:
                        other_unknown += 1
        return infeasible, strong_unknown, other_unknown, moved

    best = min(permutations(indices), key=score)
    if score(best) >= score(indices):
        return stops
    return {target: stops[source] for target, source in enumerate(best, 1)}
