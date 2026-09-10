"""Optional checkpoint state for measured daily repairs, never user intent."""

from datetime import date, time
from typing import Literal

from pydantic import AwareDatetime, Field, model_validator

from backend.contracts.v4.base import Identifier, V4ContractModel, require_unique


class ScheduleRepairAttempt(V4ContractModel):
    fingerprint: Identifier
    action: str = Field(max_length=80)
    outcome: Literal["pending", "improved", "rejected", "unchanged"] = "pending"
    failure_code: str | None = Field(default=None, max_length=300)
    rejected_candidate_ids: tuple[Identifier, ...] = ()


class ScheduleRepairIssue(V4ContractModel):
    issue_id: Identifier
    service_date: date
    kind: Literal["hard_time", "meal", "coverage", "gap", "dining_route", "evening"]
    priority: int = Field(ge=0, le=4, strict=True)
    code: str
    field_path: str
    period: str | None = None
    start: time | None = None
    end: time | None = None
    previous_node_id: Identifier | None = None
    next_node_id: Identifier | None = None
    missing_minutes: int = Field(default=0, ge=0, strict=True)
    status: Literal["pending", "resolved", "exhausted", "unavailable"] = "pending"
    attempts: tuple[ScheduleRepairAttempt, ...] = Field(default=(), max_length=2)
    last_batch: int = Field(default=0, ge=0, strict=True)
    stop_reason: str | None = None


class PlannerScheduleRepairState(V4ContractModel):
    policy_version: Literal["daily-repair-1"] = "daily-repair-1"
    turn_id: Identifier | None = None
    started_at: AwareDatetime | None = None
    deadline_at: AwareDatetime | None = None
    initial_deadline_at: AwareDatetime | None = None
    call_cutoff_at: AwareDatetime | None = None
    budget_seconds: int = Field(default=300, ge=1, le=300, strict=True)
    remaining_seconds: float | None = Field(default=None, ge=0)
    batch_number: int = Field(default=0, ge=0, strict=True)
    issues: tuple[ScheduleRepairIssue, ...] = ()
    status: Literal["planning", "complete", "partial"] = "planning"
    stop_reason: str | None = None
    measured_schedule_id: Identifier | None = None

    @model_validator(mode="after")
    def coherent_budget_and_issues(self) -> "PlannerScheduleRepairState":
        require_unique([issue.issue_id for issue in self.issues], "schedule repair issues")
        if self.started_at and self.deadline_at:
            duration = (self.deadline_at - self.started_at).total_seconds()
            if not 0 < duration <= self.budget_seconds:
                raise ValueError("repair deadline must stay inside the original turn budget")
        if self.status == "complete" and any(
            issue.status != "resolved" and issue.kind != "evening" for issue in self.issues
        ):
            raise ValueError("unresolved scheduling issues cannot be marked complete")
        return self
