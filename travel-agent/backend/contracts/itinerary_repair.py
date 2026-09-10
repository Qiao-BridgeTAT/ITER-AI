"""V3-40 contracts for bounded, auditable itinerary repair."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from enum import StrEnum
from typing import Any, Literal
from uuid import UUID

from pydantic import AwareDatetime, ConfigDict, Field, model_validator

from backend.contracts.base import ContractModel
from backend.contracts.common import NonEmptyText, ShortText
from backend.contracts.enums import CostCategory
from backend.contracts.itinerary_draft import (
    CostValidationDraft,
    DraftScheduledDay,
    ScheduleValidationDraft,
)
from backend.contracts.itinerary_validation import (
    ItineraryValidationRequest,
    ItineraryValidationResult,
    RepairAction,
    ValidationStatus,
)

ITINERARY_REPAIR_REQUEST_SCHEMA_RULE: dict[str, Any] = {"x-travel-itinerary-repair-request": True}
ITINERARY_REPAIR_RESULT_SCHEMA_RULE: dict[str, Any] = {"x-travel-itinerary-repair-result": True}


class ImmutableRepairModel(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RepairRunStatus(StrEnum):
    NOT_NEEDED = "not_needed"
    REPAIRED = "repaired"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RepairRoundStatus(StrEnum):
    APPLIED = "applied"
    REJECTED = "rejected"
    NO_CANDIDATE = "no_candidate"
    CANCELLED = "cancelled"


class RepairProposal(ImmutableRepairModel):
    """A deterministic, sourced alternative supplied to the bounded repair loop."""

    proposal_id: UUID
    round_number: int = Field(ge=1, le=2, strict=True)
    issue_id: UUID
    action: RepairAction
    affected_dates: tuple[date, ...] = Field(default=(), max_length=2)
    replacement_days: tuple[DraftScheduledDay, ...] = Field(default=(), max_length=2)
    replacement_cost_draft: CostValidationDraft | None = None
    reason: ShortText
    source_reference_ids: tuple[NonEmptyText, ...] = Field(default=())

    @model_validator(mode="after")
    def proposal_is_local_and_actionable(self) -> RepairProposal:
        _unique(self.affected_dates, "repair proposal dates")
        _unique(
            [item.service_date for item in self.replacement_days],
            "repair replacement day dates",
        )
        _unique(self.source_reference_ids, "repair proposal sources")
        unsupported = {
            RepairAction.NONE,
            RepairAction.REQUERY_FACT,
            RepairAction.FIX_REFERENCE,
        }
        if self.action in unsupported:
            raise ValueError("repair proposal requires a deterministic local action")
        replacement_dates = tuple(item.service_date for item in self.replacement_days)
        if self.action is RepairAction.RECALCULATE_COST:
            if self.affected_dates or self.replacement_days or self.replacement_cost_draft is None:
                raise ValueError("cost repair requires only a replacement cost draft")
        elif not self.affected_dates or replacement_dates != self.affected_dates:
            raise ValueError("schedule repair dates must match replacement days in order")
        if self.action is RepairAction.REPLACE_CANDIDATE:
            if self.replacement_cost_draft is None:
                raise ValueError("candidate replacement must also replace its cost draft")
            if not self.source_reference_ids:
                raise ValueError("candidate replacement requires source references")
        return self


class ItineraryRepairRequest(ImmutableRepairModel):
    model_config = ConfigDict(json_schema_extra=ITINERARY_REPAIR_REQUEST_SCHEMA_RULE)

    request_id: UUID
    trip_id: UUID
    input_state_version: int = Field(ge=0, strict=True)
    generation_id: UUID
    validation_request: ItineraryValidationRequest
    validation_result: ItineraryValidationResult
    proposals: tuple[RepairProposal, ...] = Field(default=(), max_length=20)
    max_rounds: Literal[2] = 2

    @model_validator(mode="after")
    def repair_inputs_are_current(self) -> ItineraryRepairRequest:
        validation_request = self.validation_request
        validation_result = self.validation_result
        if (
            validation_request.trip_id != self.trip_id
            or validation_result.trip_id != self.trip_id
            or validation_request.input_state_version != self.input_state_version
            or validation_result.input_state_version != self.input_state_version
            or validation_result.request_id != validation_request.request_id
        ):
            raise ValueError("repair input must reference one current validation run")
        _unique([item.proposal_id for item in self.proposals], "repair proposal IDs")
        _unique(
            [(item.round_number, item.issue_id) for item in self.proposals],
            "repair proposal round and issue keys",
        )
        known_issue_ids = {item.issue_id for item in validation_result.issues}
        if any(item.issue_id not in known_issue_ids for item in self.proposals):
            raise ValueError("repair proposal must reference a reported validation issue")
        return self


class RepairRoundRecord(ImmutableRepairModel):
    round_number: int = Field(ge=1, le=2, strict=True)
    status: RepairRoundStatus
    issue_ids: tuple[UUID, ...]
    actions: tuple[RepairAction, ...]
    affected_dates: tuple[date, ...] = ()
    changed_activity_ids: tuple[UUID, ...] = ()
    changed_transport_leg_ids: tuple[UUID, ...] = ()
    changed_cost_categories: tuple[CostCategory, ...] = ()
    before_status: ValidationStatus
    after_status: ValidationStatus
    before_hard_conflict_count: int = Field(ge=0, strict=True)
    after_hard_conflict_count: int = Field(ge=0, strict=True)
    before_fingerprint: NonEmptyText
    after_fingerprint: NonEmptyText
    message: ShortText

    @model_validator(mode="after")
    def round_record_is_consistent(self) -> RepairRoundRecord:
        _unique(self.issue_ids, "repair round issue IDs")
        _unique(self.actions, "repair round actions")
        _unique(self.affected_dates, "repair round dates")
        _unique(self.changed_activity_ids, "repair round activity IDs")
        _unique(self.changed_transport_leg_ids, "repair round transport IDs")
        _unique(self.changed_cost_categories, "repair round cost categories")
        changed = bool(
            self.changed_activity_ids
            or self.changed_transport_leg_ids
            or self.changed_cost_categories
            or self.before_fingerprint != self.after_fingerprint
        )
        if self.status is RepairRoundStatus.APPLIED and (not self.issue_ids or not changed):
            raise ValueError("applied repair round requires issues and a real change")
        if (
            self.status is RepairRoundStatus.CANCELLED
            and self.before_fingerprint != self.after_fingerprint
        ):
            raise ValueError("cancelled repair round cannot commit a change")
        return self


class ItineraryRepairResult(ImmutableRepairModel):
    model_config = ConfigDict(json_schema_extra=ITINERARY_REPAIR_RESULT_SCHEMA_RULE)

    algorithm_version: str = Field(pattern=r"^[1-9]\d*\.\d+\.\d+$")
    request_id: UUID
    trip_id: UUID
    input_state_version: int = Field(ge=0, strict=True)
    generation_id: UUID
    status: RepairRunStatus
    rounds: tuple[RepairRoundRecord, ...] = Field(default=(), max_length=2)
    best_schedule_draft: ScheduleValidationDraft
    best_cost_draft: CostValidationDraft
    final_validation: ItineraryValidationResult
    remaining_issue_ids: tuple[UUID, ...]
    strict_ready: bool
    generated_at: AwareDatetime

    @model_validator(mode="after")
    def result_status_matches_validation(self) -> ItineraryRepairResult:
        _unique([item.round_number for item in self.rounds], "repair round numbers")
        _unique(self.remaining_issue_ids, "remaining repair issue IDs")
        if tuple(item.round_number for item in self.rounds) != tuple(
            range(1, len(self.rounds) + 1)
        ):
            raise ValueError("repair rounds must be contiguous and ordered")
        if set(self.remaining_issue_ids) != {
            item.issue_id for item in self.final_validation.issues
        }:
            raise ValueError("remaining issue IDs must match final validation")
        expected_ready = self.final_validation.status is not ValidationStatus.BLOCKED
        if self.strict_ready != expected_ready:
            raise ValueError("strict readiness must follow final validation status")
        if self.status is RepairRunStatus.NOT_NEEDED:
            if self.rounds or self.final_validation.status is not ValidationStatus.VALID:
                raise ValueError("not-needed repair requires an initially valid itinerary")
        elif self.status is RepairRunStatus.REPAIRED:
            if not self.rounds or not expected_ready:
                raise ValueError("repaired result requires applied work and no hard conflict")
        elif self.status is RepairRunStatus.PARTIAL:
            if self.final_validation.status is not ValidationStatus.REVIEW:
                raise ValueError("partial repair must end in review status")
        elif self.status is RepairRunStatus.FAILED:
            if self.final_validation.status is not ValidationStatus.BLOCKED:
                raise ValueError("failed repair must retain a hard conflict")
        elif not self.rounds or self.rounds[-1].status is not RepairRoundStatus.CANCELLED:
            raise ValueError("cancelled repair must record its cancellation boundary")
        return self


def _unique(values: Sequence[object], label: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must be unique")


V3_ITINERARY_REPAIR_CONTRACTS: tuple[type[ContractModel], ...] = (
    ItineraryRepairRequest,
    ItineraryRepairResult,
)
