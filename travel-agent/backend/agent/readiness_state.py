"""Stable runtime records for V2 readiness and final-supplement boundaries."""

from __future__ import annotations

from enum import StrEnum
from uuid import UUID

from pydantic import ConfigDict, Field, model_validator

from backend.agent.semantic_operations import SemanticTarget
from backend.contracts.base import ContractModel
from backend.contracts.common import ShortText
from backend.contracts.enums import Confidence


class ReadinessGapCode(StrEnum):
    DESTINATION_MISSING = "destination_missing"
    DESTINATION_AMBIGUOUS = "destination_ambiguous"
    DATE_MISSING = "date_missing"
    DATE_AMBIGUOUS = "date_ambiguous"
    STATE_CONFLICT = "state_conflict"
    INTENT_AMBIGUITY = "intent_ambiguity"
    SPECIAL_CONSTRAINT = "special_constraint"
    FIXED_ARRANGEMENT = "fixed_arrangement"
    EARLY_GENERATION_BLOCKER = "early_generation_blocker"
    EXTERNAL_FACT = "external_fact"
    LOW_PRIORITY_DETAIL = "low_priority_detail"


class FinalSupplementStatus(StrEnum):
    NOT_REQUESTED = "not_requested"
    AWAITING_RESPONSE = "awaiting_response"
    CONFIRMED = "confirmed"


class ReadinessQuestion(ContractModel):
    model_config = ConfigDict(frozen=True)

    question_id: UUID
    concern_id: ShortText
    gap_code: ReadinessGapCode
    prompt: ShortText
    resolution_goal: ShortText
    asked_at_state_version: int = Field(ge=1, strict=True)
    answer_targets: tuple[SemanticTarget, ...] = ()
    related_conflict_id: UUID | None = None

    @model_validator(mode="after")
    def answer_requirement_matches_gap(self) -> ReadinessQuestion:
        if not self.answer_targets:
            raise ValueError("critical question requires at least one semantic answer target")
        if len(set(self.answer_targets)) != len(self.answer_targets):
            raise ValueError("critical question answer targets must be unique")
        if self.gap_code is ReadinessGapCode.STATE_CONFLICT:
            if self.related_conflict_id is None:
                raise ValueError("state-conflict question requires its conflict reference")
        elif self.related_conflict_id is not None:
            raise ValueError("only a state-conflict question can reference a conflict")
        return self


class CriticalQuestionResolutionProof(ContractModel):
    """Model judgement whose ownership and merged effects are verified by application code."""

    model_config = ConfigDict(frozen=True)

    question_id: UUID
    resolution_goal: ShortText
    trip_id: UUID
    source_message_id: UUID
    resolved: bool
    reason: ShortText
    evidence: ShortText
    operation_ids: tuple[UUID, ...] = ()
    confidence: Confidence

    @model_validator(mode="after")
    def operation_references_match_resolution(self) -> CriticalQuestionResolutionProof:
        if len(set(self.operation_ids)) != len(self.operation_ids):
            raise ValueError("critical-question proof operation IDs must be unique")
        if self.resolved and not self.operation_ids:
            raise ValueError("a resolved critical question requires at least one operation")
        return self


class CriticalQuestionResolutionRecord(ContractModel):
    model_config = ConfigDict(frozen=True)

    proof: CriticalQuestionResolutionProof
    resolved_at_state_version: int = Field(ge=1, strict=True)


class ReadinessAssumption(ContractModel):
    assumption_id: UUID
    concern_id: ShortText
    gap_code: ReadinessGapCode
    description: ShortText
    reason: ShortText
    recorded_state_version: int = Field(ge=1, strict=True)
    active: bool = True
    resolved_state_version: int | None = Field(default=None, ge=1, strict=True)

    @model_validator(mode="after")
    def resolution_matches_status(self) -> ReadinessAssumption:
        if self.active and self.resolved_state_version is not None:
            raise ValueError("an active readiness assumption cannot be resolved")
        if not self.active and self.resolved_state_version is None:
            raise ValueError("an inactive readiness assumption requires a resolution version")
        return self


class ReadinessRuntimeState(ContractModel):
    critical_questions_asked: int = Field(default=0, ge=0, le=3, strict=True)
    pending_critical_question: ReadinessQuestion | None = None
    critical_question_history: list[ReadinessQuestion] = Field(default_factory=list)
    critical_question_resolutions: list[CriticalQuestionResolutionRecord] = Field(
        default_factory=list
    )
    final_supplement_status: FinalSupplementStatus = FinalSupplementStatus.NOT_REQUESTED
    final_supplement_requested_state_version: int | None = Field(default=None, ge=1, strict=True)
    final_supplement_confirmed_state_version: int | None = Field(default=None, ge=1, strict=True)
    assumptions: list[ReadinessAssumption] = Field(default_factory=list)
    last_updated_state_version: int = Field(default=0, ge=0, strict=True)

    @model_validator(mode="after")
    def lifecycle_is_consistent(self) -> ReadinessRuntimeState:
        question_ids = [question.question_id for question in self.critical_question_history]
        if len(set(question_ids)) != len(question_ids):
            raise ValueError("critical-question history IDs must be unique")
        if len(self.critical_question_history) > self.critical_questions_asked:
            raise ValueError("critical-question history cannot exceed the asked count")
        resolutions_by_id = {
            record.proof.question_id: record for record in self.critical_question_resolutions
        }
        if len(resolutions_by_id) != len(self.critical_question_resolutions):
            raise ValueError("critical-question resolution IDs must be unique")
        history_by_id = {
            question.question_id: question for question in self.critical_question_history
        }
        for question_id, record in resolutions_by_id.items():
            question = history_by_id.get(question_id)
            if question is None:
                raise ValueError("critical-question resolution requires its question history")
            if record.proof.resolution_goal != question.resolution_goal:
                raise ValueError("critical-question resolution goal does not match its question")
            if not record.proof.resolved:
                raise ValueError("only resolved critical-question proofs can be recorded")
            if record.resolved_at_state_version <= question.asked_at_state_version:
                raise ValueError("critical-question resolution must be newer than its question")
        if self.pending_critical_question is not None:
            if self.critical_questions_asked == 0:
                raise ValueError("a pending critical question must be counted")
            if self.final_supplement_status is not FinalSupplementStatus.NOT_REQUESTED:
                raise ValueError(
                    "critical and final-supplement questions cannot be pending together"
                )
            if (
                self.pending_critical_question.asked_at_state_version
                > self.last_updated_state_version
            ):
                raise ValueError("pending critical question cannot be newer than readiness state")
            if self.pending_critical_question.question_id not in history_by_id:
                raise ValueError("pending critical question requires its question history")
            if self.pending_critical_question.question_id in resolutions_by_id:
                raise ValueError("a resolved critical question cannot remain pending")

        requested = self.final_supplement_requested_state_version
        confirmed = self.final_supplement_confirmed_state_version
        if self.final_supplement_status is FinalSupplementStatus.NOT_REQUESTED:
            if requested is not None or confirmed is not None:
                raise ValueError("unrequested final supplement cannot contain version references")
        elif self.final_supplement_status is FinalSupplementStatus.AWAITING_RESPONSE:
            if requested is None or confirmed is not None:
                raise ValueError("awaiting final supplement requires only its request version")
        elif requested is None or confirmed is None or confirmed < requested:
            raise ValueError("confirmed final supplement requires ordered version references")

        version_references = [
            version for version in (requested, confirmed) if version is not None
        ] + [assumption.recorded_state_version for assumption in self.assumptions]
        version_references += [
            assumption.resolved_state_version
            for assumption in self.assumptions
            if assumption.resolved_state_version is not None
        ]
        version_references += [
            record.resolved_at_state_version for record in self.critical_question_resolutions
        ]
        if any(version > self.last_updated_state_version for version in version_references):
            raise ValueError("readiness record cannot be newer than readiness state")

        concern_ids = [assumption.concern_id for assumption in self.assumptions]
        if len(set(concern_ids)) != len(concern_ids):
            raise ValueError("readiness assumptions must have unique concern IDs")
        assumption_ids = [assumption.assumption_id for assumption in self.assumptions]
        if len(set(assumption_ids)) != len(assumption_ids):
            raise ValueError("readiness assumptions must have unique IDs")
        return self
