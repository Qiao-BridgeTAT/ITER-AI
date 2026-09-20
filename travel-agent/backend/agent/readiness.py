"""Deterministic readiness, critical-question budget, and final supplement check."""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from uuid import UUID, uuid5

from pydantic import Field, model_validator

from backend.agent.readiness_state import (
    CriticalQuestionResolutionProof,
    CriticalQuestionResolutionRecord,
    FinalSupplementStatus,
    ReadinessAssumption,
    ReadinessGapCode,
    ReadinessQuestion,
    ReadinessRuntimeState,
)
from backend.agent.semantic_operations import (
    ConstraintOperation,
    SemanticOperation,
    SemanticOperationKind,
    SemanticTarget,
    semantic_operation_fingerprint,
)
from backend.agent.state_merge import (
    SemanticMergeOutcome,
    SemanticTripState,
    restore_semantic_state,
)
from backend.contracts.base import ContractModel
from backend.contracts.common import ShortText
from backend.contracts.enums import Confidence, ConstraintKind, EvidenceSource

MAX_CRITICAL_QUESTIONS = 3
FINAL_SUPPLEMENT_PROMPT = "还有需要补充或调整的吗？没有的话，我就整理旅行任务书。"
_READINESS_NAMESPACE = UUID("d0785691-d1c9-4b05-af20-35b33c7f95c8")
_NON_ASSUMABLE_GAPS = {
    ReadinessGapCode.DESTINATION_MISSING,
    ReadinessGapCode.DESTINATION_AMBIGUOUS,
    ReadinessGapCode.DATE_MISSING,
    ReadinessGapCode.DATE_AMBIGUOUS,
    ReadinessGapCode.STATE_CONFLICT,
}
_DEFAULT_ANSWER_TARGETS: dict[ReadinessGapCode, tuple[SemanticTarget, ...]] = {
    ReadinessGapCode.DESTINATION_MISSING: (SemanticTarget.DESTINATION,),
    ReadinessGapCode.DESTINATION_AMBIGUOUS: (SemanticTarget.DESTINATION,),
    ReadinessGapCode.DATE_MISSING: (SemanticTarget.DATE_RANGE,),
    ReadinessGapCode.DATE_AMBIGUOUS: (SemanticTarget.DATE_RANGE,),
    ReadinessGapCode.SPECIAL_CONSTRAINT: (SemanticTarget.SPECIAL_CONSTRAINTS,),
    ReadinessGapCode.FIXED_ARRANGEMENT: (SemanticTarget.SPECIAL_CONSTRAINTS,),
}
_DEFAULT_RESOLUTION_GOALS: dict[ReadinessGapCode, str] = {
    ReadinessGapCode.DESTINATION_MISSING: "确认本次旅行的单一目的地",
    ReadinessGapCode.DESTINATION_AMBIGUOUS: "消除本次旅行目的地的歧义",
    ReadinessGapCode.DATE_MISSING: "确认本次旅行一至五天的具体日期",
    ReadinessGapCode.DATE_AMBIGUOUS: "消除本次旅行日期的歧义",
    ReadinessGapCode.STATE_CONFLICT: "确认冲突信息中本次旅行应采用的版本",
    ReadinessGapCode.SPECIAL_CONSTRAINT: "确认会影响本次旅行安排的特殊限制",
    ReadinessGapCode.FIXED_ARRANGEMENT: "确认本次旅行已经固定的时间和安排",
}
_ANSWER_OUTCOMES = {
    SemanticMergeOutcome.APPLIED,
    SemanticMergeOutcome.REPLACED,
    SemanticMergeOutcome.REMOVED,
}
_EXPLICIT_ANSWER_SOURCES = {EvidenceSource.CARD, EvidenceSource.DIALOGUE}


class ReadinessResolutionSource(StrEnum):
    UNRESOLVED = "unresolved"
    PERSONAL_DEFAULTS = "personal_defaults"
    CURRENT_DIALOGUE = "current_dialogue"
    ATTACHMENT = "attachment"
    PROVIDER = "provider"
    SYSTEM_COMPUTABLE = "system_computable"


class ReadinessDecision(StrEnum):
    ASK_CRITICAL_QUESTION = "ask_critical_question"
    WAITING_FOR_CRITICAL_ANSWER = "waiting_for_critical_answer"
    BLOCKED_REQUIRED_INPUT = "blocked_required_input"
    REQUEST_FINAL_SUPPLEMENT = "request_final_supplement"
    WAITING_FOR_FINAL_SUPPLEMENT = "waiting_for_final_supplement"
    READY_FOR_TASK_BOOK = "ready_for_task_book"


class ReadinessFailureCode(StrEnum):
    STATE_VERSION_CONFLICT = "state_version_conflict"
    QUESTION_NOT_PENDING = "question_not_pending"
    QUESTION_NOT_ANSWERED = "question_not_answered"
    QUESTION_EVIDENCE_INVALID = "question_evidence_invalid"
    FINAL_SUPPLEMENT_NOT_PENDING = "final_supplement_not_pending"
    STALE_FINAL_SUPPLEMENT = "stale_final_supplement"


class FinalSupplementResponseKind(StrEnum):
    NO_MORE_INFORMATION = "no_more_information"
    START_TASK_BOOK = "start_task_book"
    SUPPLEMENT = "supplement"
    MODIFICATION = "modification"
    QUESTION = "question"


class ReadinessConcern(ContractModel):
    """A possible gap after model understanding, before asking the user."""

    concern_id: ShortText
    gap_code: ReadinessGapCode
    summary: ShortText
    materially_changes_plan: bool
    resolution_source: ReadinessResolutionSource = ReadinessResolutionSource.UNRESOLVED
    safe_conservative_assumption: bool = False
    question: ShortText | None = None
    resolution_goal: ShortText | None = None
    assumption_description: ShortText | None = None
    assumption_reason: ShortText | None = None
    answer_targets: tuple[SemanticTarget, ...] = ()
    related_conflict_id: UUID | None = None

    @model_validator(mode="after")
    def evidence_and_fallback_match_resolution(self) -> ReadinessConcern:
        if len(set(self.answer_targets)) != len(self.answer_targets):
            raise ValueError("readiness answer targets must be unique")
        if self.gap_code is ReadinessGapCode.STATE_CONFLICT:
            if (
                self.resolution_source is ReadinessResolutionSource.UNRESOLVED
                and self.materially_changes_plan
                and not self.safe_conservative_assumption
                and self.related_conflict_id is None
            ):
                raise ValueError("unresolved state conflict requires its conflict reference")
        elif self.related_conflict_id is not None:
            raise ValueError("only a state-conflict concern can reference a conflict")
        if self.gap_code in _NON_ASSUMABLE_GAPS and self.safe_conservative_assumption:
            raise ValueError("destination, date and state conflicts cannot use assumptions")
        if self.resolution_source is not ReadinessResolutionSource.UNRESOLVED:
            if self.safe_conservative_assumption:
                raise ValueError("a resolved readiness concern cannot require an assumption")
            return self
        if not self.materially_changes_plan:
            return self
        if self.safe_conservative_assumption:
            if self.assumption_description is None or self.assumption_reason is None:
                raise ValueError("a safe readiness fallback requires an assumption and reason")
        else:
            if self.question is None:
                raise ValueError("an unsafe material readiness gap requires one focused question")
            if not _answer_targets(self):
                raise ValueError("an unsafe material readiness gap requires an answer target")
        return self


class ReadinessAssessment(ContractModel):
    state: SemanticTripState
    decision: ReadinessDecision
    question: ReadinessQuestion | None = None
    final_supplement_prompt: ShortText | None = None
    blocking_concerns: list[ReadinessConcern] = Field(default_factory=list)
    active_assumptions: list[ReadinessAssumption] = Field(default_factory=list)
    critical_questions_remaining: int = Field(ge=0, le=3, strict=True)
    direct_generation_requested: bool = False

    @model_validator(mode="after")
    def payload_matches_decision(self) -> ReadinessAssessment:
        question_decisions = {
            ReadinessDecision.ASK_CRITICAL_QUESTION,
            ReadinessDecision.WAITING_FOR_CRITICAL_ANSWER,
        }
        if (self.question is not None) != (self.decision in question_decisions):
            raise ValueError("critical question payload does not match readiness decision")
        supplement_decisions = {
            ReadinessDecision.REQUEST_FINAL_SUPPLEMENT,
            ReadinessDecision.WAITING_FOR_FINAL_SUPPLEMENT,
        }
        if (self.final_supplement_prompt is not None) != (self.decision in supplement_decisions):
            raise ValueError("final supplement prompt does not match readiness decision")
        if self.blocking_concerns and self.decision not in {
            ReadinessDecision.ASK_CRITICAL_QUESTION,
            ReadinessDecision.WAITING_FOR_CRITICAL_ANSWER,
            ReadinessDecision.BLOCKED_REQUIRED_INPUT,
        }:
            raise ValueError("blocking concerns require a blocking readiness decision")
        return self


class FinalSupplementTransition(ContractModel):
    state: SemanticTripState
    ready_for_task_book: bool
    requires_reassessment: bool
    answer_question_then_resume: bool

    @model_validator(mode="after")
    def exactly_one_transition_is_selected(self) -> FinalSupplementTransition:
        selected = sum(
            (
                self.ready_for_task_book,
                self.requires_reassessment,
                self.answer_question_then_resume,
            )
        )
        if selected != 1:
            raise ValueError("final supplement response requires exactly one next transition")
        return self


class ReadinessError(ValueError):
    def __init__(self, code: ReadinessFailureCode, message: str) -> None:
        super().__init__(message)
        self.code = code


def assess_readiness(
    state: SemanticTripState,
    *,
    expected_state_version: int,
    concerns: Sequence[ReadinessConcern] = (),
    direct_generation_requested: bool = False,
) -> ReadinessAssessment:
    """Choose one readiness action without querying facts the system can obtain."""

    _require_state_version(state, expected_state_version)
    runtime = state.readiness.model_copy(deep=True)

    if runtime.pending_critical_question is not None:
        return _assessment(
            state,
            ReadinessDecision.WAITING_FOR_CRITICAL_ANSWER,
            question=runtime.pending_critical_question,
            blockers=_critical_concerns(state, concerns),
            direct_generation_requested=direct_generation_requested,
        )

    runtime, final_boundary_changed = _invalidate_stale_final_boundary(state, runtime)
    if runtime.final_supplement_status is FinalSupplementStatus.CONFIRMED:
        return _assessment(
            state,
            ReadinessDecision.READY_FOR_TASK_BOOK,
            direct_generation_requested=direct_generation_requested,
        )
    if runtime.final_supplement_status is FinalSupplementStatus.AWAITING_RESPONSE:
        return _assessment(
            state,
            ReadinessDecision.WAITING_FOR_FINAL_SUPPLEMENT,
            final_prompt=FINAL_SUPPLEMENT_PROMPT,
            direct_generation_requested=direct_generation_requested,
        )

    normalized_concerns = _all_concerns(state, concerns)
    safe_fallbacks = [concern for concern in normalized_concerns if _needs_assumption(concern)]
    critical_concerns = [
        concern for concern in normalized_concerns if _needs_critical_question(concern)
    ]
    next_version = state.state_version + 1
    runtime, assumptions_changed = _sync_assumptions(runtime, safe_fallbacks, next_version)

    if critical_concerns:
        if runtime.critical_questions_asked >= MAX_CRITICAL_QUESTIONS:
            next_state = (
                _state_with_readiness(state, runtime, next_version)
                if assumptions_changed or final_boundary_changed
                else state.model_copy(deep=True)
            )
            return _assessment(
                next_state,
                ReadinessDecision.BLOCKED_REQUIRED_INPUT,
                blockers=critical_concerns,
                direct_generation_requested=direct_generation_requested,
            )

        selected = min(critical_concerns, key=_concern_priority)
        question = ReadinessQuestion(
            question_id=uuid5(
                _READINESS_NAMESPACE,
                f"{state.trip_id}:{next_version}:{selected.concern_id}",
            ),
            concern_id=selected.concern_id,
            gap_code=selected.gap_code,
            prompt=selected.question or "请补充这项会影响行程的信息。",
            resolution_goal=(
                selected.resolution_goal
                or _DEFAULT_RESOLUTION_GOALS.get(selected.gap_code)
                or selected.summary
            ),
            asked_at_state_version=next_version,
            answer_targets=_answer_targets(selected),
            related_conflict_id=selected.related_conflict_id,
        )
        runtime = runtime.model_copy(
            update={
                "critical_questions_asked": runtime.critical_questions_asked + 1,
                "pending_critical_question": question,
                "critical_question_history": [
                    *runtime.critical_question_history,
                    question,
                ],
                "final_supplement_status": FinalSupplementStatus.NOT_REQUESTED,
                "final_supplement_requested_state_version": None,
                "final_supplement_confirmed_state_version": None,
            },
            deep=True,
        )
        next_state = _state_with_readiness(state, runtime, next_version)
        return _assessment(
            next_state,
            ReadinessDecision.ASK_CRITICAL_QUESTION,
            question=question,
            blockers=critical_concerns,
            direct_generation_requested=direct_generation_requested,
        )

    runtime = runtime.model_copy(
        update={
            "final_supplement_status": FinalSupplementStatus.AWAITING_RESPONSE,
            "final_supplement_requested_state_version": next_version,
            "final_supplement_confirmed_state_version": None,
        },
        deep=True,
    )
    next_state = _state_with_readiness(state, runtime, next_version)
    return _assessment(
        next_state,
        ReadinessDecision.REQUEST_FINAL_SUPPLEMENT,
        final_prompt=FINAL_SUPPLEMENT_PROMPT,
        direct_generation_requested=direct_generation_requested,
    )


def record_critical_answer(
    state: SemanticTripState,
    *,
    expected_state_version: int,
    proof: CriticalQuestionResolutionProof,
    answer_operations: Sequence[SemanticOperation],
    current_source_message_id: UUID,
    current_user_text: str | None = None,
) -> SemanticTripState:
    """Close one critical question after semantic judgement and deterministic evidence checks."""

    state = restore_semantic_state(state.model_dump(mode="json"))
    proof = CriticalQuestionResolutionProof.model_validate(proof.model_dump(mode="json"))
    _require_state_version(state, expected_state_version)
    pending = state.readiness.pending_critical_question
    if pending is None or pending.question_id != proof.question_id:
        raise ReadinessError(
            ReadinessFailureCode.QUESTION_NOT_PENDING,
            "critical question is not pending for this trip",
        )
    if (
        proof.trip_id != state.trip_id
        or proof.source_message_id != current_source_message_id
        or proof.resolution_goal != pending.resolution_goal
    ):
        raise ReadinessError(
            ReadinessFailureCode.QUESTION_EVIDENCE_INVALID,
            "critical answer proof ownership or resolution goal is invalid",
        )
    if not proof.resolved or proof.confidence is not Confidence.HIGH:
        raise ReadinessError(
            ReadinessFailureCode.QUESTION_NOT_ANSWERED,
            "model judgement did not resolve the current critical question with high confidence",
        )
    if state.state_version <= pending.asked_at_state_version:
        raise ReadinessError(
            ReadinessFailureCode.QUESTION_NOT_ANSWERED,
            "critical answer must be merged before closing the question",
        )
    operations_by_id = {operation.operation_id: operation for operation in answer_operations}
    if len(operations_by_id) != len(answer_operations):
        raise ReadinessError(
            ReadinessFailureCode.QUESTION_EVIDENCE_INVALID,
            "critical answer operations contain duplicate IDs",
        )
    claimed_operations: list[SemanticOperation] = []
    for operation_id in proof.operation_ids:
        operation = operations_by_id.get(operation_id)
        if operation is None:
            raise ReadinessError(
                ReadinessFailureCode.QUESTION_EVIDENCE_INVALID,
                "critical answer proof references an operation outside the current input",
            )
        if (
            operation.trip_id != state.trip_id
            or operation.evidence.source_trip_id != state.trip_id
            or operation.evidence.source_message_id != current_source_message_id
        ):
            raise ReadinessError(
                ReadinessFailureCode.QUESTION_EVIDENCE_INVALID,
                "critical answer operation ownership does not match the current input",
            )
        claimed_operations.append(operation)
    if current_user_text is not None:
        if any(
            operation.evidence.source is not EvidenceSource.DIALOGUE
            for operation in claimed_operations
        ) or _normalized_text(proof.evidence) not in _normalized_text(current_user_text):
            raise ReadinessError(
                ReadinessFailureCode.QUESTION_EVIDENCE_INVALID,
                "critical answer evidence is not present in the current user message",
            )
    elif any(
        operation.evidence.source is not EvidenceSource.CARD for operation in claimed_operations
    ):
        raise ReadinessError(
            ReadinessFailureCode.QUESTION_EVIDENCE_INVALID,
            "non-text critical answers require current attachment evidence",
        )

    accepted_operations = _accepted_answer_operations(state, pending, claimed_operations)
    if (
        len(accepted_operations) != len(claimed_operations)
        or not accepted_operations
        or not _answer_postcondition_holds(state, pending)
    ):
        raise ReadinessError(
            ReadinessFailureCode.QUESTION_NOT_ANSWERED,
            "merged operations did not resolve the pending critical question",
        )
    next_version = state.state_version + 1
    resolution_record = CriticalQuestionResolutionRecord(
        proof=proof,
        resolved_at_state_version=next_version,
    )
    runtime = state.readiness.model_copy(
        update={
            "pending_critical_question": None,
            "critical_question_resolutions": [
                *state.readiness.critical_question_resolutions,
                resolution_record,
            ],
        },
        deep=True,
    )
    return _state_with_readiness(state, runtime, next_version)


def record_final_supplement_response(
    state: SemanticTripState,
    *,
    expected_state_version: int,
    response: FinalSupplementResponseKind,
) -> FinalSupplementTransition:
    """Apply the distinct information-collection confirmation boundary."""

    _require_state_version(state, expected_state_version)
    runtime = state.readiness
    if runtime.final_supplement_status is not FinalSupplementStatus.AWAITING_RESPONSE:
        raise ReadinessError(
            ReadinessFailureCode.FINAL_SUPPLEMENT_NOT_PENDING,
            "final supplement confirmation is not pending",
        )
    requested_version = runtime.final_supplement_requested_state_version
    assert requested_version is not None

    if response is FinalSupplementResponseKind.QUESTION:
        return FinalSupplementTransition(
            state=state.model_copy(deep=True),
            ready_for_task_book=False,
            requires_reassessment=False,
            answer_question_then_resume=True,
        )

    next_version = state.state_version + 1
    if response in {
        FinalSupplementResponseKind.NO_MORE_INFORMATION,
        FinalSupplementResponseKind.START_TASK_BOOK,
    }:
        if state.state_version != requested_version:
            raise ReadinessError(
                ReadinessFailureCode.STALE_FINAL_SUPPLEMENT,
                "final supplement confirmation is stale after state changes",
            )
        confirmed_runtime = runtime.model_copy(
            update={
                "final_supplement_status": FinalSupplementStatus.CONFIRMED,
                "final_supplement_confirmed_state_version": next_version,
            },
            deep=True,
        )
        return FinalSupplementTransition(
            state=_state_with_readiness(state, confirmed_runtime, next_version),
            ready_for_task_book=True,
            requires_reassessment=False,
            answer_question_then_resume=False,
        )

    if state.state_version <= requested_version:
        raise ReadinessError(
            ReadinessFailureCode.STALE_FINAL_SUPPLEMENT,
            "supplement or modification must be merged before reassessment",
        )
    reset_runtime = runtime.model_copy(
        update={
            "final_supplement_status": FinalSupplementStatus.NOT_REQUESTED,
            "final_supplement_requested_state_version": None,
            "final_supplement_confirmed_state_version": None,
        },
        deep=True,
    )
    return FinalSupplementTransition(
        state=_state_with_readiness(state, reset_runtime, next_version),
        ready_for_task_book=False,
        requires_reassessment=True,
        answer_question_then_resume=False,
    )


def _all_concerns(
    state: SemanticTripState,
    supplied: Sequence[ReadinessConcern],
) -> list[ReadinessConcern]:
    concerns = list(supplied)
    for conflict in state.conflicts:
        conflict_target = _conflict_target(state, conflict.state_key)
        concerns.append(
            ReadinessConcern(
                concern_id=f"state-conflict:{conflict.conflict_id}",
                gap_code=ReadinessGapCode.STATE_CONFLICT,
                summary="当前旅行资料包含尚未解决的冲突",
                materially_changes_plan=True,
                question="我发现有两种互相冲突的说法，你希望以哪一个为准？",
                answer_targets=(conflict_target,),
                related_conflict_id=conflict.conflict_id,
            )
        )
    destination_gap_already_supplied = any(
        concern.gap_code
        in {ReadinessGapCode.DESTINATION_MISSING, ReadinessGapCode.DESTINATION_AMBIGUOUS}
        and _needs_critical_question(concern)
        for concern in supplied
    )
    if not _has_target(state, SemanticTarget.DESTINATION) and not destination_gap_already_supplied:
        concerns.append(
            ReadinessConcern(
                concern_id="core.destination",
                gap_code=ReadinessGapCode.DESTINATION_MISSING,
                summary="尚未确定单一目的地",
                materially_changes_plan=True,
                question="这次想去哪个城市？",
            )
        )
    date_gap_already_supplied = any(
        concern.gap_code in {ReadinessGapCode.DATE_MISSING, ReadinessGapCode.DATE_AMBIGUOUS}
        and _needs_critical_question(concern)
        for concern in supplied
    )
    if not _has_target(state, SemanticTarget.DATE_RANGE) and not date_gap_already_supplied:
        concerns.append(
            ReadinessConcern(
                concern_id="core.date_range",
                gap_code=ReadinessGapCode.DATE_MISSING,
                summary="尚未确定一至五天的具体日期",
                materially_changes_plan=True,
                question="这次准备哪几天出行？",
            )
        )
    concern_ids = [concern.concern_id for concern in concerns]
    if len(set(concern_ids)) != len(concern_ids):
        raise ValueError("readiness concern IDs must be unique")
    return concerns


def _critical_concerns(
    state: SemanticTripState,
    concerns: Sequence[ReadinessConcern],
) -> list[ReadinessConcern]:
    return [
        concern for concern in _all_concerns(state, concerns) if _needs_critical_question(concern)
    ]


def _has_target(state: SemanticTripState, target: SemanticTarget) -> bool:
    return any(
        entry.operation.target is target
        and entry.operation.operation in {SemanticOperationKind.SET, SemanticOperationKind.OVERRIDE}
        for entry in state.entries
    )


def _answer_targets(concern: ReadinessConcern) -> tuple[SemanticTarget, ...]:
    if concern.answer_targets:
        return concern.answer_targets
    return _DEFAULT_ANSWER_TARGETS.get(concern.gap_code, ())


def _conflict_target(state: SemanticTripState, state_key: str) -> SemanticTarget:
    for entry in state.entries:
        if entry.state_key == state_key:
            return entry.operation.target
    raise ValueError("readiness conflict does not reference an active semantic entry")


def _accepted_answer_operations(
    state: SemanticTripState,
    question: ReadinessQuestion,
    operations: Sequence[SemanticOperation],
) -> list[SemanticOperation]:
    audits = {audit.operation_id: audit for audit in state.audit_log}
    accepted: list[SemanticOperation] = []
    for operation in operations:
        if operation.trip_id != state.trip_id or operation.evidence.source_trip_id != state.trip_id:
            continue
        if operation.evidence.source not in _EXPLICIT_ANSWER_SOURCES:
            continue
        if operation.target not in question.answer_targets:
            continue
        audit = audits.get(operation.operation_id)
        if (
            audit is None
            or audit.state_version <= question.asked_at_state_version
            or audit.outcome not in _ANSWER_OUTCOMES
            or audit.operation_fingerprint != semantic_operation_fingerprint(operation)
        ):
            continue
        if question.gap_code is ReadinessGapCode.FIXED_ARRANGEMENT and (
            not isinstance(operation, ConstraintOperation)
            or operation.value.kind is not ConstraintKind.SCHEDULE
        ):
            continue
        accepted.append(operation)
    return accepted


def _answer_postcondition_holds(
    state: SemanticTripState,
    question: ReadinessQuestion,
) -> bool:
    if question.gap_code in {
        ReadinessGapCode.DESTINATION_MISSING,
        ReadinessGapCode.DESTINATION_AMBIGUOUS,
    }:
        return _has_target(state, SemanticTarget.DESTINATION)
    if question.gap_code in {
        ReadinessGapCode.DATE_MISSING,
        ReadinessGapCode.DATE_AMBIGUOUS,
    }:
        return _has_target(state, SemanticTarget.DATE_RANGE)
    if question.gap_code is ReadinessGapCode.STATE_CONFLICT:
        conflict_id = question.related_conflict_id
        return conflict_id is not None and all(
            conflict.conflict_id != conflict_id for conflict in state.conflicts
        )
    return True


def _normalized_text(value: str) -> str:
    return " ".join(value.split())


def _needs_critical_question(concern: ReadinessConcern) -> bool:
    return (
        concern.materially_changes_plan
        and concern.resolution_source is ReadinessResolutionSource.UNRESOLVED
        and not concern.safe_conservative_assumption
    )


def _needs_assumption(concern: ReadinessConcern) -> bool:
    return (
        concern.materially_changes_plan
        and concern.resolution_source is ReadinessResolutionSource.UNRESOLVED
        and concern.safe_conservative_assumption
    )


def _concern_priority(concern: ReadinessConcern) -> tuple[int, str]:
    order = {
        ReadinessGapCode.STATE_CONFLICT: 0,
        ReadinessGapCode.DESTINATION_AMBIGUOUS: 10,
        ReadinessGapCode.DESTINATION_MISSING: 11,
        ReadinessGapCode.DATE_AMBIGUOUS: 20,
        ReadinessGapCode.DATE_MISSING: 21,
        ReadinessGapCode.SPECIAL_CONSTRAINT: 30,
        ReadinessGapCode.FIXED_ARRANGEMENT: 40,
        ReadinessGapCode.INTENT_AMBIGUITY: 50,
        ReadinessGapCode.EARLY_GENERATION_BLOCKER: 60,
        ReadinessGapCode.EXTERNAL_FACT: 70,
        ReadinessGapCode.LOW_PRIORITY_DETAIL: 80,
    }
    return order[concern.gap_code], concern.concern_id


def _sync_assumptions(
    runtime: ReadinessRuntimeState,
    concerns: Sequence[ReadinessConcern],
    state_version: int,
) -> tuple[ReadinessRuntimeState, bool]:
    active_ids = {concern.concern_id for concern in concerns}
    assumptions = [assumption.model_copy(deep=True) for assumption in runtime.assumptions]
    changed = False
    by_concern = {assumption.concern_id: index for index, assumption in enumerate(assumptions)}

    for index, assumption in enumerate(assumptions):
        if assumption.active and assumption.concern_id not in active_ids:
            assumptions[index] = assumption.model_copy(
                update={"active": False, "resolved_state_version": state_version}
            )
            changed = True

    for concern in concerns:
        description = concern.assumption_description
        reason = concern.assumption_reason
        assert description is not None and reason is not None
        existing_index = by_concern.get(concern.concern_id)
        if existing_index is None:
            assumptions.append(
                ReadinessAssumption(
                    assumption_id=uuid5(
                        _READINESS_NAMESPACE,
                        f"{concern.concern_id}:assumption",
                    ),
                    concern_id=concern.concern_id,
                    gap_code=concern.gap_code,
                    description=description,
                    reason=reason,
                    recorded_state_version=state_version,
                )
            )
            changed = True
            continue
        existing = assumptions[existing_index]
        if (
            not existing.active
            or existing.description != description
            or existing.reason != reason
            or existing.gap_code is not concern.gap_code
        ):
            assumptions[existing_index] = existing.model_copy(
                update={
                    "gap_code": concern.gap_code,
                    "description": description,
                    "reason": reason,
                    "recorded_state_version": state_version,
                    "active": True,
                    "resolved_state_version": None,
                }
            )
            changed = True
    return runtime.model_copy(update={"assumptions": assumptions}, deep=True), changed


def _invalidate_stale_final_boundary(
    state: SemanticTripState,
    runtime: ReadinessRuntimeState,
) -> tuple[ReadinessRuntimeState, bool]:
    reference = (
        runtime.final_supplement_confirmed_state_version
        if runtime.final_supplement_status is FinalSupplementStatus.CONFIRMED
        else runtime.final_supplement_requested_state_version
    )
    if reference is None or reference == state.state_version:
        return runtime, False
    return (
        runtime.model_copy(
            update={
                "final_supplement_status": FinalSupplementStatus.NOT_REQUESTED,
                "final_supplement_requested_state_version": None,
                "final_supplement_confirmed_state_version": None,
            },
            deep=True,
        ),
        True,
    )


def _state_with_readiness(
    state: SemanticTripState,
    runtime: ReadinessRuntimeState,
    state_version: int,
) -> SemanticTripState:
    payload = state.model_dump(mode="json")
    runtime = runtime.model_copy(update={"last_updated_state_version": state_version}, deep=True)
    payload["state_version"] = state_version
    payload["readiness"] = runtime.model_dump(mode="json")
    return restore_semantic_state(payload)


def _assessment(
    state: SemanticTripState,
    decision: ReadinessDecision,
    *,
    question: ReadinessQuestion | None = None,
    final_prompt: str | None = None,
    blockers: Sequence[ReadinessConcern] = (),
    direct_generation_requested: bool = False,
) -> ReadinessAssessment:
    return ReadinessAssessment(
        state=state,
        decision=decision,
        question=question,
        final_supplement_prompt=final_prompt,
        blocking_concerns=list(blockers),
        active_assumptions=[
            assumption for assumption in state.readiness.assumptions if assumption.active
        ],
        critical_questions_remaining=(
            MAX_CRITICAL_QUESTIONS - state.readiness.critical_questions_asked
        ),
        direct_generation_requested=direct_generation_requested,
    )


def _require_state_version(state: SemanticTripState, expected_state_version: int) -> None:
    if state.state_version != expected_state_version:
        raise ReadinessError(
            ReadinessFailureCode.STATE_VERSION_CONFLICT,
            "readiness state version is stale",
        )
