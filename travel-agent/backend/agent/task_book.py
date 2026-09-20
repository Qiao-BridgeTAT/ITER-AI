"""Deterministic task-book projection, confirmation, and planning gate."""

from __future__ import annotations

import json
from datetime import date
from enum import StrEnum
from uuid import UUID, uuid5

from pydantic import model_validator

from backend.agent.readiness_state import FinalSupplementStatus
from backend.agent.semantic_operations import (
    AttractionIntentOperation,
    ConstraintOperation,
    DateRangeOperation,
    DestinationOperation,
    DiningPreferenceKind,
    DiningPreferenceOperation,
    ExperiencePreferenceOperation,
    LodgingPreferenceOperation,
    PacePreferenceOperation,
    SemanticOperation,
    SemanticOperationBatch,
    SemanticPersistenceScope,
    SemanticTarget,
    TaskBookConfirmationOperation,
    TransportPreferenceOperation,
)
from backend.agent.state_merge import (
    SemanticMergeOutcome,
    SemanticStateEntry,
    SemanticTripState,
    merge_state,
    restore_semantic_state,
)
from backend.agent.task_book_state import (
    SemanticTaskBook,
    SemanticTaskBookStatus,
    TaskBookAssumptionItem,
    TaskBookAttractionItem,
    TaskBookConstraintItem,
    TaskBookLodgingItem,
    TaskBookPreferenceItem,
    TaskBookRestaurantItem,
)
from backend.contracts.base import ContractModel
from backend.contracts.enums import Confidence

_TASK_BOOK_NAMESPACE = UUID("a543c3ef-7965-49c3-b12f-e10ad8c2b4c0")
_PREFERENCE_TARGETS = {
    SemanticTarget.DINING_PREFERENCES,
    SemanticTarget.LODGING_PREFERENCES,
    SemanticTarget.TRANSPORT_PREFERENCES,
    SemanticTarget.PACE_PREFERENCES,
    SemanticTarget.EXPERIENCE_PREFERENCES,
}


class TaskBookBuildStatus(StrEnum):
    CREATED = "created"
    EXISTING = "existing"


class TaskBookFailureCode(StrEnum):
    STATE_VERSION_CONFLICT = "state_version_conflict"
    NOT_READY = "not_ready"
    INVALID_SEMANTIC_STATE = "invalid_semantic_state"
    TASK_BOOK_NOT_FOUND = "task_book_not_found"
    TASK_BOOK_STALE = "task_book_stale"
    TASK_BOOK_ALREADY_CONFIRMED = "task_book_already_confirmed"
    TASK_BOOK_NOT_CONFIRMED = "task_book_not_confirmed"
    CROSS_TRIP = "cross_trip"


class TaskBookBuildResult(ContractModel):
    state: SemanticTripState
    task_book: SemanticTaskBook
    status: TaskBookBuildStatus

    @model_validator(mode="after")
    def task_book_is_current(self) -> TaskBookBuildResult:
        if self.state.task_book != self.task_book:
            raise ValueError("task-book build result must expose the state's current task book")
        return self


class TaskBookConfirmationResult(ContractModel):
    state: SemanticTripState
    task_book: SemanticTaskBook
    ready_for_planning: bool = True

    @model_validator(mode="after")
    def confirmation_is_the_planning_boundary(self) -> TaskBookConfirmationResult:
        if self.state.task_book != self.task_book:
            raise ValueError("confirmed task book must be the state's current task book")
        if self.task_book.status is not SemanticTaskBookStatus.CONFIRMED:
            raise ValueError("planning boundary requires a confirmed task book")
        return self


class TaskBookError(ValueError):
    def __init__(self, code: TaskBookFailureCode, message: str) -> None:
        super().__init__(message)
        self.code = code


def build_task_book(
    state: SemanticTripState,
    *,
    expected_state_version: int,
    business_date: date,
) -> TaskBookBuildResult:
    """Project a pending task book from the stable semantic state only."""

    state = restore_semantic_state(state.model_dump(mode="json"))
    _require_state_version(state, expected_state_version)
    _require_resolved_critical_questions(state)
    if state.task_book is not None:
        return TaskBookBuildResult(
            state=state.model_copy(deep=True),
            task_book=state.task_book.model_copy(deep=True),
            status=TaskBookBuildStatus.EXISTING,
        )
    readiness = state.readiness
    if (
        readiness.final_supplement_status is not FinalSupplementStatus.CONFIRMED
        or readiness.final_supplement_confirmed_state_version != state.state_version
    ):
        raise TaskBookError(
            TaskBookFailureCode.NOT_READY,
            "task book requires the current final-supplement confirmation",
        )
    if state.conflicts:
        raise TaskBookError(
            TaskBookFailureCode.INVALID_SEMANTIC_STATE,
            "task book cannot be projected while semantic conflicts remain",
        )

    destination_entry = _exact_entry(state, SemanticTarget.DESTINATION)
    date_entry = _exact_entry(state, SemanticTarget.DATE_RANGE)
    destination_operation = destination_entry.operation
    date_operation = date_entry.operation
    if (
        not isinstance(destination_operation, DestinationOperation)
        or destination_operation.value is None
    ):
        raise TaskBookError(
            TaskBookFailureCode.INVALID_SEMANTIC_STATE,
            "task book requires one concrete destination",
        )
    if not isinstance(date_operation, DateRangeOperation) or date_operation.value is None:
        raise TaskBookError(
            TaskBookFailureCode.INVALID_SEMANTIC_STATE,
            "task book requires one concrete date range",
        )

    preference_entries = _effective_preference_entries(state)
    preferences = tuple(_preference_item(entry) for entry in preference_entries)
    attraction_entries = _entries_for(state, SemanticTarget.ATTRACTION_INTENTS)
    attraction_intents = tuple(
        TaskBookAttractionItem(
            source_operation_id=operation.operation_id,
            place_id=operation.value.place_id,
            place_name=operation.value.place_name,
            intent=operation.value.intent,
        )
        for operation in (entry.operation for entry in attraction_entries)
        if isinstance(operation, AttractionIntentOperation)
    )
    important_restaurants = tuple(
        _restaurant_item(operation)
        for operation in (entry.operation for entry in preference_entries)
        if isinstance(operation, DiningPreferenceOperation)
        and operation.value.kind is DiningPreferenceKind.SPECIFIC_RESTAURANT
    )
    lodging_direction = tuple(
        TaskBookLodgingItem(
            source_operation_id=operation.operation_id,
            kind=operation.value.kind,
            value=operation.value.value,
            place_id=operation.value.place_id,
        )
        for operation in (entry.operation for entry in preference_entries)
        if isinstance(operation, LodgingPreferenceOperation)
    )
    constraint_entries = _entries_for(state, SemanticTarget.SPECIAL_CONSTRAINTS)
    key_constraints = tuple(
        TaskBookConstraintItem(
            source_operation_id=operation.operation_id,
            kind=operation.value.kind,
            description=operation.value.description,
        )
        for operation in (entry.operation for entry in constraint_entries)
        if isinstance(operation, ConstraintOperation)
    )
    assumptions = tuple(
        TaskBookAssumptionItem(
            assumption_id=assumption.assumption_id,
            description=assumption.description,
            reason=assumption.reason,
        )
        for assumption in sorted(
            (item for item in state.readiness.assumptions if item.active),
            key=lambda item: str(item.assumption_id),
        )
    )

    included_entries = [
        destination_entry,
        date_entry,
        *preference_entries,
        *attraction_entries,
        *constraint_entries,
    ]
    source_operation_ids = tuple(
        dict.fromkeys(entry.operation.operation_id for entry in included_entries)
    )
    next_version = state.state_version + 1
    identity_payload = {
        "trip_id": str(state.trip_id),
        "source_state_version": state.state_version,
        "source_operation_ids": [str(item) for item in source_operation_ids],
        "assumption_ids": [str(item.assumption_id) for item in assumptions],
    }
    task_book = SemanticTaskBook.model_validate(
        {
            "task_book_id": str(
                uuid5(
                    _TASK_BOOK_NAMESPACE,
                    json.dumps(identity_payload, ensure_ascii=False, sort_keys=True),
                )
            ),
            "trip_id": str(state.trip_id),
            "revision": next_version,
            "source_state_version": state.state_version,
            "published_state_version": next_version,
            "destination": destination_operation.value.model_dump(mode="json"),
            "date_range": date_operation.value.model_dump(mode="json"),
            "preferences": [item.model_dump(mode="json") for item in preferences],
            "attraction_intents": [item.model_dump(mode="json") for item in attraction_intents],
            "important_restaurants": [
                item.model_dump(mode="json") for item in important_restaurants
            ],
            "lodging_direction": [item.model_dump(mode="json") for item in lodging_direction],
            "key_constraints": [item.model_dump(mode="json") for item in key_constraints],
            "assumptions": [item.model_dump(mode="json") for item in assumptions],
            "source_operation_ids": [str(item) for item in source_operation_ids],
        },
        context={"today": business_date},
    )
    runtime = readiness.model_copy(
        update={
            "final_supplement_confirmed_state_version": next_version,
            "last_updated_state_version": next_version,
        },
        deep=True,
    )
    payload = state.model_dump(mode="json")
    payload["state_version"] = next_version
    payload["readiness"] = runtime.model_dump(mode="json")
    payload["task_book"] = task_book.model_dump(mode="json")
    next_state = restore_semantic_state(payload)
    return TaskBookBuildResult(
        state=next_state,
        task_book=task_book,
        status=TaskBookBuildStatus.CREATED,
    )


def _require_resolved_critical_questions(state: SemanticTripState) -> None:
    runtime = state.readiness
    if runtime.pending_critical_question is not None:
        raise TaskBookError(
            TaskBookFailureCode.NOT_READY,
            "task book cannot be projected while a critical question is unresolved",
        )
    questions = {question.question_id: question for question in runtime.critical_question_history}
    resolutions = {
        record.proof.question_id: record for record in runtime.critical_question_resolutions
    }
    if len(questions) != runtime.critical_questions_asked or set(resolutions) != set(questions):
        raise TaskBookError(
            TaskBookFailureCode.NOT_READY,
            "task book requires verified evidence for every critical question",
        )
    audits = {audit.operation_id: audit for audit in state.audit_log}
    accepted_outcomes = {
        SemanticMergeOutcome.APPLIED,
        SemanticMergeOutcome.REPLACED,
        SemanticMergeOutcome.REMOVED,
    }
    for question_id, question in questions.items():
        record = resolutions[question_id]
        proof = record.proof
        if (
            not proof.resolved
            or proof.confidence is not Confidence.HIGH
            or proof.trip_id != state.trip_id
            or proof.resolution_goal != question.resolution_goal
            or record.resolved_at_state_version > state.state_version
        ):
            raise TaskBookError(
                TaskBookFailureCode.NOT_READY,
                "task book critical-question evidence is inconsistent",
            )
        for operation_id in proof.operation_ids:
            audit = audits.get(operation_id)
            if (
                audit is None
                or audit.outcome not in accepted_outcomes
                or audit.state_version <= question.asked_at_state_version
                or audit.state_version >= record.resolved_at_state_version
            ):
                raise TaskBookError(
                    TaskBookFailureCode.NOT_READY,
                    "task book critical-question operation evidence is invalid",
                )


def confirm_task_book(
    state: SemanticTripState,
    *,
    task_book_id: UUID,
    confirmation: TaskBookConfirmationOperation,
    expected_state_version: int,
    business_date: date,
) -> TaskBookConfirmationResult:
    """Confirm exactly the pending task-book version currently stored in the state."""

    _require_state_version(state, expected_state_version)
    task_book = state.task_book
    if task_book is None:
        raise TaskBookError(TaskBookFailureCode.TASK_BOOK_NOT_FOUND, "no current task book")
    if task_book.trip_id != state.trip_id or confirmation.trip_id != state.trip_id:
        raise TaskBookError(TaskBookFailureCode.CROSS_TRIP, "task-book confirmation is cross-trip")
    if task_book.task_book_id != task_book_id:
        raise TaskBookError(
            TaskBookFailureCode.TASK_BOOK_STALE,
            "task-book confirmation references an old version",
        )
    if task_book.status is SemanticTaskBookStatus.CONFIRMED:
        raise TaskBookError(
            TaskBookFailureCode.TASK_BOOK_ALREADY_CONFIRMED,
            "task book is already confirmed",
        )
    if task_book.published_state_version != state.state_version:
        raise TaskBookError(
            TaskBookFailureCode.TASK_BOOK_STALE,
            "task book is stale after semantic state changes",
        )

    operation_batch = SemanticOperationBatch.model_validate(
        {
            "trip_id": str(state.trip_id),
            "operations": [confirmation.model_dump(mode="json")],
        },
        context={"today": business_date},
    )
    merged = merge_state(
        state,
        operation_batch,
        expected_state_version=state.state_version,
        business_date=business_date,
    ).state
    confirmed_payload = task_book.model_dump(mode="json")
    confirmed_payload.update(
        {
            "status": SemanticTaskBookStatus.CONFIRMED.value,
            "confirmation_operation_id": str(confirmation.operation_id),
            "confirmed_state_version": merged.state_version,
        }
    )
    confirmed = SemanticTaskBook.model_validate(
        confirmed_payload,
        context={"today": business_date},
    )
    state_payload = merged.model_dump(mode="json")
    state_payload["task_book"] = confirmed.model_dump(mode="json")
    confirmed_state = restore_semantic_state(state_payload)
    return TaskBookConfirmationResult(state=confirmed_state, task_book=confirmed)


def require_confirmed_task_book(
    state: SemanticTripState,
    *,
    expected_state_version: int,
) -> SemanticTaskBook:
    """Deterministic gate used before any formal planning node may run."""

    _require_state_version(state, expected_state_version)
    task_book = state.task_book
    if task_book is None or task_book.status is not SemanticTaskBookStatus.CONFIRMED:
        raise TaskBookError(
            TaskBookFailureCode.TASK_BOOK_NOT_CONFIRMED,
            "formal planning requires a confirmed task book",
        )
    if (
        task_book.confirmed_state_version is None
        or task_book.confirmed_state_version > state.state_version
    ):
        raise TaskBookError(
            TaskBookFailureCode.TASK_BOOK_STALE,
            "task-book confirmation version is not valid for this state",
        )
    if not any(
        entry.operation.operation_id == task_book.confirmation_operation_id
        and entry.operation.target is SemanticTarget.TASK_BOOK_CONFIRMATION
        for entry in state.entries
    ):
        raise TaskBookError(
            TaskBookFailureCode.TASK_BOOK_STALE,
            "task-book confirmation was invalidated by a later modification",
        )
    return task_book.model_copy(deep=True)


def _exact_entry(state: SemanticTripState, target: SemanticTarget) -> SemanticStateEntry:
    entries = _entries_for(state, target)
    if len(entries) != 1:
        raise TaskBookError(
            TaskBookFailureCode.INVALID_SEMANTIC_STATE,
            f"task book requires exactly one active {target.value}",
        )
    return entries[0]


def _entries_for(state: SemanticTripState, target: SemanticTarget) -> list[SemanticStateEntry]:
    return sorted(
        (entry for entry in state.entries if entry.operation.target is target),
        key=lambda entry: entry.state_key,
    )


def _effective_preference_entries(state: SemanticTripState) -> list[SemanticStateEntry]:
    entries = [entry for entry in state.entries if entry.operation.target in _PREFERENCE_TARGETS]
    entries.sort(
        key=lambda entry: (
            entry.operation.persistence_scope is SemanticPersistenceScope.CURRENT_TRIP,
            entry.applied_state_version,
            entry.state_key,
        )
    )
    effective: dict[str, SemanticStateEntry] = {}
    for entry in entries:
        operation = entry.operation
        if (
            isinstance(operation, DiningPreferenceOperation)
            and operation.persistence_scope is SemanticPersistenceScope.CURRENT_TRIP
        ):
            if operation.value.kind is DiningPreferenceKind.OPEN_TO_ANY:
                effective = {
                    key: value for key, value in effective.items() if not key.startswith("dining:")
                }
            else:
                effective.pop("dining:*", None)
        effective[_preference_dimension(operation)] = entry
    return sorted(effective.values(), key=lambda entry: entry.state_key)


def _preference_dimension(operation: SemanticOperation) -> str:
    impact = operation.impact_scope
    impact_key = f"{impact.kind.value}:{impact.day_number or ''}:{impact.item_id or ''}"
    if isinstance(operation, DiningPreferenceOperation):
        if operation.value.kind is DiningPreferenceKind.OPEN_TO_ANY:
            return "dining:*"
        return (
            f"dining:{impact_key}:{operation.value.kind.value}:"
            f"{operation.value.place_id or ''}:{operation.value.value or ''}"
        ).casefold()
    if isinstance(operation, LodgingPreferenceOperation):
        return f"lodging:{impact_key}:{operation.value.kind.value}"
    if isinstance(operation, TransportPreferenceOperation):
        return f"transport:{impact_key}:{operation.value.kind.value}"
    if isinstance(operation, PacePreferenceOperation):
        return f"pace:{impact_key}"
    if isinstance(operation, ExperiencePreferenceOperation):
        return f"experience:{impact_key}:{operation.value.kind.value}"
    return f"{operation.target.value}:{impact_key}:{operation.operation_id}"


def _preference_item(entry: SemanticStateEntry) -> TaskBookPreferenceItem:
    operation = entry.operation
    return TaskBookPreferenceItem(
        source_operation_id=operation.operation_id,
        target=operation.target,
        persistence_scope=operation.persistence_scope,
        impact_scope=operation.impact_scope,
        summary=_preference_summary(operation),
    )


def _restaurant_item(operation: DiningPreferenceOperation) -> TaskBookRestaurantItem:
    name = operation.value.value
    intent = operation.value.restaurant_intent
    if name is None or intent is None:
        raise TaskBookError(
            TaskBookFailureCode.INVALID_SEMANTIC_STATE,
            "specific restaurant preference requires its recorded name",
        )
    return TaskBookRestaurantItem(
        source_operation_id=operation.operation_id,
        name=name,
        place_id=operation.value.place_id,
        intent=intent,
    )


def _preference_summary(operation: SemanticOperation) -> str:
    if isinstance(operation, DiningPreferenceOperation):
        if operation.value.kind is DiningPreferenceKind.OPEN_TO_ANY:
            return "饮食没有特别限制"
        return f"饮食 {operation.value.kind.value}：{operation.value.value}"
    if isinstance(operation, LodgingPreferenceOperation):
        return f"住宿 {operation.value.kind.value}：{operation.value.value}"
    if isinstance(operation, TransportPreferenceOperation):
        value = (
            operation.value.mobility_tolerance
            or operation.value.transit_taxi_level
            or operation.value.preferred_mode
            or operation.value.note
        )
        return f"交通 {operation.value.kind.value}：{value}"
    if isinstance(operation, PacePreferenceOperation):
        value = operation.value.level or operation.value.relative_adjustment
        return f"旅行节奏：{value}"
    if isinstance(operation, ExperiencePreferenceOperation):
        value = operation.value.note or operation.value.level or operation.value.relative_adjustment
        return f"体验 {operation.value.kind.value}：{value}"
    raise TaskBookError(
        TaskBookFailureCode.INVALID_SEMANTIC_STATE,
        "unsupported preference operation in task-book projection",
    )


def _require_state_version(state: SemanticTripState, expected_state_version: int) -> None:
    if state.state_version != expected_state_version:
        raise TaskBookError(
            TaskBookFailureCode.STATE_VERSION_CONFLICT,
            "task-book state version is stale",
        )
