"""Deterministic V2 semantic-state merging with provenance and version guards."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date
from enum import StrEnum
from uuid import UUID, uuid5

from pydantic import Field, model_validator

from backend.agent.readiness_state import ReadinessRuntimeState
from backend.agent.semantic_operations import (
    AttractionIntentOperation,
    DiningPreferenceKind,
    DiningPreferenceOperation,
    ExperiencePreferenceKind,
    ExperiencePreferenceOperation,
    LodgingPreferenceKind,
    LodgingPreferenceOperation,
    SemanticImpactScope,
    SemanticOperation,
    SemanticOperationBatch,
    SemanticOperationKind,
    SemanticPersistenceScope,
    SemanticTarget,
    semantic_operation_fingerprint,
    semantic_operation_key,
    stable_effect_json,
)
from backend.agent.task_book_state import SemanticTaskBook, SemanticTaskBookStatus
from backend.contracts.base import ContractModel
from backend.contracts.common import NonEmptyText, ShortText
from backend.contracts.enums import AttractionIntent, EvidenceSource

_CONFLICT_NAMESPACE = UUID("ad8698b1-9369-4f6e-8409-117f63530651")
_EXPLICIT_USER_SOURCES = {EvidenceSource.CARD, EvidenceSource.DIALOGUE}
_CONFIRMATION_TARGETS = {
    SemanticTarget.TASK_BOOK_CONFIRMATION,
    SemanticTarget.PLAN_CONFIRMATION,
}


class SemanticStatePartition(StrEnum):
    TRIP_IDENTITY = "trip_identity"
    PERSONAL_DEFAULTS = "personal_defaults"
    TRIP_PREFERENCES = "trip_preferences"
    CONSTRAINTS = "constraints"
    DISCOVERY = "discovery"
    PLANNING = "planning"


class SemanticMergeOutcome(StrEnum):
    APPLIED = "applied"
    REPLACED = "replaced"
    REMOVED = "removed"
    CONFLICT = "conflict"
    IGNORED = "ignored"


class SemanticMergeStatus(StrEnum):
    APPLIED = "applied"
    PARTIAL = "partial"
    CONFLICT = "conflict"
    NO_CHANGE = "no_change"


class SemanticMergeFailureCode(StrEnum):
    CROSS_TRIP = "cross_trip"
    STATE_VERSION_CONFLICT = "state_version_conflict"
    OPERATION_ID_REUSE = "operation_id_reuse"


class RecomputeDomain(StrEnum):
    DISCOVERY = "discovery"
    PROVIDER_FACTS = "provider_facts"
    LODGING_STRATEGY = "lodging_strategy"
    TASK_BOOK = "task_book"
    ITINERARY = "itinerary"
    COST = "cost"
    MAP = "map"


class SemanticStateEntry(ContractModel):
    state_key: NonEmptyText
    partition: SemanticStatePartition
    operation: SemanticOperation
    applied_state_version: int = Field(ge=1, strict=True)
    confirmed: bool = False


class SemanticMergeAudit(ContractModel):
    operation_id: UUID
    operation_fingerprint: NonEmptyText
    outcome: SemanticMergeOutcome
    state_key: NonEmptyText
    state_version: int = Field(ge=1, strict=True)
    replaced_operation_id: UUID | None = None


class SemanticMergeConflict(ContractModel):
    conflict_id: UUID
    trip_id: UUID
    incoming_operation_id: UUID
    existing_operation_id: UUID
    state_key: NonEmptyText
    reason: ShortText
    detected_state_version: int = Field(ge=1, strict=True)


class SemanticInvalidation(ContractModel):
    caused_by_operation_id: UUID
    state_version: int = Field(ge=1, strict=True)
    invalidated_entry_keys: list[NonEmptyText] = Field(default_factory=list)
    recompute_domains: list[RecomputeDomain] = Field(min_length=1)


class SemanticMergeImpact(ContractModel):
    operation_id: UUID
    target: SemanticTarget
    impact_scope: SemanticImpactScope
    recompute_domains: list[RecomputeDomain] = Field(min_length=1)
    global_replan_required: bool


class SemanticTripState(ContractModel):
    """Stable semantic notebook, separate from the legacy phase-oriented state."""

    trip_id: UUID
    state_version: int = Field(default=0, ge=0, strict=True)
    entries: list[SemanticStateEntry] = Field(default_factory=list)
    superseded_entries: list[SemanticStateEntry] = Field(default_factory=list)
    conflicts: list[SemanticMergeConflict] = Field(default_factory=list)
    audit_log: list[SemanticMergeAudit] = Field(default_factory=list)
    invalidations: list[SemanticInvalidation] = Field(default_factory=list)
    readiness: ReadinessRuntimeState = Field(default_factory=ReadinessRuntimeState)
    task_book: SemanticTaskBook | None = None

    @model_validator(mode="after")
    def records_are_owned_and_consistent(self) -> SemanticTripState:
        entry_keys = [entry.state_key for entry in self.entries]
        if len(set(entry_keys)) != len(entry_keys):
            raise ValueError("semantic state entry keys must be unique")
        audit_ids = [audit.operation_id for audit in self.audit_log]
        if len(set(audit_ids)) != len(audit_ids):
            raise ValueError("semantic state operation audit IDs must be unique")
        conflict_ids = [conflict.conflict_id for conflict in self.conflicts]
        if len(set(conflict_ids)) != len(conflict_ids):
            raise ValueError("semantic state conflict IDs must be unique")

        for entry in self.entries:
            if entry.operation.trip_id != self.trip_id:
                raise ValueError("semantic state entry belongs to another trip")
            if entry.operation.evidence.source_trip_id != self.trip_id:
                raise ValueError("semantic state evidence belongs to another trip")
            if entry.state_key != semantic_operation_key(entry.operation):
                raise ValueError("semantic state entry key does not match its operation")
            if entry.partition is not _partition_for(entry.operation):
                raise ValueError("semantic state entry is stored in the wrong partition")
            if entry.applied_state_version > self.state_version:
                raise ValueError("semantic state entry cannot be newer than the state")
        for entry in self.superseded_entries:
            if entry.operation.trip_id != self.trip_id:
                raise ValueError("superseded semantic entry belongs to another trip")
            if entry.operation.evidence.source_trip_id != self.trip_id:
                raise ValueError("superseded semantic evidence belongs to another trip")
            if entry.state_key != semantic_operation_key(entry.operation):
                raise ValueError("superseded semantic entry key does not match its operation")
            if entry.applied_state_version > self.state_version:
                raise ValueError("superseded semantic entry cannot be newer than the state")
        for conflict in self.conflicts:
            if conflict.trip_id != self.trip_id:
                raise ValueError("semantic conflict belongs to another trip")
            if conflict.detected_state_version > self.state_version:
                raise ValueError("semantic conflict cannot be newer than the state")
        for audit in self.audit_log:
            if audit.state_version > self.state_version:
                raise ValueError("semantic audit cannot be newer than the state")
        for invalidation in self.invalidations:
            if invalidation.state_version > self.state_version:
                raise ValueError("semantic invalidation cannot be newer than the state")
        if self.readiness.last_updated_state_version > self.state_version:
            raise ValueError("readiness state cannot be newer than semantic state")
        if self.task_book is not None:
            if self.task_book.trip_id != self.trip_id:
                raise ValueError("semantic task book belongs to another trip")
            if self.task_book.published_state_version > self.state_version:
                raise ValueError("semantic task book cannot be newer than its state")
            if self.task_book.status is SemanticTaskBookStatus.CONFIRMED and not any(
                entry.operation.operation_id == self.task_book.confirmation_operation_id
                and entry.operation.target is SemanticTarget.TASK_BOOK_CONFIRMATION
                for entry in self.entries
            ):
                raise ValueError("confirmed task book requires its active confirmation entry")
        return self


class SemanticMergeResult(ContractModel):
    state: SemanticTripState
    previous_state_version: int = Field(ge=0, strict=True)
    status: SemanticMergeStatus
    applied_operation_ids: list[UUID] = Field(default_factory=list)
    replaced_operation_ids: list[UUID] = Field(default_factory=list)
    removed_operation_ids: list[UUID] = Field(default_factory=list)
    ignored_operation_ids: list[UUID] = Field(default_factory=list)
    duplicate_operation_ids: list[UUID] = Field(default_factory=list)
    conflicts: list[SemanticMergeConflict] = Field(default_factory=list)
    impacts: list[SemanticMergeImpact] = Field(default_factory=list)
    invalidations: list[SemanticInvalidation] = Field(default_factory=list)


class SemanticMergeError(ValueError):
    def __init__(self, code: SemanticMergeFailureCode, message: str) -> None:
        super().__init__(message)
        self.code = code


def initial_semantic_state(trip_id: UUID) -> SemanticTripState:
    return SemanticTripState(trip_id=trip_id)


def restore_semantic_state(payload: object) -> SemanticTripState:
    """Validate a persisted snapshot without reapplying input-time future-date rules."""

    return SemanticTripState.model_validate(
        payload,
        context={"restore_historical_semantic_state": True},
    )


def merge_state(
    state: SemanticTripState,
    batch: SemanticOperationBatch,
    *,
    expected_state_version: int,
    business_date: date,
    allow_personal_defaults_write: bool = False,
) -> SemanticMergeResult:
    """Merge one validated semantic input without mutating the supplied snapshot."""

    # Treat even an already-constructed Pydantic instance as untrusted input. Frozen
    # models prevent ordinary mutation, while this round-trip also rejects instances
    # forged through unchecked copy/update or low-level attribute access.
    batch = SemanticOperationBatch.model_validate(
        batch.model_dump(mode="json"),
        context={
            "today": business_date,
            "allow_personal_defaults_write": allow_personal_defaults_write,
        },
    )

    if batch.trip_id != state.trip_id:
        raise SemanticMergeError(
            SemanticMergeFailureCode.CROSS_TRIP,
            "semantic operation batch belongs to another trip",
        )

    audits_by_id = {audit.operation_id: audit for audit in state.audit_log}
    duplicate_ids: list[UUID] = []
    new_operations: list[SemanticOperation] = []
    for operation in batch.operations:
        previous = audits_by_id.get(operation.operation_id)
        if previous is None:
            new_operations.append(operation)
            continue
        if previous.operation_fingerprint != semantic_operation_fingerprint(operation):
            raise SemanticMergeError(
                SemanticMergeFailureCode.OPERATION_ID_REUSE,
                "semantic operation_id was reused with different content",
            )
        duplicate_ids.append(operation.operation_id)

    if not new_operations:
        return SemanticMergeResult(
            state=state.model_copy(deep=True),
            previous_state_version=state.state_version,
            status=SemanticMergeStatus.NO_CHANGE,
            duplicate_operation_ids=duplicate_ids,
        )
    if expected_state_version != state.state_version:
        raise SemanticMergeError(
            SemanticMergeFailureCode.STATE_VERSION_CONFLICT,
            "semantic state version is stale",
        )

    next_version = state.state_version + 1
    entries = {entry.state_key: entry.model_copy(deep=True) for entry in state.entries}
    audits = [audit.model_copy(deep=True) for audit in state.audit_log]
    all_conflicts = [conflict.model_copy(deep=True) for conflict in state.conflicts]
    all_invalidations = [item.model_copy(deep=True) for item in state.invalidations]
    superseded_entries = [item.model_copy(deep=True) for item in state.superseded_entries]
    result_conflicts: list[SemanticMergeConflict] = []
    result_invalidations: list[SemanticInvalidation] = []
    impacts: list[SemanticMergeImpact] = []
    applied_ids: list[UUID] = []
    replaced_ids: list[UUID] = []
    removed_ids: list[UUID] = []
    ignored_ids: list[UUID] = []

    for operation in new_operations:
        key = semantic_operation_key(operation)
        existing = entries.get(key)
        dining_conflict = _open_dining_conflict(entries.values(), operation)
        replaced_id: UUID | None
        if dining_conflict is not None:
            existing = dining_conflict
            outcome, replaced_id = (
                SemanticMergeOutcome.CONFLICT,
                dining_conflict.operation.operation_id,
            )
        else:
            outcome, replaced_id = _merge_operation(entries, operation, next_version)

        if outcome is SemanticMergeOutcome.CONFLICT:
            assert existing is not None
            conflict = _conflict_for(
                state.trip_id,
                operation,
                existing,
                key,
                next_version,
                reason=(
                    "open-to-any dining conflicts with concrete dining requirements"
                    if dining_conflict is not None
                    else "same-priority explicit values require clarification"
                ),
            )
            result_conflicts.append(conflict)
            all_conflicts.append(conflict)
        elif outcome is SemanticMergeOutcome.APPLIED:
            all_conflicts = [conflict for conflict in all_conflicts if conflict.state_key != key]
            applied_ids.append(operation.operation_id)
            impact = _impact_for(operation)
            impacts.append(impact)
            invalidation = _invalidate_after_change(
                entries,
                operation,
                next_version,
                impact.recompute_domains,
            )
            if invalidation is not None:
                result_invalidations.append(invalidation)
                all_invalidations.append(invalidation)
        elif outcome is SemanticMergeOutcome.REPLACED:
            assert existing is not None
            superseded_entries.append(existing.model_copy(deep=True))
            all_conflicts = [conflict for conflict in all_conflicts if conflict.state_key != key]
            replaced_ids.append(operation.operation_id)
            impact = _impact_for(operation)
            impacts.append(impact)
            invalidation = _invalidate_after_change(
                entries,
                operation,
                next_version,
                impact.recompute_domains,
            )
            if invalidation is not None:
                result_invalidations.append(invalidation)
                all_invalidations.append(invalidation)
        elif outcome is SemanticMergeOutcome.REMOVED:
            assert existing is not None
            superseded_entries.append(existing.model_copy(deep=True))
            all_conflicts = [conflict for conflict in all_conflicts if conflict.state_key != key]
            removed_ids.append(operation.operation_id)
            impact = _impact_for(operation)
            impacts.append(impact)
            invalidation = _invalidate_after_change(
                entries,
                operation,
                next_version,
                impact.recompute_domains,
            )
            if invalidation is not None:
                result_invalidations.append(invalidation)
                all_invalidations.append(invalidation)
        else:
            ignored_ids.append(operation.operation_id)

        audits.append(
            SemanticMergeAudit(
                operation_id=operation.operation_id,
                operation_fingerprint=semantic_operation_fingerprint(operation),
                outcome=outcome,
                state_key=key,
                state_version=next_version,
                replaced_operation_id=replaced_id,
            )
        )

        if operation.target in _CONFIRMATION_TARGETS and (
            outcome in {SemanticMergeOutcome.APPLIED, SemanticMergeOutcome.REPLACED}
            and operation.operation is SemanticOperationKind.CONFIRM
        ):
            entries = {
                entry_key: entry.model_copy(
                    update={"confirmed": True, "applied_state_version": next_version}
                )
                for entry_key, entry in entries.items()
                if entry.operation.persistence_scope is SemanticPersistenceScope.CURRENT_TRIP
            } | {
                entry_key: entry
                for entry_key, entry in entries.items()
                if entry.operation.persistence_scope is SemanticPersistenceScope.PERSONAL_DEFAULTS
            }

    task_book = state.task_book.model_copy(deep=True) if state.task_book is not None else None
    if any(
        RecomputeDomain.TASK_BOOK in invalidation.recompute_domains
        for invalidation in result_invalidations
    ):
        task_book = None

    merged_state = SemanticTripState(
        trip_id=state.trip_id,
        state_version=next_version,
        entries=sorted(entries.values(), key=lambda item: item.state_key),
        superseded_entries=superseded_entries,
        conflicts=all_conflicts,
        audit_log=audits,
        invalidations=all_invalidations,
        readiness=state.readiness.model_copy(deep=True),
        task_book=task_book,
    )
    status = _merge_status(
        changed=bool(applied_ids or replaced_ids or removed_ids),
        conflicted=bool(result_conflicts),
        ignored=bool(ignored_ids),
    )
    return SemanticMergeResult(
        state=merged_state,
        previous_state_version=state.state_version,
        status=status,
        applied_operation_ids=applied_ids,
        replaced_operation_ids=replaced_ids,
        removed_operation_ids=removed_ids,
        ignored_operation_ids=ignored_ids,
        duplicate_operation_ids=duplicate_ids,
        conflicts=result_conflicts,
        impacts=impacts,
        invalidations=result_invalidations,
    )


def _merge_operation(
    entries: dict[str, SemanticStateEntry],
    operation: SemanticOperation,
    state_version: int,
) -> tuple[SemanticMergeOutcome, UUID | None]:
    key = semantic_operation_key(operation)
    existing = entries.get(key)
    if existing is None:
        if operation.operation is SemanticOperationKind.DELETE:
            return SemanticMergeOutcome.IGNORED, None
        entries[key] = _entry_for(operation, state_version)
        return SemanticMergeOutcome.APPLIED, None

    if stable_effect_json(existing.operation) == stable_effect_json(operation):
        return SemanticMergeOutcome.IGNORED, existing.operation.operation_id

    incoming_source = operation.evidence.source
    if existing.confirmed and incoming_source not in _EXPLICIT_USER_SOURCES:
        return SemanticMergeOutcome.IGNORED, existing.operation.operation_id

    incoming_priority = _information_priority(operation)
    existing_priority = _information_priority(existing.operation)
    if incoming_priority < existing_priority:
        return SemanticMergeOutcome.IGNORED, existing.operation.operation_id

    if operation.operation is SemanticOperationKind.DELETE:
        del entries[key]
        return SemanticMergeOutcome.REMOVED, existing.operation.operation_id

    if (
        operation.operation
        in {
            SemanticOperationKind.OVERRIDE,
            SemanticOperationKind.NEGATE,
            SemanticOperationKind.CONFIRM,
            SemanticOperationKind.REOPEN,
        }
        or incoming_priority > existing_priority
    ):
        entries[key] = _entry_for(operation, state_version)
        return SemanticMergeOutcome.REPLACED, existing.operation.operation_id

    return SemanticMergeOutcome.CONFLICT, existing.operation.operation_id


def _open_dining_conflict(
    entries: Iterable[SemanticStateEntry],
    incoming: SemanticOperation,
) -> SemanticStateEntry | None:
    if not isinstance(incoming, DiningPreferenceOperation):
        return None
    incoming_is_open = incoming.value.kind is DiningPreferenceKind.OPEN_TO_ANY
    for entry in entries:
        current = entry.operation
        if not isinstance(current, DiningPreferenceOperation):
            continue
        if current.persistence_scope is not incoming.persistence_scope:
            continue
        if current.impact_scope != incoming.impact_scope:
            continue
        current_is_open = current.value.kind is DiningPreferenceKind.OPEN_TO_ANY
        if current_is_open != incoming_is_open:
            return entry
    return None


def _entry_for(operation: SemanticOperation, state_version: int) -> SemanticStateEntry:
    return SemanticStateEntry(
        state_key=semantic_operation_key(operation),
        partition=_partition_for(operation),
        operation=operation,
        applied_state_version=state_version,
    )


def _partition_for(operation: SemanticOperation) -> SemanticStatePartition:
    if operation.persistence_scope is SemanticPersistenceScope.PERSONAL_DEFAULTS:
        return SemanticStatePartition.PERSONAL_DEFAULTS
    prefix = operation.target.value.split(".", maxsplit=1)[0]
    return SemanticStatePartition(prefix)


def _information_priority(operation: SemanticOperation) -> int:
    source_priority = {
        EvidenceSource.SYSTEM_DEFAULT: 1_000,
        EvidenceSource.AGENT_INFERENCE: 2_000,
        EvidenceSource.PROVIDER: 3_000,
        EvidenceSource.COLD_START: 5_000,
        EvidenceSource.CARD: 6_000,
        EvidenceSource.DIALOGUE: 7_000,
    }[operation.evidence.source]
    if operation.target is SemanticTarget.SPECIAL_CONSTRAINTS:
        source_priority += 250
    if operation.target in _CONFIRMATION_TARGETS:
        source_priority += 200
    if operation.evidence.source in _EXPLICIT_USER_SOURCES and operation.operation in {
        SemanticOperationKind.OVERRIDE,
        SemanticOperationKind.NEGATE,
        SemanticOperationKind.DELETE,
    }:
        # A deliberate edit is newer and stronger than the channel that originally
        # captured the old value. This keeps click and text edits behaviorally equal.
        source_priority += 2_000
    if isinstance(operation, AttractionIntentOperation) and operation.value.intent in {
        AttractionIntent.MUST,
        AttractionIntent.AVOID,
    }:
        source_priority += 100
    if isinstance(operation, DiningPreferenceOperation) and operation.value.kind in {
        DiningPreferenceKind.SPECIFIC_RESTAURANT,
        DiningPreferenceKind.AVOIDANCE,
    }:
        source_priority += 100
    if isinstance(operation, LodgingPreferenceOperation) and operation.value.kind in {
        LodgingPreferenceKind.SPECIFIC_HOTEL,
        LodgingPreferenceKind.AREA,
        LodgingPreferenceKind.TRANSIT_NODE,
    }:
        source_priority += 100
    return source_priority


def _conflict_for(
    trip_id: UUID,
    incoming: SemanticOperation,
    existing: SemanticStateEntry,
    state_key: str,
    state_version: int,
    *,
    reason: str,
) -> SemanticMergeConflict:
    fingerprint = ":".join(
        sorted((str(incoming.operation_id), str(existing.operation.operation_id)))
    )
    return SemanticMergeConflict(
        conflict_id=uuid5(_CONFLICT_NAMESPACE, f"{trip_id}:{state_key}:{fingerprint}"),
        trip_id=trip_id,
        incoming_operation_id=incoming.operation_id,
        existing_operation_id=existing.operation.operation_id,
        state_key=state_key,
        reason=reason,
        detected_state_version=state_version,
    )


def _impact_for(operation: SemanticOperation) -> SemanticMergeImpact:
    domains = {
        SemanticTarget.DESTINATION: (
            RecomputeDomain.DISCOVERY,
            RecomputeDomain.PROVIDER_FACTS,
            RecomputeDomain.LODGING_STRATEGY,
            RecomputeDomain.TASK_BOOK,
            RecomputeDomain.ITINERARY,
            RecomputeDomain.COST,
            RecomputeDomain.MAP,
        ),
        SemanticTarget.DATE_RANGE: (
            RecomputeDomain.PROVIDER_FACTS,
            RecomputeDomain.LODGING_STRATEGY,
            RecomputeDomain.TASK_BOOK,
            RecomputeDomain.ITINERARY,
            RecomputeDomain.COST,
        ),
        SemanticTarget.ATTRACTION_INTENTS: (
            RecomputeDomain.DISCOVERY,
            RecomputeDomain.LODGING_STRATEGY,
            RecomputeDomain.TASK_BOOK,
            RecomputeDomain.ITINERARY,
            RecomputeDomain.COST,
            RecomputeDomain.MAP,
        ),
        SemanticTarget.DINING_PREFERENCES: (
            RecomputeDomain.DISCOVERY,
            RecomputeDomain.LODGING_STRATEGY,
            RecomputeDomain.TASK_BOOK,
            RecomputeDomain.ITINERARY,
            RecomputeDomain.COST,
            RecomputeDomain.MAP,
        ),
        SemanticTarget.LODGING_PREFERENCES: (
            RecomputeDomain.PROVIDER_FACTS,
            RecomputeDomain.LODGING_STRATEGY,
            RecomputeDomain.TASK_BOOK,
            RecomputeDomain.ITINERARY,
            RecomputeDomain.COST,
            RecomputeDomain.MAP,
        ),
        SemanticTarget.TRANSPORT_PREFERENCES: (
            RecomputeDomain.PROVIDER_FACTS,
            RecomputeDomain.LODGING_STRATEGY,
            RecomputeDomain.TASK_BOOK,
            RecomputeDomain.ITINERARY,
            RecomputeDomain.COST,
        ),
        SemanticTarget.PACE_PREFERENCES: (
            RecomputeDomain.TASK_BOOK,
            RecomputeDomain.ITINERARY,
            RecomputeDomain.COST,
        ),
        SemanticTarget.EXPERIENCE_PREFERENCES: (
            RecomputeDomain.DISCOVERY,
            RecomputeDomain.TASK_BOOK,
            RecomputeDomain.ITINERARY,
        ),
        SemanticTarget.SPECIAL_CONSTRAINTS: (
            RecomputeDomain.LODGING_STRATEGY,
            RecomputeDomain.TASK_BOOK,
            RecomputeDomain.ITINERARY,
        ),
        SemanticTarget.TASK_BOOK_CONFIRMATION: (RecomputeDomain.TASK_BOOK,),
        SemanticTarget.PLAN_CONFIRMATION: (RecomputeDomain.ITINERARY,),
    }[operation.target]
    return SemanticMergeImpact(
        operation_id=operation.operation_id,
        target=operation.target,
        impact_scope=operation.impact_scope,
        recompute_domains=list(domains),
        global_replan_required=(
            operation.target in {SemanticTarget.DESTINATION, SemanticTarget.DATE_RANGE}
            or (
                operation.impact_scope.kind.value == "whole_trip"
                and operation.target is SemanticTarget.PACE_PREFERENCES
            )
        ),
    )


def _invalidate_after_change(
    entries: dict[str, SemanticStateEntry],
    operation: SemanticOperation,
    state_version: int,
    domains: list[RecomputeDomain],
) -> SemanticInvalidation | None:
    if operation.target in _CONFIRMATION_TARGETS:
        return None

    invalidated_keys = set(_confirmation_keys(entries))
    if operation.target is SemanticTarget.DESTINATION:
        invalidated_keys.update(
            key
            for key, entry in entries.items()
            if entry.applied_state_version < state_version
            and _is_destination_bound(entry.operation)
        )
    for key in invalidated_keys:
        entries.pop(key, None)
    if not invalidated_keys and not domains:
        return None
    return SemanticInvalidation(
        caused_by_operation_id=operation.operation_id,
        state_version=state_version,
        invalidated_entry_keys=sorted(invalidated_keys),
        recompute_domains=domains,
    )


def _confirmation_keys(entries: dict[str, SemanticStateEntry]) -> Iterable[str]:
    return (
        key for key, entry in entries.items() if entry.operation.target in _CONFIRMATION_TARGETS
    )


def _is_destination_bound(operation: SemanticOperation) -> bool:
    if operation.target is SemanticTarget.ATTRACTION_INTENTS:
        return True
    if isinstance(operation, ExperiencePreferenceOperation):
        return operation.value.kind is ExperiencePreferenceKind.CITY_THEME
    if isinstance(operation, DiningPreferenceOperation):
        return operation.value.kind is DiningPreferenceKind.SPECIFIC_RESTAURANT
    if isinstance(operation, LodgingPreferenceOperation):
        return operation.value.kind in {
            LodgingPreferenceKind.AREA,
            LodgingPreferenceKind.TRANSIT_NODE,
            LodgingPreferenceKind.SPECIFIC_HOTEL,
        }
    return False


def _merge_status(*, changed: bool, conflicted: bool, ignored: bool) -> SemanticMergeStatus:
    if conflicted and changed:
        return SemanticMergeStatus.PARTIAL
    if conflicted:
        return SemanticMergeStatus.CONFLICT
    if changed:
        return SemanticMergeStatus.APPLIED
    if ignored:
        return SemanticMergeStatus.NO_CHANGE
    return SemanticMergeStatus.NO_CHANGE
