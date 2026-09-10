"""Deterministic V4 semantic-operation merge into an uncommitted working aggregate."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from uuid import UUID, uuid5

from backend.contracts.v4.enums import (
    ConfidenceLevel,
    DiscoverySection,
    PendingInteractionKind,
    TaskBookStatus,
)
from backend.contracts.v4.semantic_operations import (
    AddConditionalRequirementOperation,
    ConfirmFinalSupplementOperation,
    ConfirmTaskBookOperation,
    ExcludeConcreteEntityOperation,
    ExcludePreferenceDirectionOperation,
    ResolveConflictOperation,
    RevokePriorIntentOperation,
    SelectConcreteEntityOperation,
    SelectPreferenceDirectionOperation,
    SemanticDomainV4,
    SemanticOperationProposal,
    SemanticTargetV4,
    SetDelegationScopeOperation,
    SetExistingBookingOperation,
    SetLodgingClassPreferenceOperation,
    SetNoPreferenceOperation,
    SetNotApplicableOperation,
    SetTripBasicsOperation,
)
from backend.contracts.v4.state import (
    ConcreteIntentState,
    ConfirmedTaskBookRef,
    DiscoveryRuntimeState,
    PreferenceDirectionState,
    TripSemanticState,
)
from backend.contracts.v4.task_book import BookingReference, DelegatedScope, MoneyRange

_OPERATION_NAMESPACE = UUID("f35e5a94-2a65-4b88-b8d3-0ea39d7eb8d4")


class V4SemanticMergeError(ValueError):
    """One or more model-proposed operations cannot enter the working aggregate."""


@dataclass(frozen=True, slots=True)
class AcceptedV4Operation:
    operation_id: UUID
    proposal: SemanticOperationProposal


@dataclass(frozen=True, slots=True)
class WorkingAggregateMerge:
    semantic_state: TripSemanticState
    runtime_state: DiscoveryRuntimeState
    accepted_operations: tuple[AcceptedV4Operation, ...]
    invalidated_interaction_ids: tuple[str, ...]
    affected_sections: frozenset[DiscoverySection]


def stable_operation_id(turn_id: UUID, local_operation_key: str) -> UUID:
    return uuid5(_OPERATION_NAMESPACE, f"{turn_id}:{local_operation_key}")


def task_book_confirmation_fingerprint(
    task_book_id: str,
    task_book_version: int,
    state_version: int,
) -> str:
    """Bind one confirmation interaction to the current semantic version."""

    return hashlib.sha256(
        json.dumps(
            {
                "task_book_id": task_book_id,
                "version": task_book_version,
                "state_version": state_version,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def merge_v4_operations(
    semantic_state: TripSemanticState,
    runtime_state: DiscoveryRuntimeState,
    proposals: list[SemanticOperationProposal],
    *,
    turn_id: UUID,
    allowed_source_refs: set[str],
    known_entity_refs: set[str],
    advance_version: bool = True,
) -> WorkingAggregateMerge:
    """Validate and apply operations without mutating or persisting the supplied State."""

    if semantic_state.trip_id != runtime_state.trip_id:
        raise V4SemanticMergeError("semantic and runtime states belong to different trips")
    if semantic_state.state_version != runtime_state.state_version:
        raise V4SemanticMergeError("semantic and runtime states do not share one version")
    local_keys = [item.root.local_operation_key for item in proposals]
    if len(set(local_keys)) != len(local_keys):
        raise V4SemanticMergeError("local operation keys must be unique within one decision")

    next_version = semantic_state.state_version + (1 if advance_version else 0)
    working = semantic_state.model_copy(
        update={"state_version": next_version},
        deep=True,
    )
    accepted: list[AcceptedV4Operation] = []
    affected: set[DiscoverySection] = set()
    for wrapped in proposals:
        proposal = wrapped.root
        if not set(proposal.source_refs).issubset(allowed_source_refs):
            raise V4SemanticMergeError("operation references evidence outside this turn")
        _validate_target_domain(proposal)
        if isinstance(proposal, ConfirmFinalSupplementOperation):
            pending = runtime_state.pending_interaction
            if (
                proposal.target is not SemanticTargetV4.FINAL_SUPPLEMENT
                or pending is None
                or pending.status.value != "active"
                or pending.kind is not PendingInteractionKind.FREE_TEXT_QUESTION
                or pending.section is not DiscoverySection.FINAL_SUPPLEMENT
                or pending.target_ids != ["final_supplement"]
                or pending.based_on_state_version != runtime_state.state_version
            ):
                raise V4SemanticMergeError(
                    "final supplement confirmation requires current active interaction"
                )
        if isinstance(proposal, ConfirmTaskBookOperation):
            candidate = runtime_state.task_book_candidate
            if (
                candidate is None
                or candidate.status is not TaskBookStatus.AWAITING_CONFIRMATION
                or candidate.task_book_id != proposal.task_book_id
                or candidate.value.version != proposal.task_book_version
                or candidate.based_on_state_version != proposal.based_on_state_version
                or candidate.based_on_state_version != semantic_state.state_version
            ):
                raise V4SemanticMergeError("task book confirmation is stale or mismatched")
        if (
            isinstance(
                proposal,
                (SelectConcreteEntityOperation, ExcludeConcreteEntityOperation),
            )
            and proposal.canonical_entity_id not in known_entity_refs
        ):
            raise V4SemanticMergeError("concrete entity was not resolved or server-signed")
        if (
            isinstance(proposal, SetExistingBookingOperation)
            and proposal.canonical_entity_id is not None
            and proposal.canonical_entity_id not in known_entity_refs
        ):
            raise V4SemanticMergeError("booking entity was not resolved or server-signed")
        if (
            isinstance(proposal, SetTripBasicsOperation)
            and proposal.destination_canonical_id is not None
            and proposal.destination_canonical_id not in known_entity_refs
        ):
            raise V4SemanticMergeError("destination was not resolved or server-signed")
        if proposal.confidence is ConfidenceLevel.LOW and isinstance(
            proposal,
            (
                SelectConcreteEntityOperation,
                ExcludeConcreteEntityOperation,
                SetExistingBookingOperation,
                AddConditionalRequirementOperation,
            ),
        ):
            raise V4SemanticMergeError("low-confidence material operation requires clarification")

        operation_id = stable_operation_id(turn_id, proposal.local_operation_key)
        operation_ref = str(operation_id)
        working = _apply_operation(working, proposal, operation_ref)
        accepted.append(AcceptedV4Operation(operation_id, wrapped))
        affected.update(_affected_sections(proposal))

    invalidated_interaction_ids: list[str] = []
    pending = runtime_state.pending_interaction
    confirming_task_book = any(
        isinstance(item.proposal.root, ConfirmTaskBookOperation) for item in accepted
    )
    semantic_modification = bool(accepted) and not confirming_task_book
    if semantic_modification and working.confirmed_task_book_ref is not None:
        working = working.model_copy(update={"confirmed_task_book_ref": None}, deep=True)
    task_book_candidate = runtime_state.task_book_candidate
    invalidates_task_book = semantic_modification and task_book_candidate is not None
    if pending is not None and (
        pending.section in affected
        or confirming_task_book
        or (invalidates_task_book and pending.kind is PendingInteractionKind.CONFIRMATION)
    ):
        invalidated_interaction_ids.append(pending.interaction_id)
        pending = None
    if confirming_task_book and task_book_candidate is not None:
        confirmation = next(
            item.proposal.root
            for item in accepted
            if isinstance(item.proposal.root, ConfirmTaskBookOperation)
        )
        assert isinstance(confirmation, ConfirmTaskBookOperation)
        task_book_candidate = task_book_candidate.model_copy(
            update={
                "status": TaskBookStatus.CONFIRMED,
                "value": task_book_candidate.value.model_copy(
                    update={
                        "status": TaskBookStatus.CONFIRMED,
                        "confirmed_at": confirmation.confirmed_at,
                    },
                    deep=True,
                ),
            },
            deep=True,
        )
    elif semantic_modification and task_book_candidate is not None:
        task_book_candidate = task_book_candidate.model_copy(
            update={
                "status": TaskBookStatus.SUPERSEDED,
                "value": task_book_candidate.value.model_copy(
                    update={
                        "status": TaskBookStatus.SUPERSEDED,
                        "confirmed_at": None,
                    },
                    deep=True,
                ),
            },
            deep=True,
        )
    working_runtime = runtime_state.model_copy(
        update={
            "state_version": next_version,
            "pending_interaction": pending,
            "task_book_candidate": task_book_candidate,
        },
        deep=True,
    )
    return WorkingAggregateMerge(
        semantic_state=working,
        runtime_state=working_runtime,
        accepted_operations=tuple(accepted),
        invalidated_interaction_ids=tuple(invalidated_interaction_ids),
        affected_sections=frozenset(affected),
    )


def advance_without_operations(
    semantic_state: TripSemanticState,
    runtime_state: DiscoveryRuntimeState,
) -> WorkingAggregateMerge:
    """Advance the shared conversation version for a valid zero-write user turn."""

    if semantic_state.trip_id != runtime_state.trip_id:
        raise V4SemanticMergeError("semantic and runtime states belong to different trips")
    if semantic_state.state_version != runtime_state.state_version:
        raise V4SemanticMergeError("semantic and runtime states do not share one version")
    next_version = semantic_state.state_version + 1
    pending = runtime_state.pending_interaction
    task_book_candidate = runtime_state.task_book_candidate
    if (
        task_book_candidate is not None
        and task_book_candidate.status is TaskBookStatus.AWAITING_CONFIRMATION
    ):
        task_book_candidate = task_book_candidate.model_copy(
            update={
                "based_on_state_version": next_version,
                "value": task_book_candidate.value.model_copy(
                    update={"based_on_state_version": next_version},
                    deep=True,
                ),
            },
            deep=True,
        )
        if (
            pending is not None
            and pending.kind is PendingInteractionKind.CONFIRMATION
            and task_book_candidate.task_book_id in pending.target_ids
        ):
            pending = pending.model_copy(
                update={
                    "based_on_state_version": next_version,
                    "dependency_fingerprint": task_book_confirmation_fingerprint(
                        task_book_candidate.task_book_id,
                        task_book_candidate.value.version,
                        next_version,
                    ),
                },
                deep=True,
            )
    return WorkingAggregateMerge(
        semantic_state=semantic_state.model_copy(update={"state_version": next_version}, deep=True),
        runtime_state=runtime_state.model_copy(
            update={
                "state_version": next_version,
                "pending_interaction": pending,
                "task_book_candidate": task_book_candidate,
            },
            deep=True,
        ),
        accepted_operations=(),
        invalidated_interaction_ids=(),
        affected_sections=frozenset(),
    )


def _apply_operation(
    state: TripSemanticState,
    proposal: object,
    operation_ref: str,
) -> TripSemanticState:
    if isinstance(proposal, SetTripBasicsOperation):
        date_update: dict[str, object] = {}
        if proposal.start_date is not None and proposal.end_date is not None:
            date_update = {
                "start_date": proposal.start_date,
                "end_date": proposal.end_date,
                "duration_days": (proposal.end_date - proposal.start_date).days + 1,
                "date_source_operation_refs": [operation_ref],
            }
        elif proposal.duration_days is not None:
            basics = state.trip_basics
            if basics.start_date is not None and basics.end_date is not None:
                current_duration = (basics.end_date - basics.start_date).days + 1
                if current_duration != proposal.duration_days:
                    raise V4SemanticMergeError(
                        "duration_days conflicts with the existing date range"
                    )
            date_update = {
                "duration_days": proposal.duration_days,
                "date_source_operation_refs": [operation_ref],
            }
        basics = state.trip_basics.model_copy(
            update={
                **(
                    {
                        "destination_name": proposal.destination_name,
                        "destination_canonical_id": proposal.destination_canonical_id,
                        "destination_source_operation_refs": [operation_ref],
                    }
                    if proposal.destination_name is not None
                    else {}
                ),
                **date_update,
                **(
                    {
                        "travelers": list(proposal.travelers),
                        "traveler_source_operation_refs": [operation_ref],
                    }
                    if proposal.travelers is not None
                    else {}
                ),
                **(
                    {
                        "trip_goals": list(proposal.trip_goals),
                        "trip_goal_source_operation_refs": [operation_ref],
                    }
                    if proposal.trip_goals is not None
                    else {}
                ),
            },
            deep=True,
        )
        return state.model_copy(update={"trip_basics": basics}, deep=True)
    if isinstance(proposal, SetNoPreferenceOperation):
        return state
    if isinstance(proposal, SetNotApplicableOperation):
        if proposal.target in {
            SemanticTargetV4.LODGING_AREA,
            SemanticTargetV4.LODGING_CLASS,
            SemanticTargetV4.LODGING_BOOKING,
        }:
            lodging = state.lodging.model_copy(
                update={
                    "not_applicable": True,
                    "area_preferences": [],
                    "hotel_quality_tier": None,
                    "property_type_preferences": [],
                    "nightly_budget": None,
                    "facility_requirements": [],
                    "class_preference_source_operation_refs": [],
                    "existing_bookings": [],
                    "delegation": None,
                },
                deep=True,
            )
            return state.model_copy(update={"lodging": lodging}, deep=True)
        return state
    if isinstance(proposal, SetDelegationScopeOperation):
        existing = next(
            (item for item in state.delegations if item.domain == proposal.domain.value),
            None,
        )
        delegated = DelegatedScope(
            domain=proposal.domain.value,
            delegated_targets=list(
                dict.fromkeys(
                    [
                        *(existing.delegated_targets if existing is not None else []),
                        *proposal.delegated_targets,
                    ]
                )
            ),
            boundary_refs=list(
                dict.fromkeys(
                    [
                        *(existing.boundary_refs if existing is not None else []),
                        *proposal.boundary_refs,
                    ]
                )
            ),
            source_operation_refs=list(
                dict.fromkeys(
                    [
                        *(existing.source_operation_refs if existing is not None else []),
                        operation_ref,
                    ]
                )
            ),
        )
        delegations = [item for item in state.delegations if item.domain != delegated.domain]
        delegations.append(delegated)
        updates: dict[str, object] = {"delegations": delegations}
        if proposal.domain is SemanticDomainV4.ATTRACTION:
            updates["attractions"] = state.attractions.model_copy(
                update={"delegation": delegated}, deep=True
            )
        elif proposal.domain is SemanticDomainV4.DINING:
            updates["dining"] = state.dining.model_copy(update={"delegation": delegated}, deep=True)
        elif proposal.domain is SemanticDomainV4.LODGING:
            updates["lodging"] = state.lodging.model_copy(
                update={"delegation": delegated, "not_applicable": False}, deep=True
            )
        return state.model_copy(update=updates, deep=True)
    if isinstance(proposal, SetExistingBookingOperation):
        existing_booking = next(
            (
                item
                for item in state.existing_bookings
                if item.booking_kind == proposal.booking_kind
                and item.user_description == proposal.user_description
            ),
            None,
        )
        booking = BookingReference(
            booking_id=f"booking:{operation_ref}",
            booking_kind=proposal.booking_kind,
            user_description=proposal.user_description,
            canonical_entity_id=(
                proposal.canonical_entity_id
                if proposal.canonical_entity_id is not None
                else (
                    existing_booking.canonical_entity_id if existing_booking is not None else None
                )
            ),
            start_date=(
                proposal.start_date
                if proposal.start_date is not None
                else (existing_booking.start_date if existing_booking is not None else None)
            ),
            end_date=(
                proposal.end_date
                if proposal.end_date is not None
                else existing_booking.end_date
                if existing_booking is not None
                else None
            ),
            source_operation_refs=list(
                dict.fromkeys(
                    [
                        *(
                            existing_booking.source_operation_refs
                            if existing_booking is not None
                            else []
                        ),
                        operation_ref,
                    ]
                )
            ),
        )
        bookings = [
            item
            for item in state.existing_bookings
            if not (
                item.booking_kind == booking.booking_kind
                and item.user_description == booking.user_description
            )
        ]
        bookings.append(booking)
        updates = {"existing_bookings": bookings}
        if proposal.domain is SemanticDomainV4.LODGING:
            lodging_bookings = [
                item
                for item in state.lodging.existing_bookings
                if item.user_description != booking.user_description
            ]
            lodging_bookings.append(booking)
            updates["lodging"] = state.lodging.model_copy(
                update={"existing_bookings": lodging_bookings, "not_applicable": False},
                deep=True,
            )
        return state.model_copy(update=updates, deep=True)
    if isinstance(
        proposal,
        (SelectPreferenceDirectionOperation, ExcludePreferenceDirectionOperation),
    ):
        selected = isinstance(proposal, SelectPreferenceDirectionOperation)
        direction = PreferenceDirectionState(
            direction_id=proposal.direction_id,
            label=proposal.label,
            description=proposal.description,
            tags=proposal.tags,
            search_query=proposal.search_query,
            selected=selected,
            source_operation_refs=[operation_ref],
        )
        if proposal.domain is SemanticDomainV4.ATTRACTION:
            values = _replace_direction(state.attractions.preference_directions, direction)
            return state.model_copy(
                update={
                    "attractions": state.attractions.model_copy(
                        update={"preference_directions": values}, deep=True
                    )
                },
                deep=True,
            )
        if proposal.domain is SemanticDomainV4.DINING:
            values = _replace_direction(state.dining.preference_directions, direction)
            return state.model_copy(
                update={
                    "dining": state.dining.model_copy(
                        update={"preference_directions": values}, deep=True
                    )
                },
                deep=True,
            )
        if proposal.domain is SemanticDomainV4.TRANSPORT:
            # Qwen already resolved the semantic domain. Preserve that grounded
            # label instead of silently accepting an operation with no effect.
            pace_values = list(state.transport_and_pace.pace_preferences)
            pace_values = (
                _append_unique(pace_values, proposal.label)
                if selected
                else [value for value in pace_values if value != proposal.label]
            )
            return state.model_copy(
                update={
                    "transport_and_pace": state.transport_and_pace.model_copy(
                        update={"pace_preferences": pace_values}, deep=True
                    )
                },
                deep=True,
            )
        if proposal.target is SemanticTargetV4.LODGING_AREA:
            values = _replace_direction(state.lodging.area_preferences, direction)
            return state.model_copy(
                update={
                    "lodging": state.lodging.model_copy(
                        update={"area_preferences": values, "not_applicable": False},
                        deep=True,
                    )
                },
                deep=True,
            )
        if proposal.target is SemanticTargetV4.LODGING_CLASS and selected:
            normalized = proposal.direction_id.casefold()
            tier = normalized if normalized in {"economy", "comfort", "upscale", "luxury"} else None
            property_types = list(state.lodging.property_type_preferences)
            if tier is None and proposal.label not in property_types:
                property_types.append(proposal.label)
            return state.model_copy(
                update={
                    "lodging": state.lodging.model_copy(
                        update={
                            "hotel_quality_tier": tier or state.lodging.hotel_quality_tier,
                            "property_type_preferences": property_types,
                            "class_preference_source_operation_refs": _append_unique(
                                state.lodging.class_preference_source_operation_refs,
                                operation_ref,
                            ),
                            "not_applicable": False,
                        },
                        deep=True,
                    )
                },
                deep=True,
            )
        return state
    if isinstance(proposal, SetLodgingClassPreferenceOperation):
        property_types = list(state.lodging.property_type_preferences)
        if proposal.property_type and proposal.property_type not in property_types:
            property_types.append(proposal.property_type)
        budget = (
            MoneyRange(
                minimum_minor=proposal.nightly_budget_minimum_minor,
                maximum_minor=proposal.nightly_budget_maximum_minor,
            )
            if proposal.nightly_budget_minimum_minor is not None
            or proposal.nightly_budget_maximum_minor is not None
            else state.lodging.nightly_budget
        )
        return state.model_copy(
            update={
                "lodging": state.lodging.model_copy(
                    update={
                        "hotel_quality_tier": (
                            proposal.hotel_quality_tier or state.lodging.hotel_quality_tier
                        ),
                        "property_type_preferences": property_types,
                        "nightly_budget": budget,
                        "class_preference_source_operation_refs": _append_unique(
                            state.lodging.class_preference_source_operation_refs,
                            operation_ref,
                        ),
                        "not_applicable": False,
                    },
                    deep=True,
                )
            },
            deep=True,
        )
    if isinstance(proposal, (SelectConcreteEntityOperation, ExcludeConcreteEntityOperation)):
        disposition = (
            proposal.disposition if isinstance(proposal, SelectConcreteEntityOperation) else "avoid"
        )
        intent = ConcreteIntentState(
            canonical_entity_id=proposal.canonical_entity_id,
            display_name=proposal.display_name,
            disposition=disposition,
            source_operation_refs=[operation_ref],
        )
        if proposal.domain is SemanticDomainV4.ATTRACTION:
            selected_values, exclusions = _replace_entity(
                state.attractions.concrete_intents,
                state.attractions.exclusions,
                intent,
            )
            return state.model_copy(
                update={
                    "attractions": state.attractions.model_copy(
                        update={
                            "concrete_intents": selected_values,
                            "exclusions": exclusions,
                        },
                        deep=True,
                    )
                },
                deep=True,
            )
        selected_values, exclusions = _replace_entity(
            state.dining.concrete_restaurant_intents,
            state.dining.exclusions,
            intent,
        )
        return state.model_copy(
            update={
                "dining": state.dining.model_copy(
                    update={
                        "concrete_restaurant_intents": selected_values,
                        "exclusions": exclusions,
                    },
                    deep=True,
                )
            },
            deep=True,
        )
    if isinstance(proposal, RevokePriorIntentOperation):
        return _revoke_operation_ref(state, proposal.prior_operation_id)
    if isinstance(proposal, AddConditionalRequirementOperation):
        text = (
            proposal.required_outcome
            if proposal.condition.casefold() in {"always", "本次旅行", "全程"}
            else f"{proposal.condition}：{proposal.required_outcome}"
        )
        if proposal.domain is SemanticDomainV4.DINING:
            dining_requirements = _append_unique(state.dining.requirements, text)
            return state.model_copy(
                update={
                    "dining": state.dining.model_copy(update={"requirements": dining_requirements})
                },
                deep=True,
            )
        if proposal.domain is SemanticDomainV4.LODGING:
            lodging_requirements = _append_unique(state.lodging.facility_requirements, text)
            return state.model_copy(
                update={
                    "lodging": state.lodging.model_copy(
                        update={
                            "facility_requirements": lodging_requirements,
                            "class_preference_source_operation_refs": _append_unique(
                                state.lodging.class_preference_source_operation_refs,
                                operation_ref,
                            ),
                            "not_applicable": False,
                        }
                    )
                },
                deep=True,
            )
        if proposal.domain is SemanticDomainV4.TRANSPORT:
            pace_preferences = _append_unique(
                state.transport_and_pace.pace_preferences,
                text,
            )
            return state.model_copy(
                update={
                    "transport_and_pace": state.transport_and_pace.model_copy(
                        update={"pace_preferences": pace_preferences},
                        deep=True,
                    ),
                    "constraints": _append_unique(state.constraints, text),
                },
                deep=True,
            )
        return state.model_copy(
            update={"constraints": _append_unique(state.constraints, text)}, deep=True
        )
    if isinstance(proposal, ResolveConflictOperation):
        conflicts = [
            item
            for item in state.unresolved_conflicts
            if str(item.conflict_id) != proposal.conflict_id
        ]
        return state.model_copy(update={"unresolved_conflicts": conflicts}, deep=True)
    if isinstance(proposal, ConfirmFinalSupplementOperation):
        if proposal.supplement_text:
            return state.model_copy(
                update={"constraints": _append_unique(state.constraints, proposal.supplement_text)},
                deep=True,
            )
        return state
    if isinstance(proposal, ConfirmTaskBookOperation):
        return state.model_copy(
            update={
                "confirmed_task_book_ref": ConfirmedTaskBookRef(
                    task_book_id=proposal.task_book_id,
                    task_book_version=proposal.task_book_version,
                    based_on_state_version=proposal.based_on_state_version,
                )
            },
            deep=True,
        )
    raise V4SemanticMergeError("unsupported V4 semantic operation")


def _validate_target_domain(proposal: object) -> None:
    domain_targets = {
        SemanticDomainV4.ATTRACTION: {
            SemanticTargetV4.ATTRACTION_PREFERENCE,
            SemanticTargetV4.ATTRACTION_ENTITY,
        },
        SemanticDomainV4.DINING: {
            SemanticTargetV4.DINING_PREFERENCE,
            SemanticTargetV4.DINING_REQUIREMENT,
            SemanticTargetV4.DINING_ENTITY,
        },
        SemanticDomainV4.LODGING: {
            SemanticTargetV4.LODGING_AREA,
            SemanticTargetV4.LODGING_CLASS,
            SemanticTargetV4.LODGING_BOOKING,
        },
        SemanticDomainV4.TRANSPORT: {SemanticTargetV4.TRANSPORT_AND_PACE},
        SemanticDomainV4.GENERAL: {
            SemanticTargetV4.TRIP_BASICS,
            SemanticTargetV4.GENERAL_CONSTRAINT,
            SemanticTargetV4.FINAL_SUPPLEMENT,
            SemanticTargetV4.TASK_BOOK_CONFIRMATION,
            SemanticTargetV4.CONFLICT,
        },
    }
    domain = getattr(proposal, "domain", None)
    target = getattr(proposal, "target", None)
    if domain is not None and target not in domain_targets[domain]:
        raise V4SemanticMergeError("operation target does not match its domain")


def _replace_direction(
    values: list[PreferenceDirectionState],
    incoming: PreferenceDirectionState,
) -> list[PreferenceDirectionState]:
    previous = next((item for item in values if item.direction_id == incoming.direction_id), None)
    if previous:
        # A later text-only toggle must not erase the signed card's semantics.
        incoming = incoming.model_copy(
            update={
                "description": incoming.description or previous.description,
                "tags": incoming.tags or previous.tags,
                "search_query": incoming.search_query or previous.search_query,
                "source_operation_refs": list(
                    dict.fromkeys(
                        [
                            *previous.source_operation_refs,
                            *incoming.source_operation_refs,
                        ]
                    )
                ),
            },
            deep=True,
        )
    return [item for item in values if item.direction_id != incoming.direction_id] + [incoming]


def _replace_entity(
    selected: list[ConcreteIntentState],
    excluded: list[ConcreteIntentState],
    incoming: ConcreteIntentState,
) -> tuple[list[ConcreteIntentState], list[ConcreteIntentState]]:
    selected = [
        item for item in selected if item.canonical_entity_id != incoming.canonical_entity_id
    ]
    excluded = [
        item for item in excluded if item.canonical_entity_id != incoming.canonical_entity_id
    ]
    if incoming.disposition == "avoid":
        excluded.append(incoming)
    else:
        selected.append(incoming)
    return selected, excluded


def _append_unique(values: list[str], incoming: str) -> list[str]:
    return values if incoming in values else [*values, incoming]


def _revoke_operation_ref(state: TripSemanticState, prior_ref: str) -> TripSemanticState:
    trip_basics = state.trip_basics.model_copy(
        update={
            "travelers": (
                []
                if prior_ref in state.trip_basics.traveler_source_operation_refs
                else state.trip_basics.travelers
            ),
            "trip_goals": (
                []
                if prior_ref in state.trip_basics.trip_goal_source_operation_refs
                else state.trip_basics.trip_goals
            ),
            "traveler_source_operation_refs": [
                ref for ref in state.trip_basics.traveler_source_operation_refs if ref != prior_ref
            ],
            "trip_goal_source_operation_refs": [
                ref for ref in state.trip_basics.trip_goal_source_operation_refs if ref != prior_ref
            ],
        },
        deep=True,
    )
    attractions = state.attractions.model_copy(
        update={
            "preference_directions": [
                item
                for item in state.attractions.preference_directions
                if prior_ref not in item.source_operation_refs
            ],
            "concrete_intents": [
                item
                for item in state.attractions.concrete_intents
                if prior_ref not in item.source_operation_refs
            ],
            "exclusions": [
                item
                for item in state.attractions.exclusions
                if prior_ref not in item.source_operation_refs
            ],
        },
        deep=True,
    )
    dining = state.dining.model_copy(
        update={
            "preference_directions": [
                item
                for item in state.dining.preference_directions
                if prior_ref not in item.source_operation_refs
            ],
            "concrete_restaurant_intents": [
                item
                for item in state.dining.concrete_restaurant_intents
                if prior_ref not in item.source_operation_refs
            ],
            "exclusions": [
                item
                for item in state.dining.exclusions
                if prior_ref not in item.source_operation_refs
            ],
        },
        deep=True,
    )
    lodging = state.lodging.model_copy(
        update={
            "area_preferences": [
                item
                for item in state.lodging.area_preferences
                if prior_ref not in item.source_operation_refs
            ],
            "existing_bookings": [
                item
                for item in state.lodging.existing_bookings
                if prior_ref not in item.source_operation_refs
            ],
            "hotel_quality_tier": (
                None
                if prior_ref in state.lodging.class_preference_source_operation_refs
                else state.lodging.hotel_quality_tier
            ),
            "property_type_preferences": (
                []
                if prior_ref in state.lodging.class_preference_source_operation_refs
                else state.lodging.property_type_preferences
            ),
            "nightly_budget": (
                None
                if prior_ref in state.lodging.class_preference_source_operation_refs
                else state.lodging.nightly_budget
            ),
            "facility_requirements": (
                []
                if prior_ref in state.lodging.class_preference_source_operation_refs
                else state.lodging.facility_requirements
            ),
            "class_preference_source_operation_refs": [
                ref
                for ref in state.lodging.class_preference_source_operation_refs
                if ref != prior_ref
            ],
        },
        deep=True,
    )
    return state.model_copy(
        update={
            "trip_basics": trip_basics,
            "attractions": attractions,
            "dining": dining,
            "lodging": lodging,
            "existing_bookings": [
                item
                for item in state.existing_bookings
                if prior_ref not in item.source_operation_refs
            ],
            "delegations": [
                item for item in state.delegations if prior_ref not in item.source_operation_refs
            ],
        },
        deep=True,
    )


def _affected_sections(proposal: object) -> set[DiscoverySection]:
    target = getattr(proposal, "target", None)
    if isinstance(proposal, SetNotApplicableOperation) and target in {
        SemanticTargetV4.LODGING_AREA,
        SemanticTargetV4.LODGING_CLASS,
        SemanticTargetV4.LODGING_BOOKING,
    }:
        # This operation clears the entire lodging projection, regardless of
        # which lodging card originated it. Revalidate all affected evidence.
        return {
            DiscoverySection.LODGING_AREA_PREFERENCE,
            DiscoverySection.LODGING_CLASS_PREFERENCE,
            DiscoverySection.FINAL_SUPPLEMENT,
        }
    if isinstance(proposal, SetTripBasicsOperation):
        if any(
            (
                proposal.destination_name is not None,
                proposal.travelers is not None,
                proposal.trip_goals is not None,
            )
        ):
            return set(DiscoverySection)
        # A date-only revision changes Provider facts and the formal schedule,
        # not the user's already evidenced experience, dining or lodging choices.
        return {DiscoverySection.OTHER, DiscoverySection.FINAL_SUPPLEMENT}
    if not isinstance(target, SemanticTargetV4):
        return set()
    return {
        SemanticTargetV4.TRIP_BASICS: set(DiscoverySection),
        SemanticTargetV4.ATTRACTION_PREFERENCE: {
            DiscoverySection.ATTRACTION_PREFERENCE,
            DiscoverySection.ATTRACTION_SPECIFIC,
            DiscoverySection.LODGING_AREA_PREFERENCE,
            DiscoverySection.LODGING_CLASS_PREFERENCE,
            DiscoverySection.FINAL_SUPPLEMENT,
        },
        SemanticTargetV4.ATTRACTION_ENTITY: {
            DiscoverySection.ATTRACTION_SPECIFIC,
            DiscoverySection.LODGING_AREA_PREFERENCE,
            DiscoverySection.LODGING_CLASS_PREFERENCE,
            DiscoverySection.FINAL_SUPPLEMENT,
        },
        SemanticTargetV4.DINING_PREFERENCE: {
            DiscoverySection.DINING_PREFERENCE,
            DiscoverySection.DINING_SPECIFIC,
            DiscoverySection.LODGING_AREA_PREFERENCE,
            DiscoverySection.LODGING_CLASS_PREFERENCE,
            DiscoverySection.FINAL_SUPPLEMENT,
        },
        SemanticTargetV4.DINING_REQUIREMENT: {
            DiscoverySection.DINING_PREFERENCE,
            DiscoverySection.LODGING_AREA_PREFERENCE,
            DiscoverySection.LODGING_CLASS_PREFERENCE,
            DiscoverySection.FINAL_SUPPLEMENT,
        },
        SemanticTargetV4.DINING_ENTITY: {
            DiscoverySection.DINING_SPECIFIC,
            DiscoverySection.LODGING_AREA_PREFERENCE,
            DiscoverySection.LODGING_CLASS_PREFERENCE,
            DiscoverySection.FINAL_SUPPLEMENT,
        },
        SemanticTargetV4.LODGING_AREA: {
            DiscoverySection.LODGING_AREA_PREFERENCE,
            DiscoverySection.FINAL_SUPPLEMENT,
        },
        SemanticTargetV4.LODGING_CLASS: {
            DiscoverySection.LODGING_CLASS_PREFERENCE,
            DiscoverySection.FINAL_SUPPLEMENT,
        },
        SemanticTargetV4.LODGING_BOOKING: {
            DiscoverySection.LODGING_AREA_PREFERENCE,
            DiscoverySection.LODGING_CLASS_PREFERENCE,
            DiscoverySection.FINAL_SUPPLEMENT,
        },
        SemanticTargetV4.TRANSPORT_AND_PACE: {
            DiscoverySection.OTHER,
            DiscoverySection.FINAL_SUPPLEMENT,
        },
        SemanticTargetV4.GENERAL_CONSTRAINT: {
            DiscoverySection.OTHER,
            DiscoverySection.FINAL_SUPPLEMENT,
        },
        SemanticTargetV4.FINAL_SUPPLEMENT: {DiscoverySection.FINAL_SUPPLEMENT},
        SemanticTargetV4.TASK_BOOK_CONFIRMATION: set(),
        SemanticTargetV4.CONFLICT: set(DiscoverySection),
    }[target]


__all__ = [
    "AcceptedV4Operation",
    "V4SemanticMergeError",
    "WorkingAggregateMerge",
    "advance_without_operations",
    "merge_v4_operations",
    "stable_operation_id",
    "task_book_confirmation_fingerprint",
]
