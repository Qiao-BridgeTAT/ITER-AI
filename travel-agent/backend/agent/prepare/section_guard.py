"""Program-owned V4 discovery order and evidence-backed completion rules."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID

from backend.contracts.v4.enums import (
    CompletionMode,
    CoverageStatus,
    DiscoverySection,
)
from backend.contracts.v4.semantic_operations import (
    ConfirmFinalSupplementOperation,
    SemanticOperationProposal,
    SemanticTargetV4,
    SetDelegationScopeOperation,
    SetExistingBookingOperation,
    SetNoPreferenceOperation,
    SetNotApplicableOperation,
)
from backend.contracts.v4.state import (
    DiscoveryRuntimeState,
    SectionCoverage,
    TripSemanticState,
)
from backend.domain.discovery.state_merge import AcceptedV4Operation

DISCOVERY_ORDER = (
    DiscoverySection.OTHER,
    DiscoverySection.ATTRACTION_PREFERENCE,
    DiscoverySection.ATTRACTION_SPECIFIC,
    DiscoverySection.DINING_PREFERENCE,
    DiscoverySection.DINING_SPECIFIC,
    DiscoverySection.LODGING_AREA_PREFERENCE,
    DiscoverySection.LODGING_CLASS_PREFERENCE,
    DiscoverySection.FINAL_SUPPLEMENT,
)


@dataclass(frozen=True, slots=True)
class SectionGuardResult:
    runtime_state: DiscoveryRuntimeState
    completed_sections: tuple[DiscoverySection, ...]
    reopened_sections: tuple[DiscoverySection, ...]
    blocking_reasons: tuple[str, ...]


def recalculate_section_coverage(
    semantic_state: TripSemanticState,
    runtime_state: DiscoveryRuntimeState,
    accepted_operations: tuple[AcceptedV4Operation, ...],
    *,
    affected_sections: frozenset[DiscoverySection],
) -> SectionGuardResult:
    """Recalculate only from durable semantics and authorized evidence."""

    version = semantic_state.state_version
    coverage = {
        section: value.model_copy(deep=True)
        for section, value in runtime_state.section_coverage.items()
    }
    completed: list[DiscoverySection] = []
    reopened: list[DiscoverySection] = []
    operations = [(item.operation_id, item.proposal) for item in accepted_operations]
    for section in DISCOVERY_ORDER:
        previous = coverage[section]
        evidence = _completion_evidence(
            section,
            semantic_state,
            operations,
            previous_coverage=previous,
        )
        should_recalculate = (
            section is DiscoverySection.OTHER
            or section in affected_sections
            or previous.status
            not in {
                CoverageStatus.COMPLETE,
                CoverageStatus.NOT_APPLICABLE,
            }
        )
        if not should_recalculate:
            continue
        if evidence is not None:
            mode, evidence_refs, status = evidence
            coverage[section] = SectionCoverage(
                status=status,
                required_targets=[section.value],
                covered_targets=[section.value],
                completion_mode=mode,
                evidence_refs=evidence_refs,
                completed_at_state_version=version,
            )
            if previous.status not in {CoverageStatus.COMPLETE, CoverageStatus.NOT_APPLICABLE}:
                completed.append(section)
            continue
        if previous.status in {CoverageStatus.COMPLETE, CoverageStatus.NOT_APPLICABLE}:
            coverage[section] = SectionCoverage(
                status=CoverageStatus.REOPENED,
                required_targets=[section.value],
            )
            reopened.append(section)
        elif section in affected_sections or previous.status is CoverageStatus.NOT_STARTED:
            coverage[section] = SectionCoverage(
                status=(
                    CoverageStatus.IN_PROGRESS if section in affected_sections else previous.status
                ),
                required_targets=[section.value],
            )

    current = _current_section(runtime_state.current_section, coverage, reopened)
    return SectionGuardResult(
        runtime_state=runtime_state.model_copy(
            update={"section_coverage": coverage, "current_section": current},
            deep=True,
        ),
        completed_sections=tuple(completed),
        reopened_sections=tuple(reopened),
        blocking_reasons=tuple(
            f"unresolved_conflict:{item.conflict_id}"
            for item in semantic_state.unresolved_conflicts
        ),
    )


def _completion_evidence(
    section: DiscoverySection,
    semantic: TripSemanticState,
    operations: Sequence[tuple[UUID, SemanticOperationProposal]],
    *,
    previous_coverage: SectionCoverage,
) -> tuple[CompletionMode, list[str], CoverageStatus] | None:
    relevant = [
        (str(operation_id), wrapped.root)
        for operation_id, wrapped in operations
        if _operation_covers(section, wrapped.root)
    ]
    explicit_not_applicable = [
        operation_id
        for operation_id, proposal in relevant
        if isinstance(proposal, SetNotApplicableOperation)
    ]
    if explicit_not_applicable:
        return (
            CompletionMode.NOT_APPLICABLE,
            explicit_not_applicable,
            CoverageStatus.NOT_APPLICABLE,
        )
    if (
        section
        in {
            DiscoverySection.LODGING_AREA_PREFERENCE,
            DiscoverySection.LODGING_CLASS_PREFERENCE,
        }
        and semantic.lodging.not_applicable
        and previous_coverage.status is CoverageStatus.NOT_APPLICABLE
        and previous_coverage.completion_mode is CompletionMode.NOT_APPLICABLE
        and previous_coverage.evidence_refs
    ):
        # Attraction and dining choices intentionally invalidate lodging
        # recommendations, but they cannot revoke a durable user conclusion
        # that lodging itself is not applicable. The accepting operation may
        # belong to an earlier committed turn, so preserve its runtime evidence
        # instead of requiring it to appear in this turn's operation batch.
        return (
            CompletionMode.NOT_APPLICABLE,
            list(previous_coverage.evidence_refs),
            CoverageStatus.NOT_APPLICABLE,
        )
    existing_booking = [
        operation_id
        for operation_id, proposal in relevant
        if isinstance(proposal, SetExistingBookingOperation)
    ]
    if existing_booking or (
        section
        in {
            DiscoverySection.LODGING_AREA_PREFERENCE,
            DiscoverySection.LODGING_CLASS_PREFERENCE,
        }
        and semantic.lodging.existing_bookings
    ):
        refs = existing_booking or sorted(
            {
                ref
                for booking in semantic.lodging.existing_bookings
                for ref in booking.source_operation_refs
            }
        )
        return CompletionMode.EXISTING_BOOKING, refs, CoverageStatus.COMPLETE
    delegated = [
        operation_id
        for operation_id, proposal in relevant
        if isinstance(proposal, SetDelegationScopeOperation)
    ]
    if delegated or semantic_delegation_refs(section, semantic):
        return (
            CompletionMode.DELEGATED,
            delegated or semantic_delegation_refs(section, semantic),
            CoverageStatus.COMPLETE,
        )
    no_preference = [
        operation_id
        for operation_id, proposal in relevant
        if isinstance(proposal, SetNoPreferenceOperation)
    ]
    if no_preference:
        return CompletionMode.EXPLICITLY_NONE, no_preference, CoverageStatus.COMPLETE
    selected_refs = _semantic_selection_refs(section, semantic)
    selected_refs.extend(
        operation_id
        for operation_id, proposal in relevant
        if isinstance(proposal, ConfirmFinalSupplementOperation)
    )
    selected_refs = sorted(set(selected_refs))
    if selected_refs:
        return CompletionMode.SELECTED, selected_refs, CoverageStatus.COMPLETE
    return None


def _operation_covers(section: DiscoverySection, proposal: object) -> bool:
    if isinstance(proposal, SetDelegationScopeOperation):
        return delegation_covers_section(section, proposal.delegated_targets)
    if isinstance(proposal, SetNotApplicableOperation) and proposal.target in {
        SemanticTargetV4.LODGING_AREA,
        SemanticTargetV4.LODGING_CLASS,
    }:
        return section in {
            DiscoverySection.LODGING_AREA_PREFERENCE,
            DiscoverySection.LODGING_CLASS_PREFERENCE,
        }
    target = getattr(proposal, "target", None)
    target_sections = {
        SemanticTargetV4.TRIP_BASICS: {DiscoverySection.OTHER},
        SemanticTargetV4.ATTRACTION_PREFERENCE: {DiscoverySection.ATTRACTION_PREFERENCE},
        SemanticTargetV4.ATTRACTION_ENTITY: {DiscoverySection.ATTRACTION_SPECIFIC},
        SemanticTargetV4.DINING_PREFERENCE: {DiscoverySection.DINING_PREFERENCE},
        SemanticTargetV4.DINING_REQUIREMENT: {DiscoverySection.DINING_PREFERENCE},
        SemanticTargetV4.DINING_ENTITY: {DiscoverySection.DINING_SPECIFIC},
        SemanticTargetV4.LODGING_AREA: {DiscoverySection.LODGING_AREA_PREFERENCE},
        SemanticTargetV4.LODGING_CLASS: {DiscoverySection.LODGING_CLASS_PREFERENCE},
        SemanticTargetV4.LODGING_BOOKING: {
            DiscoverySection.LODGING_AREA_PREFERENCE,
            DiscoverySection.LODGING_CLASS_PREFERENCE,
        },
        SemanticTargetV4.TRANSPORT_AND_PACE: {DiscoverySection.OTHER},
        SemanticTargetV4.GENERAL_CONSTRAINT: {DiscoverySection.OTHER},
        SemanticTargetV4.FINAL_SUPPLEMENT: {DiscoverySection.FINAL_SUPPLEMENT},
        SemanticTargetV4.CONFLICT: set(),
    }
    if not isinstance(target, SemanticTargetV4):
        return False
    return section in target_sections.get(target, set())


def delegation_covers_section(section: DiscoverySection, targets: Sequence[str]) -> bool:
    # Model semantic targets and signed-card section names denote the same scope.
    aliases = {
        DiscoverySection.ATTRACTION_PREFERENCE: "attraction_preference",
        DiscoverySection.ATTRACTION_SPECIFIC: "attraction_entity",
        DiscoverySection.DINING_PREFERENCE: "dining_preference",
        DiscoverySection.DINING_SPECIFIC: "dining_entity",
        DiscoverySection.LODGING_AREA_PREFERENCE: "lodging_area",
        DiscoverySection.LODGING_CLASS_PREFERENCE: "lodging_class",
    }
    return section in aliases and bool({section.value, aliases[section]} & set(targets))


def semantic_delegation_refs(
    section: DiscoverySection,
    semantic: TripSemanticState,
) -> list[str]:
    delegation = None
    if section in {
        DiscoverySection.ATTRACTION_PREFERENCE,
        DiscoverySection.ATTRACTION_SPECIFIC,
    }:
        delegation = semantic.attractions.delegation
    elif section in {
        DiscoverySection.DINING_PREFERENCE,
        DiscoverySection.DINING_SPECIFIC,
    }:
        delegation = semantic.dining.delegation
    elif section in {
        DiscoverySection.LODGING_AREA_PREFERENCE,
        DiscoverySection.LODGING_CLASS_PREFERENCE,
    }:
        delegation = semantic.lodging.delegation
    if delegation is None or not delegation_covers_section(section, delegation.delegated_targets):
        return []
    return list(delegation.source_operation_refs)


def _semantic_selection_refs(
    section: DiscoverySection,
    semantic: TripSemanticState,
) -> list[str]:
    if section is DiscoverySection.OTHER:
        basics = semantic.trip_basics
        if (
            basics.destination_name
            and basics.destination_canonical_id
            and basics.start_date
            and basics.end_date
            and basics.duration_days
        ):
            return sorted(
                {
                    *basics.destination_source_operation_refs,
                    *basics.date_source_operation_refs,
                }
            ) or ["semantic:trip-basics"]
        return []
    if section is DiscoverySection.ATTRACTION_PREFERENCE:
        return [
            ref
            for item in semantic.attractions.preference_directions
            for ref in item.source_operation_refs
        ]
    if section is DiscoverySection.ATTRACTION_SPECIFIC:
        return [
            ref
            for item in semantic.attractions.concrete_intents
            for ref in item.source_operation_refs
        ]
    if section is DiscoverySection.DINING_PREFERENCE:
        refs = [
            ref
            for item in semantic.dining.preference_directions
            for ref in item.source_operation_refs
        ]
        if semantic.dining.requirements or semantic.dining.allergies or semantic.dining.avoidances:
            refs.append("semantic:dining-requirements")
        return refs
    if section is DiscoverySection.DINING_SPECIFIC:
        return [
            ref
            for item in semantic.dining.concrete_restaurant_intents
            for ref in item.source_operation_refs
        ]
    if section is DiscoverySection.LODGING_AREA_PREFERENCE:
        return [
            ref for item in semantic.lodging.area_preferences for ref in item.source_operation_refs
        ]
    if section is DiscoverySection.LODGING_CLASS_PREFERENCE:
        state_refs = list(semantic.lodging.class_preference_source_operation_refs)
        if any(
            (
                semantic.lodging.hotel_quality_tier,
                semantic.lodging.hotel_quality_tiers,
                semantic.lodging.property_type_preferences,
                semantic.lodging.nightly_budget,
                semantic.lodging.facility_requirements,
            )
        ):
            return state_refs or ["semantic:lodging-class"]
        return []
    return []


def _current_section(
    previous: DiscoverySection,
    coverage: dict[DiscoverySection, SectionCoverage],
    reopened: list[DiscoverySection],
) -> DiscoverySection:
    if reopened:
        earliest = min(reopened, key=DISCOVERY_ORDER.index)
        if previous not in DISCOVERY_ORDER or DISCOVERY_ORDER.index(
            earliest
        ) <= DISCOVERY_ORDER.index(previous):
            return earliest
    if previous not in DISCOVERY_ORDER:
        return previous
    index = DISCOVERY_ORDER.index(previous)
    while index < len(DISCOVERY_ORDER) - 1 and coverage[DISCOVERY_ORDER[index]].status in {
        CoverageStatus.COMPLETE,
        CoverageStatus.NOT_APPLICABLE,
    }:
        index += 1
    return DISCOVERY_ORDER[index]


__all__ = ["DISCOVERY_ORDER", "SectionGuardResult", "recalculate_section_coverage"]
