"""Explicit V2/V3 read compatibility and one-time V4 upgrade preview."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from backend.agent.state_merge import SemanticTripState
from backend.contracts.enums import EvidenceSource
from backend.contracts.state import TripState
from backend.contracts.v4.enums import (
    CompletionMode,
    CoverageStatus,
    DiscoverySection,
    InteractionStatus,
    PendingInteractionKind,
)
from backend.contracts.v4.state import (
    ColdStartProfileSnapshot,
    DiscoveryRuntimeState,
    PendingInteraction,
    SectionCoverage,
    TripSemanticState,
    V4TripStateEnvelope,
    initial_section_coverage,
)
from backend.domain.discovery.semantic_state import adapt_legacy_semantic_state


class LegacyUpgradeAuthorizationError(PermissionError):
    pass


@dataclass(frozen=True)
class LegacyUpgradePreview:
    semantic_state: TripSemanticState
    discovery_runtime_state: DiscoveryRuntimeState
    warnings: tuple[str, ...]
    source_protocol: str


def preview_legacy_upgrade(
    trip_state: TripState,
    semantic_state: SemanticTripState,
    *,
    legacy_pending_question_id: str | None = None,
    legacy_pending_state_version: int | None = None,
) -> LegacyUpgradePreview:
    if trip_state.trip_id != semantic_state.trip_id:
        raise ValueError("legacy TripState and semantic state belong to different trips")

    semantic = adapt_legacy_semantic_state(semantic_state)
    if trip_state.personal_defaults is not None and trip_state.cold_start_completed_at is not None:
        semantic = semantic.model_copy(
            update={
                "cold_start_profile_snapshot": ColdStartProfileSnapshot(
                    profile_version=1,
                    captured_at=trip_state.cold_start_completed_at,
                    preferences=trip_state.personal_defaults,
                    source_evidence_refs=["legacy:cold-start-profile-v1"],
                )
            }
        )

    coverage = _coverage_from_explicit_semantics(semantic)
    current_section = next(
        (
            section
            for section in _ordered_sections()
            if coverage[section].status is not CoverageStatus.COMPLETE
            and coverage[section].status is not CoverageStatus.NOT_APPLICABLE
        ),
        DiscoverySection.FINAL_SUPPLEMENT,
    )
    pending = None
    warnings = [
        "legacy readiness flags were not interpreted as V4 coverage",
        "legacy system-selected hotel candidates were not interpreted as user confirmation",
        "legacy task-book candidates require regeneration against the V4 state version",
    ]
    if legacy_pending_question_id is not None:
        status = (
            InteractionStatus.ACTIVE
            if legacy_pending_state_version == semantic.state_version
            else InteractionStatus.SUPERSEDED
        )
        pending = PendingInteraction(
            interaction_id=legacy_pending_question_id,
            kind=PendingInteractionKind.FREE_TEXT_QUESTION,
            section=current_section,
            target_ids=[f"legacy-question:{legacy_pending_question_id}"],
            based_on_state_version=min(
                legacy_pending_state_version or 0,
                semantic.state_version,
            ),
            dependency_fingerprint=f"legacy:{legacy_pending_question_id}",
            status=status,
        )
        if status is InteractionStatus.SUPERSEDED:
            warnings.append("legacy pending question had no current version and was superseded")

    runtime = DiscoveryRuntimeState(
        trip_id=str(trip_state.trip_id),
        state_version=semantic.state_version,
        current_section=current_section,
        section_coverage=coverage,
        pending_interaction=pending,
    )
    return LegacyUpgradePreview(
        semantic_state=semantic,
        discovery_runtime_state=runtime,
        warnings=tuple(warnings),
        source_protocol=trip_state.schema_version,
    )


def authorize_one_time_v4_upgrade(
    preview: LegacyUpgradePreview,
    *,
    explicit_authorization: bool,
) -> V4TripStateEnvelope:
    """Build a writable V4 envelope only after an explicit application decision."""

    if not explicit_authorization:
        raise LegacyUpgradeAuthorizationError("legacy-to-V4 write requires explicit authorization")
    return V4TripStateEnvelope(
        semantic_state=preview.semantic_state,
        discovery_runtime_state=preview.discovery_runtime_state,
    )


def _coverage_from_explicit_semantics(
    semantic: TripSemanticState,
) -> dict[DiscoverySection, SectionCoverage]:
    coverage = initial_section_coverage()
    version = semantic.state_version

    def complete(section: DiscoverySection, evidence_refs: list[str]) -> None:
        coverage[section] = SectionCoverage(
            status=CoverageStatus.COMPLETE,
            required_targets=[section.value],
            covered_targets=[section.value],
            completion_mode=CompletionMode.SELECTED,
            evidence_refs=evidence_refs,
            completed_at_state_version=version,
        )

    basics = semantic.trip_basics
    destination_evidence = _operation_refs_for_targets(semantic, {"trip_identity.destination"})
    date_evidence = _operation_refs_for_targets(semantic, {"trip_identity.date_range"})
    if (
        basics.destination_name
        and basics.start_date
        and basics.end_date
        and basics.travelers
        and basics.trip_goals
        and destination_evidence
        and date_evidence
    ):
        complete(
            DiscoverySection.OTHER,
            sorted(set(destination_evidence + date_evidence)),
        )

    attraction_directions = [
        item for item in semantic.attractions.preference_directions if item.coverage_eligible
    ]
    if attraction_directions:
        complete(
            DiscoverySection.ATTRACTION_PREFERENCE,
            _refs_from_direction_items(attraction_directions),
        )
    attraction_items = [
        *semantic.attractions.concrete_intents,
        *semantic.attractions.exclusions,
    ]
    if attraction_items:
        complete(DiscoverySection.ATTRACTION_SPECIFIC, _refs_from_intents(attraction_items))

    dining_evidence = _refs_from_direction_items(
        [item for item in semantic.dining.preference_directions if item.coverage_eligible]
    )
    dining_evidence.extend(
        _operation_refs_for_targets(semantic, {"trip_preferences.dining"})
        if any(
            (
                semantic.dining.requirements,
                semantic.dining.allergies,
                semantic.dining.avoidances,
            )
        )
        else []
    )
    if dining_evidence:
        complete(DiscoverySection.DINING_PREFERENCE, sorted(set(dining_evidence)))
    restaurant_items = [
        *semantic.dining.concrete_restaurant_intents,
        *semantic.dining.exclusions,
    ]
    if restaurant_items:
        complete(DiscoverySection.DINING_SPECIFIC, _refs_from_intents(restaurant_items))

    # Old named-hotel candidate cards are deliberately not completion evidence.
    lodging_area_preferences = [
        item for item in semantic.lodging.area_preferences if item.coverage_eligible
    ]
    if lodging_area_preferences:
        complete(
            DiscoverySection.LODGING_AREA_PREFERENCE,
            _refs_from_direction_items(lodging_area_preferences),
        )
    if any(
        (
            semantic.lodging.hotel_quality_tier,
            semantic.lodging.property_type_preferences,
            semantic.lodging.nightly_budget,
            semantic.lodging.facility_requirements,
        )
    ):
        lodging_refs = _operation_refs_for_targets(semantic, {"planning.lodging"})
        if lodging_refs:
            complete(DiscoverySection.LODGING_CLASS_PREFERENCE, lodging_refs)
    return coverage


def _operation_refs_for_targets(
    semantic: TripSemanticState,
    targets: set[str],
) -> list[str]:
    refs: list[str] = []
    for entry in semantic.entries:
        if entry.operation.target.value not in targets:
            continue
        if entry.operation.evidence.source not in {
            EvidenceSource.CARD,
            EvidenceSource.DIALOGUE,
        }:
            continue
        refs.append(str(entry.operation.operation_id))
    return refs


class _SourceRefCarrier(Protocol):
    source_operation_refs: list[str]


def _refs_from_direction_items(items: Sequence[_SourceRefCarrier]) -> list[str]:
    refs: list[str] = []
    for item in items:
        refs.extend(item.source_operation_refs)
    return sorted(set(refs))


def _refs_from_intents(items: Sequence[_SourceRefCarrier]) -> list[str]:
    return _refs_from_direction_items(items)


def _ordered_sections() -> tuple[DiscoverySection, ...]:
    return (
        DiscoverySection.OTHER,
        DiscoverySection.ATTRACTION_PREFERENCE,
        DiscoverySection.ATTRACTION_SPECIFIC,
        DiscoverySection.DINING_PREFERENCE,
        DiscoverySection.DINING_SPECIFIC,
        DiscoverySection.LODGING_AREA_PREFERENCE,
        DiscoverySection.LODGING_CLASS_PREFERENCE,
        DiscoverySection.FINAL_SUPPLEMENT,
    )


__all__ = [
    "LegacyUpgradeAuthorizationError",
    "LegacyUpgradePreview",
    "authorize_one_time_v4_upgrade",
    "preview_legacy_upgrade",
]
