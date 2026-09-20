"""Program-owned final guard and source-bound V4 task-book materialization."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import NAMESPACE_URL, uuid5

from backend.agent.prepare.section_guard import semantic_delegation_refs
from backend.contracts.v4.content_quality import (
    attraction_direction_quality_issue,
    visible_text_quality_issue,
)
from backend.contracts.v4.enums import (
    CompletionMode,
    CoverageStatus,
    DiscoverySection,
    TaskBookStatus,
)
from backend.contracts.v4.semantic_operations import AttractionDisposition, DiningDisposition
from backend.contracts.v4.state import (
    ConcreteIntentState,
    DiscoveryRuntimeState,
    TaskBookCandidate,
    TripSemanticState,
)
from backend.contracts.v4.task_book import (
    AttractionDirection,
    DestinationAndDates,
    DiningDirection,
    EvidenceBackedText,
    LodgingDirection,
    PaceAndTransport,
    TaskBookEntityIntent,
    TaskBookV4,
    TravelersAndTripGoal,
)
from backend.domain.discovery.cold_start import saved_preference_notes


@dataclass(frozen=True, slots=True)
class TaskBookGuardResult:
    ready: bool
    blocking_reasons: tuple[str, ...]


class TaskBookFinalGuard:
    """Require evidence-backed completion, not merely a displayed or asked section."""

    def evaluate(
        self,
        semantic: TripSemanticState,
        runtime: DiscoveryRuntimeState,
    ) -> TaskBookGuardResult:
        reasons: list[str] = []
        if semantic.state_version != runtime.state_version:
            reasons.append("state_version_mismatch")
        for section, coverage in runtime.section_coverage.items():
            if coverage.status not in {
                CoverageStatus.COMPLETE,
                CoverageStatus.NOT_APPLICABLE,
            }:
                reasons.append(f"section_incomplete:{section.value}")
            elif not coverage.evidence_refs or coverage.completion_mode is None:
                reasons.append(f"section_without_evidence:{section.value}")
            elif not _completion_matches_semantic(section, coverage.completion_mode, semantic):
                reasons.append(f"completion_mode_mismatch:{section.value}")
        basics = semantic.trip_basics
        if not basics.destination_name or not basics.destination_canonical_id:
            reasons.append("destination_incomplete")
        if not basics.start_date or not basics.end_date or not basics.duration_days:
            reasons.append("dates_incomplete")
        reasons.extend(_semantic_content_reasons(semantic))
        reasons.extend(_semantic_entity_reasons(semantic))
        if semantic.unresolved_conflicts:
            reasons.extend(
                f"blocking_conflict:{item.conflict_id}" for item in semantic.unresolved_conflicts
            )
        final = runtime.section_coverage[DiscoverySection.FINAL_SUPPLEMENT]
        if final.status not in {CoverageStatus.COMPLETE, CoverageStatus.NOT_APPLICABLE}:
            reasons.append("final_supplement_incomplete")
        return TaskBookGuardResult(
            ready=not reasons, blocking_reasons=tuple(dict.fromkeys(reasons))
        )


def _semantic_content_reasons(semantic: TripSemanticState) -> list[str]:
    """Reject unreadable projections even when a section has formal evidence."""

    checks: list[tuple[str, str, int]] = []
    basics = semantic.trip_basics
    if basics.destination_name is not None:
        checks.append(("destination", basics.destination_name, 2))
    checks.extend((f"trip_goal:{index}", value, 2) for index, value in enumerate(basics.trip_goals))
    for prefix, preference_values in (
        ("attraction_preference", semantic.attractions.preference_directions),
        ("dining_preference", semantic.dining.preference_directions),
        ("lodging_area", semantic.lodging.area_preferences),
    ):
        checks.extend(
            (f"{prefix}:{index}", item.label, 2) for index, item in enumerate(preference_values)
        )
    for prefix, intent_values in (
        ("attraction_intent", semantic.attractions.concrete_intents),
        ("attraction_exclusion", semantic.attractions.exclusions),
        ("dining_intent", semantic.dining.concrete_restaurant_intents),
        ("dining_exclusion", semantic.dining.exclusions),
        ("lodging_hotel_intent", semantic.lodging.user_named_hotel_intents),
    ):
        checks.extend(
            (f"{prefix}:{index}", item.display_name, 2) for index, item in enumerate(intent_values)
        )
    for prefix, text_values in (
        ("dining_requirement", semantic.dining.requirements),
        ("dining_allergy", semantic.dining.allergies),
        ("dining_avoidance", semantic.dining.avoidances),
        ("lodging_property_type", semantic.lodging.property_type_preferences),
        ("lodging_facility", semantic.lodging.facility_requirements),
        ("constraint", semantic.constraints),
        ("pace", semantic.transport_and_pace.pace_preferences),
        ("transport", semantic.transport_and_pace.transport_preferences),
    ):
        checks.extend((f"{prefix}:{index}", value, 2) for index, value in enumerate(text_values))
    checks.extend(
        (f"existing_booking:{index}", item.user_description, 2)
        for index, item in enumerate(semantic.existing_bookings)
    )
    reasons: list[str] = []
    for path, value, minimum in checks:
        issue = visible_text_quality_issue(value, minimum_units=minimum)
        if issue is None and path.startswith("attraction_preference:"):
            issue = attraction_direction_quality_issue(value)
        if issue is not None:
            reasons.append(f"invalid_semantic_text:{path}:{issue}")
    return reasons


def _semantic_entity_reasons(semantic: TripSemanticState) -> list[str]:
    """Last-line domain consistency checks on the task-book projection.

    Original taxonomy and identity are verified before signing a card or
    accepting a tool entity. Here we reject mismatched dispositions, duplicate
    cross-domain identities and obvious facility/business category leakage;
    this is not a substitute for those Provider source checks.
    """

    reasons: list[str] = []
    domains = (
        (
            "attraction",
            (*semantic.attractions.concrete_intents, *semantic.attractions.exclusions),
            {item.value for item in AttractionDisposition},
            r"(?:餐厅|餐馆|炒货店|便利店|超市|4S店|旗舰店|办公大厦|秘书处|售票处|停车场|分会场|展位)$",
        ),
        (
            "dining",
            (*semantic.dining.concrete_restaurant_intents, *semantic.dining.exclusions),
            {item.value for item in DiningDisposition},
            r"(?:博物馆|博物院|公园|地铁站|火车站|办公大厦|4S店|停车场)$",
        ),
    )
    entity_domains: dict[str, str] = {}
    for domain, intents, dispositions, wrong_kind in domains:
        seen: set[str] = set()
        for index, item in enumerate(intents):
            path = f"{domain}:{index}"
            if item.disposition not in dispositions:
                reasons.append(f"entity_disposition_mismatch:{path}")
            if not item.source_operation_refs:
                reasons.append(f"entity_evidence_missing:{path}")
            if re.search(wrong_kind, item.display_name.strip(), re.IGNORECASE):
                reasons.append(f"entity_category_mismatch:{path}")
            if item.canonical_entity_id in seen:
                reasons.append(f"duplicate_entity_intent:{path}")
            seen.add(item.canonical_entity_id)
            previous_domain = entity_domains.get(item.canonical_entity_id)
            if previous_domain is not None and previous_domain != domain:
                reasons.append(f"cross_domain_entity_conflict:{path}")
            entity_domains[item.canonical_entity_id] = domain
    for index, area_preference in enumerate(semantic.lodging.area_preferences):
        place_label = area_preference.label.rsplit("·", maxsplit=1)[-1].strip()
        if re.search(
            r"(?:餐厅|餐馆|炒货店|便利店|超市|4S店|旗舰店|公司|写字楼|办公大厦)$"
            r"|公寓|[（(][^）)]*店[）)]",
            place_label,
            re.IGNORECASE,
        ):
            reasons.append(f"entity_category_mismatch:lodging_area:{index}")
    return reasons


def _completion_matches_semantic(
    section: DiscoverySection,
    mode: CompletionMode,
    semantic: TripSemanticState,
) -> bool:
    if mode is CompletionMode.EXPLICITLY_NONE:
        return section in {
            DiscoverySection.ATTRACTION_PREFERENCE,
            DiscoverySection.ATTRACTION_SPECIFIC,
            DiscoverySection.DINING_PREFERENCE,
            DiscoverySection.DINING_SPECIFIC,
        }
    if mode is CompletionMode.EXISTING_BOOKING:
        return section in {
            DiscoverySection.LODGING_AREA_PREFERENCE,
            DiscoverySection.LODGING_CLASS_PREFERENCE,
        } and bool(semantic.lodging.existing_bookings)
    if mode is CompletionMode.NOT_APPLICABLE:
        return (
            section
            in {
                DiscoverySection.LODGING_AREA_PREFERENCE,
                DiscoverySection.LODGING_CLASS_PREFERENCE,
            }
            and semantic.lodging.not_applicable
        )
    if mode is CompletionMode.DELEGATED:
        return bool(semantic_delegation_refs(section, semantic))
    if mode is not CompletionMode.SELECTED:
        return False
    if section is DiscoverySection.OTHER:
        basics = semantic.trip_basics
        return bool(basics.destination_name and basics.destination_canonical_id)
    if section is DiscoverySection.ATTRACTION_PREFERENCE:
        return bool(semantic.attractions.preference_directions)
    if section is DiscoverySection.ATTRACTION_SPECIFIC:
        return bool(semantic.attractions.concrete_intents or semantic.attractions.exclusions)
    if section is DiscoverySection.DINING_PREFERENCE:
        return bool(
            semantic.dining.preference_directions
            or semantic.dining.requirements
            or semantic.dining.allergies
            or semantic.dining.avoidances
        )
    if section is DiscoverySection.DINING_SPECIFIC:
        return bool(semantic.dining.concrete_restaurant_intents or semantic.dining.exclusions)
    if section is DiscoverySection.LODGING_AREA_PREFERENCE:
        return bool(semantic.lodging.area_preferences)
    if section is DiscoverySection.LODGING_CLASS_PREFERENCE:
        return bool(
            semantic.lodging.hotel_quality_tier
            or semantic.lodging.hotel_quality_tiers
            or semantic.lodging.property_type_preferences
            or semantic.lodging.nightly_budget
            or semantic.lodging.facility_requirements
        )
    return section is DiscoverySection.FINAL_SUPPLEMENT


class TaskBookBuilder:
    def __init__(
        self,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._clock = clock
        self._guard = TaskBookFinalGuard()

    def preflight(
        self,
        semantic: TripSemanticState,
        runtime: DiscoveryRuntimeState,
    ) -> TaskBookGuardResult:
        """Expose the unchanged final guard before task-book execution starts."""

        return self._guard.evaluate(semantic, runtime)

    def build(
        self,
        semantic: TripSemanticState,
        runtime: DiscoveryRuntimeState,
    ) -> TaskBookCandidate:
        guard = self.preflight(semantic, runtime)
        if not guard.ready:
            raise ValueError("task book final guard failed: " + ",".join(guard.blocking_reasons))
        basics = semantic.trip_basics
        assert basics.destination_name is not None
        assert basics.destination_canonical_id is not None
        assert basics.start_date is not None
        assert basics.end_date is not None
        assert basics.duration_days is not None
        previous = runtime.task_book_candidate
        version = previous.value.version + 1 if previous is not None else 1
        task_book_id = (
            previous.task_book_id
            if previous is not None
            else str(uuid5(NAMESPACE_URL, f"v4-task-book:{semantic.trip_id}"))
        )
        created_at = self._now()
        source_refs = _all_source_refs(semantic, runtime)
        value = TaskBookV4(
            task_book_id=task_book_id,
            version=version,
            based_on_state_version=semantic.state_version,
            status=TaskBookStatus.AWAITING_CONFIRMATION,
            created_at=created_at,
            destination_and_dates=DestinationAndDates(
                destination_name=basics.destination_name,
                destination_canonical_id=basics.destination_canonical_id,
                start_date=basics.start_date,
                end_date=basics.end_date,
                duration_days=basics.duration_days,
                source_evidence_refs=(
                    list(
                        dict.fromkeys(
                            [
                                *basics.destination_source_operation_refs,
                                *basics.date_source_operation_refs,
                            ]
                        )
                    )
                    or ["semantic:trip-basics"]
                ),
            ),
            travelers_and_trip_goal=TravelersAndTripGoal(
                travelers=basics.travelers,
                trip_goals=[
                    EvidenceBackedText(
                        value=item,
                        source_evidence_refs=(
                            basics.trip_goal_source_operation_refs or ["semantic:trip-basics"]
                        ),
                    )
                    for item in basics.trip_goals
                ],
            ),
            pace_and_transport=_pace_and_transport(semantic),
            attraction_direction=_attraction_direction(semantic),
            dining_direction=_dining_direction(semantic),
            lodging_direction=_lodging_direction(semantic),
            hard_constraints=[
                EvidenceBackedText(
                    value=item,
                    source_evidence_refs=["semantic:general-constraint"],
                )
                for item in semantic.constraints
            ],
            existing_bookings=semantic.existing_bookings,
            tradeoffs_and_assumptions=[
                *saved_preference_notes(semantic),
                EvidenceBackedText(
                    value=(
                        "营业时间、票务、路线、具体酒店与价格将在正式规划时通过实时能力重新核验。"
                    ),
                    source_evidence_refs=["system:v4-planning-boundary"],
                ),
            ],
            unresolved_non_blocking_items=[],
            source_evidence_refs=source_refs,
        )
        return TaskBookCandidate(
            task_book_id=task_book_id,
            based_on_state_version=semantic.state_version,
            status=TaskBookStatus.AWAITING_CONFIRMATION,
            created_at=created_at,
            unresolved_items=[],
            value=value,
        )

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("task-book clock must return an aware datetime")
        return value.astimezone(UTC)


def project_plan_change_task_book(
    confirmed: TaskBookV4,
    semantic: TripSemanticState,
    runtime: DiscoveryRuntimeState,
) -> TaskBookV4:
    """Project accepted non-core edits without minting a new confirmed task book.

    The durable confirmed book remains immutable.  This view exists only while a
    validated PlanChangeRequest is replanned, so Planner can see updated soft
    preferences while all identity, version and core-authority fields stay bound
    to the confirmed book.
    """

    basics = semantic.trip_basics
    projected_attractions = _attraction_direction(semantic)
    projected_dining = _dining_direction(semantic)
    projected_lodging = _lodging_direction(semantic)
    protected_pairs = (
        (
            "destination_and_dates",
            (
                basics.destination_name,
                basics.destination_canonical_id,
                basics.start_date,
                basics.end_date,
                basics.duration_days,
            ),
            (
                confirmed.destination_and_dates.destination_name,
                confirmed.destination_and_dates.destination_canonical_id,
                confirmed.destination_and_dates.start_date,
                confirmed.destination_and_dates.end_date,
                confirmed.destination_and_dates.duration_days,
            ),
        ),
        (
            "travelers_and_trip_goal",
            (basics.travelers, basics.trip_goals),
            (
                confirmed.travelers_and_trip_goal.travelers,
                [item.value for item in confirmed.travelers_and_trip_goal.trip_goals],
            ),
        ),
        (
            "attraction_must_visit",
            projected_attractions.must_visit,
            confirmed.attraction_direction.must_visit,
        ),
        (
            "attraction_exclusions",
            projected_attractions.exclusions,
            confirmed.attraction_direction.exclusions,
        ),
        (
            "dining_destination_restaurants",
            projected_dining.destination_restaurants,
            confirmed.dining_direction.destination_restaurants,
        ),
        (
            "dining_excluded_restaurants",
            projected_dining.excluded_restaurants,
            confirmed.dining_direction.excluded_restaurants,
        ),
        (
            "dining_hard_requirements",
            projected_dining.hard_requirements,
            confirmed.dining_direction.hard_requirements,
        ),
        (
            "lodging_existing_booking",
            projected_lodging.existing_booking,
            confirmed.lodging_direction.existing_booking,
        ),
        ("existing_bookings", semantic.existing_bookings, confirmed.existing_bookings),
    )
    changed_protected = [name for name, current, baseline in protected_pairs if current != baseline]
    if changed_protected:
        raise ValueError(
            "core task-book fields require reconfirmation: " + ",".join(changed_protected)
        )
    return confirmed.model_copy(
        update={
            "pace_and_transport": _pace_and_transport(semantic),
            "attraction_direction": projected_attractions,
            "dining_direction": projected_dining,
            "lodging_direction": projected_lodging,
            "hard_constraints": [
                EvidenceBackedText(
                    value=item,
                    source_evidence_refs=["semantic:general-constraint"],
                )
                for item in semantic.constraints
            ],
            "source_evidence_refs": list(
                dict.fromkeys(
                    [*confirmed.source_evidence_refs, *_all_source_refs(semantic, runtime)]
                )
            ),
        },
        deep=True,
    )


def _pace_and_transport(semantic: TripSemanticState) -> PaceAndTransport:
    return PaceAndTransport(
        pace_preferences=[
            EvidenceBackedText(
                value=item,
                source_evidence_refs=["semantic:transport-and-pace"],
            )
            for item in semantic.transport_and_pace.pace_preferences
        ],
        transport_preferences=[
            EvidenceBackedText(
                value=item,
                source_evidence_refs=["semantic:transport-and-pace"],
            )
            for item in semantic.transport_and_pace.transport_preferences
        ],
    )


def _lodging_direction(semantic: TripSemanticState) -> LodgingDirection:
    return LodgingDirection(
        area_preferences=[
            _direction_text(item.label, item.selected, item.source_operation_refs)
            for item in semantic.lodging.area_preferences
        ],
        hotel_quality_tier=semantic.lodging.hotel_quality_tier,
        hotel_quality_tiers=semantic.lodging.hotel_quality_tiers,
        search_examples=[
            example
            for area in semantic.lodging.area_preferences
            if area.selected
            for example in area.lodging_examples
        ],
        property_type_preferences=[
            EvidenceBackedText(
                value=item,
                source_evidence_refs=(
                    semantic.lodging.class_preference_source_operation_refs
                    or ["semantic:lodging-class"]
                ),
            )
            for item in semantic.lodging.property_type_preferences
        ],
        nightly_budget=semantic.lodging.nightly_budget,
        facility_requirements=[
            EvidenceBackedText(
                value=item,
                source_evidence_refs=(
                    semantic.lodging.class_preference_source_operation_refs
                    or ["semantic:lodging-class"]
                ),
            )
            for item in semantic.lodging.facility_requirements
        ],
        existing_booking=(
            semantic.lodging.existing_bookings[0] if semantic.lodging.existing_bookings else None
        ),
        delegated_scope=semantic.lodging.delegation,
        not_applicable=semantic.lodging.not_applicable,
    )


def _attraction_direction(semantic: TripSemanticState) -> AttractionDirection:
    intents = semantic.attractions.concrete_intents
    exclusions = semantic.attractions.exclusions
    return AttractionDirection(
        preferences=[
            _direction_text(item.label, item.selected, item.source_operation_refs)
            for item in semantic.attractions.preference_directions
        ],
        must_visit=[
            _entity_intent(item, AttractionDisposition.MUST)
            for item in intents
            if item.disposition == AttractionDisposition.MUST.value
        ],
        wanted=[
            _entity_intent(item, AttractionDisposition.WANT)
            for item in intents
            if item.disposition == AttractionDisposition.WANT.value
        ],
        if_convenient=[
            _entity_intent(item, AttractionDisposition.IF_CONVENIENT)
            for item in intents
            if item.disposition == AttractionDisposition.IF_CONVENIENT.value
        ],
        exclusions=[_entity_intent(item, AttractionDisposition.AVOID) for item in exclusions],
        delegated_scope=semantic.attractions.delegation,
    )


def _dining_direction(semantic: TripSemanticState) -> DiningDirection:
    intents = semantic.dining.concrete_restaurant_intents
    exclusions = semantic.dining.exclusions
    return DiningDirection(
        preferences=[
            _direction_text(item.label, item.selected, item.source_operation_refs)
            for item in semantic.dining.preference_directions
        ],
        hard_requirements=[
            EvidenceBackedText(
                value=item,
                source_evidence_refs=["semantic:dining-requirements"],
            )
            for item in [
                *semantic.dining.requirements,
                *semantic.dining.allergies,
                *semantic.dining.avoidances,
            ]
        ],
        destination_restaurants=[
            _entity_intent(item, DiningDisposition.DESTINATION)
            for item in intents
            if item.disposition == DiningDisposition.DESTINATION.value
        ],
        if_convenient_restaurants=[
            _entity_intent(item, DiningDisposition.IF_CONVENIENT)
            for item in intents
            if item.disposition == DiningDisposition.IF_CONVENIENT.value
        ],
        excluded_restaurants=[_entity_intent(item, DiningDisposition.AVOID) for item in exclusions],
        delegated_scope=semantic.dining.delegation,
    )


def _direction_text(label: str, selected: bool, refs: list[str]) -> EvidenceBackedText:
    return EvidenceBackedText(
        value=label if selected else f"排除偏好方向：{label}",
        source_evidence_refs=refs,
    )


def _entity_intent(
    item: ConcreteIntentState,
    disposition: AttractionDisposition | DiningDisposition,
) -> TaskBookEntityIntent:
    return TaskBookEntityIntent(
        canonical_entity_id=item.canonical_entity_id,
        display_name=item.display_name,
        disposition=disposition,
        source_operation_refs=item.source_operation_refs,
        facts_to_verify=["opening_hours", "route", "availability"],
    )


def _all_source_refs(
    semantic: TripSemanticState,
    runtime: DiscoveryRuntimeState,
) -> list[str]:
    refs: list[str] = [
        "semantic:trip-basics",
        "system:v4-planning-boundary",
    ]
    refs.extend(semantic.trip_basics.traveler_source_operation_refs)
    refs.extend(semantic.trip_basics.trip_goal_source_operation_refs)
    refs.extend(semantic.trip_basics.destination_source_operation_refs)
    refs.extend(semantic.trip_basics.date_source_operation_refs)
    if semantic.cold_start_profile_snapshot is not None:
        refs.extend(semantic.cold_start_profile_snapshot.source_evidence_refs)
    refs.extend(
        f"memory:{memory.memory_id}"
        for memory in semantic.long_term_memory_snapshot or ()
        if memory.kind == "preference"
    )
    refs.extend(
        ref for coverage in runtime.section_coverage.values() for ref in coverage.evidence_refs
    )
    refs.extend(
        ref
        for values in (
            semantic.attractions.preference_directions,
            semantic.attractions.concrete_intents,
            semantic.attractions.exclusions,
            semantic.dining.preference_directions,
            semantic.dining.concrete_restaurant_intents,
            semantic.dining.exclusions,
            semantic.lodging.area_preferences,
        )
        for item in values
        for ref in item.source_operation_refs
    )
    for delegation in semantic.delegations:
        refs.extend(delegation.source_operation_refs)
    for booking in semantic.existing_bookings:
        refs.extend(booking.source_operation_refs)
    refs.extend(semantic.lodging.class_preference_source_operation_refs)
    return list(dict.fromkeys(refs))


__all__ = [
    "TaskBookBuilder",
    "TaskBookFinalGuard",
    "TaskBookGuardResult",
    "project_plan_change_task_book",
]
