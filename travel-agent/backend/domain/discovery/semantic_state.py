"""One-way, explicit adaptation from the V2 semantic notebook to V4."""

from __future__ import annotations

from backend.agent.semantic_operations import (
    AttractionIntentOperation,
    ConstraintOperation,
    DateRangeOperation,
    DestinationOperation,
    DiningPreferenceKind,
    DiningPreferenceOperation,
    ExperiencePreferenceOperation,
    LodgingPreferenceKind,
    LodgingPreferenceOperation,
    PacePreferenceOperation,
    SemanticOperationKind,
    TransportPreferenceOperation,
)
from backend.agent.state_merge import SemanticTripState
from backend.agent.task_book_state import SemanticTaskBookStatus
from backend.contracts.v4.state import (
    AttractionSemanticProjection,
    ConcreteIntentState,
    ConfirmedTaskBookRef,
    DiningSemanticProjection,
    LodgingSemanticProjection,
    PreferenceDirectionState,
    TransportAndPaceProjection,
    TripBasicsProjection,
    TripSemanticState,
)


def adapt_legacy_semantic_state(state: SemanticTripState) -> TripSemanticState:
    """Create one V4 semantic source without inferring discovery completion.

    Only active, explicitly represented semantic entries are projected.  Old
    readiness flags and old system-proposed hotel selections are intentionally
    ignored; compatibility progress is handled by a separate explicit adapter.
    """

    destination_name: str | None = None
    destination_id: str | None = None
    destination_source_refs: list[str] = []
    start_date = None
    end_date = None
    date_source_refs: list[str] = []
    attractions: list[ConcreteIntentState] = []
    attraction_exclusions: list[ConcreteIntentState] = []
    attraction_directions: list[PreferenceDirectionState] = []
    dining_directions: list[PreferenceDirectionState] = []
    dining_requirements: list[str] = []
    dining_allergies: list[str] = []
    dining_avoidances: list[str] = []
    restaurants: list[ConcreteIntentState] = []
    restaurant_exclusions: list[ConcreteIntentState] = []
    lodging_areas: list[PreferenceDirectionState] = []
    lodging_quality: str | None = None
    lodging_types: list[str] = []
    named_hotels: list[ConcreteIntentState] = []
    transport: list[str] = []
    pace: list[str] = []
    constraints: list[str] = []

    for entry in state.entries:
        operation = entry.operation
        source_ref = str(operation.operation_id)
        if isinstance(operation, DestinationOperation) and operation.value is not None:
            destination_name = operation.value.display_name
            destination_id = operation.value.city_id
            destination_source_refs = [source_ref]
        elif isinstance(operation, DateRangeOperation) and operation.value is not None:
            start_date = operation.value.start_date
            end_date = operation.value.end_date
            date_source_refs = [source_ref]
        elif isinstance(operation, AttractionIntentOperation):
            intent = ConcreteIntentState(
                canonical_entity_id=str(operation.value.place_id),
                display_name=operation.value.place_name or str(operation.value.place_id),
                disposition=operation.value.intent.value,
                source_operation_refs=[source_ref],
            )
            if operation.value.intent.value == "avoid":
                attraction_exclusions.append(intent)
            else:
                attractions.append(intent)
        elif isinstance(operation, DiningPreferenceOperation):
            value = operation.value
            if value.kind is DiningPreferenceKind.SPECIFIC_RESTAURANT and value.place_id:
                restaurant = ConcreteIntentState(
                    canonical_entity_id=str(value.place_id),
                    display_name=value.value or str(value.place_id),
                    disposition=(
                        value.restaurant_intent.value
                        if value.restaurant_intent
                        else "if_convenient"
                    ),
                    source_operation_refs=[source_ref],
                )
                if restaurant.disposition == "avoid":
                    restaurant_exclusions.append(restaurant)
                else:
                    restaurants.append(restaurant)
            elif value.kind is DiningPreferenceKind.ALLERGY and value.value:
                dining_allergies.append(value.value)
            elif value.kind is DiningPreferenceKind.AVOIDANCE and value.value:
                dining_avoidances.append(value.value)
            elif value.kind is DiningPreferenceKind.DIETARY_REQUIREMENT and value.value:
                dining_requirements.append(value.value)
            elif value.value:
                dining_directions.append(
                    PreferenceDirectionState(
                        direction_id=f"legacy:{source_ref}",
                        label=value.value,
                        selected=operation.operation
                        not in {SemanticOperationKind.DELETE, SemanticOperationKind.NEGATE},
                        source_operation_refs=[source_ref],
                    )
                )
        elif isinstance(operation, LodgingPreferenceOperation):
            lodging_value = operation.value
            if lodging_value.kind in {
                LodgingPreferenceKind.AREA,
                LodgingPreferenceKind.TRANSIT_NODE,
            }:
                lodging_areas.append(
                    PreferenceDirectionState(
                        direction_id=f"legacy:{source_ref}",
                        label=lodging_value.value,
                        selected=True,
                        source_operation_refs=[source_ref],
                    )
                )
            elif lodging_value.kind is LodgingPreferenceKind.QUALITY:
                normalized_quality = lodging_value.value.casefold()
                if normalized_quality in {"economy", "comfort", "upscale", "luxury"}:
                    lodging_quality = normalized_quality
            elif (
                lodging_value.kind is LodgingPreferenceKind.SPECIFIC_HOTEL
                and lodging_value.place_id
            ):
                # Preserve the user intent, but compatibility never treats this as
                # a V4 system-recommended hotel confirmation or coverage evidence.
                named_hotels.append(
                    ConcreteIntentState(
                        canonical_entity_id=str(lodging_value.place_id),
                        display_name=lodging_value.value,
                        disposition="want",
                        source_operation_refs=[source_ref],
                    )
                )
            elif lodging_value.kind is LodgingPreferenceKind.OTHER:
                lodging_types.append(lodging_value.value)
        elif isinstance(operation, TransportPreferenceOperation):
            transport.append(operation.value.model_dump_json())
        elif isinstance(operation, PacePreferenceOperation):
            pace.append(operation.value.model_dump_json())
        elif isinstance(operation, ExperiencePreferenceOperation):
            label = operation.value.note or operation.value.theme_id
            if label:
                # Legacy experience directions are preserved as long-tail entries;
                # they are not enough by themselves to close a V4 section.
                attraction_directions.append(
                    PreferenceDirectionState(
                        direction_id=f"legacy-experience:{source_ref}",
                        label=label,
                        selected=True,
                        source_operation_refs=[source_ref],
                        coverage_eligible=False,
                    )
                )
        elif isinstance(operation, ConstraintOperation):
            constraints.append(operation.value.description)

    duration_days = None
    if start_date is not None and end_date is not None:
        duration_days = (end_date - start_date).days + 1
    confirmed_ref = None
    if state.task_book is not None and state.task_book.status is SemanticTaskBookStatus.CONFIRMED:
        confirmed_ref = ConfirmedTaskBookRef(
            task_book_id=str(state.task_book.task_book_id),
            task_book_version=state.task_book.revision,
            based_on_state_version=state.task_book.source_state_version,
        )

    return TripSemanticState(
        trip_id=str(state.trip_id),
        state_version=state.state_version,
        trip_basics=TripBasicsProjection(
            destination_name=destination_name,
            destination_canonical_id=destination_id,
            start_date=start_date,
            end_date=end_date,
            duration_days=duration_days,
            destination_source_operation_refs=destination_source_refs,
            date_source_operation_refs=date_source_refs,
        ),
        attractions=AttractionSemanticProjection(
            preference_directions=attraction_directions,
            concrete_intents=attractions,
            exclusions=attraction_exclusions,
        ),
        dining=DiningSemanticProjection(
            preference_directions=dining_directions,
            requirements=dining_requirements,
            allergies=dining_allergies,
            avoidances=dining_avoidances,
            concrete_restaurant_intents=restaurants,
            exclusions=restaurant_exclusions,
        ),
        lodging=LodgingSemanticProjection(
            area_preferences=lodging_areas,
            hotel_quality_tier=lodging_quality,
            property_type_preferences=lodging_types,
            user_named_hotel_intents=named_hotels,
        ),
        transport_and_pace=TransportAndPaceProjection(
            transport_preferences=transport,
            pace_preferences=pace,
        ),
        constraints=constraints,
        existing_bookings=[],
        entries=state.entries,
        superseded_entries=state.superseded_entries,
        unresolved_conflicts=state.conflicts,
        invalidations=state.invalidations,
        audit_log=state.audit_log,
        confirmed_task_book_ref=confirmed_ref,
    )


def current_trip_values_override_cold_start(
    *,
    explicit_values: list[str],
    cold_start_values: list[str],
) -> list[str]:
    """Freeze precedence: explicit current-trip values replace profile defaults."""

    return list(explicit_values if explicit_values else cold_start_values)


__all__ = [
    "adapt_legacy_semantic_state",
    "current_trip_values_override_cold_start",
]
