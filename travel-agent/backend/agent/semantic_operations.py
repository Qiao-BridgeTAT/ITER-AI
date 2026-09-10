"""Validated intermediate operations between language understanding and TripState."""

from __future__ import annotations

import json
from datetime import date
from enum import StrEnum
from typing import Annotated, Any, Literal, TypeAlias
from uuid import UUID

from pydantic import ConfigDict, Field, ValidationInfo, model_validator

from backend.contracts.base import ContractModel
from backend.contracts.common import NonEmptyText, ShortText
from backend.contracts.enums import (
    AttractionIntent,
    Confidence,
    ConstraintKind,
    EvidenceSource,
    MobilityTolerance,
    RestaurantIntent,
    TransportMode,
)
from backend.contracts.trip_setup import one_year_after


class ImmutableSemanticModel(ContractModel):
    """Deep building block for semantic contracts that must not change after validation."""

    model_config = ConfigDict(frozen=True)


class SemanticOperationKind(StrEnum):
    SET = "set"
    APPEND = "append"
    DELETE = "delete"
    NEGATE = "negate"
    OVERRIDE = "override"
    CONFIRM = "confirm"
    REOPEN = "reopen"


class SemanticTarget(StrEnum):
    DESTINATION = "trip_identity.destination"
    DATE_RANGE = "trip_identity.date_range"
    ATTRACTION_INTENTS = "discovery.attraction_intents"
    DINING_PREFERENCES = "trip_preferences.dining"
    LODGING_PREFERENCES = "planning.lodging"
    TRANSPORT_PREFERENCES = "trip_preferences.transport"
    PACE_PREFERENCES = "trip_preferences.pace"
    EXPERIENCE_PREFERENCES = "trip_preferences.experience_style"
    SPECIAL_CONSTRAINTS = "constraints.special"
    TASK_BOOK_CONFIRMATION = "planning.task_book_confirmation"
    PLAN_CONFIRMATION = "planning.plan_confirmation"


class SemanticPersistenceScope(StrEnum):
    CURRENT_TRIP = "current_trip"
    PERSONAL_DEFAULTS = "personal_defaults"


class SemanticImpactKind(StrEnum):
    WHOLE_TRIP = "whole_trip"
    SPECIFIC_DAY = "specific_day"
    SPECIFIC_ITEM = "specific_item"


class SemanticImpactScope(ImmutableSemanticModel):
    """Where an operation applies inside the selected persistence destination."""

    kind: SemanticImpactKind
    day_number: int | None = Field(default=None, ge=1, le=5, strict=True)
    item_id: UUID | None = None

    @model_validator(mode="after")
    def reference_matches_kind(self) -> SemanticImpactScope:
        if self.kind is SemanticImpactKind.WHOLE_TRIP:
            if self.day_number is not None or self.item_id is not None:
                raise ValueError("whole-trip impact cannot contain a day or item reference")
        elif self.kind is SemanticImpactKind.SPECIFIC_DAY:
            if self.day_number is None or self.item_id is not None:
                raise ValueError("day impact requires only day_number")
        elif self.day_number is not None or self.item_id is None:
            raise ValueError("item impact requires only item_id")
        return self


def whole_trip_impact() -> SemanticImpactScope:
    return SemanticImpactScope(kind=SemanticImpactKind.WHOLE_TRIP)


class DiningPreferenceKind(StrEnum):
    CUISINE = "cuisine"
    DIETARY_REQUIREMENT = "dietary_requirement"
    ALLERGY = "allergy"
    AVOIDANCE = "avoidance"
    SPECIFIC_RESTAURANT = "specific_restaurant"
    OPEN_TO_ANY = "open_to_any"


class LodgingPreferenceKind(StrEnum):
    AREA = "area"
    TRANSIT_NODE = "transit_node"
    SPECIFIC_HOTEL = "specific_hotel"
    QUALITY = "quality"
    PRICE = "price"
    OTHER = "other"


class TransportPreferenceKind(StrEnum):
    WALKING_TOLERANCE = "walking_tolerance"
    BIKE_TOLERANCE = "bike_tolerance"
    TRANSIT_TAXI_BALANCE = "transit_taxi_balance"
    PREFERRED_MODE = "preferred_mode"
    OTHER = "other"


class ExperiencePreferenceKind(StrEnum):
    CLASSIC_TO_NICHE = "classic_to_niche"
    BREADTH_TO_DEPTH = "breadth_to_depth"
    CITY_THEME = "city_theme"
    OTHER = "other"


class OperationEvidence(ImmutableSemanticModel):
    """Normalized provenance only; original user text is deliberately excluded."""

    source: EvidenceSource
    source_trip_id: UUID
    source_message_id: UUID | None = None
    source_attachment_id: UUID | None = None
    source_claim_id: ShortText | None = None

    @model_validator(mode="after")
    def source_references_are_consistent(self) -> OperationEvidence:
        if self.source is EvidenceSource.CARD:
            if self.source_message_id is None or self.source_attachment_id is None:
                raise ValueError("card evidence requires message and attachment references")
        elif self.source is EvidenceSource.DIALOGUE:
            if self.source_message_id is None:
                raise ValueError("dialogue evidence requires a message reference")
            if self.source_attachment_id is not None:
                raise ValueError("dialogue evidence cannot contain an attachment reference")
        else:
            if self.source_attachment_id is not None:
                raise ValueError("only card evidence can contain an attachment reference")
            if self.source_claim_id is not None:
                raise ValueError("only card or dialogue evidence can reference a semantic claim")
        return self


class DestinationValue(ImmutableSemanticModel):
    """Registry identity stays open-ended so cities are data, not code branches."""

    city_id: NonEmptyText
    display_name: NonEmptyText


class SemanticDateRange(ImmutableSemanticModel):
    start_date: date
    end_date: date

    @model_validator(mode="after")
    def is_a_future_trip_of_one_to_five_days(self, info: ValidationInfo) -> SemanticDateRange:
        day_count = (self.end_date - self.start_date).days + 1
        if not 1 <= day_count <= 5:
            raise ValueError("semantic trip date range must be between 1 and 5 days")
        if (info.context or {}).get("restore_historical_semantic_state", False):
            return self
        context_today = info.context.get("today") if info.context else None
        if not isinstance(context_today, date):
            raise ValueError("semantic date validation requires an explicit business date")
        today = context_today
        if self.start_date < today:
            raise ValueError("semantic trip start_date cannot be in the past")
        if self.start_date > one_year_after(today):
            raise ValueError("semantic trip start_date cannot be more than one year ahead")
        return self


class AttractionIntentValue(ImmutableSemanticModel):
    place_id: UUID
    intent: AttractionIntent
    place_name: NonEmptyText | None = None


class DiningPreferenceValue(ImmutableSemanticModel):
    kind: DiningPreferenceKind
    value: ShortText | None = None
    place_id: UUID | None = None
    restaurant_intent: RestaurantIntent | None = None

    @model_validator(mode="after")
    def kind_matches_value(self) -> DiningPreferenceValue:
        if self.kind is DiningPreferenceKind.OPEN_TO_ANY:
            if (
                self.value is not None
                or self.place_id is not None
                or self.restaurant_intent is not None
            ):
                raise ValueError(
                    "open_to_any cannot contain a value, place reference or restaurant intent"
                )
            return self
        if self.value is None:
            raise ValueError("a concrete dining preference requires a value")
        if self.kind is DiningPreferenceKind.SPECIFIC_RESTAURANT:
            if self.restaurant_intent is None:
                raise ValueError("a specific restaurant requires an explicit restaurant intent")
        elif self.place_id is not None or self.restaurant_intent is not None:
            raise ValueError(
                "only a specific restaurant can contain a place reference or restaurant intent"
            )
        return self


class LodgingPreferenceValue(ImmutableSemanticModel):
    kind: LodgingPreferenceKind
    value: ShortText
    place_id: UUID | None = None

    @model_validator(mode="after")
    def place_reference_matches_kind(self) -> LodgingPreferenceValue:
        if self.kind is not LodgingPreferenceKind.SPECIFIC_HOTEL and self.place_id is not None:
            raise ValueError("only a specific hotel can contain a place reference")
        return self


class TransportPreferenceValue(ImmutableSemanticModel):
    kind: TransportPreferenceKind
    mobility_tolerance: MobilityTolerance | None = None
    transit_taxi_level: int | None = Field(default=None, ge=1, le=5, strict=True)
    preferred_mode: TransportMode | None = None
    note: ShortText | None = None

    @model_validator(mode="after")
    def kind_has_exactly_one_matching_value(self) -> TransportPreferenceValue:
        fields = {
            "mobility_tolerance": self.mobility_tolerance,
            "transit_taxi_level": self.transit_taxi_level,
            "preferred_mode": self.preferred_mode,
            "note": self.note,
        }
        expected = {
            TransportPreferenceKind.WALKING_TOLERANCE: "mobility_tolerance",
            TransportPreferenceKind.BIKE_TOLERANCE: "mobility_tolerance",
            TransportPreferenceKind.TRANSIT_TAXI_BALANCE: "transit_taxi_level",
            TransportPreferenceKind.PREFERRED_MODE: "preferred_mode",
            TransportPreferenceKind.OTHER: "note",
        }[self.kind]
        present = {name for name, value in fields.items() if value is not None}
        if present != {expected}:
            raise ValueError("transport preference kind requires exactly its matching value")
        return self


class PacePreferenceValue(ImmutableSemanticModel):
    """Absolute level or relative change; 1 is intensive and 5 is relaxed."""

    level: int | None = Field(default=None, ge=1, le=5, strict=True)
    relative_adjustment: int | None = Field(default=None, ge=-2, le=2, strict=True)

    @model_validator(mode="after")
    def has_one_non_neutral_value(self) -> PacePreferenceValue:
        if (self.level is None) == (self.relative_adjustment is None):
            raise ValueError("pace preference requires exactly one level or relative adjustment")
        if self.relative_adjustment == 0:
            raise ValueError("pace relative adjustment cannot be neutral")
        return self


class ExperiencePreferenceValue(ImmutableSemanticModel):
    """Absolute level or relative change on one stable experience dimension."""

    kind: ExperiencePreferenceKind
    level: int | None = Field(default=None, ge=1, le=5, strict=True)
    relative_adjustment: int | None = Field(default=None, ge=-2, le=2, strict=True)
    note: ShortText | None = None
    theme_id: ShortText | None = None

    @model_validator(mode="after")
    def value_matches_kind(self) -> ExperiencePreferenceValue:
        if self.kind is ExperiencePreferenceKind.CITY_THEME:
            if (
                self.theme_id is None
                or self.note is None
                or self.level is not None
                or self.relative_adjustment is not None
            ):
                raise ValueError("city theme preference requires only theme_id and note")
            return self
        if self.kind is ExperiencePreferenceKind.OTHER:
            if (
                self.note is None
                or self.theme_id is not None
                or self.level is not None
                or self.relative_adjustment is not None
            ):
                raise ValueError("other experience preference requires only a note")
            return self
        if self.note is not None or self.theme_id is not None:
            raise ValueError("scaled experience preference cannot contain a note")
        if (self.level is None) == (self.relative_adjustment is None):
            raise ValueError(
                "experience preference requires exactly one level or relative adjustment"
            )
        if self.relative_adjustment == 0:
            raise ValueError("experience relative adjustment cannot be neutral")
        return self


class ConstraintValue(ImmutableSemanticModel):
    kind: ConstraintKind
    description: ShortText


class OperationMetadata(ImmutableSemanticModel):
    operation_id: UUID
    trip_id: UUID
    operation: SemanticOperationKind
    evidence: OperationEvidence
    confidence: Confidence
    persistence_scope: SemanticPersistenceScope = SemanticPersistenceScope.CURRENT_TRIP
    impact_scope: SemanticImpactScope = Field(default_factory=whole_trip_impact)


_VALUE_WRITES = {
    SemanticOperationKind.SET,
    SemanticOperationKind.APPEND,
    SemanticOperationKind.OVERRIDE,
}
_VALUE_REMOVALS = {
    SemanticOperationKind.DELETE,
    SemanticOperationKind.NEGATE,
}
_COLLECTION_OPERATIONS = _VALUE_WRITES | _VALUE_REMOVALS


class DestinationOperation(OperationMetadata):
    target: Literal[SemanticTarget.DESTINATION]
    value: DestinationValue | None = None

    @model_validator(mode="after")
    def operation_matches_scalar(self) -> DestinationOperation:
        _validate_scalar_operation(self.operation, self.value, "destination")
        _require_current_trip(self.persistence_scope, "destination")
        _require_whole_trip(self.impact_scope, "destination")
        return self


class DateRangeOperation(OperationMetadata):
    target: Literal[SemanticTarget.DATE_RANGE]
    value: SemanticDateRange | None = None

    @model_validator(mode="after")
    def operation_matches_scalar(self) -> DateRangeOperation:
        _validate_scalar_operation(self.operation, self.value, "date range")
        _require_current_trip(self.persistence_scope, "date range")
        _require_whole_trip(self.impact_scope, "date range")
        return self


class AttractionIntentOperation(OperationMetadata):
    target: Literal[SemanticTarget.ATTRACTION_INTENTS]
    value: AttractionIntentValue

    @model_validator(mode="after")
    def operation_matches_collection(self) -> AttractionIntentOperation:
        _validate_collection_operation(self.operation, "attraction intent")
        _require_current_trip(self.persistence_scope, "attraction intent")
        if (
            self.impact_scope.kind is not SemanticImpactKind.SPECIFIC_ITEM
            or self.impact_scope.item_id != self.value.place_id
        ):
            raise ValueError("attraction intent must affect its referenced place")
        if (
            self.operation is SemanticOperationKind.NEGATE
            and self.value.intent is not AttractionIntent.AVOID
        ):
            raise ValueError("negated attraction intent must use avoid")
        return self


class DiningPreferenceOperation(OperationMetadata):
    target: Literal[SemanticTarget.DINING_PREFERENCES]
    value: DiningPreferenceValue

    @model_validator(mode="after")
    def operation_matches_collection(self) -> DiningPreferenceOperation:
        _validate_collection_operation(self.operation, "dining preference")
        if self.operation is SemanticOperationKind.NEGATE:
            valid_negation = self.value.kind is DiningPreferenceKind.AVOIDANCE or (
                self.value.kind is DiningPreferenceKind.SPECIFIC_RESTAURANT
                and self.value.restaurant_intent is RestaurantIntent.AVOID
            )
            if not valid_negation:
                raise ValueError(
                    "negated dining preference must use avoidance or an avoided restaurant"
                )
        if (
            self.persistence_scope is SemanticPersistenceScope.PERSONAL_DEFAULTS
            and self.value.kind
            in {
                DiningPreferenceKind.SPECIFIC_RESTAURANT,
                DiningPreferenceKind.OPEN_TO_ANY,
            }
        ):
            raise ValueError("specific or open dining choices only belong to the current trip")
        if self.value.kind is DiningPreferenceKind.SPECIFIC_RESTAURANT:
            if self.value.place_id is None:
                if self.impact_scope.kind is not SemanticImpactKind.WHOLE_TRIP:
                    raise ValueError(
                        "an unresolved restaurant name must remain a whole-trip recall clue"
                    )
            elif (
                self.impact_scope.kind is not SemanticImpactKind.SPECIFIC_ITEM
                or self.impact_scope.item_id != self.value.place_id
            ):
                raise ValueError("resolved restaurant preference must affect its place")
        elif self.impact_scope.kind is SemanticImpactKind.SPECIFIC_ITEM:
            raise ValueError("generic dining preference cannot affect a specific item")
        _validate_personal_default_provenance(self)
        return self


class LodgingPreferenceOperation(OperationMetadata):
    target: Literal[SemanticTarget.LODGING_PREFERENCES]
    value: LodgingPreferenceValue

    @model_validator(mode="after")
    def operation_matches_collection(self) -> LodgingPreferenceOperation:
        _validate_collection_operation(self.operation, "lodging preference")
        if (
            self.persistence_scope is SemanticPersistenceScope.PERSONAL_DEFAULTS
            and self.value.kind
            in {
                LodgingPreferenceKind.AREA,
                LodgingPreferenceKind.TRANSIT_NODE,
                LodgingPreferenceKind.SPECIFIC_HOTEL,
            }
        ):
            raise ValueError("trip-specific lodging places cannot become personal defaults")
        if self.value.kind is LodgingPreferenceKind.SPECIFIC_HOTEL:
            if (
                self.impact_scope.kind is not SemanticImpactKind.SPECIFIC_ITEM
                or self.impact_scope.item_id != self.value.place_id
            ):
                raise ValueError("specific hotel preference must affect its place")
        elif self.impact_scope.kind is SemanticImpactKind.SPECIFIC_ITEM:
            raise ValueError("generic lodging preference cannot affect a specific item")
        _validate_personal_default_provenance(self)
        return self


class TransportPreferenceOperation(OperationMetadata):
    target: Literal[SemanticTarget.TRANSPORT_PREFERENCES]
    value: TransportPreferenceValue

    @model_validator(mode="after")
    def operation_matches_collection(self) -> TransportPreferenceOperation:
        _validate_collection_operation(self.operation, "transport preference")
        if self.impact_scope.kind is SemanticImpactKind.SPECIFIC_ITEM:
            raise ValueError("transport preference can affect a day or the whole trip")
        _validate_personal_default_provenance(self)
        return self


class PacePreferenceOperation(OperationMetadata):
    target: Literal[SemanticTarget.PACE_PREFERENCES]
    value: PacePreferenceValue

    @model_validator(mode="after")
    def operation_matches_collection(self) -> PacePreferenceOperation:
        _validate_collection_operation(self.operation, "pace preference")
        if self.impact_scope.kind is SemanticImpactKind.SPECIFIC_ITEM:
            raise ValueError("pace preference can affect a day or the whole trip")
        _validate_personal_default_provenance(self)
        return self


class ExperiencePreferenceOperation(OperationMetadata):
    target: Literal[SemanticTarget.EXPERIENCE_PREFERENCES]
    value: ExperiencePreferenceValue

    @model_validator(mode="after")
    def operation_matches_collection(self) -> ExperiencePreferenceOperation:
        _validate_collection_operation(self.operation, "experience preference")
        if self.impact_scope.kind is SemanticImpactKind.SPECIFIC_ITEM:
            raise ValueError("experience preference can affect a day or the whole trip")
        _validate_personal_default_provenance(self)
        return self


class ConstraintOperation(OperationMetadata):
    target: Literal[SemanticTarget.SPECIAL_CONSTRAINTS]
    value: ConstraintValue

    @model_validator(mode="after")
    def operation_matches_collection(self) -> ConstraintOperation:
        _validate_collection_operation(self.operation, "special constraint")
        _require_current_trip(self.persistence_scope, "special constraint")
        return self


class TaskBookConfirmationOperation(OperationMetadata):
    target: Literal[SemanticTarget.TASK_BOOK_CONFIRMATION]
    value: None = None

    @model_validator(mode="after")
    def operation_matches_confirmation(self) -> TaskBookConfirmationOperation:
        _validate_confirmation_operation(self)
        return self


class PlanConfirmationOperation(OperationMetadata):
    target: Literal[SemanticTarget.PLAN_CONFIRMATION]
    value: None = None

    @model_validator(mode="after")
    def operation_matches_confirmation(self) -> PlanConfirmationOperation:
        _validate_confirmation_operation(self)
        return self


SemanticOperation: TypeAlias = Annotated[
    DestinationOperation
    | DateRangeOperation
    | AttractionIntentOperation
    | DiningPreferenceOperation
    | LodgingPreferenceOperation
    | TransportPreferenceOperation
    | PacePreferenceOperation
    | ExperiencePreferenceOperation
    | ConstraintOperation
    | TaskBookConfirmationOperation
    | PlanConfirmationOperation,
    Field(discriminator="target"),
]


class SemanticOperationBatch(ImmutableSemanticModel):
    """Atomic proposed effects from one semantic input, before state merging."""

    trip_id: UUID
    operations: tuple[SemanticOperation, ...] = Field(max_length=50)

    @model_validator(mode="after")
    def operations_are_owned_and_non_conflicting(
        self, info: ValidationInfo
    ) -> SemanticOperationBatch:
        operation_ids = [operation.operation_id for operation in self.operations]
        if len(set(operation_ids)) != len(operation_ids):
            raise ValueError("semantic operation_id must not contain duplicates")
        for operation in self.operations:
            if operation.trip_id != self.trip_id:
                raise ValueError("semantic operation belongs to another trip")
            if operation.evidence.source_trip_id != self.trip_id:
                raise ValueError("semantic operation evidence belongs to another trip")
            if operation.persistence_scope is SemanticPersistenceScope.PERSONAL_DEFAULTS and not (
                info.context or {}
            ).get("allow_personal_defaults_write", False):
                raise ValueError("personal defaults require application-authorized persistence")

        subjects: dict[tuple[str, str, str, str], UUID] = {}
        dining_scopes: dict[str, set[str]] = {}
        for operation in self.operations:
            key = _subject_key(operation)
            if key in subjects:
                raise ValueError("semantic operations contain mutually exclusive effects")
            subjects[key] = operation.operation_id
            if isinstance(operation, DiningPreferenceOperation):
                dining_scopes.setdefault(_scope_key(operation), set()).add(key[3])
        if any("*" in values and len(values) > 1 for values in dining_scopes.values()):
            raise ValueError("open dining preference conflicts with concrete preferences")
        return self

    def normalized_effects(self) -> list[dict[str, Any]]:
        """Compare click and text semantics while deliberately ignoring provenance."""

        effects = [semantic_effect(operation) for operation in self.operations]
        return sorted(
            effects,
            key=lambda effect: json.dumps(effect, ensure_ascii=False, sort_keys=True),
        )


def semantic_effect(operation: SemanticOperation) -> dict[str, Any]:
    value = getattr(operation, "value", None)
    return {
        "operation": operation.operation.value,
        "target": operation.target.value,
        "persistence_scope": operation.persistence_scope.value,
        "impact_scope": operation.impact_scope.model_dump(mode="json"),
        "value": value.model_dump(mode="json") if value is not None else None,
    }


def _validate_scalar_operation(
    operation: SemanticOperationKind,
    value: ContractModel | None,
    label: str,
) -> None:
    allowed = {
        SemanticOperationKind.SET,
        SemanticOperationKind.DELETE,
        SemanticOperationKind.NEGATE,
        SemanticOperationKind.OVERRIDE,
    }
    if operation not in allowed:
        raise ValueError(f"{label} does not support {operation.value}")
    if operation in {SemanticOperationKind.SET, SemanticOperationKind.OVERRIDE} and value is None:
        raise ValueError(f"{label} write requires a value")
    if operation is SemanticOperationKind.NEGATE and value is None:
        raise ValueError(f"{label} negation requires the rejected value")
    if operation is SemanticOperationKind.DELETE and value is not None:
        raise ValueError(f"{label} deletion acts on the current scalar and has no value")


def _validate_collection_operation(operation: SemanticOperationKind, label: str) -> None:
    if operation not in _COLLECTION_OPERATIONS:
        raise ValueError(f"{label} does not support {operation.value}")


def _validate_confirmation_operation(operation: OperationMetadata) -> None:
    if operation.operation not in {
        SemanticOperationKind.CONFIRM,
        SemanticOperationKind.REOPEN,
    }:
        raise ValueError("confirmation targets only support confirm or reopen")
    _require_current_trip(operation.persistence_scope, "confirmation")
    _require_whole_trip(operation.impact_scope, "confirmation")
    if operation.evidence.source not in {EvidenceSource.CARD, EvidenceSource.DIALOGUE}:
        raise ValueError("confirmation requires explicit user evidence")
    if operation.confidence is not Confidence.HIGH:
        raise ValueError("confirmation requires high-confidence evidence")


def _validate_personal_default_provenance(operation: OperationMetadata) -> None:
    if operation.persistence_scope is not SemanticPersistenceScope.PERSONAL_DEFAULTS:
        return
    _require_whole_trip(operation.impact_scope, "personal default")
    if operation.evidence.source not in {
        EvidenceSource.COLD_START,
        EvidenceSource.CARD,
        EvidenceSource.DIALOGUE,
    }:
        raise ValueError("personal defaults require explicit user evidence")
    if operation.confidence is not Confidence.HIGH:
        raise ValueError("personal defaults require high-confidence evidence")


def _require_current_trip(scope: SemanticPersistenceScope, label: str) -> None:
    if scope is not SemanticPersistenceScope.CURRENT_TRIP:
        raise ValueError(f"{label} can only affect the current trip")


def _require_whole_trip(scope: SemanticImpactScope, label: str) -> None:
    if scope.kind is not SemanticImpactKind.WHOLE_TRIP:
        raise ValueError(f"{label} must affect the whole trip")


def _scope_key(operation: SemanticOperation) -> str:
    return json.dumps(
        {
            "persistence": operation.persistence_scope.value,
            "impact": operation.impact_scope.model_dump(mode="json"),
        },
        sort_keys=True,
    )


def _subject_key(operation: SemanticOperation) -> tuple[str, str, str, str]:
    value = getattr(operation, "value", None)
    if isinstance(value, AttractionIntentValue):
        subject = str(value.place_id)
    elif isinstance(value, DiningPreferenceValue):
        if value.kind is DiningPreferenceKind.OPEN_TO_ANY:
            subject = "*"
        else:
            subject = f"{value.place_id or ''}:{value.value or ''}".casefold()
    elif isinstance(value, LodgingPreferenceValue):
        subject = f"{value.kind.value}:{value.place_id or ''}:{value.value}".casefold()
    elif isinstance(value, TransportPreferenceValue):
        subject = value.kind.value
    elif isinstance(value, PacePreferenceValue):
        subject = "pace"
    elif isinstance(value, ExperiencePreferenceValue):
        subject = (
            f"{value.kind.value}:{value.theme_id}"
            if value.kind is ExperiencePreferenceKind.CITY_THEME
            else value.kind.value
        )
    elif isinstance(value, ConstraintValue):
        subject = f"{value.kind.value}:{value.description}".casefold()
    else:
        # Scalar and confirmation targets can have at most one effect per input.
        subject = "scalar"
    return (
        operation.target.value,
        operation.persistence_scope.value,
        json.dumps(operation.impact_scope.model_dump(mode="json"), sort_keys=True),
        subject,
    )


def stable_effect_json(operation: SemanticOperation) -> str:
    """Deterministic representation for audit fingerprints, never raw input text."""

    return json.dumps(semantic_effect(operation), ensure_ascii=False, sort_keys=True)


def semantic_operation_key(operation: SemanticOperation) -> str:
    """Stable state-address key including persistence, impact and collection subject."""

    return json.dumps(_subject_key(operation), ensure_ascii=False, separators=(",", ":"))


def semantic_operation_fingerprint(operation: SemanticOperation) -> str:
    """Full normalized fingerprint for idempotency and operation-ID reuse protection."""

    return json.dumps(
        operation.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
