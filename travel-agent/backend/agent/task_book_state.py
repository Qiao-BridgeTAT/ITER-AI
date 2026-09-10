"""Stable V2 task-book records embedded in the semantic trip state."""

from __future__ import annotations

from enum import StrEnum
from uuid import UUID

from pydantic import ConfigDict, Field, model_validator

from backend.agent.semantic_operations import (
    DestinationValue,
    DiningPreferenceKind,
    LodgingPreferenceKind,
    SemanticDateRange,
    SemanticImpactScope,
    SemanticPersistenceScope,
    SemanticTarget,
)
from backend.contracts.base import ContractModel
from backend.contracts.common import ShortText
from backend.contracts.enums import AttractionIntent, ConstraintKind, RestaurantIntent


class ImmutableTaskBookModel(ContractModel):
    model_config = ConfigDict(frozen=True)


class SemanticTaskBookStatus(StrEnum):
    PENDING = "pending"
    CONFIRMED = "confirmed"


class TaskBookPreferenceItem(ImmutableTaskBookModel):
    source_operation_id: UUID
    target: SemanticTarget
    persistence_scope: SemanticPersistenceScope
    impact_scope: SemanticImpactScope
    summary: ShortText

    @model_validator(mode="after")
    def target_is_a_preference(self) -> TaskBookPreferenceItem:
        if self.target not in {
            SemanticTarget.DINING_PREFERENCES,
            SemanticTarget.LODGING_PREFERENCES,
            SemanticTarget.TRANSPORT_PREFERENCES,
            SemanticTarget.PACE_PREFERENCES,
            SemanticTarget.EXPERIENCE_PREFERENCES,
        }:
            raise ValueError("task-book preference item must reference a preference target")
        return self


class TaskBookAttractionItem(ImmutableTaskBookModel):
    source_operation_id: UUID
    place_id: UUID
    place_name: ShortText | None = None
    intent: AttractionIntent


class TaskBookRestaurantItem(ImmutableTaskBookModel):
    source_operation_id: UUID
    name: ShortText
    place_id: UUID | None = None
    kind: DiningPreferenceKind = DiningPreferenceKind.SPECIFIC_RESTAURANT
    intent: RestaurantIntent

    @model_validator(mode="after")
    def only_specific_restaurants_are_materialized(self) -> TaskBookRestaurantItem:
        if self.kind is not DiningPreferenceKind.SPECIFIC_RESTAURANT:
            raise ValueError("task-book restaurant item must be a specific restaurant")
        return self


class TaskBookLodgingItem(ImmutableTaskBookModel):
    source_operation_id: UUID
    kind: LodgingPreferenceKind
    value: ShortText
    place_id: UUID | None = None


class TaskBookConstraintItem(ImmutableTaskBookModel):
    source_operation_id: UUID
    kind: ConstraintKind
    description: ShortText


class TaskBookAssumptionItem(ImmutableTaskBookModel):
    assumption_id: UUID
    description: ShortText
    reason: ShortText


class SemanticTaskBook(ImmutableTaskBookModel):
    task_book_id: UUID
    trip_id: UUID
    revision: int = Field(ge=1, strict=True)
    source_state_version: int = Field(ge=1, strict=True)
    published_state_version: int = Field(ge=1, strict=True)
    destination: DestinationValue
    date_range: SemanticDateRange
    preferences: tuple[TaskBookPreferenceItem, ...] = ()
    attraction_intents: tuple[TaskBookAttractionItem, ...] = ()
    important_restaurants: tuple[TaskBookRestaurantItem, ...] = ()
    lodging_direction: tuple[TaskBookLodgingItem, ...] = ()
    key_constraints: tuple[TaskBookConstraintItem, ...] = ()
    tradeoffs: tuple[ShortText, ...] = ()
    omitted_strong_desires: tuple[ShortText, ...] = ()
    assumptions: tuple[TaskBookAssumptionItem, ...] = ()
    source_operation_ids: tuple[UUID, ...] = Field(min_length=2)
    status: SemanticTaskBookStatus = SemanticTaskBookStatus.PENDING
    confirmation_operation_id: UUID | None = None
    confirmed_state_version: int | None = Field(default=None, ge=1, strict=True)

    @model_validator(mode="after")
    def versions_references_and_status_are_consistent(self) -> SemanticTaskBook:
        if self.published_state_version <= self.source_state_version:
            raise ValueError("task book must be published after its source state")
        if self.revision != self.published_state_version:
            raise ValueError("task-book revision must use its published state version")
        if len(set(self.source_operation_ids)) != len(self.source_operation_ids):
            raise ValueError("task-book source operation IDs must be unique")
        if len(set(self.tradeoffs)) != len(self.tradeoffs):
            raise ValueError("task-book tradeoffs must be unique")
        if len(set(self.omitted_strong_desires)) != len(self.omitted_strong_desires):
            raise ValueError("task-book omitted desires must be unique")
        if self.status is SemanticTaskBookStatus.PENDING:
            if (
                self.confirmation_operation_id is not None
                or self.confirmed_state_version is not None
            ):
                raise ValueError("pending task book cannot contain confirmation references")
        elif self.confirmation_operation_id is None or self.confirmed_state_version is None:
            raise ValueError("confirmed task book requires confirmation references")
        elif self.confirmed_state_version <= self.published_state_version:
            raise ValueError("task book must be confirmed after publication")
        return self
