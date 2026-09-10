"""V3-42 contracts for scoped modification and dependency-aware replanning."""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from uuid import UUID

from pydantic import AwareDatetime, ConfigDict, Field, model_validator

from backend.agent.semantic_operations import SemanticImpactScope, SemanticTarget
from backend.agent.state_merge import RecomputeDomain
from backend.contracts.base import ContractModel
from backend.contracts.common import ShortText


class ImmutableModificationModel(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PlanArtifactKind(StrEnum):
    CANDIDATES = "candidates"
    PROVIDER_FACTS = "provider_facts"
    LODGING_STRATEGY = "lodging_strategy"
    HOTEL = "hotel"
    TASK_BOOK = "task_book"
    SCHEDULE_DAY = "schedule_day"
    MEAL_WINDOW = "meal_window"
    ROUTES = "routes"
    COST = "cost"
    VALIDATION = "validation"
    MAP = "map"


class PlanDayDependency(ImmutableModificationModel):
    day_number: int = Field(ge=1, le=5, strict=True)
    service_date: date
    activity_ids: tuple[UUID, ...] = ()
    attraction_place_ids: tuple[UUID, ...] = ()
    restaurant_place_ids: tuple[UUID, ...] = ()
    route_leg_ids: tuple[UUID, ...] = ()

    @model_validator(mode="after")
    def references_are_unique(self) -> PlanDayDependency:
        for values in (
            self.activity_ids,
            self.attraction_place_ids,
            self.restaurant_place_ids,
            self.route_leg_ids,
        ):
            if len(set(values)) != len(values):
                raise ValueError("plan day dependency references must be unique")
        return self


class PlanDependencyIndex(ImmutableModificationModel):
    trip_id: UUID
    plan_version_id: UUID
    city_id: str = Field(min_length=1)
    days: tuple[PlanDayDependency, ...] = Field(min_length=1, max_length=5)
    hotel_place_id: UUID | None = None

    @model_validator(mode="after")
    def days_are_ordered_and_unique(self) -> PlanDependencyIndex:
        expected = tuple(range(1, len(self.days) + 1))
        if tuple(day.day_number for day in self.days) != expected:
            raise ValueError("plan dependency days must be complete and ordered")
        if len({day.service_date for day in self.days}) != len(self.days):
            raise ValueError("plan dependency dates must be unique")
        return self


class PendingPlanModification(ImmutableModificationModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        json_schema_extra={"x-travel-pending-plan-modification": True},
    )

    modification_id: UUID
    trip_id: UUID
    generation_id: UUID
    source_message_id: UUID
    base_state_version: int = Field(ge=0, strict=True)
    base_plan_version_id: UUID
    base_confirmed_version_id: UUID | None = None
    targets: tuple[SemanticTarget, ...] = Field(min_length=1)
    impact_scopes: tuple[SemanticImpactScope, ...] = Field(min_length=1)
    recompute_domains: tuple[RecomputeDomain, ...] = Field(min_length=1)
    affected_artifacts: tuple[PlanArtifactKind, ...] = Field(min_length=1)
    affected_day_numbers: tuple[int, ...] = ()
    affected_item_ids: tuple[UUID, ...] = ()
    preserved_day_numbers: tuple[int, ...] = ()
    preserved_hotel: bool
    preserved_attraction_place_ids: tuple[UUID, ...] = ()
    global_replan_required: bool
    changed_summary: ShortText
    preserved_summary: ShortText
    scope_reason: ShortText
    expanded_scope_reason: ShortText | None = None
    dependency_index: PlanDependencyIndex
    created_at: AwareDatetime

    @model_validator(mode="after")
    def scope_is_consistent_with_the_base_plan(self) -> PendingPlanModification:
        if (
            self.dependency_index.trip_id != self.trip_id
            or self.dependency_index.plan_version_id != self.base_plan_version_id
        ):
            raise ValueError("modification dependency index must match its base plan")
        if len(self.targets) != len(self.impact_scopes):
            raise ValueError("each modification target requires one impact scope")
        for values in (
            self.recompute_domains,
            self.affected_artifacts,
            self.affected_day_numbers,
            self.affected_item_ids,
            self.preserved_day_numbers,
            self.preserved_attraction_place_ids,
        ):
            if len(set(values)) != len(values):
                raise ValueError("modification scope references must be unique")
        known_days = {day.day_number for day in self.dependency_index.days}
        if not set(self.affected_day_numbers) <= known_days:
            raise ValueError("affected days must exist in the base plan")
        if not set(self.preserved_day_numbers) <= known_days:
            raise ValueError("preserved days must exist in the base plan")
        if set(self.affected_day_numbers) & set(self.preserved_day_numbers):
            raise ValueError("one day cannot be both affected and preserved")
        if self.global_replan_required and (
            set(self.affected_day_numbers) != known_days or self.preserved_day_numbers
        ):
            raise ValueError("global replanning must affect every day")
        if SemanticTarget.LODGING_PREFERENCES in self.targets:
            required_lodging_artifacts = {
                PlanArtifactKind.TASK_BOOK,
                PlanArtifactKind.HOTEL,
                PlanArtifactKind.SCHEDULE_DAY,
                PlanArtifactKind.ROUTES,
            }
            if not required_lodging_artifacts <= set(self.affected_artifacts):
                raise ValueError(
                    "lodging modification must update the task book, hotel, daily boundaries, "
                    "and routes"
                )
        return self


V3_PLAN_MODIFICATION_CONTRACTS: tuple[type[ContractModel], ...] = (
    PlanDayDependency,
    PlanDependencyIndex,
    PendingPlanModification,
)
