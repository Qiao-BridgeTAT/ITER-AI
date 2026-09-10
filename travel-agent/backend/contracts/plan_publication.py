"""V3-41 contracts for publishing one complete, validated plan atomically."""

from __future__ import annotations

from typing import Any, Literal
from uuid import UUID

from pydantic import AwareDatetime, ConfigDict, Field, model_validator

from backend.contracts.base import ContractModel
from backend.contracts.candidate_recall import RecalledCandidate
from backend.contracts.commands import IdempotencyKey
from backend.contracts.cost_estimation import CostCoverageStatus, TripCostEstimate
from backend.contracts.daily_scheduling import DailyScheduleResult, SchedulePlaceFact
from backend.contracts.enums import DataAvailability
from backend.contracts.events import MapUpdatePayload
from backend.contracts.hotel_selection import HotelDecisionStatus, HotelSelectionResult
from backend.contracts.itinerary import Assumption
from backend.contracts.itinerary_repair import ItineraryRepairResult, RepairRunStatus
from backend.contracts.itinerary_validation import (
    DailyWeatherCoverage,
    ItineraryValidationRequest,
    ItineraryValidationResult,
    ValidationIssue,
    ValidationStatus,
)

PLAN_PUBLICATION_REQUEST_SCHEMA_RULE: dict[str, Any] = {"x-travel-plan-publication-request": True}
PUBLISHED_PLAN_SCHEMA_RULE: dict[str, Any] = {
    "x-travel-published-plan": True,
}


class ImmutablePublicationModel(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PublishedPlan(ImmutablePublicationModel):
    """The sole formal plan source stored with a stable trip version.

    Absence represents empty or failed generation; availability represents partial coverage;
    TripState.phase distinguishes draft from confirmed. Only normalized selected-place,
    opening, route, hotel, weather and cost facts cross the Provider boundary.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        json_schema_extra=PUBLISHED_PLAN_SCHEMA_RULE,
    )

    plan_version_id: UUID
    publication_key: IdempotencyKey
    generation_id: UUID
    parent_version_id: UUID | None = None
    base_confirmed_version_id: UUID | None = None
    trip_id: UUID
    input_state_version: int = Field(ge=0, strict=True)
    result_contract_version: Literal["1.0.0"] | None = None
    availability: DataAvailability | None = None
    places: tuple[SchedulePlaceFact, ...] = ()
    selected_candidates: tuple[RecalledCandidate, ...] = ()
    hotel_selection: HotelSelectionResult | None = None
    weather: tuple[DailyWeatherCoverage, ...] = ()
    schedule: DailyScheduleResult
    cost_estimate: TripCostEstimate
    validation: ItineraryValidationResult
    issues: tuple[ValidationIssue, ...]
    assumptions: tuple[Assumption, ...] = ()
    map_projection: MapUpdatePayload
    published_at: AwareDatetime

    @model_validator(mode="after")
    def all_artifacts_share_one_publication_boundary(self) -> PublishedPlan:
        if (
            self.schedule.trip_id != self.trip_id
            or self.cost_estimate.trip_id != self.trip_id
            or self.validation.trip_id != self.trip_id
        ):
            raise ValueError("published plan artifacts must belong to one trip")
        if (
            self.schedule.input_state_version != self.input_state_version
            or self.cost_estimate.input_state_version != self.input_state_version
            or self.validation.input_state_version != self.input_state_version
        ):
            raise ValueError("published plan artifacts must use one input state version")
        if (
            self.schedule.task_book_id != self.cost_estimate.task_book_id
            or self.schedule.task_book_id != self.validation.task_book_id
            or self.schedule.task_book_revision != self.cost_estimate.task_book_revision
            or self.schedule.task_book_revision != self.validation.task_book_revision
        ):
            raise ValueError("published plan artifacts must use one confirmed task book")
        if (
            self.schedule.start_date != self.cost_estimate.start_date
            or self.schedule.end_date != self.cost_estimate.end_date
        ):
            raise ValueError("published schedule and cost dates must match")
        if self.validation.status is ValidationStatus.BLOCKED:
            raise ValueError("a plan with hard conflicts cannot be published")
        if self.issues != self.validation.issues:
            raise ValueError("published issues must be the final validation issues")
        if len({item.assumption_id for item in self.assumptions}) != len(self.assumptions):
            raise ValueError("published assumption IDs must be unique")

        formal_fields_present = bool(
            self.availability is not None
            or self.places
            or self.selected_candidates
            or self.hotel_selection is not None
            or self.weather
        )
        if self.result_contract_version is None:
            if formal_fields_present:
                raise ValueError("formal result fields require result_contract_version")
        else:
            self._validate_formal_result_projection()

        known_place_ids = set(self.validation_request_places)
        marker_ids = {item.place_id for item in self.map_projection.markers}
        if not marker_ids <= known_place_ids:
            raise ValueError("published map markers must reference scheduled trip places")
        if any(
            route.from_place_id not in known_place_ids or route.to_place_id not in known_place_ids
            for route in self.map_projection.routes
        ):
            raise ValueError("published map routes must reference scheduled trip places")
        if self.map_projection.selected_day_index >= len(self.schedule.days):
            raise ValueError("published map day must exist in the schedule")
        return self

    def _validate_formal_result_projection(self) -> None:
        if self.availability is None or self.availability is DataAvailability.MISSING:
            raise ValueError("a published formal result must be available or partial")
        if not self.places or self.hotel_selection is None or not self.weather:
            raise ValueError("a formal result requires places, hotel decision and daily weather")

        place_ids = [item.place_id for item in self.places]
        if len(set(place_ids)) != len(place_ids):
            raise ValueError("formal result place IDs must be unique")
        if any(item.city_id != self.schedule.city_id for item in self.places):
            raise ValueError("formal result places must belong to the schedule city")

        activity_place_ids = {
            item.place_id for day in self.schedule.days for item in day.activities
        }
        if not activity_place_ids <= set(place_ids):
            raise ValueError("every scheduled activity requires a formal place fact")

        selected_place_ids = [item.place.place_id for item in self.selected_candidates]
        if len(set(selected_place_ids)) != len(selected_place_ids):
            raise ValueError("formal result selected place IDs must be unique")
        if not set(selected_place_ids) <= activity_place_ids:
            raise ValueError("selected candidates must be used by the published schedule")
        if any(item.place.city_id != self.schedule.city_id for item in self.selected_candidates):
            raise ValueError("selected candidates must belong to the schedule city")

        hotel = self.hotel_selection
        if (
            hotel.trip_id != self.trip_id
            or hotel.city_id != self.schedule.city_id
            or hotel.input_state_version != self.input_state_version
            or hotel.check_in != self.schedule.start_date
            or hotel.check_out != self.schedule.end_date
        ):
            raise ValueError("formal result hotel must share the publication boundary")
        if hotel.night_count and hotel.decision_status is not HotelDecisionStatus.FINAL:
            raise ValueError("an overnight formal result requires one final hotel")
        if hotel.night_count and any(
            day.start_place_id != hotel.selected_hotel_place_id
            or day.end_place_id != hotel.selected_hotel_place_id
            for day in self.schedule.days
        ):
            raise ValueError("every overnight day must use the formal result hotel")

        expected_dates = tuple(day.service_date for day in self.schedule.days)
        if tuple(item.service_date for item in self.weather) != expected_dates:
            raise ValueError("formal result weather must cover each trip date in order")

        if self.availability is not self.derived_availability:
            raise ValueError("formal result availability must be derived from its items")

    @property
    def derived_availability(self) -> DataAvailability:
        assert self.hotel_selection is not None
        return derive_published_plan_availability(
            schedule=self.schedule,
            cost_estimate=self.cost_estimate,
            validation=self.validation,
            places=self.places,
            selected_candidates=self.selected_candidates,
            hotel_selection=self.hotel_selection,
            weather=self.weather,
        )

    @property
    def validation_request_places(self) -> tuple[UUID, ...]:
        """Every place referenced by the strict schedule and its daily boundaries."""

        place_ids = {item.place_id for day in self.schedule.days for item in day.activities}
        place_ids.update(day.start_place_id for day in self.schedule.days)
        place_ids.update(day.end_place_id for day in self.schedule.days)
        return tuple(place_ids)


class PlanPublicationRequest(ImmutablePublicationModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        json_schema_extra=PLAN_PUBLICATION_REQUEST_SCHEMA_RULE,
    )

    request_id: UUID
    publication_key: IdempotencyKey
    trip_id: UUID
    generation_id: UUID
    expected_state_version: int = Field(ge=0, strict=True)
    plan_version_id: UUID
    parent_version_id: UUID | None = None
    base_confirmed_version_id: UUID | None = None
    validation_request: ItineraryValidationRequest
    repair_result: ItineraryRepairResult
    selected_candidates: tuple[RecalledCandidate, ...] = ()
    assumptions: tuple[Assumption, ...] = ()
    map_projection: MapUpdatePayload

    @model_validator(mode="after")
    def request_is_a_publishable_repair_result(self) -> PlanPublicationRequest:
        if (
            self.validation_request.trip_id != self.trip_id
            or self.repair_result.trip_id != self.trip_id
            or self.validation_request.input_state_version != self.expected_state_version
            or self.repair_result.input_state_version != self.expected_state_version
            or self.repair_result.generation_id != self.generation_id
            or self.repair_result.final_validation.request_id != self.validation_request.request_id
        ):
            raise ValueError("publication must reference one current validation and repair run")
        if not self.repair_result.strict_ready or self.repair_result.status in {
            RepairRunStatus.FAILED,
            RepairRunStatus.CANCELLED,
        }:
            raise ValueError("only a non-blocked repair result can be published")
        if len({item.assumption_id for item in self.assumptions}) != len(self.assumptions):
            raise ValueError("publication assumption IDs must be unique")
        selected_place_ids = [item.place.place_id for item in self.selected_candidates]
        if len(set(selected_place_ids)) != len(selected_place_ids):
            raise ValueError("publication selected place IDs must be unique")
        schedule_place_ids = {
            item.place_id
            for day in self.repair_result.best_schedule_draft.days
            for item in day.activities
        }
        if not set(selected_place_ids) <= schedule_place_ids:
            raise ValueError("publication candidates must be present in the final schedule")
        city_id = self.validation_request.scheduling_request.city_id
        if any(item.place.city_id != city_id for item in self.selected_candidates):
            raise ValueError("publication candidates must belong to the planning city")
        return self


class PlanPublicationResult(ImmutablePublicationModel):
    request_id: UUID
    publication_key: IdempotencyKey
    trip_id: UUID
    generation_id: UUID
    plan_version_id: UUID
    state_version: int = Field(ge=1, strict=True)
    idempotent_replay: bool
    published_at: AwareDatetime


V3_PLAN_PUBLICATION_CONTRACTS: tuple[type[ContractModel], ...] = (
    PlanPublicationRequest,
    PublishedPlan,
    PlanPublicationResult,
)


def derive_published_plan_availability(
    *,
    schedule: DailyScheduleResult,
    cost_estimate: TripCostEstimate,
    validation: ItineraryValidationResult,
    places: tuple[SchedulePlaceFact, ...],
    selected_candidates: tuple[RecalledCandidate, ...],
    hotel_selection: HotelSelectionResult,
    weather: tuple[DailyWeatherCoverage, ...],
) -> DataAvailability:
    """Aggregate only explicit item statuses; one missing item never mutates its siblings."""

    degraded = (
        schedule.status is DataAvailability.PARTIAL
        or validation.status is ValidationStatus.REVIEW
        or any(
            item.availability is not DataAvailability.AVAILABLE
            for day in schedule.days
            for item in day.activities
        )
        or any(
            item.availability is not DataAvailability.AVAILABLE
            for day in schedule.days
            for item in day.transport_legs
        )
        or any(item.opening_availability is not DataAvailability.AVAILABLE for item in places)
        or any(item.availability is not DataAvailability.AVAILABLE for item in selected_candidates)
        or any(item.availability is not DataAvailability.AVAILABLE for item in weather)
        or any(
            item.status in {CostCoverageStatus.PARTIAL, CostCoverageStatus.MISSING}
            for item in cost_estimate.categories
        )
        or (
            hotel_selection.night_count > 0
            and hotel_selection.status is not DataAvailability.AVAILABLE
        )
    )
    return DataAvailability.PARTIAL if degraded else DataAvailability.AVAILABLE
