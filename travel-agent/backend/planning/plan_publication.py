"""Prepare one V3 plan for the existing atomic stable-state repository boundary."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from uuid import UUID

from backend.contracts.enums import DataAvailability, TripPhase
from backend.contracts.itinerary_validation import DailyWeatherCoverage, ValidationStatus
from backend.contracts.plan_publication import (
    PlanPublicationRequest,
    PublishedPlan,
    derive_published_plan_availability,
)
from backend.contracts.state import TripState
from backend.planning.city_registry import CityRegistry, default_city_registry
from backend.planning.itinerary_validation import ItineraryValidationService


class PlanPublicationError(RuntimeError):
    """Raised before persistence when a plan is stale or not publishable."""


class StalePlanGenerationError(PlanPublicationError):
    pass


class PlanPublicationService:
    """Build the complete next snapshot; persistence remains one DB transaction."""

    def __init__(
        self,
        *,
        validation_service: ItineraryValidationService | None = None,
        city_registry: CityRegistry | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._validation = validation_service or ItineraryValidationService(clock=clock)
        self._city_registry = city_registry or default_city_registry()
        self._clock = clock

    async def prepare_stable_state(
        self,
        state: TripState,
        request: PlanPublicationRequest,
        *,
        is_generation_active: Callable[[UUID, UUID], Awaitable[bool]],
    ) -> TripState:
        request = PlanPublicationRequest.model_validate(
            request.model_dump(mode="json"),
            context={"today": request.validation_request.scheduling_request.business_date},
        )
        self._validate_state_boundary(state, request)
        if not await is_generation_active(request.trip_id, request.generation_id):
            raise StalePlanGenerationError("the planning generation is no longer active")

        strict_request = request.validation_request.model_copy(
            update={
                "schedule_draft": request.repair_result.best_schedule_draft,
                "cost_draft": request.repair_result.best_cost_draft,
            }
        )
        final_validation = self._validation.validate(strict_request)
        if final_validation.status is ValidationStatus.BLOCKED:
            raise PlanPublicationError("a plan with hard conflicts cannot be published")
        schedule, cost = self._validation.publish_strict(
            strict_request,
            final_validation,
        )
        weather = strict_request.weather or tuple(
            DailyWeatherCoverage(
                service_date=day.service_date,
                availability=DataAvailability.MISSING,
                night_condition_available=False,
                missing_reason="本次规划未取得该日期的天气资料。",
            )
            for day in schedule.days
        )
        scheduled_place_ids = {
            place_id
            for day in schedule.days
            for place_id in (
                day.start_place_id,
                *(item.place_id for item in day.activities),
                day.end_place_id,
            )
        }
        published_places = tuple(
            item
            for item in strict_request.scheduling_request.places
            if item.place_id in scheduled_place_ids
        )

        published_at = self._clock()
        if published_at.tzinfo is None or published_at.utcoffset() is None:
            raise PlanPublicationError("plan publication clock must return an aware datetime")
        availability = derive_published_plan_availability(
            schedule=schedule,
            cost_estimate=cost,
            validation=final_validation,
            places=published_places,
            selected_candidates=request.selected_candidates,
            hotel_selection=strict_request.scheduling_request.hotel_result,
            weather=weather,
        )
        published_plan = PublishedPlan(
            plan_version_id=request.plan_version_id,
            publication_key=request.publication_key,
            generation_id=request.generation_id,
            parent_version_id=request.parent_version_id,
            base_confirmed_version_id=request.base_confirmed_version_id,
            trip_id=request.trip_id,
            input_state_version=request.expected_state_version,
            result_contract_version="1.0.0",
            availability=availability,
            places=published_places,
            selected_candidates=request.selected_candidates,
            hotel_selection=strict_request.scheduling_request.hotel_result,
            weather=weather,
            schedule=schedule,
            cost_estimate=cost,
            validation=final_validation,
            issues=final_validation.issues,
            assumptions=request.assumptions,
            map_projection=request.map_projection,
            published_at=published_at,
        )

        # A second check closes the expensive validation window. The realtime caller performs
        # this under its trip lock immediately before the atomic repository commit as well.
        if not await is_generation_active(request.trip_id, request.generation_id):
            raise StalePlanGenerationError("the planning generation became stale before publish")

        return TripState.model_validate(
            {
                **state.model_dump(mode="json"),
                "phase": TripPhase.DRAFT_READY,
                "state_version": state.state_version + 1,
                "current_plan_version_id": str(request.plan_version_id),
                "base_confirmed_version_id": (
                    str(request.base_confirmed_version_id)
                    if request.base_confirmed_version_id is not None
                    else None
                ),
                "published_plan": published_plan.model_dump(mode="json"),
                "pending_plan_modification": None,
                "map_view": request.map_projection.model_dump(mode="json"),
                "active_generation_id": None,
            }
        )

    def _validate_state_boundary(self, state: TripState, request: PlanPublicationRequest) -> None:
        if state.trip_id != request.trip_id:
            raise PlanPublicationError("publication trip does not match the stable state")
        if state.state_version != request.expected_state_version:
            raise PlanPublicationError("publication input state version is stale")
        if state.phase is not TripPhase.PLANNING:
            raise PlanPublicationError("only the planning phase can publish a new draft")
        if state.task_book is None or state.task_book.status.value != "confirmed":
            raise PlanPublicationError("publication requires a confirmed task book")
        planning_city_id = request.validation_request.scheduling_request.city_id
        state_city_id = (
            state.city_id
            if state.city_id is not None
            else self._city_registry.resolve(state.city.value).city_id
            if state.city is not None
            else None
        )
        if state_city_id != planning_city_id:
            raise PlanPublicationError("publication city does not match the stable state")
        if state.date_range is None or (
            state.date_range.start_date
            != request.validation_request.scheduling_request.task_book.date_range.start_date
            or state.date_range.end_date
            != request.validation_request.scheduling_request.task_book.date_range.end_date
        ):
            raise PlanPublicationError("publication dates do not match the stable state")
        hotel_result = request.validation_request.scheduling_request.hotel_result
        if hotel_result.night_count and (
            state.selected_hotel is None
            or hotel_result.selected_hotel_place_id != state.selected_hotel.place_id
        ):
            raise PlanPublicationError("publication hotel does not match the stable state")
        if state.base_confirmed_version_id != request.base_confirmed_version_id:
            raise PlanPublicationError("base confirmed version changed during planning")
        if request.parent_version_id != state.current_plan_version_id:
            raise PlanPublicationError("parent version must be the state used for planning")
