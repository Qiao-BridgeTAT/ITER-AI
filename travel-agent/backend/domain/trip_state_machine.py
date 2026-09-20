"""Deterministic P0-09 phase transitions for the unified trip state."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from backend.contracts.enums import CityCode, TripPhase
from backend.contracts.state import TripState


class InvalidTripTransition(ValueError):
    """Raised when a caller attempts to skip or reverse a protected phase."""


_ALLOWED_TRANSITIONS: dict[TripPhase, frozenset[TripPhase]] = {
    TripPhase.COLD_START: frozenset({TripPhase.CITY_SELECTION}),
    TripPhase.CITY_SELECTION: frozenset({TripPhase.CITY_BRIEF}),
    TripPhase.CITY_BRIEF: frozenset({TripPhase.TRIP_SETUP}),
    TripPhase.TRIP_SETUP: frozenset({TripPhase.INTEREST_SELECTION}),
    TripPhase.INTEREST_SELECTION: frozenset(
        {TripPhase.ATTRACTION_SELECTION, TripPhase.TASK_REFLECTION}
    ),
    TripPhase.ATTRACTION_SELECTION: frozenset(
        {TripPhase.DINING_SELECTION, TripPhase.TASK_REFLECTION}
    ),
    TripPhase.DINING_SELECTION: frozenset({TripPhase.LODGING_SELECTION, TripPhase.TASK_REFLECTION}),
    TripPhase.LODGING_SELECTION: frozenset({TripPhase.TASK_REFLECTION}),
    TripPhase.TASK_REFLECTION: frozenset({TripPhase.PLANNING}),
    TripPhase.PLANNING: frozenset({TripPhase.DRAFT_READY}),
    TripPhase.DRAFT_READY: frozenset({TripPhase.REVISING, TripPhase.CONFIRMED}),
    TripPhase.REVISING: frozenset({TripPhase.PLANNING}),
    TripPhase.CONFIRMED: frozenset({TripPhase.REVISING}),
}


class TripStateMachine:
    @staticmethod
    def allowed_targets(phase: TripPhase) -> frozenset[TripPhase]:
        return _ALLOWED_TRANSITIONS[phase]

    @staticmethod
    def transition(
        state: TripState,
        target: TripPhase,
        *,
        new_plan_version_id: UUID | None = None,
    ) -> TripState:
        if target not in _ALLOWED_TRANSITIONS[state.phase]:
            raise InvalidTripTransition(f"cannot transition from {state.phase} to {target}")
        if state.phase is TripPhase.COLD_START and (
            state.personal_defaults is None or state.cold_start_completed_at is None
        ):
            raise InvalidTripTransition("cold start must be completed before city selection")

        update: dict[str, Any] = {"phase": target, "state_version": state.state_version + 1}
        if state.phase is TripPhase.CONFIRMED and target is TripPhase.REVISING:
            if state.current_plan_version_id is None:
                raise InvalidTripTransition("confirmed state has no current plan version")
            update["base_confirmed_version_id"] = state.current_plan_version_id
        if state.phase is TripPhase.PLANNING and target is TripPhase.DRAFT_READY:
            if new_plan_version_id is None:
                raise InvalidTripTransition("publishing a draft requires a new plan version id")
            update["current_plan_version_id"] = new_plan_version_id
        elif new_plan_version_id is not None:
            raise InvalidTripTransition("new plan version id is only valid when publishing a draft")
        return TripState.model_validate({**state.model_dump(), **update})

    @staticmethod
    def change_city(state: TripState, city: CityCode) -> TripState:
        if state.phase in {TripPhase.COLD_START, TripPhase.CITY_SELECTION}:
            raise InvalidTripTransition("change_city is only valid after a city was selected")
        cleared: dict[str, Any] = {
            "city": city,
            "phase": TripPhase.CITY_BRIEF,
            "state_version": state.state_version + 1,
            "city_brief": None,
            "profile_decision": None,
            "resolved_preferences": None,
            "fixed_events": [],
            "city_theme_selection": None,
            "attraction_feedback": [],
            "dining_option_card": None,
            "dining_preferences": None,
            "restaurant_feedback": [],
            "hotel_favorites": [],
            "anchors": [],
            "candidate_places": [],
            "lodging_strategies": [],
            "selected_hotel": None,
            "task_book": None,
            "itinerary": None,
            "cost_estimate": None,
            "published_plan": None,
            "pending_plan_modification": None,
            "issues": [],
            "assumptions": [],
            "current_plan_version_id": None,
            "base_confirmed_version_id": None,
            "active_generation_id": None,
        }
        return TripState.model_validate({**state.model_dump(), **cleared})
