"""Formal hotel recommendation and published-plan change contracts for V4."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from backend.contracts.v4.base import DisplayText, Identifier, V4ContractModel, require_unique
from backend.contracts.v4.planner_observations import HotelObservation
from backend.contracts.v4.planner_refs import HotelOfferRef, PlannerScope
from backend.contracts.v4.semantic_operations import SemanticOperationProposal


class BestHotelRecommendation(V4ContractModel):
    role: Literal["best"] = "best"
    hotel_offer_ref: HotelOfferRef
    area_reason: DisplayText
    route_fit: DisplayText
    quality_and_price_fit: DisplayText
    main_tradeoff: DisplayText


class AlternativeHotelRecommendation(V4ContractModel):
    role: Literal["better_value", "alternative_location_or_experience"]
    hotel_offer_ref: HotelOfferRef
    difference_from_best: DisplayText


class SelectedHotelRecommendation(V4ContractModel):
    """The single hotel chosen as this plan's lodging baseline.

    This is a planning recommendation, not a booking or inventory guarantee.
    """

    selection_id: Identifier
    scope: PlannerScope
    hotel_observation_id: Identifier
    hotel_offer_ref: HotelOfferRef
    area_reason: DisplayText
    route_fit: DisplayText
    selection_reason: DisplayText
    main_tradeoff: DisplayText


class FormalHotelRecommendationSet(V4ContractModel):
    """Exactly one best hotel and two intentionally differentiated alternatives."""

    recommendation_set_id: Identifier
    scope: PlannerScope
    hotel_observation_id: Identifier
    recommended_hotel: BestHotelRecommendation
    alternative_hotels: tuple[AlternativeHotelRecommendation, ...] = Field(
        min_length=2,
        max_length=2,
    )
    reason_summary: DisplayText

    @model_validator(mode="after")
    def roles_and_offers_are_distinct_and_current(self) -> FormalHotelRecommendationSet:
        expected_roles = {"better_value", "alternative_location_or_experience"}
        if {item.role for item in self.alternative_hotels} != expected_roles:
            raise ValueError(
                "hotel alternatives require one value and one location/experience role"
            )
        all_refs = (
            self.recommended_hotel.hotel_offer_ref,
            *(item.hotel_offer_ref for item in self.alternative_hotels),
        )
        if any(item.hotel_observation_id != self.hotel_observation_id for item in all_refs):
            raise ValueError("hotel recommendations must use the current HotelObservation")
        require_unique([item.offer_id for item in all_refs], "recommended hotel offer IDs")
        require_unique([item.property_id for item in all_refs], "recommended hotel properties")
        return self


def validate_hotel_recommendations_against_observation(
    recommendations: FormalHotelRecommendationSet,
    observation: HotelObservation,
) -> None:
    """Reject stale, unavailable, fixed-booking, or unsourced hotel selections."""

    if observation.mode != "search":
        raise ValueError("formal 1+2 hotel recommendations require a search observation")
    if recommendations.hotel_observation_id != observation.hotel_observation_id:
        raise ValueError("hotel recommendation set references a different observation")
    if (
        recommendations.scope.trip_id != observation.scope.trip_id
        or recommendations.scope.generation_id != observation.scope.generation_id
        or recommendations.scope.task_book_id != observation.scope.task_book_id
        or recommendations.scope.task_book_version != observation.scope.task_book_version
    ):
        raise ValueError("hotel recommendation set and observation have different ownership")
    available = {
        item.offer_ref.offer_id: item
        for item in observation.offers
        if item.availability_status in {"available", "limited"}
    }
    selected = (
        recommendations.recommended_hotel.hotel_offer_ref,
        *(item.hotel_offer_ref for item in recommendations.alternative_hotels),
    )
    if any(item.offer_id not in available for item in selected):
        raise ValueError("hotel recommendation references an unavailable or unknown offer")


def validate_selected_hotel_against_observation(
    selection: SelectedHotelRecommendation,
    observation: HotelObservation,
) -> None:
    """Validate one current hotel identity without upgrading unknown inventory."""

    if observation.mode != "search":
        raise ValueError("selected hotel requires a search observation")
    if selection.hotel_observation_id != observation.hotel_observation_id:
        raise ValueError("selected hotel references a different observation")
    if (
        selection.scope.trip_id != observation.scope.trip_id
        or selection.scope.generation_id != observation.scope.generation_id
        or selection.scope.task_book_id != observation.scope.task_book_id
        or selection.scope.task_book_version != observation.scope.task_book_version
    ):
        raise ValueError("selected hotel and observation have different ownership")
    offer = next(
        (item for item in observation.offers if item.offer_ref == selection.hotel_offer_ref),
        None,
    )
    if offer is None or offer.availability_status == "unavailable":
        raise ValueError("selected hotel references an unavailable offer")


class PlanChangeRequest(V4ContractModel):
    """Validated user intent against an immutable published plan version."""

    plan_change_request_id: Identifier
    trip_id: Identifier
    base_plan_id: Identifier
    base_plan_version: int = Field(ge=1, strict=True)
    user_message_id: Identifier
    proposed_semantic_operations: list[SemanticOperationProposal] = Field(max_length=50)
    semantic_operation_ids: tuple[Identifier, ...] = Field(max_length=50)
    requested_scope: Literal["reply_only", "local_replan", "full_replan", "task_book_change"]

    @model_validator(mode="after")
    def scope_matches_semantic_effects(self) -> PlanChangeRequest:
        local_keys = [item.root.local_operation_key for item in self.proposed_semantic_operations]
        require_unique(local_keys, "PlanChange semantic operation local keys")
        require_unique(self.semantic_operation_ids, "PlanChange semantic operation IDs")
        if len(self.semantic_operation_ids) != len(self.proposed_semantic_operations):
            raise ValueError("each PlanChange semantic operation requires its server-owned ID")
        if self.requested_scope == "reply_only" and self.proposed_semantic_operations:
            raise ValueError("reply_only cannot contain semantic mutation operations")
        if self.requested_scope != "reply_only" and not self.proposed_semantic_operations:
            raise ValueError("a replan or task-book change requires semantic operations")
        return self


V4_PLAN_CHANGE_CONTRACTS: tuple[type[V4ContractModel], ...] = (
    FormalHotelRecommendationSet,
    SelectedHotelRecommendation,
    PlanChangeRequest,
)
