"""P0-07 anchor, activity-cluster, and single-base lodging contracts."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Any
from uuid import UUID

from pydantic import ConfigDict, Field, model_validator

from backend.contracts.base import ContractModel
from backend.contracts.common import CnyAmountRange, NonEmptyText, ShortText
from backend.contracts.enums import (
    AnchorRole,
    HotelSelectionSource,
    LodgingAnchorDecision,
    LodgingCenterKind,
)
from backend.contracts.places import Gcj02Coordinates

LODGING_PLAN_SCHEMA_RULE: dict[str, Any] = {
    "allOf": [
        {
            "if": {
                "properties": {"night_count": {"const": 0}},
                "required": ["night_count"],
            },
            "then": {
                "properties": {
                    "strategies": {"type": "array", "maxItems": 0},
                    "selected_hotel": {"type": "null"},
                }
            },
            "else": {
                "properties": {
                    "strategies": {"type": "array", "minItems": 2, "maxItems": 3},
                    "selected_hotel": {"not": {"type": "null"}},
                },
                "required": ["selected_hotel"],
            },
        }
    ],
    "x-travel-lodging-plan": {
        "startField": "trip_start_date",
        "endField": "trip_end_date",
        "nightCountField": "night_count",
        "anchorsField": "anchors",
        "clustersField": "activity_clusters",
        "strategiesField": "strategies",
        "favoritesField": "favorites",
        "selectedHotelField": "selected_hotel",
    },
}


def _unique(values: Sequence[str | UUID], field_name: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} must not contain duplicates")


class Anchor(ContractModel):
    anchor_id: NonEmptyText
    role: AnchorRole
    place_id: UUID | None = None
    fixed_event_id: NonEmptyText | None = None
    lodging_weight: int = Field(ge=0, le=100, strict=True)
    reason: ShortText

    @model_validator(mode="after")
    def role_and_lodging_influence_are_valid(self) -> Anchor:
        if self.role is AnchorRole.FIXED_EVENT:
            if self.fixed_event_id is None:
                raise ValueError("fixed-event anchor requires fixed_event_id")
        elif self.place_id is None:
            raise ValueError("place-based anchor requires place_id")
        if self.role in {
            AnchorRole.CONVENIENT_ATTRACTION,
            AnchorRole.CONVENIENT_RESTAURANT,
        }:
            if self.lodging_weight != 0:
                raise ValueError("convenient candidates cannot influence the lodging center")
        elif self.lodging_weight == 0:
            raise ValueError("strong anchors must carry a positive lodging weight")
        return self


class AnchorSet(ContractModel):
    anchors: list[Anchor] = Field(default_factory=list)

    @model_validator(mode="after")
    def anchor_ids_are_unique(self) -> AnchorSet:
        _unique([anchor.anchor_id for anchor in self.anchors], "anchor_id")
        return self


class ActivityCluster(ContractModel):
    cluster_id: NonEmptyText
    label: NonEmptyText
    anchor_ids: list[NonEmptyText] = Field(
        min_length=1,
        json_schema_extra={"uniqueItems": True},
    )
    center: Gcj02Coordinates
    representative_place_ids: list[UUID] = Field(
        default_factory=list,
        json_schema_extra={"uniqueItems": True},
    )

    @model_validator(mode="after")
    def references_are_unique(self) -> ActivityCluster:
        _unique(self.anchor_ids, "anchor_ids")
        _unique(self.representative_place_ids, "representative_place_ids")
        return self


class RepresentativeHotel(ContractModel):
    place_id: UUID
    offer_id: NonEmptyText | None = None
    price: CnyAmountRange | None = None
    fit_reason: ShortText
    tradeoff: ShortText


class LodgingAreaStrategy(ContractModel):
    strategy_id: NonEmptyText
    label: NonEmptyText
    center_kind: LodgingCenterKind
    center_name: NonEmptyText
    anchor_ids: list[NonEmptyText] = Field(
        min_length=1,
        json_schema_extra={"uniqueItems": True},
    )
    transit_nodes: list[NonEmptyText] = Field(
        default_factory=list,
        json_schema_extra={"uniqueItems": True},
    )
    why_it_fits: ShortText
    tradeoffs: list[ShortText] = Field(min_length=1)
    representative_hotels: list[RepresentativeHotel] = Field(min_length=1, max_length=5)

    @model_validator(mode="after")
    def strategy_references_are_unique(self) -> LodgingAreaStrategy:
        _unique(self.anchor_ids, "anchor_ids")
        _unique(self.transit_nodes, "transit_nodes")
        _unique(
            [hotel.place_id for hotel in self.representative_hotels],
            "representative hotel place_id",
        )
        return self


class HotelFavoritesSubmission(ContractModel):
    hotel_place_ids: list[UUID] = Field(
        default_factory=list,
        json_schema_extra={"uniqueItems": True},
    )

    @model_validator(mode="after")
    def favorites_are_unique(self) -> HotelFavoritesSubmission:
        _unique(self.hotel_place_ids, "hotel_place_ids")
        return self


class PrebookedHotel(ContractModel):
    place_id: UUID
    check_in: date
    check_out: date

    @model_validator(mode="after")
    def stay_has_at_least_one_night(self) -> PrebookedHotel:
        if self.check_out <= self.check_in:
            raise ValueError("prebooked hotel check_out must be after check_in")
        return self


class LodgingAnchorAssessment(ContractModel):
    prebooked_hotels: list[PrebookedHotel] = Field(default_factory=list)
    decision: LodgingAnchorDecision
    selected_base_place_id: UUID | None = None

    @model_validator(mode="after")
    def decision_matches_prebooked_hotels(self) -> LodgingAnchorAssessment:
        place_ids = [hotel.place_id for hotel in self.prebooked_hotels]
        _unique(place_ids, "prebooked hotel place_id")
        count = len(place_ids)
        if count == 0:
            if (
                self.decision is not LodgingAnchorDecision.NO_PREBOOKED_HOTEL
                or self.selected_base_place_id is not None
            ):
                raise ValueError("zero prebooked hotels must use no_prebooked_hotel")
            return self
        if self.selected_base_place_id is not None and self.selected_base_place_id not in place_ids:
            raise ValueError("selected base hotel must be one of the prebooked hotels")
        if count == 1:
            if (
                self.decision is not LodgingAnchorDecision.SINGLE_BASE_READY
                or self.selected_base_place_id != place_ids[0]
            ):
                raise ValueError("one prebooked hotel must become the selected single base")
            return self
        if self.selected_base_place_id is None:
            if self.decision is not LodgingAnchorDecision.REQUIRES_SINGLE_BASE_CLARIFICATION:
                raise ValueError("multiple hotels require a single-base clarification")
        elif self.decision is not LodgingAnchorDecision.SINGLE_BASE_READY:
            raise ValueError("a clarified base must use single_base_ready")
        return self

    @property
    def requires_clarification(self) -> bool:
        return self.decision is LodgingAnchorDecision.REQUIRES_SINGLE_BASE_CLARIFICATION


class SelectedHotel(ContractModel):
    place_id: UUID
    strategy_id: NonEmptyText | None = None
    source: HotelSelectionSource
    check_in: date
    check_out: date
    night_count: int = Field(ge=1, le=4, strict=True)
    reason: ShortText

    @model_validator(mode="after")
    def stay_length_matches_night_count(self) -> SelectedHotel:
        if (self.check_out - self.check_in).days != self.night_count:
            raise ValueError("selected hotel must cover every declared night")
        return self


class LodgingPlan(ContractModel):
    model_config = ConfigDict(json_schema_extra=LODGING_PLAN_SCHEMA_RULE)

    trip_start_date: date
    trip_end_date: date
    night_count: int = Field(ge=0, le=4, strict=True)
    anchors: list[Anchor] = Field(default_factory=list)
    activity_clusters: list[ActivityCluster] = Field(default_factory=list, max_length=3)
    strategies: list[LodgingAreaStrategy] = Field(default_factory=list, max_length=3)
    favorites: list[UUID] = Field(default_factory=list, json_schema_extra={"uniqueItems": True})
    selected_hotel: SelectedHotel | None = None

    @model_validator(mode="after")
    def single_base_covers_the_trip(self) -> LodgingPlan:
        day_count = (self.trip_end_date - self.trip_start_date).days + 1
        if day_count < 1 or day_count > 5:
            raise ValueError("lodging plan trip length must be between 1 and 5 days")
        if self.night_count != day_count - 1:
            raise ValueError("night_count must equal day_count minus one")
        _unique([anchor.anchor_id for anchor in self.anchors], "anchor_id")
        _unique([cluster.cluster_id for cluster in self.activity_clusters], "cluster_id")
        _unique([strategy.strategy_id for strategy in self.strategies], "strategy_id")
        _unique(self.favorites, "favorites")

        anchor_ids = {anchor.anchor_id for anchor in self.anchors}
        for cluster in self.activity_clusters:
            if not set(cluster.anchor_ids) <= anchor_ids:
                raise ValueError("activity clusters may only reference declared anchors")
        for strategy in self.strategies:
            if not set(strategy.anchor_ids) <= anchor_ids:
                raise ValueError("lodging strategies may only reference declared anchors")

        if self.night_count == 0:
            if self.strategies or self.selected_hotel is not None:
                raise ValueError("a one-day trip has no lodging strategies or selected hotel")
            return self
        if not 2 <= len(self.strategies) <= 3:
            raise ValueError("an overnight trip requires two or three lodging strategies")
        if self.selected_hotel is None:
            raise ValueError("an overnight lodging plan requires one selected hotel")
        if (
            self.selected_hotel.check_in != self.trip_start_date
            or self.selected_hotel.check_out != self.trip_end_date
            or self.selected_hotel.night_count != self.night_count
        ):
            raise ValueError("the selected hotel must cover all default lodging nights")

        strategies_by_id = {strategy.strategy_id: strategy for strategy in self.strategies}
        candidate_ids = {
            hotel.place_id
            for strategy in self.strategies
            for hotel in strategy.representative_hotels
        }
        if not set(self.favorites) <= candidate_ids:
            raise ValueError("hotel favorites must belong to the lodging candidate set")
        selected_strategy = None
        if self.selected_hotel.strategy_id is not None:
            selected_strategy = strategies_by_id.get(self.selected_hotel.strategy_id)
            if selected_strategy is None:
                raise ValueError("selected hotel strategy_id must reference a declared strategy")
        elif self.selected_hotel.source is not HotelSelectionSource.PREBOOKED:
            raise ValueError("candidate-selected hotel requires a lodging strategy_id")

        if selected_strategy is not None:
            strategy_hotel_ids = {
                hotel.place_id for hotel in selected_strategy.representative_hotels
            }
            if self.selected_hotel.place_id not in strategy_hotel_ids:
                raise ValueError("selected hotel must belong to its referenced lodging strategy")
        if self.selected_hotel.source is HotelSelectionSource.USER_FAVORITES:
            if self.selected_hotel.place_id not in self.favorites:
                raise ValueError("favorite-selected hotel must belong to the favorites set")
        elif (
            self.selected_hotel.source is HotelSelectionSource.SYSTEM_CANDIDATES
            and self.selected_hotel.place_id not in candidate_ids
        ):
            raise ValueError("system-selected hotel must belong to the candidate set")
        return self


P0_LODGING_CONTRACTS: tuple[type[ContractModel], ...] = (
    AnchorSet,
    LodgingAnchorAssessment,
    HotelFavoritesSubmission,
    LodgingPlan,
)
