"""V3-35 contracts for one route-aware lodging base strategy."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from enum import StrEnum
from uuid import UUID

from pydantic import AwareDatetime, ConfigDict, Field, model_validator

from backend.contracts.base import ContractModel
from backend.contracts.common import CnyAmountRange, NonEmptyText, ShortText
from backend.contracts.enums import (
    AnchorRole,
    DataAvailability,
    MobilityTolerance,
)
from backend.contracts.places import Gcj02Coordinates
from backend.contracts.spatial_planning import SpatialPlanningResult


class ImmutableLodgingStrategyModel(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LodgingBaseKind(StrEnum):
    AREA = "area"
    TRANSIT_NODE = "transit_node"
    FIXED_HOTEL = "fixed_hotel"


class LodgingClusterAccess(ImmutableLodgingStrategyModel):
    """Source-backed door-to-cluster access facts for one possible base."""

    cluster_id: UUID
    status: DataAvailability
    transit_minutes: int | None = Field(default=None, ge=0, le=360, strict=True)
    transit_transfer_count: int | None = Field(default=None, ge=0, le=12, strict=True)
    transit_last_mile_walk_m: int | None = Field(default=None, ge=0, le=20_000, strict=True)
    taxi_minutes: int | None = Field(default=None, ge=0, le=360, strict=True)
    taxi_cost: CnyAmountRange | None = None
    cycling_minutes: int | None = Field(default=None, ge=0, le=360, strict=True)
    source_reference_ids: tuple[NonEmptyText, ...] = Field(min_length=1)
    missing_fields: tuple[NonEmptyText, ...] = ()
    missing_reason: ShortText | None = None

    @model_validator(mode="after")
    def availability_and_metrics_are_consistent(self) -> LodgingClusterAccess:
        _unique(self.source_reference_ids, "lodging access source references")
        _unique(self.missing_fields, "lodging access missing fields")
        has_route = any(
            value is not None
            for value in (self.transit_minutes, self.taxi_minutes, self.cycling_minutes)
        )
        transit_details = (
            self.transit_transfer_count is not None or self.transit_last_mile_walk_m is not None
        )
        if transit_details and self.transit_minutes is None:
            raise ValueError("transit details require a transit duration")
        if self.taxi_cost is not None and self.taxi_minutes is None:
            raise ValueError("taxi cost requires a door-to-door taxi duration")
        if self.status is DataAvailability.AVAILABLE:
            if not has_route or self.missing_fields or self.missing_reason is not None:
                raise ValueError("available lodging access requires usable complete route facts")
        elif self.status is DataAvailability.PARTIAL:
            if not has_route or not self.missing_fields or self.missing_reason is None:
                raise ValueError("partial lodging access requires route facts and missing details")
        elif has_route or not self.missing_fields or self.missing_reason is None:
            raise ValueError("missing lodging access requires an explicit unknown route cost")
        return self


class LodgingBaseCandidate(ImmutableLodgingStrategyModel):
    candidate_id: UUID
    city_id: NonEmptyText
    kind: LodgingBaseKind
    label: NonEmptyText
    center: Gcj02Coordinates
    transit_node_place_id: UUID | None = None
    quality_level: int | None = Field(default=None, ge=1, le=5, strict=True)
    typical_nightly_price: CnyAmountRange | None = None
    cluster_accesses: tuple[LodgingClusterAccess, ...] = Field(min_length=1, max_length=16)
    source_reference_ids: tuple[NonEmptyText, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def kind_and_references_are_consistent(self) -> LodgingBaseCandidate:
        if self.kind is LodgingBaseKind.FIXED_HOTEL:
            raise ValueError("fixed hotels are declared through fixed_hotel_node_id")
        if self.kind is LodgingBaseKind.TRANSIT_NODE:
            if self.transit_node_place_id is None:
                raise ValueError("transit-node candidates require a place reference")
        elif self.transit_node_place_id is not None:
            raise ValueError("only transit-node candidates may reference a transit place")
        _unique(
            [access.cluster_id for access in self.cluster_accesses],
            "lodging candidate cluster accesses",
        )
        _unique(self.source_reference_ids, "lodging candidate source references")
        return self


class LodgingStrategyPreferences(ImmutableLodgingStrategyModel):
    """Current-trip preference projection; one is low/compact, five is high/relaxed."""

    transit_taxi_level: int = Field(ge=1, le=5, strict=True)
    walking_tolerance: MobilityTolerance
    cycling_tolerance: MobilityTolerance
    pace_level: int = Field(ge=1, le=5, strict=True)
    quality_level: int = Field(ge=1, le=5, strict=True)
    value_priority_level: int = Field(ge=1, le=5, strict=True)


class LodgingStrategyRequest(ImmutableLodgingStrategyModel):
    request_id: UUID
    trip_id: UUID
    input_state_version: int = Field(ge=0, strict=True)
    city_id: NonEmptyText
    start_date: date
    end_date: date
    spatial_result: SpatialPlanningResult
    preferences: LodgingStrategyPreferences
    candidates: tuple[LodgingBaseCandidate, ...] = Field(default=(), max_length=8)
    fixed_hotel_node_id: UUID | None = None
    fixed_hotel_accesses: tuple[LodgingClusterAccess, ...] = Field(default=(), max_length=16)

    @model_validator(mode="after")
    def trip_boundary_and_references_are_consistent(self) -> LodgingStrategyRequest:
        day_count = (self.end_date - self.start_date).days + 1
        if not 1 <= day_count <= 5:
            raise ValueError("lodging strategy date range must contain 1 to 5 days")
        if (
            self.spatial_result.trip_id != self.trip_id
            or self.spatial_result.city_id != self.city_id
            or self.spatial_result.input_state_version != self.input_state_version
        ):
            raise ValueError("lodging strategy must use the current trip spatial result")
        _unique([candidate.candidate_id for candidate in self.candidates], "lodging candidate IDs")
        if any(candidate.city_id != self.city_id for candidate in self.candidates):
            raise ValueError("every lodging candidate must belong to the trip city")
        cluster_ids = {cluster.cluster_id for cluster in self.spatial_result.clusters}
        for candidate in self.candidates:
            if {item.cluster_id for item in candidate.cluster_accesses} != cluster_ids:
                raise ValueError("every lodging candidate must assess every activity cluster")
        fixed_access_ids = {item.cluster_id for item in self.fixed_hotel_accesses}
        if len(fixed_access_ids) != len(self.fixed_hotel_accesses):
            raise ValueError("fixed-hotel cluster accesses must be unique")
        if self.fixed_hotel_node_id is None:
            if self.fixed_hotel_accesses:
                raise ValueError("fixed-hotel access facts require a fixed hotel")
        else:
            anchors = {anchor.node_id: anchor for anchor in self.spatial_result.anchors}
            fixed = anchors.get(self.fixed_hotel_node_id)
            if fixed is None or fixed.role is not AnchorRole.FIXED_HOTEL:
                raise ValueError("fixed hotel must reference a fixed-hotel spatial anchor")
            trip_dates = {
                self.start_date.fromordinal(day)
                for day in range(
                    self.start_date.toordinal(),
                    self.end_date.toordinal() + 1,
                )
            }
            if not trip_dates <= set(fixed.available_dates):
                raise ValueError("fixed hotel must cover every trip date")
            if self.candidates:
                raise ValueError("a fixed hotel skips competing lodging candidates")
            if fixed_access_ids != cluster_ids:
                raise ValueError("a fixed hotel must assess every activity cluster")
        if day_count == 1:
            if self.candidates or self.fixed_hotel_node_id is not None:
                raise ValueError("a day trip must not create a lodging base strategy")
        elif self.fixed_hotel_node_id is None and not self.candidates:
            raise ValueError("an overnight trip requires lodging candidates or a fixed hotel")
        return self

    @property
    def night_count(self) -> int:
        return (self.end_date - self.start_date).days


class LodgingCommuteSummary(ImmutableLodgingStrategyModel):
    assessed_cluster_count: int = Field(ge=1, le=16, strict=True)
    known_cluster_count: int = Field(ge=0, le=16, strict=True)
    typical_minutes: int | None = Field(default=None, ge=0, strict=True)
    maximum_minutes: int | None = Field(default=None, ge=0, strict=True)
    typical_transfers: float | None = Field(default=None, ge=0, le=12)
    typical_last_mile_walk_m: int | None = Field(default=None, ge=0, strict=True)
    typical_taxi_cost: CnyAmountRange | None = None
    additional_minutes_vs_fastest: int | None = Field(default=None, ge=0, strict=True)

    @model_validator(mode="after")
    def known_count_matches_summary(self) -> LodgingCommuteSummary:
        if self.known_cluster_count > self.assessed_cluster_count:
            raise ValueError("known lodging clusters cannot exceed assessed clusters")
        if self.known_cluster_count == 0:
            if any(
                value is not None
                for value in (
                    self.typical_minutes,
                    self.maximum_minutes,
                    self.typical_transfers,
                    self.typical_last_mile_walk_m,
                    self.typical_taxi_cost,
                    self.additional_minutes_vs_fastest,
                )
            ):
                raise ValueError("unknown lodging commute cannot report summary metrics")
        elif self.typical_minutes is None or self.maximum_minutes is None:
            raise ValueError("known lodging commute requires typical and maximum minutes")
        elif self.maximum_minutes < self.typical_minutes:
            raise ValueError("maximum lodging commute cannot be below typical commute")
        return self


class LodgingStrategyScore(ImmutableLodgingStrategyModel):
    coverage: int = Field(ge=0, le=100, strict=True)
    transport_fit: int = Field(ge=0, le=100, strict=True)
    walking_fit: int = Field(ge=0, le=100, strict=True)
    route_simplicity: int = Field(ge=0, le=100, strict=True)
    comfort_fit: int = Field(ge=0, le=100, strict=True)
    quality_fit: int = Field(ge=0, le=100, strict=True)
    value_fit: int = Field(ge=0, le=100, strict=True)
    total: int = Field(ge=0, le=100, strict=True)


class LodgingBaseStrategy(ImmutableLodgingStrategyModel):
    strategy_id: UUID
    rank: int = Field(ge=1, le=3, strict=True)
    candidate_id: UUID | None = None
    kind: LodgingBaseKind
    label: NonEmptyText
    center: Gcj02Coordinates
    fixed_hotel_node_id: UUID | None = None
    covered_cluster_ids: tuple[UUID, ...] = Field(min_length=1, max_length=16)
    covered_anchor_ids: tuple[UUID, ...] = Field(min_length=1, max_length=16)
    commute: LodgingCommuteSummary
    score: LodgingStrategyScore
    typical_nightly_price: CnyAmountRange | None = None
    advantages: tuple[ShortText, ...] = Field(min_length=1, max_length=4)
    tradeoffs: tuple[ShortText, ...] = Field(min_length=1, max_length=4)
    applicable_conditions: tuple[ShortText, ...] = Field(min_length=1, max_length=4)
    explanation: ShortText
    source_reference_ids: tuple[NonEmptyText, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def kind_and_references_are_consistent(self) -> LodgingBaseStrategy:
        _unique(self.covered_cluster_ids, "lodging strategy clusters")
        _unique(self.covered_anchor_ids, "lodging strategy anchors")
        _unique(self.source_reference_ids, "lodging strategy source references")
        if self.kind is LodgingBaseKind.FIXED_HOTEL:
            if self.fixed_hotel_node_id is None or self.candidate_id is not None:
                raise ValueError("fixed lodging strategy requires only its fixed hotel anchor")
        elif self.candidate_id is None or self.fixed_hotel_node_id is not None:
            raise ValueError("candidate lodging strategy requires only its candidate ID")
        return self


class LodgingStrategyResult(ImmutableLodgingStrategyModel):
    algorithm_version: str = Field(pattern=r"^[1-9]\d*\.\d+\.\d+$")
    request_id: UUID
    trip_id: UUID
    input_state_version: int = Field(ge=0, strict=True)
    city_id: NonEmptyText
    night_count: int = Field(ge=0, le=4, strict=True)
    status: DataAvailability
    strategies: tuple[LodgingBaseStrategy, ...] = Field(default=(), max_length=3)
    selected_strategy_id: UUID | None = None
    degradation_reasons: tuple[ShortText, ...] = ()
    generated_at: AwareDatetime

    @model_validator(mode="after")
    def one_base_and_references_are_consistent(self) -> LodgingStrategyResult:
        _unique([strategy.strategy_id for strategy in self.strategies], "lodging strategy IDs")
        ranks = [strategy.rank for strategy in self.strategies]
        if ranks != list(range(1, len(ranks) + 1)):
            raise ValueError("lodging strategy ranks must be contiguous and ordered")
        _unique(self.degradation_reasons, "lodging degradation reasons")
        if self.night_count == 0:
            if self.strategies or self.selected_strategy_id is not None:
                raise ValueError("a day trip cannot select a lodging strategy")
        else:
            if not self.strategies or self.selected_strategy_id is None:
                raise ValueError("an overnight trip must select exactly one lodging strategy")
            if self.selected_strategy_id not in {
                strategy.strategy_id for strategy in self.strategies
            }:
                raise ValueError("selected lodging strategy must be one of the declared strategies")
        if self.status is DataAvailability.AVAILABLE and self.degradation_reasons:
            raise ValueError("available lodging strategy result cannot contain degradation")
        if self.status is DataAvailability.PARTIAL and not self.degradation_reasons:
            raise ValueError("partial lodging strategy result requires degradation reasons")
        return self


def _unique(values: Sequence[object], label: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must be unique")


V3_LODGING_STRATEGY_CONTRACTS: tuple[type[ContractModel], ...] = (
    LodgingStrategyRequest,
    LodgingStrategyResult,
)
