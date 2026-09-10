"""V3-36 contracts for seven hotel options and an explicitly finalized hotel."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import AwareDatetime, ConfigDict, Field, HttpUrl, model_validator

from backend.contracts.base import ContractModel
from backend.contracts.common import CnyAmountRange, NonEmptyText, ShortText
from backend.contracts.enums import DataAvailability, ProviderCode
from backend.contracts.lodging_strategy import (
    LodgingBaseKind,
    LodgingStrategyPreferences,
    LodgingStrategyResult,
)
from backend.contracts.places import Gcj02Coordinates
from backend.providers.contracts import ProviderFailureCode


class ImmutableHotelSelectionModel(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class HotelSelectionSource(StrEnum):
    USER_CHOICE = "user_choice"
    AGENT_DELEGATED = "agent_delegated"
    PREBOOKED = "prebooked"


class HotelDecisionMode(StrEnum):
    NOT_REQUIRED = "not_required"
    AWAIT_USER = "await_user"
    USER_CHOICE = "user_choice"
    AGENT_DELEGATED = "agent_delegated"
    PREBOOKED = "prebooked"


class HotelDecisionStatus(StrEnum):
    NOT_REQUIRED = "not_required"
    AWAITING_USER = "awaiting_user"
    FINAL = "final"


class HotelPriceBand(StrEnum):
    BUDGET = "budget"
    MID_RANGE = "mid_range"
    UPSCALE = "upscale"
    LUXURY = "luxury"


class FixedHotelInput(ImmutableHotelSelectionModel):
    node_id: UUID
    place_id: UUID
    name: NonEmptyText
    coordinates: Gcj02Coordinates
    source_reference_ids: tuple[NonEmptyText, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def references_are_unique(self) -> FixedHotelInput:
        _unique(self.source_reference_ids, "fixed hotel source references")
        return self


class HotelSelectionRequest(ImmutableHotelSelectionModel):
    request_id: UUID
    trip_id: UUID
    input_state_version: int = Field(ge=0, strict=True)
    city_id: NonEmptyText
    check_in: date
    check_out: date
    lodging_result: LodgingStrategyResult
    preferences: LodgingStrategyPreferences
    fixed_hotel: FixedHotelInput | None = None
    decision_mode: HotelDecisionMode = HotelDecisionMode.AWAIT_USER
    chosen_hotel_place_id: UUID | None = None
    candidate_count: Literal[7] = 7

    @model_validator(mode="after")
    def dates_strategy_and_fixed_hotel_are_consistent(self) -> HotelSelectionRequest:
        if self.check_out < self.check_in:
            raise ValueError("hotel check_out must not precede check_in")
        night_count = (self.check_out - self.check_in).days
        if not 0 <= night_count <= 4:
            raise ValueError("hotel selection must cover zero to four nights")
        if (
            self.lodging_result.trip_id != self.trip_id
            or self.lodging_result.city_id != self.city_id
            or self.lodging_result.input_state_version != self.input_state_version
        ):
            raise ValueError("hotel selection must use the current trip lodging result")
        if self.lodging_result.night_count != night_count:
            raise ValueError("hotel search nights must match the lodging strategy nights")
        if night_count == 0:
            if self.lodging_result.selected_strategy_id is not None:
                raise ValueError("a day trip cannot contain a selected lodging strategy")
            if (
                self.fixed_hotel is not None
                or self.decision_mode is not HotelDecisionMode.NOT_REQUIRED
                or self.chosen_hotel_place_id is not None
            ):
                raise ValueError("a day trip cannot select or request a hotel")
            return self
        selected = next(
            (
                item
                for item in self.lodging_result.strategies
                if item.strategy_id == self.lodging_result.selected_strategy_id
            ),
            None,
        )
        if selected is None:
            raise ValueError("hotel selection requires an existing selected lodging strategy")
        if selected.kind is LodgingBaseKind.FIXED_HOTEL:
            if self.fixed_hotel is None:
                raise ValueError("a fixed lodging strategy requires the booked hotel identity")
            if self.fixed_hotel.node_id != selected.fixed_hotel_node_id:
                raise ValueError("booked hotel must match the fixed lodging strategy anchor")
            if (
                self.decision_mode is not HotelDecisionMode.PREBOOKED
                or self.chosen_hotel_place_id is not None
            ):
                raise ValueError("a booked hotel must use the prebooked decision mode")
        elif self.fixed_hotel is not None:
            raise ValueError("only a fixed lodging strategy may contain a booked hotel")
        elif self.decision_mode is HotelDecisionMode.PREBOOKED:
            raise ValueError("prebooked decision mode requires a fixed lodging strategy")
        elif self.decision_mode is HotelDecisionMode.USER_CHOICE:
            if self.chosen_hotel_place_id is None:
                raise ValueError("user hotel choice requires one chosen candidate")
        elif self.chosen_hotel_place_id is not None:
            raise ValueError("only user choice may provide a chosen hotel candidate")
        if self.decision_mode is HotelDecisionMode.NOT_REQUIRED:
            raise ValueError("an overnight trip requires a hotel decision mode")
        return self

    @property
    def night_count(self) -> int:
        return (self.check_out - self.check_in).days


class HotelSelectionScore(ImmutableHotelSelectionModel):
    strategy_match: int = Field(ge=0, le=100, strict=True)
    commute_fit: int = Field(ge=0, le=100, strict=True)
    price_fit: int = Field(ge=0, le=100, strict=True)
    rating_fit: int = Field(ge=0, le=100, strict=True)
    fact_completeness: int = Field(ge=0, le=100, strict=True)
    preference_boost: int = Field(ge=0, le=100, strict=True)
    total: int = Field(ge=0, le=100, strict=True)


class HotelSelectionCandidate(ImmutableHotelSelectionModel):
    hotel_place_id: UUID
    strategy_id: UUID
    provider: ProviderCode
    source_hotel_id: NonEmptyText
    source_offer_id: NonEmptyText
    name: NonEmptyText
    hotel_type: NonEmptyText | None = None
    brand_name: NonEmptyText | None = None
    price_band: HotelPriceBand | None = None
    address: NonEmptyText | None = None
    coordinates: Gcj02Coordinates | None = None
    check_in: date
    check_out: date
    night_count: int = Field(ge=1, le=4, strict=True)
    availability: DataAvailability
    room_price: CnyAmountRange | None = None
    rating: float | None = Field(default=None, ge=0, le=5, allow_inf_nan=False)
    image_urls: tuple[HttpUrl, ...] = ()
    detail_url: HttpUrl | None = None
    distance_to_strategy_center_m: int | None = Field(default=None, ge=0, strict=True)
    missing_fields: tuple[NonEmptyText, ...] = ()
    missing_reason: ShortText | None = None
    score: HotelSelectionScore
    fit_reason: ShortText
    source_reference_ids: tuple[NonEmptyText, ...] = Field(min_length=1)
    fetched_at: AwareDatetime

    @model_validator(mode="after")
    def dates_availability_and_sources_are_consistent(self) -> HotelSelectionCandidate:
        if (self.check_out - self.check_in).days != self.night_count:
            raise ValueError("hotel candidate nights must match its date range")
        _unique(self.missing_fields, "hotel candidate missing fields")
        _unique(self.source_reference_ids, "hotel candidate source references")
        present = {
            "hotel_type": self.hotel_type is not None,
            "price_band": self.price_band is not None,
            "address": self.address is not None,
            "coordinates": self.coordinates is not None,
            "room_price": self.room_price is not None,
            "rating": self.rating is not None,
            "image_urls": bool(self.image_urls),
            "detail_url": self.detail_url is not None,
        }
        for field in self.missing_fields:
            if present.get(field, False):
                raise ValueError(f"{field} cannot be both present and declared missing")
        if self.availability is DataAvailability.AVAILABLE:
            if not all(present.values()) or self.missing_fields or self.missing_reason is not None:
                raise ValueError(
                    "available hotel candidate requires complete display and route facts"
                )
        elif self.availability is DataAvailability.PARTIAL:
            if not any(present.values()) or not self.missing_fields or self.missing_reason is None:
                raise ValueError(
                    "partial hotel candidate requires data and explicit missing details"
                )
        elif any(present.values()) or not self.missing_fields or self.missing_reason is None:
            raise ValueError("missing hotel candidate requires no usable facts and a reason")
        return self


class HotelRouteBaseline(ImmutableHotelSelectionModel):
    hotel_place_id: UUID
    strategy_id: UUID
    availability: DataAvailability
    coordinates: Gcj02Coordinates | None = None
    source_reference_ids: tuple[NonEmptyText, ...] = Field(min_length=1)
    missing_reason: ShortText | None = None

    @model_validator(mode="after")
    def availability_matches_coordinates(self) -> HotelRouteBaseline:
        _unique(self.source_reference_ids, "hotel route baseline sources")
        if self.availability is DataAvailability.AVAILABLE:
            if self.coordinates is None or self.missing_reason is not None:
                raise ValueError("available hotel route baseline requires coordinates")
        elif self.coordinates is not None or self.missing_reason is None:
            raise ValueError("unavailable hotel route baseline requires a missing reason")
        return self


class HotelSelectionResult(ImmutableHotelSelectionModel):
    algorithm_version: str = Field(pattern=r"^[1-9]\d*\.\d+\.\d+$")
    request_id: UUID
    trip_id: UUID
    input_state_version: int = Field(ge=0, strict=True)
    city_id: NonEmptyText
    check_in: date
    check_out: date
    night_count: int = Field(ge=0, le=4, strict=True)
    selected_strategy_id: UUID | None = None
    status: DataAvailability
    decision_status: HotelDecisionStatus
    target_candidate_count: Literal[7] = 7
    candidates: tuple[HotelSelectionCandidate, ...] = Field(default=(), max_length=7)
    selected_hotel_place_id: UUID | None = None
    selection_source: HotelSelectionSource | None = None
    route_baseline: HotelRouteBaseline | None = None
    provider_failure_code: ProviderFailureCode | None = None
    missing_reason: ShortText | None = None
    degradation_reasons: tuple[ShortText, ...] = ()
    generated_at: AwareDatetime

    @model_validator(mode="after")
    def single_hotel_and_strategy_references_are_consistent(self) -> HotelSelectionResult:
        if (self.check_out - self.check_in).days != self.night_count:
            raise ValueError("hotel selection result nights must match its date range")
        _unique([item.hotel_place_id for item in self.candidates], "hotel result place IDs")
        _unique(self.degradation_reasons, "hotel result degradation reasons")
        if any(
            item.strategy_id != self.selected_strategy_id
            or item.check_in != self.check_in
            or item.check_out != self.check_out
            or item.night_count != self.night_count
            for item in self.candidates
        ):
            raise ValueError("every hotel candidate must use the selected strategy and stay dates")
        if self.night_count == 0:
            if (
                any(
                    value is not None
                    for value in (
                        self.selected_strategy_id,
                        self.selected_hotel_place_id,
                        self.selection_source,
                        self.route_baseline,
                        self.provider_failure_code,
                        self.missing_reason,
                    )
                )
                or self.candidates
                or self.degradation_reasons
                or self.decision_status is not HotelDecisionStatus.NOT_REQUIRED
            ):
                raise ValueError("a day trip cannot contain hotel selection data")
            return self
        if self.selected_strategy_id is None:
            raise ValueError("an overnight hotel result requires a lodging strategy")
        if self.status is DataAvailability.MISSING:
            if (
                self.candidates
                or self.selected_hotel_place_id is not None
                or self.selection_source is not None
                or self.route_baseline is not None
                or self.missing_reason is None
                or self.decision_status is not HotelDecisionStatus.AWAITING_USER
            ):
                raise ValueError("missing hotel result must preserve only the strategy and reason")
            return self
        if not self.candidates:
            raise ValueError("available hotel result requires hotel candidates")
        if self.decision_status is HotelDecisionStatus.AWAITING_USER:
            if any(
                value is not None
                for value in (
                    self.selected_hotel_place_id,
                    self.selection_source,
                    self.route_baseline,
                )
            ):
                raise ValueError("awaiting-user hotel result cannot contain a final hotel")
            if self.status is DataAvailability.AVAILABLE and self.degradation_reasons:
                raise ValueError("available hotel candidates cannot contain degradation")
            if self.status is DataAvailability.PARTIAL and not self.degradation_reasons:
                raise ValueError("partial hotel candidates require explicit degradation reasons")
            return self
        if self.decision_status is not HotelDecisionStatus.FINAL:
            raise ValueError("overnight hotel result must await a user or be final")
        if self.selected_hotel_place_id is None:
            raise ValueError("final hotel result requires one selected hotel")
        selected = next(
            (
                item
                for item in self.candidates
                if item.hotel_place_id == self.selected_hotel_place_id
            ),
            None,
        )
        if selected is None or self.selection_source is None or self.route_baseline is None:
            raise ValueError(
                "selected hotel and route baseline must reference a returned candidate"
            )
        if (
            self.route_baseline.hotel_place_id != selected.hotel_place_id
            or self.route_baseline.strategy_id != selected.strategy_id
        ):
            raise ValueError("hotel route baseline must match the selected hotel and strategy")
        if self.status is DataAvailability.AVAILABLE:
            if (
                selected.availability is not DataAvailability.AVAILABLE
                or self.provider_failure_code is not None
                or self.missing_reason is not None
                or self.degradation_reasons
            ):
                raise ValueError("available hotel result cannot contain degraded selected data")
        elif not self.degradation_reasons:
            raise ValueError("partial hotel result requires explicit degradation reasons")
        return self


def _unique(values: Sequence[object], label: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must be unique")


V3_HOTEL_SELECTION_CONTRACTS: tuple[type[ContractModel], ...] = (
    HotelSelectionRequest,
    HotelSelectionResult,
)
