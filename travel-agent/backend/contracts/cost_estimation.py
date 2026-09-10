"""V3-38 contracts for source-backed per-person trip cost estimation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import date, datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import UUID

from pydantic import AwareDatetime, ConfigDict, Field, model_validator

from backend.contracts.base import ContractModel
from backend.contracts.common import CnyAmountRange, NonEmptyText, ShortText
from backend.contracts.daily_scheduling import (
    DailyScheduleResult,
    DailySchedulingRequest,
    ScheduleActivityKind,
)
from backend.contracts.enums import CostCategory, DataAvailability, ExcludedCostKind

COST_ESTIMATION_REQUEST_SCHEMA_RULE: dict[str, Any] = {"x-travel-cost-estimation-request": True}
TRIP_COST_ESTIMATE_SCHEMA_RULE: dict[str, Any] = {"x-travel-trip-cost-estimate": True}


class ImmutableCostModel(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CostSubjectKind(StrEnum):
    ACTIVITY = "activity"
    TRANSPORT_LEG = "transport_leg"
    HOTEL_NIGHT = "hotel_night"


class CostPriceBasis(StrEnum):
    PER_PERSON = "per_person"
    PER_VEHICLE = "per_vehicle"
    PER_ROOM_NIGHT = "per_room_night"


class CostCoverageStatus(StrEnum):
    AVAILABLE = "available"
    PARTIAL = "partial"
    MISSING = "missing"
    NOT_APPLICABLE = "not_applicable"


class OriginalMoneyRange(ImmutableCostModel):
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    minimum_minor: int = Field(ge=0, strict=True)
    maximum_minor: int = Field(ge=0, strict=True)

    @model_validator(mode="after")
    def range_is_ordered(self) -> OriginalMoneyRange:
        if self.maximum_minor < self.minimum_minor:
            raise ValueError("original money maximum cannot be below minimum")
        return self


class ExchangeRateFact(ImmutableCostModel):
    exchange_rate_id: NonEmptyText
    source_currency: str = Field(pattern=r"^[A-Z]{3}$")
    target_currency: str = Field(default="CNY", pattern=r"^CNY$")
    cny_fen_numerator: int = Field(gt=0, strict=True)
    source_minor_denominator: int = Field(gt=0, strict=True)
    source_reference_ids: tuple[NonEmptyText, ...] = Field(min_length=1)
    fetched_at: AwareDatetime

    @model_validator(mode="after")
    def exchange_sources_are_unique(self) -> ExchangeRateFact:
        if self.source_currency == self.target_currency:
            raise ValueError("CNY prices do not require an exchange-rate fact")
        _unique(self.source_reference_ids, "exchange-rate sources")
        return self


class CostPriceFact(ImmutableCostModel):
    price_fact_id: NonEmptyText
    subject_kind: CostSubjectKind
    subject_id: UUID
    service_date: date
    category: CostCategory
    basis: CostPriceBasis
    availability: DataAvailability
    original_amount: OriginalMoneyRange | None = None
    source_reference_ids: tuple[NonEmptyText, ...] = Field(min_length=1)
    fetched_at: AwareDatetime
    missing_reason: ShortText | None = None

    @model_validator(mode="after")
    def price_semantics_are_consistent(self) -> CostPriceFact:
        _unique(self.source_reference_ids, "price-fact sources")
        expected_basis = {
            CostCategory.ATTRACTION_TICKETS: CostPriceBasis.PER_PERSON,
            CostCategory.DINING: CostPriceBasis.PER_PERSON,
            CostCategory.LODGING: CostPriceBasis.PER_ROOM_NIGHT,
        }.get(self.category)
        if expected_basis is not None and self.basis is not expected_basis:
            raise ValueError("price basis does not match its cost category")
        if self.availability is DataAvailability.MISSING:
            if self.original_amount is not None or self.missing_reason is None:
                raise ValueError("missing price requires no amount and an explicit reason")
        elif self.original_amount is None:
            raise ValueError("usable price requires an original amount")
        elif self.availability is DataAvailability.AVAILABLE and self.missing_reason is not None:
            raise ValueError("available price cannot contain a missing reason")
        elif self.availability is DataAvailability.PARTIAL and self.missing_reason is None:
            raise ValueError("partial price requires an explicit missing reason")
        return self


class CostEstimationRequest(ImmutableCostModel):
    model_config = ConfigDict(json_schema_extra=COST_ESTIMATION_REQUEST_SCHEMA_RULE)

    request_id: UUID
    trip_id: UUID
    input_state_version: int = Field(ge=0, strict=True)
    business_time: AwareDatetime
    party_size: int = Field(default=1, ge=1, le=20, strict=True)
    maximum_fact_age_hours: int = Field(default=24, ge=1, le=168, strict=True)
    scheduling_request: DailySchedulingRequest
    schedule_result: DailyScheduleResult
    price_facts: tuple[CostPriceFact, ...]
    exchange_rates: tuple[ExchangeRateFact, ...] = ()

    @model_validator(mode="after")
    def facts_cover_the_current_schedule(self) -> CostEstimationRequest:
        if (
            self.scheduling_request.trip_id != self.trip_id
            or self.scheduling_request.input_state_version != self.input_state_version
            or self.schedule_result.trip_id != self.trip_id
            or self.schedule_result.input_state_version != self.input_state_version
            or self.schedule_result.request_id != self.scheduling_request.request_id
        ):
            raise ValueError("cost estimation requires the current scheduling request and result")
        _unique([item.price_fact_id for item in self.price_facts], "price fact IDs")
        _unique(
            [
                (item.subject_kind, item.subject_id, item.service_date, item.category)
                for item in self.price_facts
            ],
            "price subject keys",
        )
        _unique([item.exchange_rate_id for item in self.exchange_rates], "exchange-rate IDs")
        _unique([item.source_currency for item in self.exchange_rates], "exchange currencies")
        activity_by_id = {
            item.activity_id: item for day in self.schedule_result.days for item in day.activities
        }
        leg_by_id = {
            item.leg_id: (day.service_date, item)
            for day in self.schedule_result.days
            for item in day.transport_legs
        }
        selected_hotel_id = self.scheduling_request.hotel_result.selected_hotel_place_id
        hotel_dates = {
            self.schedule_result.start_date.fromordinal(day)
            for day in range(
                self.schedule_result.start_date.toordinal(),
                self.schedule_result.end_date.toordinal(),
            )
        }
        expected_keys: set[tuple[CostSubjectKind, UUID, date, CostCategory]] = set()
        for day in self.schedule_result.days:
            for activity in day.activities:
                category = _activity_cost_category(activity.kind)
                if category is not None:
                    expected_keys.add(
                        (CostSubjectKind.ACTIVITY, activity.activity_id, day.service_date, category)
                    )
            for leg in day.transport_legs:
                expected_keys.add(
                    (
                        CostSubjectKind.TRANSPORT_LEG,
                        leg.leg_id,
                        day.service_date,
                        CostCategory.LOCAL_TRANSPORT,
                    )
                )
        if selected_hotel_id is not None:
            expected_keys.update(
                (
                    CostSubjectKind.HOTEL_NIGHT,
                    selected_hotel_id,
                    service_date,
                    CostCategory.LODGING,
                )
                for service_date in hotel_dates
            )
        actual_keys = {
            (item.subject_kind, item.subject_id, item.service_date, item.category)
            for item in self.price_facts
        }
        if actual_keys != expected_keys:
            raise ValueError("price facts must cover every priced schedule subject exactly once")
        for fact in self.price_facts:
            if fact.subject_kind is CostSubjectKind.ACTIVITY:
                referenced_activity = activity_by_id.get(fact.subject_id)
                if (
                    referenced_activity is None
                    or referenced_activity.service_date != fact.service_date
                ):
                    raise ValueError("activity price fact references an unknown scheduled activity")
            elif fact.subject_kind is CostSubjectKind.TRANSPORT_LEG:
                referenced_leg = leg_by_id.get(fact.subject_id)
                if referenced_leg is None or referenced_leg[0] != fact.service_date:
                    raise ValueError("transport price fact references an unknown scheduled leg")
            elif (
                selected_hotel_id is None
                or fact.subject_id != selected_hotel_id
                or fact.service_date not in hotel_dates
            ):
                raise ValueError("hotel price fact must reference the final hotel and a stay night")
            _validate_fact_freshness(
                fact.fetched_at,
                self.business_time,
                self.maximum_fact_age_hours,
                "price fact",
            )
        rates = {item.source_currency: item for item in self.exchange_rates}
        for rate in self.exchange_rates:
            _validate_fact_freshness(
                rate.fetched_at,
                self.business_time,
                self.maximum_fact_age_hours,
                "exchange-rate fact",
            )
        currencies = {
            fact.original_amount.currency
            for fact in self.price_facts
            if fact.original_amount is not None and fact.original_amount.currency != "CNY"
        }
        if currencies != set(rates):
            raise ValueError("every non-CNY price requires exactly one current exchange rate")
        return self


class CostEstimateLine(ImmutableCostModel):
    price_fact_id: NonEmptyText
    subject_kind: CostSubjectKind
    subject_id: UUID
    service_date: date
    category: CostCategory
    basis: CostPriceBasis
    share_divisor: int = Field(ge=1, strict=True)
    availability: DataAvailability
    amount_per_person: CnyAmountRange | None = None
    original_amount: OriginalMoneyRange | None = None
    exchange_rate_id: NonEmptyText | None = None
    exchange_rate_fetched_at: AwareDatetime | None = None
    source_reference_ids: tuple[NonEmptyText, ...] = Field(min_length=1)
    fetched_at: AwareDatetime
    missing_reason: ShortText | None = None

    @model_validator(mode="after")
    def amount_and_sources_match_status(self) -> CostEstimateLine:
        _unique(self.source_reference_ids, "cost-line sources")
        if self.availability is DataAvailability.MISSING:
            if (
                any(
                    value is not None
                    for value in (
                        self.amount_per_person,
                        self.original_amount,
                        self.exchange_rate_id,
                        self.exchange_rate_fetched_at,
                    )
                )
                or self.missing_reason is None
            ):
                raise ValueError("missing cost line cannot fabricate an amount")
        elif self.amount_per_person is None or self.original_amount is None:
            raise ValueError("usable cost line requires original and per-person amounts")
        elif self.original_amount.currency == "CNY" and (
            self.exchange_rate_id is not None or self.exchange_rate_fetched_at is not None
        ):
            raise ValueError("CNY cost line cannot claim an exchange rate")
        elif self.original_amount.currency != "CNY" and (
            self.exchange_rate_id is None or self.exchange_rate_fetched_at is None
        ):
            raise ValueError("converted cost line requires its exchange-rate reference")
        elif self.availability is DataAvailability.AVAILABLE and self.missing_reason is not None:
            raise ValueError("available cost line cannot contain a missing reason")
        elif self.availability is DataAvailability.PARTIAL and self.missing_reason is None:
            raise ValueError("partial cost line requires a reason")
        return self


class CategoryCostSummary(ImmutableCostModel):
    category: CostCategory
    status: CostCoverageStatus
    amount_per_person: CnyAmountRange | None = None
    priced_item_count: int = Field(ge=0, strict=True)
    missing_item_count: int = Field(ge=0, strict=True)
    source_reference_ids: tuple[NonEmptyText, ...] = ()
    note: ShortText | None = None

    @model_validator(mode="after")
    def coverage_matches_amount(self) -> CategoryCostSummary:
        _unique(self.source_reference_ids, "cost-summary sources")
        if self.status is CostCoverageStatus.NOT_APPLICABLE:
            if (
                self.amount_per_person is not None
                or self.priced_item_count
                or self.missing_item_count
                or self.source_reference_ids
                or self.note is None
            ):
                raise ValueError("not-applicable cost requires only an explanation")
        elif self.status is CostCoverageStatus.MISSING:
            if (
                self.amount_per_person is not None
                or self.priced_item_count
                or not self.missing_item_count
                or self.note is None
            ):
                raise ValueError("missing category cost requires missing items and no amount")
        elif self.amount_per_person is None or not self.priced_item_count:
            raise ValueError("available category cost requires priced amounts")
        elif self.status is CostCoverageStatus.AVAILABLE:
            if self.missing_item_count or self.note is not None:
                raise ValueError("available category cost cannot contain missing items")
        elif not self.missing_item_count or self.note is None:
            raise ValueError("partial category cost requires priced and missing items")
        return self


class DailyCostEstimate(ImmutableCostModel):
    service_date: date
    lines: tuple[CostEstimateLine, ...]
    categories: tuple[CategoryCostSummary, ...] = Field(min_length=4, max_length=4)
    known_subtotal_per_person: CnyAmountRange | None = None

    @model_validator(mode="after")
    def categories_and_subtotal_are_exact(self) -> DailyCostEstimate:
        _unique([item.price_fact_id for item in self.lines], "daily price fact IDs")
        _validate_category_set(self.categories)
        _validate_summaries(self.categories, self.lines)
        if any(item.service_date != self.service_date for item in self.lines):
            raise ValueError("daily cost lines must belong to their service date")
        expected = _sum_ranges(
            item.amount_per_person for item in self.categories if item.amount_per_person is not None
        )
        if self.known_subtotal_per_person != expected:
            raise ValueError("daily known subtotal must equal category amounts")
        return self


class TripCostEstimate(ImmutableCostModel):
    model_config = ConfigDict(json_schema_extra=TRIP_COST_ESTIMATE_SCHEMA_RULE)

    algorithm_version: str = Field(pattern=r"^[1-9]\d*\.\d+\.\d+$")
    request_id: UUID
    schedule_request_id: UUID
    trip_id: UUID
    input_state_version: int = Field(ge=0, strict=True)
    task_book_id: UUID
    task_book_revision: int = Field(ge=1, strict=True)
    start_date: date
    end_date: date
    currency: str = Field(default="CNY", pattern=r"^CNY$")
    per_person: Literal[True] = True
    party_size: int = Field(ge=1, le=20, strict=True)
    lodging_share_divisor: Literal[2] = 2
    days: tuple[DailyCostEstimate, ...] = Field(min_length=1, max_length=5)
    categories: tuple[CategoryCostSummary, ...] = Field(min_length=4, max_length=4)
    known_total_per_person: CnyAmountRange | None = None
    excluded_costs: tuple[ExcludedCostKind, ...]
    pricing_note: ShortText
    generated_at: AwareDatetime

    @model_validator(mode="after")
    def totals_and_scope_are_exact(self) -> TripCostEstimate:
        _unique([item.service_date for item in self.days], "cost-estimate dates")
        expected_dates = tuple(
            self.start_date.fromordinal(day)
            for day in range(self.start_date.toordinal(), self.end_date.toordinal() + 1)
        )
        if tuple(item.service_date for item in self.days) != expected_dates:
            raise ValueError("daily costs must cover the trip exactly once and in order")
        _unique(self.excluded_costs, "excluded costs")
        if set(self.excluded_costs) != set(ExcludedCostKind):
            raise ValueError("airfare, rail and intercity transport must remain excluded")
        _validate_category_set(self.categories)
        lines = tuple(line for day in self.days for line in day.lines)
        _unique([item.price_fact_id for item in lines], "trip price fact IDs")
        _validate_summaries(self.categories, lines)
        for line in lines:
            expected_divisor = (
                self.party_size
                if line.basis is CostPriceBasis.PER_VEHICLE
                else 2
                if line.basis is CostPriceBasis.PER_ROOM_NIGHT
                else 1
            )
            if line.share_divisor != expected_divisor:
                raise ValueError("cost-line share divisor must follow the trip pricing basis")
        daily_total = _sum_ranges(
            day.known_subtotal_per_person
            for day in self.days
            if day.known_subtotal_per_person is not None
        )
        category_total = _sum_ranges(
            item.amount_per_person for item in self.categories if item.amount_per_person is not None
        )
        if (
            self.known_total_per_person != daily_total
            or self.known_total_per_person != category_total
        ):
            raise ValueError("trip total must equal both daily and category totals")
        return self


def _activity_cost_category(kind: ScheduleActivityKind) -> CostCategory | None:
    if kind is ScheduleActivityKind.ATTRACTION:
        return CostCategory.ATTRACTION_TICKETS
    if kind is ScheduleActivityKind.RESTAURANT:
        return CostCategory.DINING
    return None


def _validate_fact_freshness(
    fetched_at: datetime,
    business_time: datetime,
    maximum_age_hours: int,
    label: str,
) -> None:
    if fetched_at > business_time:
        raise ValueError(f"{label} cannot come from the future")
    age_seconds = (business_time - fetched_at).total_seconds()
    if age_seconds > maximum_age_hours * 3600:
        raise ValueError(f"{label} is older than the accepted estimation window")


def _validate_category_set(categories: Sequence[CategoryCostSummary]) -> None:
    if {item.category for item in categories} != set(CostCategory):
        raise ValueError("cost summary must contain each supported category exactly once")


def _validate_summaries(
    categories: Sequence[CategoryCostSummary],
    lines: Sequence[CostEstimateLine],
) -> None:
    for summary in categories:
        applicable = tuple(item for item in lines if item.category is summary.category)
        priced = tuple(item for item in applicable if item.amount_per_person is not None)
        missing = tuple(
            item
            for item in applicable
            if item.amount_per_person is None or item.availability is DataAvailability.PARTIAL
        )
        expected_status = (
            CostCoverageStatus.NOT_APPLICABLE
            if not applicable
            else CostCoverageStatus.MISSING
            if not priced
            else CostCoverageStatus.PARTIAL
            if missing
            else CostCoverageStatus.AVAILABLE
        )
        expected_sources = {source for item in applicable for source in item.source_reference_ids}
        if (
            summary.status is not expected_status
            or summary.amount_per_person
            != _sum_ranges(
                item.amount_per_person for item in priced if item.amount_per_person is not None
            )
            or summary.priced_item_count != len(priced)
            or summary.missing_item_count != len(missing)
            or set(summary.source_reference_ids) != expected_sources
        ):
            raise ValueError("cost category summary must be derived from its cost lines")


def _sum_ranges(values: Iterable[CnyAmountRange]) -> CnyAmountRange | None:
    materialized = tuple(values)
    if not materialized:
        return None
    return CnyAmountRange(
        minimum_fen=sum(item.minimum_fen for item in materialized),
        maximum_fen=sum(item.maximum_fen for item in materialized),
    )


def _unique(values: Sequence[object], label: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must be unique")


V3_COST_ESTIMATION_CONTRACTS: tuple[type[ContractModel], ...] = (
    CostEstimationRequest,
    TripCostEstimate,
)
