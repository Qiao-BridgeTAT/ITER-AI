"""Deterministic V3-38 conversion and aggregation of source-backed trip costs."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import UTC, date, datetime
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal

from backend.contracts.common import CnyAmountRange
from backend.contracts.cost_estimation import (
    CategoryCostSummary,
    CostCoverageStatus,
    CostEstimateLine,
    CostEstimationRequest,
    CostPriceBasis,
    CostPriceFact,
    DailyCostEstimate,
    ExchangeRateFact,
    TripCostEstimate,
)
from backend.contracts.enums import CostCategory, DataAvailability, ExcludedCostKind

COST_ESTIMATION_ALGORITHM_VERSION = "1.0.0"
_CATEGORIES = tuple(CostCategory)


class CostEstimationService:
    """Build an auditable per-person CNY estimate without inventing missing prices."""

    def __init__(
        self,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._clock = clock

    def estimate(self, request: CostEstimationRequest) -> TripCostEstimate:
        request = CostEstimationRequest.model_validate(
            request.model_dump(mode="json"),
            context={"today": request.scheduling_request.business_date},
        )
        rates = {item.source_currency: item for item in request.exchange_rates}
        lines = tuple(
            self._line(fact, party_size=request.party_size, rates=rates)
            for fact in request.price_facts
        )
        days = tuple(
            self._day(service_date, lines)
            for service_date in _trip_dates(
                request.schedule_result.start_date,
                request.schedule_result.end_date,
            )
        )
        categories = tuple(_summarize(category, lines) for category in _CATEGORIES)
        generated_at = self._clock()
        if generated_at.tzinfo is None or generated_at.utcoffset() is None:
            raise ValueError("cost estimation clock must return an aware datetime")
        return TripCostEstimate(
            algorithm_version=COST_ESTIMATION_ALGORITHM_VERSION,
            request_id=request.request_id,
            schedule_request_id=request.schedule_result.request_id,
            trip_id=request.trip_id,
            input_state_version=request.input_state_version,
            task_book_id=request.schedule_result.task_book_id,
            task_book_revision=request.schedule_result.task_book_revision,
            start_date=request.schedule_result.start_date,
            end_date=request.schedule_result.end_date,
            party_size=request.party_size,
            days=days,
            categories=categories,
            known_total_per_person=_sum_ranges(
                item.known_subtotal_per_person
                for item in days
                if item.known_subtotal_per_person is not None
            ),
            excluded_costs=tuple(ExcludedCostKind),
            pricing_note=(
                "按每人估算：餐饮和门票使用人均价，本地交通按同行人数分摊，"
                "住宿每晚按两人分摊；不含往返大交通。缺失价格不会按零元计算。"
            ),
            generated_at=generated_at.astimezone(UTC),
        )

    def _day(
        self,
        service_date: date,
        all_lines: tuple[CostEstimateLine, ...],
    ) -> DailyCostEstimate:
        lines = tuple(item for item in all_lines if item.service_date == service_date)
        categories = tuple(_summarize(category, lines) for category in _CATEGORIES)
        return DailyCostEstimate(
            service_date=service_date,
            lines=lines,
            categories=categories,
            known_subtotal_per_person=_sum_ranges(
                item.amount_per_person for item in categories if item.amount_per_person is not None
            ),
        )

    def _line(
        self,
        fact: CostPriceFact,
        *,
        party_size: int,
        rates: dict[str, ExchangeRateFact],
    ) -> CostEstimateLine:
        if fact.availability is DataAvailability.MISSING:
            return CostEstimateLine(
                price_fact_id=fact.price_fact_id,
                subject_kind=fact.subject_kind,
                subject_id=fact.subject_id,
                service_date=fact.service_date,
                category=fact.category,
                basis=fact.basis,
                share_divisor=_share_divisor(fact.basis, party_size),
                availability=fact.availability,
                source_reference_ids=fact.source_reference_ids,
                fetched_at=fact.fetched_at,
                missing_reason=fact.missing_reason,
            )
        original = fact.original_amount
        if original is None:  # defended by the request contract
            raise ValueError("usable price fact is missing its amount")
        rate = rates.get(original.currency)
        converted = _convert_to_cny(original.minimum_minor, original.maximum_minor, rate)
        divisor = _share_divisor(fact.basis, party_size)
        amount = CnyAmountRange(
            minimum_fen=_floor_div(converted.minimum_fen, divisor),
            maximum_fen=_ceil_div(converted.maximum_fen, divisor),
        )
        sources = tuple(
            dict.fromkeys(
                (
                    *fact.source_reference_ids,
                    *(rate.source_reference_ids if rate is not None else ()),
                )
            )
        )
        return CostEstimateLine(
            price_fact_id=fact.price_fact_id,
            subject_kind=fact.subject_kind,
            subject_id=fact.subject_id,
            service_date=fact.service_date,
            category=fact.category,
            basis=fact.basis,
            share_divisor=divisor,
            availability=fact.availability,
            amount_per_person=amount,
            original_amount=original,
            exchange_rate_id=rate.exchange_rate_id if rate is not None else None,
            exchange_rate_fetched_at=rate.fetched_at if rate is not None else None,
            source_reference_ids=sources,
            fetched_at=fact.fetched_at,
            missing_reason=fact.missing_reason,
        )


def _summarize(
    category: CostCategory,
    lines: Iterable[CostEstimateLine],
) -> CategoryCostSummary:
    applicable = tuple(item for item in lines if item.category is category)
    if not applicable:
        return CategoryCostSummary(
            category=category,
            status=CostCoverageStatus.NOT_APPLICABLE,
            priced_item_count=0,
            missing_item_count=0,
            note="本日没有这一类已选项目。",
        )
    priced = tuple(item for item in applicable if item.amount_per_person is not None)
    missing = tuple(
        item
        for item in applicable
        if item.amount_per_person is None or item.availability is DataAvailability.PARTIAL
    )
    sources = tuple(
        dict.fromkeys(source for item in applicable for source in item.source_reference_ids)
    )
    if not priced:
        return CategoryCostSummary(
            category=category,
            status=CostCoverageStatus.MISSING,
            priced_item_count=0,
            missing_item_count=len(missing),
            source_reference_ids=sources,
            note="当前没有可用价格，未按零元计入。",
        )
    if missing:
        return CategoryCostSummary(
            category=category,
            status=CostCoverageStatus.PARTIAL,
            amount_per_person=_sum_ranges(item.amount_per_person for item in priced),
            priced_item_count=len(priced),
            missing_item_count=len(missing),
            source_reference_ids=sources,
            note="仅合计已有来源的价格，缺失部分未按零元计入。",
        )
    return CategoryCostSummary(
        category=category,
        status=CostCoverageStatus.AVAILABLE,
        amount_per_person=_sum_ranges(item.amount_per_person for item in priced),
        priced_item_count=len(priced),
        missing_item_count=0,
        source_reference_ids=sources,
    )


def _convert_to_cny(
    minimum_minor: int,
    maximum_minor: int,
    rate: ExchangeRateFact | None,
) -> CnyAmountRange:
    if rate is None:
        return CnyAmountRange(minimum_fen=minimum_minor, maximum_fen=maximum_minor)
    denominator = Decimal(rate.source_minor_denominator)
    factor = Decimal(rate.cny_fen_numerator) / denominator
    return CnyAmountRange(
        minimum_fen=int((Decimal(minimum_minor) * factor).to_integral_value(ROUND_FLOOR)),
        maximum_fen=int((Decimal(maximum_minor) * factor).to_integral_value(ROUND_CEILING)),
    )


def _share_divisor(basis: CostPriceBasis, party_size: int) -> int:
    if basis is CostPriceBasis.PER_VEHICLE:
        return party_size
    if basis is CostPriceBasis.PER_ROOM_NIGHT:
        return 2
    return 1


def _floor_div(value: int, divisor: int) -> int:
    return value // divisor


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _sum_ranges(values: Iterable[CnyAmountRange | None]) -> CnyAmountRange | None:
    materialized = tuple(item for item in values if item is not None)
    if not materialized:
        return None
    return CnyAmountRange(
        minimum_fen=sum(item.minimum_fen for item in materialized),
        maximum_fen=sum(item.maximum_fen for item in materialized),
    )


def _trip_dates(start_date: date, end_date: date) -> tuple[date, ...]:
    return tuple(
        start_date.fromordinal(day)
        for day in range(start_date.toordinal(), end_date.toordinal() + 1)
    )
