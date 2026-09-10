"""Permissive V3-39 draft contracts inspected before strict publication."""

from __future__ import annotations

from datetime import date, time
from uuid import UUID

from pydantic import AwareDatetime, ConfigDict

from backend.contracts.base import ContractModel
from backend.contracts.common import NonEmptyText, ShortText
from backend.contracts.cost_estimation import (
    CostCoverageStatus,
    CostPriceBasis,
    CostSubjectKind,
    TripCostEstimate,
)
from backend.contracts.daily_scheduling import (
    DailyScheduleResult,
    ScheduleActivityKind,
    SchedulePauseKind,
)
from backend.contracts.enums import (
    AnchorRole,
    CostCategory,
    DataAvailability,
    ExcludedCostKind,
)
from backend.providers.contracts import RouteMode


class ImmutableDraftModel(ContractModel):
    """Drafts remain immutable but deliberately omit semantic cross-field validators."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class DraftAmountRange(ImmutableDraftModel):
    currency: str
    minimum_fen: int
    maximum_fen: int


class DraftScheduledActivity(ImmutableDraftModel):
    activity_id: UUID
    node_id: UUID
    place_id: UUID
    kind: ScheduleActivityKind
    role: AnchorRole
    title: NonEmptyText
    service_date: date
    start_time: time
    end_time: time
    duration_minutes: int
    availability: DataAvailability
    source_reference_ids: tuple[NonEmptyText, ...] = ()
    missing_reason: ShortText | None = None
    timing_notice: ShortText | None = None


class DraftScheduledPause(ImmutableDraftModel):
    pause_id: UUID
    kind: SchedulePauseKind
    service_date: date
    start_time: time
    end_time: time
    duration_minutes: int
    reason: ShortText


class DraftScheduledTransport(ImmutableDraftModel):
    leg_id: UUID
    origin_place_id: UUID
    destination_place_id: UUID
    departure_time: time
    arrival_time: time
    mode: RouteMode
    availability: DataAvailability
    distance_m: int
    duration_minutes: int
    walking_m: int
    buffer_minutes: int
    source_reference_ids: tuple[NonEmptyText, ...] = ()
    missing_reason: ShortText | None = None


class DraftScheduledDay(ImmutableDraftModel):
    service_date: date
    start_place_id: UUID
    end_place_id: UUID
    start_time: time
    end_time: time
    activities: tuple[DraftScheduledActivity, ...] = ()
    pauses: tuple[DraftScheduledPause, ...] = ()
    transport_legs: tuple[DraftScheduledTransport, ...] = ()
    active_minutes: int
    walking_m: int
    cycling_m: int = 0
    meal_minutes: int
    rest_minutes: int
    buffer_minutes: int


class DraftUnscheduledStrongDesire(ImmutableDraftModel):
    node_id: UUID
    place_id: UUID
    role: AnchorRole
    reason: ShortText
    source_reference_ids: tuple[NonEmptyText, ...] = ()


class ScheduleValidationDraft(ImmutableDraftModel):
    algorithm_version: str
    request_id: UUID
    trip_id: UUID
    input_state_version: int
    task_book_id: UUID
    task_book_revision: int
    city_id: NonEmptyText
    start_date: date
    end_date: date
    status: DataAvailability
    days: tuple[DraftScheduledDay, ...]
    unscheduled_strong_desires: tuple[DraftUnscheduledStrongDesire, ...] = ()
    degradation_reasons: tuple[ShortText, ...] = ()
    provider_fact_ids: tuple[NonEmptyText, ...] = ()
    generated_at: AwareDatetime

    @classmethod
    def from_result(cls, result: DailyScheduleResult) -> ScheduleValidationDraft:
        return cls.model_validate(result.model_dump(mode="json"))


class DraftCostEstimateLine(ImmutableDraftModel):
    price_fact_id: NonEmptyText
    subject_kind: CostSubjectKind
    subject_id: UUID
    service_date: date
    category: CostCategory
    basis: CostPriceBasis
    share_divisor: int
    availability: DataAvailability
    amount_per_person: DraftAmountRange | None = None
    original_amount: dict[str, object] | None = None
    exchange_rate_id: NonEmptyText | None = None
    exchange_rate_fetched_at: AwareDatetime | None = None
    source_reference_ids: tuple[NonEmptyText, ...] = ()
    fetched_at: AwareDatetime
    missing_reason: ShortText | None = None


class DraftCategoryCostSummary(ImmutableDraftModel):
    category: CostCategory
    status: CostCoverageStatus
    amount_per_person: DraftAmountRange | None = None
    priced_item_count: int
    missing_item_count: int
    source_reference_ids: tuple[NonEmptyText, ...] = ()
    note: ShortText | None = None


class DraftDailyCostEstimate(ImmutableDraftModel):
    service_date: date
    lines: tuple[DraftCostEstimateLine, ...] = ()
    categories: tuple[DraftCategoryCostSummary, ...] = ()
    known_subtotal_per_person: DraftAmountRange | None = None


class CostValidationDraft(ImmutableDraftModel):
    algorithm_version: str
    request_id: UUID
    schedule_request_id: UUID
    trip_id: UUID
    input_state_version: int
    task_book_id: UUID
    task_book_revision: int
    start_date: date
    end_date: date
    currency: str
    per_person: bool
    party_size: int
    lodging_share_divisor: int
    days: tuple[DraftDailyCostEstimate, ...]
    categories: tuple[DraftCategoryCostSummary, ...]
    known_total_per_person: DraftAmountRange | None = None
    excluded_costs: tuple[ExcludedCostKind, ...] = ()
    pricing_note: ShortText
    generated_at: AwareDatetime

    @classmethod
    def from_estimate(cls, estimate: TripCostEstimate) -> CostValidationDraft:
        return cls.model_validate(estimate.model_dump(mode="json"))


V3_ITINERARY_DRAFT_CONTRACTS: tuple[type[ContractModel], ...] = (
    ScheduleValidationDraft,
    CostValidationDraft,
)
