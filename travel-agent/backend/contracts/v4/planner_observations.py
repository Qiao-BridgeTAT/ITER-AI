"""Typed evidence requests and safe Planner observations for V4."""

from __future__ import annotations

from collections import Counter
from datetime import date, time
from typing import Literal

from pydantic import AwareDatetime, Field, model_validator

from backend.contracts.common import CnyAmountRange
from backend.contracts.places import Gcj02Coordinates
from backend.contracts.v4.base import (
    Digest,
    DisplayText,
    Identifier,
    V4ContractModel,
    require_unique,
)
from backend.contracts.v4.enums import (
    AskUserReasonCode,
    CandidateEntityKind,
    CrossClusterReasonCode,
    InteractionStatus,
    PlannerCapability,
)
from backend.contracts.v4.planner_refs import (
    CandidateRef,
    FixedCommitmentRef,
    HotelOfferRef,
    PlannerScope,
    candidate_ref_key,
)


class CandidateRecallArguments(V4ContractModel):
    domain: CandidateEntityKind
    gap_code: Identifier
    task_book_preference_refs: tuple[Identifier, ...] = Field(min_length=1)
    nearby_candidate_refs: tuple[CandidateRef, ...] = ()
    nearby_cluster_refs: tuple[Identifier, ...] = ()
    limit: int = Field(ge=1, le=20, strict=True)


class PlaceFactsArguments(V4ContractModel):
    candidate_refs: tuple[CandidateRef, ...] = Field(min_length=1, max_length=20)
    fact_kinds: tuple[Identifier, ...] = Field(min_length=1)


class OpeningHoursArguments(V4ContractModel):
    candidate_refs: tuple[CandidateRef, ...] = Field(min_length=1, max_length=20)
    service_dates: tuple[date, ...] = Field(min_length=1, max_length=5)


class TicketAvailabilityArguments(V4ContractModel):
    candidate_refs: tuple[CandidateRef, ...] = Field(min_length=1, max_length=20)
    service_dates: tuple[date, ...] = Field(min_length=1, max_length=5)
    party_size_ref: Identifier


class WeatherForecastArguments(V4ContractModel):
    destination_ref: Identifier
    service_dates: tuple[date, ...] = Field(min_length=1, max_length=5)
    weather_fields: tuple[Identifier, ...] = Field(min_length=1)


class SpatialRouteEndpoint(V4ContractModel):
    kind: Literal["candidate", "cluster", "hotel_offer", "fixed_commitment"]
    reference_id: Identifier


class SpatialRoutePair(V4ContractModel):
    origin: SpatialRouteEndpoint
    destination: SpatialRouteEndpoint

    @model_validator(mode="after")
    def endpoints_are_distinct(self) -> SpatialRoutePair:
        if self.origin == self.destination:
            raise ValueError("route endpoints must be distinct")
        return self


class RouteComparisonDay(V4ContractModel):
    service_date: date
    ordered_endpoints: tuple[SpatialRouteEndpoint, ...] = Field(max_length=24)

    @model_validator(mode="after")
    def uses_exact_objects_not_cluster_proxies(self) -> RouteComparisonDay:
        if any(item.kind == "cluster" for item in self.ordered_endpoints):
            raise ValueError("route comparison requires exact entities, not cluster proxies")
        if any(
            left == right
            for left, right in zip(self.ordered_endpoints, self.ordered_endpoints[1:], strict=False)
        ):
            raise ValueError("route comparison cannot contain repeated adjacent endpoints")
        return self


class RouteComparisonInput(V4ContractModel):
    baseline_days: tuple[RouteComparisonDay, ...] = Field(min_length=1, max_length=5)
    proposed_days: tuple[RouteComparisonDay, ...] = Field(min_length=1, max_length=5)
    transport_mode: Literal["public_transit", "taxi", "walking", "driving"]

    @model_validator(mode="after")
    def compares_same_dates_and_object_multiset(self) -> RouteComparisonInput:
        baseline_dates = [day.service_date for day in self.baseline_days]
        proposed_dates = [day.service_date for day in self.proposed_days]
        if baseline_dates != proposed_dates or baseline_dates != sorted(set(baseline_dates)):
            raise ValueError("route comparison must cover the same ordered unique trip dates")

        def identities(days: tuple[RouteComparisonDay, ...]) -> Counter[tuple[str, str]]:
            return Counter(
                (item.kind, item.reference_id) for day in days for item in day.ordered_endpoints
            )

        if identities(self.baseline_days) != identities(self.proposed_days):
            raise ValueError(
                "route comparison cannot gain savings by removing or replacing objects"
            )
        return self


class SpatialRoutesArguments(V4ContractModel):
    endpoint_pairs: tuple[SpatialRoutePair, ...] = Field(min_length=1, max_length=40)
    transport_modes: tuple[Literal["public_transit", "taxi", "walking", "driving"], ...] = Field(
        min_length=1
    )
    departure_service_date: date
    departure_window: tuple[time, time] | None = None
    comparison: RouteComparisonInput | None = None

    @model_validator(mode="after")
    def pairs_modes_and_window_are_valid(self) -> SpatialRoutesArguments:
        require_unique(
            [
                (
                    item.origin.kind,
                    item.origin.reference_id,
                    item.destination.kind,
                    item.destination.reference_id,
                )
                for item in self.endpoint_pairs
            ],
            "spatial route endpoint pairs",
        )
        require_unique(self.transport_modes, "spatial route transport modes")
        if self.comparison:
            if self.comparison.transport_mode not in self.transport_modes:
                raise ValueError("comparison mode must be queried")
            required_pairs = {
                (left, right)
                for day in (*self.comparison.baseline_days, *self.comparison.proposed_days)
                for left, right in zip(
                    day.ordered_endpoints, day.ordered_endpoints[1:], strict=False
                )
            }
            if not required_pairs <= {
                (pair.origin, pair.destination) for pair in self.endpoint_pairs
            }:
                raise ValueError("every comparison leg must be included in endpoint_pairs")
        if (
            self.departure_window is not None
            and self.departure_window[1] < self.departure_window[0]
        ):
            raise ValueError("route departure window cannot run backwards")
        return self


class HotelSearchArguments(V4ContractModel):
    search_keyword: str | None = Field(default=None, min_length=1, max_length=100)
    check_in_date: date
    check_out_date: date
    party_size_ref: Identifier
    lodging_preference_refs: tuple[Identifier, ...] = ()
    budget_constraint_ref: Identifier | None = None
    facility_constraint_refs: tuple[Identifier, ...] = ()
    activity_cluster_refs: tuple[Identifier, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def stay_dates_and_refs_are_valid(self) -> HotelSearchArguments:
        nights = (self.check_out_date - self.check_in_date).days
        if not 1 <= nights <= 4:
            raise ValueError("hotel search must cover one to four nights")
        require_unique(self.lodging_preference_refs, "lodging preference refs")
        require_unique(self.facility_constraint_refs, "hotel facility refs")
        require_unique(self.activity_cluster_refs, "hotel activity clusters")
        return self


class HotelOfferRefreshArguments(V4ContractModel):
    offer_ref: HotelOfferRef
    check_in_date: date
    check_out_date: date

    @model_validator(mode="after")
    def stay_dates_are_valid(self) -> HotelOfferRefreshArguments:
        nights = (self.check_out_date - self.check_in_date).days
        if not 1 <= nights <= 4:
            raise ValueError("hotel offer refresh must cover one to four nights")
        return self


CapabilityArguments = (
    CandidateRecallArguments
    | PlaceFactsArguments
    | OpeningHoursArguments
    | TicketAvailabilityArguments
    | WeatherForecastArguments
    | SpatialRoutesArguments
    | HotelSearchArguments
    | HotelOfferRefreshArguments
)


class PlannerCapabilityRequest(V4ContractModel):
    request_id: Identifier
    scope: PlannerScope
    capability: PlannerCapability
    purpose: Literal[
        "complete_initial_evidence",
        "resolve_validation_issue",
        "compare_route_alternatives",
        "verify_fixed_commitment",
        "refresh_stale_hotel_offer",
        "expand_candidate_gap",
    ]
    service_dates: tuple[date, ...] = ()
    based_on_issue_ids: tuple[Identifier, ...] = ()
    blocking: bool
    arguments: CapabilityArguments

    @model_validator(mode="after")
    def capability_matches_typed_arguments(self) -> PlannerCapabilityRequest:
        expected_type: dict[PlannerCapability, type[V4ContractModel]] = {
            PlannerCapability.CANDIDATE_RECALL: CandidateRecallArguments,
            PlannerCapability.PLACE_FACTS: PlaceFactsArguments,
            PlannerCapability.OPENING_HOURS: OpeningHoursArguments,
            PlannerCapability.TICKET_AVAILABILITY: TicketAvailabilityArguments,
            PlannerCapability.WEATHER_FORECAST: WeatherForecastArguments,
            PlannerCapability.SPATIAL_ROUTES: SpatialRoutesArguments,
            PlannerCapability.HOTEL_SEARCH: HotelSearchArguments,
            PlannerCapability.HOTEL_OFFER_REFRESH: HotelOfferRefreshArguments,
        }
        if not isinstance(self.arguments, expected_type[self.capability]):
            raise ValueError("Planner capability does not match its typed arguments")
        require_unique(self.service_dates, "capability service dates")
        require_unique(self.based_on_issue_ids, "capability issue IDs")
        argument_dates = getattr(self.arguments, "service_dates", None)
        if argument_dates is not None and tuple(argument_dates) != self.service_dates:
            raise ValueError("top-level and capability argument service dates must match")
        if self.purpose == "resolve_validation_issue" and not self.based_on_issue_ids:
            raise ValueError("validation issue evidence requires based_on_issue_ids")
        if self.purpose != "resolve_validation_issue" and self.based_on_issue_ids:
            raise ValueError("only issue-resolution evidence may carry based_on_issue_ids")
        return self


class SpatialCluster(V4ContractModel):
    cluster_id: Identifier
    candidate_refs: tuple[CandidateRef, ...] = Field(min_length=1)
    representative_area_refs: tuple[Identifier, ...] = ()


class RouteQueryFailure(V4ContractModel):
    """Safe recovery metadata, not evidence that endpoints are unreachable."""

    code: Identifier
    upstream_code: str | None = Field(default=None, pattern=r"^[0-9]{3,5}$")
    retryable: bool
    attempts: int = Field(ge=1, le=2, strict=True)
    query_count: int = Field(ge=1, strict=True)
    retry_after: AwareDatetime | None = None


class SpatialRouteEdge(V4ContractModel):
    route_edge_id: Identifier
    origin: SpatialRouteEndpoint
    destination: SpatialRouteEndpoint
    transport_mode: Literal["public_transit", "taxi", "walking", "driving"]
    status: Literal["available", "partial", "missing"]
    duration_minutes: int | None = Field(default=None, ge=0, strict=True)
    distance_meters: int | None = Field(default=None, ge=0, strict=True)
    transfer_count: int | None = Field(default=None, ge=0, strict=True)
    fare: CnyAmountRange | None = Field(default=None, exclude_if=lambda v: v is None)
    polyline: tuple[Gcj02Coordinates, ...] = ()
    fact_reference_ids: tuple[Identifier, ...] = ()
    missing_reason: DisplayText | None = None
    query_failure: RouteQueryFailure | None = Field(default=None, exclude_if=lambda v: v is None)

    @model_validator(mode="after")
    def status_controls_route_values(self) -> SpatialRouteEdge:
        if self.origin == self.destination:
            raise ValueError("spatial route edge endpoints must be distinct")
        require_unique(self.fact_reference_ids, "spatial route fact references")
        if self.query_failure is not None and self.status != "missing":
            raise ValueError("only missing routes may retain a query failure")
        if len(self.polyline) == 1:
            raise ValueError("route polyline must be empty or contain at least two points")
        if self.status == "available":
            if (
                self.duration_minutes is None
                or self.distance_meters is None
                or not self.fact_reference_ids
                or self.missing_reason is not None
            ):
                raise ValueError("available route requires sourced duration and distance")
        elif self.status == "partial":
            if (
                self.duration_minutes is None
                or not self.fact_reference_ids
                or self.missing_reason is None
            ):
                raise ValueError("partial route requires sourced duration and a missing reason")
        elif (
            self.duration_minutes is not None
            or self.distance_meters is not None
            or self.transfer_count is not None
            or self.fare is not None
            or self.polyline
            or self.fact_reference_ids
            or self.missing_reason is None
        ):
            raise ValueError("missing route may contain only an explicit missing reason")
        return self


class InterClusterCost(V4ContractModel):
    from_cluster_id: Identifier
    to_cluster_id: Identifier
    route_edge_ids: tuple[Identifier, ...] = Field(min_length=1)
    duration_minutes: int = Field(ge=0, strict=True)
    distance_meters: int | None = Field(default=None, ge=0, strict=True)

    @model_validator(mode="after")
    def clusters_and_edges_are_valid(self) -> InterClusterCost:
        if self.from_cluster_id == self.to_cluster_id:
            raise ValueError("inter-cluster cost requires distinct clusters")
        require_unique(self.route_edge_ids, "inter-cluster route edges")
        return self


class SpatialOutlier(V4ContractModel):
    candidate_ref: CandidateRef
    reason_summary: DisplayText
    route_edge_ids: tuple[Identifier, ...] = ()


class MissingRouteFact(V4ContractModel):
    origin: SpatialRouteEndpoint
    destination: SpatialRouteEndpoint
    transport_modes: tuple[Literal["public_transit", "taxi", "walking", "driving"], ...] = Field(
        min_length=1
    )
    reason_summary: DisplayText


class SpatialObservation(V4ContractModel):
    observation_id: Identifier
    scope: PlannerScope
    clusters: tuple[SpatialCluster, ...]
    route_edges: tuple[SpatialRouteEdge, ...] = ()
    inter_cluster_costs: tuple[InterClusterCost, ...] = ()
    outliers: tuple[SpatialOutlier, ...] = ()
    missing_route_facts: tuple[MissingRouteFact, ...] = ()
    source_reference_ids: tuple[Identifier, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def cluster_and_route_references_are_consistent(self) -> SpatialObservation:
        cluster_ids = [item.cluster_id for item in self.clusters]
        require_unique(cluster_ids, "spatial cluster IDs")
        route_ids = [item.route_edge_id for item in self.route_edges]
        require_unique(route_ids, "spatial route edge IDs")
        require_unique(self.source_reference_ids, "spatial observation sources")
        candidate_keys = [
            candidate_ref_key(reference)
            for cluster in self.clusters
            for reference in cluster.candidate_refs
        ]
        require_unique(candidate_keys, "spatially clustered candidates")
        route_id_set = set(route_ids)
        cluster_id_set = set(cluster_ids)
        for cost in self.inter_cluster_costs:
            if {cost.from_cluster_id, cost.to_cluster_id} - cluster_id_set:
                raise ValueError("inter-cluster cost references an unknown cluster")
            if set(cost.route_edge_ids) - route_id_set:
                raise ValueError("inter-cluster cost references an unknown route edge")
        for outlier in self.outliers:
            if set(outlier.route_edge_ids) - route_id_set:
                raise ValueError("spatial outlier references an unknown route edge")
        return self


class DayGroupingCrossClusterSegment(V4ContractModel):
    from_cluster_id: Identifier
    to_cluster_id: Identifier
    candidate_refs: tuple[CandidateRef, ...] = Field(min_length=1)
    reason_code: CrossClusterReasonCode
    supporting_intent_or_fact_refs: tuple[Identifier, ...] = Field(min_length=1)
    route_edge_ids: tuple[Identifier, ...] = Field(min_length=1)
    comparison_observation_ref: Identifier | None = None

    @model_validator(mode="after")
    def evidence_is_sufficient(self) -> DayGroupingCrossClusterSegment:
        if self.from_cluster_id == self.to_cluster_id:
            raise ValueError("day grouping cross-cluster segment requires distinct clusters")
        require_unique(
            [candidate_ref_key(item) for item in self.candidate_refs],
            "day grouping cross-cluster candidates",
        )
        require_unique(self.route_edge_ids, "day grouping route edge IDs")
        if (
            self.reason_code is CrossClusterReasonCode.VERIFIED_GLOBAL_ROUTE_IMPROVEMENT
            and self.comparison_observation_ref is None
        ):
            raise ValueError("verified global route improvement requires comparison evidence")
        return self


class DayGrouping(V4ContractModel):
    service_date: date
    primary_cluster_id: Identifier
    ordered_candidate_refs: tuple[CandidateRef, ...]
    cross_cluster_segments: tuple[DayGroupingCrossClusterSegment, ...] = ()

    @model_validator(mode="after")
    def candidate_order_and_cross_segments_are_unique(self) -> DayGrouping:
        ordered_keys = [candidate_ref_key(item) for item in self.ordered_candidate_refs]
        require_unique(ordered_keys, "day grouping candidates")
        covered = [
            candidate_ref_key(reference)
            for segment in self.cross_cluster_segments
            for reference in segment.candidate_refs
        ]
        require_unique(covered, "day grouping cross-cluster coverage")
        if set(covered) - set(ordered_keys):
            raise ValueError("cross-cluster segment contains an ungrouped candidate")
        return self


class DayGroupingDecision(V4ContractModel):
    scope: PlannerScope
    candidate_pool_revision: int = Field(ge=1, strict=True)
    spatial_observation_id: Identifier
    days: tuple[DayGrouping, ...] = Field(min_length=1, max_length=5)

    @model_validator(mode="after")
    def days_are_unique_and_ordered(self) -> DayGroupingDecision:
        dates = [item.service_date for item in self.days]
        require_unique(dates, "day grouping dates")
        if dates != sorted(dates):
            raise ValueError("day grouping dates must be ordered")
        for day in self.days:
            for reference in day.ordered_candidate_refs:
                if reference.candidate_pool_revision != self.candidate_pool_revision:
                    raise ValueError("day grouping CandidateRef uses a stale pool revision")
        return self


class HotelStaySegment(V4ContractModel):
    check_in_date: date
    check_out_date: date
    nights: int = Field(ge=1, le=4, strict=True)
    activity_cluster_ids: tuple[Identifier, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def nights_match_dates(self) -> HotelStaySegment:
        if (self.check_out_date - self.check_in_date).days != self.nights:
            raise ValueError("hotel stay nights must match check-in and check-out dates")
        require_unique(self.activity_cluster_ids, "hotel stay activity clusters")
        return self


class AppliedHotelConstraints(V4ContractModel):
    quality_tiers: tuple[Literal["economy", "comfort", "upscale", "luxury"], ...] = ()
    area_refs: tuple[Identifier, ...] = ()
    quality_tier: Identifier | None = None
    nightly_budget_ref: Identifier | None = None
    property_types: tuple[Identifier, ...] = ()
    facility_requirement_refs: tuple[Identifier, ...] = ()
    source_task_book_refs: tuple[Identifier, ...] = ()


class FixedBookingObservation(V4ContractModel):
    commitment_ref: FixedCommitmentRef
    property_id: Identifier | None = None
    verification_status: Literal["verified", "ambiguous", "unavailable"]
    conflict_issue_refs: tuple[Identifier, ...] = ()


class MoneyAmount(V4ContractModel):
    currency: Identifier
    amount_minor: int = Field(ge=1, strict=True)


class MoneyAmountRange(V4ContractModel):
    """A bounded Provider reference range, not a bookable stay total."""

    currency: Identifier
    minimum_minor: int = Field(ge=1, strict=True)
    maximum_minor: int = Field(ge=1, strict=True)

    @model_validator(mode="after")
    def bounds_are_ordered(self) -> MoneyAmountRange:
        if self.maximum_minor < self.minimum_minor:
            raise ValueError("money range maximum cannot be below minimum")
        return self


class HotelClusterCommute(V4ContractModel):
    cluster_id: Identifier
    route_observation_ref: Identifier
    duration_minutes: int | None = Field(default=None, ge=0, strict=True)


class HotelOfferObservation(V4ContractModel):
    offer_ref: HotelOfferRef
    property_name: DisplayText
    area_ref: Identifier
    availability_status: Literal["available", "limited", "unknown", "unavailable"]
    room_and_price_fact_refs: tuple[Identifier, ...] = ()
    reference_price: MoneyAmountRange | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="Provider list reference price per room-night; not a bookable total.",
    )
    reference_price_fact_refs: tuple[Identifier, ...] = Field(
        default=(),
        exclude_if=lambda value: not value,
    )
    total_price: MoneyAmount | None = None
    price_missing: bool
    quality_feature_refs: tuple[Identifier, ...] = ()
    facility_feature_refs: tuple[Identifier, ...] = ()
    commute_to_clusters: tuple[HotelClusterCommute, ...] = ()
    source_reference_ids: tuple[Identifier, ...] = Field(min_length=1)
    observed_at: AwareDatetime
    expires_at: AwareDatetime | None = None

    @model_validator(mode="after")
    def price_sources_and_expiry_are_consistent(self) -> HotelOfferObservation:
        if self.price_missing != (self.total_price is None):
            raise ValueError("unknown hotel price must be null and never encoded as zero")
        if self.total_price is not None and not self.room_and_price_fact_refs:
            raise ValueError("hotel price requires source fact references")
        if (self.reference_price is None) != (not self.reference_price_fact_refs):
            raise ValueError("hotel reference price and its fact references must appear together")
        require_unique(self.room_and_price_fact_refs, "hotel price fact references")
        require_unique(self.reference_price_fact_refs, "hotel reference price fact references")
        require_unique(self.quality_feature_refs, "hotel quality feature references")
        require_unique(self.facility_feature_refs, "hotel facility references")
        require_unique(self.source_reference_ids, "hotel offer source references")
        require_unique(
            [item.cluster_id for item in self.commute_to_clusters],
            "hotel commute cluster IDs",
        )
        if self.expires_at is not None and self.expires_at <= self.observed_at:
            raise ValueError("hotel offer expires_at must be after observed_at")
        return self


class HotelQueryAttempt(V4ContractModel):
    """Safe diagnostics only; never vendor payloads, URLs or credentials."""

    stage: Literal["search", "identity"] = "search"
    anchor_name: str | None = None
    search_keyword: str | None = None
    outcome: Literal["results", "empty", "failed"]
    result_count: int = Field(default=0, ge=0)
    error_code: Identifier | None = None
    retryable: bool = False
    attempts: int = Field(default=1, ge=1)
    observed_at: AwareDatetime


class HotelObservation(V4ContractModel):
    query_status: Literal["available", "empty", "failed", "unverified"] | None = None
    query_origin: Literal["prepare_handoff", "planner_query"] | None = None
    query_attempts: tuple[HotelQueryAttempt, ...] = ()
    search_keyword: str | None = Field(
        default=None, min_length=1, max_length=100, exclude_if=lambda value: value is None
    )
    hotel_observation_id: Identifier
    scope: PlannerScope
    request_id: Identifier
    mode: Literal["not_applicable", "fixed_booking_verification", "search"]
    status: Literal["complete", "partial", "unavailable"]
    observed_at: AwareDatetime
    expires_at: AwareDatetime | None = None
    stay_segments: tuple[HotelStaySegment, ...] = ()
    applied_constraints: AppliedHotelConstraints
    fixed_booking: FixedBookingObservation | None = None
    offers: tuple[HotelOfferObservation, ...] = ()
    missing_fact_kinds: tuple[Identifier, ...] = ()
    source_reference_ids: tuple[Identifier, ...] = ()

    @model_validator(mode="after")
    def mode_status_and_snapshot_references_are_consistent(self) -> HotelObservation:
        require_unique(self.missing_fact_kinds, "hotel missing fact kinds")
        require_unique(self.source_reference_ids, "hotel observation sources")
        if self.expires_at is not None and self.expires_at <= self.observed_at:
            raise ValueError("HotelObservation expires_at must be after observed_at")
        dates = [(item.check_in_date, item.check_out_date) for item in self.stay_segments]
        require_unique(dates, "hotel stay segments")
        ordered = sorted(self.stay_segments, key=lambda item: item.check_in_date)
        if any(
            current.check_out_date > following.check_in_date
            for current, following in zip(ordered, ordered[1:], strict=False)
        ):
            raise ValueError("hotel stay segments cannot overlap")

        if self.mode == "not_applicable":
            if self.stay_segments or self.fixed_booking is not None or self.offers:
                raise ValueError("day-trip HotelObservation cannot contain stays or hotels")
        elif self.mode == "fixed_booking_verification":
            if not self.stay_segments or self.fixed_booking is None or self.offers:
                raise ValueError("fixed booking mode requires only a fixed booking observation")
            fixed = self.fixed_booking.commitment_ref
            if (
                fixed.task_book_id != self.scope.task_book_id
                or fixed.task_book_version != self.scope.task_book_version
            ):
                raise ValueError("fixed hotel booking must belong to the confirmed task book")
        elif not self.stay_segments or self.fixed_booking is not None:
            raise ValueError("hotel search mode requires stays and forbids fixed_booking")

        offer_keys = [item.offer_ref.offer_id for item in self.offers]
        require_unique(offer_keys, "hotel offer IDs")
        require_unique(
            [item.offer_ref.property_id for item in self.offers],
            "hotel offer properties",
        )
        for offer in self.offers:
            if offer.offer_ref.hotel_observation_id != self.hotel_observation_id:
                raise ValueError("hotel offer must bind to this HotelObservation")
        if self.status == "complete" and self.missing_fact_kinds:
            raise ValueError("complete HotelObservation cannot declare missing facts")
        if self.status == "unavailable" and self.offers:
            raise ValueError("unavailable HotelObservation cannot expose usable offers")
        if self.mode == "search" and self.status == "complete" and not self.offers:
            raise ValueError("complete hotel search requires at least one offer")
        return self


class PlannerValidationIssue(V4ContractModel):
    issue_id: Identifier
    code: Literal[
        "candidate_not_in_pool",
        "required_missing",
        "forbidden_included",
        "opening_conflict",
        "reservation_conflict",
        "time_overlap",
        "route_unavailable",
        "route_cost_exceeded",
        "unjustified_cross_cluster",
        "walking_limit_exceeded",
        "pace_limit_exceeded",
        "half_day_without_visit",
        "unfilled_sightseeing_gap",
        "meal_constraint_violation",
        "meal_gap",
        "hotel_night_gap",
        "hotel_unavailable",
        "hotel_constraint_conflict",
        "missing_price",
        "budget_exceeded",
        "unscheduled_strong_intent",
        "stale_evidence",
        "stale_revision",
    ]
    severity: Literal["warning", "error", "blocking"]
    scope_kind: Literal["day", "item", "route_edge", "hotel", "cost", "global"]
    affected_dates: tuple[date, ...] = ()
    draft_item_ids: tuple[Identifier, ...] = ()
    candidate_refs: tuple[CandidateRef, ...] = ()
    route_edge_ids: tuple[Identifier, ...] = ()
    hotel_offer_refs: tuple[HotelOfferRef, ...] = ()
    fact_reference_ids: tuple[Identifier, ...] = ()
    violated_constraint_refs: tuple[Identifier, ...] = ()
    message_summary: DisplayText
    allowed_actions: tuple[
        Literal[
            "request_evidence",
            "move_item",
            "reorder_item",
            "replace_item",
            "remove_item",
            "change_window",
            "change_transport",
            "change_hotel",
            "ask_user",
        ],
        ...,
    ] = Field(min_length=1)
    user_authority_required: bool

    @model_validator(mode="after")
    def target_and_allowed_actions_are_concrete(self) -> PlannerValidationIssue:
        for name, values in (
            ("affected_dates", self.affected_dates),
            ("draft_item_ids", self.draft_item_ids),
            ("route_edge_ids", self.route_edge_ids),
            ("fact_reference_ids", self.fact_reference_ids),
            ("violated_constraint_refs", self.violated_constraint_refs),
            ("allowed_actions", self.allowed_actions),
        ):
            require_unique(values, name)
        require_unique(
            [candidate_ref_key(item) for item in self.candidate_refs],
            "validation issue candidate references",
        )
        require_unique(
            [(item.hotel_observation_id, item.offer_id) for item in self.hotel_offer_refs],
            "validation issue hotel offers",
        )
        has_target = {
            "day": bool(self.affected_dates),
            "item": bool(self.draft_item_ids or self.candidate_refs),
            "route_edge": bool(self.route_edge_ids),
            "hotel": bool(self.hotel_offer_refs or self.violated_constraint_refs),
            "cost": bool(self.violated_constraint_refs or self.fact_reference_ids),
            "global": bool(
                self.affected_dates
                or self.draft_item_ids
                or self.candidate_refs
                or self.route_edge_ids
                or self.hotel_offer_refs
                or self.violated_constraint_refs
            ),
        }[self.scope_kind]
        if not has_target:
            raise ValueError("Planner validation issue must identify a concrete target")
        if self.user_authority_required and "ask_user" not in self.allowed_actions:
            raise ValueError("user-authority issue must allow ask_user")
        return self


class PlannerValidationObservation(V4ContractModel):
    observation_id: Identifier
    scope: PlannerScope
    draft_id: Identifier
    draft_revision: int = Field(ge=1, strict=True)
    materialized_schedule_id: Identifier
    materialized_schedule_revision: int = Field(ge=1, strict=True)
    cost_draft_id: Identifier
    cost_draft_revision: int = Field(ge=1, strict=True)
    validator_version: Identifier
    validation_fingerprint: Digest
    result: Literal["passed", "repairable", "requires_user", "insufficient_evidence", "fatal"]
    issues: tuple[PlannerValidationIssue, ...] = ()
    globally_affected_dates: tuple[date, ...] = ()
    checked_at: AwareDatetime

    @model_validator(mode="after")
    def result_matches_issue_permissions(self) -> PlannerValidationObservation:
        require_unique([item.issue_id for item in self.issues], "Planner validation issue IDs")
        require_unique(self.globally_affected_dates, "globally affected dates")
        blocking = [item for item in self.issues if item.severity in {"error", "blocking"}]
        if self.result == "passed" and blocking:
            raise ValueError("passed validation cannot contain error or blocking issues")
        if self.result != "passed" and not self.issues:
            raise ValueError("non-passed validation requires at least one concrete issue")
        if self.result == "repairable" and not any(
            set(item.allowed_actions)
            & {
                "move_item",
                "reorder_item",
                "replace_item",
                "remove_item",
                "change_transport",
                "change_hotel",
            }
            and not item.user_authority_required
            for item in self.issues
        ):
            raise ValueError("repairable validation requires an authorized repair action")
        if self.result == "requires_user" and not any(
            item.user_authority_required and "ask_user" in item.allowed_actions
            for item in self.issues
        ):
            raise ValueError("requires_user needs a user-authority issue")
        if self.result == "insufficient_evidence" and not any(
            "request_evidence" in item.allowed_actions for item in self.issues
        ):
            raise ValueError("insufficient evidence must allow request_evidence")
        return self


class PlannerInteractionOption(V4ContractModel):
    option_id: Identifier
    semantic_action: Identifier
    affected_refs: tuple[Identifier, ...] = Field(min_length=1)
    verified_impact_summary: DisplayText


class PlannerInteraction(V4ContractModel):
    interaction_id: Identifier
    scope: PlannerScope
    reason_code: AskUserReasonCode
    issue_ids: tuple[Identifier, ...] = Field(min_length=1)
    question: str | None = Field(default=None, max_length=240)
    decision_scope: Literal["global", "date", "item", "hotel"]
    affected_dates: tuple[date, ...] = ()
    option_contracts: tuple[PlannerInteractionOption, ...] = Field(min_length=1)
    allow_free_text: bool
    based_on_workspace_revision: int = Field(ge=0, strict=True)
    resume_token: Identifier
    status: InteractionStatus

    @model_validator(mode="after")
    def interaction_is_actionable_and_current(self) -> PlannerInteraction:
        require_unique(self.issue_ids, "Planner interaction issue IDs")
        require_unique(self.affected_dates, "Planner interaction affected dates")
        require_unique(
            [item.option_id for item in self.option_contracts],
            "Planner interaction option IDs",
        )
        if self.decision_scope == "date" and not self.affected_dates:
            raise ValueError("date-scoped Planner interaction requires affected_dates")
        if self.based_on_workspace_revision != self.scope.workspace_revision:
            raise ValueError("Planner interaction must bind to its scope workspace revision")
        return self


class PlannerReadinessIssue(V4ContractModel):
    """A sourced, pre-materialization blocker; never a final validation result."""

    issue_id: Identifier
    code: Literal[
        "required_entity_missing",
        "strong_opening_conflict",
        "fixed_booking_conflict",
        "missing_booking_detail",
        "missing_evidence",
    ]
    candidate_refs: tuple[CandidateRef, ...] = ()
    fixed_commitment_refs: tuple[FixedCommitmentRef, ...] = ()
    affected_dates: tuple[date, ...] = ()
    fact_reference_ids: tuple[Identifier, ...] = Field(min_length=1)
    reason_summary: DisplayText
    user_authority_required: bool
    ask_user_reason: AskUserReasonCode | None = None
    option_contracts: tuple[PlannerInteractionOption, ...] = ()

    @model_validator(mode="after")
    def permissions_require_actionable_evidence(self) -> PlannerReadinessIssue:
        require_unique(self.fact_reference_ids, "readiness issue evidence")
        require_unique(self.affected_dates, "readiness issue dates")
        require_unique([item.option_id for item in self.option_contracts], "readiness options")
        if self.user_authority_required:
            if (
                self.code in {"missing_evidence", "required_entity_missing"}
                or self.ask_user_reason is None
                or not self.option_contracts
                or not (self.candidate_refs or self.fixed_commitment_refs)
            ):
                raise ValueError("user authority requires a verified target and actionable options")
        elif self.ask_user_reason is not None or self.option_contracts:
            raise ValueError("internal evidence gaps cannot ask the user to solve them")
        return self


class PlannerReadinessObservation(V4ContractModel):
    observation_id: Identifier
    scope: PlannerScope
    candidate_pool_revision: int = Field(ge=1, strict=True)
    checked_at: AwareDatetime
    issues: tuple[PlannerReadinessIssue, ...] = ()

    @model_validator(mode="after")
    def issues_bind_to_current_inputs(self) -> PlannerReadinessObservation:
        require_unique([item.issue_id for item in self.issues], "readiness issue IDs")
        for issue in self.issues:
            for reference in issue.candidate_refs:
                if reference.candidate_pool_revision != self.candidate_pool_revision:
                    raise ValueError("readiness candidate uses a stale pool revision")
            for fixed_reference in issue.fixed_commitment_refs:
                if (
                    fixed_reference.task_book_id != self.scope.task_book_id
                    or fixed_reference.task_book_version != self.scope.task_book_version
                ):
                    raise ValueError("readiness fixed commitment belongs to another task book")
        return self


V4_PLANNER_OBSERVATION_CONTRACTS: tuple[type[V4ContractModel], ...] = (
    PlannerCapabilityRequest,
    SpatialObservation,
    DayGroupingDecision,
    HotelObservation,
    PlannerValidationObservation,
    PlannerInteraction,
    PlannerReadinessObservation,
)
