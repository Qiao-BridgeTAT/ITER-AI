"""Checkpointable, normalized Planner evidence without raw Provider payloads."""

from __future__ import annotations

from datetime import date
from typing import Literal
from uuid import NAMESPACE_URL, uuid5

from pydantic import AwareDatetime, Field, JsonValue, model_validator

from backend.contracts.common import CnyAmountRange
from backend.contracts.places import Gcj02Coordinates
from backend.contracts.v4.base import (
    Digest,
    DisplayText,
    Identifier,
    V4ContractModel,
    require_unique,
)
from backend.contracts.v4.enums import CandidateEntityKind, PlannerCapability
from backend.contracts.v4.planner_observations import RouteComparisonInput, SpatialRouteEdge
from backend.contracts.v4.planner_refs import CandidateRef, PlannerScope
from backend.providers.contracts import ProviderDateHours


class PlannerCandidateOrigin(V4ContractModel):
    """Identity lookup only; committed cards never substitute for fresh place facts."""

    canonical_entity_id: Identifier
    provider: Literal["amap"] = "amap"
    provider_entity_id: Identifier
    source_message_id: Identifier
    source_option_id: Identifier

    @model_validator(mode="after")
    def source_identity_matches_canonical_id(self) -> PlannerCandidateOrigin:
        if self.canonical_entity_id != str(uuid5(NAMESPACE_URL, f"amap:{self.provider_entity_id}")):
            raise ValueError(
                "candidate origin cannot transfer a Provider identity to another entity"
            )
        return self


class PlannerPlaceEvidence(V4ContractModel):
    canonical_entity_id: Identifier
    entity_kind: CandidateEntityKind
    display_name: DisplayText
    city_id: Identifier
    coordinates: Gcj02Coordinates
    provider: Literal["amap"]
    provider_entity_id: Identifier
    provider_parent_place_id: Identifier | None = Field(
        default=None, exclude_if=lambda v: v is None
    )
    provider_typecode: Identifier
    address: DisplayText | None = None
    rating: float | None = Field(default=None, ge=0, le=5, exclude_if=lambda v: v is None)
    average_cost: CnyAmountRange | None = Field(default=None, exclude_if=lambda v: v is None)
    business_fact_reference_id: Identifier | None = Field(
        default=None, exclude_if=lambda v: v is None
    )
    business_observed_at: AwareDatetime | None = Field(default=None, exclude_if=lambda v: v is None)
    fact_reference_id: Identifier
    observed_at: AwareDatetime

    @model_validator(mode="after")
    def identity_is_bound_to_provider_entity(self) -> PlannerPlaceEvidence:
        if self.canonical_entity_id != str(uuid5(NAMESPACE_URL, f"amap:{self.provider_entity_id}")):
            raise ValueError("Planner place identity must match its exact Provider entity")
        return self


class PlannerHoursEvidence(V4ContractModel):
    canonical_entity_id: Identifier
    provider_entity_id: Identifier
    fact_reference_id: Identifier
    observed_at: AwareDatetime
    expires_at: AwareDatetime
    days: tuple[ProviderDateHours, ...] = Field(min_length=1, max_length=5)

    @model_validator(mode="after")
    def dates_and_expiry_are_valid(self) -> PlannerHoursEvidence:
        require_unique([day.service_date for day in self.days], "Planner hours dates")
        if self.expires_at <= self.observed_at:
            raise ValueError("Planner hours must have a positive evidence lifetime")
        return self


class PlannerVisitDurationEstimate(V4ContractModel):
    """Advisory duration, deliberately not a verified opening-hours fact."""

    canonical_entity_id: Identifier
    minimum_minutes: int = Field(ge=15, le=600, strict=True)
    maximum_minutes: int = Field(ge=15, le=600, strict=True)
    source: Literal["llm_estimate", "category_estimate"]
    context_fingerprint: Digest

    @model_validator(mode="after")
    def range_is_ordered(self) -> PlannerVisitDurationEstimate:
        if self.maximum_minutes < self.minimum_minutes:
            raise ValueError("visit duration maximum must not precede minimum")
        return self


class PlannerWeatherEvidence(V4ContractModel):
    service_date: date
    condition_day: DisplayText | None = None
    condition_night: DisplayText | None = None
    high_celsius: int | None = None
    low_celsius: int | None = None
    source_name: DisplayText | None = Field(default=None, exclude_if=lambda value: value is None)
    forecast_kind: Literal["forecast", "outlook"] = Field(
        default="forecast", exclude_if=lambda value: value == "forecast"
    )
    observed_at: AwareDatetime
    fact_reference_id: Identifier


class PlannerTicketEvidence(V4ContractModel):
    canonical_entity_id: Identifier
    service_date: date
    product_count: int = Field(ge=0, strict=True)
    reference_price: CnyAmountRange | None = Field(default=None, exclude_if=lambda v: v is None)
    source_offer_ids: tuple[Identifier, ...] = Field(default=(), exclude_if=lambda v: not v)
    admission_status: Literal["free", "paid", "unknown"] = Field(
        default="unknown", exclude_if=lambda v: v == "unknown"
    )
    admission_source_ids: tuple[Identifier, ...] = Field(default=(), exclude_if=lambda v: not v)
    availability: Literal["unknown"] = "unknown"
    reason_summary: DisplayText
    observed_at: AwareDatetime
    fact_reference_id: Identifier

    @model_validator(mode="after")
    def reference_price_has_bound_offer_sources(self) -> PlannerTicketEvidence:
        if (self.reference_price is None) != (not self.source_offer_ids):
            raise ValueError("ticket reference price requires matching source offer identities")
        require_unique(self.source_offer_ids, "ticket source offers")
        require_unique(self.admission_source_ids, "ticket admission sources")
        if self.admission_status != "unknown" and not self.admission_source_ids:
            raise ValueError("known admission status requires an exact-entity source")
        if self.admission_status == "free" and (
            self.reference_price is None or self.reference_price.maximum_fen != 0
        ):
            raise ValueError("free basic admission must have a sourced zero reference price")
        return self


class PlannerHotelLocationEvidence(V4ContractModel):
    property_id: Identifier
    provider_entity_id: Identifier
    coordinates: Gcj02Coordinates
    display_name: DisplayText | None = Field(default=None, exclude_if=lambda v: v is None)
    address: DisplayText | None = Field(default=None, exclude_if=lambda v: v is None)
    rating: float | None = Field(default=None, ge=0, le=5, exclude_if=lambda v: v is None)
    fact_reference_id: Identifier
    observed_at: AwareDatetime


class PlannerCapabilityObservation(V4ContractModel):
    """Receipt binds a typed request to normalized facts held in the workspace."""

    observation_id: Identifier
    scope: PlannerScope
    request_id: Identifier
    capability: PlannerCapability
    status: Literal["complete", "partial", "unavailable", "invalid_request"]
    candidate_refs: tuple[CandidateRef, ...] = ()
    service_dates: tuple[date, ...] = ()
    fact_reference_ids: tuple[Identifier, ...] = ()
    artifact_reference_ids: tuple[Identifier, ...] = ()
    reason_summary: DisplayText
    observed_at: AwareDatetime

    @model_validator(mode="after")
    def successful_observation_has_evidence(self) -> PlannerCapabilityObservation:
        require_unique(self.fact_reference_ids, "capability fact references")
        require_unique(self.artifact_reference_ids, "capability artifact references")
        require_unique(self.service_dates, "capability observation dates")
        if self.status == "complete" and not (
            self.fact_reference_ids or self.artifact_reference_ids
        ):
            raise ValueError("complete capability observation must point to real evidence")
        return self


class PlannerGuardViolation(V4ContractModel):
    """One actionable, privacy-safe reason a Planner proposal was rejected."""

    violation_id: Identifier
    attempted_action: Identifier
    stage: Literal[
        "schema",
        "reference",
        "business",
        "materialize",
        "validate",
        "final",
    ]
    code: Identifier
    field_path: Identifier | None = None
    object_index: int | None = Field(default=None, ge=0, strict=True)
    expected_rule: DisplayText
    legal_option_keys: tuple[Identifier, ...] = ()
    related_issue_ids: tuple[Identifier, ...] = ()
    preserve_field_paths: tuple[Identifier, ...] = ()
    allowed_next_actions: tuple[Identifier, ...] = Field(min_length=1)
    failure_fingerprint: Digest
    minimal_valid_fragment: dict[str, JsonValue] | None = None

    @model_validator(mode="after")
    def lists_are_unique(self) -> PlannerGuardViolation:
        for label, values in (
            ("guard legal options", self.legal_option_keys),
            ("guard related issues", self.related_issue_ids),
            ("guard preserve paths", self.preserve_field_paths),
            ("guard allowed next actions", self.allowed_next_actions),
        ):
            require_unique(values, label)
        return self


class PlannerGuardObservation(V4ContractModel):
    observation_id: Identifier
    attempted_action: Identifier
    code: Identifier
    message: DisplayText
    based_on_workspace_revision: int = Field(ge=0, strict=True)
    violations: tuple[PlannerGuardViolation, ...] = Field(default=(), max_length=4)
    failure_fingerprint: Identifier | None = None

    @model_validator(mode="after")
    def structured_feedback_is_consistent(self) -> PlannerGuardObservation:
        if not self.violations:
            if self.failure_fingerprint is not None:
                raise ValueError("guard fingerprint requires structured violations")
            return self
        if any(item.attempted_action != self.attempted_action for item in self.violations):
            raise ValueError("guard violations must match the attempted action")
        if self.failure_fingerprint != self.violations[0].failure_fingerprint:
            raise ValueError("guard observation fingerprint must match its root violation")
        return self


class PlannerInteractionAnswer(V4ContractModel):
    answer_id: Identifier
    interaction_id: Identifier
    option_id: Identifier
    semantic_action: Literal["keep_task_book", "revise_task_book", "supply_booking_detail"]
    user_text: DisplayText | None = None
    source_turn_id: Identifier


class PlannerRouteComparisonObservation(V4ContractModel):
    observation_id: Identifier
    scope: PlannerScope
    candidate_pool_revision: int = Field(ge=1, strict=True)
    comparison: RouteComparisonInput
    route_edges: tuple[SpatialRouteEdge, ...] = Field(min_length=1)
    baseline_duration_minutes: int = Field(ge=0, strict=True)
    proposed_duration_minutes: int = Field(ge=0, strict=True)
    observed_at: AwareDatetime
    expires_at: AwareDatetime

    @model_validator(mode="after")
    def totals_are_computed_from_exact_verified_legs(self) -> PlannerRouteComparisonObservation:
        require_unique([edge.route_edge_id for edge in self.route_edges], "comparison route edges")
        if self.expires_at <= self.observed_at:
            raise ValueError("comparison evidence must have a positive lifetime")
        edges = {(edge.origin, edge.destination): edge for edge in self.route_edges}
        if len(edges) != len(self.route_edges) or any(
            edge.status != "available"
            or edge.duration_minutes is None
            or not edge.fact_reference_ids
            or edge.transport_mode != self.comparison.transport_mode
            for edge in self.route_edges
        ):
            raise ValueError("comparison requires complete exact legs in one transport mode")
        totals = []
        for days in (self.comparison.baseline_days, self.comparison.proposed_days):
            total = 0
            for day in days:
                for pair in zip(day.ordered_endpoints, day.ordered_endpoints[1:], strict=False):
                    edge = edges.get(pair)
                    if edge is None or edge.duration_minutes is None:
                        raise ValueError("comparison route leg is missing")
                    total += edge.duration_minutes
            totals.append(total)
        if totals != [self.baseline_duration_minutes, self.proposed_duration_minutes]:
            raise ValueError("comparison totals must exactly equal observed route costs")
        return self


V4_PLANNER_EVIDENCE_CONTRACTS = (
    PlannerCandidateOrigin,
    PlannerCapabilityObservation,
    PlannerPlaceEvidence,
    PlannerHoursEvidence,
    PlannerWeatherEvidence,
    PlannerTicketEvidence,
    PlannerHotelLocationEvidence,
    PlannerGuardViolation,
    PlannerGuardObservation,
    PlannerInteractionAnswer,
    PlannerRouteComparisonObservation,
)
