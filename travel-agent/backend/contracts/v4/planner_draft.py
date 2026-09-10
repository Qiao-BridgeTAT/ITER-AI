"""Complete, ordered working-itinerary contracts for the V4 Planner."""

from __future__ import annotations

import hashlib
import json
from datetime import date, time
from typing import Any, Literal

from pydantic import Field, model_validator

from backend.contracts.v4.base import (
    Digest,
    DisplayText,
    Identifier,
    V4ContractModel,
    require_unique,
)
from backend.contracts.v4.enums import CommitmentLevel, CrossClusterReasonCode
from backend.contracts.v4.planner_observations import SpatialRouteEndpoint
from backend.contracts.v4.planner_refs import (
    CandidateRef,
    FixedCommitmentRef,
    HotelOfferRef,
    PlannerObjectRef,
    PlannerScope,
    candidate_ref_key,
    planner_object_ref_key,
    require_same_scope_ownership,
)
from backend.contracts.v4.planner_strategy import CandidatePoolSummary


class ExpectedWindow(V4ContractModel):
    part_of_day: Literal["morning", "midday", "afternoon", "evening", "anytime"]
    earliest: time | None = None
    latest: time | None = None

    @model_validator(mode="after")
    def bounds_are_ordered(self) -> ExpectedWindow:
        if self.earliest is not None and self.latest is not None and self.latest < self.earliest:
            raise ValueError("expected window cannot run backwards")
        return self


class DraftItem(V4ContractModel):
    draft_item_id: Identifier
    position: int = Field(ge=0, strict=True)
    item_kind: Literal["visit", "dining", "fixed_event", "arrival", "departure"]
    object_ref: PlannerObjectRef
    cluster_id: Identifier | None = None
    expected_window: ExpectedWindow
    meal_slot: Literal["breakfast", "lunch", "dinner", "snack"] | None = None
    duration_preference: Literal["short", "normal", "extended"] | None = None
    onsite_lunch: bool = Field(default=False, exclude_if=lambda value: not value)
    commitment_level: Literal["immutable", "strong", "soft", "filler", "neutral"]

    @model_validator(mode="after")
    def kind_and_commitment_match_reference(self) -> DraftItem:
        if self.onsite_lunch and self.item_kind != "visit":
            raise ValueError("onsite lunch is only supported inside a visit")
        if self.item_kind == "dining" and self.meal_slot is None:
            raise ValueError("a dining draft item requires a meal_slot")
        if self.item_kind != "dining" and self.meal_slot is not None:
            raise ValueError("only a dining draft item may contain a meal_slot")
        if isinstance(self.object_ref, CandidateRef):
            if self.commitment_level == "immutable":
                raise ValueError("ordinary candidate references cannot claim immutable commitment")
            if self.item_kind in {"fixed_event", "arrival", "departure"}:
                raise ValueError("candidate references may only produce visit or dining items")
            if self.object_ref.entity_kind.value == "restaurant" and self.item_kind != "dining":
                raise ValueError("restaurant candidates must be dining items")
            if self.object_ref.entity_kind.value != "restaurant" and self.item_kind == "dining":
                raise ValueError("only restaurant candidates may be dining items")
            return self

        expected_kind = {
            "arrival": "arrival",
            "departure": "departure",
            "reservation": "fixed_event",
            "existing_booking": "fixed_event",
        }.get(self.object_ref.commitment_kind)
        if expected_kind is None or self.item_kind != expected_kind:
            raise ValueError("fixed commitment kind does not match the draft item kind")
        if self.commitment_level != "immutable":
            raise ValueError("fixed commitments must remain immutable in the draft")
        return self


class ExpectedRouteCost(V4ContractModel):
    duration_minutes: int = Field(ge=0, le=1_440, strict=True)
    distance_meters: int | None = Field(default=None, ge=0, strict=True)
    transfer_count: int | None = Field(default=None, ge=0, le=20, strict=True)


class CrossClusterSegment(V4ContractModel):
    from_cluster_id: Identifier
    to_cluster_id: Identifier
    covered_item_ids: tuple[Identifier, ...] = Field(min_length=1)
    reason_code: CrossClusterReasonCode
    supporting_intent_or_fact_refs: tuple[Identifier, ...] = Field(min_length=1)
    route_edge_ids: tuple[Identifier, ...] = Field(min_length=1)
    expected_route_cost: ExpectedRouteCost
    comparison_observation_ref: Identifier | None = None

    @model_validator(mode="after")
    def segment_has_evidence_and_distinct_clusters(self) -> CrossClusterSegment:
        if self.from_cluster_id == self.to_cluster_id:
            raise ValueError("a cross-cluster segment requires distinct clusters")
        require_unique(self.covered_item_ids, "cross-cluster covered item IDs")
        require_unique(
            self.supporting_intent_or_fact_refs,
            "cross-cluster supporting references",
        )
        require_unique(self.route_edge_ids, "cross-cluster route edge IDs")
        if (
            self.reason_code is CrossClusterReasonCode.VERIFIED_GLOBAL_ROUTE_IMPROVEMENT
            and self.comparison_observation_ref is None
        ):
            raise ValueError("verified_global_route_improvement requires a comparison observation")
        return self


class DraftRouteModeSelection(V4ContractModel):
    origin: SpatialRouteEndpoint
    destination: SpatialRouteEndpoint
    transport_mode: Literal["taxi", "public_transit", "walking"]


class WorkingItineraryDay(V4ContractModel):
    service_date: date
    day_kind: Literal["active", "arrival_departure", "rest"]
    day_theme: DisplayText
    primary_cluster_id: Identifier | None = None
    ordered_items: tuple[DraftItem, ...] = ()
    cross_cluster_segments: tuple[CrossClusterSegment, ...] = ()
    route_mode_selections: tuple[DraftRouteModeSelection, ...] = Field(
        default=(), exclude_if=lambda v: not v
    )
    dining_goals: tuple[Literal["breakfast", "lunch", "dinner", "snack"], ...] = ()
    transport_preferences: tuple[Literal["public_transit", "taxi", "walking", "driving"], ...] = (
        Field(min_length=1)
    )

    @model_validator(mode="after")
    def ordering_and_cross_cluster_coverage_are_exact(self) -> WorkingItineraryDay:
        onsite_count = sum(item.onsite_lunch for item in self.ordered_items)
        if onsite_count > 1 or (
            onsite_count and any(item.meal_slot == "lunch" for item in self.ordered_items)
        ):
            raise ValueError("choose one onsite lunch or one restaurant lunch, not both")
        if self.day_kind == "active" and self.primary_cluster_id is None:
            raise ValueError("an active day requires a primary_cluster_id")
        item_ids = [item.draft_item_id for item in self.ordered_items]
        require_unique(item_ids, "draft item IDs within a day")
        expected_positions = list(range(len(self.ordered_items)))
        if [item.position for item in self.ordered_items] != expected_positions:
            raise ValueError("draft item positions must be contiguous and match array order")
        require_unique(self.dining_goals, "daily dining goals")
        require_unique(self.transport_preferences, "daily transport preferences")
        require_unique(
            [
                (
                    item.origin.kind,
                    item.origin.reference_id,
                    item.destination.kind,
                    item.destination.reference_id,
                )
                for item in self.route_mode_selections
            ],
            "route mode selection endpoints",
        )

        item_by_id = {item.draft_item_id: item for item in self.ordered_items}
        covered_ids = [
            item_id
            for segment in self.cross_cluster_segments
            for item_id in segment.covered_item_ids
        ]
        require_unique(covered_ids, "cross-cluster item coverage")
        unknown = set(covered_ids) - set(item_by_id)
        if unknown:
            raise ValueError("cross-cluster segment references an unknown draft item")

        cross_cluster_ids = {
            item.draft_item_id
            for item in self.ordered_items
            if item.cluster_id is not None and item.cluster_id != self.primary_cluster_id
        }
        # Optional legacy explanation metadata. Actual adjacent routes are
        # derived from ordered items by the materializer.
        if covered_ids and set(covered_ids) != cross_cluster_ids:
            raise ValueError("when supplied, cross-cluster coverage must remain exact")
        for segment in self.cross_cluster_segments:
            if self.primary_cluster_id not in {segment.from_cluster_id, segment.to_cluster_id}:
                raise ValueError(
                    "a daily cross-cluster segment must connect to the primary cluster"
                )
            other_cluster = (
                segment.to_cluster_id
                if segment.from_cluster_id == self.primary_cluster_id
                else segment.from_cluster_id
            )
            if any(
                item_by_id[item_id].cluster_id != other_cluster
                for item_id in segment.covered_item_ids
            ):
                raise ValueError("covered items must belong to the segment's non-primary cluster")
        return self


class LodgingBaseline(V4ContractModel):
    mode: Literal["not_applicable", "fixed", "selected_offer", "unresolved"]
    fixed_commitment_ref: FixedCommitmentRef | None = None
    selected_offer_ref: HotelOfferRef | None = None
    unresolved_reason: Literal["provider_unavailable", "no_verified_hotel"] | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )

    @model_validator(mode="after")
    def mode_selects_exactly_one_reference(self) -> LodgingBaseline:
        if self.mode == "not_applicable":
            if (
                self.fixed_commitment_ref is not None
                or self.selected_offer_ref is not None
                or self.unresolved_reason is not None
            ):
                raise ValueError("not-applicable lodging cannot contain a hotel reference")
        elif self.mode == "fixed":
            if self.fixed_commitment_ref is None or self.selected_offer_ref is not None:
                raise ValueError("fixed lodging requires only fixed_commitment_ref")
            if self.fixed_commitment_ref.commitment_kind not in {
                "named_hotel",
                "existing_booking",
            }:
                raise ValueError("fixed lodging must reference a hotel commitment")
            if self.unresolved_reason is not None:
                raise ValueError("fixed lodging cannot contain an unresolved reason")
        elif self.mode == "selected_offer":
            if self.selected_offer_ref is None or self.fixed_commitment_ref is not None:
                raise ValueError("selected-offer lodging requires only selected_offer_ref")
            if self.unresolved_reason is not None:
                raise ValueError("selected lodging cannot contain an unresolved reason")
        elif (
            self.fixed_commitment_ref is not None
            or self.selected_offer_ref is not None
            or self.unresolved_reason is None
        ):
            raise ValueError("unresolved lodging requires only an unresolved reason")
        return self


class DiscardableObject(V4ContractModel):
    draft_item_id: Identifier
    mode: Literal["materializer_may_omit", "planner_review_if_infeasible"]
    trigger_codes: tuple[
        Literal["capacity_conflict", "route_conflict", "opening_conflict", "budget_conflict"],
        ...,
    ] = Field(min_length=1)
    discard_rank: int = Field(ge=0, strict=True)
    authorization_ref: Identifier
    reason_summary: DisplayText

    @model_validator(mode="after")
    def triggers_are_unique(self) -> DiscardableObject:
        require_unique(self.trigger_codes, "discard trigger codes")
        return self


class UnassignedIntent(V4ContractModel):
    candidate_ref: CandidateRef
    commitment_level: Literal["strong", "soft"]
    reason_code: Literal[
        "infeasible_date",
        "capacity_conflict",
        "route_conflict",
        "opening_conflict",
        "budget_conflict",
        "duplicate_experience",
        "planner_tradeoff",
        "user_requested",
        "awaiting_user",
    ]
    supporting_observation_refs: tuple[Identifier, ...] = Field(min_length=1)
    requires_user_resolution: bool

    @model_validator(mode="after")
    def strong_intent_requires_user_resolution(self) -> UnassignedIntent:
        require_unique(self.supporting_observation_refs, "unassigned intent observations")
        if self.commitment_level == "strong" and not self.requires_user_resolution:
            raise ValueError("an unassigned strong intent requires user resolution")
        return self


class WorkingItineraryDraft(V4ContractModel):
    """Planner-owned semantic skeleton before deterministic materialization."""

    draft_id: Identifier
    draft_revision: int = Field(ge=1, strict=True)
    content_digest: Digest
    scope: PlannerScope
    based_on_strategy_revision: int = Field(ge=1, strict=True)
    candidate_pool_revision: int = Field(ge=1, strict=True)
    spatial_observation_id: Identifier
    hotel_observation_id: Identifier | None = None
    lodging_baseline: LodgingBaseline
    days: tuple[WorkingItineraryDay, ...] = Field(min_length=1, max_length=5)
    discardable_objects: tuple[DiscardableObject, ...] = ()
    unassigned_intents: tuple[UnassignedIntent, ...] = ()
    reason_summary: DisplayText

    @model_validator(mode="after")
    def days_refs_and_discard_permissions_are_consistent(self) -> WorkingItineraryDraft:
        dates = [day.service_date for day in self.days]
        require_unique(dates, "working itinerary dates")
        if dates != sorted(dates):
            raise ValueError("working itinerary days must be date ordered")

        all_items = [item for day in self.days for item in day.ordered_items]
        item_ids = [item.draft_item_id for item in all_items]
        require_unique(item_ids, "draft item IDs")
        object_keys = [planner_object_ref_key(item.object_ref) for item in all_items]
        require_unique(object_keys, "draft object references")
        items_by_id = {item.draft_item_id: item for item in all_items}

        discard_ids = [item.draft_item_id for item in self.discardable_objects]
        require_unique(discard_ids, "discardable draft item IDs")
        require_unique(
            [item.discard_rank for item in self.discardable_objects],
            "discard ranks",
        )
        if set(discard_ids) - set(items_by_id):
            raise ValueError("discardable_objects must reference existing draft items")
        for discard in self.discardable_objects:
            commitment = items_by_id[discard.draft_item_id].commitment_level
            if commitment in {"immutable", "strong"}:
                raise ValueError("immutable and strong items cannot be discardable")
            if discard.mode == "materializer_may_omit" and commitment not in {
                "filler",
                "neutral",
            }:
                raise ValueError("only filler or neutral items may be omitted by materializer")
            if discard.mode == "planner_review_if_infeasible" and commitment != "soft":
                raise ValueError("planner-review discard authority is limited to soft items")

        unassigned_keys = [
            candidate_ref_key(item.candidate_ref) for item in self.unassigned_intents
        ]
        require_unique(unassigned_keys, "unassigned intent candidate references")
        assigned_candidate_keys = {
            candidate_ref_key(item.object_ref)
            for item in all_items
            if isinstance(item.object_ref, CandidateRef)
        }
        if assigned_candidate_keys & set(unassigned_keys):
            raise ValueError("a candidate cannot be both assigned and unassigned")

        if self.lodging_baseline.mode == "selected_offer":
            selected = self.lodging_baseline.selected_offer_ref
            if self.hotel_observation_id is None or (
                selected is not None and selected.hotel_observation_id != self.hotel_observation_id
            ):
                raise ValueError("selected hotel offer must use the current HotelObservation")
        elif (
            self.hotel_observation_id is not None and self.lodging_baseline.mode == "not_applicable"
        ):
            raise ValueError("a day trip cannot retain a HotelObservation")
        if self.lodging_baseline.fixed_commitment_ref is not None:
            fixed = self.lodging_baseline.fixed_commitment_ref
            if (
                fixed.task_book_id != self.scope.task_book_id
                or fixed.task_book_version != self.scope.task_book_version
            ):
                raise ValueError("fixed lodging must belong to the confirmed task book")
        return self


def validate_draft_against_pool(
    draft: WorkingItineraryDraft,
    pool: CandidatePoolSummary,
    *,
    expected_service_dates: tuple[date, ...] | None = None,
) -> None:
    """Validate current-pool membership, commitment integrity and date coverage."""

    require_same_scope_ownership(draft.scope, pool.scope)
    if draft.candidate_pool_revision != pool.revision:
        raise ValueError("draft must use the current candidate pool revision")

    entries = {candidate_ref_key(item.candidate_ref): item for item in pool.candidates}
    assigned_keys: set[tuple[str, int, str]] = set()
    for day in draft.days:
        for item in day.ordered_items:
            if not isinstance(item.object_ref, CandidateRef):
                if item.object_ref not in pool.fixed_commitments:
                    raise ValueError("draft references an unknown fixed commitment")
                continue
            key = candidate_ref_key(item.object_ref)
            entry = entries.get(key)
            if entry is None or entry.candidate_ref != item.object_ref:
                raise ValueError("draft references a candidate outside the current pool")
            if entry.commitment_level is CommitmentLevel.FORBIDDEN:
                raise ValueError("forbidden candidates cannot enter a working draft")
            if entry.eligibility in {"unavailable", "excluded"}:
                raise ValueError("an unavailable candidate cannot enter a working draft")
            if item.cluster_id is None or item.cluster_id not in entry.cluster_ids:
                raise ValueError("draft item must retain its observed candidate cluster")
            if item.commitment_level != entry.commitment_level.value:
                raise ValueError("draft cannot change a candidate's commitment level")
            if day.service_date in entry.infeasible_dates:
                raise ValueError("draft assigns a candidate to a known infeasible date")
            if entry.feasible_dates and day.service_date not in entry.feasible_dates:
                raise ValueError("draft assigns a candidate outside its feasible dates")
            assigned_keys.add(key)

    required = {key for key, entry in entries.items() if entry.selection_permission == "required"}
    unassigned = {candidate_ref_key(item.candidate_ref) for item in draft.unassigned_intents}
    if not required <= assigned_keys | unassigned:
        raise ValueError("required candidates must be assigned or recorded as unassigned blockers")
    soft = {key for key, entry in entries.items() if entry.commitment_level is CommitmentLevel.SOFT}
    if not soft <= assigned_keys | unassigned:
        raise ValueError("every soft intent must be assigned or explicitly recorded as unassigned")
    for intent in draft.unassigned_intents:
        entry = entries.get(candidate_ref_key(intent.candidate_ref))
        if entry is None or entry.candidate_ref != intent.candidate_ref:
            raise ValueError("unassigned intent references a candidate outside the pool")
        if intent.commitment_level != entry.commitment_level.value:
            raise ValueError("unassigned intent cannot change commitment level")

    if expected_service_dates is not None:
        require_unique(expected_service_dates, "expected service dates")
        actual_dates = tuple(day.service_date for day in draft.days)
        if actual_dates != tuple(sorted(expected_service_dates)):
            raise ValueError("working draft must contain every trip date exactly once")


def canonical_planning_projection(draft: WorkingItineraryDraft) -> dict[str, Any]:
    """Return only semantic planning fields used to detect an effective change."""

    days: list[dict[str, Any]] = []
    for day in draft.days:
        days.append(
            {
                "service_date": day.service_date.isoformat(),
                "day_kind": day.day_kind,
                "primary_cluster_id": day.primary_cluster_id,
                "ordered_items": [
                    {
                        "draft_item_id": item.draft_item_id,
                        "position": item.position,
                        "item_kind": item.item_kind,
                        "object_ref": item.object_ref.model_dump(mode="json"),
                        "cluster_id": item.cluster_id,
                        "expected_window": item.expected_window.model_dump(mode="json"),
                        "meal_slot": item.meal_slot,
                        "duration_preference": item.duration_preference,
                        **({"onsite_lunch": True} if item.onsite_lunch else {}),
                        "commitment_level": item.commitment_level,
                    }
                    for item in day.ordered_items
                ],
                "cross_cluster_segments": [
                    segment.model_dump(mode="json") for segment in day.cross_cluster_segments
                ],
                "transport_preferences": sorted(day.transport_preferences),
                **(
                    {
                        "route_mode_selections": [
                            item.model_dump(mode="json") for item in day.route_mode_selections
                        ]
                    }
                    if day.route_mode_selections
                    else {}
                ),
            }
        )
    return {
        "lodging_baseline": draft.lodging_baseline.model_dump(mode="json"),
        "days": days,
        "discardable_objects": sorted(
            [
                {
                    "draft_item_id": item.draft_item_id,
                    "mode": item.mode,
                    "trigger_codes": sorted(item.trigger_codes),
                    "discard_rank": item.discard_rank,
                    "authorization_ref": item.authorization_ref,
                }
                for item in draft.discardable_objects
            ],
            key=lambda value: value["draft_item_id"],
        ),
        "unassigned_intents": sorted(
            [
                {
                    "candidate_ref": item.candidate_ref.model_dump(mode="json"),
                    "commitment_level": item.commitment_level,
                    "reason_code": item.reason_code,
                    "supporting_observation_refs": sorted(item.supporting_observation_refs),
                    "requires_user_resolution": item.requires_user_resolution,
                }
                for item in draft.unassigned_intents
            ],
            key=lambda value: (
                value["candidate_ref"]["candidate_pool_id"],
                value["candidate_ref"]["candidate_id"],
            ),
        ),
    }


def planning_projection_digest(draft: WorkingItineraryDraft) -> str:
    """Compute a deterministic digest of the canonical planning projection."""

    encoded = json.dumps(
        canonical_planning_projection(draft),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


V4_PLANNER_DRAFT_CONTRACTS: tuple[type[V4ContractModel], ...] = (WorkingItineraryDraft,)
