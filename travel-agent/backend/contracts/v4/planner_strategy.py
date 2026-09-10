"""Candidate-pool and complete strategy contracts for the V4 Planner."""

from __future__ import annotations

from datetime import date, time
from typing import Literal

from pydantic import AwareDatetime, Field, model_validator

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
    CommitmentLevel,
    CrossClusterReasonCode,
    PlannerCapability,
)
from backend.contracts.v4.planner_refs import (
    CandidateRef,
    FixedCommitmentRef,
    PlannerObjectRef,
    PlannerScope,
    candidate_ref_key,
    fixed_commitment_ref_key,
    planner_object_ref_key,
    require_same_scope_ownership,
)

MAX_DAILY_MAJOR_ACTIVITIES = 4


class CandidateAdvisoryFeatures(V4ContractModel):
    preference_fit: Literal["low", "medium", "high", "unknown"]
    city_representativeness: Literal["low", "medium", "high", "unknown"]
    accessibility_fit: Literal["low", "medium", "high", "unknown"]
    typical_duration_minutes: int | None = Field(default=None, ge=1, le=1_440, strict=True)
    price_band: Identifier | None = None


class CandidatePoolEntry(V4ContractModel):
    candidate_ref: CandidateRef
    display_name: DisplayText
    entity_kind: CandidateEntityKind
    commitment_level: CommitmentLevel
    source_intent_refs: tuple[Identifier, ...] = ()
    selection_permission: Literal["required", "allowed", "filler_only", "forbidden"]
    eligibility: Literal["eligible", "needs_evidence", "unavailable", "excluded"]
    fact_reference_ids: tuple[Identifier, ...] = ()
    cluster_ids: tuple[Identifier, ...] = ()
    feasible_dates: tuple[date, ...] = ()
    infeasible_dates: tuple[date, ...] = ()
    missing_fact_kinds: tuple[Identifier, ...] = ()
    fixed_window_refs: tuple[Identifier, ...] = ()
    advisory_features: CandidateAdvisoryFeatures

    @model_validator(mode="after")
    def permission_and_dates_match_commitment(self) -> CandidatePoolEntry:
        if self.entity_kind is not self.candidate_ref.entity_kind:
            raise ValueError("candidate entry entity_kind must match its CandidateRef")
        for field_name, values in (
            ("source_intent_refs", self.source_intent_refs),
            ("fact_reference_ids", self.fact_reference_ids),
            ("cluster_ids", self.cluster_ids),
            ("feasible_dates", self.feasible_dates),
            ("infeasible_dates", self.infeasible_dates),
            ("missing_fact_kinds", self.missing_fact_kinds),
            ("fixed_window_refs", self.fixed_window_refs),
        ):
            require_unique(values, field_name)
        if set(self.feasible_dates) & set(self.infeasible_dates):
            raise ValueError("a candidate date cannot be both feasible and infeasible")
        if (
            self.commitment_level in {CommitmentLevel.IMMUTABLE, CommitmentLevel.STRONG}
            and self.selection_permission != "required"
        ):
            raise ValueError(
                "immutable and strong candidates require selection_permission=required"
            )
        if self.commitment_level is CommitmentLevel.FORBIDDEN:
            if self.selection_permission != "forbidden":
                raise ValueError("forbidden candidates require selection_permission=forbidden")
        elif self.selection_permission == "forbidden":
            raise ValueError("only forbidden candidates may have forbidden selection permission")
        if self.selection_permission == "filler_only" and self.commitment_level not in {
            CommitmentLevel.FILLER,
            CommitmentLevel.NEUTRAL,
        }:
            raise ValueError("filler_only is limited to filler or neutral candidates")
        if self.selection_permission == "required" and self.eligibility in {
            "unavailable",
            "excluded",
        }:
            raise ValueError("an unavailable required candidate must be reported as a blocker")
        return self


class CandidatePoolSummary(V4ContractModel):
    """Read-only, sourced candidate observation; ranking remains advisory."""

    candidate_pool_id: Identifier
    revision: int = Field(ge=1, strict=True)
    scope: PlannerScope
    pool_fingerprint: Digest
    spatial_observation_id: Identifier | None = None
    generated_at: AwareDatetime
    source_reference_ids: tuple[Identifier, ...] = Field(min_length=1)
    candidates: tuple[CandidatePoolEntry, ...]
    fixed_commitments: tuple[FixedCommitmentRef, ...] = ()
    missing_required_candidate_refs: tuple[Identifier, ...] = ()

    @model_validator(mode="after")
    def refs_belong_to_current_pool_and_task_book(self) -> CandidatePoolSummary:
        require_unique(self.source_reference_ids, "candidate pool source_reference_ids")
        require_unique(
            self.missing_required_candidate_refs,
            "missing_required_candidate_refs",
        )
        candidate_ids = [entry.candidate_ref.candidate_id for entry in self.candidates]
        require_unique(candidate_ids, "candidate IDs")
        canonical_ids = [
            (entry.entity_kind, entry.candidate_ref.canonical_entity_id)
            for entry in self.candidates
        ]
        require_unique(canonical_ids, "canonical candidate identities")
        for entry in self.candidates:
            candidate_reference = entry.candidate_ref
            if (
                candidate_reference.candidate_pool_id != self.candidate_pool_id
                or candidate_reference.candidate_pool_revision != self.revision
            ):
                raise ValueError("CandidateRef must use the current pool ID and revision")
        fixed_keys = [fixed_commitment_ref_key(item) for item in self.fixed_commitments]
        require_unique(fixed_keys, "fixed commitment references")
        for fixed_reference in self.fixed_commitments:
            if (
                fixed_reference.task_book_id != self.scope.task_book_id
                or fixed_reference.task_book_version != self.scope.task_book_version
            ):
                raise ValueError("fixed commitment must belong to the confirmed task book")
        return self

    def candidate_by_id(self) -> dict[str, CandidatePoolEntry]:
        return {entry.candidate_ref.candidate_id: entry for entry in self.candidates}


class StrategyAnchorPolicy(V4ContractModel):
    immutable_refs: tuple[FixedCommitmentRef, ...] = ()
    strong_candidate_refs: tuple[CandidateRef, ...] = ()
    soft_candidate_refs: tuple[CandidateRef, ...] = ()
    filler_candidate_refs: tuple[CandidateRef, ...] = ()

    @model_validator(mode="after")
    def anchor_sets_are_unique_and_disjoint(self) -> StrategyAnchorPolicy:
        require_unique(
            [fixed_commitment_ref_key(item) for item in self.immutable_refs],
            "immutable_refs",
        )
        candidate_sets = {
            "strong_candidate_refs": self.strong_candidate_refs,
            "soft_candidate_refs": self.soft_candidate_refs,
            "filler_candidate_refs": self.filler_candidate_refs,
        }
        keys_by_name: dict[str, set[tuple[str, int, str]]] = {}
        for name, values in candidate_sets.items():
            keys = [candidate_ref_key(item) for item in values]
            require_unique(keys, name)
            keys_by_name[name] = set(keys)
        if any(
            keys_by_name[left] & keys_by_name[right]
            for left, right in (
                ("strong_candidate_refs", "soft_candidate_refs"),
                ("strong_candidate_refs", "filler_candidate_refs"),
                ("soft_candidate_refs", "filler_candidate_refs"),
            )
        ):
            raise ValueError("candidate anchor bands must be disjoint")
        return self


class CandidatePriority(V4ContractModel):
    candidate_ref: CandidateRef
    priority_band: Literal["preserve_first", "normal", "drop_first"]
    reason_code: Literal[
        "user_commitment",
        "trip_theme",
        "spatial_fit",
        "dining_role",
        "delegated_choice",
    ]


class PreferredTimeWindow(V4ContractModel):
    earliest: time | None = None
    latest: time | None = None

    @model_validator(mode="after")
    def bounds_are_ordered(self) -> PreferredTimeWindow:
        if self.earliest is not None and self.latest is not None and self.latest < self.earliest:
            raise ValueError("preferred time window cannot run backwards")
        return self


class ActivityTargetRange(V4ContractModel):
    minimum: int = Field(ge=0, le=MAX_DAILY_MAJOR_ACTIVITIES, strict=True)
    maximum: int = Field(ge=0, le=MAX_DAILY_MAJOR_ACTIVITIES, strict=True)

    @model_validator(mode="after")
    def bounds_are_ordered(self) -> ActivityTargetRange:
        if self.maximum < self.minimum:
            raise ValueError("major activity target maximum cannot be below minimum")
        return self


class WalkingPolicy(V4ContractModel):
    goal: Literal["minimize", "balanced", "no_preference"]
    hard_limit_ref: Identifier | None = None


class RestPolicy(V4ContractModel):
    mode: Literal["regular", "as_needed", "minimal"]
    preferred_windows: tuple[Identifier, ...] = ()
    hard_requirement_refs: tuple[Identifier, ...] = ()

    @model_validator(mode="after")
    def references_are_unique(self) -> RestPolicy:
        require_unique(self.preferred_windows, "preferred rest windows")
        require_unique(self.hard_requirement_refs, "rest hard requirements")
        return self


class PerDateCapacityOverride(V4ContractModel):
    service_date: date
    day_kind: Literal["active", "arrival_departure", "rest"]
    available_window_ref: Identifier | None = None
    activity_load: Literal["light", "normal", "full"]


class DailyCapacityPolicy(V4ContractModel):
    pace_profile: Literal["relaxed", "balanced", "intensive", "custom"]
    preferred_start_window: PreferredTimeWindow
    preferred_end_window: PreferredTimeWindow
    major_activity_target: ActivityTargetRange
    walking_policy: WalkingPolicy
    rest_policy: RestPolicy
    source_constraint_refs: tuple[Identifier, ...] = ()
    per_date_overrides: tuple[PerDateCapacityOverride, ...] = ()

    @model_validator(mode="after")
    def dates_and_references_are_unique(self) -> DailyCapacityPolicy:
        require_unique(self.source_constraint_refs, "daily capacity source constraints")
        dates = [item.service_date for item in self.per_date_overrides]
        require_unique(dates, "per-date capacity overrides")
        if dates != sorted(dates):
            raise ValueError("per-date capacity overrides must be date ordered")
        return self


class SpatialPolicy(V4ContractModel):
    preferred_transport_modes: tuple[
        Literal["public_transit", "taxi", "walking", "driving"], ...
    ] = Field(min_length=1)
    prefer_single_primary_cluster: bool
    split_large_cluster: bool
    merge_adjacent_clusters: bool
    allowed_cross_cluster_reasons: tuple[CrossClusterReasonCode, ...]

    @model_validator(mode="after")
    def modes_and_reasons_are_unique(self) -> SpatialPolicy:
        require_unique(self.preferred_transport_modes, "preferred transport modes")
        require_unique(self.allowed_cross_cluster_reasons, "allowed cross-cluster reasons")
        return self


class LodgingPolicy(V4ContractModel):
    mode: Literal["not_applicable", "fixed", "search"]
    fixed_commitment_ref: FixedCommitmentRef | None = None
    preferred_area_refs: tuple[Identifier, ...] = ()
    selection_objectives: tuple[
        Literal[
            "minimize_total_commute",
            "minimize_walking",
            "transit_convenience",
            "better_value",
            "preferred_atmosphere",
            "facility_fit",
        ],
        ...,
    ] = ()
    budget_constraint_ref: Identifier | None = None
    facility_constraint_refs: tuple[Identifier, ...] = ()

    @model_validator(mode="after")
    def mode_controls_fixed_reference(self) -> LodgingPolicy:
        require_unique(self.preferred_area_refs, "preferred lodging areas")
        require_unique(self.selection_objectives, "lodging objectives")
        require_unique(self.facility_constraint_refs, "lodging facility constraints")
        if self.mode == "fixed" and self.fixed_commitment_ref is None:
            raise ValueError("fixed lodging policy requires a fixed commitment")
        if self.mode != "fixed" and self.fixed_commitment_ref is not None:
            raise ValueError("only fixed lodging policy may contain a fixed commitment")
        if self.mode == "not_applicable" and (
            self.preferred_area_refs
            or self.selection_objectives
            or self.budget_constraint_ref is not None
            or self.facility_constraint_refs
        ):
            raise ValueError("not-applicable lodging cannot contain search preferences")
        return self


class RequiredMealWindow(V4ContractModel):
    service_date: date
    meal: Literal["breakfast", "lunch", "dinner"]
    window_ref: Identifier | None = None


class DiningPolicy(V4ContractModel):
    destination_restaurant_refs: tuple[CandidateRef, ...] = ()
    required_meal_windows: tuple[RequiredMealWindow, ...] = ()
    flexible_meal_placement: Literal[
        "near_route",
        "near_primary_cluster",
        "near_hotel",
        "delegated",
    ]
    dietary_constraint_refs: tuple[Identifier, ...] = ()

    @model_validator(mode="after")
    def dining_references_are_unique(self) -> DiningPolicy:
        require_unique(
            [candidate_ref_key(item) for item in self.destination_restaurant_refs],
            "destination restaurant references",
        )
        require_unique(
            [(item.service_date, item.meal) for item in self.required_meal_windows],
            "required meal windows",
        )
        require_unique(self.dietary_constraint_refs, "dietary constraints")
        return self


class ConflictPolicy(V4ContractModel):
    hard_guard_refs: tuple[Identifier, ...] = ()
    flexible_tradeoff_order: tuple[
        Literal[
            "retain_soft_attractions",
            "retain_destination_meals",
            "minimize_walking",
            "minimize_transfers",
            "minimize_total_cost",
            "preserve_late_start",
            "maximize_local_distinctiveness",
            "minimize_cross_cluster",
            "lodging_experience",
        ],
        ...,
    ] = Field(min_length=1)
    ask_user_trigger_codes: tuple[AskUserReasonCode, ...] = ()

    @model_validator(mode="after")
    def guards_and_soft_order_are_unique(self) -> ConflictPolicy:
        require_unique(self.hard_guard_refs, "hard guard references")
        require_unique(self.flexible_tradeoff_order, "flexible tradeoff order")
        require_unique(self.ask_user_trigger_codes, "ask-user trigger codes")
        return self


class PendingEvidenceNeed(V4ContractModel):
    evidence_need_id: Identifier
    capability: PlannerCapability
    target_refs: tuple[PlannerObjectRef, ...] = ()
    affected_dates: tuple[date, ...] = ()
    blocking: bool

    @model_validator(mode="after")
    def targets_and_dates_are_unique(self) -> PendingEvidenceNeed:
        require_unique(
            [planner_object_ref_key(item) for item in self.target_refs],
            "pending evidence targets",
        )
        require_unique(self.affected_dates, "pending evidence dates")
        return self


class NonBlockingAssumption(V4ContractModel):
    assumption_id: Identifier
    summary: DisplayText
    source_ref: Identifier


class PlanningStrategy(V4ContractModel):
    """Complete, auditable Planner strategy snapshot."""

    strategy_id: Identifier
    strategy_revision: int = Field(ge=1, strict=True)
    scope: PlannerScope
    candidate_pool_revision: int = Field(ge=1, strict=True)
    core_experience_summary: DisplayText
    anchor_policy: StrategyAnchorPolicy
    candidate_priority: tuple[CandidatePriority, ...] = ()
    daily_capacity_policy: DailyCapacityPolicy
    spatial_policy: SpatialPolicy
    lodging_policy: LodgingPolicy
    dining_policy: DiningPolicy
    conflict_policy: ConflictPolicy
    pending_evidence: tuple[PendingEvidenceNeed, ...] = ()
    non_blocking_assumptions: tuple[NonBlockingAssumption, ...] = ()
    reason_summary: DisplayText

    @model_validator(mode="after")
    def strategy_references_are_internally_consistent(self) -> PlanningStrategy:
        candidate_refs = (
            *self.anchor_policy.strong_candidate_refs,
            *self.anchor_policy.soft_candidate_refs,
            *self.anchor_policy.filler_candidate_refs,
            *(item.candidate_ref for item in self.candidate_priority),
            *self.dining_policy.destination_restaurant_refs,
        )
        for candidate_reference in candidate_refs:
            if candidate_reference.candidate_pool_revision != self.candidate_pool_revision:
                raise ValueError("strategy CandidateRef must use its candidate_pool_revision")
        require_unique(
            [candidate_ref_key(item.candidate_ref) for item in self.candidate_priority],
            "candidate priority references",
        )
        require_unique(
            [item.evidence_need_id for item in self.pending_evidence],
            "pending evidence IDs",
        )
        require_unique(
            [item.assumption_id for item in self.non_blocking_assumptions],
            "non-blocking assumption IDs",
        )
        for fixed_reference in self.anchor_policy.immutable_refs:
            _require_fixed_ref_matches_scope(fixed_reference, self.scope)
        if self.lodging_policy.fixed_commitment_ref is not None:
            _require_fixed_ref_matches_scope(self.lodging_policy.fixed_commitment_ref, self.scope)
        return self


def _require_fixed_ref_matches_scope(reference: FixedCommitmentRef, scope: PlannerScope) -> None:
    if (
        reference.task_book_id != scope.task_book_id
        or reference.task_book_version != scope.task_book_version
    ):
        raise ValueError("fixed commitment does not belong to the strategy task book")


def validate_strategy_against_pool(
    strategy: PlanningStrategy,
    pool: CandidatePoolSummary,
) -> None:
    """Apply contextual pool and commitment guards omitted from bare JSON Schema."""

    require_same_scope_ownership(strategy.scope, pool.scope)
    if strategy.candidate_pool_revision != pool.revision:
        raise ValueError("strategy must use the current candidate pool revision")

    entries = {candidate_ref_key(item.candidate_ref): item for item in pool.candidates}
    referenced = (
        *strategy.anchor_policy.strong_candidate_refs,
        *strategy.anchor_policy.soft_candidate_refs,
        *strategy.anchor_policy.filler_candidate_refs,
        *(item.candidate_ref for item in strategy.candidate_priority),
        *strategy.dining_policy.destination_restaurant_refs,
    )
    for reference in referenced:
        entry = entries.get(candidate_ref_key(reference))
        if entry is None or entry.candidate_ref != reference:
            raise ValueError("strategy references a candidate outside the current pool")
        if entry.commitment_level is CommitmentLevel.FORBIDDEN:
            raise ValueError("strategy cannot reference a forbidden candidate")

    required_strong = {
        key for key, entry in entries.items() if entry.commitment_level is CommitmentLevel.STRONG
    }
    actual_strong = {
        candidate_ref_key(item) for item in strategy.anchor_policy.strong_candidate_refs
    }
    if actual_strong != required_strong:
        raise ValueError("strategy must completely cover current strong candidate commitments")

    expected_soft = {
        key for key, entry in entries.items() if entry.commitment_level is CommitmentLevel.SOFT
    }
    if {
        candidate_ref_key(item) for item in strategy.anchor_policy.soft_candidate_refs
    } != expected_soft:
        raise ValueError("strategy must retain every soft intent, even when later unassigned")

    expected_fixed = {fixed_commitment_ref_key(item) for item in pool.fixed_commitments}
    actual_fixed = {
        fixed_commitment_ref_key(item) for item in strategy.anchor_policy.immutable_refs
    }
    if actual_fixed != expected_fixed:
        raise ValueError("strategy must completely cover current immutable commitments")

    levels = {
        "soft_candidate_refs": CommitmentLevel.SOFT,
        "filler_candidate_refs": CommitmentLevel.FILLER,
    }
    for field_name, expected_level in levels.items():
        for reference in getattr(strategy.anchor_policy, field_name):
            actual = entries[candidate_ref_key(reference)].commitment_level
            allowed = (
                {CommitmentLevel.FILLER, CommitmentLevel.NEUTRAL}
                if expected_level is CommitmentLevel.FILLER
                else {expected_level}
            )
            if actual not in allowed:
                raise ValueError(
                    f"{field_name} contains a candidate from the wrong commitment band"
                )


V4_PLANNER_STRATEGY_CONTRACTS: tuple[type[V4ContractModel], ...] = (
    CandidatePoolSummary,
    PlanningStrategy,
)
