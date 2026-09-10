"""Compact Qwen wire contract: local keys in, server-owned public artifacts out."""

from __future__ import annotations

from datetime import date, time
from typing import Annotated, Literal, TypeAlias

from pydantic import Field, RootModel, model_validator

from backend.contracts.v4.base import DisplayText, Identifier, V4ContractModel
from backend.contracts.v4.enums import (
    CandidateEntityKind,
    CrossClusterReasonCode,
    PlannerCapability,
)
from backend.contracts.v4.planner_draft import ExpectedWindow
from backend.contracts.v4.planner_strategy import (
    ConflictPolicy,
    DailyCapacityPolicy,
    RequiredMealWindow,
    SpatialPolicy,
)

HotelKey = Annotated[Identifier, Field(pattern=r"^h[1-9][0-9]*$")]
CandidateKey = Annotated[Identifier, Field(pattern=r"^c[1-9][0-9]*$")]
RepairObjectKey = Annotated[
    Identifier,
    Field(
        pattern=r"^[cf][1-9][0-9]*$",
        description="完整复制当前对象短键，例如 c2 或 f1；不能只填 c/f。",
    ),
]
ValidationIssueKey = Annotated[
    Identifier,
    Field(
        pattern=r"^v[1-9][0-9]*$",
        description="完整复制 current_issue_keys 中的问题短键，例如 v1；不能只填 v。",
    ),
]


class ModelPlanStop(V4ContractModel):
    """One semantic stop choice; all formal schedule fields are server-owned."""

    candidate_key: CandidateKey
    part_of_day: Literal["morning", "midday", "afternoon", "evening", "anytime"] = "anytime"
    meal_slot: Literal["lunch", "dinner"] | None = None
    duration_preference: Literal["short", "normal", "extended"] | None = None
    onsite_lunch: bool = False


class ModelPlanDay(V4ContractModel):
    """A model-owned daily theme and ordered list of current candidate keys."""

    day_index: int = Field(ge=1, le=5, strict=True)
    theme: DisplayText
    stops: tuple[ModelPlanStop, ...] = Field(default=(), max_length=8)


class ModelPlanIntent(V4ContractModel):
    """Compact Qwen output compiled into the formal Planner contracts by code."""

    days: tuple[ModelPlanDay, ...] = Field(min_length=1, max_length=5)
    selected_hotel_key: HotelKey | None = None
    overall_rationale: DisplayText


class ModelCandidatePriority(V4ContractModel):
    candidate_key: Identifier
    priority_band: Literal["preserve_first", "normal", "drop_first"]
    reason_code: Literal[
        "user_commitment", "trip_theme", "spatial_fit", "dining_role", "delegated_choice"
    ]


class ModelLodgingPolicy(V4ContractModel):
    mode: Literal["not_applicable", "fixed", "search"]
    fixed_commitment_key: Identifier | None = None
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


class ModelMainMealWindow(RequiredMealWindow):
    """New model requests are narrower than readable historical strategies."""

    meal: Literal["lunch", "dinner"]


class ModelDiningPolicy(V4ContractModel):
    required_meal_windows: tuple[ModelMainMealWindow, ...]
    flexible_meal_placement: Literal[
        "near_route", "near_primary_cluster", "near_hotel", "delegated"
    ]
    dietary_constraint_refs: tuple[Identifier, ...] = ()


class ModelEvidenceNeed(V4ContractModel):
    local_key: Identifier
    capability: PlannerCapability
    target_keys: tuple[Identifier, ...] = Field(
        default=(),
        description="仅当前 c/f 键；酒店或天气等整体需求用空数组，不填 h/g。",
    )
    affected_dates: tuple[date, ...] = ()
    blocking: bool


class ModelAssumption(V4ContractModel):
    local_key: Identifier
    summary: DisplayText
    source_ref: Identifier = Field(
        description=(
            "One existing task_book_reference_keys key or evidence_keys key. "
            "Never invent assumption:N or a JSON path."
        )
    )


class ModelStrategy(V4ContractModel):
    core_experience_summary: DisplayText
    candidate_priority: tuple[ModelCandidatePriority, ...]
    daily_capacity_policy: DailyCapacityPolicy
    spatial_policy: SpatialPolicy
    lodging_policy: ModelLodgingPolicy
    dining_policy: ModelDiningPolicy
    conflict_policy: ConflictPolicy
    pending_evidence: tuple[ModelEvidenceNeed, ...]
    non_blocking_assumptions: tuple[ModelAssumption, ...]
    reason_summary: DisplayText


class ModelDraftItem(V4ContractModel):
    object_key: Identifier = Field(
        description=(
            "只引用当前 candidates 的 c 键或 fixed_commitments 的 f 键。"
            "h 酒店仅放 lodging_baseline，不得作为活动；没有真实 f 键就不添加到达/离开固定事件。"
        )
    )
    item_kind: Literal["visit", "dining", "fixed_event", "arrival", "departure"]
    expected_window: ExpectedWindow
    meal_slot: Literal["lunch", "dinner"] | None = None
    duration_preference: Literal["short", "normal", "extended"] | None = None


class ModelCrossClusterSegment(V4ContractModel):
    from_cluster_key: Identifier
    to_cluster_key: Identifier
    covered_object_keys: tuple[Identifier, ...] = Field(
        min_length=1,
        description=(
            "直接引用同一天 ordered_items.object_key 的 c/f 短键，不创造另一层活动编号。"
            "恰好覆盖属于本段非主簇的活动；一天内所有非主簇活动必须各覆盖一次，不能包含主簇项。"
        ),
    )
    reason_code: CrossClusterReasonCode
    supporting_evidence_keys: tuple[Identifier, ...] = Field(
        min_length=1,
        description=(
            "只填当轮 evidence_keys 的键，如 e1、spatial、hotel 或 intent:c1:1；"
            "不能填 c/g/r/h 对象键或原始 ID。"
        ),
    )
    route_edge_keys: tuple[Identifier, ...] = Field(
        min_length=1,
        description=(
            "引用当前 routes 的 r 键，须覆盖这段实际相邻地点的进出路线，不能填事实 e 键。"
            "每个实际有向边界只能选择一个 r 键；同一端点、不同交通方式是备选，不能全部填入。"
        ),
    )
    comparison_observation_key: Identifier | None = None


class ModelDraftDay(V4ContractModel):
    service_date: date
    day_kind: Literal["active", "arrival_departure", "rest"]
    day_theme: DisplayText
    primary_cluster_key: Identifier | None = None
    ordered_items: tuple[ModelDraftItem, ...]
    cross_cluster_segments: tuple[ModelCrossClusterSegment, ...] = Field(
        default=(),
        description=(
            "只描述 ordered_items 中实际相邻地点跨越不同活动簇的边界。"
            "lodging_baseline 的酒店不属于 ordered_items，酒店与活动簇不同也不能创建本字段。"
            "当天所有 ordered_items 都在 primary_cluster_key 时必须为 []。"
        ),
    )
    dining_goals: tuple[Literal["lunch", "dinner"], ...]
    transport_preferences: tuple[Literal["public_transit", "taxi", "walking", "driving"], ...] = (
        Field(min_length=1)
    )


class ModelDiscardable(V4ContractModel):
    object_key: Identifier = Field(description="直接引用已安排的 ordered_items.object_key。")
    mode: Literal["materializer_may_omit", "planner_review_if_infeasible"]
    trigger_codes: tuple[
        Literal["capacity_conflict", "route_conflict", "opening_conflict", "budget_conflict"], ...
    ] = Field(min_length=1)
    discard_rank: int = Field(ge=0, strict=True)
    authorization_key: Identifier = Field(
        description=(
            "只填当轮 evidence_keys 的键，如 pool 或 strategy；"
            "不得创造 materializer/constraint 等键。"
        )
    )
    reason_summary: DisplayText


class ModelUnassigned(V4ContractModel):
    candidate_key: Identifier = Field(description="只允许当前 commitment=soft/strong 的候选 c 键。")
    reason_code: Literal[
        "infeasible_date",
        "capacity_conflict",
        "route_conflict",
        "opening_conflict",
        "budget_conflict",
        "duplicate_experience",
        "awaiting_user",
    ]
    observation_keys: tuple[Identifier, ...] = Field(
        min_length=1,
        description=(
            "引用当轮 evidence_keys：容量依据用 strategy，路线依据用 spatial，"
            "具体事实用已有 e/fact 键。"
        ),
    )
    requires_user_resolution: bool


class ModelLodgingBaseline(V4ContractModel):
    mode: Literal["not_applicable", "fixed", "search", "selected_offer"] | None = Field(
        default=None,
        description=("兼容旧模型输出；服务器从已接受 lodging_policy 编译正式 mode，不信任本字段。"),
    )
    fixed_commitment_key: Identifier | None = Field(
        default=None,
        description="兼容旧模型输出；固定住宿引用由服务器从已接受策略编译。",
    )
    selected_offer_key: HotelKey | None = Field(
        default=None,
        description="从 available_hotel_offer_keys 选综合最佳 h 键，不填名称或公共 ID。",
    )
    better_value_offer_key: HotelKey | None = Field(
        default=None,
        description="从 available_hotel_offer_keys 选更高性价比备选 h 键。",
    )
    alternative_experience_offer_key: HotelKey | None = Field(
        default=None,
        description="从 available_hotel_offer_keys 选不同区位或体验备选 h 键。",
    )

    @model_validator(mode="after")
    def search_roles_are_distinct_when_present(self) -> ModelLodgingBaseline:
        role_keys = tuple(
            key
            for key in (
                self.selected_offer_key,
                self.better_value_offer_key,
                self.alternative_experience_offer_key,
            )
            if key is not None
        )
        if len(set(role_keys)) != len(role_keys):
            raise ValueError("hotel recommendation roles require distinct h keys")
        return self


class ModelWorkingDraft(V4ContractModel):
    lodging_baseline: ModelLodgingBaseline
    days: tuple[ModelDraftDay, ...] = Field(min_length=1, max_length=5)
    discardable_objects: tuple[ModelDiscardable, ...]
    unassigned_intents: tuple[ModelUnassigned, ...]
    reason_summary: DisplayText


class ModelRecallArgs(V4ContractModel):
    capability: Literal["candidate_recall"]
    domain: CandidateEntityKind
    gap_code: Identifier
    task_book_preference_refs: tuple[Identifier, ...] = Field(min_length=1)
    nearby_candidate_keys: tuple[Identifier, ...] = ()
    nearby_cluster_keys: tuple[Identifier, ...] = ()
    limit: int = Field(ge=1, le=20, strict=True)


class ModelPlaceArgs(V4ContractModel):
    capability: Literal["place_facts"]
    candidate_keys: tuple[Identifier, ...] = Field(min_length=1, max_length=20)
    fact_kinds: tuple[Identifier, ...] = Field(min_length=1)


class ModelHoursArgs(V4ContractModel):
    capability: Literal["opening_hours"]
    candidate_keys: tuple[Identifier, ...] = Field(min_length=1, max_length=20)
    service_dates: tuple[date, ...] = Field(min_length=1, max_length=5)


class ModelTicketArgs(V4ContractModel):
    capability: Literal["ticket_availability"]
    candidate_keys: tuple[Identifier, ...] = Field(min_length=1, max_length=20)
    service_dates: tuple[date, ...] = Field(min_length=1, max_length=5)
    party_size_ref: Identifier


class ModelWeatherArgs(V4ContractModel):
    capability: Literal["weather_forecast"]
    destination_ref: Identifier
    service_dates: tuple[date, ...] = Field(min_length=1, max_length=5)
    weather_fields: tuple[Identifier, ...] = Field(min_length=1)


class ModelRoutePair(V4ContractModel):
    origin_key: Identifier
    destination_key: Identifier


class ModelComparisonDay(V4ContractModel):
    service_date: date
    ordered_endpoint_keys: tuple[Identifier, ...] = Field(max_length=24)


class ModelRouteComparison(V4ContractModel):
    baseline_days: tuple[ModelComparisonDay, ...] = Field(min_length=1, max_length=5)
    proposed_days: tuple[ModelComparisonDay, ...] = Field(min_length=1, max_length=5)
    transport_mode: Literal["public_transit", "taxi", "walking", "driving"]


class ModelRouteArgs(V4ContractModel):
    capability: Literal["spatial_routes"]
    endpoint_pairs: tuple[ModelRoutePair, ...] = Field(min_length=1, max_length=40)
    transport_modes: tuple[Literal["public_transit", "taxi", "walking", "driving"], ...] = Field(
        min_length=1
    )
    departure_service_date: date
    departure_window: tuple[time, time] | None = None
    comparison: ModelRouteComparison | None = None


class ModelHotelArgs(V4ContractModel):
    capability: Literal["hotel_search"]
    check_in_date: date
    check_out_date: date
    party_size_ref: Identifier
    lodging_preference_refs: tuple[Identifier, ...] = ()
    budget_constraint_ref: Identifier | None = None
    facility_constraint_refs: tuple[Identifier, ...] = ()
    activity_cluster_keys: tuple[Identifier, ...] = Field(min_length=1)


class ModelHotelRefreshArgs(V4ContractModel):
    capability: Literal["hotel_offer_refresh"]
    offer_key: HotelKey = Field(
        description="从 available_hotel_offer_keys 选择 h 键；不使用 Provider 商品 ID 或 UUID。"
    )
    check_in_date: date
    check_out_date: date


ModelCapabilityArgs = Annotated[
    ModelRecallArgs
    | ModelPlaceArgs
    | ModelHoursArgs
    | ModelTicketArgs
    | ModelWeatherArgs
    | ModelRouteArgs
    | ModelHotelArgs
    | ModelHotelRefreshArgs,
    Field(discriminator="capability"),
]


class ModelCapabilityRequest(V4ContractModel):
    local_key: Identifier
    purpose: Literal[
        "complete_initial_evidence",
        "resolve_validation_issue",
        "compare_route_alternatives",
        "verify_fixed_commitment",
        "refresh_stale_hotel_offer",
        "expand_candidate_gap",
    ]
    blocking: bool
    based_on_issue_keys: tuple[Identifier, ...] = Field(
        default=(),
        description=(
            "Required current readiness b keys only for resolve_validation_issue; "
            "empty for other purposes."
        ),
    )
    arguments: ModelCapabilityArgs


class _ModelRelativePlacement(V4ContractModel):
    placement: Literal["before", "after", "at_end"]
    relative_to_key: RepairObjectKey | None = None

    @model_validator(mode="after")
    def relative_key_matches_placement(self) -> _ModelRelativePlacement:
        if self.placement == "at_end":
            if self.relative_to_key is not None:
                raise ValueError("at_end forbids relative_to_key")
        elif self.relative_to_key is None:
            raise ValueError("before/after placement requires relative_to_key")
        return self


class ModelMoveRepairChoice(_ModelRelativePlacement):
    operation: Literal["move"] = "move"
    target_key: RepairObjectKey
    destination_date: date


class ModelReorderRepairChoice(_ModelRelativePlacement):
    operation: Literal["reorder"] = "reorder"
    target_key: RepairObjectKey


class ModelReplaceRepairChoice(V4ContractModel):
    operation: Literal["replace"] = "replace"
    target_key: RepairObjectKey
    replacement_key: CandidateKey


class ModelOmitSoftRepairChoice(V4ContractModel):
    """Omit an optional candidate; the legacy operation name remains wire-compatible."""

    operation: Literal["omit_soft"] = "omit_soft"
    target_key: CandidateKey = Field(
        description="待移除的 soft/filler/neutral 候选完整 c 键；不能删除 strong/immutable。"
    )


class ModelChangeWindowRepairChoice(V4ContractModel):
    operation: Literal["change_window"] = "change_window"
    target_key: RepairObjectKey
    preferred_window: ExpectedWindow


class ModelChangeTransportRepairChoice(V4ContractModel):
    operation: Literal["change_transport"] = "change_transport"
    service_date: date
    from_key: RepairObjectKey
    to_key: RepairObjectKey
    transport_preferences: tuple[Literal["public_transit", "taxi", "walking", "driving"], ...] = (
        Field(min_length=1)
    )


class ModelChangeHotelRepairChoice(V4ContractModel):
    operation: Literal["change_hotel"] = "change_hotel"
    hotel_offer_key: HotelKey


class ModelRequestEvidenceRepairChoice(V4ContractModel):
    operation: Literal["request_evidence"] = "request_evidence"
    requests: tuple[ModelCapabilityRequest, ...] = Field(min_length=1, max_length=4)


class ModelAskUserRepairChoice(V4ContractModel):
    operation: Literal["ask_user"] = "ask_user"


ModelRepairChoice: TypeAlias = Annotated[
    ModelMoveRepairChoice
    | ModelReorderRepairChoice
    | ModelReplaceRepairChoice
    | ModelOmitSoftRepairChoice
    | ModelChangeWindowRepairChoice
    | ModelChangeTransportRepairChoice
    | ModelChangeHotelRepairChoice
    | ModelRequestEvidenceRepairChoice
    | ModelAskUserRepairChoice,
    Field(discriminator="operation"),
]


class ModelRepairIntent(V4ContractModel):
    """Narrow semantic repair; all formal Patch metadata remains server-owned."""

    issue_keys: tuple[ValidationIssueKey, ...] = Field(min_length=1, max_length=4)
    choice: ModelRepairChoice
    reason_summary: DisplayText

    @model_validator(mode="after")
    def issue_keys_are_unique(self) -> ModelRepairIntent:
        if len(set(self.issue_keys)) != len(self.issue_keys):
            raise ValueError("repair issue_keys must be unique")
        return self


ModelPatchRepairChoice: TypeAlias = Annotated[
    ModelMoveRepairChoice
    | ModelReorderRepairChoice
    | ModelReplaceRepairChoice
    | ModelOmitSoftRepairChoice
    | ModelChangeWindowRepairChoice
    | ModelChangeTransportRepairChoice
    | ModelChangeHotelRepairChoice,
    Field(discriminator="operation"),
]


class ModelPatchRepairIntent(V4ContractModel):
    """Schema used only when Validator authorizes a semantic draft Patch."""

    issue_keys: tuple[ValidationIssueKey, ...] = Field(min_length=1, max_length=4)
    choice: ModelPatchRepairChoice
    reason_summary: DisplayText

    @model_validator(mode="after")
    def issue_keys_are_unique(self) -> ModelPatchRepairIntent:
        if len(set(self.issue_keys)) != len(self.issue_keys):
            raise ValueError("repair issue_keys must be unique")
        return self


class ModelPlanChangePatchIntent(V4ContractModel):
    """Published-plan edit choices; all authority and version fields are server-owned."""

    choices: tuple[ModelPatchRepairChoice, ...] = Field(min_length=1, max_length=5)
    reason_summary: DisplayText

    @model_validator(mode="after")
    def choices_are_not_exact_duplicates(self) -> ModelPlanChangePatchIntent:
        fingerprints = [choice.model_dump_json() for choice in self.choices]
        if len(set(fingerprints)) != len(fingerprints):
            raise ValueError("published-plan edit choices must be distinct")
        return self


class ModelEvidenceRepairIntent(V4ContractModel):
    """Schema used only when missing evidence is the current blocking cause."""

    issue_keys: tuple[ValidationIssueKey, ...] = Field(min_length=1, max_length=4)
    choice: ModelRequestEvidenceRepairChoice
    reason_summary: DisplayText

    @model_validator(mode="after")
    def issue_keys_are_unique(self) -> ModelEvidenceRepairIntent:
        if len(set(self.issue_keys)) != len(self.issue_keys):
            raise ValueError("repair issue_keys must be unique")
        return self


class ModelDecisionBase(V4ContractModel):
    current_goal: DisplayText
    reason_summary: DisplayText
    remaining_blockers: tuple[Identifier, ...] = Field(min_length=1)


class ModelStrategyDecision(ModelDecisionBase):
    action: Literal["build_or_update_strategy"]
    mode: Literal["initialize", "replace"]
    base_strategy_revision: int | None = Field(default=None, ge=1, strict=True)
    strategy: ModelStrategy


class ModelEvidenceDecision(ModelDecisionBase):
    action: Literal["request_evidence"]
    requests: tuple[ModelCapabilityRequest, ...] = Field(
        min_length=1,
        max_length=4,
        description="每批 1–4 项独立请求，其中 hotel_search 和 hotel_offer_refresh 合计最多一项。",
    )
    resume_goal: DisplayText


class ModelDraftDecision(ModelDecisionBase):
    action: Literal["materialize_draft"]
    draft: ModelWorkingDraft


class ModelAskDecision(ModelDecisionBase):
    action: Literal["ask_user"]
    issue_keys: tuple[Identifier, ...] = Field(min_length=1, max_length=4)


class ModelPlannerDecision(
    RootModel[
        Annotated[
            ModelStrategyDecision | ModelEvidenceDecision | ModelDraftDecision | ModelAskDecision,
            Field(discriminator="action"),
        ]
    ]
):
    """V4-04 enables only pre-materialization actions from the six-action public union."""


class ModelNonIssueCapabilityRequest(ModelCapabilityRequest):
    purpose: Literal[
        "complete_initial_evidence",
        "compare_route_alternatives",
        "verify_fixed_commitment",
        "refresh_stale_hotel_offer",
        "expand_candidate_gap",
    ] = Field(description="当前没有 readiness issue，不能选择 resolve_validation_issue。")
    based_on_issue_keys: tuple[Identifier, ...] = Field(default=(), max_length=0)


class ModelNonIssueEvidenceDecision(ModelEvidenceDecision):
    requests: tuple[ModelNonIssueCapabilityRequest, ...] = Field(
        min_length=1,
        max_length=4,
        description="每批 1–4 项独立请求，其中 hotel_search 和 hotel_offer_refresh 合计最多一项。",
    )


class ModelPlannerDecisionAfterStrategyWithoutIssues(
    RootModel[
        Annotated[
            ModelNonIssueEvidenceDecision | ModelDraftDecision,
            Field(discriminator="action"),
        ]
    ]
):
    """A stable policy can only gather evidence or propose the actual draft."""


class ModelPlannerDecisionAfterStrategyWithIssues(
    RootModel[
        Annotated[
            ModelEvidenceDecision | ModelDraftDecision | ModelAskDecision,
            Field(discriminator="action"),
        ]
    ]
):
    """Readiness issues may authorize ask, but never a cosmetic policy loop."""
