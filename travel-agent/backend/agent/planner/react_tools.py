"""Planner tool bindings; all semantic changes originate in native model calls."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextlib import suppress
from datetime import UTC, date, datetime, time, timedelta
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator

from backend.agent.model_gateway import ModelToolDefinition
from backend.agent.planner.decision_contracts import (
    CandidateKey,
    HotelKey,
    ModelAskDecision,
    ModelHotelArgs,
    ModelHotelRefreshArgs,
    ModelHoursArgs,
    ModelPlaceArgs,
    ModelPlanChangePatchIntent,
    ModelPlanDay,
    ModelPlanIntent,
    ModelPlannerDecision,
    ModelPlanStop,
    ModelRouteArgs,
    ModelTicketArgs,
    ModelWeatherArgs,
)
from backend.agent.planner.dependencies import (
    prepare_workspace_for_evidence_refresh,
    rebind_semantic_artifacts_after_evidence,
)
from backend.agent.planner.dining_context import dining_place_blocked, mark_dining_admitted
from backend.agent.planner.evidence import (
    PlannerEvidenceBackend,
    _category,
    _place_evidence,
    _real_category,
)
from backend.agent.planner.graph import _guard_local_day_changes
from backend.agent.planner.guards import guard_ask_user, validate_working_draft
from backend.agent.planner.materializer import PlannerDraftMaterializer
from backend.agent.planner.observation_router import EvidenceUpdate, observe_capability_results
from backend.agent.planner.plan_intent_compiler import (
    compile_default_strategy_decision,
    compile_plan_intent_decision,
)
from backend.agent.planner.proposals import PlannerReferenceCatalog, resolve_model_decision
from backend.agent.planner.react_context import draft_model_view
from backend.agent.planner.react_review import evidence_digest, has_current_review
from backend.agent.planner.react_runtime import (
    AgentSession,
    AgentTool,
    ReActRuntime,
    ToolOutcome,
    react_memory,
)
from backend.agent.planner.repair_compiler import compile_model_plan_change_intent
from backend.agent.planner.spatial import build_spatial_observation
from backend.agent.planner.validator import PlannerDraftValidator
from backend.agent.planner.workspace import PlannerGuardError, advance, task_book_references
from backend.contracts.enums import PlaceCategory, ProviderCode
from backend.contracts.v4.base import V4ContractModel
from backend.contracts.v4.enums import CandidateEntityKind, PlannerCapability, PlannerStatus
from backend.contracts.v4.planner_decision import (
    AskUserPayload,
    BuildOrUpdateStrategyPayload,
    MaterializeDraftPayload,
    RequestEvidencePayload,
    ReviseDraftPayload,
)
from backend.contracts.v4.planner_draft import WorkingItineraryDraft, planning_projection_digest
from backend.contracts.v4.planner_evidence import (
    PlannerCapabilityObservation,
    PlannerVisitDurationEstimate,
)
from backend.contracts.v4.planner_observations import (
    OpeningHoursArguments,
    SpatialRouteEdge,
    TicketAvailabilityArguments,
)
from backend.contracts.v4.planner_patch import atomic_apply_itinerary_patch
from backend.contracts.v4.planner_react import MAX_EFFECTIVE_REVISIONS, BoundReview, ReviewVerdict
from backend.contracts.v4.planner_workspace import PlannerWorkspaceState
from backend.persistence.outbox_repository import canonical_json_hash
from backend.planning.dining_geometry import corridor_polygon
from backend.providers.contracts import (
    KeywordPlaceSearchRequest,
    NearbyPlaceSearchRequest,
    PolygonPlaceSearchRequest,
)
from backend.providers.request_budget import RequestBudgetExceeded


class EmptyArgs(V4ContractModel):
    pass


class SearchArgs(V4ContractModel):
    keywords: str = Field(min_length=1, max_length=100)
    category: Literal["attraction", "restaurant"]
    next_candidate_key: str | None = None
    nearby_candidate_key: str | None = None
    radius_m: int = Field(default=1500, ge=100, le=5000)


class CorridorSearchArgs(V4ContractModel):
    before_candidate_key: CandidateKey
    after_candidate_key: CandidateKey
    category: Literal["attraction", "restaurant"]
    keywords: str = Field(min_length=1, max_length=100)


class AgentPlanStop(ModelPlanStop):
    candidate_key: CandidateKey = Field(
        description="当前 candidates 中的 c 键，每个候选全程只出现一次；"
        "同一景区及内部子地点不重复安排。"
    )
    part_of_day: Literal["morning", "midday", "afternoon", "evening", "anytime"] = Field(
        default="anytime",
        description="按餐次表达游览位置：morning=午餐前，afternoon=午餐后至晚餐前，"
        "evening=晚餐后，anytime=灵活；midday用于午餐或跨午餐游览。"
        "程序按这个餐次分组生成顺序，stops 只决定同一组内的先后。",
    )
    meal_slot: Literal["lunch", "dinner"] | None = Field(
        default=None, description="餐厅必填 lunch 或 dinner，每天各至多一次；景点留空。"
    )
    duration_preference: Literal["short", "normal", "extended"] | None = Field(
        default=None, description="在景点合理游览时长范围内选择短游、正常或深游；不包含交通和用餐。"
    )
    onsite_lunch: bool = Field(
        default=False,
        description="建议最长游览至少240分钟的大型景区可安排园内午餐暂停，"
        "此时当天不再另排午餐餐厅；不代表已核实食品准入或再入园。",
    )


class AgentAskArgs(V4ContractModel):
    candidate_key: CandidateKey | None = Field(
        default=None, description="需要用户取舍的必去或必吃候选；想去、顺路去自行取舍。"
    )
    issue_keys: tuple[str, ...] = Field(
        default=(),
        max_length=4,
        description="也可选择程序报告的用户权限问题，与 candidate_key 二选一。",
    )
    reason: str = Field(
        min_length=1,
        max_length=160,
        description="简短说明已知不合适之处；估算距离不能称为实际通勤。",
    )

    @model_validator(mode="after")
    def one_subject(self) -> AgentAskArgs:
        if bool(self.candidate_key) == bool(self.issue_keys):
            raise ValueError("choose candidate_key or issue_keys")
        return self


class AgentPlanDay(ModelPlanDay):
    start_time: time | None = Field(
        default=None, description="本日出发软目标，如08:00；可按游览和交通适度调整。"
    )
    end_time: time | None = Field(
        default=None,
        description="本日结束软目标，如21:00；实际时间由程序计算，允许适度提前或延后。",
    )
    stops: tuple[AgentPlanStop, ...] = Field(
        default=(),
        max_length=8,
        description=(
            "选择地点及餐次时段；程序按午餐前→午餐→两餐间→晚餐→夜游编译，"
            "同一时段内保留这里的排列顺序。"
            "每个 candidate_key 全程只出现一次。"
        ),
    )
    transport_preferences: tuple[Literal["public_transit", "taxi", "walking", "driving"], ...] = (
        Field(
            default=(),
            max_length=4,
            description=(
                "按优先顺序填写本日交通方式：先查并采用首个可用方式，失败才查后备方式。"
                "不会因公交耗时长而自动改成打车；需要省时须自行调整顺序并重新试算。"
                "省略时沿用任务书偏好。"
            ),
        )
    )


class AgentVisitDuration(V4ContractModel):
    candidate_key: CandidateKey
    minimum_minutes: int = Field(ge=15, le=600, strict=True)
    maximum_minutes: int = Field(ge=15, le=600, strict=True)

    @model_validator(mode="after")
    def ordered_range(self) -> AgentVisitDuration:
        if self.minimum_minutes > self.maximum_minutes:
            raise ValueError("minimum_minutes must not exceed maximum_minutes")
        return self


class AgentCapacityTradeoff(V4ContractModel):
    candidate_key: CandidateKey = Field(
        description="本次不安排的必去/想去景点或必吃/想吃餐厅，保留原偏好并说明取舍原因。"
    )
    reason: str = Field(min_length=1, max_length=160, description="给用户的具体取舍原因。")

    @model_validator(mode="after")
    def meaningful_reason(self) -> AgentCapacityTradeoff:
        if not self.reason.strip():
            raise ValueError("capacity tradeoff requires a nonblank reason")
        return self


class AgentPlanIntent(ModelPlanIntent):
    days: tuple[AgentPlanDay, ...] = Field(
        min_length=1,
        max_length=5,
        description="完整覆盖本次旅行的各日安排；按日期升序，每天选择各餐次时段的地点。",
    )
    selected_hotel_key: HotelKey | None = Field(
        default=None,
        description="从当前 hotels 字典选择一个酒店商品 h 键；"
        "需要住宿且尚未查询时先获取商品；已查询无可核实商品、已有预订或不需住宿时留空。",
    )
    visit_duration_estimates: tuple[AgentVisitDuration, ...] = Field(
        default=(),
        max_length=20,
        description=(
            "已选景点的 typical_duration_minutes 和 suggested_visit_duration 均为空时，"
            "须在本次写草稿时给出基本到深游的分钟范围；已有合理范围的可复用。"
            "仅模型建议，不是营业事实，不包含交通、排队或用餐；不能为填空档任意放大。"
            "同一候选只填一次；餐厅不填。修改范围后会重新计算并须重新评审。"
        ),
    )
    capacity_tradeoffs: tuple[AgentCapacityTradeoff, ...] = Field(
        default=(),
        max_length=20,
        description="必去景点容纳不下，或必吃餐厅绕路、缺少配套游览等不利于整体安排时，逐项说明未安排原因；"
        "无需先询问，不改变用户原意愿。已明确回答仍要去的不能省略。不发生取舍时留空。",
    )


class HotelLocationArgs(V4ContractModel):
    keywords: str = Field(min_length=1, max_length=100)


class InsertPlanStop(V4ContractModel):
    action: Literal["insert_stop"]
    day_index: int = Field(ge=1, le=5)
    after_candidate_key: CandidateKey | None = Field(
        default=None,
        description="插入到本日这个已排候选之后；null 表示组内首站。"
        "跨餐次以 stop.part_of_day 为准，同一餐次内保留此先后关系。",
    )
    stop: AgentPlanStop


class RemovePlanStop(V4ContractModel):
    action: Literal["remove_stop"]
    candidate_key: CandidateKey = Field(description="移除当前已排的一站；移动地点时先移除再插入。")


class UpdatePlanStop(V4ContractModel):
    action: Literal["update_stop"]
    candidate_key: CandidateKey = Field(
        description="仅修改当前已排的一站；尚未安排的候选用 insert_stop。"
    )
    duration_preference: Literal["short", "normal", "extended"] | None = None
    onsite_lunch: bool | None = None
    meal_slot: Literal["lunch", "dinner"] | None = None
    part_of_day: Literal["morning", "midday", "afternoon", "evening", "anytime"] | None = None


class ChangeDayTransport(V4ContractModel):
    action: Literal["change_transport"]
    day_index: int = Field(ge=1, le=5)
    transport_preferences: tuple[Literal["public_transit", "taxi", "walking", "driving"], ...] = (
        Field(min_length=1, max_length=4)
    )


class ChangeDayTiming(V4ContractModel):
    action: Literal["change_day_timing"]
    day_index: int = Field(ge=1, le=5)
    start_time: time
    end_time: time

    @model_validator(mode="after")
    def ordered_times(self) -> ChangeDayTiming:
        if self.start_time >= self.end_time:
            raise ValueError("start_time must precede end_time")
        return self


class AgentPlanEdits(V4ContractModel):
    edits: tuple[
        InsertPlanStop | RemovePlanStop | UpdatePlanStop | ChangeDayTransport | ChangeDayTiming, ...
    ] = Field(
        default=(),
        max_length=20,
        description="按顺序应用明确改动；移动一站用先移除再插入，同批保存为一次修订。未指定的安排保留。",
    )
    selected_hotel_key: HotelKey | None = Field(
        default=None,
        description="需要补选或更换酒店时填当前 hotels 的 h 键；留空保留原选择。"
        "仅选酒店时 edits 留空。",
    )
    reason: str = Field(min_length=1, max_length=300)
    visit_duration_estimates: tuple[AgentVisitDuration, ...] = ()
    capacity_tradeoffs: tuple[AgentCapacityTradeoff, ...] = Field(
        default=(),
        max_length=20,
        description="本次取舍必去景点或必吃餐厅须说明具体原因；已有取舍记录自动保留。",
    )

    @model_validator(mode="after")
    def has_edit(self) -> AgentPlanEdits:
        if not self.edits and self.selected_hotel_key is None:
            raise ValueError("provide edits or selected_hotel_key")
        return self


class AgentWeatherArgs(ModelWeatherArgs):
    capability: Literal["weather_forecast"] = "weather_forecast"
    destination_ref: Literal["destination"] = Field(
        default="destination", description="本次已确认任务书的目的地引用，固定使用 destination"
    )


class AgentHoursArgs(ModelHoursArgs):
    capability: Literal["opening_hours"] = "opening_hours"
    candidate_keys: tuple[CandidateKey, ...] = Field(
        min_length=1,
        max_length=20,
        description="从当前 candidates 字典选择 c1、c2 等短键；不能使用 UUID、POI ID 或地点名",
    )


class AgentHotelArgs(V4ContractModel):
    """Agent chooses the search; confirmed trip requirements are supplied by code.

    Optional legacy fields still parse pending native-tool receipts on recovery.
    They are omitted from the new schema dialogue's advertised parameter set.
    """

    capability: Literal["hotel_search"] = "hotel_search"
    party_size_ref: Literal["party"] = "party"
    search_keyword: str | None = Field(default=None, min_length=1, max_length=100)
    activity_cluster_keys: tuple[str, ...] = Field(min_length=1)
    check_in_date: date | None = None
    check_out_date: date | None = None
    lodging_preference_refs: tuple[str, ...] = ()
    budget_constraint_ref: str | None = None
    facility_constraint_refs: tuple[str, ...] = ()


class AgentHotelRefreshArgs(ModelHotelRefreshArgs):
    capability: Literal["hotel_offer_refresh"] = "hotel_offer_refresh"
    offer_key: HotelKey = Field(
        description="当前 hotels 字典中的 h 键；不使用 Provider 商品 ID。",
    )


class AgentTicketArgs(ModelTicketArgs):
    capability: Literal["ticket_availability"] = "ticket_availability"
    candidate_keys: tuple[CandidateKey, ...] = Field(
        min_length=1,
        max_length=20,
        description="从当前 candidates 字典选择 c1、c2 等短键，不使用 UUID 或 POI ID",
    )
    party_size_ref: Literal["party"] = "party"
    service_dates: tuple[date, ...] = Field(
        min_length=1,
        max_length=5,
        description=(
            "只传这些景点实际计划游览的日期。此工具会查询每个候选与每个日期的所有组合；"
            "不同日期的景点请拆成独立工具调用并行提出，不要统一传全部旅行日期。"
        ),
    )


class PlannerToolBindings:
    def __init__(
        self,
        evidence: PlannerEvidenceBackend,
        materializer: PlannerDraftMaterializer,
        validator: PlannerDraftValidator,
    ) -> None:
        self.evidence = evidence
        self.materializer = materializer
        self.validator = validator
        self.runtime: ReActRuntime | None = None

    def tools(self) -> tuple[AgentTool, ...]:
        def tool(
            name: str,
            description: str,
            model: type[BaseModel],
            handler: Callable[[dict[str, Any], AgentSession], Awaitable[ToolOutcome]],
            *,
            read_only: bool = True,
            reviewer: bool = False,
            query_group: str | None = None,
        ) -> AgentTool:
            return AgentTool(
                ModelToolDefinition(
                    name=name, description=description, parameters=model.model_json_schema()
                ),
                handler,
                read_only,
                reviewer,
                query_group,
            )

        result = [
            tool(
                "search_along_route",
                "按前后两候选坐标搜索沿途矩形内的景点或餐厅；程序计算范围，"
                "通过本地高德 MCP 查询并核验入池，不自动插入行程。不能代表真实交通路线。",
                CorridorSearchArgs,
                self.search_corridor,
                reviewer=True,
            ),
            tool(
                "search_hotel_locations",
                "通过高德 MCP 查酒店分店位置；不代表指定日期有房或真实房价，商品须另查 FlyAI",
                HotelLocationArgs,
                self.search_hotel_locations,
                reviewer=True,
            ),
            tool(
                "patch_plan",
                "根据本次用户授权局部修改已有方案",
                ModelPlanChangePatchIntent,
                self.patch,
                read_only=False,
            ),
            tool(
                "search_places",
                "按关键词或候选地点周边补搜真实景点或餐厅并核验加入候选池",
                SearchArgs,
                self.search,
                reviewer=True,
            ),
            tool(
                "write_plan",
                "保存完整草稿或替换当前草稿，包含各日游览、午晚餐和所需住宿。"
                "成功返回 ok=true 后查看 current_draft，无需重写确认；局部修改限于授权日期。",
                AgentPlanIntent,
                self.write,
                read_only=False,
            ),
            tool(
                "edit_plan",
                "按明确差异修订已保存草稿：增加/移除地点、调整游览深度或园内午餐、改变本日交通。"
                "没有指定的内容保留；只说修改理由不会改变安排，必须列出实际 edits。"
                "一次保存计一轮修订。",
                AgentPlanEdits,
                self.edit,
                read_only=False,
            ),
            tool(
                "check_plan",
                "补查当前草稿所需营业与相邻路线，计算时间轴、费用并校验；不替换地点或交通方式。"
                "返回实际计算和问题；仅送审时可直接使用评审工具，无需重复试算。",
                EmptyArgs,
                self.check,
                read_only=False,
                reviewer=True,
            ),
            tool(
                "review_plan",
                "将当前草稿送交独立评审；缺少当前版本计算时先自动试算，再由评审员核验，不修改草稿",
                EmptyArgs,
                self.review,
                read_only=False,
            ),
            tool(
                "submit_review",
                "提交当前版本的独立评审结论与具体问题；"
                "拒绝必须有具体 error/blocking，不能仅列警告却拒绝",
                ReviewVerdict,
                self.submit_review,
                read_only=False,
                reviewer=True,
            ),
            tool(
                "finish_plan",
                "输出当前完整行程，程序计算时间和费用并附上检查问题。"
                "尽量先评审和修正；仍有问题或无法完成复核时也可交付，不必等待全部通过。",
                EmptyArgs,
                self.finish,
                read_only=False,
            ),
            tool(
                "ask_user",
                "必去或必吃确实不合适时，说明原因并询问仍要安排还是本次不去；也可请求裁决程序发现的硬冲突。回答前暂停，想去和顺路去无需询问。",
                AgentAskArgs,
                self.ask,
                read_only=False,
            ),
        ]
        for name, description, model in (
            ("lookup_hours", "核验候选在旅行日期的营业及最后入场信息", AgentHoursArgs),
            ("lookup_weather", "查询本次已确认旅行目的地的天气", AgentWeatherArgs),
            ("lookup_routes", "查询指定地点间真实路线时间与费用", ModelRouteArgs),
            ("lookup_place", "核验已有候选地点身份与详情", ModelPlaceArgs),
            (
                "lookup_tickets",
                "可选查询景点参考票价；不查询用户预约状态或预约余量，缺价不阻止排程",
                AgentTicketArgs,
            ),
            (
                "search_hotels",
                "按活动区域或关键词查询酒店商品；程序自动带入已确认的入住日期、预算及设施要求。"
                "可与营业等独立查询并行，同批只允许一个酒店商品查询",
                AgentHotelArgs,
            ),
            (
                "refresh_hotel",
                "核验指定酒店商品的本次入住日期价格。同批只允许一个酒店商品查询",
                AgentHotelRefreshArgs,
            ),
        ):

            async def query(
                args: dict[str, Any], session: AgentSession, model: type[BaseModel] = model
            ) -> ToolOutcome:
                return await self.query(model.model_validate(args), session)

            # Independent fact queries merge under the session lock; competing
            # replacements of the same hotel offer set cannot share a batch.
            result.append(
                tool(
                    name,
                    description,
                    model,
                    query,
                    query_group="hotel_offers"
                    if name in {"search_hotels", "refresh_hotel"}
                    else None,
                    reviewer=True,
                )
            )
        return tuple(result)

    async def search_hotel_locations(
        self, args: dict[str, Any], session: AgentSession
    ) -> ToolOutcome:
        query = HotelLocationArgs.model_validate(args)
        city = self.evidence.registry.provider_scope(
            session.context.book.destination_and_dates.destination_name, ProviderCode.AMAP
        )
        response = await self.evidence.providers.places.search_places(
            KeywordPlaceSearchRequest(
                city=city, query=query.keywords, category_hint=PlaceCategory.HOTEL
            )
        )
        return ToolOutcome(
            {
                "source": "amap_mcp",
                "query": query.keywords,
                "observed_at": response.fetched_at.isoformat(),
                "locations": [
                    item.model_dump(mode="json", exclude={"raw_payload"})
                    for item in response.items
                    if item.category is PlaceCategory.HOTEL
                ],
                "price_status": "unknown",
                "availability_status": "unknown",
                "next": "search_hotels; verify exact branch name, address and coordinates",
            }
        )

    async def query(self, args: BaseModel, session: AgentSession) -> ToolOutcome:
        w = session.workspace
        if isinstance(args, AgentHotelArgs):
            book = session.context.book
            references = task_book_references(book)
            args = ModelHotelArgs.model_validate(
                {
                    "capability": "hotel_search",
                    "party_size_ref": "party",
                    "check_in_date": book.destination_and_dates.start_date,
                    "check_out_date": book.destination_and_dates.end_date,
                    "lodging_preference_refs": [
                        k for k in references if k == "lodging" or k.startswith("area:")
                    ],
                    "budget_constraint_ref": "lodging_budget"
                    if "lodging_budget" in references
                    else None,
                    "facility_constraint_refs": [
                        k for k in references if k.startswith("facility:")
                    ],
                    **args.model_dump(exclude_unset=True),
                }
            )
        decision = resolve_model_decision(
            ModelPlannerDecision.model_validate(
                {
                    "action": "request_evidence",
                    "current_goal": "核实本次规划事实",
                    "reason_summary": "由 Agent 主动查询",
                    "remaining_blockers": ["evidence_pending"],
                    "resume_goal": "依据查询结果继续决策",
                    "requests": [
                        {
                            "local_key": str(uuid4()),
                            "purpose": "complete_initial_evidence",
                            "blocking": False,
                            "arguments": args.model_dump(mode="json"),
                        }
                    ],
                }
            ),
            w,
        )
        assert isinstance(decision.payload, RequestEvidencePayload)
        request = decision.payload.capability_requests[0]
        if isinstance(request.arguments, TicketAvailabilityArguments):
            requested_tickets = {
                (ref.canonical_entity_id, day)
                for ref in request.arguments.candidate_refs
                for day in request.arguments.service_dates
            }
            cached_tickets = self.evidence.fresh_ticket_facts(w)
            if requested_tickets <= cached_tickets.keys():
                self.evidence._validate_request(
                    request, w, session.context.book, cached_observation=True
                )
                return ToolOutcome(
                    {
                        "cached": True,
                        "tickets": [
                            cached_tickets[key].model_dump(mode="json")
                            for key in sorted(requested_tickets)
                        ],
                        "status": "partial",
                        "note": "复用近期参考票价；仅作为费用资料，不核验用户预约或余量。",
                    }
                )
        if isinstance(request.arguments, OpeningHoursArguments):
            requested = {r.canonical_entity_id for r in request.arguments.candidate_refs}
            dates = set(request.arguments.service_dates)
            cached = tuple(
                h
                for h in w.hours_evidence
                if h.canonical_entity_id in requested
                and h.expires_at > self.evidence.clock()
                and dates <= {day.service_date for day in h.days}
            )
            if len(cached) == len(requested):
                # Cross-agent reuse is an observation, not another I/O attempt.
                # Validate identity, authority and dates even on the cache path.
                self.evidence._validate_request(
                    request, w, session.context.book, cached_observation=True
                )
                return ToolOutcome(
                    {
                        "cached": True,
                        "hours": [h.model_dump(mode="json") for h in cached],
                        "status": "partial"
                        if any(
                            day.status in {"unknown", "conflict"}
                            for h in cached
                            for day in h.days
                            if day.service_date in dates
                        )
                        else "complete",
                        "note": "复用工作区已有营业证据；未知仍为未知，未发起新的外部查询。",
                    }
                )
        self.evidence.validate_batch((request,), w, session.context.book)
        update = self.evidence._cached_update(request, w)
        if update is None:
            update = await self.evidence._execute(
                request, w, session.context.book, session.context.cancellation
            )

        async def apply(current: PlannerWorkspaceState) -> PlannerWorkspaceState:
            following = await self.merge(current, update, session)
            return advance(following, decision_trace=(*following.decision_trace, decision))

        result: dict[str, Any] = {"observation": update.observation.model_dump(mode="json")}
        for field in ("places", "hours", "weather", "tickets", "hotel", "route_edges"):
            value = getattr(update, field)
            result[field] = (
                [v.model_dump(mode="json") for v in value]
                if isinstance(value, tuple)
                else value.model_dump(mode="json")
                if value
                else None
            )
        return ToolOutcome(result, apply)

    async def merge(
        self, current: PlannerWorkspaceState, update: EvidenceUpdate, session: AgentSession
    ) -> PlannerWorkspaceState:
        book = session.context.book
        now = self.evidence.clock()
        detached = prepare_workspace_for_evidence_refresh(current)
        following = observe_capability_results(detached, book, (update,), now)
        if following.spatial_observation is None:
            following = await build_spatial_observation(
                following,
                book,
                routes=self.evidence.providers.routes,
                city=self.evidence.registry.provider_scope(
                    book.destination_and_dates.destination_name, ProviderCode.AMAP
                ),
                cancellation=session.context.cancellation,
                now=now,
                query_routes=False,
                retained_route_edges=self._reusable_route_edges(current, following, session, now),
            )
        if current.working_itinerary is not None:
            following = rebind_semantic_artifacts_after_evidence(
                current, following, refresh_clusters=True
            )
        elif following.planning_strategy is None:
            decision = compile_default_strategy_decision(following, book)
            assert isinstance(decision.payload, BuildOrUpdateStrategyPayload)
            following = advance(following, planning_strategy=decision.payload.proposed_strategy)
        return following

    @staticmethod
    def _reusable_route_edges(
        before: PlannerWorkspaceState,
        after: PlannerWorkspaceState,
        session: AgentSession,
        now: datetime,
    ) -> tuple[SpatialRouteEdge, ...]:
        from backend.agent.planner.location_capabilities import endpoint_coordinates

        fresh = {
            fact.fact_reference_id
            for fact in after.verified_facts
            if fact.expires_at is not None and fact.expires_at > now
        }
        reusable = []
        for edge in before.route_evidence:
            if edge.origin.kind == "cluster" or edge.destination.kind == "cluster":
                continue
            if edge.status == "available" and (
                not edge.fact_reference_ids or not set(edge.fact_reference_ids) <= fresh
            ):
                continue
            try:
                if any(
                    endpoint_coordinates(endpoint, before, session.context.book)
                    != endpoint_coordinates(endpoint, after, session.context.book)
                    for endpoint in (edge.origin, edge.destination)
                ):
                    continue
            except (PlannerGuardError, KeyError, StopIteration):
                continue
            reusable.append(edge)
        return tuple(reusable)

    async def search_corridor(self, args: dict[str, Any], session: AgentSession) -> ToolOutcome:
        query = CorridorSearchArgs.model_validate(args)
        return await self.search(
            {
                "keywords": query.keywords,
                "category": query.category,
                "nearby_candidate_key": query.before_candidate_key,
                "next_candidate_key": query.after_candidate_key,
            },
            session,
        )

    async def search(self, args: dict[str, Any], session: AgentSession) -> ToolOutcome:
        query = SearchArgs.model_validate(args)
        book = session.context.book
        city = self.evidence.registry.provider_scope(
            book.destination_and_dates.destination_name, ProviderCode.AMAP
        )
        kind = CandidateEntityKind(query.category)
        if query.nearby_candidate_key:
            ref = PlannerReferenceCatalog(session.workspace).candidate(
                query.nearby_candidate_key, field="nearby_candidate_key"
            )
            place = next(
                (
                    p
                    for p in session.workspace.place_evidence
                    if p.canonical_entity_id == ref.canonical_entity_id
                ),
                None,
            )
            if place is None or place.coordinates is None:
                raise PlannerGuardError(
                    "search_requires_verified_endpoint_coordinates:先核验地点坐标或改用可定位的前后地点"
                )
            if query.next_candidate_key:
                other_ref = PlannerReferenceCatalog(session.workspace).candidate(
                    query.next_candidate_key, field="after_candidate_key"
                )
                other = next(
                    (
                        p
                        for p in session.workspace.place_evidence
                        if p.canonical_entity_id == other_ref.canonical_entity_id
                    ),
                    None,
                )
                if other is None or other.coordinates is None or place.coordinates is None:
                    raise PlannerGuardError("corridor_search_requires_endpoint_coordinates")
                response = await self.evidence.providers.places.search_polygon(
                    PolygonPlaceSearchRequest(
                        city=city,
                        polygon=corridor_polygon(place.coordinates, other.coordinates),
                        category_hint=PlaceCategory.ATTRACTION
                        if query.category == "attraction"
                        else PlaceCategory.RESTAURANT,
                        typecodes=("110000" if query.category == "attraction" else "050000",),
                        query=query.keywords,
                    )
                )
            else:
                response = await self.evidence.providers.places.search_nearby(
                    NearbyPlaceSearchRequest(
                        city=city,
                        query=query.keywords,
                        category_hint=_category(kind),
                        center=place.coordinates,
                        radius_m=query.radius_m,
                    )
                )
        else:
            response = await self.evidence.providers.places.search_places(
                KeywordPlaceSearchRequest(
                    city=city, query=query.keywords, category_hint=_category(kind)
                )
            )
        places = tuple(
            {
                _place_evidence(p, kind).canonical_entity_id: _place_evidence(p, kind)
                for p in response.items
                if p.provider_typecode
                and _real_category(p, kind)
                and p.provider_city_code
                and p.provider_city_code[: 2 if city.provider_city_code.endswith("0000") else 4]
                == city.provider_city_code[: 2 if city.provider_city_code.endswith("0000") else 4]
            }.values()
        )
        places = tuple(
            p
            for p in places
            if kind is not CandidateEntityKind.RESTAURANT or not dining_place_blocked(p, book)
        )

        async def apply(w: PlannerWorkspaceState) -> PlannerWorkspaceState:
            if not places:
                return w
            if kind is CandidateEntityKind.RESTAURANT:
                w = mark_dining_admitted(w, (p.canonical_entity_id for p in places))
            update = EvidenceUpdate(
                places=places,
                observation=PlannerCapabilityObservation(
                    observation_id=str(uuid4()),
                    scope=w.current_scope,
                    request_id=str(uuid4()),
                    capability=PlannerCapability.PLACE_FACTS,
                    status="complete",
                    fact_reference_ids=tuple(p.fact_reference_id for p in places),
                    reason_summary=f"MCP 关键词补搜：{query.keywords}；已核验实体、类型及城市。",
                    observed_at=datetime.now(UTC),
                ),
            )
            return await self.merge(w, update, session)

        return ToolOutcome(
            {
                "places": [p.model_dump(mode="json") for p in places],
                "query": query.model_dump(),
                "missing": not places,
            },
            apply,
        )

    async def edit(self, args: dict[str, Any], session: AgentSession) -> ToolOutcome:
        chosen = AgentPlanEdits.model_validate(args)
        current = draft_model_view(session.workspace)
        if current is None:
            raise PlannerGuardError("edit_plan_requires_saved_draft")
        hotel_only = bool(chosen.selected_hotel_key) and not (
            chosen.edits or chosen.visit_duration_estimates or chosen.capacity_tradeoffs
        )
        days = {}
        for day in current["days"]:
            stops = []
            if day["route_mode_selections"] and not hotel_only:
                raise PlannerGuardError("use_patch_plan_to_preserve_per_leg_transport")
            for stop in day["stops"]:
                if "candidate_key" not in stop:
                    continue  # Confirmed fixed commitments are recompiled from the book.
                window = stop["expected_window"]
                if (window["earliest"] or window["latest"]) and not hotel_only:
                    raise PlannerGuardError("use_patch_plan_to_preserve_exact_windows")
                stops.append(
                    {
                        "candidate_key": stop["candidate_key"],
                        "part_of_day": window["part_of_day"],
                        "meal_slot": stop["meal_slot"],
                        "duration_preference": stop["duration_preference"],
                        "onsite_lunch": stop["onsite_lunch"],
                    }
                )
            days[day["day_index"]] = {
                "day_index": day["day_index"],
                "theme": day["theme"],
                "stops": stops,
                "transport_preferences": day["transport_preferences"],
                "start_time": day.get("start_time"),
                "end_time": day.get("end_time"),
            }
        for edit in chosen.edits:
            if isinstance(edit, (InsertPlanStop, ChangeDayTransport, ChangeDayTiming)):
                if edit.day_index not in days:
                    raise PlannerGuardError("edit_day_outside_current_trip")
                day = days[edit.day_index]
                if isinstance(edit, ChangeDayTiming):
                    day.update(start_time=edit.start_time, end_time=edit.end_time)
                    continue
                if isinstance(edit, ChangeDayTransport):
                    day["transport_preferences"] = list(edit.transport_preferences)
                    continue
                existing = [
                    index
                    for index, value in days.items()
                    if any(s["candidate_key"] == edit.stop.candidate_key for s in value["stops"])
                ]
                if existing:
                    raise PlannerGuardError(
                        f"edit_candidate_already_scheduled:{edit.stop.candidate_key}:"
                        f"days={existing}:repair=移动地点须先 remove_stop 再 insert_stop；"
                        "只改餐次或时段用 update_stop，不重复插入"
                    )
                keys = [stop["candidate_key"] for stop in day["stops"]]
                if edit.after_candidate_key is not None and edit.after_candidate_key not in keys:
                    raise PlannerGuardError(
                        f"edit_anchor_must_be_in_target_day:{edit.after_candidate_key}:"
                        f"day={edit.day_index}:available_anchors={keys}:"
                        "repair=选择此前操作后仍在本日的锚点，或用 null 插入到第一站"
                    )
                index = keys.index(edit.after_candidate_key) + 1 if edit.after_candidate_key else 0
                day["stops"].insert(index, edit.stop.model_dump(mode="json"))
            else:
                matches = [
                    (day, stop)
                    for day in days.values()
                    for stop in day["stops"]
                    if stop["candidate_key"] == edit.candidate_key
                ]
                if len(matches) != 1:
                    raise PlannerGuardError(
                        f"edit_target_must_be_a_current_scheduled_stop:{edit.candidate_key}:"
                        "repair=仅修改或移除当前已排地点；新增候选用 insert_stop"
                    )
                day, stop = matches[0]
                if isinstance(edit, RemovePlanStop):
                    day["stops"].remove(stop)
                else:
                    values = edit.model_dump(exclude={"action", "candidate_key"}, exclude_none=True)
                    if not values:
                        raise PlannerGuardError("edit_update_requires_actual_fields")
                    stop.update(values)
        return await self.write(
            {
                "days": list(days.values()),
                "selected_hotel_key": chosen.selected_hotel_key or current["selected_hotel_key"],
                "overall_rationale": chosen.reason,
                "visit_duration_estimates": [
                    e.model_dump(mode="json") for e in chosen.visit_duration_estimates
                ],
                "capacity_tradeoffs": [
                    e.model_dump(mode="json") for e in chosen.capacity_tradeoffs
                ],
            },
            session,
            preserve_day_themes=True,
            preserve_day_layout=hotel_only,
        )

    async def write(
        self,
        args: dict[str, Any],
        session: AgentSession,
        *,
        preserve_day_themes: bool = False,
        preserve_day_layout: bool = False,
    ) -> ToolOutcome:
        if not session.context.allow_semantic_repair:
            raise PlannerGuardError("semantic_change_not_authorized")
        chosen = _compile_meal_relative_order(
            AgentPlanIntent.model_validate(args),
            session.workspace,
            preserve_unchanged_days=preserve_day_themes
            or bool(
                session.workspace.plan_change_request
                and session.workspace.plan_change_request.requested_scope == "local_replan"
            ),
        )
        intent = ModelPlanIntent.model_validate(
            chosen.model_dump(
                exclude={
                    "days": {"__all__": {"transport_preferences", "start_time", "end_time"}},
                    "visit_duration_estimates": True,
                    "capacity_tradeoffs": True,
                }
            )
        )
        old = session.workspace.working_itinerary
        tradeoffs = {item.candidate_key: item.reason for item in chosen.capacity_tradeoffs}
        if len(tradeoffs) != len(chosen.capacity_tradeoffs):
            raise PlannerGuardError("capacity_tradeoff_candidate_must_be_unique")
        if old is not None and session.state.effective_revisions >= MAX_EFFECTIVE_REVISIONS:
            raise PlannerGuardError("planner_revision_budget_exhausted")

        def apply(w: PlannerWorkspaceState) -> PlannerWorkspaceState:
            old = w.working_itinerary
            prepared = prepare_workspace_for_evidence_refresh(w)
            prepared = self._with_visit_estimates(prepared, chosen, session)
            duration_changed = prepared.visit_duration_estimates != w.visit_duration_estimates
            # Replacing the semantic draft does not change the existing strategy.
            prepared = prepared.model_copy(update={"planning_strategy": w.planning_strategy})
            decision = compile_plan_intent_decision(
                intent,
                prepared,
                session.context.book,
                base_draft=old,
                preserve_agent_choices=True,
                capacity_tradeoffs=tradeoffs,
            )
            payload = decision.payload
            assert isinstance(payload, MaterializeDraftPayload)
            draft = payload.proposed_working_draft
            modes = {day.day_index: day.transport_preferences for day in chosen.days}
            day_choices = {day.day_index: day for day in chosen.days}
            old_days = {day.service_date: day for day in old.days} if old is not None else {}
            following_days = []
            for index, day in enumerate(draft.days, 1):
                previous = old_days.get(day.service_date)
                if (
                    preserve_day_themes
                    and previous is not None
                    and (
                        [item.object_ref for item in previous.ordered_items]
                        == [item.object_ref for item in day.ordered_items]
                    )
                ):
                    day = day.model_copy(update={"day_theme": previous.day_theme})
                if modes.get(index):
                    day = day.model_copy(update={"transport_preferences": modes[index]})
                choice = day_choices[index]
                day = day.model_copy(
                    update={"start_time": choice.start_time, "end_time": choice.end_time}
                )
                if day.start_time and day.end_time and day.start_time >= day.end_time:
                    raise PlannerGuardError("day_start_time_must_precede_end_time")
                following_days.append(day)
            draft = draft.model_copy(
                update={
                    "days": old.days
                    if preserve_day_layout and old is not None
                    else tuple(following_days)
                }
            )
            draft = draft.model_copy(update={"content_digest": planning_projection_digest(draft)})
            decision = decision.model_copy(
                update={"payload": payload.model_copy(update={"proposed_working_draft": draft})}
            )
            dates = w.accepted_local_change_dates or session.context.local_change_dates
            if w.plan_change_request and w.plan_change_request.requested_scope == "local_replan":
                if not dates:
                    raise PlannerGuardError("use_patch_plan_to_establish_authorized_dates")
                if old is not None:
                    _guard_local_day_changes(old, draft, dates)
                    if old.lodging_baseline != draft.lodging_baseline:
                        raise PlannerGuardError("use_authorized_patch_to_change_hotel")
            if old is not None and not duration_changed:
                _require_effective_change(old, draft)
            validate_working_draft(draft, prepared, session.context.book, datetime.now(UTC))
            return advance(
                prepared,
                working_itinerary=draft,
                selected_hotel=payload.selected_hotel,
                hotel_recommendations=None,
                unresolved_decisions=(),
                status=PlannerStatus.DRAFT_READY,
                decision_trace=(*w.decision_trace, decision),
                react_state=react_memory(w).model_copy(
                    update={
                        "review": None,
                        "effective_revisions": react_memory(w).effective_revisions
                        + int(old is not None),
                    }
                ),
            )

        return ToolOutcome({"ok": True, "draft_saved": True}, apply)

    @staticmethod
    def _with_visit_estimates(
        w: PlannerWorkspaceState, chosen: AgentPlanIntent, session: AgentSession
    ) -> PlannerWorkspaceState:
        catalog = PlannerReferenceCatalog(w)
        previous = {item.canonical_entity_id: item for item in w.visit_duration_estimates}
        selected = {
            stop.candidate_key: session.context.book.destination_and_dates.start_date
            + timedelta(days=day.day_index - 1)
            for day in chosen.days
            for stop in day.stops
        }
        seen = set()
        dates = w.accepted_local_change_dates or session.context.local_change_dates or frozenset()
        for estimate in chosen.visit_duration_estimates:
            key = estimate.candidate_key
            entry = catalog.candidates.get(key)
            if key in seen or key not in selected or entry is None:
                raise PlannerGuardError("visit_estimate_requires_unique_selected_candidate")
            seen.add(key)
            if entry.entity_kind is not CandidateEntityKind.ATTRACTION:
                raise PlannerGuardError("visit_estimate_requires_attraction")
            if entry.advisory_features.typical_duration_minutes is not None:
                raise PlannerGuardError("use_existing_candidate_duration_with_duration_preference")
            identity = entry.candidate_ref.canonical_entity_id
            old = previous.get(identity)
            if old and (old.minimum_minutes, old.maximum_minutes) == (
                estimate.minimum_minutes,
                estimate.maximum_minutes,
            ):
                continue
            if (
                w.plan_change_request
                and w.plan_change_request.requested_scope == "local_replan"
                and selected[key] not in dates
            ):
                raise PlannerGuardError("visit_estimate_outside_authorized_dates")
            previous[identity] = PlannerVisitDurationEstimate(
                canonical_entity_id=identity,
                minimum_minutes=estimate.minimum_minutes,
                maximum_minutes=estimate.maximum_minutes,
                source="llm_estimate",
                context_fingerprint=canonical_json_hash(
                    {
                        "task_book": session.context.book.model_dump(mode="json"),
                        "entity": identity,
                        "estimate": estimate.model_dump(),
                    }
                ),
            )
        return w.model_copy(update={"visit_duration_estimates": tuple(previous.values())})

    async def patch(self, args: dict[str, Any], session: AgentSession) -> ToolOutcome:
        if not session.context.allow_semantic_repair:
            raise PlannerGuardError("semantic_change_not_authorized")
        intent = ModelPlanChangePatchIntent.model_validate(args)

        def apply(w: PlannerWorkspaceState) -> PlannerWorkspaceState:
            if w.plan_change_request is None or w.working_itinerary is None:
                raise PlannerGuardError("planner_patch_requires_user_authority")
            if react_memory(w).effective_revisions >= MAX_EFFECTIVE_REVISIONS:
                raise PlannerGuardError("planner_revision_budget_exhausted")
            decision = compile_model_plan_change_intent(intent, w.plan_change_request, w)
            payload = decision.payload
            assert isinstance(payload, ReviseDraftPayload)
            draft = atomic_apply_itinerary_patch(
                w.working_itinerary,
                payload.itinerary_patch,
                candidate_pool=w.candidate_pool,
                allowed_authority_refs=frozenset(w.plan_change_request.semantic_operation_ids),
            )
            _require_effective_change(w.working_itinerary, draft)
            return advance(
                w,
                working_itinerary=draft,
                selected_hotel=payload.selected_hotel or w.selected_hotel,
                hotel_recommendations=payload.hotel_recommendations or w.hotel_recommendations,
                materialized_schedule=None,
                cost_draft=None,
                validation_report=None,
                validation_observation=None,
                status=PlannerStatus.PLANNING,
                decision_trace=(*w.decision_trace, decision),
                react_state=react_memory(w).model_copy(
                    update={
                        "review": None,
                        "effective_revisions": react_memory(w).effective_revisions + 1,
                    }
                ),
            )

        return ToolOutcome({"ok": True, "next": "check_plan"}, apply)

    async def checked(
        self, w: PlannerWorkspaceState, session: AgentSession
    ) -> PlannerWorkspaceState:
        if w.working_itinerary is None:
            raise PlannerGuardError("planner_draft_required")
        # Missing lookup budget is not permission to invent facts, nor to
        # discard a computable partial draft. The validator reports gaps.
        if not (w.react_state and w.react_state.route_refresh_only):
            with suppress(RequestBudgetExceeded):
                w = await self.evidence.ensure_selected_hours(
                    w, session.context.book, session.context.cancellation
                )
        with suppress(RequestBudgetExceeded):
            w = await self.evidence.ensure_selected_itinerary_routes(
                w, session.context.book, session.context.cancellation
            )
        materialized = await self.materializer.materialize(
            w,
            session.context.book,
            session.context.cancellation,
            input_state_version=session.context.input_state_version,
            allow_time_savings=False,
        )
        w = advance(
            w,
            materialized_schedule=materialized.schedule,
            cost_draft=materialized.cost,
            validation_observation=None,
            validation_report=None,
        )
        result = await self.validator.validate(
            w, session.context.book, session.context.cancellation
        )
        return advance(
            w, validation_observation=result.observation, validation_report=result.legacy_report
        )

    async def check(self, args: dict[str, Any], session: AgentSession) -> ToolOutcome:
        checked = session.workspace
        if checked.validation_observation is None:
            checked = await self.checked(checked, session)
        assert (
            checked.validation_observation and checked.materialized_schedule and checked.cost_draft
        )
        return ToolOutcome(
            {
                "validation": checked.validation_observation.model_dump(mode="json"),
                "schedule": checked.materialized_schedule.model_dump(mode="json"),
                "cost": checked.cost_draft.model_dump(mode="json"),
            },
            lambda current: checked.model_copy(update={"react_state": current.react_state}),
        )

    async def review(self, args: dict[str, Any], session: AgentSession) -> ToolOutcome:
        if session.workspace.validation_observation is None:
            await session.progress("runtime", "先核对这版安排的时间和费用，再交给评审检查。")
            # Calculation performs budgeted I/O and persists reservations. Do not
            # hold the session mutation lock while it is awaiting those writes.
            checked = await self.checked(session.workspace, session)
            await session.update(
                lambda w: checked.model_copy(update={"react_state": w.react_state})
            )
        if not has_current_review(session.workspace):
            assert self.runtime is not None
            await self.runtime.review(session)
        review = session.state.review
        if review is None:
            raise PlannerGuardError("reviewer_did_not_submit_verdict")
        return ToolOutcome({"review": review.model_dump(mode="json")})

    async def submit_review(self, args: dict[str, Any], session: AgentSession) -> ToolOutcome:
        verdict = ReviewVerdict.model_validate(args)
        w = session.workspace
        draft, validation = w.working_itinerary, w.validation_observation
        if draft is None or validation is None:
            raise PlannerGuardError("review_requires_current_calculation")
        if verdict.accepted and validation.result != "passed":
            raise PlannerGuardError("review_cannot_override_hard_validation")
        if not verdict.accepted and not any(
            issue.severity != "warning" for issue in verdict.issues
        ):
            # Check only new submissions; historical serialized verdicts remain
            # readable. The program never turns a rejected verdict into approval.
            raise PlannerGuardError("review_rejection_requires_actionable_error")
        review = BoundReview(
            draft_revision=draft.draft_revision,
            draft_digest=draft.content_digest,
            evidence_digest=evidence_digest(w),
            validation_fingerprint=validation.validation_fingerprint,
            verdict=verdict,
            reviewed_at=datetime.now(UTC),
        )
        return ToolOutcome(
            {"ok": True},
            lambda w: advance(
                w,
                react_state=react_memory(w).model_copy(
                    update={"review": review, "review_in_progress": False}
                ),
            ),
        )

    async def finish(self, args: dict[str, Any], session: AgentSession) -> ToolOutcome:
        from backend.agent.planner.result_delivery import prepare_result_delivery

        if session.workspace.working_itinerary is None:
            raise PlannerGuardError("planner_draft_required")
        from backend.agent.planner.hotel_status import require_hotel_action

        require_hotel_action(session.workspace, session.context.book)
        checked = await prepare_result_delivery(
            session.workspace,
            session.context.book,
            session.context.cancellation,
            materializer=self.materializer,
            validator=self.validator,
            input_state_version=session.context.input_state_version,
        )
        return ToolOutcome(
            {"ok": True, "result_ready": True},
            lambda w: checked.model_copy(update={"react_state": w.react_state}),
        )

    async def ask(self, args: dict[str, Any], session: AgentSession) -> ToolOutcome:
        from backend.agent.planner.candidate_tradeoffs import ask_required_candidate

        query = AgentAskArgs.model_validate(args)
        if query.candidate_key:
            ref = PlannerReferenceCatalog(session.workspace).candidate(
                query.candidate_key, field="candidate_key"
            )
            return ToolOutcome(
                {"awaiting_user": True}, lambda w: ask_required_candidate(w, ref, query.reason)
            )
        decision = resolve_model_decision(
            ModelPlannerDecision(
                ModelAskDecision(
                    action="ask_user",
                    current_goal="请求用户取舍",
                    reason_summary=query.reason,
                    remaining_blockers=query.issue_keys,
                    issue_keys=query.issue_keys,
                )
            ),
            session.workspace,
        )
        guard_ask_user(decision, session.workspace)
        payload = decision.payload
        assert isinstance(payload, AskUserPayload)
        return ToolOutcome(
            {"awaiting_user": True},
            lambda w: advance(
                w,
                active_interaction=payload.user_decision_request,
                decision_trace=(*w.decision_trace, decision),
                unresolved_decisions=payload.blocking_issue_ids,
                user_interrupt_count=w.user_interrupt_count + 1,
                status=PlannerStatus.AWAITING_USER,
            ),
        )


def _compile_meal_relative_order(
    chosen: AgentPlanIntent,
    workspace: PlannerWorkspaceState,
    *,
    preserve_unchanged_days: bool = False,
) -> AgentPlanIntent:
    """Compile the model's meal groups, preserving order within each group.

    Explicit meal-relative labels own the phase; the list owns within-phase
    order. Unlabelled visits retain their position relative to meal anchors.
    This neither picks places nor adjusts durations. Untouched checkpoint days
    retain their existing order, including when another day is locally edited.
    """
    old = draft_model_view(workspace)
    old_days = {d["day_index"]: d for d in old["days"]} if old else {}
    days = []
    for day in chosen.days:
        previous = old_days.get(day.day_index)
        signature = [
            (s.candidate_key, s.part_of_day, s.meal_slot, s.onsite_lunch) for s in day.stops
        ]
        if (
            preserve_unchanged_days
            and previous
            and signature
            == [
                (
                    s["candidate_key"],
                    s["expected_window"]["part_of_day"],
                    s["meal_slot"],
                    s["onsite_lunch"],
                )
                for s in previous["stops"]
                if "candidate_key" in s
            ]
        ):
            days.append(day)
            continue
        lunch = next(
            (i for i, s in enumerate(day.stops) if s.meal_slot == "lunch" or s.onsite_lunch),
            None,
        )
        dinner = next((i for i, s in enumerate(day.stops) if s.meal_slot == "dinner"), None)

        def phase(
            item: tuple[int, AgentPlanStop],
            lunch_index: int | None = lunch,
            dinner_index: int | None = dinner,
        ) -> int:
            position, stop = item
            if stop.meal_slot == "dinner":
                return 3
            if stop.meal_slot == "lunch" or stop.onsite_lunch:
                return 1
            explicit = {"morning": 0, "afternoon": 2, "evening": 4}.get(stop.part_of_day)
            if explicit is not None:
                return explicit
            if dinner_index is not None and position > dinner_index:
                return 4
            if lunch_index is not None and position > lunch_index:
                return 2
            return 0

        stops = tuple(stop for _, stop in sorted(enumerate(day.stops), key=phase))
        days.append(day.model_copy(update={"stops": stops}))
    return chosen.model_copy(update={"days": tuple(days)})


def _require_effective_change(before: WorkingItineraryDraft, after: WorkingItineraryDraft) -> None:
    """Technical IDs and regenerated rationale do not consume a semantic revision."""
    try:
        _guard_local_day_changes(before, after, frozenset())
    except PlannerGuardError:
        return
    if before.lodging_baseline != after.lodging_baseline:
        return
    raise PlannerGuardError("planner_revision_noop")
