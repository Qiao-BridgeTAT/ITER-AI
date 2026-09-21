"""Version dispatcher: old checkpoints keep their engine; new Agent never falls back."""

import asyncio
from contextlib import AsyncExitStack
from datetime import UTC, datetime, timedelta
from typing import Any

from backend.agent.model_gateway import ModelCancellation, ModelGateway, ModelMessage, ModelRole
from backend.agent.planner.dependencies import prepare_workspace_for_evidence_refresh
from backend.agent.planner.evidence import PlannerEvidenceBackend
from backend.agent.planner.graph import PlannerAgentGraph, PlannerGraphContext
from backend.agent.planner.plan_intent_compiler import (
    build_automatic_hotel_request,
    compile_default_strategy_decision,
)
from backend.agent.planner.prepare_durations import inherit_prepare_visit_durations
from backend.agent.planner.react_context import agent_context
from backend.agent.planner.react_mcp_tools import discovered_mcp_tools
from backend.agent.planner.react_runtime import AgentSession, ReActRuntime, react_memory
from backend.agent.planner.react_tools import PlannerToolBindings
from backend.agent.planner.react_web_tools import discovered_web_tools
from backend.agent.planner.route_refresh import requests_route_refresh_only
from backend.agent.planner.workspace import advance
from backend.contracts.v4.enums import PlannerStatus
from backend.contracts.v4.plan_change import PlanChangeRequest
from backend.contracts.v4.planner_decision import BuildOrUpdateStrategyPayload
from backend.contracts.v4.planner_react import (
    DEFAULT_PLANNER_CALL_LIMIT,
    PLANNER_EXECUTION_SECONDS,
    PlannerReactState,
)
from backend.contracts.v4.planner_workspace import PlannerWorkspaceState
from backend.contracts.v4.task_book import TaskBookV4
from backend.providers.amap_mcp import AmapMcpRouter
from backend.providers.request_budget import RequestBudgetExceeded, active_request_budget
from backend.providers.tavily_mcp import TavilyMcpClient, WebSearchMcpError


class PlannerEngineRouter(PlannerAgentGraph):
    def __init__(
        self,
        legacy: PlannerAgentGraph,
        react_evidence: PlannerEvidenceBackend,
        *,
        new_engine: bool = False,
        time_limit_enabled: bool = True,
        max_decisions: int = DEFAULT_PLANNER_CALL_LIMIT,
        mcp: AmapMcpRouter | None = None,
        react_gateway: ModelGateway | None = None,
        web_search: TavilyMcpClient | None = None,
    ) -> None:
        self.legacy = legacy
        self.react_evidence = react_evidence
        self.new_engine = new_engine
        self.time_limit_enabled = time_limit_enabled
        if not 1 <= max_decisions <= 24:
            raise ValueError("Planner max_decisions must be between 1 and 24")
        self.max_decisions = max_decisions
        self.mcp = mcp
        self.react_gateway = react_gateway
        self.web_search = web_search

    def __getattr__(self, name: str) -> Any:
        return getattr(self.legacy, name)

    def pin_new_run(self, workspace: PlannerWorkspaceState) -> PlannerWorkspaceState:
        # Called only for a newly admitted generation, never when restoring it.
        return workspace.model_copy(
            update={
                "react_state": PlannerReactState(
                    deadline_at=datetime.now(UTC) + timedelta(seconds=PLANNER_EXECUTION_SECONDS),
                    time_limit_disabled=not self.time_limit_enabled,
                    planner_call_limit=self.max_decisions,
                    dialogue_mode="json_schema",
                )
                if self.new_engine
                else None
            }
        )

    def begin_answer_segment(self, workspace: PlannerWorkspaceState) -> PlannerWorkspaceState:
        """A newly accepted user answer starts work; crash/cancel recovery never calls this."""
        if workspace.react_state is None:
            return workspace
        return workspace.model_copy(
            update={
                "react_state": react_memory(workspace).model_copy(
                    update={
                        "deadline_at": datetime.now(UTC)
                        + timedelta(seconds=PLANNER_EXECUTION_SECONDS),
                        "time_limit_disabled": not self.time_limit_enabled,
                        "planner_call_limit": self.max_decisions,
                        "planner_calls": 0,
                        "external_requests": 0,
                        "web_search_calls": 0,
                        "web_search_blocked": False,
                        "effective_revisions": 0,
                        "review": None,
                        "reviewer_calls": 0,
                        "reviewer_query_rounds": 0,
                        "reviewer_messages": (),
                        "reviewer_batches": (),
                        "review_in_progress": False,
                        "stop_reason": None,
                    }
                )
            }
        )

    def resume_with_configured_budget(
        self, workspace: PlannerWorkspaceState
    ) -> PlannerWorkspaceState:
        """Only an admitted user resume may start a new exhausted execution segment.

        Automatic/crash recovery never calls this method. Immutable workspace
        revisions retain the previous segment's counters and deadline.
        """
        state = workspace.react_state
        if (
            state is not None
            and workspace.status is PlannerStatus.FAILED
            and (
                state.stop_reason
                in {
                    "deadline_exceeded",
                    "decision_time_budget_exhausted",
                    "planner_decision_budget_exhausted",
                    "external_budget_exhausted",
                    "repeated_malformed_decision",
                    "repeated_failed_action",
                }
                or (
                    not state.time_limit_disabled
                    and state.deadline_at <= datetime.now(UTC) + timedelta(seconds=20)
                )
            )
        ):
            renewed = self.begin_answer_segment(workspace)
            assert renewed.react_state is not None
            return advance(
                renewed,
                react_state=renewed.react_state.model_copy(update={"review": state.review}),
            )
        if (
            state is None
            or state.stop_reason != "planner_decision_budget_exhausted"
            or self.max_decisions <= state.planner_call_limit
        ):
            return workspace
        return advance(
            workspace,
            react_state=state.model_copy(
                update={"planner_call_limit": self.max_decisions, "stop_reason": None}
            ),
        )

    async def invoke(
        self, workspace: PlannerWorkspaceState, context: PlannerGraphContext
    ) -> PlannerWorkspaceState:
        if workspace.react_state is None:
            return await self.legacy.invoke(workspace, context)
        if workspace.status is PlannerStatus.READY_TO_PUBLISH:
            return workspace
        session = AgentSession(workspace, context)
        token = active_request_budget.set(session.budget)
        try:
            async with asyncio.timeout(session.timeout(reserve=20)):
                return await self._invoke_react(session, context)
        except (TimeoutError, RequestBudgetExceeded) as error:
            await session.memory(
                stop_reason="deadline_exceeded"
                if isinstance(error, TimeoutError)
                else "external_budget_exhausted"
            )
            if session.workspace.status not in {
                PlannerStatus.READY_TO_PUBLISH,
                PlannerStatus.AWAITING_USER,
            }:
                await session.update(lambda w: advance(w, status=PlannerStatus.FAILED))
            return session.workspace
        finally:
            active_request_budget.reset(token)

    async def _invoke_react(
        self, session: AgentSession, context: PlannerGraphContext
    ) -> PlannerWorkspaceState:
        workspace = session.workspace
        if not workspace.initial_evidence_ready:
            await session.progress("runtime", "先核对已确认任务和候选地点。")

            async def checkpoint(value: PlannerWorkspaceState) -> None:
                await session.update(
                    lambda current: value.model_copy(update={"react_state": current.react_state})
                )

            initialized = await self.react_evidence.initialize(
                workspace,
                context.book,
                context.cancellation,
                checkpoint=checkpoint,
                agent_driven=True,
            )
            await checkpoint(inherit_prepare_visit_durations(initialized))
        if session.workspace.planning_strategy is None:
            decision = compile_default_strategy_decision(session.workspace, context.book)
            assert isinstance(decision.payload, BuildOrUpdateStrategyPayload)
            strategy = decision.payload.proposed_strategy
            await session.update(
                lambda w: advance(
                    w,
                    planning_strategy=strategy,
                    decision_trace=(*w.decision_trace, decision),
                )
            )
        bindings = PlannerToolBindings(
            self.react_evidence, self.legacy.materializer, self.legacy.validator
        )
        await self._inherit_prepare_hotels(session, bindings)
        if session.state.route_refresh_only and session.workspace.validation_observation is None:
            await session.progress("runtime", "正在补查缺失路线，保留地点顺序并更新交通时间。")
            checked = await bindings.checked(session.workspace, session)
            await session.update(
                lambda w: checked.model_copy(update={"react_state": w.react_state})
            )

        def data(w: PlannerWorkspaceState) -> dict[str, Any]:
            return agent_context(w, context.book)

        tools = bindings.tools()
        if self.mcp is not None and not session.state.route_refresh_only:
            replaced = {
                "search_places",
                "search_hotel_locations",
                "lookup_place",
                "lookup_routes",
            }
            tools = tuple(t for t in tools if t.definition.name not in replaced)
            try:
                discovered = await discovered_mcp_tools(self.mcp, bindings)
                if not any(t.definition.name == "maps_polygon_search" for t in discovered):
                    tools = tuple(t for t in tools if t.definition.name != "search_along_route")
                tools += tuple(t for t in discovered if t.definition.name != "maps_polygon_search")
            except RequestBudgetExceeded:
                await session.progress("runtime", "查询额度已用完，继续检查已有资料和安排。")
        async with AsyncExitStack() as stack:
            if (
                self.web_search is not None
                and not session.state.route_refresh_only
                and session.state.web_search_calls < 4
                and not session.state.web_search_blocked
                and session.state.external_requests < 80
            ):
                try:
                    web = await stack.enter_async_context(self.web_search.connect())
                    tools += await discovered_web_tools(web)
                except WebSearchMcpError:
                    await session.progress("runtime", "联网搜索暂时不可用，先用已有资料继续规划。")
            if session.state.route_refresh_only:
                tools = tuple(
                    t
                    for t in tools
                    if t.definition.name
                    in {
                        "check_plan",
                        "review_plan",
                        "submit_review",
                        "finish_plan",
                    }
                )
            runtime = ReActRuntime(self._gateway_for(session.workspace), tools, data)
            bindings.runtime = runtime
            result = await runtime.invoke(session)
        if result.status not in {PlannerStatus.READY_TO_PUBLISH, PlannerStatus.AWAITING_USER}:
            await session.update(lambda w: advance(w, status=PlannerStatus.FAILED))
        return session.workspace

    async def _inherit_prepare_hotels(
        self, session: AgentSession, bindings: PlannerToolBindings
    ) -> None:
        """Join the selected Prepare request/cache before model hotel selection."""
        if (
            session.state.route_refresh_only
            or not session.context.book.lodging_direction.search_examples
        ):
            return
        request = build_automatic_hotel_request(session.workspace, session.context.book)
        if request is None:
            return
        from backend.agent.planner.location_capabilities import execute_location_capability

        await session.progress("runtime", "正在接收住宿预查资料，核对酒店位置。")
        update = await execute_location_capability(
            request,
            session.workspace,
            session.context.book,
            providers=self.react_evidence.providers,
            registry=self.react_evidence.registry,
            cancellation=session.context.cancellation,
            now=self.react_evidence.clock(),
            reuse_prepare=True,
        )
        await session.update(lambda w: bindings.merge(w, update, session))
        if update.hotel is not None:
            message = {
                "available": "已接收可用酒店资料，接下来结合行程选择住宿。",
                "empty": "住宿预查正常返回，但没有匹配结果，已保留这个缺口。",
                "failed": "住宿预查或位置核验失败，已记录具体原因。",
                "unverified": "已查到酒店候选，但暂时没有通过条件与位置核验的酒店。",
            }.get(update.hotel.query_status or "")
            if message:
                await session.progress("runtime", message)

    def _gateway_for(self, workspace: PlannerWorkspaceState) -> ModelGateway:
        if workspace.react_state and workspace.react_state.dialogue_mode == "json_schema":
            if self.react_gateway is None:
                raise ValueError("schema Planner requires its dedicated Flash gateway")
            return self.react_gateway
        return self.legacy.gateway

    async def compose_response(
        self,
        workspace: PlannerWorkspaceState,
        book: TaskBookV4,
        cancellation: ModelCancellation,
        *,
        change_request: PlanChangeRequest | None = None,
        gateway: ModelGateway | None = None,
    ) -> str:
        # Execution status is authoritative. An LLM must not reinterpret a
        # timeout as missing inventory, prices, or an unavailable hotel provider.
        if workspace.react_state and workspace.status is PlannerStatus.FAILED:
            reason = {
                "deadline_exceeded": "本轮规划时间已用完",
                "decision_time_budget_exhausted": "本轮规划时间已用完",
                "planner_decision_budget_exhausted": "本轮规划尝试次数已用完",
                "external_budget_exhausted": "本轮查询次数已用完",
                "repeated_malformed_decision": "本轮方案格式修正未完成",
                "repeated_failed_action": "本轮仍有未解决的执行问题",
            }.get(workspace.react_state.stop_reason or "")
            if reason:
                cancellation.raise_if_cancelled("planner_failure_response")
                saved = (
                    "已保留当前安排和查询资料"
                    if workspace.working_itinerary is not None
                    else "已保留任务书和查询资料"
                )
                return f"{reason}，尚未完成本次行程规划。{saved}，可以点击继续规划。"
        return await self.legacy.compose_response(
            workspace,
            book,
            cancellation,
            change_request=change_request,
            gateway=gateway or self._gateway_for(workspace),
        )

    async def apply_plan_change(
        self,
        workspace: PlannerWorkspaceState,
        context: PlannerGraphContext,
        *,
        change_request: PlanChangeRequest,
        user_text: str,
        refresh_evidence: bool = False,
    ) -> PlannerWorkspaceState:
        if workspace.react_state is None:
            return await self.legacy.apply_plan_change(
                workspace,
                context,
                change_request=change_request,
                user_text=user_text,
                refresh_evidence=refresh_evidence,
            )
        route_only = requests_route_refresh_only(user_text) and not refresh_evidence
        if change_request.requested_scope == "full_replan" and not route_only:
            workspace = prepare_workspace_for_evidence_refresh(workspace)
        if route_only:
            workspace = advance(
                workspace,
                cost_draft=None,
                validation_report=None,
                validation_observation=None,
            )
        workspace = advance(
            workspace,
            plan_change_request=change_request,
            react_state=react_memory(workspace).model_copy(
                update={
                    "messages": (ModelMessage(role=ModelRole.USER, content=user_text),),
                    "route_refresh_only": route_only,
                    "review": None,
                }
            ),
        )
        await context.checkpoint(workspace)
        return await self.invoke(workspace, context)
