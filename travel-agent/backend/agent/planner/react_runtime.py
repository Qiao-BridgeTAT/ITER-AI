"""Compiled ReAct loops with durable tool receipts and a separate Reviewer context."""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from typing import Any, Literal, TypedDict, cast

from jsonschema import Draft202012Validator  # type: ignore[import-untyped]
from langchain_core.messages import AIMessage, AnyMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import NodeError
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, ToolRuntime
from langgraph.prebuilt.tool_node import ToolCallRequest, ToolInvocationError
from langgraph.runtime import Runtime
from langgraph.types import Command, RetryPolicy, interrupt
from pydantic import ValidationError

from backend.agent.model_gateway import (
    ModelAuditMetadata,
    ModelFailureCode,
    ModelGateway,
    ModelGatewayError,
    ModelMessage,
    ModelRequest,
    ModelRole,
    ModelToolCall,
    ModelToolDefinition,
)
from backend.agent.planner.decision_feedback import (
    EMPTY_SEARCH_ACTION,
    decision_obstacles,
    empty_place_search,
    repair_action,
    repeated_empty_search,
)
from backend.agent.planner.graph import PlannerGraphContext
from backend.agent.planner.prepared_facts import fully_cached_fact_tools
from backend.agent.planner.react_context import (
    candidate_argument_feedback,
    compact_candidate_facts,
    duplicate_candidate_issues,
    edit_candidate_feedback,
    meal_conflict_feedback,
    model_projection,
    recent_dialogue,
)
from backend.agent.planner.react_prompts import PROMPT_VERSION, agent_system_prompt
from backend.agent.planner.react_review import evidence_digest, has_current_review
from backend.agent.planner.react_schema import bind_plan_tool_schemas
from backend.agent.planner.workspace import PlannerGuardError, advance
from backend.agent.schema_tool_dialogue import SchemaDecisionError, generate_schema_tool_turn
from backend.agent.tool_dialogue import MAX_TOOL_BATCH_SIZE, generate_tool_turn
from backend.agent.tool_schema_errors import tool_schema_error_details
from backend.contracts.v4.enums import PlannerStatus
from backend.contracts.v4.planner_react import (
    MAX_EFFECTIVE_REVISIONS,
    PlannerReactState,
    ToolReceipt,
)
from backend.contracts.v4.planner_workspace import PlannerWorkspaceState
from backend.persistence.outbox_repository import canonical_json_hash
from backend.providers.contracts import ProviderError
from backend.providers.request_budget import (
    RequestBudget,
    RequestBudgetExceeded,
    active_request_budget,
)

Source = Literal["planner", "reviewer"]
PUBLIC_TOOL_ACTIONS = {
    "maps_text_search": "正在搜索相关地点。",
    "maps_around_search": "正在查找附近合适的地点。",
    "maps_polygon_search": "正在查找沿途合适的地点。",
    "search_along_route": "正在查找沿途合适的地点。",
    "maps_search_detail": "正在核对地点详情。",
    "search_places": "正在搜索相关地点。",
    "search_hotel_locations": "正在核对酒店位置。",
    "lookup_place": "正在核对地点详情。",
    "lookup_hours": "正在核对营业时间。",
    "lookup_weather": "正在查询旅行期间的天气。",
    "lookup_tickets": "正在查询门票信息。",
    "search_hotels": "正在查询住宿和入住日期价格。",
    "refresh_hotel": "正在核对这家酒店的入住日期价格。",
    "write_plan": "正在整理这版行程。",
    "edit_plan": "正在修改需要调整的安排。",
    "patch_plan": "正在更新需要调整的安排。",
    "check_plan": "正在计算时间和费用。",
    "review_plan": "正在评审这版行程。",
    "submit_review": "正在整理检查结论。",
    "finish_plan": "正在整理完整行程和出行提示。",
    "ask_user": "正在整理需要你确认的选择。",
}
Mutation = Callable[
    [PlannerWorkspaceState], PlannerWorkspaceState | Awaitable[PlannerWorkspaceState]
]


def react_memory(workspace: PlannerWorkspaceState) -> PlannerReactState:
    if workspace.react_state is None:
        raise PlannerGuardError("react_engine_not_pinned")
    return workspace.react_state


@dataclass(frozen=True)
class ToolOutcome:
    result: dict[str, Any]
    apply: Mutation | None = None


@dataclass(frozen=True)
class AgentTool:
    definition: ModelToolDefinition
    handler: Callable[[dict[str, Any], AgentSession], Awaitable[ToolOutcome]]
    read_only: bool = True
    reviewer_allowed: bool = False
    exclusive_query_group: str | None = None


class AgentSession:
    """Single owner serializes workspace writes, including parallel I/O reservations."""

    def __init__(self, workspace: PlannerWorkspaceState, context: PlannerGraphContext) -> None:
        if workspace.react_state is None:
            raise PlannerGuardError("react_engine_not_pinned")
        self.workspace = workspace
        self.context = context
        self.lock = asyncio.Lock()
        self.active_tools: set[asyncio.Task[Any]] = set()
        self.budget = RequestBudget(
            None if self.state.time_limit_disabled else self.state.deadline_at,
            used=self.state.external_requests,
            persist=self.reserve_external,
        )

    @property
    def state(self) -> PlannerReactState:
        assert self.workspace.react_state is not None
        return self.workspace.react_state

    async def update(self, mutation: Mutation) -> None:
        async with self.lock:
            self.context.cancellation.raise_if_cancelled("react_checkpoint")
            following = mutation(self.workspace)
            if inspect.isawaitable(following):
                following = await following
            # A calculation may have started before its I/O reservations advanced
            # the durable revision. Its semantic result must follow those writes.
            following = following.model_copy(
                update={
                    "workspace_revision": max(
                        following.workspace_revision, self.workspace.workspace_revision + 1
                    )
                }
            )
            interaction = following.active_interaction
            if interaction is not None and following.status is PlannerStatus.AWAITING_USER:
                revision = following.workspace_revision
                following = following.model_copy(
                    update={
                        "active_interaction": interaction.model_copy(
                            update={
                                "based_on_workspace_revision": revision,
                                "scope": interaction.scope.model_copy(
                                    update={"workspace_revision": revision}
                                ),
                            }
                        )
                    }
                )
            await self.context.checkpoint(following)
            self.workspace = following

    async def memory(self, **changes: Any) -> None:
        await self.update(
            lambda w: advance(w, react_state=react_memory(w).model_copy(update=changes))
        )

    async def reserve_external(self, used: int) -> None:
        await self.memory(external_requests=used)

    async def progress(self, source: Source | Literal["tool", "runtime"], text: str) -> None:
        await self.context.progress(f"agent.{source}", text)

    def remaining(self) -> float:
        return self.budget.remaining()

    def timeout(self, *, reserve: float = 0) -> float | None:
        return None if self.state.time_limit_disabled else max(0.01, self.remaining() - reserve)


class LoopState(TypedDict, total=False):
    next: str
    messages: list[AnyMessage]


@dataclass(frozen=True)
class LoopContext:
    session: AgentSession
    source: Source


class ToolArgumentFailure(ValueError):
    def __init__(self, code: str, details: list[dict[str, Any]] | None = None) -> None:
        self.code = code
        self.details = details
        super().__init__(code)


def tool_error_content(error: Exception) -> str:
    """Domain-safe formatting hook; ToolNode owns exception-to-ToolMessage flow."""
    details = None
    if isinstance(error, ToolArgumentFailure):
        code, details = error.code, error.details
    elif isinstance(error, PlannerGuardError):
        code = error.code
    elif isinstance(error, ProviderError):
        code = f"provider_{error.code.value}"
    elif isinstance(error, TimeoutError):
        code = "tool_timeout"
    elif isinstance(error, (ValidationError, ToolInvocationError)):
        source = error.source if isinstance(error, ToolInvocationError) else error
        code = "tool_contract_invalid"
        details = [
            dict(item)
            for item in source.errors(
                include_url=False, include_input=False, include_context=False
            )[:5]
        ]
    elif isinstance(error, RequestBudgetExceeded):
        code = "external_budget_exhausted"
    else:
        # Never turn cancellation, graph interrupts, persistence failures or
        # programming errors into an ordinary observation.
        raise error
    result: dict[str, Any] = {"ok": False, "error": code}
    if code.startswith(
        ("planner_plan_same_venue_duplicate", "planner_plan_missing_required_restaurant")
    ):
        result["next_action"] = repair_action(code)
    if code == "empty_search_requires_different_plan_or_area":
        result["next_action"] = EMPTY_SEARCH_ACTION
    if details:
        result["validation_issues"] = details
    if code.startswith("mcp_coordinate_invalid:"):
        result["next_action"] = (
            "坐标参数错误：高德使用 经度,纬度；不要交换顺序。"
            "请直接复制当前目标 candidates 中的 mcp_location 后重试。未发起外部查询。"
        )
    hints = {
        "dependent_or_mutating_tools_must_run_alone": (
            "本批工具均未执行。草稿修改、试算、评审、发布等独占操作须单独调用；"
            "酒店查询可以和营业等查询并行，但同批只能有一个酒店商品查询。拆分后可保留原有效参数。"
        ),
        "tool_batch_limit_exceeded": "本批未执行。同轮最多四项独立查询，先读取结果再决定后续行动。",
        "duplicate_requests_in_tool_batch": "本批未执行。相同名称和参数只保留一项。",
        "identical_failed_request_requires_changed_arguments": (
            "此前相同草稿与证据上的调用已失败。修改相关草稿、取得新证据或改变有效参数后才可再试。"
        ),
        "review_rejection_requires_actionable_error": (
            "拒绝结论与全部为 warning 的问题清单矛盾，本次结论尚未保存。"
            "请依据任务、事实和实际时间轴重新判断：存在具体错误时说明依据与修订；"
            "只剩允许披露的缺口时可通过并保留警告。不能仅为满足格式把未知事实改成硬错误。"
        ),
        "planner_revision_noop": (
            "这份草稿已保存，本次没有实际修改。可调用 review_plan 检查现有版本；"
            "若已被评审拒绝，需根据具体问题改变安排，不能原样送审。"
        ),
    }
    if code in hints:
        result["next_action"] = hints[code]
    return json.dumps(result, ensure_ascii=False)


class ReActRuntime:
    def __init__(
        self,
        gateway: ModelGateway,
        tools: tuple[AgentTool, ...],
        context_data: Callable[[PlannerWorkspaceState], dict[str, Any]],
    ) -> None:
        self.gateway = gateway
        self.tools = {tool.definition.name: tool for tool in tools}
        if len(self.tools) != len(tools):
            raise ValueError("duplicate tool registration")
        self.context_data = context_data
        self.tool_node = ToolNode(
            [self._native_tool(tool) for tool in tools],
            handle_tool_errors=tool_error_content,
            awrap_tool_call=self._durable_tool_call,
        )
        self.checkpointer = InMemorySaver()
        self.compiled = self._compile(self.checkpointer)
        self.reviewer = self._compile()

    def _compile(self, checkpointer: InMemorySaver | None = None) -> Any:
        graph = StateGraph(LoopState, context_schema=LoopContext)
        graph.add_node(
            "decide",
            cast(Any, self._decide),
            retry_policy=RetryPolicy(
                max_attempts=2,
                initial_interval=0.25,
                max_interval=1,
                jitter=False,
                retry_on=lambda error: (
                    isinstance(error, ModelGatewayError)
                    and error.retryable
                    and error.code
                    in {
                        ModelFailureCode.RATE_LIMITED,
                        ModelFailureCode.UNAVAILABLE,
                        ModelFailureCode.TIMEOUT,
                    }
                ),
            ),
            error_handler=cast(Any, self._decision_error),
        )
        graph.add_node("prepare_tools", cast(Any, self._prepare_tools))
        graph.add_node("tools", self.tool_node)
        graph.add_node("observe", cast(Any, self._observe))
        graph.add_node("await_user", cast(Any, self._await_user))
        graph.add_edge("await_user", "decide")
        graph.add_conditional_edges(
            START,
            self._restore_route,
            {"decide": "decide", "tools": "prepare_tools", "await_user": "await_user"},
        )
        graph.add_conditional_edges(
            "decide", lambda s: s["next"], {"tools": "prepare_tools", "end": END}
        )
        graph.add_edge("prepare_tools", "tools")
        graph.add_edge("tools", "observe")
        graph.add_conditional_edges(
            "observe",
            lambda s: s["next"],
            {"decide": "decide", "await_user": "await_user", "end": END},
        )
        return graph.compile(checkpointer=checkpointer)

    def _restore_route(self, state: LoopState, runtime: Runtime[LoopContext]) -> str:
        ctx = runtime.context
        interaction = ctx.session.workspace.active_interaction
        if (
            ctx.source == "planner"
            and interaction
            and interaction.interaction_id != ctx.session.state.resumed_interaction_id
        ):
            return "await_user"
        messages = self._messages(ctx)
        if messages and messages[-1].tool_calls:
            return "tools"
        return "decide"

    def _messages(self, ctx: LoopContext) -> tuple[ModelMessage, ...]:
        state = ctx.session.state
        return state.messages if ctx.source == "planner" else state.reviewer_messages

    def _allowed(self, source: Source) -> dict[str, AgentTool]:
        return {
            name: tool
            for name, tool in self.tools.items()
            if source == "planner"
            and name != "submit_review"
            or source == "reviewer"
            and tool.reviewer_allowed
        }

    @staticmethod
    def _invalid_batch(
        calls: tuple[ModelToolCall, ...], allowed: dict[str, AgentTool]
    ) -> str | None:
        if len(calls) > MAX_TOOL_BATCH_SIZE:
            return "tool_batch_limit_exceeded"
        requests = [
            canonical_json_hash(
                {"name": call.function.name, "arguments": json.loads(call.function.arguments)}
            )
            for call in calls
        ]
        if len(set(requests)) != len(requests):
            return "duplicate_requests_in_tool_batch"
        selected = [allowed[call.function.name] for call in calls if call.function.name in allowed]
        groups = [tool.exclusive_query_group for tool in selected if tool.exclusive_query_group]
        if len(calls) > 1 and (
            any(not tool.read_only for tool in selected) or len(groups) != len(set(groups))
        ):
            return "dependent_or_mutating_tools_must_run_alone"
        return None

    async def invoke(self, session: AgentSession) -> PlannerWorkspaceState:
        token = active_request_budget.set(session.budget)
        try:
            if session.state.stop_reason is not None:
                await session.memory(stop_reason=None)
            from backend.agent.planner.human_checkpoint import dump_interrupt, restore_interrupt

            config = {
                "configurable": {"thread_id": session.workspace.generation_id, "checkpoint_ns": ""},
                "recursion_limit": 5
                * max(1, session.state.planner_call_limit - session.state.planner_calls)
                + 8,
            }
            if session.state.native_interrupt_checkpoint:
                await restore_interrupt(
                    self.checkpointer, config, session.state.native_interrupt_checkpoint
                )
            interaction = session.workspace.active_interaction
            answer = next(
                (
                    a
                    for a in reversed(session.workspace.interaction_answers)
                    if interaction and a.interaction_id == interaction.interaction_id
                ),
                None,
            )
            async with asyncio.timeout(session.timeout()):
                context = LoopContext(session, "planner")
                if (
                    answer
                    and interaction
                    and interaction.interaction_id != session.state.resumed_interaction_id
                ):
                    if session.state.native_interrupt_checkpoint is None:
                        # Recovery after the app persisted the question but before
                        # the graph snapshot: reconstruct the pure interrupt node.
                        await self.compiled.ainvoke(
                            {"next": "await_user"}, context=context, config=config
                        )
                    await self.compiled.ainvoke(
                        Command(resume=answer.answer_id), context=context, config=config
                    )
                else:
                    await self.compiled.ainvoke({"next": "decide"}, context=context, config=config)
                if session.workspace.status is PlannerStatus.AWAITING_USER:
                    await session.memory(
                        native_interrupt_checkpoint=await dump_interrupt(self.checkpointer, config)
                    )
        except TimeoutError:
            await session.memory(stop_reason="deadline_exceeded")
        finally:
            # ToolNode owns dispatch/concurrency. Join outstanding I/O when the
            # surrounding product run is cancelled or an unexpected error escapes.
            pending = [task for task in session.active_tools if task is not asyncio.current_task()]
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            active_request_budget.reset(token)
        return session.workspace

    async def _await_user(self, _: LoopState, runtime: Runtime[LoopContext]) -> LoopState:
        session = runtime.context.session
        interaction = session.workspace.active_interaction
        if interaction is None:
            raise PlannerGuardError("planner_interaction_missing")
        answer_id = interrupt(
            {
                "interaction_id": interaction.interaction_id,
                "question": interaction.question,
                "options": [
                    option.model_dump(mode="json") for option in interaction.option_contracts
                ],
            }
        )
        answer = next(
            (
                a
                for a in session.workspace.interaction_answers
                if a.interaction_id == interaction.interaction_id and a.answer_id == answer_id
            ),
            None,
        )
        if answer is None:
            raise PlannerGuardError("planner_native_resume_requires_accepted_user_answer")
        await session.memory(
            resumed_interaction_id=interaction.interaction_id, native_interrupt_checkpoint=None
        )
        return {"next": "decide"}

    async def review(self, session: AgentSession) -> None:
        if not session.state.review_in_progress:
            await session.memory(
                review_in_progress=True,
                reviewer_calls=0,
                reviewer_query_rounds=0,
                reviewer_messages=(),
                reviewer_batches=(),
                review=None,
            )
        await self.reviewer.ainvoke(
            {"next": "decide"},
            context=LoopContext(session, "reviewer"),
            config={"recursion_limit": 26},
        )

    async def _decision_error(
        self, _: LoopState, error: NodeError, runtime: Runtime[LoopContext]
    ) -> Command[str]:
        failure = error.error
        if (
            not isinstance(failure, ModelGatewayError)
            or failure.code is not ModelFailureCode.MALFORMED_RESPONSE
        ):
            raise failure
        ctx = runtime.context
        session = ctx.session
        feedback = ModelMessage(
            role=ModelRole.USER,
            content=json.dumps(
                {
                    "model_response_error": (
                        "上一轮输出不符合本轮响应合同，未执行工具。"
                        "请根据以下字段问题修正，再按当前 Schema 返回完整决策。"
                    ),
                    "validation_issues": list(failure.validation_issues),
                    **(
                        {
                            "rejected_decision": failure.rejected_decision,
                            "repair_scope": "这份决策未执行、未保存。优先在原方案上修正上述格式；"
                            "无需仅因格式错误重查已取得的事实。修正后仍须通过工具参数与业务校验。",
                        }
                        if isinstance(failure, SchemaDecisionError)
                        else {}
                    ),
                },
                ensure_ascii=False,
            ),
        )
        messages = self._messages(ctx)
        if messages and messages[-1] == feedback:
            await session.memory(stop_reason="repeated_malformed_decision")
            return Command(goto=END)
        key = "messages" if ctx.source == "planner" else "reviewer_messages"
        await session.memory(**{key: (*messages, feedback)})
        return Command(goto="decide")

    async def _decide(self, _: LoopState, runtime: Runtime[LoopContext]) -> LoopState:
        ctx = runtime.context
        session = ctx.session
        if session.remaining() <= 20:
            await session.memory(stop_reason="decision_time_budget_exhausted")
            return {"next": "end"}
        calls = (
            session.state.planner_calls if ctx.source == "planner" else session.state.reviewer_calls
        )
        if calls >= (session.state.planner_call_limit if ctx.source == "planner" else 5):
            if ctx.source == "planner":
                await session.memory(stop_reason="planner_decision_budget_exhausted")
            return {"next": "end"}
        # Reserve before requesting the model; restart does not refund interrupted calls.
        await session.memory(
            **{("planner_calls" if ctx.source == "planner" else "reviewer_calls"): calls + 1}
        )
        data = self.context_data(session.workspace)
        data["decision_obstacles"] = decision_obstacles(session.workspace)
        if ctx.source == "reviewer":
            data.pop("review", None)
        data["remaining_budget"] = {
            "revisions": max(0, MAX_EFFECTIVE_REVISIONS - session.state.effective_revisions),
            "time_limit_enabled": not session.state.time_limit_disabled,
            "seconds": None if session.state.time_limit_disabled else int(session.remaining()),
            "seconds_available_for_actions": None
            if session.state.time_limit_disabled
            else max(0, int(session.remaining()) - 20),
            "external_requests": 80 - session.state.external_requests,
            "web_search_calls": 0
            if session.state.web_search_blocked
            else 4 - session.state.web_search_calls,
            **(
                {
                    "planner_decisions": session.state.planner_call_limit
                    - session.state.planner_calls,
                }
                if ctx.source == "planner"
                else {
                    "reviewer_decisions": 5 - session.state.reviewer_calls,
                    "reviewer_query_rounds": 2 - session.state.reviewer_query_rounds,
                }
            ),
        }
        # A successful replacement supersedes errors in the previous draft
        # proposal. Full receipts/messages remain durable for audit/recovery.
        last_write = max(
            (
                index
                for index, receipt in enumerate(session.state.receipts)
                if receipt.status == "completed"
                and receipt.call.function.name
                in (
                    {"write_plan", "edit_plan", "patch_plan"}
                    if ctx.source == "planner"
                    else {"write_plan", "edit_plan", "patch_plan", "submit_review"}
                )
            ),
            default=-1,
        )
        # Keep the original actionable error once. Subsequent replay denials
        # must not push the actual root cause out of the four-error context.
        unique_failures: dict[str, ToolReceipt] = {}
        for receipt in session.state.receipts[last_write + 1 :]:
            if receipt.source == ctx.source and receipt.status == "failed" and receipt.result:
                unique_failures.setdefault(receipt.fingerprint, receipt)
        data["recent_failed_actions"] = [
            {
                "tool": receipt.call.function.name,
                **(
                    {"arguments": json.loads(receipt.call.function.arguments)}
                    if receipt.read_only
                    else {}
                ),
                "error": json.loads(receipt.result).get("error"),
                "validation_issues": json.loads(receipt.result).get("validation_issues"),
                "next_action": json.loads(receipt.result).get("next_action"),
                **(
                    {"repair_context": repair}
                    if (
                        repair := meal_conflict_feedback(
                            json.loads(receipt.result).get("error"), session.workspace
                        )
                    )
                    else {}
                ),
            }
            for receipt in unique_failures.values()
            if receipt.result
        ][-4:]
        rejected_plan = next(
            (
                receipt
                for receipt in reversed(session.state.receipts[last_write + 1 :])
                if receipt.source == ctx.source
                and receipt.status == "failed"
                and receipt.call.function.name in {"write_plan", "edit_plan", "patch_plan"}
                and receipt.result
                and json.loads(receipt.result).get("error")
                != "identical_failed_request_requires_changed_arguments"
            ),
            None,
        )
        if rejected_plan:
            data["rejected_plan_attempt"] = {
                "status": "not_saved",
                "tool": rejected_plan.call.function.name,
                "arguments": json.loads(rejected_plan.call.function.arguments),
                "feedback": json.loads(rejected_plan.result or "{}"),
                "next_action": repair_action(json.loads(rejected_plan.result or "{}").get("error"))
                + "这份提案尚未保存；以 current_draft 为已保存版本，修改成功后再送审。",
            }
        allowed = self._allowed(ctx.source)
        cached_tools = fully_cached_fact_tools(
            session.workspace, session.context.book, datetime.now(UTC)
        )
        for name in cached_tools:
            allowed.pop(name, None)
        if cached_tools:
            data["cached_query_tools"] = {
                "names": sorted(cached_tools),
                "note": "当前候选和旅行日期的这些查询均已有有效结果，已直接提供在当前事实中；"
                "未知仍是未知。同一查询不会产生新证据，补查其他来源可用联网搜索。"
                "新增候选或结果过期后可再次查询。",
            }
        if session.state.web_search_blocked or session.state.web_search_calls >= 4:
            # Existing excerpts stay in shared context. Do not invite the model
            # to spend another decision on a service that cannot accept requests.
            allowed.pop("tavily_search", None)
            allowed.pop("tavily_extract", None)
        if session.state.external_requests >= 80:
            # Cached facts are already in context. No fresh query is executable;
            # keep local edits, calculation, review and guarded finalization.
            allowed = {name: tool for name, tool in allowed.items() if not tool.read_only}
        if session.workspace.validation_observation is not None:
            # The current calculation already appears in the review context.
            # A fact-changing query invalidates it and exposes calculation again.
            allowed.pop("check_plan", None)
        if ctx.source == "planner" and has_current_review(session.workspace):
            # Repeating an unchanged rejected review does not fix the plan.
            # Editing or acquiring new evidence will expose review again.
            allowed.pop("review_plan", None)
        readiness = session.workspace.readiness_observation
        validation = session.workspace.validation_observation
        decision_needed = any(
            issue.user_authority_required for issue in (readiness.issues if readiness else ())
        ) or any(
            issue.user_authority_required and "ask_user" in issue.allowed_actions
            for issue in (validation.issues if validation else ())
        )
        decision_needed = decision_needed or any(
            entry.commitment_level.value == "strong"
            for entry in session.workspace.candidate_pool.candidates
        )
        if not decision_needed:
            allowed = {name: tool for name, tool in allowed.items() if name != "ask_user"}
        if session.workspace.plan_change_request is None:
            allowed = {n: t for n, t in allowed.items() if n != "patch_plan"}
        if session.workspace.working_itinerary is None:
            allowed.pop("edit_plan", None)
        if (
            not session.context.allow_semantic_repair
            or session.state.effective_revisions >= MAX_EFFECTIVE_REVISIONS
        ):
            allowed = {
                n: t
                for n, t in allowed.items()
                if n not in {"write_plan", "edit_plan", "patch_plan"}
            }
        if (
            session.workspace.working_itinerary is not None
            and session.state.effective_revisions >= MAX_EFFECTIVE_REVISIONS
        ):
            # No remaining edit can consume a new candidate. Close the current
            # version using existing facts; review_plan still performs necessary
            # current-plan calculations before its independent assessment.
            closing_tools = (
                {"review_plan", "finish_plan", "ask_user"}
                if ctx.source == "planner"
                else {"check_plan", "submit_review"}
            )
            allowed = {name: tool for name, tool in allowed.items() if name in closing_tools}
            data["completion_guidance"] = (
                f"本段 {MAX_EFFECTIVE_REVISIONS} 轮有效修订已用完，不再查找无法采用的新候选。"
                "当前版未评审则完成评审；已有当前结论则直接交付完整行程，附具体提示。"
                "不重复试算或送审以寻求通过，不能将未解决问题描述为已解决。"
            )
        if ctx.source == "reviewer" and session.state.reviewer_query_rounds >= 2:
            allowed = {
                name: tool
                for name, tool in allowed.items()
                if name in {"submit_review", "check_plan"}
            }
        forced = None
        if (
            ctx.source == "planner"
            and session.workspace.working_itinerary is not None
            and session.state.review is None
            and session.remaining() <= 100
            and "review_plan" in allowed
        ):
            # Prefer a review before optional searches under the remaining-time limit.
            # obtain that verdict before further optional searches or edits.
            # This does not approve the plan, choose fixes, or renew the deadline.
            forced = "review_plan"
            allowed = {forced: allowed[forced]}
        if ctx.source == "reviewer" and (
            calls >= 4
            or calls == 3
            and session.workspace.validation_observation is None
            or session.remaining() <= 100
        ):
            # Bound the review's reporting phase, never the verdict. The model
            # still owns every issue and approval/rejection; no new query can
            # displace calculation and an explicit outcome at budget exhaustion.
            # Keep time in the same segment for a Planner edit, re-review and
            # final response instead of letting the first review consume it all.
            forced = (
                "check_plan"
                if session.workspace.validation_observation is None
                else "submit_review"
            )
            allowed = {name: tool for name, tool in allowed.items() if name == forced}
        definitions = tuple(tool.definition for tool in allowed.values())
        if session.state.dialogue_mode == "json_schema":
            definitions = bind_plan_tool_schemas(definitions, session.workspace)
        projected = model_projection(data)
        if session.state.dialogue_mode == "json_schema":
            projected = compact_candidate_facts(projected)
        request = ModelRequest(
            messages=[
                ModelMessage(
                    role=ModelRole.SYSTEM,
                    content=agent_system_prompt(ctx.source, allowed),
                ),
                *recent_dialogue(self._messages(ctx), workspace=session.workspace),
                ModelMessage(
                    role=ModelRole.USER,
                    content=json.dumps(projected, ensure_ascii=False, separators=(",", ":")),
                ),
            ],
            tools=definitions,
            forced_tool_name=forced,
            max_output_tokens=6000,
            audit=ModelAuditMetadata(
                stage=f"react_{ctx.source}",
                node="decide",
                attempt=calls + 1,
                contract_version=PROMPT_VERSION,
            ),
        )
        async with asyncio.timeout(session.timeout(reserve=20)):
            generate = (
                partial(
                    generate_schema_tool_turn,
                    parallel_tool_names=frozenset(
                        name for name, tool in allowed.items() if tool.read_only
                    ),
                )
                if session.state.dialogue_mode == "json_schema"
                else generate_tool_turn
            )
            turn = await generate(
                self.gateway,
                request,
                cancellation=session.context.cancellation,
                on_public_text=lambda text: session.progress(ctx.source, text),
            )
        prior_ids = {r.call.id for r in session.state.receipts}
        if any(call.id in prior_ids for call in turn.message.tool_calls):
            raise PlannerGuardError("duplicate_tool_call_id")
        validation = session.workspace.validation_observation
        invalid_mutation_batch = self._invalid_batch(turn.message.tool_calls, allowed)
        receipts = tuple(
            ToolReceipt(
                call=call,
                fingerprint=canonical_json_hash(
                    {
                        "name": call.function.name,
                        "arguments": json.loads(call.function.arguments),
                        # Rejecting a batch never executes its individual tools.
                        # Splitting that batch is a valid repair, even with the
                        # same arguments and unchanged evidence.
                        "invalid_mutation_batch": invalid_mutation_batch,
                        "draft": session.workspace.working_itinerary.content_digest
                        if session.workspace.working_itinerary
                        else None,
                        "evidence": evidence_digest(session.workspace),
                        "validation": validation.validation_fingerprint if validation else None,
                        "review": session.state.review.model_dump(mode="json")
                        if session.state.review
                        else None,
                    }
                ),
                # A well-formed request for an unavailable tool is a permission
                # rejection observation, never executable or a transport crash.
                read_only=allowed[call.function.name].read_only
                if call.function.name in allowed
                else True,
                source=ctx.source,
                status="pending"
                if call.function.name in allowed and call.id not in turn.argument_errors
                else "failed",
                result=None
                if call.function.name in allowed and call.id not in turn.argument_errors
                else json.dumps(
                    {
                        "ok": False,
                        "error": "tool_arguments_or_permission_invalid",
                        **(
                            {
                                "validation_issues": candidate_argument_feedback(
                                    call.function.name,
                                    json.loads(call.function.arguments),
                                    turn.argument_errors[call.id],
                                    session.workspace,
                                )
                            }
                            if call.id in turn.argument_errors
                            else {"allowed_tools": list(allowed)}
                        ),
                    }
                ),
            )
            for call in turn.message.tool_calls
        )
        key = "messages" if ctx.source == "planner" else "reviewer_messages"
        # Model decisions and pending calls are durable before any tool starts.
        await session.memory(
            **{
                key: (*self._messages(ctx), turn.message),
                "receipts": (*session.state.receipts, *receipts),
            }
        )
        if not receipts:
            await session.memory(stop_reason="model_returned_without_completion_request")
            return {"next": "end"}
        return {"next": "tools"}

    def _native_tool(self, tool: AgentTool) -> StructuredTool:
        async def run(runtime: ToolRuntime[LoopContext], **arguments: Any) -> ToolMessage:
            return await self._run_tool(tool, arguments, runtime)

        return StructuredTool(
            name=tool.definition.name,
            description=tool.definition.description,
            args_schema=tool.definition.parameters,
            coroutine=run,
        )

    async def _prepare_tools(self, _: LoopState, runtime: Runtime[LoopContext]) -> LoopState:
        """Apply batch permissions once; all actual dispatch belongs to ToolNode."""
        ctx, session = runtime.context, runtime.context.session
        message = self._messages(ctx)[-1]
        calls = message.tool_calls
        failure = self._invalid_batch(calls, self._allowed(ctx.source))
        batch_id = canonical_json_hash([call.id for call in calls])
        if (
            ctx.source == "reviewer"
            and any(c.function.name not in {"submit_review", "check_plan"} for c in calls)
            and batch_id not in session.state.reviewer_batches
        ):
            if session.state.reviewer_query_rounds >= 2:
                failure = "reviewer_query_budget_exhausted_use_check_plan_or_submit_review"
            else:
                await session.memory(
                    reviewer_query_rounds=session.state.reviewer_query_rounds + 1,
                    reviewer_batches=(*session.state.reviewer_batches, batch_id),
                )
        # Persist denial before native execution; a denied batch cannot partially
        # execute or regain permission on process recovery. Repeated denials are
        # bounded too, even when no underlying tool ran.
        ids = {call.id for call in calls}
        receipts = []
        for receipt in session.state.receipts:
            rejection = None
            if receipt.call.id in ids and receipt.status != "completed":
                if any(
                    r.status == "failed"
                    and r.fingerprint == receipt.fingerprint
                    and r.call.id not in ids
                    for r in session.state.receipts
                ):
                    rejection = "identical_failed_request_requires_changed_arguments"
                elif receipt.status == "pending":
                    rejection = failure
            receipts.append(
                receipt.model_copy(
                    update={
                        "status": "failed",
                        "result": tool_error_content(ToolArgumentFailure(rejection)),
                    }
                )
                if rejection
                else receipt
            )
        if tuple(receipts) != session.state.receipts:
            await session.memory(receipts=tuple(receipts))
        return {
            "messages": [
                AIMessage(
                    content=message.content,
                    tool_calls=[
                        {
                            "type": "tool_call",
                            "id": call.id,
                            "name": call.function.name,
                            "args": json.loads(call.function.arguments),
                        }
                        for call in calls
                    ],
                )
            ]
        }

    async def _durable_tool_call(
        self,
        request: ToolCallRequest,
        execute: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage:
        """Native ToolNode hook for receipt replay and the product commit boundary."""
        context = request.runtime.context
        assert isinstance(context, LoopContext)
        session = context.session
        receipt = next(r for r in session.state.receipts if r.call.id == request.tool_call["id"])
        if receipt.status != "pending":
            assert receipt.result is not None
            return ToolMessage(
                content=receipt.result,
                name=receipt.call.function.name,
                tool_call_id=receipt.call.id,
                status="error" if receipt.status == "failed" else "success",
            )
        task = asyncio.current_task()
        assert task is not None
        session.active_tools.add(task)
        try:
            await session.progress(
                "tool",
                (
                    "正在查询实际路线。"
                    if receipt.call.function.name.startswith("maps_direction_")
                    or receipt.call.function.name == "lookup_routes"
                    else PUBLIC_TOOL_ACTIONS.get(receipt.call.function.name, "正在核对相关信息。")
                ),
            )
            response = await execute(request)
            if not isinstance(response, ToolMessage) or not isinstance(response.content, str):
                raise TypeError("Planner tools must return a textual ToolMessage")
            if response.status == "error":
                await session.update(
                    lambda w: advance(
                        w,
                        react_state=react_memory(w).model_copy(
                            update={
                                "receipts": tuple(
                                    r.model_copy(
                                        update={"status": "failed", "result": response.content}
                                    )
                                    if r.call.id == receipt.call.id
                                    else r
                                    for r in react_memory(w).receipts
                                )
                            }
                        ),
                    )
                )
            await session.progress(
                "tool",
                "这一步未完成，已记录原因。"
                if response.status == "error"
                else "这次查询没有找到地点，已记录查询范围。"
                if empty_place_search(receipt.call.function.name, json.loads(response.content))
                else "这一步已完成。",
            )
            return response
        finally:
            session.active_tools.discard(task)

    async def _run_tool(
        self, tool: AgentTool, args: dict[str, Any], runtime: ToolRuntime[LoopContext]
    ) -> ToolMessage:
        """Business validation and atomic mutation inside a native StructuredTool."""
        ctx, session = runtime.context, runtime.context.session
        receipt = next(r for r in session.state.receipts if r.call.id == runtime.tool_call_id)
        name = tool.definition.name
        if not session.context.allow_semantic_repair and name in {
            "write_plan",
            "edit_plan",
            "patch_plan",
        }:
            raise PlannerGuardError("semantic_change_not_authorized")
        if name not in self._allowed(ctx.source):
            raise PlannerGuardError("tool_arguments_or_permission_invalid")
        # MCP declares JSON Schema, not a Pydantic class. StructuredTool retains
        # that exact schema; enforce it here without coercing provider arguments.
        errors = list(Draft202012Validator(tool.definition.parameters).iter_errors(args))
        if errors:
            raise ToolArgumentFailure(
                "tool_arguments_or_permission_invalid",
                candidate_argument_feedback(
                    name,
                    args,
                    tool_schema_error_details(errors, schema=tool.definition.parameters),
                    session.workspace,
                ),
            )
        ids = {call.id for call in self._messages(ctx)[-1].tool_calls}
        if any(
            r.status == "failed" and r.fingerprint == receipt.fingerprint and r.call.id not in ids
            for r in session.state.receipts
        ):
            raise PlannerGuardError("identical_failed_request_requires_changed_arguments")
        session.context.cancellation.raise_if_cancelled("react_tool")
        if repeated_empty_search(name, args, session.workspace):
            raise PlannerGuardError("empty_search_requires_different_plan_or_area")
        try:
            async with asyncio.timeout(session.timeout(reserve=20)):
                outcome = await tool.handler(args, session)
        except PlannerGuardError as error:
            details = edit_candidate_feedback(error.code, args, session.workspace)
            if details:
                raise ToolArgumentFailure(error.code, details) from None
            raise
        if outcome.result.get("missing") is True or (
            isinstance(outcome.result.get("observation"), dict)
            and outcome.result["observation"].get("status") in {"unavailable", "invalid_request"}
        ):
            # The observation still carries useful missing-data evidence. Persist
            # it atomically with its receipt, and let ToolNode return error status.
            outcome.result.update(ok=False, error="query_returned_no_verified_evidence")
        if empty_place_search(name, outcome.result):
            outcome.result["next_action"] = EMPTY_SEARCH_ACTION
        result = json.dumps(outcome.result, ensure_ascii=False, default=str)
        if len(result) > 150_000:
            raise PlannerGuardError("tool_result_too_large")

        async def complete(w: PlannerWorkspaceState) -> PlannerWorkspaceState:
            nonlocal result
            try:
                changed = outcome.apply(w) if outcome.apply is not None else w
                if inspect.isawaitable(changed):
                    changed = await changed
            except ValueError as error:
                code = (
                    error.code if isinstance(error, PlannerGuardError) else "tool_contract_invalid"
                )
                details = (
                    [
                        dict(item)
                        for item in error.errors(
                            include_url=False, include_input=False, include_context=False
                        )[:5]
                    ]
                    if isinstance(error, ValidationError)
                    else duplicate_candidate_issues(code, args) or None
                )
                raise ToolArgumentFailure(code, details) from None
            if (
                name in {"write_plan", "edit_plan", "patch_plan"}
                and changed.working_itinerary is not None
            ):
                result = json.dumps(
                    {
                        **outcome.result,
                        "saved_draft_revision": changed.working_itinerary.draft_revision,
                        "current_draft_ref": "current_draft",
                        "needs_current_review": True,
                    },
                    ensure_ascii=False,
                )
            state = react_memory(changed)
            done = receipt.model_copy(
                update={
                    "status": "failed" if outcome.result.get("ok") is False else "completed",
                    "result": result,
                }
            )
            return advance(
                changed,
                react_state=state.model_copy(
                    update={
                        "receipts": tuple(
                            done if r.call.id == receipt.call.id else r for r in state.receipts
                        )
                    }
                ),
            )

        await session.update(complete)
        return ToolMessage(
            content=result,
            name=name,
            tool_call_id=receipt.call.id,
            status="error" if outcome.result.get("ok") is False else "success",
        )

    async def _observe(self, state: LoopState, runtime: Runtime[LoopContext]) -> LoopState:
        ctx, session = runtime.context, runtime.context.session
        calls = self._messages(ctx)[-1].tool_calls
        messages = state["messages"]
        if len(messages) != len(calls) or any(not isinstance(m, ToolMessage) for m in messages):
            raise PlannerGuardError("native_tool_results_incomplete")
        results = {m.tool_call_id: m for m in messages if isinstance(m, ToolMessage)}
        if set(results) != {call.id for call in calls}:
            raise PlannerGuardError("native_tool_result_ids_mismatch")
        key = "messages" if ctx.source == "planner" else "reviewer_messages"
        await session.memory(
            **{
                key: (
                    *self._messages(ctx),
                    *(
                        ModelMessage(
                            role=ModelRole.TOOL,
                            tool_call_id=call.id,
                            content=cast(str, results[call.id].content),
                        )
                        for call in calls
                    ),
                )
            }
        )
        selected = [
            next(r for r in session.state.receipts if r.call.id == call.id) for call in calls
        ]
        if ctx.source == "planner" and all(
            json.loads(cast(str, result.result)).get("error")
            == "identical_failed_request_requires_changed_arguments"
            and sum(
                r.fingerprint == result.fingerprint
                and r.result is not None
                and json.loads(r.result).get("error")
                == "identical_failed_request_requires_changed_arguments"
                for r in session.state.receipts
            )
            >= 2
            for result in selected
        ):
            await session.memory(stop_reason="repeated_failed_action")
            return {"next": "end"}
        if ctx.source == "planner" and session.workspace.status is PlannerStatus.AWAITING_USER:
            return {"next": "await_user"}
        done = (
            session.workspace.status in {PlannerStatus.READY_TO_PUBLISH}
            if ctx.source == "planner"
            else not session.state.review_in_progress
        )
        return {"next": "end" if done else "decide"}
