"""Bounded LangGraph orchestration for the deterministic V3 planning pipeline."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Protocol, TypedDict, cast
from uuid import UUID

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime

from backend.agent.model_gateway import ModelCancellation, ModelGatewayError
from backend.agent.task_book_state import SemanticTaskBook, SemanticTaskBookStatus
from backend.contracts.enums import TripPhase
from backend.contracts.state import TripState
from backend.planning.city_registry import default_city_registry

PLANNING_GRAPH_RECURSION_LIMIT = 16
PlanningProgressSink = Callable[[str, str], Awaitable[bool]]


@dataclass(frozen=True)
class PlanningNodeContract:
    name: str
    requires: tuple[str, ...]
    produces: str
    max_visits: int = 1


PLANNING_NODE_CONTRACTS = (
    PlanningNodeContract("candidate_recall", ("request",), "recall"),
    PlanningNodeContract("candidate_ranking", ("recall",), "ranking"),
    PlanningNodeContract("spatial_planning", ("ranking",), "spatial"),
    PlanningNodeContract("lodging_strategy", ("spatial",), "lodging"),
    PlanningNodeContract("hotel_selection", ("lodging",), "hotel"),
    PlanningNodeContract("daily_scheduling", ("spatial", "hotel"), "schedule"),
    PlanningNodeContract("cost_estimation", ("schedule",), "cost"),
    PlanningNodeContract("itinerary_validation", ("schedule", "cost"), "validation"),
    PlanningNodeContract("itinerary_repair", ("validation",), "repair"),
    PlanningNodeContract("plan_publication", ("repair",), "published_state"),
)
_CONTRACTS = {contract.name: contract for contract in PLANNING_NODE_CONTRACTS}
_PROGRESS_COPY = {
    "candidate_recall": "正在整理符合这次旅行的真实地点候选。",
    "candidate_ranking": "正在结合你的取舍筛选候选。",
    "spatial_planning": "正在检查地点分布和移动关系。",
    "lodging_strategy": "正在确定更合适的住宿范围。",
    "hotel_selection": "正在核对住宿候选。",
    "daily_scheduling": "正在安排每天的顺序和时间。",
    "cost_estimation": "正在汇总可追溯的费用估算。",
    "itinerary_validation": "正在检查营业、路线和时间冲突。",
    "itinerary_repair": "正在处理可以安全修复的问题。",
    "plan_publication": "正在保存完整且可恢复的行程版本。",
}


class PlanningGraphError(RuntimeError):
    """A safe stage-classified failure that never commits a partial state."""

    def __init__(self, stage: str, message: str) -> None:
        self.stage = stage
        super().__init__(message)


@dataclass(frozen=True)
class PlanningGraphRequest:
    state: TripState
    generation_id: UUID
    semantic_task_book: SemanticTaskBook

    def validate_boundary(self) -> None:
        task_book = self.state.task_book
        if task_book is None or task_book.status.value != "confirmed":
            raise PlanningGraphError(
                "planning_gate", "formal planning requires a confirmed task book"
            )
        if self.state.phase is not TripPhase.PLANNING:
            raise PlanningGraphError("planning_gate", "formal planning requires planning phase")
        if self.state.active_generation_id != self.generation_id:
            raise PlanningGraphError("planning_gate", "planning generation is not current")
        if (
            self.semantic_task_book.status is not SemanticTaskBookStatus.CONFIRMED
            or self.semantic_task_book.trip_id != self.state.trip_id
        ):
            raise PlanningGraphError(
                "planning_gate", "formal planning requires the current confirmed semantic task book"
            )
        state_city_id = self.state.city_id or (
            self.state.city.value if self.state.city is not None else None
        )
        canonical_city_id = (
            default_city_registry().resolve(state_city_id).city_id
            if state_city_id is not None
            else None
        )
        if self.semantic_task_book.destination.city_id != canonical_city_id:
            raise PlanningGraphError(
                "planning_gate", "semantic task book destination does not match the trip"
            )


@dataclass(frozen=True)
class PlanningGraphResult:
    published_state: TripState
    artifacts: Mapping[str, object]
    trace: tuple[str, ...]
    node_visits: Mapping[str, int]


class PlanningGraphBackend(Protocol):
    """Typed stage boundary; adapters call Providers and deterministic services here."""

    async def candidate_recall(
        self,
        request: PlanningGraphRequest,
        artifacts: Mapping[str, object],
        cancellation: ModelCancellation,
    ) -> object: ...

    async def candidate_ranking(
        self,
        request: PlanningGraphRequest,
        artifacts: Mapping[str, object],
        cancellation: ModelCancellation,
    ) -> object: ...

    async def spatial_planning(
        self,
        request: PlanningGraphRequest,
        artifacts: Mapping[str, object],
        cancellation: ModelCancellation,
    ) -> object: ...

    async def lodging_strategy(
        self,
        request: PlanningGraphRequest,
        artifacts: Mapping[str, object],
        cancellation: ModelCancellation,
    ) -> object: ...

    async def hotel_selection(
        self,
        request: PlanningGraphRequest,
        artifacts: Mapping[str, object],
        cancellation: ModelCancellation,
    ) -> object: ...

    async def daily_scheduling(
        self,
        request: PlanningGraphRequest,
        artifacts: Mapping[str, object],
        cancellation: ModelCancellation,
    ) -> object: ...

    async def cost_estimation(
        self,
        request: PlanningGraphRequest,
        artifacts: Mapping[str, object],
        cancellation: ModelCancellation,
    ) -> object: ...

    async def itinerary_validation(
        self,
        request: PlanningGraphRequest,
        artifacts: Mapping[str, object],
        cancellation: ModelCancellation,
    ) -> object: ...

    async def itinerary_repair(
        self,
        request: PlanningGraphRequest,
        artifacts: Mapping[str, object],
        cancellation: ModelCancellation,
    ) -> object: ...

    async def plan_publication(
        self,
        request: PlanningGraphRequest,
        artifacts: Mapping[str, object],
        cancellation: ModelCancellation,
    ) -> TripState: ...


@dataclass(frozen=True)
class PlanningGraphContext:
    cancellation: ModelCancellation
    emit_progress: PlanningProgressSink


class PlanningGraphState(TypedDict, total=False):
    request: PlanningGraphRequest
    artifacts: dict[str, object]
    trace: list[str]
    node_visits: dict[str, int]
    published_state: TripState


class V3PlanningGraph:
    """Execute every formal planning dependency once, with no stable partial writes."""

    def __init__(self, backend: PlanningGraphBackend) -> None:
        self._backend = backend
        builder = StateGraph(PlanningGraphState, context_schema=PlanningGraphContext)
        for contract in PLANNING_NODE_CONTRACTS:
            builder.add_node(contract.name, self._node(contract.name))
        builder.add_edge(START, PLANNING_NODE_CONTRACTS[0].name)
        for current, following in zip(
            PLANNING_NODE_CONTRACTS[:-1], PLANNING_NODE_CONTRACTS[1:], strict=True
        ):
            builder.add_edge(current.name, following.name)
        builder.add_edge(PLANNING_NODE_CONTRACTS[-1].name, END)
        self._compiled = builder.compile(name="v3-planning-orchestrator")

    @property
    def compiled_graph(
        self,
    ) -> CompiledStateGraph[
        PlanningGraphState,
        PlanningGraphContext,
        PlanningGraphState,
        PlanningGraphState,
    ]:
        return self._compiled

    async def invoke(
        self,
        request: PlanningGraphRequest,
        *,
        cancellation: ModelCancellation,
        emit_progress: PlanningProgressSink,
    ) -> PlanningGraphResult:
        request.validate_boundary()
        raw = cast(
            PlanningGraphState,
            await self._compiled.ainvoke(
                PlanningGraphState(
                    request=request,
                    artifacts={},
                    trace=[],
                    node_visits={},
                ),
                config={"recursion_limit": PLANNING_GRAPH_RECURSION_LIMIT},
                context=PlanningGraphContext(
                    cancellation=cancellation,
                    emit_progress=emit_progress,
                ),
            ),
        )
        published = raw.get("published_state")
        if published is None:
            raise PlanningGraphError("plan_publication", "planning ended without publication")
        return PlanningGraphResult(
            published_state=published,
            artifacts=dict(raw.get("artifacts", {})),
            trace=tuple(raw.get("trace", ())),
            node_visits=dict(raw.get("node_visits", {})),
        )

    def _node(self, stage: str):  # type: ignore[no-untyped-def]
        async def execute(
            state: PlanningGraphState,
            runtime: Runtime[PlanningGraphContext],
        ) -> PlanningGraphState:
            contract = _CONTRACTS[stage]
            request = state["request"]
            artifacts = dict(state.get("artifacts", {}))
            visits = dict(state.get("node_visits", {}))
            missing = [
                name for name in contract.requires if name != "request" and name not in artifacts
            ]
            if missing:
                raise PlanningGraphError(stage, f"planning node is missing inputs: {missing}")
            visits[stage] = visits.get(stage, 0) + 1
            if visits[stage] > contract.max_visits:
                raise PlanningGraphError(stage, "planning node visit limit exceeded")
            runtime.context.cancellation.raise_if_cancelled(stage)
            if not await runtime.context.emit_progress(stage, _PROGRESS_COPY[stage]):
                runtime.context.cancellation.cancel()
                runtime.context.cancellation.raise_if_cancelled(stage)
            method = getattr(self._backend, stage)
            try:
                output = await _await_cancellable(
                    method(request, artifacts, runtime.context.cancellation),
                    cancellation=runtime.context.cancellation,
                    operation=stage,
                )
            except PlanningGraphError:
                raise
            except ModelGatewayError:
                raise
            except Exception as exc:
                raise PlanningGraphError(stage, f"planning stage failed: {stage}") from exc
            runtime.context.cancellation.raise_if_cancelled(stage)
            artifacts[contract.produces] = output
            update: PlanningGraphState = {
                "artifacts": artifacts,
                "trace": [*state.get("trace", ()), stage],
                "node_visits": visits,
            }
            if stage == "plan_publication":
                if not isinstance(output, TripState):
                    raise PlanningGraphError(stage, "publication must return one TripState")
                published = TripState.model_validate(output.model_dump(mode="json"))
                if (
                    published.trip_id != request.state.trip_id
                    or published.state_version != request.state.state_version + 1
                    or published.phase is not TripPhase.DRAFT_READY
                    or published.published_plan is None
                    or published.published_plan.generation_id != request.generation_id
                ):
                    raise PlanningGraphError(
                        stage,
                        "publication must return one current complete stable plan",
                    )
                update["published_state"] = published
            return update

        return execute


async def _await_cancellable(
    awaitable: Awaitable[object],
    *,
    cancellation: ModelCancellation,
    operation: str,
) -> object:
    """Cancel the in-flight stage task as soon as the generation becomes stale."""

    stage_task: asyncio.Future[object] = asyncio.ensure_future(awaitable)
    cancelled_task = asyncio.create_task(cancellation.wait_cancelled())
    try:
        done, _pending = await asyncio.wait(
            (stage_task, cancelled_task),
            return_when=asyncio.FIRST_COMPLETED,
        )
        if cancelled_task in done:
            stage_task.cancel()
            await asyncio.gather(stage_task, return_exceptions=True)
            cancellation.raise_if_cancelled(operation)
        return await stage_task
    finally:
        cancelled_task.cancel()
        await asyncio.gather(cancelled_task, return_exceptions=True)
