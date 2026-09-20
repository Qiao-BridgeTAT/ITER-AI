"""Bounded LangGraph coordinator for the V2 semantic planning boundary."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Literal, TypedDict, TypeVar, cast
from uuid import UUID

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from pydantic import Field, model_validator

from backend.agent.model_gateway import ModelCancellation, ModelGateway
from backend.agent.readiness import (
    FinalSupplementResponseKind,
    ReadinessAssessment,
    ReadinessConcern,
    ReadinessError,
    ReadinessFailureCode,
    assess_readiness,
    record_critical_answer,
    record_final_supplement_response,
)
from backend.agent.readiness_state import CriticalQuestionResolutionProof, FinalSupplementStatus
from backend.agent.response_selection import (
    ExpressionCost,
    ResponseActionKind,
    ResponseCopy,
    ResponseSelection,
    ResponseSelectionContext,
    select_response,
)
from backend.agent.semantic_operations import SemanticOperationBatch
from backend.agent.semantic_understanding import (
    IntentCode,
    RelationClassification,
    SemanticInput,
    SemanticUnderstanding,
    classify_relation,
    extract_operations,
)
from backend.agent.state_merge import SemanticMergeResult, SemanticTripState, merge_state
from backend.agent.task_book import TaskBookBuildResult, build_task_book
from backend.contracts.base import ContractModel
from backend.contracts.common import NonEmptyText
from backend.contracts.conversation import (
    ConversationAttachment,
    ExternalFactReference,
    RecommendationSetAttachment,
)
from backend.contracts.enums import EvidenceSource

GRAPH_RECURSION_LIMIT = 12
RequiredValue = TypeVar("RequiredValue")


class AgentGraphError(RuntimeError):
    """Raised when the bounded graph violates its execution contract."""


@dataclass(frozen=True)
class AgentGraphContext:
    """Non-persisted runtime controls kept outside the shared business state."""

    cancellation: ModelCancellation | None = None


class GraphNodeContract(ContractModel):
    """Auditable input, output, and visit boundary for one graph node."""

    name: NonEmptyText
    required_inputs: tuple[NonEmptyText, ...]
    outputs: tuple[NonEmptyText, ...]
    max_visits: int = Field(default=1, ge=1, le=3, strict=True)


NODE_CONTRACTS = (
    GraphNodeContract(
        name="prepare_turn",
        required_inputs=("request", "semantic_state"),
        outputs=("trace", "node_visits"),
    ),
    GraphNodeContract(
        name="classify_relation",
        required_inputs=("request",),
        outputs=("classification", "trace", "node_visits"),
    ),
    GraphNodeContract(
        name="extract_operations",
        required_inputs=("request", "classification"),
        outputs=("understanding", "trace", "node_visits"),
    ),
    GraphNodeContract(
        name="merge_state",
        required_inputs=("request", "semantic_state"),
        outputs=("semantic_state", "merge_result", "trace", "node_visits"),
    ),
    GraphNodeContract(
        name="assess_readiness",
        required_inputs=("request", "semantic_state"),
        outputs=("semantic_state", "readiness", "trace", "node_visits"),
    ),
    GraphNodeContract(
        name="select_response",
        required_inputs=("request", "readiness"),
        outputs=("response", "trace", "node_visits"),
    ),
    GraphNodeContract(
        name="build_task_book",
        required_inputs=("request", "semantic_state", "response"),
        outputs=("semantic_state", "task_book_result", "trace", "node_visits"),
    ),
)
_NODE_CONTRACT_BY_NAME = {contract.name: contract for contract in NODE_CONTRACTS}


class AgentGraphRequest(ContractModel):
    """One user turn entering the graph through text or validated attachment operations."""

    initial_state: SemanticTripState
    business_date: date
    expected_state_version: int = Field(ge=0, strict=True)
    semantic_input: SemanticInput | None = None
    attachment_operations: SemanticOperationBatch | None = None
    attachment_question_resolution: CriticalQuestionResolutionProof | None = None
    prevalidated_user_text: str | None = Field(
        default=None,
        min_length=1,
        max_length=40_000,
        repr=False,
    )
    final_supplement_response: FinalSupplementResponseKind | None = None
    readiness_concerns: tuple[ReadinessConcern, ...] = ()
    direct_generation_requested: bool = False
    ordinary_reply: ResponseCopy | None = None
    user_question_answer: ResponseCopy | None = None
    clarification_attachment: ConversationAttachment | None = None
    clarification_external_facts: tuple[ExternalFactReference, ...] = ()
    attachment_for_question_id: UUID | None = None
    clarification_expression_cost: ExpressionCost = ExpressionCost.FREE_TEXT_EASIER
    recommendation: RecommendationSetAttachment | None = None
    recommendation_external_facts: tuple[ExternalFactReference, ...] = ()
    recommendation_requested_by_user: bool = False
    recommendation_needed_for_progress: bool = False

    @model_validator(mode="after")
    def one_owned_input_at_current_version(self) -> AgentGraphRequest:
        if (self.semantic_input is None) == (self.attachment_operations is None):
            raise ValueError("graph request requires exactly one text or attachment input")
        if self.expected_state_version != self.initial_state.state_version:
            raise ValueError("graph request state version is stale")
        if self.semantic_input is not None:
            if self.semantic_input.trip_id != self.initial_state.trip_id:
                raise ValueError("semantic input belongs to another trip")
            if self.semantic_input.business_date != self.business_date:
                raise ValueError("semantic input business date must match graph context")
        if (
            self.attachment_operations is not None
            and self.attachment_operations.trip_id != self.initial_state.trip_id
        ):
            raise ValueError("attachment operations belong to another trip")
        pending = self.initial_state.readiness.pending_critical_question
        if self.attachment_operations is not None:
            if (pending is None) != (self.attachment_question_resolution is None):
                raise ValueError(
                    "attachment input must include a resolution judgement exactly when "
                    "a question is pending"
                )
        elif self.attachment_question_resolution is not None:
            raise ValueError("text input receives question resolution from semantic extraction")
        if self.prevalidated_user_text is not None:
            if self.attachment_operations is None:
                raise ValueError("prevalidated user text requires semantic operations")
            if any(
                operation.evidence.source is not EvidenceSource.DIALOGUE
                for operation in self.attachment_operations.operations
            ):
                raise ValueError("prevalidated user text requires dialogue evidence")
        return self


class AgentGraphResult(ContractModel):
    state: SemanticTripState
    classification: RelationClassification | None = None
    understanding: SemanticUnderstanding | None = None
    merge_result: SemanticMergeResult | None = None
    readiness: ReadinessAssessment
    response: ResponseSelection
    task_book_result: TaskBookBuildResult | None = None
    trace: tuple[NonEmptyText, ...]
    node_visits: dict[NonEmptyText, int]


class AgentGraphState(TypedDict, total=False):
    request: AgentGraphRequest
    semantic_state: SemanticTripState
    classification: RelationClassification
    understanding: SemanticUnderstanding
    merge_result: SemanticMergeResult | None
    readiness: ReadinessAssessment
    response: ResponseSelection
    task_book_result: TaskBookBuildResult
    trace: list[str]
    node_visits: dict[str, int]


class V2AgentGraph:
    """Single shared-state graph covering V2 understanding through task-book projection."""

    def __init__(self, gateway: ModelGateway) -> None:
        self._gateway = gateway
        builder = StateGraph(AgentGraphState, context_schema=AgentGraphContext)
        builder.add_node("prepare_turn", self._prepare_turn)
        builder.add_node("classify_relation", self._classify_relation)
        builder.add_node("extract_operations", self._extract_operations)
        builder.add_node("merge_state", self._merge_state)
        builder.add_node("assess_readiness", self._assess_readiness)
        builder.add_node("select_response", self._select_response)
        builder.add_node("build_task_book", self._build_task_book)
        builder.add_edge(START, "prepare_turn")
        builder.add_conditional_edges(
            "prepare_turn",
            self._route_input,
            {"text": "classify_relation", "attachment": "merge_state"},
        )
        builder.add_edge("classify_relation", "extract_operations")
        builder.add_edge("extract_operations", "merge_state")
        builder.add_edge("merge_state", "assess_readiness")
        builder.add_edge("assess_readiness", "select_response")
        builder.add_conditional_edges(
            "select_response",
            self._route_response,
            {"task_book": "build_task_book", "done": END},
        )
        builder.add_edge("build_task_book", END)
        self._compiled = builder.compile(name="v2-agent-coordinator")

    @property
    def compiled_graph(
        self,
    ) -> CompiledStateGraph[
        AgentGraphState,
        AgentGraphContext,
        AgentGraphState,
        AgentGraphState,
    ]:
        return self._compiled

    async def invoke(
        self,
        request: AgentGraphRequest,
        *,
        cancellation: ModelCancellation | None = None,
    ) -> AgentGraphResult:
        """Run one bounded turn; cancellation is checked by model-backed nodes."""

        raw_result = cast(
            AgentGraphState,
            await self._compiled.ainvoke(
                AgentGraphState(
                    request=request,
                    semantic_state=request.initial_state.model_copy(deep=True),
                    trace=[],
                    node_visits={},
                ),
                config={"recursion_limit": GRAPH_RECURSION_LIMIT},
                context=AgentGraphContext(cancellation=cancellation),
            ),
        )
        return AgentGraphResult(
            state=_required(raw_result, "semantic_state", SemanticTripState),
            classification=raw_result.get("classification"),
            understanding=raw_result.get("understanding"),
            merge_result=raw_result.get("merge_result"),
            readiness=_required(raw_result, "readiness", ReadinessAssessment),
            response=_required(raw_result, "response", ResponseSelection),
            task_book_result=raw_result.get("task_book_result"),
            trace=tuple(raw_result.get("trace", ())),
            node_visits=raw_result.get("node_visits", {}),
        )

    async def _prepare_turn(self, state: AgentGraphState) -> AgentGraphState:
        return _visited(state, "prepare_turn")

    def _route_input(self, state: AgentGraphState) -> Literal["text", "attachment"]:
        request = state["request"]
        return "text" if request.semantic_input is not None else "attachment"

    async def _classify_relation(
        self,
        state: AgentGraphState,
        runtime: Runtime[AgentGraphContext],
    ) -> AgentGraphState:
        request = state["request"]
        semantic_input = _semantic_input_for_state(request.semantic_input, state["semantic_state"])
        if semantic_input is None:
            raise AgentGraphError("classification requires a semantic text input")
        classification = await classify_relation(
            self._gateway,
            semantic_input,
            cancellation=runtime.context.cancellation,
        )
        return {"classification": classification, **_visited(state, "classify_relation")}

    async def _extract_operations(
        self,
        state: AgentGraphState,
        runtime: Runtime[AgentGraphContext],
    ) -> AgentGraphState:
        request = state["request"]
        semantic_input = _semantic_input_for_state(request.semantic_input, state["semantic_state"])
        classification = state.get("classification")
        if semantic_input is None or classification is None:
            raise AgentGraphError("extraction requires text input and classification")
        understanding = await extract_operations(
            self._gateway,
            semantic_input,
            classification,
            cancellation=runtime.context.cancellation,
        )
        return {"understanding": understanding, **_visited(state, "extract_operations")}

    async def _merge_state(self, state: AgentGraphState) -> AgentGraphState:
        request = state["request"]
        semantic_state = state["semantic_state"]
        batch = request.attachment_operations
        understanding = state.get("understanding")
        if batch is None and understanding is not None and understanding.operations:
            batch = SemanticOperationBatch.model_validate(
                {
                    "trip_id": str(semantic_state.trip_id),
                    "operations": [
                        operation.model_dump(mode="json") for operation in understanding.operations
                    ],
                },
                context={
                    "today": request.business_date,
                    "allow_personal_defaults_write": (
                        request.semantic_input.allow_personal_defaults_write
                        if request.semantic_input is not None
                        else False
                    ),
                },
            )

        merge_result: SemanticMergeResult | None = None
        merged_operations = () if batch is None else tuple(batch.operations)
        if batch is not None:
            merge_result = merge_state(
                semantic_state,
                batch,
                expected_state_version=semantic_state.state_version,
                business_date=request.business_date,
                allow_personal_defaults_write=(
                    request.semantic_input.allow_personal_defaults_write
                    if request.semantic_input is not None
                    else False
                ),
            )
            semantic_state = merge_result.state

        pending = request.initial_state.readiness.pending_critical_question
        resolution_proof = request.attachment_question_resolution
        if resolution_proof is None and understanding is not None:
            resolution_proof = understanding.extraction.pending_question_resolution
        if pending is not None and resolution_proof is not None:
            semantic_input = request.semantic_input
            try:
                semantic_state = record_critical_answer(
                    semantic_state,
                    expected_state_version=semantic_state.state_version,
                    proof=resolution_proof,
                    answer_operations=merged_operations,
                    current_source_message_id=(
                        semantic_input.source_message_id
                        if semantic_input is not None
                        else resolution_proof.source_message_id
                    ),
                    current_user_text=(
                        semantic_input.user_text
                        if semantic_input is not None
                        else request.prevalidated_user_text
                    ),
                )
            except ReadinessError as error:
                if error.code not in {
                    ReadinessFailureCode.QUESTION_NOT_ANSWERED,
                    ReadinessFailureCode.QUESTION_EVIDENCE_INVALID,
                }:
                    raise

        final_supplement_response = request.final_supplement_response
        if final_supplement_response is None and understanding is not None:
            intent_codes = {
                understanding.classification.primary_intent.code,
                *(intent.code for intent in understanding.classification.secondary_intents),
            }
            if (
                request.initial_state.readiness.final_supplement_status
                is FinalSupplementStatus.AWAITING_RESPONSE
                and not understanding.operations
                and intent_codes
                & {
                    IntentCode.CONFIRM_UNDERSTANDING,
                    IntentCode.NO_PREFERENCE,
                    IntentCode.START_PLANNING,
                    IntentCode.CONTINUE,
                }
            ):
                final_supplement_response = FinalSupplementResponseKind.NO_MORE_INFORMATION

        if final_supplement_response is not None:
            transition = record_final_supplement_response(
                semantic_state,
                expected_state_version=semantic_state.state_version,
                response=final_supplement_response,
            )
            semantic_state = transition.state

        return {
            "semantic_state": semantic_state,
            "merge_result": merge_result,
            **_visited(state, "merge_state"),
        }

    async def _assess_readiness(self, state: AgentGraphState) -> AgentGraphState:
        request = state["request"]
        assessment = assess_readiness(
            state["semantic_state"],
            expected_state_version=state["semantic_state"].state_version,
            concerns=request.readiness_concerns,
            direct_generation_requested=request.direct_generation_requested,
        )
        return {
            "semantic_state": assessment.state,
            "readiness": assessment,
            **_visited(state, "assess_readiness"),
        }

    async def _select_response(self, state: AgentGraphState) -> AgentGraphState:
        request = state["request"]
        ordinary_reply = request.ordinary_reply or _social_acknowledgement(
            state.get("classification")
        )
        response = select_response(
            ResponseSelectionContext(
                readiness=state["readiness"],
                ordinary_reply=ordinary_reply,
                user_question_answer=request.user_question_answer,
                clarification_attachment=request.clarification_attachment,
                clarification_external_facts=request.clarification_external_facts,
                attachment_for_question_id=request.attachment_for_question_id,
                clarification_expression_cost=request.clarification_expression_cost,
                recommendation=request.recommendation,
                recommendation_external_facts=request.recommendation_external_facts,
                recommendation_requested_by_user=request.recommendation_requested_by_user,
                recommendation_needed_for_progress=request.recommendation_needed_for_progress,
            )
        )
        return {"response": response, **_visited(state, "select_response")}

    def _route_response(self, state: AgentGraphState) -> Literal["task_book", "done"]:
        return "task_book" if state["response"].action is ResponseActionKind.TASK_BOOK else "done"

    async def _build_task_book(self, state: AgentGraphState) -> AgentGraphState:
        request = state["request"]
        result = build_task_book(
            state["semantic_state"],
            expected_state_version=state["semantic_state"].state_version,
            business_date=request.business_date,
        )
        return {
            "semantic_state": result.state,
            "task_book_result": result,
            **_visited(state, "build_task_book"),
        }


def _visited(state: AgentGraphState, node_name: str) -> AgentGraphState:
    contract = _NODE_CONTRACT_BY_NAME[node_name]
    missing = [name for name in contract.required_inputs if name not in state]
    if missing:
        raise AgentGraphError(f"{node_name} missing required inputs: {', '.join(missing)}")
    visits = dict(state.get("node_visits", {}))
    visits[node_name] = visits.get(node_name, 0) + 1
    if visits[node_name] > contract.max_visits:
        raise AgentGraphError(f"{node_name} exceeded its bounded visit count")
    return {"trace": [*state.get("trace", []), node_name], "node_visits": visits}


def _required(
    result: Mapping[str, object],
    key: str,
    expected_type: type[RequiredValue],
) -> RequiredValue:
    value = result.get(key)
    if not isinstance(value, expected_type):
        raise AgentGraphError(f"graph result is missing {key}")
    return value


def _semantic_input_for_state(
    semantic_input: SemanticInput | None,
    state: SemanticTripState,
) -> SemanticInput | None:
    if semantic_input is None:
        return None
    pending = state.readiness.pending_critical_question
    return semantic_input.model_copy(
        update={
            "pending_question": pending.prompt if pending is not None else None,
            "pending_question_id": pending.question_id if pending is not None else None,
            "pending_question_resolution_goal": (
                pending.resolution_goal if pending is not None else None
            ),
        },
        deep=True,
    )


def _social_acknowledgement(
    classification: RelationClassification | None,
) -> ResponseCopy | None:
    if classification is None:
        return None
    code = classification.primary_intent.code
    copy_by_code = {
        IntentCode.GREETING: "你好，我们来一起规划这次旅行。",
        IntentCode.THANKS: "不客气。",
        IntentCode.EMOTION: "我听到了。",
        IntentCode.CHAT: "我明白你的意思了。",
        IntentCode.OUT_OF_SCOPE: "这条信息暂时不属于本次旅行规划。",
        IntentCode.UNCLEAR: "我还没理解这句话具体想表达什么。",
        IntentCode.INPUT_NOISE: "我还没有从这句话里识别到明确的旅行信息。",
        IntentCode.TEST_OR_PROBE: "我正在正常工作，可以直接告诉我旅行需求。",
    }
    text = copy_by_code.get(code)
    return ResponseCopy(text=text) if text is not None else None
