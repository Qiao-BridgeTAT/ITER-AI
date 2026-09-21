"""Versioned, serializable ReAct working memory and public progress contracts."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import AwareDatetime, Field, model_validator

from backend.agent.model_gateway import ModelMessage, ModelToolCall
from backend.contracts.v4.base import Identifier, V4ContractModel
from backend.contracts.v4.memory import UserMemoryView

REACT_ENGINE: Literal["langgraph-react-2"] = "langgraph-react-2"
REACT_CHECKPOINT: Literal["v4-planner-react-2"] = "v4-planner-react-2"
MAX_EFFECTIVE_REVISIONS = 6
PLANNER_EXECUTION_SECONDS = 480
DEFAULT_PLANNER_CALL_LIMIT = 24


class ReviewIssue(V4ContractModel):
    code: str = Field(min_length=1, max_length=100)
    severity: Literal["warning", "error", "blocking"] = Field(
        description=(
            "warning：允许明确披露的事实缺失或非阻断建议。error/blocking：具体的已知冲突、"
            "用户要求违反或实质排程风险，需说明依据。营业未知不等于已证实闭馆；"
            "参考房价不等于已证实超预算，不能仅以未知为由判定硬冲突。"
        )
    )
    target: str = Field(min_length=1, max_length=256)
    evidence_refs: tuple[str, ...] = ()
    description: str = Field(min_length=1, max_length=1000)
    suggestion: str = Field(min_length=1, max_length=1000)


class ReviewVerdict(V4ContractModel):
    accepted: bool = Field(
        description=(
            "根据当前实际时间轴、任务要求、事实和程序报告作出独立结论。"
            "只有明确披露的允许缺口及建议时可以通过；存在真实冲突或实质质量问题时拒绝。"
        )
    )
    summary: str = Field(min_length=1, max_length=1500)
    issues: tuple[ReviewIssue, ...] = Field(default=(), max_length=30)

    @model_validator(mode="after")
    def no_approval_with_errors(self) -> ReviewVerdict:
        if self.accepted and any(issue.severity != "warning" for issue in self.issues):
            raise ValueError("review with errors cannot approve a draft")
        return self


class BoundReview(V4ContractModel):
    draft_revision: int = Field(ge=1)
    draft_digest: str
    evidence_digest: str
    validation_fingerprint: str
    verdict: ReviewVerdict
    reviewed_at: AwareDatetime


class ToolReceipt(V4ContractModel):
    call: ModelToolCall
    fingerprint: str
    status: Literal["pending", "completed", "failed"] = "pending"
    result: str | None = Field(default=None, max_length=150_000)
    expires_at: AwareDatetime | None = None
    read_only: bool
    source: Literal["planner", "reviewer"] = "planner"


class PlannerReactState(V4ContractModel):
    native_interrupt_checkpoint: dict[str, Any] | None = None
    resumed_interaction_id: str | None = None

    engine_version: Literal["langgraph-react-2"] = REACT_ENGINE
    checkpoint_version: Literal["v4-planner-react-2"] = REACT_CHECKPOINT
    # Additive checkpoint migration: absent means the original native protocol.
    # New runs explicitly pin json_schema; recovery never repins the mode.
    dialogue_mode: Literal["native_tools", "json_schema"] = "native_tools"
    # Keep the nominal deadline for segment identity/elapsed-time auditing.
    # Old checkpoints remain bounded; only explicitly configured new runs opt out.
    deadline_at: AwareDatetime
    time_limit_disabled: bool = False
    route_refresh_only: bool = False
    long_term_memories: tuple[UserMemoryView, ...] = ()
    planner_call_limit: int = Field(default=12, ge=1, le=24)
    planner_calls: int = Field(default=0, ge=0, le=24)
    external_requests: int = Field(default=0, ge=0, le=80)
    web_search_calls: int = Field(default=0, ge=0, le=4)
    web_search_blocked: bool = False
    effective_revisions: int = Field(default=0, ge=0, le=MAX_EFFECTIVE_REVISIONS)
    messages: tuple[ModelMessage, ...] = ()
    receipts: tuple[ToolReceipt, ...] = ()
    review: BoundReview | None = None
    reviewer_messages: tuple[ModelMessage, ...] = ()
    reviewer_query_rounds: int = Field(default=0, ge=0, le=2)
    reviewer_batches: tuple[str, ...] = ()
    reviewer_calls: int = Field(default=0, ge=0, le=5)
    review_in_progress: bool = False
    stop_reason: str | None = None

    @model_validator(mode="after")
    def decisions_within_admitted_limit(self) -> PlannerReactState:
        if self.planner_calls > self.planner_call_limit:
            raise ValueError("planner decisions exceed this run's admitted limit")
        return self


class AgentProgressEntry(V4ContractModel):
    event_id: Identifier
    trip_id: Identifier
    turn_id: Identifier
    generation_id: Identifier
    progress_index: int = Field(ge=1, strict=True)
    source: Literal["planner", "reviewer", "tool", "runtime"]
    text: str = Field(min_length=1, max_length=1000)
    emitted_at: AwareDatetime
