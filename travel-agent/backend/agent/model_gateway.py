"""Vendor-neutral model generation boundary used by Agent application code."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Generic, Literal, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelContract(BaseModel):
    """Strict immutable contract that never accepts vendor response fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ModelToolFunction(ModelContract):
    name: str = Field(min_length=1, max_length=128)
    arguments: str = Field(default="{}", max_length=131_072, repr=False)


class ModelToolCall(ModelContract):
    id: str = Field(min_length=1, max_length=256)
    type: Literal["function"] = "function"
    function: ModelToolFunction


class ModelToolDefinition(ModelContract):
    name: str = Field(min_length=1, max_length=128)
    description: str = Field(min_length=1, max_length=8_192)
    parameters: dict[str, Any]


class ModelToolDelta(ModelContract):
    index: int = Field(ge=0, le=79, strict=True)
    id: str | None = None
    name: str = ""
    arguments: str = Field(default="", repr=False)


class ModelMessage(ModelContract):
    role: ModelRole
    content: str = Field(default="", repr=False)
    tool_calls: tuple[ModelToolCall, ...] = Field(default=(), exclude_if=lambda v: not v)
    tool_call_id: str | None = Field(default=None, exclude_if=lambda v: v is None)

    @model_validator(mode="after")
    def validate_tool_message(self) -> ModelMessage:
        if self.role is ModelRole.TOOL:
            if not self.tool_call_id or self.tool_calls or not self.content:
                raise ValueError("tool result requires content and tool_call_id")
        elif self.tool_call_id is not None:
            raise ValueError("only tool results may contain tool_call_id")
        if self.tool_calls and self.role is not ModelRole.ASSISTANT:
            raise ValueError("only assistant may request tools")
        if not self.content and not self.tool_calls:
            raise ValueError("message must contain content or tool calls")
        return self


class ModelAuditMetadata(ModelContract):
    """Internal call identity; excluded from the provider HTTP request."""

    stage: str = Field(min_length=1, max_length=96)
    node: str | None = Field(default=None, max_length=96)
    contract_version: str | None = Field(default=None, max_length=96)
    repair: bool = False
    attempt: int | None = Field(default=None, ge=1, le=100, strict=True)
    parent_llm_call_id: str | None = Field(default=None, max_length=64)
    repair_of_call_id: str | None = Field(default=None, max_length=64)
    model_switch_from: str | None = Field(default=None, max_length=128)
    model_switch_reason: str | None = Field(default=None, max_length=500)


class ModelRequest(ModelContract):
    messages: list[ModelMessage] = Field(min_length=1, repr=False)
    max_output_tokens: int = Field(default=1_024, ge=1, le=32_768, strict=True)
    structured_output_mode: Literal["json_schema", "json_object"] = "json_schema"
    output_schema_override: dict[str, Any] | None = Field(default=None, repr=False)
    request_timeout_seconds: int | None = Field(default=None, ge=1, le=300, strict=True)
    temperature_override: float | None = Field(default=None, ge=0.0, le=2.0)
    # Opt-in for complex structured decisions only; never used for public prose.
    thinking_budget_tokens: int | None = Field(default=None, ge=1, le=8_192, strict=True)
    reasoning_timeout_seconds: int | None = Field(default=None, ge=1, le=300, strict=True)
    audit: ModelAuditMetadata | None = Field(default=None, exclude=True, repr=False)
    tools: tuple[ModelToolDefinition, ...] = Field(default=(), repr=False)
    parallel_tool_calls: bool = True
    forced_tool_name: str | None = None

    @model_validator(mode="after")
    def creative_temperature_requires_json_object(self) -> ModelRequest:
        if self.tools and self.thinking_budget_tokens is not None:
            raise ValueError("native tool dialogue uses non-thinking mode")
        if len({tool.name for tool in self.tools}) != len(self.tools):
            raise ValueError("tool names must be unique")
        if self.forced_tool_name is not None and self.forced_tool_name not in {
            tool.name for tool in self.tools
        }:
            raise ValueError("forced tool must be offered in this request")
        pending: set[str] = set()
        used: set[str] = set()
        for message in self.messages:
            if message.role is ModelRole.TOOL:
                if message.tool_call_id not in pending:
                    raise ValueError("tool result must match one pending call")
                pending.remove(message.tool_call_id)
            else:
                if pending:
                    raise ValueError("all tool results must precede the next message")
                for call in message.tool_calls:
                    if call.id in used:
                        raise ValueError("tool call ids must be unique")
                    used.add(call.id)
                    pending.add(call.id)
        if pending:
            raise ValueError("cannot call model before pending tool results are supplied")
        if self.thinking_budget_tokens is not None and (
            self.structured_output_mode != "json_object"
            or self.thinking_budget_tokens >= self.max_output_tokens
        ):
            raise ValueError("bounded thinking requires json_object and an answer token allowance")
        if self.reasoning_timeout_seconds is not None and self.thinking_budget_tokens is None:
            raise ValueError("reasoning timeout may only override bounded thinking requests")
        return self


class ModelUsage(ModelContract):
    prompt_tokens: int = Field(ge=0, strict=True)
    completion_tokens: int = Field(ge=0, strict=True)
    total_tokens: int = Field(ge=0, strict=True)


class ModelStreamChunk(ModelContract):
    delta: str = ""
    finish_reason: str | None = None
    usage: ModelUsage | None = None
    audit_call_id: str | None = Field(default=None, exclude=True, repr=False)
    tool_call_deltas: tuple[ModelToolDelta, ...] = ()


StructuredValue = TypeVar("StructuredValue", bound=BaseModel)


class ModelStructuredResult(ModelContract, Generic[StructuredValue]):
    value: StructuredValue
    usage: ModelUsage | None = None
    audit_call_id: str | None = Field(default=None, exclude=True, repr=False)


class ModelFailureCode(StrEnum):
    DISABLED = "disabled"
    AUTHENTICATION_FAILED = "authentication_failed"
    PERMISSION_DENIED = "permission_denied"
    BILLING_UNAVAILABLE = "billing_unavailable"
    RATE_LIMITED = "rate_limited"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    MALFORMED_RESPONSE = "malformed_response"
    UNAVAILABLE = "unavailable"
    UPSTREAM_ERROR = "upstream_error"
    AUDIT_UNAVAILABLE = "audit_unavailable"


class ModelGatewayError(RuntimeError):
    """Safe classified model failure without prompts, user text or credentials."""

    def __init__(
        self,
        code: ModelFailureCode,
        operation: str,
        *,
        retryable: bool,
        validation_issues: tuple[str, ...] = (),
        audit_call_id: str | None = None,
    ) -> None:
        self.code = code
        self.operation = operation
        self.retryable = retryable
        self.validation_issues = validation_issues
        self.audit_call_id = audit_call_id
        super().__init__(f"model {operation} failed: {code.value}")

    @property
    def requires_runtime_recovery(self) -> bool:
        """These failures cannot be repaired by changing an LLM's output."""
        return self.code in {
            ModelFailureCode.DISABLED,
            ModelFailureCode.AUTHENTICATION_FAILED,
            ModelFailureCode.PERMISSION_DENIED,
            ModelFailureCode.BILLING_UNAVAILABLE,
            ModelFailureCode.AUDIT_UNAVAILABLE,
        }


@dataclass(frozen=True)
class ModelRuntimeConfig:
    """Versioned runtime choices kept outside prompts and business services."""

    model_name: str
    temperature: float
    timeout_seconds: float
    prompt_version: str


class ModelCancellation:
    """Cooperative cancellation token shared across gateway implementations."""

    def __init__(self) -> None:
        self._event = asyncio.Event()

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self) -> None:
        self._event.set()

    async def wait_cancelled(self) -> None:
        await self._event.wait()

    def raise_if_cancelled(self, operation: str) -> None:
        if self.is_cancelled:
            raise ModelGatewayError(
                ModelFailureCode.CANCELLED,
                operation,
                retryable=False,
            )


class ModelGateway(Protocol):
    """Only model capability visible to Agent application services."""

    @property
    def config(self) -> ModelRuntimeConfig: ...

    def stream_text(
        self,
        request: ModelRequest,
        *,
        cancellation: ModelCancellation | None = None,
    ) -> AsyncIterator[ModelStreamChunk]: ...

    async def generate_structured(
        self,
        request: ModelRequest,
        output_type: type[StructuredValue],
        *,
        cancellation: ModelCancellation | None = None,
        validation_context: Mapping[str, Any] | None = None,
    ) -> ModelStructuredResult[StructuredValue]: ...


class DisabledModelGateway:
    """Safe no-network gateway for local and CI environments without credentials."""

    def __init__(self, config: ModelRuntimeConfig) -> None:
        self._config = config

    @property
    def config(self) -> ModelRuntimeConfig:
        return self._config

    async def stream_text(
        self,
        request: ModelRequest,
        *,
        cancellation: ModelCancellation | None = None,
    ) -> AsyncIterator[ModelStreamChunk]:
        del request, cancellation
        raise ModelGatewayError(ModelFailureCode.DISABLED, "stream_text", retryable=False)
        yield  # pragma: no cover - keeps this method an async iterator

    async def generate_structured(
        self,
        request: ModelRequest,
        output_type: type[StructuredValue],
        *,
        cancellation: ModelCancellation | None = None,
        validation_context: Mapping[str, Any] | None = None,
    ) -> ModelStructuredResult[StructuredValue]:
        del request, output_type, cancellation, validation_context
        raise ModelGatewayError(ModelFailureCode.DISABLED, "structured", retryable=False)
