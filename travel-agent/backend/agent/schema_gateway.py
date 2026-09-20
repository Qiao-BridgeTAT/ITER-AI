"""Use the Flash gateway's API schema contract for Prepare dining structured tasks."""

from collections.abc import AsyncIterator, Mapping
from typing import Any, TypeVar

from pydantic import BaseModel

from backend.agent.model_audit import ModelAuditRecorder
from backend.agent.model_gateway import (
    ModelCancellation,
    ModelGateway,
    ModelRequest,
    ModelRuntimeConfig,
    ModelStreamChunk,
    ModelStructuredResult,
)
from backend.discovery.cards.attraction_schema import attraction_schema

Output = TypeVar("Output", bound=BaseModel)


class SchemaModelGateway:
    def __init__(self, gateway: ModelGateway) -> None:
        self.gateway = gateway

    @property
    def config(self) -> ModelRuntimeConfig:
        return self.gateway.config

    @property
    def audit_recorder(self) -> ModelAuditRecorder | None:
        return getattr(self.gateway, "audit_recorder", None)

    async def generate_structured(
        self,
        request: ModelRequest,
        output_type: type[Output],
        *,
        cancellation: ModelCancellation | None = None,
        validation_context: Mapping[str, Any] | None = None,
    ) -> ModelStructuredResult[Output]:
        request = request.model_copy(
            update={
                "structured_output_mode": "json_schema",
                "output_schema_override": request.output_schema_override
                or attraction_schema(output_type),
                "thinking_budget_tokens": None,
                "reasoning_timeout_seconds": None,
            }
        )
        return await self.gateway.generate_structured(
            request, output_type, cancellation=cancellation, validation_context=validation_context
        )

    def stream_text(
        self, request: ModelRequest, *, cancellation: ModelCancellation | None = None
    ) -> AsyncIterator[ModelStreamChunk]:
        return self.gateway.stream_text(request, cancellation=cancellation)
