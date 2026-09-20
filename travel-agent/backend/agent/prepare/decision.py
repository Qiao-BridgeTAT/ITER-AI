"""Prepare decision contract and non-checkpointed runtime dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Protocol
from zoneinfo import ZoneInfo

from sqlalchemy.ext.asyncio import AsyncSession

from backend.agent.model_gateway import ModelCancellation, ModelGateway
from backend.contracts.v4.prepare import PrepareDecision


class PrepareProviderRegistry(Protocol):
    """Runtime-only lookup boundary; normalized observations cross into State."""

    def supports(self, capability: str) -> bool: ...


@dataclass(frozen=True, slots=True)
class PrepareRuntimeContext:
    model_gateway: ModelGateway
    provider_registry: PrepareProviderRegistry
    cancellation: ModelCancellation
    user_id: str
    business_date: date
    timezone: ZoneInfo
    database_session: AsyncSession
    trace_id: str

    def __getstate__(self) -> None:
        raise TypeError(
            "PrepareRuntimeContext contains live dependencies and is not checkpointable"
        )


__all__ = ["PrepareDecision", "PrepareProviderRegistry", "PrepareRuntimeContext"]
