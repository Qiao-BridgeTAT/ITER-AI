"""Planner runtime dependencies deliberately excluded from checkpoint State."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Protocol
from zoneinfo import ZoneInfo

from sqlalchemy.ext.asyncio import AsyncSession

from backend.agent.model_gateway import ModelCancellation, ModelGateway


class PlannerProviderRegistry(Protocol):
    def supports(self, capability: str) -> bool: ...


@dataclass(frozen=True, slots=True)
class PlannerRuntimeContext:
    model_gateway: ModelGateway
    provider_registry: PlannerProviderRegistry
    cancellation: ModelCancellation
    user_id: str
    business_date: date
    timezone: ZoneInfo
    database_session: AsyncSession
    trace_id: str

    def __getstate__(self) -> None:
        raise TypeError(
            "PlannerRuntimeContext contains live dependencies and is not checkpointable"
        )


__all__ = ["PlannerProviderRegistry", "PlannerRuntimeContext"]
