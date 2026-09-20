"""Route formal planning through the V3 graph in every conversation runtime."""

from __future__ import annotations

from uuid import UUID

from backend.application.realtime_command_service import (
    CommandExecutionContext,
    CommandExecutionResult,
    RealtimeCommandUseCase,
)
from backend.contracts.commands import TaskBookConfirmCommand


class PlanningDispatchJourneyUseCase:
    """Keep Replay/Fake conversation behavior while sharing one formal planner."""

    def __init__(
        self,
        conversation: RealtimeCommandUseCase,
        planning: RealtimeCommandUseCase,
    ) -> None:
        self._conversation = conversation
        self._planning = planning

    @property
    def formal_planning_graph_wired(self) -> bool:
        return bool(getattr(self._planning, "formal_planning_graph_wired", False))

    async def execute(self, context: CommandExecutionContext) -> CommandExecutionResult:
        target = (
            self._planning
            if isinstance(context.command, TaskBookConfirmCommand)
            else self._conversation
        )
        return await target.execute(context)

    async def cancel_generation(self, generation_id: UUID) -> None:
        for target in (self._planning, self._conversation):
            cancel = getattr(target, "cancel_generation", None)
            if cancel is not None:
                await cancel(generation_id)
