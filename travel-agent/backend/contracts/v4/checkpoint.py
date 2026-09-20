"""Recoverable V4 checkpoint envelope with no live runtime dependencies."""

from __future__ import annotations

from typing import Literal

from pydantic import AwareDatetime, Field, model_validator

from backend.contracts.v4.base import Identifier, V4ContractModel, require_unique
from backend.contracts.v4.conversation import ConversationEventV4, ConversationMessageV4
from backend.contracts.v4.planner_workspace import PlannerWorkspaceState
from backend.contracts.v4.state import V4TripStateEnvelope


class V4CheckpointEnvelope(V4ContractModel):
    checkpoint_version: Literal["v4-agent-1"] = "v4-agent-1"
    protocol_version: Literal["v4"] = "v4"
    trip_id: Identifier
    turn_id: Identifier
    trip_state_version: int = Field(ge=0, strict=True)
    state: V4TripStateEnvelope
    planner_workspace: PlannerWorkspaceState | None = None
    authoritative_messages: tuple[ConversationMessageV4, ...] = ()
    terminal_event: ConversationEventV4 | None = None
    outbox_cursor: Identifier | None = None
    updated_at: AwareDatetime

    @model_validator(mode="after")
    def checkpoint_is_owned_and_recoverable(self) -> V4CheckpointEnvelope:
        if self.state.semantic_state.trip_id != self.trip_id:
            raise ValueError("checkpoint state belongs to another trip")
        if self.state.semantic_state.state_version != self.trip_state_version:
            raise ValueError("checkpoint state version does not match its envelope")
        require_unique(
            (message.message_id for message in self.authoritative_messages),
            "checkpoint message_id",
        )
        if any(message.trip_id != self.trip_id for message in self.authoritative_messages):
            raise ValueError("checkpoint contains a message from another trip")
        if self.planner_workspace is not None:
            task_book_ref = self.state.semantic_state.confirmed_task_book_ref
            if task_book_ref is None:
                raise ValueError("Planner workspace requires a confirmed V4 task-book reference")
            if (
                self.planner_workspace.trip_id != self.trip_id
                or self.planner_workspace.based_on_task_book_id != task_book_ref.task_book_id
                or self.planner_workspace.based_on_task_book_version
                != task_book_ref.task_book_version
            ):
                raise ValueError("Planner workspace cannot change confirmed task-book identity")
        if self.terminal_event is not None:
            event = self.terminal_event.root
            if event.trip_id != self.trip_id or event.turn_id != self.turn_id:
                raise ValueError("checkpoint terminal event belongs to another turn")
            if event.event_type not in {
                "assistant.completed",
                "turn.cancelled",
                "turn.failed",
            }:
                raise ValueError("checkpoint terminal_event must be terminal")
        return self


V4_CHECKPOINT_CONTRACTS = (V4CheckpointEnvelope,)
