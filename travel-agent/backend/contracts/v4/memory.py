"""Only user-authored, explicitly saved long-term memory is accepted."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import AwareDatetime, Field, model_validator

from backend.contracts.v4.base import V4ContractModel


class CreateUserMemory(V4ContractModel):
    kind: Literal["preference", "feedback"]
    text: str = Field(min_length=1, max_length=2000)
    trip_id: UUID | None = None
    source_message_id: UUID | None = None
    explicitly_confirmed: Literal[True]

    @model_validator(mode="after")
    def feedback_has_source(self) -> CreateUserMemory:
        if self.kind == "feedback" and (self.trip_id is None or self.source_message_id is None):
            raise ValueError("trip feedback requires its original user message")
        if self.source_message_id is not None and self.trip_id is None:
            raise ValueError("message source requires a trip")
        return self


class UserMemoryView(V4ContractModel):
    memory_id: UUID
    kind: Literal["preference", "feedback"]
    text: str
    trip_id: UUID | None = None
    source_message_id: UUID | None = None
    created_at: AwareDatetime


class UserMemoryList(V4ContractModel):
    memories: tuple[UserMemoryView, ...] = ()
