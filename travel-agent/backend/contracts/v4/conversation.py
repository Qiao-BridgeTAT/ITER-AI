"""V4 committed conversation messages, events, and refresh snapshot."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import AwareDatetime, Field, RootModel, model_validator

from backend.contracts.v4.base import (
    Digest,
    DisplayText,
    Identifier,
    V4ContractModel,
    require_unique,
)
from backend.contracts.v4.cards import (
    AttractionPreferenceCard,
    DiningPreferenceCard,
    LodgingAreaPreferenceCard,
    LodgingClassPreferenceCard,
    SpecificCandidateCard,
)
from backend.contracts.v4.enums import AskUserReasonCode, InteractionStatus, PlannerStatus
from backend.contracts.v4.planner_observations import PlannerInteractionOption
from backend.contracts.v4.planner_publication import PlannerPublishedPlan
from backend.contracts.v4.state import PendingInteraction, V4TripStateEnvelope
from backend.contracts.v4.task_book import TaskBookV4

V4AttachmentValue = (
    AttractionPreferenceCard
    | DiningPreferenceCard
    | LodgingAreaPreferenceCard
    | LodgingClassPreferenceCard
    | SpecificCandidateCard
    | TaskBookV4
    | PlannerPublishedPlan
)


class V4Attachment(RootModel[V4AttachmentValue]):
    pass


class ConversationMessageV4(V4ContractModel):
    message_id: Identifier
    trip_id: Identifier
    turn_id: Identifier
    generation_id: Identifier
    role: Literal["user", "assistant", "system"]
    message_type: Literal["text", "card", "task_book", "plan", "status"]
    text: DisplayText
    state_version: int = Field(ge=0, strict=True)
    generation_mode: Literal["user", "qwen", "fallback", "system"]
    content_hash: Digest
    attachments: list[V4Attachment] = Field(default_factory=list)
    status: Literal["accepted", "processed", "committed"]
    created_at: AwareDatetime

    @model_validator(mode="after")
    def message_mode_matches_role(self) -> ConversationMessageV4:
        if self.role == "user" and self.generation_mode != "user":
            raise ValueError("user message requires user generation_mode")
        if self.role == "assistant" and self.generation_mode not in {"qwen", "fallback"}:
            raise ValueError("assistant message requires qwen or fallback generation_mode")
        if self.status != "committed" and self.role == "assistant":
            raise ValueError("assistant messages are user-visible only after commit")
        return self


class EventBase(V4ContractModel):
    event_id: Identifier
    event_type: str
    protocol_version: Literal["v4"] = "v4"
    trip_id: Identifier
    turn_id: Identifier
    generation_id: Identifier
    sequence: int = Field(ge=0, strict=True)
    emitted_at: AwareDatetime


class TurnAcceptedEvent(EventBase):
    event_type: Literal["turn.accepted"]
    base_state_version: int = Field(ge=0, strict=True)


class AgentStatusEvent(EventBase):
    event_type: Literal["agent.status"]
    status_code: Identifier
    message: DisplayText


class StateCommittedEvent(EventBase):
    event_type: Literal["state.committed"]
    message_id: Identifier
    state_version: int = Field(ge=1, strict=True)
    base_state_version: int = Field(ge=0, strict=True)
    committed_state_version: int = Field(ge=1, strict=True)
    invalidated_interaction_ids: list[Identifier] = Field(default_factory=list)

    @model_validator(mode="after")
    def state_versions_advance_once(self) -> StateCommittedEvent:
        if self.committed_state_version != self.base_state_version + 1:
            raise ValueError("state commit event must advance the version exactly once")
        if self.state_version != self.committed_state_version:
            raise ValueError("event state_version must equal committed_state_version")
        return self


class AssistantStartedEvent(EventBase):
    event_type: Literal["assistant.started"]
    message_id: Identifier
    state_version: int = Field(ge=1, strict=True)


class AssistantDeltaEvent(EventBase):
    event_type: Literal["assistant.delta"]
    message_id: Identifier
    state_version: int = Field(ge=1, strict=True)
    chunk_index: int = Field(ge=0, strict=True)
    delta: str = Field(min_length=1)
    content_hash: Digest


class AttachmentReadyEvent(EventBase):
    event_type: Literal["attachment.ready"]
    message_id: Identifier
    state_version: int = Field(ge=1, strict=True)
    attachment_id: Identifier
    interaction_id: Identifier | None = None


class AssistantCompletedEvent(EventBase):
    event_type: Literal["assistant.completed"]
    message_id: Identifier
    state_version: int = Field(ge=1, strict=True)
    content_hash: Digest
    generation_mode: Literal["qwen", "fallback"]
    outbox_cursor: Identifier
    outcome: Literal[
        "answered",
        "card_ready",
        "task_book_ready",
        "plan_ready",
        "awaiting_user",
    ]


class TurnCancelledEvent(EventBase):
    event_type: Literal["turn.cancelled"]
    failure_code: Identifier


class TurnFailedEvent(EventBase):
    event_type: Literal["turn.failed"]
    failure_code: Identifier
    retryable: bool


ConversationEventValue = Annotated[
    TurnAcceptedEvent
    | AgentStatusEvent
    | StateCommittedEvent
    | AssistantStartedEvent
    | AssistantDeltaEvent
    | AttachmentReadyEvent
    | AssistantCompletedEvent
    | TurnCancelledEvent
    | TurnFailedEvent,
    Field(discriminator="event_type"),
]


class ConversationEventV4(RootModel[ConversationEventValue]):
    pass


class PlannerInteractionPublicView(V4ContractModel):
    """Only fields required to render and answer the current user decision."""

    interaction_id: Identifier
    reason_code: AskUserReasonCode
    option_contracts: tuple[PlannerInteractionOption, ...] = Field(min_length=1)
    resume_token: Identifier
    status: InteractionStatus


class PlannerWorkspacePublicView(V4ContractModel):
    """Minimal resumable Planner state safe to expose to the browser."""

    trip_id: Identifier
    generation_id: Identifier
    based_on_task_book_id: Identifier
    based_on_task_book_version: int = Field(ge=1, strict=True)
    workspace_revision: int = Field(ge=0, strict=True)
    last_interaction_action: (
        Literal[
            "keep_task_book",
            "revise_task_book",
            "supply_booking_detail",
        ]
        | None
    ) = None
    active_interaction: PlannerInteractionPublicView | None = None
    status: PlannerStatus


class ConversationSnapshotV4(V4ContractModel):
    trip_state: V4TripStateEnvelope
    messages: list[ConversationMessageV4] = Field(default_factory=list)
    pending_interaction: PendingInteraction | None = None
    terminal_event: ConversationEventV4 | None = None
    last_outbox_cursor: Identifier | None = None
    active_generation_id: Identifier | None = None
    planner_workspace: PlannerWorkspacePublicView | None = None
    snapshot_at: AwareDatetime

    @model_validator(mode="after")
    def snapshot_is_authoritative_and_ordered(self) -> ConversationSnapshotV4:
        trip_id = self.trip_state.semantic_state.trip_id
        if self.planner_workspace is not None:
            planner = self.planner_workspace
            book = self.trip_state.discovery_runtime_state.task_book_candidate
            confirmed = self.trip_state.semantic_state.confirmed_task_book_ref
            if (
                planner.trip_id != trip_id
                or book is None
                or confirmed is None
                or planner.based_on_task_book_id != confirmed.task_book_id
                or planner.based_on_task_book_version != confirmed.task_book_version
            ):
                raise ValueError(
                    "snapshot Planner workspace must belong to current confirmed task book"
                )
        if any(message.trip_id != trip_id for message in self.messages):
            raise ValueError("snapshot contains a message from another trip")
        require_unique((message.message_id for message in self.messages), "message_id")
        state_versions = [message.state_version for message in self.messages]
        if state_versions != sorted(state_versions):
            raise ValueError("snapshot messages must be ordered by committed state version")
        if any(
            version > self.trip_state.semantic_state.state_version for version in state_versions
        ):
            raise ValueError("snapshot message cannot be newer than trip state")
        if self.pending_interaction != self.trip_state.discovery_runtime_state.pending_interaction:
            raise ValueError("snapshot pending interaction must match runtime state")
        if self.terminal_event is not None:
            event = self.terminal_event.root
            if event.trip_id != trip_id:
                raise ValueError("snapshot terminal event belongs to another trip")
            if event.event_type not in {
                "assistant.completed",
                "turn.cancelled",
                "turn.failed",
            }:
                raise ValueError("snapshot terminal_event must be terminal")
            if isinstance(event, AssistantCompletedEvent):
                if event.state_version != self.trip_state.semantic_state.state_version:
                    raise ValueError("completed event must describe the current committed state")
                matching_messages = [
                    message for message in self.messages if message.message_id == event.message_id
                ]
                if len(matching_messages) != 1:
                    raise ValueError("completed event must reference one committed message")
                message = matching_messages[0]
                if (
                    message.content_hash != event.content_hash
                    or message.state_version != event.state_version
                    or message.generation_id != event.generation_id
                ):
                    raise ValueError("completed event does not match its committed message")
                if (
                    self.last_outbox_cursor is not None
                    and event.outbox_cursor != self.last_outbox_cursor
                ):
                    raise ValueError("completed event and snapshot outbox cursor must match")
        return self


class ConversationHistoryWindow(V4ContractModel):
    """Read-only UI pagination; hashes still identify ORIGINAL committed messages.

    The listed messages omit attachments in this projection only. Their complete
    immutable content remains available from the authorized message endpoint.
    Neither this projection nor its empty attachment lists may enter Agent state.
    """

    through_state_version: int = Field(ge=0, strict=True)
    before_state_version: int | None = Field(default=None, ge=0, strict=True)
    deferred_attachment_message_ids: list[Identifier] = Field(default_factory=list)


class ConversationHistoryPage(V4ContractModel):
    trip_id: Identifier
    messages: list[ConversationMessageV4] = Field(default_factory=list)
    history: ConversationHistoryWindow


class ConversationView(V4ContractModel):
    """Explicit presentation view, separate from the full recovery/Agent contract."""

    snapshot: ConversationSnapshotV4
    history: ConversationHistoryWindow


V4_CONVERSATION_CONTRACTS = (
    ConversationMessageV4,
    ConversationEventV4,
    PlannerInteractionPublicView,
    PlannerWorkspacePublicView,
    ConversationSnapshotV4,
    ConversationHistoryWindow,
    ConversationHistoryPage,
    ConversationView,
)
