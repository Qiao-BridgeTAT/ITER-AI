"""V4 client commands for the Prepare Agent conversation boundary."""

from __future__ import annotations

import re
from datetime import date
from typing import Annotated, Literal
from uuid import UUID

from pydantic import Field, RootModel, StringConstraints, field_validator, model_validator

from backend.contracts.v4.base import DisplayText, Identifier, V4ContractModel, require_unique

V4IdempotencyKey = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=8,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]+$",
    ),
]


class V4CommandEnvelope(V4ContractModel):
    protocol_version: Literal["v4"] = "v4"
    schema_version: Literal["4.0.0"] = "4.0.0"
    request_id: UUID
    idempotency_key: V4IdempotencyKey
    expected_state_version: int = Field(ge=0, strict=True)
    client_sequence: int = Field(ge=1, strict=True)


class V4UserMessagePayload(V4ContractModel):
    message_id: UUID
    text: DisplayText


class V4UserMessageCommand(V4CommandEnvelope):
    type: Literal["user_message"]
    payload: V4UserMessagePayload


class V4TripSetupPayload(V4ContractModel):
    message_id: UUID
    city_id: str = Field(pattern=r"^cn-[0-9]{6}$")
    start_date: date
    end_date: date

    @field_validator("start_date", "end_date", mode="before")
    @classmethod
    def calendar_dates_only(cls, value: object) -> object:
        if isinstance(value, str) and re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
            return value
        if type(value) is date:
            return value
        raise ValueError("trip setup requires YYYY-MM-DD calendar dates")

    @model_validator(mode="after")
    def supported_range(self) -> V4TripSetupPayload:
        if not 1 <= (self.end_date - self.start_date).days + 1 <= 5:
            raise ValueError("trip setup requires an inclusive range of one to five days")
        return self


class V4TripSetupCommand(V4CommandEnvelope):
    type: Literal["trip_setup"]
    payload: V4TripSetupPayload


class V4CardSelection(V4ContractModel):
    option_id: Identifier
    disposition: Literal[
        "selected",
        "excluded",
        "must",
        "want",
        "destination",
        "if_convenient",
        "avoid",
    ]


class V4CardAnswerPayload(V4ContractModel):
    interaction_id: UUID
    answer_id: UUID
    selections: list[V4CardSelection] = Field(default_factory=list, max_length=14)
    control_action_id: Identifier | None = None
    optional_user_text: DisplayText | None = None

    @model_validator(mode="after")
    def answer_has_one_authoritative_shape(self) -> V4CardAnswerPayload:
        if bool(self.selections) == (self.control_action_id is not None):
            raise ValueError("card answer requires selections or one control action")
        require_unique((item.option_id for item in self.selections), "option_id")
        return self


class V4CardAnswerCommand(V4CommandEnvelope):
    type: Literal["card_answer"]
    payload: V4CardAnswerPayload


class V4RetryInteractionPayload(V4ContractModel):
    interaction_id: UUID


class V4RetryInteractionCommand(V4CommandEnvelope):
    type: Literal["retry_interaction"]
    payload: V4RetryInteractionPayload


class V4TaskBookConfirmationPayload(V4ContractModel):
    task_book_id: UUID
    task_book_version: int = Field(ge=1, strict=True)


class V4TaskBookConfirmationCommand(V4CommandEnvelope):
    type: Literal["task_book_confirmation"]
    payload: V4TaskBookConfirmationPayload


class V4CancelGenerationPayload(V4ContractModel):
    generation_id: UUID


class V4CancelGenerationCommand(V4CommandEnvelope):
    type: Literal["cancel_generation"]
    payload: V4CancelGenerationPayload


class V4PlannerResumePayload(V4ContractModel):
    generation_id: UUID
    expected_workspace_revision: int = Field(ge=0, strict=True)


class V4PlannerResumeCommand(V4CommandEnvelope):
    type: Literal["planner_resume"]
    payload: V4PlannerResumePayload


class V4PlanTransportSelectionPayload(V4PlannerResumePayload):
    plan_version_id: UUID
    leg_id: UUID
    transport_mode: Literal["taxi", "public_transit", "walking"]


class V4PlanTransportSelectionCommand(V4CommandEnvelope):
    type: Literal["plan_transport_selection"]
    payload: V4PlanTransportSelectionPayload


class V4PlannerAnswerPayload(V4PlannerResumePayload):
    interaction_id: UUID
    answer_id: UUID
    resume_token: Identifier
    option_id: Identifier
    optional_user_text: DisplayText | None = None


class V4PlannerAnswerCommand(V4CommandEnvelope):
    type: Literal["planner_answer"]
    payload: V4PlannerAnswerPayload


V4ClientCommandValue = Annotated[
    V4UserMessageCommand
    | V4TripSetupCommand
    | V4CardAnswerCommand
    | V4RetryInteractionCommand
    | V4TaskBookConfirmationCommand
    | V4CancelGenerationCommand
    | V4PlannerResumeCommand
    | V4PlanTransportSelectionCommand
    | V4PlannerAnswerCommand,
    Field(discriminator="type"),
]


class V4ClientCommand(RootModel[V4ClientCommandValue]):
    """Strict public union negotiated with protocol_version=v4."""


V4_COMMAND_CONTRACTS = (V4ClientCommand,)
