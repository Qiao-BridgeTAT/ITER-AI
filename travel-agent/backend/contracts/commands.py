"""P0-10 client command envelope and typed payloads."""

from __future__ import annotations

from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import ConfigDict, Field, RootModel, StringConstraints, field_validator

from backend.contracts.base import ContractModel
from backend.contracts.cold_start import ColdStartSubmission
from backend.contracts.common import NonEmptyText, ShortText
from backend.contracts.enums import CommandType, RecommendationIntent
from backend.contracts.feedback import (
    AttractionFeedbackSubmission,
    CityThemeSelection,
    DiningPreferenceSelection,
    RestaurantFeedbackSubmission,
)
from backend.contracts.lodging import HotelFavoritesSubmission
from backend.contracts.trip_setup import (
    CityBriefAcknowledgement,
    CitySelection,
    TripSetupSubmission,
)

ProtocolVersion = Literal["2.0.0"]
SchemaVersion = Literal["2.0.0"]
IdempotencyKey = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=8,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]+$",
    ),
]


class EmptyPayload(ContractModel):
    """Explicit empty object used by action-only commands."""


class AdjustPersonalDefaultsPayload(ContractModel):
    edited_profile_text: ShortText


class UserMessagePayload(ContractModel):
    message_id: UUID
    text: NonEmptyText


class SingleChoiceAnswer(ContractModel):
    answer_type: Literal["single_choice"]
    option_id: NonEmptyText


class MultiChoiceAnswer(ContractModel):
    answer_type: Literal["multi_choice"]
    option_ids: list[NonEmptyText] = Field(
        min_length=1,
        max_length=12,
        json_schema_extra={"uniqueItems": True},
    )
    free_text: ShortText | None = None

    @field_validator("option_ids")
    @classmethod
    def option_ids_are_unique(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("option_ids must not contain duplicates")
        return value


class SliderAnswer(ContractModel):
    answer_type: Literal["slider"]
    value: float | None = Field(allow_inf_nan=False)


class RecommendationFeedback(ContractModel):
    recommendation_id: UUID
    intent: RecommendationIntent


class RecommendationFeedbackAnswer(ContractModel):
    model_config = ConfigDict(
        json_schema_extra={
            "x-travel-unique-by": {
                "arrayField": "feedback",
                "itemField": "recommendation_id",
            }
        }
    )

    answer_type: Literal["recommendation_feedback"]
    feedback: list[RecommendationFeedback] = Field(max_length=20)

    @field_validator("feedback")
    @classmethod
    def recommendation_ids_are_unique(
        cls, value: list[RecommendationFeedback]
    ) -> list[RecommendationFeedback]:
        ids = [item.recommendation_id for item in value]
        if len(set(ids)) != len(ids):
            raise ValueError("recommendation_id must not contain duplicates")
        return value


AttachmentAnswerValue = Annotated[
    SingleChoiceAnswer | MultiChoiceAnswer | SliderAnswer | RecommendationFeedbackAnswer,
    Field(discriminator="answer_type"),
]


class AttachmentAnswerPayload(ContractModel):
    attachment_id: UUID
    source_message_id: UUID
    answer: AttachmentAnswerValue


class AcceptPlanPayload(ContractModel):
    plan_version_id: UUID


class CancelGenerationPayload(ContractModel):
    generation_id: UUID


class ResetTripPayload(ContractModel):
    preserve_personal_defaults: Literal[True] = True


class CommandEnvelope(ContractModel):
    protocol_version: ProtocolVersion
    schema_version: SchemaVersion
    request_id: UUID
    idempotency_key: IdempotencyKey
    expected_state_version: int = Field(ge=0, strict=True)


class ColdStartSubmitCommand(CommandEnvelope):
    type: Literal[CommandType.COLD_START_SUBMIT]
    payload: ColdStartSubmission


class CitySelectCommand(CommandEnvelope):
    type: Literal[CommandType.CITY_SELECT]
    payload: CitySelection


class CityBriefAcknowledgeCommand(CommandEnvelope):
    type: Literal[CommandType.CITY_BRIEF_ACKNOWLEDGE]
    payload: CityBriefAcknowledgement


class PersonalDefaultsConfirmCommand(CommandEnvelope):
    type: Literal[CommandType.PERSONAL_DEFAULTS_CONFIRM]
    payload: EmptyPayload


class PersonalDefaultsAdjustCommand(CommandEnvelope):
    type: Literal[CommandType.PERSONAL_DEFAULTS_ADJUST]
    payload: AdjustPersonalDefaultsPayload


class PersonalDefaultsNotUseCommand(CommandEnvelope):
    type: Literal[CommandType.PERSONAL_DEFAULTS_NOT_USE]
    payload: EmptyPayload


class TripSetupSubmitCommand(CommandEnvelope):
    type: Literal[CommandType.TRIP_SETUP_SUBMIT]
    payload: TripSetupSubmission


class CityThemesSubmitCommand(CommandEnvelope):
    type: Literal[CommandType.CITY_THEMES_SUBMIT]
    payload: CityThemeSelection


class AttractionFeedbackSubmitCommand(CommandEnvelope):
    type: Literal[CommandType.ATTRACTION_FEEDBACK_SUBMIT]
    payload: AttractionFeedbackSubmission


class DiningPreferencesSubmitCommand(CommandEnvelope):
    type: Literal[CommandType.DINING_PREFERENCES_SUBMIT]
    payload: DiningPreferenceSelection


class RestaurantFeedbackSubmitCommand(CommandEnvelope):
    type: Literal[CommandType.RESTAURANT_FEEDBACK_SUBMIT]
    payload: RestaurantFeedbackSubmission


class HotelFavoritesSubmitCommand(CommandEnvelope):
    type: Literal[CommandType.HOTEL_FAVORITES_SUBMIT]
    payload: HotelFavoritesSubmission


class TaskBookConfirmPayload(ContractModel):
    trip_id: UUID


class TaskBookConfirmCommand(CommandEnvelope):
    type: Literal[CommandType.TASK_BOOK_CONFIRM]
    payload: TaskBookConfirmPayload


class UserMessageCommand(CommandEnvelope):
    type: Literal[CommandType.USER_MESSAGE]
    payload: UserMessagePayload


class AttachmentAnswerCommand(CommandEnvelope):
    type: Literal[CommandType.ATTACHMENT_ANSWER]
    payload: AttachmentAnswerPayload


class AcceptPlanCommand(CommandEnvelope):
    type: Literal[CommandType.ACCEPT_PLAN]
    payload: AcceptPlanPayload


class CancelGenerationCommand(CommandEnvelope):
    type: Literal[CommandType.CANCEL_GENERATION]
    payload: CancelGenerationPayload


class ResetTripCommand(CommandEnvelope):
    type: Literal[CommandType.RESET_TRIP]
    payload: ResetTripPayload


ClientCommandValue = Annotated[
    ColdStartSubmitCommand
    | CitySelectCommand
    | CityBriefAcknowledgeCommand
    | PersonalDefaultsConfirmCommand
    | PersonalDefaultsAdjustCommand
    | PersonalDefaultsNotUseCommand
    | TripSetupSubmitCommand
    | CityThemesSubmitCommand
    | AttractionFeedbackSubmitCommand
    | DiningPreferencesSubmitCommand
    | RestaurantFeedbackSubmitCommand
    | HotelFavoritesSubmitCommand
    | TaskBookConfirmCommand
    | UserMessageCommand
    | AttachmentAnswerCommand
    | AcceptPlanCommand
    | CancelGenerationCommand
    | ResetTripCommand,
    Field(discriminator="type"),
]


class ClientCommand(RootModel[ClientCommandValue]):
    """Discriminated union for every stage-0 WebSocket client command."""


P0_COMMAND_CONTRACTS: tuple[type[Any], ...] = (ClientCommand,)
