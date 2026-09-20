"""Normalize free text and attachment answers into one semantic-input boundary."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isclose
from typing import NoReturn
from uuid import UUID

from backend.contracts.commands import (
    AttachmentAnswerCommand,
    MultiChoiceAnswer,
    RecommendationFeedbackAnswer,
    SingleChoiceAnswer,
    SliderAnswer,
    UserMessageCommand,
)
from backend.contracts.conversation import (
    ChoiceOption,
    CompactChoiceAttachment,
    CompactMultiAttachment,
    ConversationMessage,
    DetailedChoiceAttachment,
    PreferenceSliderAttachment,
    RecommendationSetAttachment,
    TextMultiChoiceAttachment,
)


class AttachmentAnswerConflictCode(StrEnum):
    ATTACHMENT_NOT_FOUND = "attachment_not_found"
    ATTACHMENT_EXPIRED = "attachment_expired"
    ATTACHMENT_TRIP_CONFLICT = "attachment_trip_conflict"
    ANSWER_MODE_CONFLICT = "answer_mode_conflict"
    ANSWER_VALUE_CONFLICT = "answer_value_conflict"


class AttachmentAnswerConflict(ValueError):
    def __init__(self, code: AttachmentAnswerConflictCode, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class FreeTextSemanticInput:
    kind: str
    message_id: UUID
    text: str


@dataclass(frozen=True)
class AttachmentSemanticInput:
    kind: str
    attachment_id: UUID
    source_message_id: UUID
    answer: SingleChoiceAnswer | MultiChoiceAnswer | SliderAnswer | RecommendationFeedbackAnswer


SemanticInput = FreeTextSemanticInput | AttachmentSemanticInput


def normalize_user_message(command: UserMessageCommand) -> FreeTextSemanticInput:
    return FreeTextSemanticInput(
        kind="free_text",
        message_id=command.payload.message_id,
        text=command.payload.text,
    )


def normalize_attachment_answer(
    command: AttachmentAnswerCommand,
    *,
    current_trip_id: UUID,
    attachment_trip_id: UUID,
    source_message: ConversationMessage,
) -> AttachmentSemanticInput:
    """Validate attachment ownership and answer semantics before state operations."""

    if attachment_trip_id != current_trip_id:
        raise AttachmentAnswerConflict(
            AttachmentAnswerConflictCode.ATTACHMENT_TRIP_CONFLICT,
            "attachment belongs to another trip",
        )
    if source_message.message_id != command.payload.source_message_id:
        raise AttachmentAnswerConflict(
            AttachmentAnswerConflictCode.ATTACHMENT_NOT_FOUND,
            "source message does not match the command",
        )
    attachment = next(
        (
            item.root
            for item in source_message.attachments
            if item.root.attachment_id == command.payload.attachment_id
        ),
        None,
    )
    if attachment is None:
        raise AttachmentAnswerConflict(
            AttachmentAnswerConflictCode.ATTACHMENT_NOT_FOUND,
            "attachment was not found in the source message",
        )
    if attachment.status == "superseded" or not attachment.editable:
        raise AttachmentAnswerConflict(
            AttachmentAnswerConflictCode.ATTACHMENT_EXPIRED,
            "attachment is no longer editable",
        )

    answer = command.payload.answer
    if isinstance(attachment, (CompactChoiceAttachment, DetailedChoiceAttachment)):
        if not isinstance(answer, SingleChoiceAnswer):
            _answer_mode_conflict(attachment.kind, answer.answer_type)
        _validate_choice_ids([answer.option_id], attachment.options, 1, 1)
    elif isinstance(attachment, (CompactMultiAttachment, TextMultiChoiceAttachment)):
        if not isinstance(answer, MultiChoiceAnswer):
            _answer_mode_conflict(attachment.kind, answer.answer_type)
        _validate_choice_ids(
            answer.option_ids,
            attachment.options,
            attachment.minimum_selections,
            attachment.maximum_selections,
        )
        if (
            isinstance(attachment, TextMultiChoiceAttachment)
            and attachment.exclusive_option_id in answer.option_ids
            and len(answer.option_ids) != 1
        ):
            _answer_value_conflict("exclusive option must be selected alone")
    elif isinstance(attachment, PreferenceSliderAttachment):
        if not isinstance(answer, SliderAnswer):
            _answer_mode_conflict(attachment.kind, answer.answer_type)
        if answer.value is None:
            if not attachment.allow_no_preference:
                _answer_value_conflict("this slider requires a numeric value")
        else:
            if not attachment.minimum_value <= answer.value <= attachment.maximum_value:
                _answer_value_conflict("slider value is outside the declared range")
            steps = (answer.value - attachment.minimum_value) / attachment.step
            if not isclose(steps, round(steps), abs_tol=1e-9):
                _answer_value_conflict("slider value does not align with the declared step")
    elif isinstance(attachment, RecommendationSetAttachment):
        if not isinstance(answer, RecommendationFeedbackAnswer):
            _answer_mode_conflict(attachment.kind, answer.answer_type)
        item_ids = {item.recommendation_id for item in attachment.items}
        answer_ids = {item.recommendation_id for item in answer.feedback}
        if not answer_ids <= item_ids:
            _answer_value_conflict("recommendation feedback references an unknown item")
        if (
            not attachment.minimum_selections
            <= len(answer.feedback)
            <= attachment.maximum_selections
        ):
            _answer_value_conflict("recommendation feedback violates selection bounds")
    else:
        _answer_mode_conflict(attachment.kind, answer.answer_type)

    return AttachmentSemanticInput(
        kind="attachment_answer",
        attachment_id=command.payload.attachment_id,
        source_message_id=command.payload.source_message_id,
        answer=answer,
    )


def _validate_choice_ids(
    selected: list[str],
    options: list[ChoiceOption],
    minimum: int,
    maximum: int,
) -> None:
    option_ids = {option.option_id for option in options}
    if not set(selected) <= option_ids:
        _answer_value_conflict("answer references an unknown option")
    if not minimum <= len(selected) <= maximum:
        _answer_value_conflict("answer violates selection bounds")


def _answer_mode_conflict(attachment_kind: str, answer_type: str) -> NoReturn:
    raise AttachmentAnswerConflict(
        AttachmentAnswerConflictCode.ANSWER_MODE_CONFLICT,
        f"{answer_type} cannot answer {attachment_kind}",
    )


def _answer_value_conflict(message: str) -> NoReturn:
    raise AttachmentAnswerConflict(
        AttachmentAnswerConflictCode.ANSWER_VALUE_CONFLICT,
        message,
    )
