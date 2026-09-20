"""Deterministic selection of one primary conversational response action."""

from __future__ import annotations

import re
from enum import StrEnum
from uuid import UUID

from pydantic import model_validator

from backend.agent.readiness import ReadinessAssessment, ReadinessDecision
from backend.agent.readiness_state import ReadinessQuestion
from backend.contracts.base import ContractModel
from backend.contracts.common import ShortText
from backend.contracts.conversation import (
    CompactChoiceAttachment,
    CompactMultiAttachment,
    ConversationAttachment,
    DetailedChoiceAttachment,
    ExternalFactReference,
    PreferenceSliderAttachment,
    RecommendationSetAttachment,
    TextMultiChoiceAttachment,
)

_SENTENCE_BOUNDARY = re.compile(r"[。！？!?]+")
_SELECTABLE_ATTACHMENTS = (
    CompactChoiceAttachment,
    CompactMultiAttachment,
    DetailedChoiceAttachment,
    PreferenceSliderAttachment,
    TextMultiChoiceAttachment,
)


class ResponseActionKind(StrEnum):
    ORDINARY_REPLY = "ordinary_reply"
    CRITICAL_QUESTION = "critical_question"
    STRUCTURED_ATTACHMENT = "structured_attachment"
    CANDIDATE_RECOMMENDATION = "candidate_recommendation"
    TASK_BOOK = "task_book"


class ExpressionCost(StrEnum):
    FREE_TEXT_EASIER = "free_text_easier"
    EQUIVALENT = "equivalent"
    ATTACHMENT_EASIER = "attachment_easier"


class ResponseCopy(ContractModel):
    text: ShortText
    expanded_for_conflict: bool = False

    @model_validator(mode="after")
    def ordinary_copy_is_concise_by_default(self) -> ResponseCopy:
        if not self.expanded_for_conflict and _sentence_count(self.text) > 3:
            raise ValueError("ordinary Agent replies must not exceed three sentences")
        return self


class ResponseSelectionContext(ContractModel):
    readiness: ReadinessAssessment
    ordinary_reply: ResponseCopy | None = None
    user_question_answer: ResponseCopy | None = None
    clarification_attachment: ConversationAttachment | None = None
    clarification_external_facts: tuple[ExternalFactReference, ...] = ()
    attachment_for_question_id: UUID | None = None
    clarification_expression_cost: ExpressionCost = ExpressionCost.FREE_TEXT_EASIER
    recommendation: RecommendationSetAttachment | None = None
    recommendation_external_facts: tuple[ExternalFactReference, ...] = ()
    recommendation_requested_by_user: bool = False
    recommendation_needed_for_progress: bool = False

    @model_validator(mode="after")
    def opportunities_are_well_formed(self) -> ResponseSelectionContext:
        if (self.clarification_attachment is None) != (self.attachment_for_question_id is None):
            raise ValueError("a clarification attachment must identify its readiness question")
        if self.clarification_attachment is not None and not isinstance(
            self.clarification_attachment.root,
            _SELECTABLE_ATTACHMENTS,
        ):
            raise ValueError("clarification must use a selectable attachment")
        if self.clarification_attachment is None and self.clarification_external_facts:
            raise ValueError("clarification facts require a clarification attachment")
        if self.clarification_attachment is not None and set(
            self.clarification_attachment.root.external_fact_ids
        ) != {fact.fact_id for fact in self.clarification_external_facts}:
            raise ValueError("clarification facts must match attachment references")
        if (
            self.recommendation_requested_by_user or self.recommendation_needed_for_progress
        ) and self.recommendation is None:
            raise ValueError("a recommendation decision requires a recommendation payload")
        if self.recommendation is None and self.recommendation_external_facts:
            raise ValueError("recommendation facts require a recommendation payload")
        if (
            self.recommendation is not None
            and self.recommendation.recommendation_domain in {"attraction", "restaurant"}
            and set(self.recommendation.external_fact_ids)
            != {fact.fact_id for fact in self.recommendation_external_facts}
        ):
            raise ValueError("recommendation facts must match attachment references")
        return self


class ResponseSelection(ContractModel):
    action: ResponseActionKind
    reason: ShortText
    message: ShortText | None = None
    question: ReadinessQuestion | None = None
    attachment: ConversationAttachment | None = None
    attachment_external_facts: tuple[ExternalFactReference, ...] = ()
    recommendation: RecommendationSetAttachment | None = None
    recommendation_external_facts: tuple[ExternalFactReference, ...] = ()
    task_book_requested: bool = False
    text_reply_allowed: bool = True
    acknowledgement: ShortText | None = None

    @model_validator(mode="after")
    def payload_matches_primary_action(self) -> ResponseSelection:
        present = {
            "message": self.message is not None,
            "question": self.question is not None,
            "attachment": self.attachment is not None,
            "recommendation": self.recommendation is not None,
            "task_book": self.task_book_requested,
        }
        expected = {
            ResponseActionKind.ORDINARY_REPLY: "message",
            ResponseActionKind.CRITICAL_QUESTION: "question",
            ResponseActionKind.STRUCTURED_ATTACHMENT: "attachment",
            ResponseActionKind.CANDIDATE_RECOMMENDATION: "recommendation",
            ResponseActionKind.TASK_BOOK: "task_book",
        }[self.action]
        if present[expected] is not True or sum(present.values()) != 1:
            raise ValueError("response selection must contain exactly its primary action payload")
        if self.action is ResponseActionKind.TASK_BOOK and self.text_reply_allowed:
            raise ValueError("task-book generation is not an answer input")
        if self.acknowledgement is not None and self.action not in {
            ResponseActionKind.CRITICAL_QUESTION,
            ResponseActionKind.STRUCTURED_ATTACHMENT,
        }:
            raise ValueError("acknowledgement can only lead into a pending answer request")
        if self.recommendation is None and self.recommendation_external_facts:
            raise ValueError("recommendation facts require a recommendation response")
        if self.attachment is None and self.attachment_external_facts:
            raise ValueError("attachment facts require an attachment response")
        if self.attachment is not None and set(self.attachment.root.external_fact_ids) != {
            fact.fact_id for fact in self.attachment_external_facts
        }:
            raise ValueError("attachment facts must match attachment references")
        return self


def select_response(context: ResponseSelectionContext) -> ResponseSelection:
    """Select one action from current intent, readiness, and expression cost."""

    readiness = context.readiness

    # A direct user question interrupts collection, is answered, and then the graph
    # resumes from the same readiness boundary on the next turn.
    if context.user_question_answer is not None:
        return _ordinary(
            context.user_question_answer.text,
            "answer the user's current question before resuming collection",
        )

    # User-requested exploration outranks an Agent-initiated question. It is not a
    # fixed domain step; the upstream relation decides when it is relevant.
    if context.recommendation_requested_by_user:
        assert context.recommendation is not None
        return ResponseSelection(
            action=ResponseActionKind.CANDIDATE_RECOMMENDATION,
            reason="the user explicitly requested concrete candidates",
            recommendation=context.recommendation,
            recommendation_external_facts=context.recommendation_external_facts,
        )

    if readiness.decision in {
        ReadinessDecision.ASK_CRITICAL_QUESTION,
        ReadinessDecision.WAITING_FOR_CRITICAL_ANSWER,
    }:
        question = readiness.question
        assert question is not None
        if (
            context.clarification_attachment is not None
            and context.attachment_for_question_id == question.question_id
            and context.clarification_expression_cost is ExpressionCost.ATTACHMENT_EASIER
        ):
            return ResponseSelection(
                action=ResponseActionKind.STRUCTURED_ATTACHMENT,
                reason="the attachment lowers the cost of answering this exact question",
                attachment=context.clarification_attachment,
                attachment_external_facts=context.clarification_external_facts,
                acknowledgement=(
                    context.ordinary_reply.text if context.ordinary_reply is not None else None
                ),
            )
        return ResponseSelection(
            action=ResponseActionKind.CRITICAL_QUESTION,
            reason="one unresolved issue materially affects the trip",
            question=question,
            acknowledgement=(
                context.ordinary_reply.text if context.ordinary_reply is not None else None
            ),
        )

    if readiness.decision is ReadinessDecision.BLOCKED_REQUIRED_INPUT:
        return _ordinary(
            "还有一项会影响行程且无法安全假设的信息，请直接补充后我再继续。",
            "the critical-question budget is exhausted but required input is still missing",
        )

    if context.recommendation_needed_for_progress:
        assert context.recommendation is not None
        return ResponseSelection(
            action=ResponseActionKind.CANDIDATE_RECOMMENDATION,
            reason="concrete candidates are the lowest-cost way to continue exploration",
            recommendation=context.recommendation,
            recommendation_external_facts=context.recommendation_external_facts,
        )

    if readiness.decision in {
        ReadinessDecision.REQUEST_FINAL_SUPPLEMENT,
        ReadinessDecision.WAITING_FOR_FINAL_SUPPLEMENT,
    }:
        assert readiness.final_supplement_prompt is not None
        return _ordinary(
            readiness.final_supplement_prompt,
            "readiness passed and the mandatory final supplement check is pending",
        )

    if readiness.decision is ReadinessDecision.READY_FOR_TASK_BOOK:
        return ResponseSelection(
            action=ResponseActionKind.TASK_BOOK,
            reason="information collection closed and a task book can be projected",
            task_book_requested=True,
            text_reply_allowed=False,
        )

    return _ordinary(
        context.ordinary_reply.text
        if context.ordinary_reply is not None
        else "我已经记下了，我们可以继续聊你在意的安排。",
        "no higher-priority clarification, exploration, or task-book action applies",
    )


def _ordinary(message: str, reason: str) -> ResponseSelection:
    return ResponseSelection(
        action=ResponseActionKind.ORDINARY_REPLY,
        reason=reason,
        message=message,
    )


def _sentence_count(text: str) -> int:
    segments = [segment.strip() for segment in _SENTENCE_BOUNDARY.split(text) if segment.strip()]
    return max(1, len(segments))
