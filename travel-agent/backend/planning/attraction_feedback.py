"""Deterministic V3-32 mapping from recommendation answers to semantic operations."""

from __future__ import annotations

from datetime import date
from uuid import NAMESPACE_URL, UUID, uuid5

from backend.agent.semantic_operations import (
    AttractionIntentOperation,
    AttractionIntentValue,
    OperationEvidence,
    SemanticImpactKind,
    SemanticImpactScope,
    SemanticOperationBatch,
    SemanticOperationKind,
    SemanticPersistenceScope,
    SemanticTarget,
)
from backend.contracts.commands import RecommendationFeedbackAnswer
from backend.contracts.conversation import RecommendationSetAttachment
from backend.contracts.enums import AttractionIntent, Confidence, EvidenceSource


def recommendation_feedback_operations(
    *,
    trip_id: UUID,
    request_id: UUID,
    source_message_id: UUID,
    attachment: RecommendationSetAttachment,
    answer: RecommendationFeedbackAnswer,
    business_date: date,
) -> SemanticOperationBatch:
    """Map only explicitly answered V3 attraction items; unanswered items stay neutral."""

    if attachment.recommendation_domain != "attraction":
        raise ValueError("only attraction recommendations use V3 semantic feedback mapping")
    items = {item.recommendation_id: item for item in attachment.items}
    operations: list[AttractionIntentOperation] = []
    for feedback in answer.feedback:
        item = items.get(feedback.recommendation_id)
        if item is None or item.place_id is None:
            raise ValueError("recommendation feedback references an unknown attraction")
        intent = AttractionIntent(feedback.intent.value)
        operations.append(
            AttractionIntentOperation(
                operation_id=uuid5(
                    NAMESPACE_URL,
                    "iter:v3-attraction-feedback:"
                    f"{request_id}:{attachment.attachment_id}:{item.place_id}:{intent.value}",
                ),
                trip_id=trip_id,
                operation=(
                    SemanticOperationKind.NEGATE
                    if intent is AttractionIntent.AVOID
                    else SemanticOperationKind.OVERRIDE
                ),
                target=SemanticTarget.ATTRACTION_INTENTS,
                value=AttractionIntentValue(
                    place_id=item.place_id,
                    place_name=item.title,
                    intent=intent,
                ),
                evidence=OperationEvidence(
                    source=EvidenceSource.CARD,
                    source_trip_id=trip_id,
                    source_message_id=source_message_id,
                    source_attachment_id=attachment.attachment_id,
                ),
                confidence=Confidence.HIGH,
                persistence_scope=SemanticPersistenceScope.CURRENT_TRIP,
                impact_scope=SemanticImpactScope(
                    kind=SemanticImpactKind.SPECIFIC_ITEM,
                    item_id=item.place_id,
                ),
            )
        )
    return SemanticOperationBatch.model_validate(
        {"trip_id": str(trip_id), "operations": operations},
        context={"today": business_date},
    )
