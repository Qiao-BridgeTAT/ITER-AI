from __future__ import annotations

import hashlib
import json
from uuid import UUID

from pydantic import Field, model_validator

from backend.agent.readiness_state import CriticalQuestionResolutionProof
from backend.agent.semantic_operations import SemanticOperation
from backend.agent.state_merge import SemanticTripState
from backend.contracts.base import ContractModel
from backend.contracts.commands import AttachmentAnswerValue
from backend.contracts.enums import EvidenceSource


class AttachmentAnswerSemanticBinding(ContractModel):
    """Server-only mapping from one typed answer to its already reviewed effects."""

    attachment_id: UUID
    source_message_id: UUID
    answer: AttachmentAnswerValue
    operations: tuple[SemanticOperation, ...] = Field(min_length=1, max_length=50)
    question_resolution: CriticalQuestionResolutionProof | None = None

    @model_validator(mode="after")
    def provenance_matches_attachment(self) -> AttachmentAnswerSemanticBinding:
        operation_ids = {operation.operation_id for operation in self.operations}
        for operation in self.operations:
            evidence = operation.evidence
            if (
                evidence.source is not EvidenceSource.CARD
                or evidence.source_message_id != self.source_message_id
                or evidence.source_attachment_id != self.attachment_id
            ):
                raise ValueError("attachment binding operations require matching card evidence")
        proof = self.question_resolution
        if proof is not None:
            if proof.source_message_id != self.source_message_id:
                raise ValueError("attachment question proof must use the source message")
            if not set(proof.operation_ids) <= operation_ids:
                raise ValueError("attachment question proof references an unknown operation")
        return self


class StableAgentCheckpoint(ContractModel):
    """Private validated payload stored only after a complete graph turn."""

    semantic_state: SemanticTripState
    attachment_bindings: tuple[AttachmentAnswerSemanticBinding, ...] = ()

    @model_validator(mode="after")
    def bindings_are_unique_and_owned(self) -> StableAgentCheckpoint:
        keys = [
            (binding.attachment_id, _answer_fingerprint(binding.answer))
            for binding in self.attachment_bindings
        ]
        if len(set(keys)) != len(keys):
            raise ValueError("agent attachment answer bindings must be unique")
        trip_id = self.semantic_state.trip_id
        if any(
            operation.trip_id != trip_id
            for binding in self.attachment_bindings
            for operation in binding.operations
        ):
            raise ValueError("agent attachment binding belongs to another trip")
        return self


def _answer_fingerprint(answer: AttachmentAnswerValue) -> str:
    payload = json.dumps(answer.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
