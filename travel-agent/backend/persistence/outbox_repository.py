"""Durable, retryable delivery of already committed assistant messages."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid5

from pydantic import ValidationError
from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from backend.contracts.v4.conversation import (
    AssistantCompletedEvent,
    AssistantStartedEvent,
    AttachmentReadyEvent,
    ConversationEventV4,
    StateCommittedEvent,
)
from backend.persistence.models import OutboxEvent

_OUTBOX_EVENT_NAMESPACE = UUID("fd1f9272-6e52-4a45-a077-f4eaf7130c61")


class OutboxPersistenceError(RuntimeError):
    """Base class for durable delivery failures."""


class OutboxLeaseConflictError(OutboxPersistenceError):
    pass


def canonical_json_hash(value: Any) -> str:
    """Hash one JSON-compatible value with deterministic key and Unicode handling."""

    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def stable_text_chunks(text: str, *, chunk_size: int) -> tuple[list[str], str]:
    """Persisted Unicode-codepoint chunks, stable across dispatcher retries."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    chunks = [text[index : index + chunk_size] for index in range(0, len(text), chunk_size)]
    return chunks, f"unicode-codepoint-v1:size={chunk_size}"


@dataclass(frozen=True)
class PreparedOutboxBundle:
    payload: dict[str, Any]
    chunks: list[str]
    content_hash: str
    chunking_algorithm_version: str


def prepare_assistant_bundle(
    *,
    message_id: UUID,
    text: str,
    attachments: list[dict[str, Any]],
    generation_mode: str,
    failure_code: str | None,
    chunk_size: int,
    invalidated_interaction_ids: list[str] | None = None,
) -> PreparedOutboxBundle:
    chunks, algorithm = stable_text_chunks(text, chunk_size=chunk_size)
    committed_content = {
        "message_id": str(message_id),
        "text": text,
        "attachments": attachments,
        "generation_mode": generation_mode,
        "failure_code": failure_code,
        "invalidated_interaction_ids": invalidated_interaction_ids or [],
    }
    return PreparedOutboxBundle(
        payload=committed_content,
        chunks=chunks,
        content_hash=canonical_json_hash(committed_content),
        chunking_algorithm_version=algorithm,
    )


@dataclass(frozen=True)
class OutboxBundleRecord:
    outbox_id: UUID
    cursor: str
    trip_id: UUID
    turn_id: UUID
    generation_id: UUID
    message_id: UUID
    committed_state_version: int
    payload: dict[str, Any]
    chunks: tuple[str, ...]
    terminal_event: dict[str, Any]
    content_hash: str
    chunking_algorithm_version: str
    retry_count: int
    delivered_sequence: int
    committed_at: datetime


@dataclass(frozen=True)
class OutboxFrame:
    event_id: UUID
    cursor: str
    sequence: int
    event_type: str
    payload: dict[str, Any]


def stable_event_id(
    outbox_id: UUID,
    *,
    event_type: str,
    content_hash: str,
    chunk_index: int | None = None,
) -> UUID:
    suffix = f":{chunk_index}" if chunk_index is not None else ""
    return uuid5(
        _OUTBOX_EVENT_NAMESPACE,
        f"{outbox_id}:{event_type}{suffix}:{content_hash}",
    )


def stable_delivery_frames(bundle: OutboxBundleRecord) -> tuple[OutboxFrame, ...]:
    """Derive stable event IDs and sequences from one persisted bundle."""

    frames: list[OutboxFrame] = [
        _state_committed_frame(bundle),
        _assistant_started_frame(bundle),
    ]
    for index, chunk in enumerate(bundle.chunks):
        sequence = len(frames) + 1
        frames.append(
            OutboxFrame(
                event_id=stable_event_id(
                    bundle.outbox_id,
                    event_type="assistant.delta",
                    content_hash=bundle.content_hash,
                    chunk_index=index,
                ),
                cursor=bundle.cursor,
                sequence=sequence,
                event_type="assistant.delta",
                payload={
                    "event_id": str(
                        stable_event_id(
                            bundle.outbox_id,
                            event_type="assistant.delta",
                            content_hash=bundle.content_hash,
                            chunk_index=index,
                        )
                    ),
                    "event_type": "assistant.delta",
                    "protocol_version": "v4",
                    "trip_id": str(bundle.trip_id),
                    "turn_id": str(bundle.turn_id),
                    "generation_id": str(bundle.generation_id),
                    "sequence": sequence,
                    "emitted_at": bundle.committed_at.isoformat(),
                    "message_id": str(bundle.message_id),
                    "state_version": bundle.committed_state_version,
                    "chunk_index": index,
                    "delta": chunk,
                    "content_hash": bundle.content_hash,
                },
            )
        )
    for attachment in bundle.payload["attachments"]:
        sequence = len(frames) + 1
        attachment_id, interaction_id = _attachment_identity(attachment)
        event = AttachmentReadyEvent(
            event_id=str(
                stable_event_id(
                    bundle.outbox_id,
                    event_type="attachment.ready",
                    content_hash=bundle.content_hash,
                    chunk_index=sequence,
                )
            ),
            event_type="attachment.ready",
            trip_id=str(bundle.trip_id),
            turn_id=str(bundle.turn_id),
            generation_id=str(bundle.generation_id),
            sequence=sequence,
            emitted_at=bundle.committed_at,
            message_id=str(bundle.message_id),
            state_version=bundle.committed_state_version,
            attachment_id=attachment_id,
            interaction_id=interaction_id,
        )
        frames.append(
            OutboxFrame(
                event_id=UUID(event.event_id),
                cursor=bundle.cursor,
                sequence=sequence,
                event_type=event.event_type,
                payload=event.model_dump(mode="json"),
            )
        )
    terminal_sequence = len(frames) + 1
    frames.append(
        OutboxFrame(
            event_id=stable_event_id(
                bundle.outbox_id,
                event_type="assistant.completed",
                content_hash=bundle.content_hash,
            ),
            cursor=bundle.cursor,
            sequence=terminal_sequence,
            event_type="assistant.completed",
            payload=dict(bundle.terminal_event),
        )
    )
    return tuple(frames)


def terminal_sequence_for_bundle(prepared: PreparedOutboxBundle) -> int:
    """Return the frozen terminal sequence for a not-yet-persisted bundle."""

    return 2 + len(prepared.chunks) + len(prepared.payload["attachments"]) + 1


def _state_committed_frame(bundle: OutboxBundleRecord) -> OutboxFrame:
    event = StateCommittedEvent(
        event_id=str(
            stable_event_id(
                bundle.outbox_id,
                event_type="state.committed",
                content_hash=bundle.content_hash,
            )
        ),
        event_type="state.committed",
        trip_id=str(bundle.trip_id),
        turn_id=str(bundle.turn_id),
        generation_id=str(bundle.generation_id),
        sequence=1,
        emitted_at=bundle.committed_at,
        message_id=str(bundle.message_id),
        state_version=bundle.committed_state_version,
        base_state_version=bundle.committed_state_version - 1,
        committed_state_version=bundle.committed_state_version,
        invalidated_interaction_ids=bundle.payload["invalidated_interaction_ids"],
    )
    return OutboxFrame(
        event_id=UUID(event.event_id),
        cursor=bundle.cursor,
        sequence=event.sequence,
        event_type=event.event_type,
        payload=event.model_dump(mode="json"),
    )


def _assistant_started_frame(bundle: OutboxBundleRecord) -> OutboxFrame:
    event = AssistantStartedEvent(
        event_id=str(
            stable_event_id(
                bundle.outbox_id,
                event_type="assistant.started",
                content_hash=bundle.content_hash,
            )
        ),
        event_type="assistant.started",
        trip_id=str(bundle.trip_id),
        turn_id=str(bundle.turn_id),
        generation_id=str(bundle.generation_id),
        sequence=2,
        emitted_at=bundle.committed_at,
        message_id=str(bundle.message_id),
        state_version=bundle.committed_state_version,
    )
    return OutboxFrame(
        event_id=UUID(event.event_id),
        cursor=bundle.cursor,
        sequence=event.sequence,
        event_type=event.event_type,
        payload=event.model_dump(mode="json"),
    )


def _attachment_identity(attachment: object) -> tuple[str, str | None]:
    if not isinstance(attachment, dict):
        raise OutboxPersistenceError("V4 outbox attachment must be an object")
    attachment_id = (
        attachment.get("attachment_id")
        or attachment.get("task_book_id")
        or attachment.get("plan_version_id")
    )
    if not isinstance(attachment_id, str) or not attachment_id:
        raise OutboxPersistenceError("V4 outbox attachment is missing its stable ID")
    interaction_id = attachment.get("interaction_id")
    if interaction_id is not None and not isinstance(interaction_id, str):
        raise OutboxPersistenceError("V4 outbox interaction ID must be a string")
    return attachment_id, interaction_id


def pending_delivery_frames(bundle: OutboxBundleRecord) -> tuple[OutboxFrame, ...]:
    """Return only frames not durably acknowledged before a retry or restart."""

    return tuple(
        frame
        for frame in stable_delivery_frames(bundle)
        if frame.sequence > bundle.delivered_sequence
    )


class OutboxRepository:
    """Claims and settles outbox bundles without re-running the producing turn."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def load_by_cursor(self, cursor: str) -> OutboxBundleRecord | None:
        async with self._session_factory() as session:
            row = await session.scalar(select(OutboxEvent).where(OutboxEvent.cursor == cursor))
        return _record(row) if row is not None else None

    async def claim_pending(
        self,
        *,
        worker_id: str,
        limit: int = 20,
        lease_seconds: int = 30,
        now: datetime | None = None,
    ) -> list[OutboxBundleRecord]:
        if not worker_id:
            raise ValueError("worker_id must not be empty")
        if limit <= 0 or lease_seconds <= 0:
            raise ValueError("limit and lease_seconds must be greater than zero")
        claimed_at = now or datetime.now(UTC)
        lease_expires_at = claimed_at + timedelta(seconds=lease_seconds)
        async with self._session_factory() as session, session.begin():
            rows = (
                await session.scalars(
                    select(OutboxEvent)
                    .where(
                        or_(
                            and_(
                                OutboxEvent.delivery_status == "pending",
                                OutboxEvent.next_attempt_at <= claimed_at,
                            ),
                            and_(
                                OutboxEvent.delivery_status == "leased",
                                OutboxEvent.lease_expires_at.is_not(None),
                                OutboxEvent.lease_expires_at <= claimed_at,
                            ),
                        )
                    )
                    .order_by(OutboxEvent.next_attempt_at, OutboxEvent.created_at, OutboxEvent.id)
                    .limit(limit)
                    .with_for_update(skip_locked=True)
                )
            ).all()
            for row in rows:
                row.delivery_status = "leased"
                row.leased_by = worker_id
                row.lease_expires_at = lease_expires_at
            await session.flush()
            return [_record(row) for row in rows]

    async def claim_by_cursor(
        self,
        cursor: str,
        *,
        worker_id: str,
        lease_seconds: int = 30,
        now: datetime | None = None,
    ) -> OutboxBundleRecord | None:
        """Claim one known bundle without touching unrelated trips or deliveries."""

        if not worker_id:
            raise ValueError("worker_id must not be empty")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be greater than zero")
        claimed_at = now or datetime.now(UTC)
        async with self._session_factory() as session, session.begin():
            row = await session.scalar(
                select(OutboxEvent)
                .where(
                    OutboxEvent.cursor == cursor,
                    or_(
                        and_(
                            OutboxEvent.delivery_status == "pending",
                            OutboxEvent.next_attempt_at <= claimed_at,
                        ),
                        and_(
                            OutboxEvent.delivery_status == "leased",
                            OutboxEvent.lease_expires_at.is_not(None),
                            OutboxEvent.lease_expires_at <= claimed_at,
                        ),
                    ),
                )
                .with_for_update(skip_locked=True)
            )
            if row is None:
                return None
            row.delivery_status = "leased"
            row.leased_by = worker_id
            row.lease_expires_at = claimed_at + timedelta(seconds=lease_seconds)
            await session.flush()
            return _record(row)

    async def mark_delivered(
        self,
        outbox_id: UUID,
        *,
        worker_id: str,
        delivered_at: datetime | None = None,
    ) -> None:
        completed_at = delivered_at or datetime.now(UTC)
        async with self._session_factory() as session, session.begin():
            row = await session.scalar(
                select(OutboxEvent).where(OutboxEvent.id == outbox_id).with_for_update()
            )
            if row is None or row.delivery_status != "leased" or row.leased_by != worker_id:
                raise OutboxLeaseConflictError("outbox lease no longer belongs to this worker")
            row.delivery_status = "delivered"
            row.delivered_sequence = _row_terminal_sequence(row)
            row.delivered_at = completed_at
            row.leased_by = None
            row.lease_expires_at = None
            row.last_error_code = None
            await session.flush()

    async def mark_frame_delivered(
        self,
        outbox_id: UUID,
        *,
        worker_id: str,
        sequence: int,
        delivered_at: datetime | None = None,
    ) -> str:
        """Persist contiguous delivery progress so restart resumes after the last ack."""

        if sequence <= 0:
            raise ValueError("sequence must be greater than zero")
        completed_at = delivered_at or datetime.now(UTC)
        async with self._session_factory() as session, session.begin():
            row = await session.scalar(
                select(OutboxEvent).where(OutboxEvent.id == outbox_id).with_for_update()
            )
            if row is None or row.delivery_status != "leased" or row.leased_by != worker_id:
                raise OutboxLeaseConflictError("outbox lease no longer belongs to this worker")
            if sequence <= row.delivered_sequence:
                return row.delivery_status
            if sequence != row.delivered_sequence + 1:
                raise OutboxLeaseConflictError("outbox frames must be acknowledged contiguously")
            terminal_sequence = _row_terminal_sequence(row)
            if sequence > terminal_sequence:
                raise OutboxLeaseConflictError("outbox sequence exceeds the terminal frame")
            row.delivered_sequence = sequence
            if sequence == terminal_sequence:
                row.delivery_status = "delivered"
                row.delivered_at = completed_at
                row.leased_by = None
                row.lease_expires_at = None
                row.last_error_code = None
            await session.flush()
            return row.delivery_status

    async def mark_retry(
        self,
        outbox_id: UUID,
        *,
        worker_id: str,
        error_code: str,
        next_attempt_at: datetime,
        max_attempts: int,
    ) -> str:
        if max_attempts <= 0:
            raise ValueError("max_attempts must be greater than zero")
        async with self._session_factory() as session, session.begin():
            row = await session.scalar(
                select(OutboxEvent).where(OutboxEvent.id == outbox_id).with_for_update()
            )
            if row is None or row.delivery_status != "leased" or row.leased_by != worker_id:
                raise OutboxLeaseConflictError("outbox lease no longer belongs to this worker")
            row.retry_count += 1
            row.delivery_status = "dead" if row.retry_count >= max_attempts else "pending"
            row.next_attempt_at = next_attempt_at
            row.leased_by = None
            row.lease_expires_at = None
            row.last_error_code = error_code
            await session.flush()
            return row.delivery_status


def _record(row: OutboxEvent) -> OutboxBundleRecord:
    payload = dict(row.payload)
    required_payload_fields = {
        "message_id",
        "text",
        "attachments",
        "generation_mode",
        "failure_code",
        "invalidated_interaction_ids",
    }
    if set(payload) != required_payload_fields:
        raise OutboxPersistenceError("V4 outbox payload does not match its frozen shape")
    text = payload["text"]
    attachments = payload["attachments"]
    invalidated_interaction_ids = payload["invalidated_interaction_ids"]
    generation_mode = payload["generation_mode"]
    if (
        row.event_type != "assistant.message"
        or payload["message_id"] != str(row.message_id)
        or not isinstance(text, str)
        or not text
        or not isinstance(attachments, list)
        or not isinstance(invalidated_interaction_ids, list)
        or any(not isinstance(item, str) for item in invalidated_interaction_ids)
        or generation_mode not in {"qwen", "fallback"}
        or canonical_json_hash(payload) != row.content_hash
    ):
        raise OutboxPersistenceError("V4 outbox payload diverges from its authoritative message")

    algorithm_prefix = "unicode-codepoint-v1:size="
    if not row.chunking_algorithm_version.startswith(algorithm_prefix):
        raise OutboxPersistenceError("V4 outbox uses an unsupported chunking algorithm")
    try:
        chunk_size = int(row.chunking_algorithm_version.removeprefix(algorithm_prefix))
        expected_chunks, expected_algorithm = stable_text_chunks(text, chunk_size=chunk_size)
    except (TypeError, ValueError) as error:
        raise OutboxPersistenceError("V4 outbox has an invalid chunking algorithm") from error
    if list(row.chunks) != expected_chunks or row.chunking_algorithm_version != expected_algorithm:
        raise OutboxPersistenceError("V4 outbox chunks diverge from its committed text")

    for attachment in attachments:
        _attachment_identity(attachment)
    terminal_sequence = 2 + len(expected_chunks) + len(attachments) + 1
    if not 0 <= row.delivered_sequence <= terminal_sequence:
        raise OutboxPersistenceError("V4 outbox delivery cursor is outside its frame range")
    if (row.delivery_status == "delivered") != (row.delivered_sequence == terminal_sequence):
        raise OutboxPersistenceError("V4 outbox terminal delivery state is inconsistent")
    try:
        terminal = ConversationEventV4.model_validate(row.terminal_event).root
    except ValidationError as error:
        raise OutboxPersistenceError("V4 outbox terminal event failed validation") from error
    expected_event_id = stable_event_id(
        row.id,
        event_type="assistant.completed",
        content_hash=row.content_hash,
    )
    if not isinstance(terminal, AssistantCompletedEvent) or (
        terminal.event_id != str(expected_event_id)
        or terminal.sequence != terminal_sequence
        or terminal.message_id != str(row.message_id)
        or terminal.generation_id != str(row.generation_id)
        or terminal.trip_id != str(row.trip_id)
        or terminal.turn_id != str(row.turn_id)
        or terminal.state_version != row.committed_state_version
        or terminal.content_hash != row.content_hash
        or terminal.outbox_cursor != row.cursor
        or terminal.generation_mode != generation_mode
    ):
        raise OutboxPersistenceError("V4 outbox terminal event diverges from its bundle")
    return OutboxBundleRecord(
        outbox_id=row.id,
        cursor=row.cursor,
        trip_id=row.trip_id,
        turn_id=row.turn_id,
        generation_id=row.generation_id,
        message_id=row.message_id,
        committed_state_version=row.committed_state_version,
        payload=payload,
        chunks=tuple(row.chunks),
        terminal_event=dict(row.terminal_event),
        content_hash=row.content_hash,
        chunking_algorithm_version=row.chunking_algorithm_version,
        retry_count=row.retry_count,
        delivered_sequence=row.delivered_sequence,
        committed_at=(
            row.created_at
            if row.created_at.tzinfo is not None
            else row.created_at.replace(tzinfo=UTC)
        ),
    )


def _row_terminal_sequence(row: OutboxEvent) -> int:
    attachments = row.payload.get("attachments")
    if not isinstance(attachments, list):
        raise OutboxPersistenceError("V4 outbox attachments are invalid")
    return 2 + len(row.chunks) + len(attachments) + 1
