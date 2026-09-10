"""Encrypted append-only audit trail for model and conversation execution.

The ordinary application logger must remain metadata-only.  This module owns the
separate private payload store used to reproduce model, parsing, Guard, fallback,
and authoritative-output behaviour without leaking those payloads to stdout.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import re
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID, uuid4

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from pydantic import BaseModel

MODEL_AUDIT_EVENT_VERSION = "1.0.0"
MODEL_AUDIT_ENVELOPE_VERSION = "iter-ai-private-model-audit-v1"
_AUDIT_AAD_PREFIX = b"iter-ai:model-audit:v1:"
_LOGGER = logging.getLogger(__name__)
_CURRENT_EXECUTION: ContextVar[ModelAuditExecutionContext | None] = ContextVar(
    "model_audit_execution",
    default=None,
)
_SECRET_KEY_PARTS = frozenset(
    {
        "api_key",
        "apikey",
        "access_key",
        "authorization",
        "card_signature",
        "cookie",
        "database_url",
        "idempotency_key",
        "interaction_signature",
        "password",
        "security_code",
        "security_key",
        "secret",
        "session_token",
        "signature",
        "signed_payload",
        "token",
    }
)
_PRIVATE_REASONING_KEYS = frozenset(
    {
        "reasoning_content",
        "reasoning_text",
        "thinking_content",
        "chain_of_thought",
    }
)
_BEARER_PATTERN = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+\-/=]+")
_API_KEY_PATTERN = re.compile(r"\bsk-[A-Za-z0-9._\-]{8,}\b")
_DATABASE_URL_PATTERN = re.compile(
    r"(?i)\b(?:postgres(?:ql)?|redis|mysql)(?:\+[a-z0-9_]+)?://[^\s\"']+"
)


class ModelAuditError(RuntimeError):
    """Audit persistence failed; real model execution must fail closed."""


class ModelAuditRecorder(Protocol):
    @property
    def enabled(self) -> bool: ...

    async def record(
        self,
        event_kind: str,
        *,
        llm_call_id: str | None,
        payload: Mapping[str, Any],
    ) -> str: ...


class NoopModelAuditRecorder:
    """Explicit test-only/default recorder for directly constructed gateways."""

    @property
    def enabled(self) -> bool:
        return False

    async def record(
        self,
        event_kind: str,
        *,
        llm_call_id: str | None,
        payload: Mapping[str, Any],
    ) -> str:
        del event_kind, llm_call_id, payload
        return ""


@dataclass(slots=True)
class ModelAuditExecutionContext:
    """One admitted user/Planner turn and the model calls caused by it."""

    trace_id: str
    trip_id: str
    turn_id: str
    generation_id: str
    user_message_id: str
    input_kind: str
    user_input_full: Any = field(repr=False)
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    call_ids: list[str] = field(default_factory=list)
    _stage_attempts: dict[str, int] = field(default_factory=dict, repr=False)
    _last_call_by_stage: dict[str, str] = field(default_factory=dict, repr=False)
    _last_model_by_stage: dict[str, str] = field(default_factory=dict, repr=False)
    _recorders: list[ModelAuditRecorder] = field(default_factory=list, repr=False)

    def register_call(
        self,
        recorder: ModelAuditRecorder,
        *,
        call_id: str,
        stage: str,
        model: str,
        repair: bool,
        explicit_attempt: int | None,
        explicit_parent_call_id: str | None,
        explicit_repair_of_call_id: str | None,
        explicit_model_switch_from: str | None,
    ) -> dict[str, Any]:
        attempt = explicit_attempt or self._stage_attempts.get(stage, 0) + 1
        previous_stage_call = self._last_call_by_stage.get(stage)
        parent_call_id = explicit_parent_call_id or (self.call_ids[-1] if self.call_ids else None)
        repair_of_call_id = explicit_repair_of_call_id or (previous_stage_call if repair else None)
        prior_model = self._last_model_by_stage.get(stage)
        model_switch_from = explicit_model_switch_from or (
            prior_model if prior_model is not None and prior_model != model else None
        )
        self._stage_attempts[stage] = max(attempt, self._stage_attempts.get(stage, 0))
        self._last_call_by_stage[stage] = call_id
        self._last_model_by_stage[stage] = model
        self.call_ids.append(call_id)
        if recorder.enabled and all(existing is not recorder for existing in self._recorders):
            self._recorders.append(recorder)
        return {
            "trace_id": self.trace_id,
            "trip_id": self.trip_id,
            "turn_id": self.turn_id,
            "generation_id": self.generation_id,
            "user_message_id": self.user_message_id,
            "input_kind": self.input_kind,
            "user_input_full": self.user_input_full,
            "parent_llm_call_id": parent_call_id,
            "repair_of_call_id": repair_of_call_id,
            "attempt": attempt,
            "model_switch_from": model_switch_from,
        }

    @property
    def recorders(self) -> tuple[ModelAuditRecorder, ...]:
        return tuple(self._recorders)

    def attach_recorder(self, recorder: ModelAuditRecorder) -> None:
        if recorder.enabled and all(existing is not recorder for existing in self._recorders):
            self._recorders.append(recorder)


@contextmanager
def bind_model_audit_execution(
    context: ModelAuditExecutionContext,
) -> Iterator[ModelAuditExecutionContext]:
    """Bind turn metadata across nested LangGraph and async model calls."""

    token = _CURRENT_EXECUTION.set(context)
    try:
        yield context
    finally:
        _CURRENT_EXECUTION.reset(token)


def current_model_audit_execution() -> ModelAuditExecutionContext | None:
    return _CURRENT_EXECUTION.get()


def activate_model_audit_execution(
    context: ModelAuditExecutionContext,
) -> Token[ModelAuditExecutionContext | None]:
    return _CURRENT_EXECUTION.set(context)


def reset_model_audit_execution(token: Token[ModelAuditExecutionContext | None]) -> None:
    _CURRENT_EXECUTION.reset(token)


class EncryptedFileModelAuditRecorder:
    """Private, append-only AES-GCM event objects stored outside Git."""

    def __init__(self, root: Path, encryption_secret: str, *, ttl_days: int = 30) -> None:
        if len(encryption_secret) < 16:
            raise ValueError("model audit encryption secret must contain at least 16 characters")
        if ttl_days < 1 or ttl_days > 365:
            raise ValueError("model audit TTL must be between 1 and 365 days")
        self._root = root.resolve()
        self._events = self._root / "events"
        self._key = hashlib.sha256(
            b"iter-ai:model-audit:key:v1\0" + encryption_secret.encode("utf-8")
        ).digest()
        self._ttl = timedelta(days=ttl_days)
        self._lock = asyncio.Lock()
        self._retention_lock = asyncio.Lock()
        self._retention_stop = asyncio.Event()
        self._retention_task: asyncio.Task[None] | None = None

    @property
    def enabled(self) -> bool:
        return True

    @property
    def root(self) -> Path:
        return self._root

    async def record(
        self,
        event_kind: str,
        *,
        llm_call_id: str | None,
        payload: Mapping[str, Any],
    ) -> str:
        event_id = str(uuid4())
        now = datetime.now(UTC)
        private_event = sanitize_audit_value(
            {
                "audit_event_version": MODEL_AUDIT_EVENT_VERSION,
                "event_id": event_id,
                "event_kind": event_kind,
                "llm_call_id": llm_call_id,
                "occurred_at": now.isoformat(),
                **dict(payload),
            }
        )
        canonical = _canonical_bytes(private_event)
        nonce = os.urandom(12)
        aad = _AUDIT_AAD_PREFIX + event_id.encode("ascii")
        ciphertext = AESGCM(self._key).encrypt(nonce, canonical, aad)
        envelope = {
            "format": MODEL_AUDIT_ENVELOPE_VERSION,
            "event_id": event_id,
            "event_kind": event_kind,
            "llm_call_id": llm_call_id,
            "occurred_at": now.isoformat(),
            "expires_at": (now + self._ttl).isoformat(),
            "payload_sha256": hashlib.sha256(canonical).hexdigest(),
            "nonce_b64": base64.b64encode(nonce).decode("ascii"),
            "ciphertext_b64": base64.b64encode(ciphertext).decode("ascii"),
        }
        try:
            async with self._lock:
                await asyncio.to_thread(self._append_envelope, now, event_id, envelope)
        except (OSError, ValueError, TypeError) as error:
            raise ModelAuditError("private model audit write failed") from error
        return event_id

    async def read_events(
        self,
        *,
        access_reason: str,
        llm_call_id: str | None = None,
        trace_id: str | None = None,
        trip_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Decrypt a bounded audit selection and append a separate access event."""

        if not access_reason.strip():
            raise ValueError("audit access reason is required")
        if llm_call_id is None and trace_id is None and trip_id is None:
            raise ValueError("at least one audit scope is required")
        try:
            events = await asyncio.to_thread(self._read_all_events)
        except (OSError, ValueError, TypeError, KeyError) as error:
            raise ModelAuditError("private model audit read failed") from error
        selected = [
            event
            for event in events
            if (
                llm_call_id is None
                or event.get("llm_call_id") == llm_call_id
                or llm_call_id in event.get("llm_call_ids", [])
            )
            and (trace_id is None or event.get("trace_id") == trace_id)
            and (trip_id is None or event.get("trip_id") == trip_id)
        ]
        await self.record(
            "audit_accessed",
            llm_call_id=llm_call_id,
            payload={
                "access_reason": access_reason,
                "trace_filter": trace_id,
                "trip_filter": trip_id,
                "selected_event_ids": [event.get("event_id") for event in selected],
                "selected_count": len(selected),
            },
        )
        return selected

    async def purge_expired(self, *, now: datetime | None = None) -> int:
        cutoff = now or datetime.now(UTC)
        try:
            async with self._retention_lock:
                return await asyncio.to_thread(self._purge_expired, cutoff)
        except (OSError, ValueError, TypeError, KeyError) as error:
            raise ModelAuditError("private model audit retention cleanup failed") from error

    def start_retention_maintenance(self, *, interval_seconds: float = 3600) -> None:
        """Run retention separately from durable appends; called by API lifespan."""

        if interval_seconds <= 0:
            raise ValueError("audit retention interval must be positive")
        if self._retention_task is not None:
            return
        self._retention_stop.clear()
        self._retention_task = asyncio.create_task(
            self._maintain_retention(interval_seconds), name="model-audit-retention"
        )

    async def close(self) -> None:
        self._retention_stop.set()
        if self._retention_task is not None:
            # Do not abandon a filesystem worker halfway through retention.
            await self._retention_task
            self._retention_task = None

    async def _maintain_retention(self, interval_seconds: float) -> None:
        while not self._retention_stop.is_set():
            try:
                await self.purge_expired()
            except ModelAuditError:
                # Metadata only. Retry on the next tick; writes still fail closed
                # independently, and reads never expose expired events.
                _LOGGER.error("private model audit retention cleanup failed; retry scheduled")
            try:
                await asyncio.wait_for(self._retention_stop.wait(), timeout=interval_seconds)
            except TimeoutError:
                continue

    def _append_envelope(
        self,
        occurred_at: datetime,
        event_id: str,
        envelope: Mapping[str, Any],
    ) -> None:
        day = occurred_at.date().isoformat()
        directory = self._events / day
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self._root, 0o700)
        os.chmod(self._events, 0o700)
        os.chmod(directory, 0o700)
        path = directory / f"{time.time_ns()}-{event_id}.audit.json"
        pending = path.with_suffix(".pending")
        data = _canonical_bytes(envelope) + b"\n"
        descriptor = os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            # Publish only complete envelopes. link is atomic and refuses to
            # overwrite an existing event, even while reads/retention run.
            os.link(pending, path)
        finally:
            pending.unlink(missing_ok=True)

    def _read_all_events(self) -> list[dict[str, Any]]:
        if not self._events.exists():
            return []
        events: list[dict[str, Any]] = []
        for path in sorted(self._events.glob("*/*.audit.json")):
            try:
                envelope = json.loads(path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                continue  # Retention in another task/process removed this event.
            if envelope.get("format") != MODEL_AUDIT_ENVELOPE_VERSION:
                raise ValueError("unsupported private audit envelope")
            if datetime.fromisoformat(str(envelope["expires_at"])) <= datetime.now(UTC):
                continue
            event_id = str(envelope["event_id"])
            nonce = base64.b64decode(envelope["nonce_b64"], validate=True)
            ciphertext = base64.b64decode(envelope["ciphertext_b64"], validate=True)
            canonical = AESGCM(self._key).decrypt(
                nonce,
                ciphertext,
                _AUDIT_AAD_PREFIX + event_id.encode("ascii"),
            )
            if hashlib.sha256(canonical).hexdigest() != envelope["payload_sha256"]:
                raise ValueError("private audit payload hash mismatch")
            value = json.loads(canonical)
            if not isinstance(value, dict):
                raise ValueError("private audit event must be an object")
            events.append(value)
        return events

    def _purge_expired(self, now: datetime) -> int:
        if not self._events.exists():
            return 0
        removed = 0
        for path in sorted(self._events.glob("*/*.audit.json")):
            try:
                envelope = json.loads(path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                continue
            expires_at = datetime.fromisoformat(str(envelope["expires_at"]))
            if expires_at <= now:
                try:
                    path.unlink()
                except FileNotFoundError:
                    continue
                removed += 1
        return removed


def register_model_call(
    recorder: ModelAuditRecorder,
    *,
    call_id: str,
    stage: str,
    model: str,
    repair: bool = False,
    attempt: int | None = None,
    parent_call_id: str | None = None,
    repair_of_call_id: str | None = None,
    model_switch_from: str | None = None,
) -> dict[str, Any]:
    context = current_model_audit_execution()
    if context is None:
        return {
            "trace_id": None,
            "trip_id": None,
            "turn_id": None,
            "generation_id": None,
            "user_message_id": None,
            "input_kind": None,
            "user_input_full": None,
            "parent_llm_call_id": parent_call_id,
            "repair_of_call_id": repair_of_call_id,
            "attempt": attempt or 1,
            "model_switch_from": model_switch_from,
        }
    return context.register_call(
        recorder,
        call_id=call_id,
        stage=stage,
        model=model,
        repair=repair,
        explicit_attempt=attempt,
        explicit_parent_call_id=parent_call_id,
        explicit_repair_of_call_id=repair_of_call_id,
        explicit_model_switch_from=model_switch_from,
    )


async def record_execution_event(
    context: ModelAuditExecutionContext,
    event_kind: str,
    payload: Mapping[str, Any],
) -> None:
    """Link every call in a turn to its input, commit, failure, or delivery result."""

    base = {
        "trace_id": context.trace_id,
        "trip_id": context.trip_id,
        "turn_id": context.turn_id,
        "generation_id": context.generation_id,
        "user_message_id": context.user_message_id,
        "input_kind": context.input_kind,
        "user_input_full": context.user_input_full,
        **dict(payload),
        "llm_call_ids": list(dict.fromkeys(context.call_ids)),
    }
    for recorder in context.recorders:
        await recorder.record(event_kind, llm_call_id=None, payload=base)


async def record_model_call_annotation(
    gateway: object,
    llm_call_id: str | None,
    event_kind: str,
    payload: Mapping[str, Any],
) -> None:
    """Append downstream materialization/Guard information to one gateway call."""

    if llm_call_id is None:
        return
    recorder = getattr(gateway, "audit_recorder", None)
    if recorder is None or not getattr(recorder, "enabled", False):
        return
    context = current_model_audit_execution()
    linked: dict[str, Any] = {}
    if context is not None:
        linked = {
            "trace_id": context.trace_id,
            "trip_id": context.trip_id,
            "turn_id": context.turn_id,
            "generation_id": context.generation_id,
        }
    try:
        await recorder.record(
            event_kind,
            llm_call_id=llm_call_id,
            payload={**linked, **dict(payload)},
        )
    except ModelAuditError:
        raise
    except Exception as error:  # pragma: no cover - third-party recorder shield
        raise ModelAuditError("private model audit annotation failed") from error


def sanitize_audit_value(value: Any) -> Any:
    """Convert to JSON data while removing credentials and hidden reasoning first."""

    if isinstance(value, BaseModel):
        return sanitize_audit_value(value.model_dump(mode="json"))
    if isinstance(value, Enum):
        return sanitize_audit_value(value.value)
    if isinstance(value, (datetime, UUID, Path)):
        return str(value)
    if isinstance(value, Mapping):
        sanitized: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key)
            normalized = key.casefold().replace("-", "_")
            if normalized in _PRIVATE_REASONING_KEYS:
                sanitized[key] = "[REDACTED_PRIVATE_REASONING]"
            elif _is_secret_key(normalized):
                sanitized[key] = "[REDACTED_SECRET]"
            else:
                sanitized[key] = sanitize_audit_value(raw_value)
        return sanitized
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [sanitize_audit_value(item) for item in value]
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("ascii")
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("{", "[")):
            try:
                decoded = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                pass
            else:
                return json.dumps(
                    sanitize_audit_value(decoded),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
        value = _BEARER_PATTERN.sub("Bearer [REDACTED_SECRET]", value)
        value = _API_KEY_PATTERN.sub("[REDACTED_SECRET]", value)
        return _DATABASE_URL_PATTERN.sub("[REDACTED_DATABASE_URL]", value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)


def private_reasoning_was_redacted(value: Any) -> bool:
    if isinstance(value, Mapping):
        for raw_key, raw_value in value.items():
            key = str(raw_key).casefold().replace("-", "_")
            if key in _PRIVATE_REASONING_KEYS or private_reasoning_was_redacted(raw_value):
                return True
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(private_reasoning_was_redacted(item) for item in value)
    return False


def canonical_audit_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(sanitize_audit_value(value))).hexdigest()


def _is_secret_key(normalized: str) -> bool:
    return normalized in _SECRET_KEY_PARTS or any(
        normalized.endswith(f"_{marker}")
        for marker in (
            "access_key",
            "api_key",
            "authorization",
            "cookie",
            "password",
            "secret",
            "session_token",
            "signature",
        )
    )


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


__all__ = [
    "EncryptedFileModelAuditRecorder",
    "MODEL_AUDIT_EVENT_VERSION",
    "ModelAuditError",
    "ModelAuditExecutionContext",
    "ModelAuditRecorder",
    "NoopModelAuditRecorder",
    "activate_model_audit_execution",
    "bind_model_audit_execution",
    "canonical_audit_hash",
    "current_model_audit_execution",
    "private_reasoning_was_redacted",
    "record_execution_event",
    "record_model_call_annotation",
    "reset_model_audit_execution",
    "register_model_call",
    "sanitize_audit_value",
]
