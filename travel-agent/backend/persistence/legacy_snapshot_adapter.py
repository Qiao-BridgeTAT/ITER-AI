"""Lossless V2/V3 snapshot reads and explicit V4 upgrade preparation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal
from uuid import UUID

from pydantic import ValidationError

from backend.contracts.v4.conversation import AssistantCompletedEvent, ConversationEventV4
from backend.contracts.v4.state import (
    DiscoveryRuntimeState,
    TripSemanticState,
    V4TripStateEnvelope,
)
from backend.contracts.versions import V4_PROTOCOL_VERSION, V4_SCHEMA_VERSION
from backend.persistence.models import (
    DiscoveryRuntimeStateVersion,
    TripSemanticStateVersion,
    TripSnapshot,
)
from backend.persistence.outbox_repository import canonical_json_hash


class SnapshotCompatibilityError(ValueError):
    pass


@dataclass(frozen=True)
class PersistedTripProjection:
    kind: Literal["legacy", "v4"]
    trip_id: UUID
    state_version: int
    schema_version: str
    trip_snapshot: dict[str, Any]
    semantic_state: dict[str, Any] | None
    discovery_runtime_state: dict[str, Any] | None
    terminal_status: str | None
    terminal_event: dict[str, Any] | None
    message_cursor: UUID | None
    outbox_cursor: str | None
    requires_explicit_upgrade: bool


@dataclass(frozen=True)
class ExplicitV4UpgradeSeed:
    trip_id: UUID
    base_state_version: int
    next_state_version: int
    legacy_snapshot: dict[str, Any]
    semantic_state: dict[str, Any]
    discovery_runtime_state: dict[str, Any]
    provenance: dict[str, Any]


def decode_persisted_snapshot(
    snapshot: TripSnapshot,
    *,
    semantic_state: TripSemanticStateVersion | None = None,
    discovery_runtime_state: DiscoveryRuntimeStateVersion | None = None,
) -> PersistedTripProjection:
    """Read old rows losslessly; V4 rows require both same-version state projections."""

    if snapshot.snapshot_kind != "v4":
        return PersistedTripProjection(
            kind="legacy",
            trip_id=snapshot.trip_id,
            state_version=snapshot.state_version,
            schema_version=snapshot.schema_version,
            trip_snapshot=dict(snapshot.snapshot),
            semantic_state=None,
            discovery_runtime_state=None,
            terminal_status=snapshot.terminal_status,
            terminal_event=(
                dict(snapshot.terminal_event) if snapshot.terminal_event is not None else None
            ),
            message_cursor=snapshot.message_cursor,
            outbox_cursor=snapshot.outbox_cursor,
            requires_explicit_upgrade=True,
        )
    if semantic_state is None or discovery_runtime_state is None:
        raise SnapshotCompatibilityError("V4 snapshot is missing one of its two state projections")
    versions = {
        snapshot.state_version,
        semantic_state.state_version,
        discovery_runtime_state.state_version,
    }
    trip_ids = {snapshot.trip_id, semantic_state.trip_id, discovery_runtime_state.trip_id}
    if len(versions) != 1 or len(trip_ids) != 1:
        raise SnapshotCompatibilityError("V4 snapshot projections do not share trip and version")
    if (
        snapshot.schema_version != V4_SCHEMA_VERSION
        or semantic_state.schema_version != V4_SCHEMA_VERSION
        or discovery_runtime_state.schema_version != V4_SCHEMA_VERSION
        or snapshot.protocol_version != V4_PROTOCOL_VERSION
    ):
        raise SnapshotCompatibilityError("V4 snapshot uses an unsupported protocol or schema")
    if (
        snapshot.turn_id is None
        or snapshot.message_cursor is None
        or snapshot.outbox_cursor is None
        or snapshot.terminal_status != "committed"
        or snapshot.terminal_event is None
    ):
        raise SnapshotCompatibilityError("V4 snapshot is missing committed conversation metadata")
    try:
        envelope = V4TripStateEnvelope.model_validate(
            snapshot.snapshot,
            context={"restore_historical_semantic_state": True},
        )
        typed_semantic = TripSemanticState.model_validate(
            semantic_state.payload,
            context={"restore_historical_semantic_state": True},
        )
        typed_runtime = DiscoveryRuntimeState.model_validate(discovery_runtime_state.payload)
        terminal = ConversationEventV4.model_validate(snapshot.terminal_event).root
    except ValidationError as error:
        raise SnapshotCompatibilityError(
            "V4 snapshot payload failed contract validation"
        ) from error
    if (
        envelope.semantic_state != typed_semantic
        or envelope.discovery_runtime_state != typed_runtime
    ):
        raise SnapshotCompatibilityError("V4 snapshot envelope and projections diverge")
    if semantic_state.content_hash != canonical_json_hash(
        semantic_state.payload
    ) or discovery_runtime_state.content_hash != canonical_json_hash(
        discovery_runtime_state.payload
    ):
        raise SnapshotCompatibilityError("V4 snapshot projection content hash mismatch")
    if not isinstance(terminal, AssistantCompletedEvent) or (
        terminal.trip_id != str(snapshot.trip_id)
        or terminal.turn_id != str(snapshot.turn_id)
        or terminal.message_id != str(snapshot.message_cursor)
        or terminal.state_version != snapshot.state_version
        or terminal.outbox_cursor != snapshot.outbox_cursor
    ):
        raise SnapshotCompatibilityError("V4 snapshot terminal event diverges from its cursors")
    terminal_payload = ConversationEventV4(root=terminal).model_dump(mode="json")
    return PersistedTripProjection(
        kind="v4",
        trip_id=snapshot.trip_id,
        state_version=snapshot.state_version,
        schema_version=snapshot.schema_version,
        trip_snapshot=envelope.model_dump(mode="json"),
        semantic_state=typed_semantic.model_dump(mode="json"),
        discovery_runtime_state=typed_runtime.model_dump(mode="json"),
        terminal_status=snapshot.terminal_status,
        terminal_event=terminal_payload,
        message_cursor=snapshot.message_cursor,
        outbox_cursor=snapshot.outbox_cursor,
        requires_explicit_upgrade=False,
    )


def prepare_explicit_v4_upgrade(
    legacy: PersistedTripProjection,
    *,
    semantic_state: dict[str, Any],
    discovery_runtime_state: dict[str, Any],
    requested_by: str,
) -> ExplicitV4UpgradeSeed:
    """Validate caller-supplied V4 projections without inferring legacy confirmations."""

    if legacy.kind != "legacy" or not legacy.requires_explicit_upgrade:
        raise SnapshotCompatibilityError("only a legacy snapshot can enter explicit V4 upgrade")
    next_version = legacy.state_version + 1
    try:
        typed_semantic = TripSemanticState.model_validate(
            semantic_state,
            context={"restore_historical_semantic_state": True},
        )
        typed_runtime = DiscoveryRuntimeState.model_validate(discovery_runtime_state)
    except ValidationError as error:
        raise SnapshotCompatibilityError("upgrade seed failed the V4 state contract") from error
    if {
        typed_semantic.state_version,
        typed_runtime.state_version,
    } != {next_version}:
        raise SnapshotCompatibilityError("V4 states must advance exactly one state version")
    if {typed_semantic.trip_id, typed_runtime.trip_id} != {str(legacy.trip_id)}:
        raise SnapshotCompatibilityError("V4 states belong to another trip")
    try:
        envelope = V4TripStateEnvelope(
            semantic_state=typed_semantic,
            discovery_runtime_state=typed_runtime,
        )
    except ValidationError as error:
        raise SnapshotCompatibilityError("upgrade seed failed the V4 state contract") from error
    return ExplicitV4UpgradeSeed(
        trip_id=legacy.trip_id,
        base_state_version=legacy.state_version,
        next_state_version=next_version,
        legacy_snapshot=dict(legacy.trip_snapshot),
        semantic_state=envelope.semantic_state.model_dump(mode="json"),
        discovery_runtime_state=envelope.discovery_runtime_state.model_dump(mode="json"),
        provenance={
            "adapter": "v2-v3-explicit-to-v4-v1",
            "requested_by": requested_by,
            "legacy_schema_version": legacy.schema_version,
            "legacy_state_version": legacy.state_version,
            "inferred_confirmations": False,
        },
    )
