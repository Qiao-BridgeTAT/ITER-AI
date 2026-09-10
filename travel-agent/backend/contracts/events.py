"""P0-11 server events and JSON Patch payloads."""

from __future__ import annotations

from typing import Annotated, Any, Literal, cast
from uuid import UUID

from pydantic import (
    AwareDatetime,
    ConfigDict,
    Field,
    JsonValue,
    RootModel,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)

from backend.contracts.base import ContractModel
from backend.contracts.commands import IdempotencyKey, ProtocolVersion, SchemaVersion
from backend.contracts.common import NonEmptyText, ShortText
from backend.contracts.conversation import ConversationMessage
from backend.contracts.enums import (
    EventType,
    GenerationStatus,
    GestureKind,
    ItineraryEntryKind,
    TransportMode,
)
from backend.contracts.itinerary import Itinerary, PlanningIssue
from backend.contracts.places import Gcj02Coordinates

JSON_PATCH_SCHEMA_RULE: dict[str, Any] = {
    "allOf": [
        {
            "if": {"properties": {"op": {"enum": ["add", "replace", "test"]}}},
            "then": {"properties": {"value": {}}, "required": ["value"]},
        },
        {
            "if": {"properties": {"op": {"enum": ["move", "copy"]}}},
            "then": {"properties": {"from": {"type": "string"}}, "required": ["from"]},
        },
        {
            "if": {"properties": {"op": {"enum": ["remove", "move", "copy"]}}},
            "then": {"not": {"required": ["value"]}},
        },
        {
            "if": {"properties": {"op": {"enum": ["add", "remove", "replace", "test"]}}},
            "then": {"not": {"required": ["from"]}},
        },
    ]
}

EVENT_ENVELOPE_SCHEMA_RULE: dict[str, Any] = {
    "allOf": [
        {
            "if": {"properties": {"generation_id": {"type": "null"}}},
            "then": {"properties": {"sequence": {"const": 0, "type": "integer"}}},
            "else": {"properties": {"sequence": {"minimum": 1, "type": "integer"}}},
        }
    ]
}

STATE_PATCH_EVENT_SCHEMA_RULE: dict[str, Any] = {
    **EVENT_ENVELOPE_SCHEMA_RULE,
    "x-travel-state-version-increment": {
        "baseField": "base_state_version",
        "stateField": "state_version",
        "increment": 1,
    },
}


class JsonPatchOperation(ContractModel):
    model_config = ConfigDict(json_schema_extra=JSON_PATCH_SCHEMA_RULE)

    op: Literal["add", "remove", "replace", "move", "copy", "test"]
    path: str
    from_: str | None = Field(default=None, alias="from")
    value: JsonValue | None = None

    @model_serializer(mode="wrap")
    def serialize_wire_shape(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        payload = cast(dict[str, Any], handler(self))
        if self.from_ is None:
            payload.pop("from", None)
            payload.pop("from_", None)
        if "value" not in self.model_fields_set:
            payload.pop("value", None)
        return payload

    @model_validator(mode="after")
    def operation_fields_match_rfc_6902(self) -> JsonPatchOperation:
        if self.path and not self.path.startswith("/"):
            raise ValueError("JSON Patch path must be empty or start with a slash")
        has_value = "value" in self.model_fields_set
        has_from = self.from_ is not None
        if self.op in {"add", "replace", "test"} and not has_value:
            raise ValueError(f"{self.op} operation requires value")
        if self.op in {"move", "copy"} and not has_from:
            raise ValueError(f"{self.op} operation requires from")
        if self.op in {"remove", "move", "copy"} and has_value:
            raise ValueError(f"{self.op} operation cannot include value")
        if self.op not in {"move", "copy"} and has_from:
            raise ValueError(f"{self.op} operation cannot include from")
        return self


class StatePatchPayload(ContractModel):
    patch: list[JsonPatchOperation] = Field(min_length=1)


class StreamTokenPayload(ContractModel):
    token: NonEmptyText


class GestureReadyPayload(ContractModel):
    gesture_id: NonEmptyText
    kind: GestureKind
    title: NonEmptyText
    data: dict[str, JsonValue] = Field(default_factory=dict)
    conversation_message: ConversationMessage | None = None


class MapMarker(ContractModel):
    place_id: UUID
    label: NonEmptyText
    kind: ItineraryEntryKind
    coordinates: Gcj02Coordinates


class MapRouteSegment(ContractModel):
    from_place_id: UUID
    to_place_id: UUID
    mode: TransportMode
    polyline: list[Gcj02Coordinates] = Field(min_length=2)


class MapUpdatePayload(ContractModel):
    model_config = ConfigDict(json_schema_extra={"x-travel-map-update": True})

    selected_day_index: int = Field(ge=0, le=4, strict=True)
    markers: list[MapMarker] = Field(default_factory=list)
    routes: list[MapRouteSegment] = Field(default_factory=list)

    @model_validator(mode="after")
    def map_references_are_self_contained(self) -> MapUpdatePayload:
        marker_ids = [marker.place_id for marker in self.markers]
        if len(set(marker_ids)) != len(marker_ids):
            raise ValueError("map marker place IDs must be unique")
        marker_id_set = set(marker_ids)
        route_keys = [(route.from_place_id, route.to_place_id, route.mode) for route in self.routes]
        if len(set(route_keys)) != len(route_keys):
            raise ValueError("map route endpoint and mode keys must be unique")
        if any(
            route.from_place_id not in marker_id_set or route.to_place_id not in marker_id_set
            for route in self.routes
        ):
            raise ValueError("map routes must reference visible map markers")
        return self


class ItineraryUpdatePayload(ContractModel):
    """Compatibility-only legacy projection; formal clients use trip_state.published_plan."""

    model_config = ConfigDict(json_schema_extra={"deprecated": True})

    itinerary: Itinerary


class GenerationStatusPayload(ContractModel):
    status: GenerationStatus
    message: ShortText | None = None


class IssueEventPayload(ContractModel):
    issue: PlanningIssue


class ErrorEventPayload(ContractModel):
    code: NonEmptyText
    message: ShortText
    retryable: bool
    snapshot_required: bool = False


class EventEnvelope(ContractModel):
    model_config = ConfigDict(json_schema_extra=EVENT_ENVELOPE_SCHEMA_RULE)

    protocol_version: ProtocolVersion
    schema_version: SchemaVersion
    event_id: UUID
    request_id: UUID
    generation_id: UUID | None
    sequence: int = Field(ge=0, strict=True)
    base_state_version: int = Field(ge=0, strict=True)
    state_version: int = Field(ge=0, strict=True)
    idempotency_key: IdempotencyKey
    timestamp: AwareDatetime

    @model_validator(mode="after")
    def sequence_matches_generation(self) -> EventEnvelope:
        if self.generation_id is None and self.sequence != 0:
            raise ValueError("non-generation events must use sequence zero")
        if self.generation_id is not None and self.sequence < 1:
            raise ValueError("generation events must use a positive sequence")
        return self


class StatePatchEvent(EventEnvelope):
    model_config = ConfigDict(json_schema_extra=STATE_PATCH_EVENT_SCHEMA_RULE)

    type: Literal[EventType.STATE_PATCH]
    payload: StatePatchPayload

    @model_validator(mode="after")
    def patch_advances_one_stable_version(self) -> StatePatchEvent:
        if self.state_version != self.base_state_version + 1:
            raise ValueError("state patch must advance exactly one state version")
        return self


class StreamTokenEvent(EventEnvelope):
    type: Literal[EventType.STREAM_TOKEN]
    payload: StreamTokenPayload


class GestureReadyEvent(EventEnvelope):
    type: Literal[EventType.GESTURE_READY]
    payload: GestureReadyPayload


class MapUpdateEvent(EventEnvelope):
    type: Literal[EventType.MAP_UPDATE]
    payload: MapUpdatePayload


class ItineraryUpdateEvent(EventEnvelope):
    type: Literal[EventType.ITINERARY_UPDATE]
    payload: ItineraryUpdatePayload


class GenerationStatusEvent(EventEnvelope):
    generation_id: UUID
    sequence: int = Field(ge=1, strict=True)
    type: Literal[EventType.GENERATION_STATUS]
    payload: GenerationStatusPayload


class IssueEvent(EventEnvelope):
    type: Literal[EventType.ISSUE]
    payload: IssueEventPayload


class ErrorEvent(EventEnvelope):
    type: Literal[EventType.ERROR]
    payload: ErrorEventPayload


ServerEventValue = Annotated[
    StatePatchEvent
    | StreamTokenEvent
    | GestureReadyEvent
    | MapUpdateEvent
    | ItineraryUpdateEvent
    | GenerationStatusEvent
    | IssueEvent
    | ErrorEvent,
    Field(discriminator="type"),
]


class ServerEvent(RootModel[ServerEventValue]):
    """Discriminated union for every stage-0 WebSocket server event."""


P0_EVENT_CONTRACTS: tuple[type[Any], ...] = (ServerEvent,)
