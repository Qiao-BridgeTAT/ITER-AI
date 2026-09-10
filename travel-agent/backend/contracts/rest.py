"""P0-12 REST payloads, endpoint catalog, and ownership metadata."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Any
from uuid import UUID

from pydantic import AwareDatetime, ConfigDict, Field, HttpUrl, StringConstraints, model_validator

from backend.contracts.base import ContractModel
from backend.contracts.commands import IdempotencyKey
from backend.contracts.common import NonEmptyText, ShortText
from backend.contracts.enums import (
    ArtifactStatus,
    HttpMethod,
    OwnerType,
    RestAuthMode,
    RestOwnershipPolicy,
    RestResourceKind,
    TripPhase,
)
from backend.contracts.state import TripState

E164Phone = Annotated[str, StringConstraints(pattern=r"^\+[1-9]\d{7,14}$")]
SmsCode = Annotated[str, StringConstraints(pattern=r"^\d{4,8}$")]
NicknameText = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=20,
        pattern=r"^[^\r\n]+$",
    ),
]

EXPORT_VIEW_SCHEMA_RULE: dict[str, Any] = {
    "allOf": [
        {
            "if": {"properties": {"status": {"const": "ready"}}, "required": ["status"]},
            "then": {
                "properties": {"download_url": {"not": {"type": "null"}}},
                "required": ["download_url"],
            },
            "else": {"properties": {"download_url": {"type": "null"}}},
        }
    ]
}

REST_ENDPOINT_SCHEMA_RULE: dict[str, Any] = {
    "allOf": [
        {
            "if": {
                "properties": {"method": {"enum": ["POST", "PATCH", "DELETE"]}},
                "required": ["method"],
            },
            "then": {"properties": {"idempotency_required": {"const": True}}},
            "else": {"properties": {"idempotency_required": {"const": False}}},
        },
        {
            "if": {"properties": {"owner_check_required": {"const": True}}},
            "then": {
                "properties": {
                    "auth": {"const": "session"},
                    "resource_kind": {"not": {"type": "null"}},
                    "ownership_policy": {"not": {"const": "none"}},
                },
                "required": ["ownership_policy"],
            },
            "else": {"properties": {"ownership_policy": {"const": "none"}}},
        },
        {
            "if": {"properties": {"auth": {"const": "public"}}},
            "then": {
                "properties": {
                    "allowed_actor_types": {"type": "array", "maxItems": 0},
                    "ownership_policy": {"const": "none"},
                }
            },
            "else": {
                "properties": {"allowed_actor_types": {"type": "array", "minItems": 1}},
                "required": ["allowed_actor_types"],
            },
        },
    ]
}


def _unique(values: Sequence[str | UUID], field_name: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} must not contain duplicates")


class MutationRequestMeta(ContractModel):
    request_id: UUID
    idempotency_key: IdempotencyKey


class SmsSendRequest(ContractModel):
    phone: E164Phone


class SmsVerifyRequest(ContractModel):
    phone: E164Phone
    code: SmsCode


class AccountView(ContractModel):
    user_id: UUID
    masked_phone: NonEmptyText
    nickname: NicknameText | None = None
    created_at: AwareDatetime


class UpdateAccountRequest(ContractModel):
    nickname: NicknameText


class AnonymousSessionView(ContractModel):
    session_id: NonEmptyText
    expires_at: AwareDatetime


class AttachAnonymousTripRequest(ContractModel):
    anonymous_session_id: NonEmptyText
    trip_id: UUID


class CreateTripRequest(ContractModel):
    anonymous_session_id: NonEmptyText | None = None
    trip_id: UUID | None = None


class TripListItem(ContractModel):
    trip_id: UUID
    phase: TripPhase
    title: NonEmptyText
    updated_at: AwareDatetime


class TripListView(ContractModel):
    trips: list[TripListItem] = Field(default_factory=list)


class TripSnapshotView(ContractModel):
    state: TripState


class PreferenceValue(ContractModel):
    preference_id: UUID
    key: NonEmptyText
    value: ShortText
    active: bool = True
    updated_at: AwareDatetime


class PreferenceListView(ContractModel):
    preferences: list[PreferenceValue] = Field(default_factory=list)

    @model_validator(mode="after")
    def ids_are_unique(self) -> PreferenceListView:
        _unique([item.preference_id for item in self.preferences], "preference_id")
        return self


class PreferencePatchItem(ContractModel):
    preference_id: UUID
    value: ShortText
    active: bool = True


class PatchPreferencesRequest(ContractModel):
    preferences: list[PreferencePatchItem] = Field(min_length=1)

    @model_validator(mode="after")
    def ids_are_unique(self) -> PatchPreferencesRequest:
        _unique([item.preference_id for item in self.preferences], "preference_id")
        return self


class ConfirmPreferenceCandidatesRequest(ContractModel):
    candidate_ids: list[UUID] = Field(
        min_length=1,
        json_schema_extra={"uniqueItems": True},
    )

    @model_validator(mode="after")
    def ids_are_unique(self) -> ConfirmPreferenceCandidatesRequest:
        _unique(self.candidate_ids, "candidate_ids")
        return self


class CreateExportRequest(ContractModel):
    plan_version_id: UUID


class ExportView(ContractModel):
    model_config = ConfigDict(json_schema_extra=EXPORT_VIEW_SCHEMA_RULE)

    artifact_id: UUID
    trip_id: UUID
    plan_version_id: UUID
    status: ArtifactStatus
    download_url: HttpUrl | None = None
    expires_at: AwareDatetime | None = None

    @model_validator(mode="after")
    def ready_export_has_a_download(self) -> ExportView:
        if self.status is ArtifactStatus.READY and self.download_url is None:
            raise ValueError("ready export requires a download URL")
        if self.status is not ArtifactStatus.READY and self.download_url is not None:
            raise ValueError("only ready exports may expose a download URL")
        return self


class RestEndpointContract(ContractModel):
    model_config = ConfigDict(json_schema_extra=REST_ENDPOINT_SCHEMA_RULE)

    operation_id: NonEmptyText
    method: HttpMethod
    path: NonEmptyText
    auth: RestAuthMode
    resource_kind: RestResourceKind | None = None
    owner_check_required: bool
    allowed_actor_types: list[OwnerType] = Field(
        default_factory=list,
        json_schema_extra={"uniqueItems": True},
    )
    ownership_policy: RestOwnershipPolicy = RestOwnershipPolicy.NONE
    idempotency_required: bool
    request_model: NonEmptyText | None = None
    response_model: NonEmptyText | None = None

    @model_validator(mode="after")
    def security_flags_match_method_and_resource(self) -> RestEndpointContract:
        if not self.path.startswith("/"):
            raise ValueError("REST path must start with a slash")
        if any(segment in self.path for segment in ("quota", "invite", "share")):
            raise ValueError("stage 0 cannot expose quota, invite, or share endpoints")
        if self.method in {HttpMethod.POST, HttpMethod.PATCH, HttpMethod.DELETE}:
            if not self.idempotency_required:
                raise ValueError("every side-effecting REST endpoint requires idempotency")
        elif self.idempotency_required:
            raise ValueError("read-only REST endpoints do not require idempotency keys")
        if self.owner_check_required and (
            self.auth is not RestAuthMode.SESSION
            or self.resource_kind is None
            or self.ownership_policy is RestOwnershipPolicy.NONE
        ):
            raise ValueError("owner checks require a session and resource kind")
        if self.owner_check_required != (self.ownership_policy is not RestOwnershipPolicy.NONE):
            raise ValueError("ownership policy must match owner_check_required")
        if len(set(self.allowed_actor_types)) != len(self.allowed_actor_types):
            raise ValueError("allowed_actor_types must not contain duplicates")
        if self.auth is RestAuthMode.PUBLIC:
            if self.allowed_actor_types or self.ownership_policy is not RestOwnershipPolicy.NONE:
                raise ValueError("public endpoints cannot require a session actor or owner")
        elif not self.allowed_actor_types:
            raise ValueError("session endpoints must declare allowed actor types")
        if self.ownership_policy is RestOwnershipPolicy.ANONYMOUS_HANDOFF and (
            self.resource_kind is not RestResourceKind.TRIP
            or self.allowed_actor_types != [OwnerType.USER]
        ):
            raise ValueError("anonymous handoff is a user-only trip migration policy")
        return self


class RestContractCatalog(ContractModel):
    endpoints: list[RestEndpointContract] = Field(min_length=1)

    @model_validator(mode="after")
    def routes_and_operations_are_unique(self) -> RestContractCatalog:
        _unique([item.operation_id for item in self.endpoints], "operation_id")
        routes = [(item.method.value, item.path) for item in self.endpoints]
        if len(set(routes)) != len(routes):
            raise ValueError("method and path pairs must be unique")
        return self


REST_ENDPOINTS: tuple[RestEndpointContract, ...] = tuple(
    RestEndpointContract.model_validate(item)
    for item in (
        {
            "operation_id": "send_sms",
            "method": "POST",
            "path": "/auth/sms",
            "auth": "public",
            "resource_kind": None,
            "owner_check_required": False,
            "idempotency_required": True,
            "request_model": "SmsSendRequest",
        },
        {
            "operation_id": "verify_sms",
            "method": "POST",
            "path": "/auth/verify",
            "auth": "public",
            "resource_kind": None,
            "owner_check_required": False,
            "idempotency_required": True,
            "request_model": "SmsVerifyRequest",
            "response_model": "AccountView",
        },
        {
            "operation_id": "logout",
            "method": "POST",
            "path": "/auth/logout",
            "auth": "session",
            "resource_kind": None,
            "owner_check_required": False,
            "allowed_actor_types": ["user"],
            "idempotency_required": True,
        },
        {
            "operation_id": "get_account",
            "method": "GET",
            "path": "/account",
            "auth": "session",
            "resource_kind": "account",
            "owner_check_required": True,
            "allowed_actor_types": ["user"],
            "ownership_policy": "same_owner",
            "idempotency_required": False,
            "response_model": "AccountView",
        },
        {
            "operation_id": "update_account",
            "method": "PATCH",
            "path": "/account",
            "auth": "session",
            "resource_kind": "account",
            "owner_check_required": True,
            "allowed_actor_types": ["user"],
            "ownership_policy": "same_owner",
            "idempotency_required": True,
            "request_model": "UpdateAccountRequest",
            "response_model": "AccountView",
        },
        {
            "operation_id": "delete_account",
            "method": "DELETE",
            "path": "/account",
            "auth": "session",
            "resource_kind": "account",
            "owner_check_required": True,
            "allowed_actor_types": ["user"],
            "ownership_policy": "same_owner",
            "idempotency_required": True,
        },
        {
            "operation_id": "create_anonymous_session",
            "method": "POST",
            "path": "/anonymous-sessions",
            "auth": "public",
            "resource_kind": None,
            "owner_check_required": False,
            "idempotency_required": True,
            "response_model": "AnonymousSessionView",
        },
        {
            "operation_id": "delete_anonymous_session",
            "method": "DELETE",
            "path": "/anonymous-sessions/{session_id}",
            "auth": "session",
            "resource_kind": "anonymous_session",
            "owner_check_required": True,
            "allowed_actor_types": ["anonymous"],
            "ownership_policy": "same_owner",
            "idempotency_required": True,
        },
        {
            "operation_id": "attach_trip",
            "method": "POST",
            "path": "/auth/attach-trip",
            "auth": "session",
            "resource_kind": "trip",
            "owner_check_required": True,
            "allowed_actor_types": ["user"],
            "ownership_policy": "anonymous_handoff",
            "idempotency_required": True,
            "request_model": "AttachAnonymousTripRequest",
            "response_model": "TripSnapshotView",
        },
        {
            "operation_id": "list_trips",
            "method": "GET",
            "path": "/trips",
            "auth": "session",
            "resource_kind": "trip",
            "owner_check_required": True,
            "allowed_actor_types": ["user"],
            "ownership_policy": "same_owner",
            "idempotency_required": False,
            "response_model": "TripListView",
        },
        {
            "operation_id": "create_trip",
            "method": "POST",
            "path": "/trips",
            "auth": "session",
            "resource_kind": "trip",
            "owner_check_required": True,
            "allowed_actor_types": ["anonymous", "user"],
            "ownership_policy": "same_owner",
            "idempotency_required": True,
            "request_model": "CreateTripRequest",
            "response_model": "TripSnapshotView",
        },
        {
            "operation_id": "get_trip",
            "method": "GET",
            "path": "/trips/{trip_id}",
            "auth": "session",
            "resource_kind": "trip",
            "owner_check_required": True,
            "allowed_actor_types": ["anonymous", "user"],
            "ownership_policy": "same_owner",
            "idempotency_required": False,
            "response_model": "TripSnapshotView",
        },
        {
            "operation_id": "delete_trip",
            "method": "DELETE",
            "path": "/trips/{trip_id}",
            "auth": "session",
            "resource_kind": "trip",
            "owner_check_required": True,
            "allowed_actor_types": ["anonymous", "user"],
            "ownership_policy": "same_owner",
            "idempotency_required": True,
        },
        {
            "operation_id": "get_preferences",
            "method": "GET",
            "path": "/preferences",
            "auth": "session",
            "resource_kind": "preference",
            "owner_check_required": True,
            "allowed_actor_types": ["user"],
            "ownership_policy": "same_owner",
            "idempotency_required": False,
            "response_model": "PreferenceListView",
        },
        {
            "operation_id": "patch_preferences",
            "method": "PATCH",
            "path": "/preferences",
            "auth": "session",
            "resource_kind": "preference",
            "owner_check_required": True,
            "allowed_actor_types": ["user"],
            "ownership_policy": "same_owner",
            "idempotency_required": True,
            "request_model": "PatchPreferencesRequest",
            "response_model": "PreferenceListView",
        },
        {
            "operation_id": "save_cold_start_preference",
            "method": "POST",
            "path": "/preferences/cold-start",
            "auth": "session",
            "resource_kind": "preference",
            "owner_check_required": True,
            "allowed_actor_types": ["user"],
            "ownership_policy": "same_owner",
            "idempotency_required": True,
            "request_model": "ColdStartSubmission",
            "response_model": "PreferenceListView",
        },
        {
            "operation_id": "delete_preference",
            "method": "DELETE",
            "path": "/preferences/{preference_id}",
            "auth": "session",
            "resource_kind": "preference",
            "owner_check_required": True,
            "allowed_actor_types": ["user"],
            "ownership_policy": "same_owner",
            "idempotency_required": True,
        },
        {
            "operation_id": "confirm_preference_candidates",
            "method": "POST",
            "path": "/trips/{trip_id}/preference-candidates/confirm",
            "auth": "session",
            "resource_kind": "trip",
            "owner_check_required": True,
            "allowed_actor_types": ["user"],
            "ownership_policy": "same_owner",
            "idempotency_required": True,
            "request_model": "ConfirmPreferenceCandidatesRequest",
            "response_model": "PreferenceListView",
        },
        {
            "operation_id": "create_export",
            "method": "POST",
            "path": "/trips/{trip_id}/versions/{plan_version_id}/exports",
            "auth": "session",
            "resource_kind": "trip",
            "owner_check_required": True,
            "allowed_actor_types": ["anonymous", "user"],
            "ownership_policy": "same_owner",
            "idempotency_required": True,
            "request_model": "CreateExportRequest",
            "response_model": "ExportView",
        },
        {
            "operation_id": "get_export",
            "method": "GET",
            "path": "/exports/{artifact_id}",
            "auth": "session",
            "resource_kind": "export",
            "owner_check_required": True,
            "allowed_actor_types": ["anonymous", "user"],
            "ownership_policy": "same_owner",
            "idempotency_required": False,
            "response_model": "ExportView",
        },
    )
)

P0_REST_CATALOG = RestContractCatalog(endpoints=list(REST_ENDPOINTS))

P0_REST_CONTRACTS: tuple[type[Any], ...] = (
    MutationRequestMeta,
    SmsSendRequest,
    SmsVerifyRequest,
    AccountView,
    UpdateAccountRequest,
    AnonymousSessionView,
    AttachAnonymousTripRequest,
    CreateTripRequest,
    TripListView,
    TripSnapshotView,
    PreferenceListView,
    PatchPreferencesRequest,
    ConfirmPreferenceCandidatesRequest,
    CreateExportRequest,
    ExportView,
    RestContractCatalog,
)
