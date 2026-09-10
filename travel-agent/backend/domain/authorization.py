"""Pure owner checks used by every future REST and WebSocket adapter."""

from __future__ import annotations

from dataclasses import dataclass

from backend.contracts.common import NonEmptyText
from backend.contracts.enums import OwnerType, RestOwnershipPolicy
from backend.contracts.rest import RestEndpointContract


@dataclass(frozen=True)
class RequestActor:
    owner_type: OwnerType
    owner_id: NonEmptyText


@dataclass(frozen=True)
class ResourceOwner:
    owner_type: OwnerType
    owner_id: NonEmptyText


def actor_can_call_endpoint(actor: RequestActor, endpoint: RestEndpointContract) -> bool:
    """Apply the endpoint's explicit anonymous/user principal allowlist."""

    return actor.owner_type in endpoint.allowed_actor_types


def can_access_owned_resource(
    actor: RequestActor,
    resource: ResourceOwner,
    *,
    policy: RestOwnershipPolicy = RestOwnershipPolicy.SAME_OWNER,
    anonymous_handoff_session_id: str | None = None,
) -> bool:
    """Apply same-owner access or an explicit anonymous-to-user handoff proof."""

    if policy is RestOwnershipPolicy.NONE:
        return True
    if policy is RestOwnershipPolicy.SAME_OWNER:
        return actor.owner_type is resource.owner_type and actor.owner_id == resource.owner_id
    return (
        actor.owner_type is OwnerType.USER
        and resource.owner_type is OwnerType.ANONYMOUS
        and anonymous_handoff_session_id is not None
        and resource.owner_id == anonymous_handoff_session_id
    )
