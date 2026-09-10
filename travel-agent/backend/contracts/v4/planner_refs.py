"""Stable scope and entity references used by the V4 Planner."""

from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, Field, RootModel

from backend.contracts.v4.base import Identifier, V4ContractModel
from backend.contracts.v4.enums import CandidateEntityKind


class PlannerScope(V4ContractModel):
    """Optimistic-concurrency scope shared by one Planner decision round."""

    protocol_version: Literal["v4"] = "v4"
    trip_id: Identifier
    generation_id: Identifier
    task_book_id: Identifier
    task_book_version: int = Field(ge=1, strict=True)
    task_book_state_version: int = Field(ge=0, strict=True)
    workspace_revision: int = Field(ge=0, strict=True)


class CandidateRef(V4ContractModel):
    """Revision-bound reference to one candidate in the current pool."""

    kind: Literal["candidate"] = "candidate"
    candidate_pool_id: Identifier
    candidate_pool_revision: int = Field(ge=1, strict=True)
    candidate_id: Identifier
    entity_kind: CandidateEntityKind
    canonical_entity_id: Identifier


class FixedCommitmentRef(V4ContractModel):
    """Reference to a user-owned immutable commitment in the task book."""

    kind: Literal["fixed_commitment"] = "fixed_commitment"
    task_book_id: Identifier
    task_book_version: int = Field(ge=1, strict=True)
    commitment_id: Identifier
    commitment_kind: Literal[
        "reservation",
        "arrival",
        "departure",
        "named_hotel",
        "existing_booking",
    ]


class HotelOfferRef(V4ContractModel):
    """Inventory-snapshot-bound reference to a normalized hotel offer."""

    kind: Literal["hotel_offer"] = "hotel_offer"
    hotel_observation_id: Identifier
    offer_id: Identifier
    property_id: Identifier
    inventory_snapshot_id: Identifier


PlannerObjectRef: TypeAlias = Annotated[
    CandidateRef | FixedCommitmentRef,
    Field(discriminator="kind"),
]
LodgingObjectRef: TypeAlias = Annotated[
    FixedCommitmentRef | HotelOfferRef,
    Field(discriminator="kind"),
]


PlannerRefValue: TypeAlias = Annotated[
    CandidateRef | FixedCommitmentRef | HotelOfferRef,
    Field(discriminator="kind"),
]


class PlannerRefs(RootModel[PlannerRefValue]):
    """Public aggregate schema for the three stable Planner reference kinds."""


def candidate_ref_key(reference: CandidateRef) -> tuple[str, int, str]:
    """Return the authoritative pool identity of a candidate reference."""

    return (
        reference.candidate_pool_id,
        reference.candidate_pool_revision,
        reference.candidate_id,
    )


def fixed_commitment_ref_key(reference: FixedCommitmentRef) -> tuple[str, int, str]:
    """Return the authoritative task-book identity of a fixed commitment."""

    return reference.task_book_id, reference.task_book_version, reference.commitment_id


def planner_object_ref_key(reference: PlannerObjectRef) -> tuple[str, ...]:
    """Return a stable comparison key without relying on labels or array positions."""

    if isinstance(reference, CandidateRef):
        pool_id, revision, candidate_id = candidate_ref_key(reference)
        return "candidate", pool_id, str(revision), candidate_id
    task_book_id, version, commitment_id = fixed_commitment_ref_key(reference)
    return "fixed_commitment", task_book_id, str(version), commitment_id


def scope_ownership_key(scope: PlannerScope) -> tuple[str, str, str, int, int]:
    """Return the immutable ownership portion of a Planner scope."""

    return (
        scope.trip_id,
        scope.generation_id,
        scope.task_book_id,
        scope.task_book_version,
        scope.task_book_state_version,
    )


def require_same_scope_ownership(left: PlannerScope, right: PlannerScope) -> None:
    """Reject cross-trip, cross-generation, or cross-task-book references."""

    if scope_ownership_key(left) != scope_ownership_key(right):
        raise ValueError("Planner artifacts must share trip, generation and task-book ownership")


V4_PLANNER_REF_CONTRACTS: tuple[type[BaseModel], ...] = (
    PlannerScope,
    PlannerRefs,
    CandidateRef,
    FixedCommitmentRef,
    HotelOfferRef,
)
