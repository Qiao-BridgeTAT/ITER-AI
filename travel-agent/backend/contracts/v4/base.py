"""Strict primitives shared by V4 public contracts."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Annotated, TypeVar

from pydantic import ConfigDict, StringConstraints

from backend.contracts.base import ContractModel

Identifier = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=200, pattern=r"\S"),
]
Digest = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=16, max_length=128, pattern=r"\S"),
]
DisplayText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=2_000, pattern=r"\S"),
]
TripGoalText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=2, max_length=2_000, pattern=r"\S"),
]


class V4ContractModel(ContractModel):
    """Immutable strict model used as the V4 Pydantic source of truth."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_assignment=True)


ValueT = TypeVar("ValueT")


def require_unique(values: Iterable[ValueT], field_name: str) -> None:
    materialized = list(values)
    if len(set(materialized)) != len(materialized):
        raise ValueError(f"{field_name} must contain unique values")
