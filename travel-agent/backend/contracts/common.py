"""Small value objects shared by multiple stage-0 contracts."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field, StringConstraints, model_validator

from backend.contracts.base import ContractModel

NonEmptyText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, pattern=r"\S"),
]
ShortText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=500),
]


class CnyAmountRange(ContractModel):
    """A closed CNY range represented in integer fen."""

    currency: str = Field(default="CNY", pattern=r"^CNY$")
    minimum_fen: int = Field(ge=0, strict=True)
    maximum_fen: int = Field(ge=0, strict=True)

    @model_validator(mode="after")
    def maximum_is_not_below_minimum(self) -> CnyAmountRange:
        if self.maximum_fen < self.minimum_fen:
            raise ValueError("maximum_fen cannot be below minimum_fen")
        return self
