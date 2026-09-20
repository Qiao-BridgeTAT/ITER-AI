"""Advisory sightseeing time, separate from opening hours and route facts."""

from pydantic import Field, model_validator

from backend.contracts.v4.base import V4ContractModel


class VisitDurationRange(V4ContractModel):
    minimum_minutes: int = Field(ge=15, le=600, strict=True)
    maximum_minutes: int = Field(ge=15, le=600, strict=True)

    @model_validator(mode="after")
    def range_is_ordered(self) -> "VisitDurationRange":
        if self.maximum_minutes < self.minimum_minutes:
            raise ValueError("visit duration maximum must not precede minimum")
        return self
