"""Optional, read-only prose; never a place fact or itinerary mutation."""

from typing import Annotated, Literal
from uuid import UUID

from pydantic import Field, StringConstraints, model_validator

from backend.contracts.v4.base import V4ContractModel, require_unique

IntroductionText = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=24)
]


class PlaceIntroduction(V4ContractModel):
    place_id: UUID
    description: IntroductionText


class PlaceIntroductionView(V4ContractModel):
    trip_id: UUID
    scope_kind: Literal["card", "plan"]
    scope_id: UUID
    places: tuple[PlaceIntroduction, ...] = Field(default=(), max_length=40)

    @model_validator(mode="after")
    def unique_places(self) -> "PlaceIntroductionView":
        require_unique([str(item.place_id) for item in self.places], "place introductions")
        return self
