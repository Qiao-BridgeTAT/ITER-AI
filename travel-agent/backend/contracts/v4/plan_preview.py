"""Read-only visual enrichment, deliberately outside immutable plan content."""

from uuid import UUID

from pydantic import Field, HttpUrl, model_validator

from backend.contracts.v4.base import DisplayText, Identifier, V4ContractModel, require_unique
from backend.contracts.v4.dining_display import DiningDisplayFacts
from backend.contracts.v4.planner_evidence import PlannerWeatherEvidence


class PlannerPlacePreview(V4ContractModel):
    place_id: UUID
    image_url: HttpUrl | None = None
    image_source_ref: Identifier | None = None
    description: DisplayText | None = None
    dining_details: DiningDisplayFacts | None = None

    @model_validator(mode="after")
    def image_is_sourced(self) -> "PlannerPlacePreview":
        if (self.image_url is None) != (self.image_source_ref is None):
            raise ValueError("preview image requires a source reference")
        return self


class PlannerPlanPreview(V4ContractModel):
    trip_id: UUID
    plan_version_id: UUID
    places: tuple[PlannerPlacePreview, ...] = Field(default=(), max_length=100)
    weather_evidence: tuple[PlannerWeatherEvidence, ...] = Field(default=(), max_length=5)

    @model_validator(mode="after")
    def places_are_unique(self) -> "PlannerPlanPreview":
        require_unique([str(place.place_id) for place in self.places], "plan preview places")
        require_unique(
            [str(day.service_date) for day in self.weather_evidence], "plan preview weather dates"
        )
        return self
