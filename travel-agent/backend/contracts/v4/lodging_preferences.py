"""Lodging search leads and fixed quality selections, separate from hotel facts."""

from typing import Literal

from pydantic import Field, field_validator

from backend.contracts.v4.base import V4ContractModel

HotelQualityTier = Literal["economy", "comfort", "upscale", "luxury"]
HOTEL_STARS: dict[str, Literal[1, 2, 3, 4, 5]] = {
    "economy": 2,
    "comfort": 3,
    "upscale": 4,
    "luxury": 5,
}


class LodgingExample(V4ContractModel):
    name: str = Field(min_length=2, max_length=80)
    search_keyword: str = Field(min_length=2, max_length=120)

    @field_validator("name")
    @classmethod
    def display_copy(cls, value: str) -> str:
        return validate_lodging_copy(value)


def validate_lodging_copy(value: str) -> str:
    if any(token in value for token in ("(", ")", "（", "）", "用户")):
        raise ValueError("lodging display copy cannot contain parentheses or third person user")
    if not value.strip():
        raise ValueError("lodging display copy must be visible")
    return value


class LodgingAreaCopy(V4ContractModel):
    examples: list[LodgingExample] = Field(min_length=1, max_length=3)
    advantage: str = Field(min_length=2, max_length=50)
    tradeoff: str = Field(min_length=2, max_length=50)

    @field_validator("advantage", "tradeoff")
    @classmethod
    def display_copy(cls, value: str) -> str:
        return validate_lodging_copy(value)


class LodgingTransitCopy(LodgingAreaCopy):
    examples: list[LodgingExample] = Field(min_length=1, max_length=2)


class LodgingAttractionCopy(LodgingAreaCopy):
    examples: list[LodgingExample] = Field(min_length=2, max_length=3)


class LodgingAreaPlan(V4ContractModel):
    transit: LodgingTransitCopy
    attraction: LodgingAttractionCopy
    commercial: LodgingTransitCopy
