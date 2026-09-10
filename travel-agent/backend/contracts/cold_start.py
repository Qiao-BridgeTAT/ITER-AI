"""Complete six-question cold-start submission contract."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator

from backend.contracts.base import ContractModel
from backend.contracts.enums import DayReturn, DayStart, MobilityTolerance, PriorityGoal

Level = Literal[1, 2, 3, 4, 5]


class ColdStartSubmission(ContractModel):
    day_start: DayStart
    day_return: DayReturn
    pace_level: Level
    classic_niche_level: Level
    walking_tolerance: MobilityTolerance
    bike_tolerance: MobilityTolerance
    transit_taxi_level: Level
    priority_goals: list[PriorityGoal] = Field(
        min_length=1,
        max_length=2,
        json_schema_extra={"uniqueItems": True},
    )

    @field_validator("priority_goals")
    @classmethod
    def priority_goals_must_be_unique(cls, goals: list[PriorityGoal]) -> list[PriorityGoal]:
        if len(set(goals)) != len(goals):
            raise ValueError("priority_goals must not contain duplicates")
        return goals
