"""Single-source soft targets for the city-led attraction discovery policy."""

from dataclasses import dataclass

ATTRACTION_STRATEGY_VERSION = "attraction-v2"


@dataclass(frozen=True)
class AttractionTarget:
    minimum_target: int
    maximum: int
    city_target: int
    personalized_target: int


ATTRACTION_TARGETS = {
    1: AttractionTarget(5, 5, 3, 2),
    2: AttractionTarget(7, 7, 4, 3),
    3: AttractionTarget(9, 9, 6, 3),
    4: AttractionTarget(10, 12, 8, 4),
    5: AttractionTarget(10, 12, 8, 4),
}
