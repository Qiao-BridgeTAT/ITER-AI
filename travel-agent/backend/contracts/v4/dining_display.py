"""Optional, source-backed restaurant facts for discovery and read-only plan views."""

from pydantic import AwareDatetime, Field

from backend.contracts.common import CnyAmountRange
from backend.contracts.v4.base import DisplayText, Identifier, V4ContractModel

DINING_CITY_TARGETS = {1: 2, 2: 2, 3: 4, 4: 5, 5: 6}


class DiningDisplayFacts(V4ContractModel):
    cuisine: DisplayText | None = None
    rating: float | None = Field(default=None, gt=0, le=5, allow_inf_nan=False)
    average_cost: CnyAmountRange | None = None
    source_ref: Identifier
    source_name: DisplayText
    observed_at: AwareDatetime
