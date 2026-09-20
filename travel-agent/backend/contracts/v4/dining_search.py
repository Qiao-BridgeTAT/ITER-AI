"""Server-owned dining search hints, carried by signed preference options."""

from typing import Literal

from pydantic import Field

from backend.contracts.v4.base import DisplayText, V4ContractModel


class DiningSearchHints(V4ContractModel):
    kind: Literal["local_specialty", "regular"]
    representative_restaurants: list[DisplayText] = Field(default_factory=list, max_length=1)
    search_keywords: list[DisplayText] = Field(default_factory=list, max_length=1)
