"""Search hints are model proposals, never verified or selectable entities."""

from pydantic import Field

from backend.contracts.v4.base import DisplayText, V4ContractModel


class AttractionSearchPlace(V4ContractModel):
    name: DisplayText
    subcategory: DisplayText


class AttractionSearchHints(V4ContractModel):
    representative_places: list[AttractionSearchPlace] = Field(default_factory=list, max_length=4)
    search_queries: list[DisplayText] = Field(default_factory=list, max_length=2)
