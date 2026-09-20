"""Public tentative itinerary, deliberately separate from a published plan."""

from datetime import date
from typing import Literal

from pydantic import Field

from backend.contracts.v4.base import Digest, DisplayText, Identifier, V4ContractModel


class PlannerDraftStopPreview(V4ContractModel):
    draft_item_id: Identifier
    title: DisplayText
    time_hint: DisplayText


class PlannerDraftDayPreview(V4ContractModel):
    service_date: date
    theme: DisplayText
    transport_summary: DisplayText
    stops: tuple[PlannerDraftStopPreview, ...]


class PlannerDraftPreview(V4ContractModel):
    status: Literal["unverified"] = "unverified"
    draft_id: Identifier
    draft_revision: int = Field(ge=1, strict=True)
    content_digest: Digest
    notice: DisplayText
    days: tuple[PlannerDraftDayPreview, ...] = Field(min_length=1, max_length=5)
    lodging_summary: DisplayText
    unresolved_issues: tuple[DisplayText, ...]
