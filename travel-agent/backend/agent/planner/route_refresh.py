"""Narrow authority for refreshing transport evidence without changing choices."""

import re
from typing import Any

from backend.contracts.itinerary_draft import ScheduleValidationDraft
from backend.contracts.v4.planner_draft import WorkingItineraryDraft, canonical_planning_projection


def requests_route_refresh_only(text: str) -> bool:
    return bool(
        re.search(
            r"(?:仅|只)(?:需|要)?[^。；;\n]{0,16}(?:补查|更新|核实|查询)[^。；;\n]{0,16}路线", text
        )
        and re.search(r"保留|不改变|不调整|不修改", text)
    )


def same_planning_choices(before: WorkingItineraryDraft, after: WorkingItineraryDraft) -> bool:
    def choices(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: choices(item)
                for key, item in value.items()
                if key not in {"candidate_pool_revision", "hotel_observation_id", "scope"}
            }
        if isinstance(value, list):
            return [choices(item) for item in value]
        return value

    return bool(
        choices(canonical_planning_projection(before))
        == choices(canonical_planning_projection(after))
    )


def same_scheduled_choices(before: ScheduleValidationDraft, after: ScheduleValidationDraft) -> bool:
    def choices(schedule: ScheduleValidationDraft) -> list[Any]:
        return [
            (
                day.service_date,
                [(a.place_id, a.kind, a.duration_minutes) for a in day.activities],
                [
                    (leg.origin_place_id, leg.destination_place_id, leg.mode)
                    for leg in day.transport_legs
                ],
            )
            for day in schedule.days
        ]

    return choices(before) == choices(after)
