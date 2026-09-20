"""Immutable formal-plan publication produced by the V4 Planner safety kernel."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Collection
from typing import Any, Literal
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import AwareDatetime, Field, model_validator

from backend.contracts.events import MapUpdatePayload
from backend.contracts.itinerary_draft import CostValidationDraft, ScheduleValidationDraft
from backend.contracts.itinerary_validation import ItineraryValidationResult, ValidationStatus
from backend.contracts.v4.base import Digest, Identifier, V4ContractModel
from backend.contracts.v4.plan_change import (
    FormalHotelRecommendationSet,
    PlanChangeRequest,
    SelectedHotelRecommendation,
    validate_hotel_recommendations_against_observation,
    validate_selected_hotel_against_observation,
)
from backend.contracts.v4.planner_draft import WorkingItineraryDraft
from backend.contracts.v4.planner_evidence import (
    PlannerHotelLocationEvidence,
    PlannerHoursEvidence,
    PlannerPlaceEvidence,
    PlannerTicketEvidence,
    PlannerWeatherEvidence,
)
from backend.contracts.v4.planner_observations import (
    HotelObservation,
    PlannerValidationObservation,
    SpatialRouteEdge,
)
from backend.contracts.v4.planner_refs import FixedCommitmentRef


def scheduled_map_place_ids(
    draft: WorkingItineraryDraft,
    schedule: ScheduleValidationDraft,
    *,
    verified_lodging_place_ids: Collection[UUID] = (),
) -> set[UUID]:
    """Return real scheduled places, excluding structural timeline-only IDs."""

    place_ids = {
        place_id
        for day in schedule.days
        for place_id in (
            day.start_place_id,
            *(activity.place_id for activity in day.activities),
            day.end_place_id,
        )
    }
    empty_draft_dates = {day.service_date for day in draft.days if not day.ordered_items}
    if (
        draft.lodging_baseline.selected_offer_ref is not None
        or draft.lodging_baseline.fixed_commitment_ref is not None
    ):
        empty_draft_dates.clear()
    for day in schedule.days:
        if (
            day.service_date in empty_draft_dates
            and not day.activities
            and not day.transport_legs
            and day.start_place_id == day.end_place_id
        ):
            place_ids.discard(day.start_place_id)

    structural_commitment_ids = {
        item.object_ref.commitment_id
        for day in draft.days
        for item in day.ordered_items
        if isinstance(item.object_ref, FixedCommitmentRef)
        and item.object_ref.commitment_kind
        in {"reservation", "arrival", "departure", "existing_booking"}
    }
    place_ids.difference_update(
        activity.place_id
        for day in schedule.days
        for activity in day.activities
        if len(activity.source_reference_ids) == 1
        and activity.source_reference_ids[0] in structural_commitment_ids
    )

    lodging_place_id: UUID | None = None
    if draft.lodging_baseline.selected_offer_ref is not None:
        lodging_place_id = _planner_place_id(
            "hotel-property",
            draft.lodging_baseline.selected_offer_ref.property_id,
        )
    elif draft.lodging_baseline.fixed_commitment_ref is not None:
        lodging_place_id = _planner_place_id(
            "fixed-hotel",
            draft.lodging_baseline.fixed_commitment_ref.commitment_id,
        )
    activity_place_ids = {activity.place_id for day in schedule.days for activity in day.activities}
    if (
        lodging_place_id is not None
        and lodging_place_id not in verified_lodging_place_ids
        and lodging_place_id not in activity_place_ids
    ):
        # A lodging boundary is a deterministic scheduling anchor, not proof of
        # a real-world location. Omit only that exact anchor when no normalized
        # hotel-location fact can support a marker. Candidate places remain
        # strict even if they happen to lack map evidence.
        place_ids.discard(lodging_place_id)
    return place_ids


def verified_lodging_map_place_ids(
    draft: WorkingItineraryDraft,
    hotel_location_evidence: Collection[PlannerHotelLocationEvidence],
) -> set[UUID]:
    """Map only the active lodging boundary backed by normalized location evidence."""

    properties = {item.property_id for item in hotel_location_evidence}
    selected = draft.lodging_baseline.selected_offer_ref
    if selected is not None and selected.property_id in properties:
        return {_planner_place_id("hotel-property", selected.property_id)}
    fixed = draft.lodging_baseline.fixed_commitment_ref
    if fixed is not None and fixed.commitment_id in properties:
        return {_planner_place_id("fixed-hotel", fixed.commitment_id)}
    return set()


def _planner_place_id(prefix: str, value: str) -> UUID:
    return uuid5(NAMESPACE_URL, f"iter:v4-planner:{prefix}:{value}")


class PlannerPublishedPlan(V4ContractModel):
    """The sole V4 formal itinerary attached to one immutable plan version."""

    plan_version_id: UUID
    publication_key: Identifier
    generation_id: UUID
    trip_id: UUID
    based_on_state_version: int
    based_on_task_book_id: Identifier
    based_on_task_book_version: int
    travel_style_summary: str | None = Field(
        default=None, max_length=180, exclude_if=lambda v: v is None
    )
    best_effort_reasons: tuple[str, ...] = Field(default=(), exclude_if=lambda v: not v)
    verification_status: Literal["verified", "with_issues", "not_reviewed"] = Field(
        default="verified", exclude_if=lambda v: v == "verified"
    )
    planning_notes: tuple[str, ...] = Field(default=(), exclude_if=lambda v: not v)
    schedule_quality_status: Literal["complete", "partial"] | None = Field(
        default=None, exclude_if=lambda v: v is None
    )
    working_itinerary: WorkingItineraryDraft
    materialized_schedule: ScheduleValidationDraft
    cost_draft: CostValidationDraft
    validation_report: ItineraryValidationResult
    validation_observation: PlannerValidationObservation
    map_projection: MapUpdatePayload
    hotel_observation: HotelObservation | None = None
    selected_hotel: SelectedHotelRecommendation | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    hotel_recommendations: FormalHotelRecommendationSet | None = None
    hotel_location_evidence: tuple[PlannerHotelLocationEvidence, ...] = Field(
        default=(),
        exclude_if=lambda value: not value,
    )
    place_evidence: tuple[PlannerPlaceEvidence, ...] = ()
    hours_evidence: tuple[PlannerHoursEvidence, ...] = ()
    weather_evidence: tuple[PlannerWeatherEvidence, ...] = ()
    ticket_evidence: tuple[PlannerTicketEvidence, ...] = ()
    route_evidence: tuple[SpatialRouteEdge, ...] = ()
    change_request: PlanChangeRequest | None = None
    published_at: AwareDatetime
    content_digest: Digest

    @model_validator(mode="after")
    def publication_is_current_complete_and_digest_bound(self) -> PlannerPublishedPlan:
        draft = self.working_itinerary
        schedule = self.materialized_schedule
        cost = self.cost_draft
        observation = self.validation_observation
        if self.schedule_quality_status == "complete" and any(
            any(
                ref.startswith("internal_schedule_quality:")
                for ref in issue.violated_constraint_refs
            )
            for issue in observation.issues
        ):
            raise ValueError("schedule quality cannot be complete with unresolved quality issues")
        if self.change_request is not None and (
            self.change_request.trip_id != str(self.trip_id)
            or self.change_request.base_plan_id == str(self.plan_version_id)
            or self.change_request.requested_scope not in {"local_replan", "full_replan"}
        ):
            raise ValueError("published plan change request has invalid version ownership")
        if (
            str(schedule.trip_id) != str(self.trip_id)
            or str(cost.trip_id) != str(self.trip_id)
            or str(self.validation_report.trip_id) != str(self.trip_id)
        ):
            raise ValueError("published Planner artifacts must belong to one trip")
        if (
            str(schedule.task_book_id) != self.based_on_task_book_id
            or str(cost.task_book_id) != self.based_on_task_book_id
            or str(self.validation_report.task_book_id) != self.based_on_task_book_id
            or schedule.task_book_revision != self.based_on_task_book_version
            or cost.task_book_revision != self.based_on_task_book_version
            or self.validation_report.task_book_revision != self.based_on_task_book_version
        ):
            raise ValueError("published Planner artifacts must use one task book")
        if (
            schedule.input_state_version != self.based_on_state_version
            or cost.input_state_version != self.based_on_state_version
            or self.validation_report.input_state_version != self.based_on_state_version
        ):
            raise ValueError("published Planner artifacts must use one input state version")
        if cost.schedule_request_id != schedule.request_id:
            raise ValueError("published cost must reference the materialized schedule")
        if self.verification_status == "verified" and (
            observation.result != "passed"
            or self.validation_report.status is ValidationStatus.BLOCKED
        ):
            raise ValueError("verified Planner publication must pass current validation")
        if self.verification_status != "verified" and not self.planning_notes:
            raise ValueError("unchecked Planner publication must disclose its remaining issues")
        if (
            observation.draft_id != draft.draft_id
            or observation.draft_revision != draft.draft_revision
            or observation.materialized_schedule_id != str(schedule.request_id)
            or observation.materialized_schedule_revision != draft.draft_revision
            or observation.cost_draft_id != str(cost.request_id)
            or observation.cost_draft_revision != draft.draft_revision
        ):
            raise ValueError("published Planner validation must reference current artifacts")
        selected_mode = draft.lodging_baseline.mode == "selected_offer"
        selection_count = sum(
            value is not None for value in (self.selected_hotel, self.hotel_recommendations)
        )
        if selected_mode != (selection_count == 1):
            raise ValueError("published selected lodging requires one formal hotel selection")
        if selection_count:
            if self.hotel_observation is None:
                raise ValueError("published hotel selection requires its observation")
            if self.selected_hotel is not None:
                validate_selected_hotel_against_observation(
                    self.selected_hotel,
                    self.hotel_observation,
                )
                selected_ref = self.selected_hotel.hotel_offer_ref
                if tuple(offer.offer_ref for offer in self.hotel_observation.offers) != (
                    selected_ref,
                ):
                    raise ValueError(
                        "published selected hotel observation must contain only its selected offer"
                    )
            else:
                assert self.hotel_recommendations is not None
                validate_hotel_recommendations_against_observation(
                    self.hotel_recommendations,
                    self.hotel_observation,
                )
                selected_ref = self.hotel_recommendations.recommended_hotel.hotel_offer_ref
            if selected_ref != draft.lodging_baseline.selected_offer_ref:
                raise ValueError("published hotel must equal the scheduling baseline")
        marker_ids = {marker.place_id for marker in self.map_projection.markers}
        lodging_map_place_ids = verified_lodging_map_place_ids(
            draft,
            self.hotel_location_evidence,
        )
        if not lodging_map_place_ids and self.hotel_recommendations is not None:
            # Pre-single-hotel publications required the selected lodging marker
            # before normalized hotel-location evidence existed. Preserve those
            # already-published 1+2 plans without weakening new selected_hotel
            # publications, which still require current location evidence.
            baseline_selected_ref = draft.lodging_baseline.selected_offer_ref
            if baseline_selected_ref is not None:
                lodging_map_place_ids.add(
                    _planner_place_id("hotel-property", baseline_selected_ref.property_id)
                )
        scheduled_place_ids = scheduled_map_place_ids(
            draft,
            schedule,
            verified_lodging_place_ids=lodging_map_place_ids,
        )
        if marker_ids != scheduled_place_ids:
            raise ValueError("published map must contain every and only scheduled place")
        scheduled_route_keys = {
            (leg.origin_place_id, leg.destination_place_id)
            for day in schedule.days
            for leg in day.transport_legs
        }
        if any(
            (route.from_place_id, route.to_place_id) not in scheduled_route_keys
            for route in self.map_projection.routes
        ):
            raise ValueError("published map routes must belong to the current schedule")
        if self.content_digest != planner_publication_digest(self):
            raise ValueError("published Planner plan digest does not match its content")
        return self


def planner_publication_digest(value: PlannerPublishedPlan | dict[str, Any]) -> str:
    payload = (
        value.model_dump(mode="json") if isinstance(value, PlannerPublishedPlan) else dict(value)
    )
    payload.pop("content_digest", None)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


V4_PLANNER_PUBLICATION_CONTRACTS: tuple[type[V4ContractModel], ...] = (PlannerPublishedPlan,)
