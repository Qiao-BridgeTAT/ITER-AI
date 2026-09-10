"""P0-09 unified trip state shared by every stage-0 surface."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Any, Literal
from uuid import UUID

from pydantic import AwareDatetime, ConfigDict, Field, JsonValue, model_validator

from backend.contracts.base import ContractModel
from backend.contracts.cold_start import ColdStartSubmission
from backend.contracts.common import NonEmptyText, ShortText
from backend.contracts.conversation import ConversationMessage
from backend.contracts.enums import CityCode, HotelSelectionSource, OwnerType, TripPhase
from backend.contracts.events import MapUpdatePayload
from backend.contracts.feedback import (
    AttractionFeedback,
    CityThemeSelection,
    DiningOptionCard,
    DiningPreferenceSelection,
    RestaurantFeedback,
)
from backend.contracts.itinerary import Assumption, CostEstimate, Itinerary, PlanningIssue, TaskBook
from backend.contracts.lodging import Anchor, LodgingAreaStrategy, SelectedHotel
from backend.contracts.places import CanonicalPlace
from backend.contracts.plan_modification import PendingPlanModification
from backend.contracts.plan_publication import PublishedPlan
from backend.contracts.provider_display import ProviderDisplayProjection
from backend.contracts.trip_setup import (
    CityBriefAcknowledgement,
    FixedEvent,
    PersonalProfileDecision,
    ResolvedTripPreferences,
    SpecialConstraint,
)

TRIP_STATE_SCHEMA_RULE: dict[str, Any] = {
    "allOf": [
        {
            "if": {
                "properties": {"phase": {"const": "lodging_selection"}},
                "required": ["phase"],
            },
            "then": {
                "properties": {"dining_preferences": {"not": {"type": "null"}}},
                "required": ["dining_preferences"],
            },
        },
        {
            "if": {
                "properties": {
                    "phase": {
                        "enum": [
                            "task_reflection",
                            "planning",
                            "draft_ready",
                            "revising",
                            "confirmed",
                        ]
                    },
                    "night_count": {"type": "integer", "minimum": 1},
                },
                "required": ["phase", "night_count"],
            },
            "then": {
                "properties": {"selected_hotel": {"not": {"type": "null"}}},
                "required": ["selected_hotel"],
            },
        },
        {
            "if": {
                "properties": {
                    "phase": {"enum": ["planning", "draft_ready", "revising", "confirmed"]}
                },
                "required": ["phase"],
            },
            "then": {
                "properties": {
                    "task_book": {
                        "allOf": [
                            {"not": {"type": "null"}},
                            {
                                "type": "object",
                                "properties": {"status": {"const": "confirmed"}},
                                "required": ["status"],
                            },
                        ]
                    }
                },
                "required": ["task_book"],
            },
        },
    ],
    "x-travel-trip-state": {
        "cityField": "city",
        "cityIdField": "city_id",
        "dateRangeField": "date_range",
        "dayCountField": "day_count",
        "nightCountField": "night_count",
        "phaseField": "phase",
        "diningPreferencesField": "dining_preferences",
        "strategiesField": "lodging_strategies",
        "selectedHotelField": "selected_hotel",
        "taskBookField": "task_book",
        "itineraryField": "itinerary",
        "costField": "cost_estimate",
        "publishedPlanField": "published_plan",
        "pendingPlanModificationField": "pending_plan_modification",
        "cityRequiredPhases": [
            "city_brief",
            "trip_setup",
            "interest_selection",
            "attraction_selection",
            "dining_selection",
            "lodging_selection",
            "task_reflection",
            "planning",
            "draft_ready",
            "revising",
            "confirmed",
        ],
        "profileCompletePhases": [
            "trip_setup",
            "interest_selection",
            "attraction_selection",
            "dining_selection",
            "lodging_selection",
            "task_reflection",
            "planning",
            "draft_ready",
            "revising",
            "confirmed",
        ],
        "dateRequiredPhases": [
            "interest_selection",
            "attraction_selection",
            "dining_selection",
            "lodging_selection",
            "task_reflection",
            "planning",
            "draft_ready",
            "revising",
            "confirmed",
        ],
        "taskAndPlanPhases": [
            "task_reflection",
            "planning",
            "draft_ready",
            "revising",
            "confirmed",
        ],
        "planPhases": ["planning", "draft_ready", "revising", "confirmed"],
        "visiblePlanPhases": ["draft_ready", "revising", "confirmed"],
        "derivedPlanPhases": ["revising", "planning", "draft_ready", "confirmed"],
        "coldStartPhase": "cold_start",
        "tripSetupPhase": "trip_setup",
        "attractionSelectionPhase": "attraction_selection",
        "lodgingSelectionPhase": "lodging_selection",
        "taskReflectionPhase": "task_reflection",
    },
}


class TripOverrides(ContractModel):
    edited_profile_text: ShortText | None = None
    companion_note: ShortText | None = None
    budget_note: ShortText | None = None
    arrival_departure_note: ShortText | None = None
    lodging_share_divisor: int | None = Field(default=None, ge=1, strict=True)
    other_notes: list[ShortText] = Field(default_factory=list)


class MessageRef(ContractModel):
    message_id: UUID
    role: Literal["user", "assistant", "system"]
    created_at: AwareDatetime
    generation_id: UUID | None = None


class AgentCheckpointEnvelope(ContractModel):
    """Versioned, trip-bound wrapper for a validated private Agent checkpoint."""

    checkpoint_version: Literal["v2-agent-1"] = "v2-agent-1"
    trip_id: UUID
    trip_state_version: int = Field(ge=0, strict=True)
    payload: dict[str, JsonValue]


class TripDateRange(ContractModel):
    """Stable natural dates stored in TripState after submission validation."""

    start_date: date
    end_date: date

    @model_validator(mode="after")
    def length_is_between_one_and_five_days(self) -> TripDateRange:
        day_count = (self.end_date - self.start_date).days + 1
        if not 1 <= day_count <= 5:
            raise ValueError("stored trip date range must be between 1 and 5 days")
        return self

    @property
    def day_count(self) -> int:
        return (self.end_date - self.start_date).days + 1

    @property
    def night_count(self) -> int:
        return self.day_count - 1


class TripState(ContractModel):
    model_config = ConfigDict(json_schema_extra=TRIP_STATE_SCHEMA_RULE)

    trip_id: UUID
    owner_type: OwnerType
    owner_id: NonEmptyText
    phase: TripPhase
    state_version: int = Field(ge=0, strict=True)
    schema_version: NonEmptyText
    current_plan_version_id: UUID | None = None
    base_confirmed_version_id: UUID | None = None

    # `city` remains for legacy stage-zero fixtures. V3 uses the registry key
    # so nationwide cities are data entries rather than Python enum members.
    city: CityCode | None = None
    city_id: NonEmptyText | None = None
    date_range: TripDateRange | None = None
    day_count: int | None = Field(default=None, ge=1, le=5, strict=True)
    night_count: int | None = Field(default=None, ge=0, le=4, strict=True)

    personal_defaults: ColdStartSubmission | None = None
    cold_start_completed_at: AwareDatetime | None = None
    city_brief: CityBriefAcknowledgement | None = None
    profile_decision: PersonalProfileDecision | None = None
    resolved_preferences: ResolvedTripPreferences | None = None
    trip_overrides: TripOverrides = Field(default_factory=TripOverrides)
    fixed_events: list[FixedEvent] = Field(default_factory=list)
    special_constraints: list[SpecialConstraint] = Field(default_factory=list)

    city_theme_selection: CityThemeSelection | None = None
    attraction_feedback: list[AttractionFeedback] = Field(default_factory=list)
    dining_option_card: DiningOptionCard | None = None
    dining_preferences: DiningPreferenceSelection | None = None
    restaurant_feedback: list[RestaurantFeedback] = Field(default_factory=list)
    hotel_favorites: list[UUID] = Field(
        default_factory=list,
        json_schema_extra={"uniqueItems": True},
    )

    anchors: list[Anchor] = Field(default_factory=list)
    candidate_places: list[CanonicalPlace] = Field(default_factory=list)
    lodging_strategies: list[LodgingAreaStrategy] = Field(default_factory=list)
    selected_hotel: SelectedHotel | None = None
    task_book: TaskBook | None = None
    itinerary: Itinerary | None = None
    cost_estimate: CostEstimate | None = None
    published_plan: PublishedPlan | None = None
    pending_plan_modification: PendingPlanModification | None = None
    issues: list[PlanningIssue] = Field(default_factory=list)
    assumptions: list[Assumption] = Field(default_factory=list)
    provider_display: ProviderDisplayProjection = Field(default_factory=ProviderDisplayProjection)
    map_view: MapUpdatePayload | None = None

    messages: list[MessageRef] = Field(default_factory=list)
    conversation_messages: list[ConversationMessage] = Field(default_factory=list)
    followup_question_count: int = Field(default=0, ge=0, le=3, strict=True)
    active_generation_id: UUID | None = None
    agent_checkpoint: AgentCheckpointEnvelope | None = None

    @model_validator(mode="after")
    def phase_and_derived_state_are_consistent(self) -> TripState:
        self._validate_unique_references()
        self._validate_conversation_messages()
        if self.agent_checkpoint is not None and (
            self.agent_checkpoint.trip_id != self.trip_id
            or self.agent_checkpoint.trip_state_version != self.state_version
        ):
            raise ValueError("agent checkpoint must belong to the current stable trip version")
        if self.cold_start_completed_at is not None and self.personal_defaults is None:
            raise ValueError("cold_start_completed_at requires personal_defaults")

        city_required_phases = set(TripPhase) - {
            TripPhase.COLD_START,
            TripPhase.CITY_SELECTION,
        }
        if self.phase in city_required_phases and self.city is None and self.city_id is None:
            raise ValueError("the current phase requires a selected city")
        if self.phase is TripPhase.COLD_START and (
            self.city is not None or self.city_id is not None
        ):
            raise ValueError("cold_start cannot already contain a city")

        profile_complete_phases = {
            TripPhase.TRIP_SETUP,
            TripPhase.INTEREST_SELECTION,
            TripPhase.ATTRACTION_SELECTION,
            TripPhase.DINING_SELECTION,
            TripPhase.LODGING_SELECTION,
            TripPhase.TASK_REFLECTION,
            TripPhase.PLANNING,
            TripPhase.DRAFT_READY,
            TripPhase.REVISING,
            TripPhase.CONFIRMED,
        }
        if self.phase in profile_complete_phases and (
            self.city_brief is None
            or self.profile_decision is None
            or self.resolved_preferences is None
        ):
            raise ValueError("city brief, profile decision, and resolved preferences required")

        date_complete_phases = profile_complete_phases - {TripPhase.TRIP_SETUP}
        if self.phase in date_complete_phases and self.date_range is None:
            raise ValueError("a validated date range is required after trip setup")

        if self.profile_decision is not None and (
            self.profile_decision.mode.value == "use_defaults" and self.personal_defaults is None
        ):
            raise ValueError("use_defaults requires saved personal defaults")
        if self.date_range is None:
            if self.day_count is not None or self.night_count is not None:
                raise ValueError("day_count and night_count require a date range")
        elif (
            self.day_count != self.date_range.day_count
            or self.night_count != self.date_range.night_count
        ):
            raise ValueError("day_count and night_count must be derived from date_range")

        if self.phase is TripPhase.ATTRACTION_SELECTION and self.city_theme_selection is None:
            raise ValueError("attraction selection requires a completed city-theme decision")
        if self.phase is TripPhase.LODGING_SELECTION and self.dining_preferences is None:
            raise ValueError("lodging selection requires a completed dining preference decision")
        if self.phase is TripPhase.TASK_REFLECTION and self.task_book is None:
            raise ValueError("task_reflection requires a generated task book")

        if self.date_range is not None:
            if self.night_count == 0 and self.selected_hotel is not None:
                raise ValueError("a one-day trip cannot contain a selected hotel")
            task_and_plan_phases = {
                TripPhase.TASK_REFLECTION,
                TripPhase.PLANNING,
                TripPhase.DRAFT_READY,
                TripPhase.REVISING,
                TripPhase.CONFIRMED,
            }
            if (
                self.phase in task_and_plan_phases
                and self.date_range.night_count > 0
                and self.selected_hotel is None
            ):
                raise ValueError("an overnight trip requires a final hotel before task reflection")

        if (self.itinerary is None) != (self.cost_estimate is None):
            raise ValueError("legacy itinerary and cost estimate must appear together")
        if self.published_plan is not None and (
            self.itinerary is not None or self.cost_estimate is not None
        ):
            raise ValueError("published_plan is the sole formal plan source")

        self._validate_cross_object_consistency()
        self._validate_map_view()

        plan_phases = {
            TripPhase.PLANNING,
            TripPhase.DRAFT_READY,
            TripPhase.REVISING,
            TripPhase.CONFIRMED,
        }
        if self.phase in plan_phases and (
            self.task_book is None or self.task_book.status.value != "confirmed"
        ):
            raise ValueError("a confirmed task book is required before planning")

        visible_plan_phases = {
            TripPhase.DRAFT_READY,
            TripPhase.REVISING,
            TripPhase.CONFIRMED,
        }
        if self.phase in visible_plan_phases and (
            self.current_plan_version_id is None
            or (
                self.published_plan is None
                and (self.itinerary is None or self.cost_estimate is None)
            )
        ):
            raise ValueError("visible plan phases require a complete versioned draft")
        if self.published_plan is not None:
            if self.phase not in plan_phases:
                raise ValueError("published_plan is only valid during plan work")
            if (
                self.published_plan.trip_id != self.trip_id
                or self.published_plan.plan_version_id != self.current_plan_version_id
                or self.published_plan.map_projection != self.map_view
            ):
                raise ValueError("published plan must match the current stable trip version")
            if self.city_id is not None and self.published_plan.schedule.city_id != self.city_id:
                raise ValueError("published plan city must match the current trip city")
            if self.date_range is not None and (
                self.published_plan.schedule.start_date != self.date_range.start_date
                or self.published_plan.schedule.end_date != self.date_range.end_date
            ):
                raise ValueError("published plan dates must match the current trip dates")
            if self.phase is TripPhase.DRAFT_READY and (
                self.published_plan.input_state_version != self.state_version - 1
                or self.published_plan.base_confirmed_version_id != self.base_confirmed_version_id
            ):
                raise ValueError("newly published plan must bind its exact base state")
        if self.pending_plan_modification is not None:
            pending = self.pending_plan_modification
            if self.phase not in {TripPhase.REVISING, TripPhase.PLANNING}:
                raise ValueError("a pending plan modification requires revising or planning")
            if self.published_plan is None:
                raise ValueError("a pending plan modification requires a visible base plan")
            if (
                pending.trip_id != self.trip_id
                or pending.base_plan_version_id != self.current_plan_version_id
                or pending.base_state_version >= self.state_version
            ):
                raise ValueError("pending plan modification must bind the current stable plan")
        if self.base_confirmed_version_id is not None and self.phase not in {
            TripPhase.REVISING,
            TripPhase.PLANNING,
            TripPhase.DRAFT_READY,
            TripPhase.CONFIRMED,
        }:
            raise ValueError("base_confirmed_version_id is only valid for derived plan work")
        return self

    def _validate_map_view(self) -> None:
        if self.map_view is None:
            return
        known_place_ids = {place.place_id for place in self.candidate_places} | {
            place.place_id for place in self.provider_display.places
        }
        if self.published_plan is not None:
            known_place_ids.update(place.place_id for place in self.published_plan.places)
            selected_hotel_id = (
                self.published_plan.hotel_selection.selected_hotel_place_id
                if self.published_plan.hotel_selection is not None
                else None
            )
            if selected_hotel_id is not None:
                known_place_ids.add(selected_hotel_id)
        marker_ids = [marker.place_id for marker in self.map_view.markers]
        if len(set(marker_ids)) != len(marker_ids):
            raise ValueError("map markers must not contain duplicate place_id values")
        if not set(marker_ids) <= known_place_ids:
            raise ValueError("map markers must reference canonical trip places")
        if any(
            route.from_place_id not in known_place_ids or route.to_place_id not in known_place_ids
            for route in self.map_view.routes
        ):
            raise ValueError("map routes must reference canonical trip places")

    def _validate_cross_object_consistency(self) -> None:
        if self.task_book is not None:
            if (self.city is None and self.city_id is None) or self.date_range is None:
                raise ValueError("task book requires the state's city and date range")
            task_book_city_matches = (
                self.task_book.city is not None
                and self.task_book.city is self.city
                and self.task_book.city_id is None
            ) or (
                self.task_book.city_id is not None
                and self.task_book.city_id == self.city_id
                and self.task_book.city is None
            )
            if (
                not task_book_city_matches
                or self.task_book.start_date != self.date_range.start_date
                or self.task_book.end_date != self.date_range.end_date
            ):
                raise ValueError("task book city and dates must match the trip state")
            selected_hotel_id = self.selected_hotel.place_id if self.selected_hotel else None
            if self.task_book.selected_hotel_id != selected_hotel_id:
                raise ValueError("task book selected hotel must match the trip state")

        if self.itinerary is not None:
            if self.city is None or self.date_range is None:
                raise ValueError("itinerary requires the state's city and date range")
            if (
                self.itinerary.city is not self.city
                or self.itinerary.start_date != self.date_range.start_date
                or self.itinerary.end_date != self.date_range.end_date
            ):
                raise ValueError("itinerary city and dates must match the trip state")

        if self.selected_hotel is not None:
            if self.date_range is None:
                raise ValueError("selected hotel requires the state's date range")
            if (
                self.selected_hotel.check_in != self.date_range.start_date
                or self.selected_hotel.check_out != self.date_range.end_date
                or self.selected_hotel.night_count != self.date_range.night_count
            ):
                raise ValueError("selected hotel must cover the trip state's lodging nights")

            strategies_by_id = {
                strategy.strategy_id: strategy for strategy in self.lodging_strategies
            }
            if self.selected_hotel.strategy_id is not None:
                strategy = strategies_by_id.get(self.selected_hotel.strategy_id)
                if strategy is None:
                    raise ValueError("selected hotel references an unknown lodging strategy")
                if self.selected_hotel.place_id not in {
                    hotel.place_id for hotel in strategy.representative_hotels
                }:
                    raise ValueError("selected hotel must belong to its referenced strategy")
            elif self.selected_hotel.source is not HotelSelectionSource.PREBOOKED:
                raise ValueError("candidate-selected hotel requires a lodging strategy_id")

        anchor_ids = {anchor.anchor_id for anchor in self.anchors}
        for strategy in self.lodging_strategies:
            if not set(strategy.anchor_ids) <= anchor_ids:
                raise ValueError("lodging strategies may only reference declared state anchors")

        if self.itinerary is not None and self.cost_estimate is not None:
            daily_minimum = sum(
                day.daily_cost_per_person.minimum_fen for day in self.itinerary.days
            )
            daily_maximum = sum(
                day.daily_cost_per_person.maximum_fen for day in self.itinerary.days
            )
            if (
                daily_minimum != self.cost_estimate.total_per_person.minimum_fen
                or daily_maximum != self.cost_estimate.total_per_person.maximum_fen
            ):
                raise ValueError("state daily cost subtotals must equal the full-trip total")

    def _validate_unique_references(self) -> None:
        collections: dict[str, Sequence[str | UUID]] = {
            "attraction place_id": [item.place_id for item in self.attraction_feedback],
            "restaurant place_id": [item.place_id for item in self.restaurant_feedback],
            "hotel_favorites": self.hotel_favorites,
            "anchor_id": [item.anchor_id for item in self.anchors],
            "candidate place_id": [item.place_id for item in self.candidate_places],
            "lodging strategy_id": [item.strategy_id for item in self.lodging_strategies],
            "issue_id": [item.issue_id for item in self.issues],
            "assumption_id": [item.assumption_id for item in self.assumptions],
            "message_id": [item.message_id for item in self.messages],
        }
        for name, values in collections.items():
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must not contain duplicates")

    def _validate_conversation_messages(self) -> None:
        message_ids = [message.message_id for message in self.conversation_messages]
        if len(set(message_ids)) != len(message_ids):
            raise ValueError("conversation message_id must not contain duplicates")
        attachment_ids = [
            attachment.root.attachment_id
            for message in self.conversation_messages
            for attachment in message.attachments
        ]
        if len(set(attachment_ids)) != len(attachment_ids):
            raise ValueError("conversation attachment_id must be unique across the trip")
        if any(
            message.state_version > self.state_version for message in self.conversation_messages
        ):
            raise ValueError("conversation messages cannot be newer than the trip state")


P0_STATE_CONTRACTS: tuple[type[ContractModel], ...] = (TripState,)
