"""Runtime-only Prepare decision schemas narrowed to one guarded capability."""

from __future__ import annotations

from datetime import date
from functools import lru_cache
from typing import Annotated, ClassVar, Generic, Literal, TypeVar, cast

from pydantic import (
    Field,
    RootModel,
    StringConstraints,
    ValidationInfo,
    computed_field,
    create_model,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticCustomError

from backend.agent.prepare.semantic_transaction import normalize_compound_basics
from backend.contracts.v4.base import (
    DisplayText,
    Identifier,
    TripGoalText,
    V4ContractModel,
    require_unique,
)
from backend.contracts.v4.content_quality import (
    require_meaningful_label,
    require_meaningful_trip_goal,
)
from backend.contracts.v4.enums import (
    ConfidenceLevel,
    DiscoverySection,
    PrepareActionKind,
    PrepareDomain,
)
from backend.contracts.v4.prepare import (
    AskClarificationAction,
    ClarificationContract,
    FinalSupplementAction,
    GenerateTaskBookAction,
    HotelBookingFactsRequest,
    OpeningHoursRequest,
    PlaceFactsRequest,
    PlaceProductsRequest,
    PrepareDecision,
    PrepareNextAction,
    PublishedPlanIntent,
    ReplyGoal,
    ReplyOnlyAction,
    ResolvePlaceRequest,
    ShowPreferenceCardAction,
    ShowSpecificCardAction,
    SpatialRoutesRequest,
    TicketAvailabilityRequest,
    ToolRequest,
    ToolRequestBase,
    UseToolAction,
    WeatherForecastRequest,
)
from backend.contracts.v4.semantic_operations import (
    AddConditionalRequirementOperation,
    ExcludeConcreteEntityOperation,
    ExcludePreferenceDirectionOperation,
    ResolveConflictOperation,
    RevokePriorIntentOperation,
    SelectConcreteEntityOperation,
    SelectPreferenceDirectionOperation,
    SemanticTargetV4,
    SetDelegationScopeOperation,
    SetExistingBookingOperation,
    SetLodgingClassPreferenceOperation,
    SetNoPreferenceOperation,
    SetNotApplicableOperation,
)

RequestValue = TypeVar("RequestValue", bound=ToolRequestBase)
PlaceQuery = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=2, max_length=80, pattern=r"\S"),
]


class GroundedPlaceReference(V4ContractModel):
    query: PlaceQuery = Field(
        description="逐字复制当前用户话语中的完整地点名称，不含城市名或类别后缀。"
    )


def _with_calendar_duration(value: dict[str, object]) -> dict[str, object]:
    endpoints: list[date] = []
    for field in ("start_date", "end_date"):
        endpoint = value.get(field)
        if type(endpoint) is date:
            endpoints.append(endpoint)
        elif isinstance(endpoint, str):
            try:
                endpoints.append(date.fromisoformat(endpoint))
            except ValueError:
                return value  # Let field validation report the invalid date.
        else:
            return value  # Never infer a missing endpoint here.
    duration = (endpoints[1] - endpoints[0]).days + 1
    return {**value, "duration_days": duration} if 1 <= duration <= 5 else value


class TripDateRangeAssessment(V4ContractModel):
    """A complete date pair understood by Qwen, never a partial persisted date."""

    status: Literal["none", "needs_confirmation", "ready"] = "none"
    basis: Literal[
        "none",
        "current_explicit_range",
        "start_plus_duration",
        "contextual_completion",
        "contextual_confirmation",
    ] = "none"
    start_date: date | None = None
    end_date: date | None = None
    duration_days: int | None = Field(default=None, ge=1, le=5, strict=True)

    @model_validator(mode="before")
    @classmethod
    def derive_duration_from_complete_dates(cls, value: object) -> object:
        """Calendar arithmetic is not another independent model decision."""

        if not isinstance(value, dict) or value.get("status", "none") == "none":
            return value
        return _with_calendar_duration(value)

    @model_validator(mode="after")
    def range_matches_status_and_basis(self) -> TripDateRangeAssessment:
        if self.status == "none":
            if self.basis != "none" or any(
                value is not None for value in (self.start_date, self.end_date, self.duration_days)
            ):
                raise PydanticCustomError(
                    "date_resolution_empty",
                    "empty date assessment cannot contain a range or basis",
                )
            return self

        if self.basis == "none" or any(
            value is None for value in (self.start_date, self.end_date, self.duration_days)
        ):
            raise PydanticCustomError(
                "date_resolution_incomplete",
                "non-empty date assessment requires one complete date range",
            )
        assert self.start_date is not None
        assert self.end_date is not None
        assert self.duration_days is not None
        if self.end_date < self.start_date:
            raise PydanticCustomError(
                "date_resolution_reversed", "assessed end date cannot be before start date"
            )
        expected_duration = (self.end_date - self.start_date).days + 1
        if expected_duration > 5:
            raise PydanticCustomError(
                "date_resolution_too_long", "assessed date range cannot exceed five days"
            )
        if self.duration_days != expected_duration:
            raise ValueError("assessed duration must match the inclusive date range")
        if self.status == "needs_confirmation" and self.basis != "start_plus_duration":
            raise PydanticCustomError(
                "date_resolution_confirmation_basis",
                "only a start-plus-duration range may await confirmation",
            )
        if self.status == "ready" and self.basis == "start_plus_duration":
            raise PydanticCustomError(
                "date_resolution_unconfirmed",
                "a derived date range must be confirmed before it is ready",
            )
        return self


class NamedEntityIntent(V4ContractModel):
    query: PlaceQuery
    domain: Literal["attraction", "dining"]
    disposition: Literal["must", "want", "destination", "if_convenient", "avoid"]


AdditionalIntakeTarget = Literal[
    SemanticTargetV4.ATTRACTION_PREFERENCE,
    SemanticTargetV4.ATTRACTION_ENTITY,
    SemanticTargetV4.DINING_PREFERENCE,
    SemanticTargetV4.DINING_REQUIREMENT,
    SemanticTargetV4.DINING_ENTITY,
    SemanticTargetV4.LODGING_AREA,
    SemanticTargetV4.LODGING_CLASS,
    SemanticTargetV4.LODGING_BOOKING,
    SemanticTargetV4.TRANSPORT_AND_PACE,
    SemanticTargetV4.GENERAL_CONSTRAINT,
]


class IntakeRequirementFact(V4ContractModel):
    """A verbatim, independently meaningful requirement, without model-owned IDs."""

    target: Literal[
        "general_constraint", "transport_and_pace", "lodging_class", "dining_requirement"
    ]
    quote: DisplayText


class IntakePreferenceFact(V4ContractModel):
    """A direction is not a hard requirement; Qwen owns its explicit polarity."""

    target: Literal["attraction_preference", "dining_preference", "lodging_area"]
    quote: DisplayText
    disposition: Literal["select", "exclude"]


class TripBasicsAssessment(V4ContractModel):
    """Focused LLM check used only when a broad decision silently drops basics."""

    explicit_destination: bool
    explicit_date_range: bool
    explicit_duration_days: bool
    explicit_travelers: bool
    explicit_trip_goals: bool
    has_additional_request: bool
    explicit_lodging_not_applicable: bool
    explicitly_no_more_requirements: bool = False
    requests_attraction_cards: bool = False
    fact_capabilities: list[
        Literal[
            "place_facts",
            "opening_hours",
            "ticket_availability",
            "weather_forecast",
            "spatial_routes",
            "hotel_booking_facts",
            "place_products",
        ]
    ] = Field(default_factory=list, max_length=7)
    date_range_resolution: TripDateRangeAssessment = Field(default_factory=TripDateRangeAssessment)
    named_entity_intents: list[NamedEntityIntent] = Field(default_factory=list, max_length=8)
    required_additional_targets: list[AdditionalIntakeTarget] = Field(
        default_factory=list, max_length=12
    )
    requirement_facts: list[
        Annotated[IntakeRequirementFact | IntakePreferenceFact, Field(discriminator="target")]
    ] = Field(default_factory=list, max_length=16)

    @field_validator("required_additional_targets", mode="before")
    @classmethod
    def intake_inventory_excludes_lifecycle_actions(cls, value: object) -> object:
        if not isinstance(value, list):
            return value
        # A request to be guided to confirmation is not confirmation. The
        # existing pending-interaction and TaskBook guards still own those actions.
        excluded = {"trip_basics", "final_supplement", "task_book_confirmation", "conflict"}
        result: list[object] = []
        for item in value:
            if isinstance(item, str) and (item in excluded or item in result):
                continue
            result.append(item)
        return result

    @model_validator(mode="after")
    def lodging_conclusion_is_an_additional_request(self) -> TripBasicsAssessment:
        if self.explicit_lodging_not_applicable and not self.has_additional_request:
            raise ValueError("lodging not applicable must be marked as an additional request")
        if (
            self.explicit_date_range
            and self.date_range_resolution.status != "none"
            and not (
                self.date_range_resolution.status == "ready"
                and self.date_range_resolution.basis
                # A complete current answer can also complete the active date
                # question. Keep its contextual basis: the graph must still
                # require an active follow-up before accepting these dates.
                in {"current_explicit_range", "contextual_completion", "contextual_confirmation"}
            )
        ):
            raise PydanticCustomError(
                "date_resolution_explicit_conflict",
                "explicit date range must include a ready explicit or confirmed resolution",
            )
        if (
            not self.explicit_date_range
            and self.date_range_resolution.basis == "current_explicit_range"
        ):
            raise PydanticCustomError(
                "date_resolution_explicit_required",
                "current-message date resolution requires explicit_date_range",
            )
        return self


class CardActionObservation(V4ContractModel):
    """Safe card-capability result returned to Qwen for one bounded re-decision."""

    observation_id: Identifier
    section: DiscoverySection
    status: Literal["needs_input", "unavailable"]
    failure_code: Identifier
    missing_fields: list[Literal["destination", "duration_days"]] = Field(
        default_factory=list,
        max_length=2,
    )
    allowed_next_actions: list[PrepareActionKind] = Field(min_length=1, max_length=2)
    user_input_targets: list[Identifier] = Field(min_length=1, max_length=4)
    safe_summary: DisplayText

    @model_validator(mode="after")
    def result_has_only_bounded_non_card_choices(self) -> CardActionObservation:
        require_unique(self.missing_fields, "card observation missing fields")
        require_unique(self.allowed_next_actions, "card observation allowed actions")
        require_unique(self.user_input_targets, "card observation user input targets")
        legal_actions = {
            PrepareActionKind.ASK_CLARIFICATION,
            PrepareActionKind.REPLY_ONLY,
        }
        if not set(self.allowed_next_actions) <= legal_actions:
            raise ValueError("card observation may only allow clarification or reply")
        if self.status == "needs_input":
            if not self.missing_fields:
                raise ValueError("needs-input card observation requires missing_fields")
            if self.allowed_next_actions != [PrepareActionKind.ASK_CLARIFICATION]:
                raise ValueError("needs-input card observation must require clarification")
        elif self.missing_fields:
            raise ValueError("unavailable card observation cannot claim missing user fields")
        return self


class TaskBookActionObservation(V4ContractModel):
    """Final-guard preflight result returned for one bounded re-decision."""

    observation_id: Identifier
    action: Literal[PrepareActionKind.GENERATE_TASK_BOOK]
    status: Literal["blocked"]
    failure_code: Literal["final_supplement_incomplete"]
    blocking_reasons: list[Identifier] = Field(min_length=1, max_length=4)
    allowed_next_actions: list[PrepareActionKind] = Field(min_length=1, max_length=1)
    user_input_targets: list[Identifier] = Field(min_length=1, max_length=1)
    safe_summary: DisplayText

    @model_validator(mode="after")
    def recovery_is_limited_to_final_supplement(self) -> TaskBookActionObservation:
        require_unique(self.blocking_reasons, "task-book observation blocking reasons")
        if self.allowed_next_actions != [PrepareActionKind.FINAL_SUPPLEMENT]:
            raise ValueError("task-book observation may only return to final supplement")
        if self.user_input_targets != ["final_supplement"]:
            raise ValueError("task-book observation target must be final supplement")
        return self


class CompoundTripIntakeExtraction(V4ContractModel):
    """Focused semantics for basics + no lodging + one named anchor."""

    travelers: list[DisplayText] = Field(min_length=1, max_length=20)
    trip_goals: list[TripGoalText] = Field(min_length=1, max_length=20)
    lodging_not_applicable_reason: DisplayText
    place_query: PlaceQuery
    transport_requirements: list[DisplayText] = Field(default_factory=list, max_length=5)
    dining_requirements: list[DisplayText] = Field(default_factory=list, max_length=5)
    delegate_dining_by_route: bool

    @field_validator("transport_requirements", "dining_requirements")
    @classmethod
    def list_text_is_complete(cls, values: list[str]) -> list[str]:
        return [require_meaningful_label(value, "compound intake text") for value in values]

    @field_validator("travelers")
    @classmethod
    def travelers_are_complete(cls, values: list[str]) -> list[str]:
        return [
            value if value == "我" else require_meaningful_label(value, "compound intake traveler")
            for value in values
        ]

    @field_validator("lodging_not_applicable_reason", "place_query")
    @classmethod
    def scalar_text_is_complete(cls, value: str) -> str:
        return require_meaningful_label(value, "compound intake text")

    @field_validator("trip_goals")
    @classmethod
    def trip_goals_are_readable(cls, values: list[str]) -> list[str]:
        return [require_meaningful_trip_goal(value) for value in values]


class FinalSupplementAssessment(V4ContractModel):
    """Focused LLM judgment for closing the final discovery section."""

    explicitly_no_more_requirements: bool
    has_additional_request: bool


class PaceModificationAssessment(V4ContractModel):
    """Focused LLM judgment for a task-book pace or mobility revision."""

    explicit_pace_or_mobility_change: bool
    has_unrelated_request: bool


class TaskBookReviewModificationAssessment(V4ContractModel):
    """Node-local reading of a user's requested task-book revision.

    This assessment identifies the minimum semantic targets that the broad
    Prepare decision must not drop.  It does not choose or apply operations.
    """

    is_change_request: bool
    required_targets: list[SemanticTargetV4] = Field(default_factory=list, max_length=8)
    requested_card_section: DiscoverySection | None = None
    requests_regeneration: bool = False
    has_fact_question: bool = False

    @model_validator(mode="after")
    def targets_stay_within_task_book_revision_scope(
        self,
    ) -> TaskBookReviewModificationAssessment:
        require_unique(self.required_targets, "task-book revision targets")
        forbidden_targets = {
            SemanticTargetV4.FINAL_SUPPLEMENT,
            SemanticTargetV4.TASK_BOOK_CONFIRMATION,
            SemanticTargetV4.CONFLICT,
        }
        if set(self.required_targets) & forbidden_targets:
            raise ValueError("task-book revision assessment contains an internal target")
        card_sections = {
            DiscoverySection.ATTRACTION_PREFERENCE,
            DiscoverySection.ATTRACTION_SPECIFIC,
            DiscoverySection.DINING_PREFERENCE,
            DiscoverySection.DINING_SPECIFIC,
            DiscoverySection.LODGING_AREA_PREFERENCE,
            DiscoverySection.LODGING_CLASS_PREFERENCE,
        }
        if (
            self.requested_card_section is not None
            and self.requested_card_section not in card_sections
        ):
            raise ValueError("task-book revision requested an unsupported card section")
        if not self.is_change_request and (
            self.required_targets or self.requested_card_section is not None
        ):
            raise ValueError("non-change assessment cannot require writes or cards")
        return self


class PaceRequirementExtraction(V4ContractModel):
    """Exact user-authored pace semantics after the focused modification check."""

    condition: DisplayText = Field(
        description="用户明确表达的适用对象或范围；整体行程可写本次旅行。"
    )
    required_outcome: DisplayText = Field(
        description="完整保留用户明确提出的节奏、每日密度、休息或行动便利要求。"
    )

    @field_validator("condition", "required_outcome")
    @classmethod
    def text_is_complete(cls, value: str) -> str:
        return require_meaningful_label(value, "pace requirement text")


class ConcreteIntentReviewChoice(V4ContractModel):
    entity_key: Identifier
    disposition: Literal["must", "want", "if_convenient", "avoid", "not_requested", "unclear"]
    quote: DisplayText | None = None


class ConcreteIntentReview(V4ContractModel):
    """Small semantic check before an unverified attraction hard requirement is saved."""

    choices: list[ConcreteIntentReviewChoice] = Field(min_length=1, max_length=24)


class CardTextCoverageAssessment(V4ContractModel):
    """Qwen identifies explicit new semantics in a signed card's text answer."""

    required_targets: list[SemanticTargetV4] = Field(default_factory=list, max_length=8)
    requires_answer_first: bool


class _TripBasicsChoice(V4ContractModel):
    operation_type: Literal["set_trip_basics"] = "set_trip_basics"
    domain: Literal["general"] = "general"
    destination_name: DisplayText | None = None
    start_date: date | None = None
    end_date: date | None = None
    duration_days: int | None = Field(default=None, ge=1, le=5, strict=True)
    travelers: list[DisplayText] | None = Field(default=None, min_length=1, max_length=20)
    trip_goals: list[TripGoalText] | None = Field(default=None, min_length=1, max_length=20)
    confidence: ConfidenceLevel

    @model_validator(mode="before")
    @classmethod
    def compile_validated_dates(cls, value: object, info: ValidationInfo) -> object:
        if not isinstance(value, dict):
            return value
        # Only trusted, independently validated intake can supply dates. This
        # context is local to the gateway and never comes from model JSON.
        assessment = (info.context or {}).get("assessed_trip_date_range")
        if isinstance(assessment, TripDateRangeAssessment) and assessment.status == "ready":
            return {
                **value,
                "start_date": assessment.start_date,
                "end_date": assessment.end_date,
                "duration_days": assessment.duration_days,
            }
        # A complete explicit pair has one inclusive duration, not a second LLM
        # arithmetic choice. Missing/invalid endpoints still fail below.
        return _with_calendar_duration(value)

    @field_validator("trip_goals")
    @classmethod
    def trip_goals_are_readable(cls, values: list[str] | None) -> list[str] | None:
        if values is None:
            return None
        return [require_meaningful_trip_goal(value) for value in values]

    @model_validator(mode="after")
    def contains_a_consistent_basic_update(self) -> _TripBasicsChoice:
        if not any(
            (
                self.destination_name is not None,
                self.start_date is not None,
                self.duration_days is not None,
                self.travelers is not None,
                self.trip_goals is not None,
            )
        ):
            raise PydanticCustomError(
                "trip_basics_empty", "trip basics choice requires at least one explicit field"
            )
        if (self.start_date is None) != (self.end_date is None):
            raise PydanticCustomError(
                "trip_basics_date_pair", "trip basics choice dates must be set together"
            )
        if self.start_date is not None and self.end_date is not None:
            if self.end_date < self.start_date:
                raise PydanticCustomError(
                    "trip_basics_dates_reversed", "trip basics end date cannot be before start date"
                )
            expected_duration = (self.end_date - self.start_date).days + 1
            if expected_duration > 5:
                raise PydanticCustomError(
                    "trip_basics_dates_too_long", "trip basics date range cannot exceed five days"
                )
            if self.duration_days is not None and self.duration_days != expected_duration:
                raise ValueError("duration_days must match the inclusive date range")
        return self


class _TripBasicsOperationChoice(RootModel[_TripBasicsChoice]):
    pass


class _RuntimeSectionProposal(V4ContractModel):
    """Model proposal without the public guard's evidence validator."""

    kind: Literal["stay", "complete_and_advance", "reopen"]
    section: DiscoverySection
    reopen_sections: list[DiscoverySection] = Field(default_factory=list)
    evidence_refs: list[DisplayText] = Field(default_factory=list)


class _RuntimePrepareDecision(PrepareDecision):
    section_proposal: _RuntimeSectionProposal  # type: ignore[assignment]

    @model_validator(mode="before")
    @classmethod
    def compile_fixed_action_scope(cls, value: object) -> object:
        return _fixed_action_scope(value)


def _fixed_action_scope(value: object) -> object:
    if not isinstance(value, dict):
        return value
    action = value.get("next_action")
    if isinstance(action, dict) and action.get("kind") == "final_supplement":
        return {**value, "next_action": {**action, "requested_targets": ["final_supplement"]}}
    return value


class _CapabilityToolRequest(RootModel[RequestValue], Generic[RequestValue]):
    pass


class _CapabilityPrepareDecision(_RuntimePrepareDecision, Generic[RequestValue]):
    # The public contract remains PrepareDecision. This runtime response schema
    # narrows only the model's current tool branch before normalizing back to it.
    tool_requests: list[_CapabilityToolRequest[RequestValue]] = Field(  # type: ignore[assignment]
        default_factory=list,
        max_length=4,
    )
    next_action: UseToolAction


class _CapabilityContinuationDecision(
    _CapabilityPrepareDecision[RequestValue],
    Generic[RequestValue],
):
    semantic_operations: list[None] = Field(  # type: ignore[assignment]
        default_factory=list,
        max_length=0,
    )


class _AdditionalSemanticOperation(
    RootModel[
        Annotated[
            SetNoPreferenceOperation
            | SetNotApplicableOperation
            | SetDelegationScopeOperation
            | SetExistingBookingOperation
            | SelectPreferenceDirectionOperation
            | ExcludePreferenceDirectionOperation
            | SetLodgingClassPreferenceOperation
            | SelectConcreteEntityOperation
            | ExcludeConcreteEntityOperation
            | RevokePriorIntentOperation
            | AddConditionalRequirementOperation
            | ResolveConflictOperation,
            Field(discriminator="operation_type"),
        ]
    ]
):
    """Reuse public operations, excluding the dedicated trip_basics entry."""


class _CompoundTripBasicsDecision(V4ContractModel):
    """Required explicit basics without excluding the user's other requests."""

    trip_basics: _TripBasicsChoice
    semantic_operations: list[_AdditionalSemanticOperation] = Field(default_factory=list)
    next_action: PrepareNextAction
    tool_requests: list[ToolRequest] = Field(default_factory=list, max_length=4)
    reply_goal: ReplyGoal
    clarification: ClarificationContract | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_redundant_basics(cls, value: object) -> object:
        return _fixed_action_scope(normalize_compound_basics(value))


class _CompoundTripBasicsCapabilityDecision(_CompoundTripBasicsDecision, Generic[RequestValue]):
    tool_requests: list[_CapabilityToolRequest[RequestValue]] = Field(  # type: ignore[assignment]
        min_length=1, max_length=4
    )
    next_action: UseToolAction
    clarification: None = None


class _LodgingBookingChoice(V4ContractModel):
    operation_type: Literal["set_existing_booking"]
    domain: Literal["lodging"]
    booking_kind: Literal["lodging"]
    user_description: DisplayText
    start_date: date | None = None
    end_date: date | None = None
    confidence: ConfidenceLevel

    @model_validator(mode="after")
    def dates_are_ordered(self) -> _LodgingBookingChoice:
        if (
            self.start_date is not None
            and self.end_date is not None
            and self.end_date < self.start_date
        ):
            raise ValueError("end_date cannot be before start_date")
        return self


class _LodgingBookingPrepareDecision(_RuntimePrepareDecision):
    lodging_booking: _LodgingBookingChoice


class _LodgingBookingCapabilityDecision(
    _CapabilityPrepareDecision[RequestValue],
    Generic[RequestValue],
):
    # The required booking is separate from the user's other requests, just as
    # in the compound trip-basics contract. A compound turn is not a one-op turn.
    lodging_booking: _LodgingBookingChoice


class _DiningRequirementChoice(V4ContractModel):
    operation_type: Literal["add_conditional_requirement"]
    domain: Literal["dining"]
    condition: DisplayText
    required_outcome: DisplayText
    confidence: ConfidenceLevel


class _DiningRequirementOperationChoice(RootModel[_DiningRequirementChoice]):
    pass


class _PaceRequirementChoice(V4ContractModel):
    operation_type: Literal["add_conditional_requirement"]
    domain: Literal["transport"]
    condition: DisplayText
    required_outcome: DisplayText
    confidence: ConfidenceLevel


class _PaceRequirementOperationChoice(RootModel[_PaceRequirementChoice]):
    pass


class _DiningRequirementPrepareDecision(_RuntimePrepareDecision):
    semantic_operations: list[_DiningRequirementOperationChoice] = Field(  # type: ignore[assignment]
        min_length=1,
        max_length=1,
    )


class _PaceRequirementPrepareDecision(V4ContractModel):
    semantic_operations: list[_PaceRequirementOperationChoice] = Field(
        min_length=1,
        max_length=1,
    )
    reply_goal: ReplyGoal


class _DiningRequirementCapabilityDecision(
    _CapabilityPrepareDecision[RequestValue],
    Generic[RequestValue],
):
    semantic_operations: list[_DiningRequirementOperationChoice] = Field(  # type: ignore[assignment]
        min_length=1,
        max_length=1,
    )


class _ModelFinalSupplementAction(FinalSupplementAction):
    requested_targets: list[Identifier] = Field(default_factory=lambda: ["final_supplement"])

    @field_validator("requested_targets", mode="before")
    @classmethod
    def target_is_program_owned(cls, value: object) -> list[str]:
        return ["final_supplement"]


NoToolAction = Annotated[
    ReplyOnlyAction
    | ShowPreferenceCardAction
    | ShowSpecificCardAction
    | AskClarificationAction
    | _ModelFinalSupplementAction
    | GenerateTaskBookAction,
    Field(discriminator="kind"),
]


class _CompoundTripBasicsNoToolDecision(_CompoundTripBasicsDecision):
    next_action: NoToolAction
    tool_requests: list[None] = Field(default_factory=list, max_length=0)  # type: ignore[assignment]


class _NoToolPrepareDecision(_RuntimePrepareDecision):
    next_action: NoToolAction
    tool_requests: list[None] = Field(default_factory=list, max_length=0)  # type: ignore[assignment]


class _NoSemanticNoToolPrepareDecision(_NoToolPrepareDecision):
    semantic_operations: list[None] = Field(  # type: ignore[assignment]
        default_factory=list,
        max_length=0,
    )


class _CardNeedsInputDecision(V4ContractModel):
    next_action: AskClarificationAction
    reply_goal: ReplyGoal
    clarification: ClarificationContract


CardUnavailableAction = Annotated[
    ReplyOnlyAction | AskClarificationAction,
    Field(discriminator="kind"),
]


class _CardUnavailableDecision(V4ContractModel):
    next_action: CardUnavailableAction
    reply_goal: ReplyGoal
    clarification: ClarificationContract | None = None

    @model_validator(mode="after")
    def clarification_matches_action(self) -> _CardUnavailableDecision:
        asks = self.next_action.kind is PrepareActionKind.ASK_CLARIFICATION
        if asks != (self.clarification is not None):
            raise ValueError("card observation clarification must match the selected action")
        return self


class _TripBasicsModelDecision(V4ContractModel):
    semantic_operations: list[_TripBasicsOperationChoice] = Field(
        min_length=1,
        max_length=1,
    )
    next_action: NoToolAction
    reply_goal: ReplyGoal


class _IntakeFollowupDecision(V4ContractModel):
    """Qwen owns semantics and prose, not a forced interaction's protocol fields.

    Computed fields are absent from the model's validation Schema, but present
    when compiling into the unchanged public PrepareDecision. This model is
    selected only after the graph has established the intake follow-up gate.
    """

    followup_target: ClassVar[str] = "date_range"
    followup_reason: ClassVar[str] = "需确认本次旅行的具体起止日期。"
    reply_goal: ReplyGoal

    @model_validator(mode="before")
    @classmethod
    def ignore_legacy_protocol_fields(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        # Rolling prompt/context compatibility: the old model may echo these
        # fields (including a question in target). They have no authority here.
        payload.pop("next_action", None)
        payload.pop("clarification", None)
        if payload.get("tool_requests") == []:
            payload.pop("tool_requests")
        if (
            "semantic_operations" not in cls.model_fields
            and payload.get("semantic_operations") == []
        ):
            payload.pop("semantic_operations")
        # Never discard real tool requests, extra semantics or unknown fields.
        return payload

    @computed_field  # type: ignore[prop-decorator]
    @property
    def next_action(self) -> AskClarificationAction:
        return AskClarificationAction(
            kind=PrepareActionKind.ASK_CLARIFICATION,
            domain=PrepareDomain.GENERAL,
            requested_targets=[self.followup_target],
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def clarification(self) -> ClarificationContract:
        return ClarificationContract(target=self.followup_target, why_blocking=self.followup_reason)


class _DateRangeFollowupDecision(_IntakeFollowupDecision):
    pass


class _TripBasicsDateFollowupModelDecision(_IntakeFollowupDecision):
    semantic_operations: list[_TripBasicsOperationChoice] = Field(min_length=1, max_length=1)


class _CompoundTripBasicsDateFollowupDecision(_IntakeFollowupDecision):
    trip_basics: _TripBasicsChoice
    semantic_operations: list[_AdditionalSemanticOperation] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_redundant_basics(cls, value: object) -> object:
        return normalize_compound_basics(value)


class _TripBasicsOptionalFollowupModelDecision(_TripBasicsDateFollowupModelDecision):
    followup_target = "optional_trip_preferences"
    followup_reason = "基础信息已齐，可一次性补充其他旅行偏好。"


class _TripBasicsCompleteModelDecision(_TripBasicsModelDecision):
    next_action: ShowPreferenceCardAction


class _FinalSupplementCompleteModelDecision(V4ContractModel):
    next_action: GenerateTaskBookAction
    reply_goal: ReplyGoal


class _FinalSupplementRecoveryAction(_ModelFinalSupplementAction):
    requested_targets: list[Identifier] = Field(min_length=1, max_length=1)

    @model_validator(mode="after")
    def target_is_fixed(self) -> _FinalSupplementRecoveryAction:
        if self.requested_targets != ["final_supplement"]:
            raise ValueError("task-book recovery target is fixed")
        return self


class _TaskBookFinalSupplementRecoveryDecision(V4ContractModel):
    next_action: _FinalSupplementRecoveryAction
    reply_goal: ReplyGoal


class _AfterUpdatePreferenceCardDecision(V4ContractModel):
    next_action: ShowPreferenceCardAction
    reply_goal: ReplyGoal


class _AfterUpdateSpecificCardDecision(V4ContractModel):
    next_action: ShowSpecificCardAction
    reply_goal: ReplyGoal


class _AfterUpdateFinalSupplementDecision(V4ContractModel):
    next_action: _ModelFinalSupplementAction
    reply_goal: ReplyGoal


class _AfterTaskBookConfirmationDecision(V4ContractModel):
    next_action: ReplyOnlyAction
    reply_goal: ReplyGoal


class _LodgingBookingNoToolPrepareDecision(_NoToolPrepareDecision):
    next_action: NoToolAction
    lodging_booking: _LodgingBookingChoice
    clarification: None = None


class _ResolvedEntityChoice(V4ContractModel):
    resolution_key: Identifier | None = None


class _SelectConcreteEntityChoice(_ResolvedEntityChoice):
    operation_type: Literal["select_concrete_entity"]
    domain: Literal["attraction", "dining"]
    display_name: DisplayText
    disposition: Literal["must", "want", "destination", "if_convenient", "avoid"]
    confidence: ConfidenceLevel

    @model_validator(mode="after")
    def disposition_matches_domain(self) -> _SelectConcreteEntityChoice:
        # Validate the compact choice before the compiler binds IDs and sources.
        # Otherwise a retry receives a late, generic materialization failure.
        if self.domain == "dining" and self.disposition not in {
            "destination",
            "if_convenient",
            "avoid",
        }:
            raise ValueError("dining entity requires a dining disposition")
        if self.domain == "attraction" and self.disposition == "destination":
            raise ValueError("attraction entity requires an attraction disposition")
        return self


class _ExcludeConcreteEntityChoice(_ResolvedEntityChoice):
    operation_type: Literal["exclude_concrete_entity"]
    domain: Literal["attraction", "dining"]
    display_name: DisplayText
    confidence: ConfidenceLevel


ConcreteEntityChoiceValue = Annotated[
    _SelectConcreteEntityChoice | _ExcludeConcreteEntityChoice,
    Field(discriminator="operation_type"),
]


class _ConcreteEntityChoice(RootModel[ConcreteEntityChoiceValue]):
    pass


class _ConcreteEntityNoToolPrepareDecision(_NoToolPrepareDecision):
    next_action: NoToolAction
    semantic_operations: list[_ConcreteEntityChoice] = Field(  # type: ignore[assignment]
        max_length=8,
    )
    clarification: ClarificationContract | None = None


class _ConcreteEntityCapabilityDecision(
    _CapabilityPrepareDecision[RequestValue], Generic[RequestValue]
):
    semantic_operations: list[_ConcreteEntityChoice] = Field(  # type: ignore[assignment]
        default_factory=list, max_length=8
    )


class PrepareActionRecoveryDecision(V4ContractModel):
    """An action-only retry cannot re-extract, delete or duplicate validated intents."""

    next_action: NoToolAction
    reply_goal: ReplyGoal
    clarification: ClarificationContract | None = None


class _ConcreteClarificationDecision(_ConcreteEntityNoToolPrepareDecision):
    next_action: AskClarificationAction
    clarification: ClarificationContract


class _AttractionMustChoice(_ResolvedEntityChoice):
    operation_type: Literal["select_concrete_entity"]
    domain: Literal["attraction"]
    display_name: DisplayText
    disposition: Literal["must"]
    confidence: ConfidenceLevel


class _AttractionWantChoice(_ResolvedEntityChoice):
    operation_type: Literal["select_concrete_entity"]
    domain: Literal["attraction"]
    display_name: DisplayText
    disposition: Literal["want"]
    confidence: ConfidenceLevel


class _AttractionIfConvenientChoice(_ResolvedEntityChoice):
    operation_type: Literal["select_concrete_entity"]
    domain: Literal["attraction"]
    display_name: DisplayText
    disposition: Literal["if_convenient"]
    confidence: ConfidenceLevel


class _AttractionAvoidChoice(_ResolvedEntityChoice):
    operation_type: Literal["exclude_concrete_entity"]
    domain: Literal["attraction"]
    display_name: DisplayText
    confidence: ConfidenceLevel


class _AttractionMustOperationChoice(RootModel[_AttractionMustChoice]):
    pass


class _AttractionWantOperationChoice(RootModel[_AttractionWantChoice]):
    pass


class _AttractionIfConvenientOperationChoice(RootModel[_AttractionIfConvenientChoice]):
    pass


class _AttractionAvoidOperationChoice(RootModel[_AttractionAvoidChoice]):
    pass


class _AttractionMustNoToolPrepareDecision(_NoToolPrepareDecision):
    next_action: NoToolAction
    semantic_operations: list[_AttractionMustOperationChoice] = Field(  # type: ignore[assignment]
        min_length=1,
        max_length=1,
    )
    clarification: None = None


class _AttractionWantNoToolPrepareDecision(_NoToolPrepareDecision):
    next_action: NoToolAction
    semantic_operations: list[_AttractionWantOperationChoice] = Field(  # type: ignore[assignment]
        min_length=1,
        max_length=1,
    )
    clarification: None = None


class _AttractionIfConvenientNoToolPrepareDecision(_NoToolPrepareDecision):
    next_action: NoToolAction
    semantic_operations: list[_AttractionIfConvenientOperationChoice] = Field(  # type: ignore[assignment]
        min_length=1,
        max_length=1,
    )
    clarification: None = None


class _AttractionAvoidNoToolPrepareDecision(_NoToolPrepareDecision):
    next_action: NoToolAction
    semantic_operations: list[_AttractionAvoidOperationChoice] = Field(  # type: ignore[assignment]
        min_length=1,
        max_length=1,
    )
    clarification: None = None


class _DiningDestinationChoice(_ResolvedEntityChoice):
    operation_type: Literal["select_concrete_entity"]
    domain: Literal["dining"]
    display_name: DisplayText
    disposition: Literal["destination"]
    confidence: ConfidenceLevel


class _DiningIfConvenientChoice(_ResolvedEntityChoice):
    operation_type: Literal["select_concrete_entity"]
    domain: Literal["dining"]
    display_name: DisplayText
    disposition: Literal["if_convenient"]
    confidence: ConfidenceLevel


class _DiningAvoidChoice(_ResolvedEntityChoice):
    operation_type: Literal["exclude_concrete_entity"]
    domain: Literal["dining"]
    display_name: DisplayText
    confidence: ConfidenceLevel


class _DiningDestinationOperationChoice(RootModel[_DiningDestinationChoice]):
    pass


class _DiningIfConvenientOperationChoice(RootModel[_DiningIfConvenientChoice]):
    pass


class _DiningAvoidOperationChoice(RootModel[_DiningAvoidChoice]):
    pass


class _DiningDestinationNoToolPrepareDecision(_NoToolPrepareDecision):
    next_action: NoToolAction
    semantic_operations: list[_DiningDestinationOperationChoice] = Field(  # type: ignore[assignment]
        min_length=1,
        max_length=1,
    )
    clarification: None = None


class _DiningIfConvenientNoToolPrepareDecision(_NoToolPrepareDecision):
    next_action: NoToolAction
    semantic_operations: list[_DiningIfConvenientOperationChoice] = Field(  # type: ignore[assignment]
        min_length=1,
        max_length=1,
    )
    clarification: None = None


class _DiningAvoidNoToolPrepareDecision(_NoToolPrepareDecision):
    next_action: NoToolAction
    semantic_operations: list[_DiningAvoidOperationChoice] = Field(  # type: ignore[assignment]
        min_length=1,
        max_length=1,
    )
    clarification: None = None


_REQUIRED_CONCRETE_DECISIONS = {
    "attraction_must": _AttractionMustNoToolPrepareDecision,
    "attraction_want": _AttractionWantNoToolPrepareDecision,
    "attraction_if_convenient": _AttractionIfConvenientNoToolPrepareDecision,
    "attraction_avoid": _AttractionAvoidNoToolPrepareDecision,
    "dining_destination": _DiningDestinationNoToolPrepareDecision,
    "dining_if_convenient": _DiningIfConvenientNoToolPrepareDecision,
    "dining_avoid": _DiningAvoidNoToolPrepareDecision,
}


_CAPABILITY_DECISIONS = {
    "resolve_place": _CapabilityPrepareDecision[ResolvePlaceRequest],
    "place_facts": _CapabilityPrepareDecision[PlaceFactsRequest],
    "opening_hours": _CapabilityPrepareDecision[OpeningHoursRequest],
    "ticket_availability": _CapabilityPrepareDecision[TicketAvailabilityRequest],
    "weather_forecast": _CapabilityPrepareDecision[WeatherForecastRequest],
    "spatial_routes": _CapabilityPrepareDecision[SpatialRoutesRequest],
    "hotel_booking_facts": _CapabilityPrepareDecision[HotelBookingFactsRequest],
    "place_products": _CapabilityPrepareDecision[PlaceProductsRequest],
}

_CAPABILITY_CONTINUATION_DECISIONS = {
    "resolve_place": _CapabilityContinuationDecision[ResolvePlaceRequest],
    "place_facts": _CapabilityContinuationDecision[PlaceFactsRequest],
    "opening_hours": _CapabilityContinuationDecision[OpeningHoursRequest],
    "ticket_availability": _CapabilityContinuationDecision[TicketAvailabilityRequest],
    "weather_forecast": _CapabilityContinuationDecision[WeatherForecastRequest],
    "spatial_routes": _CapabilityContinuationDecision[SpatialRoutesRequest],
    "hotel_booking_facts": _CapabilityContinuationDecision[HotelBookingFactsRequest],
    "place_products": _CapabilityContinuationDecision[PlaceProductsRequest],
}

_CONCRETE_ENTITY_CAPABILITY_DECISIONS = {
    "resolve_place": _ConcreteEntityCapabilityDecision[ResolvePlaceRequest],
    "place_facts": _ConcreteEntityCapabilityDecision[PlaceFactsRequest],
    "opening_hours": _ConcreteEntityCapabilityDecision[OpeningHoursRequest],
    "ticket_availability": _ConcreteEntityCapabilityDecision[TicketAvailabilityRequest],
    "weather_forecast": _ConcreteEntityCapabilityDecision[WeatherForecastRequest],
    "spatial_routes": _ConcreteEntityCapabilityDecision[SpatialRoutesRequest],
    "hotel_booking_facts": _ConcreteEntityCapabilityDecision[HotelBookingFactsRequest],
    "place_products": _ConcreteEntityCapabilityDecision[PlaceProductsRequest],
}

_LODGING_BOOKING_CAPABILITY_DECISIONS = {
    "resolve_place": _LodgingBookingCapabilityDecision[ResolvePlaceRequest],
    "place_facts": _LodgingBookingCapabilityDecision[PlaceFactsRequest],
    "opening_hours": _LodgingBookingCapabilityDecision[OpeningHoursRequest],
    "ticket_availability": _LodgingBookingCapabilityDecision[TicketAvailabilityRequest],
    "weather_forecast": _LodgingBookingCapabilityDecision[WeatherForecastRequest],
    "spatial_routes": _LodgingBookingCapabilityDecision[SpatialRoutesRequest],
    "hotel_booking_facts": _LodgingBookingCapabilityDecision[HotelBookingFactsRequest],
    "place_products": _LodgingBookingCapabilityDecision[PlaceProductsRequest],
}

_DINING_REQUIREMENT_CAPABILITY_DECISIONS = {
    "resolve_place": _DiningRequirementCapabilityDecision[ResolvePlaceRequest],
    "place_facts": _DiningRequirementCapabilityDecision[PlaceFactsRequest],
    "opening_hours": _DiningRequirementCapabilityDecision[OpeningHoursRequest],
    "ticket_availability": _DiningRequirementCapabilityDecision[TicketAvailabilityRequest],
    "weather_forecast": _DiningRequirementCapabilityDecision[WeatherForecastRequest],
    "spatial_routes": _DiningRequirementCapabilityDecision[SpatialRoutesRequest],
    "hotel_booking_facts": _DiningRequirementCapabilityDecision[HotelBookingFactsRequest],
    "place_products": _DiningRequirementCapabilityDecision[PlaceProductsRequest],
}


_COMPOUND_BASICS_CAPABILITY_DECISIONS = {
    "resolve_place": _CompoundTripBasicsCapabilityDecision[ResolvePlaceRequest],
    "place_facts": _CompoundTripBasicsCapabilityDecision[PlaceFactsRequest],
    "opening_hours": _CompoundTripBasicsCapabilityDecision[OpeningHoursRequest],
    "ticket_availability": _CompoundTripBasicsCapabilityDecision[TicketAvailabilityRequest],
    "weather_forecast": _CompoundTripBasicsCapabilityDecision[WeatherForecastRequest],
    "spatial_routes": _CompoundTripBasicsCapabilityDecision[SpatialRoutesRequest],
    "hotel_booking_facts": _CompoundTripBasicsCapabilityDecision[HotelBookingFactsRequest],
    "place_products": _CompoundTripBasicsCapabilityDecision[PlaceProductsRequest],
}


def decision_contract_for_published_plan(
    contract: type[PrepareDecision], *, required: bool
) -> type[PrepareDecision]:
    """New model decisions must classify intent; persisted old records stay valid."""
    return _published_plan_contract(contract) if required else contract


@lru_cache(maxsize=64)
def _published_plan_contract(contract: type[PrepareDecision]) -> type[PrepareDecision]:
    return create_model(
        f"Published{contract.__name__}",
        __base__=contract,
        published_plan_intent=(PublishedPlanIntent, ...),
    )


def decision_contract_for_capability(
    capability: str | None,
    *,
    tools_allowed: bool = True,
    continuation: bool = False,
    no_tool_semantics: Literal["full", "none", "concrete_entity", "lodging_booking"] = "full",
    required_semantic_operation: Literal[
        "none",
        "trip_basics",
        "trip_basics_with_additions",
        "lodging_booking",
        "dining_requirement",
        "final_supplement",
        "pace_requirement",
    ] = "none",
    trip_intake_transition_action: Literal[
        "none",
        "ask_trip_dates",
        "ask_optional_preferences",
        "show_attraction_preferences",
    ] = "none",
    required_post_update_action: Literal[
        "none",
        "reply_only",
        "show_preference_card",
        "show_specific_card",
        "final_supplement",
    ] = "none",
    forbid_semantic_operations: bool = False,
    required_concrete_choice: Literal[
        "none",
        "attraction_must",
        "attraction_want",
        "attraction_if_convenient",
        "attraction_avoid",
        "dining_destination",
        "dining_if_convenient",
        "dining_avoid",
    ] = "none",
    card_observation_status: Literal["needs_input", "unavailable"] | None = None,
    task_book_observation_failure_code: Literal["final_supplement_incomplete"] | None = None,
    entity_clarification_required: bool = False,
) -> type[PrepareDecision]:
    if entity_clarification_required:
        return cast(type[PrepareDecision], _ConcreteClarificationDecision)
    if task_book_observation_failure_code == "final_supplement_incomplete":
        return cast(type[PrepareDecision], _TaskBookFinalSupplementRecoveryDecision)
    if card_observation_status == "needs_input":
        return cast(type[PrepareDecision], _CardNeedsInputDecision)
    if card_observation_status == "unavailable":
        return cast(type[PrepareDecision], _CardUnavailableDecision)
    if capability is None and trip_intake_transition_action == "ask_trip_dates":
        if required_semantic_operation == "trip_basics_with_additions":
            return cast(type[PrepareDecision], _CompoundTripBasicsDateFollowupDecision)
        if required_semantic_operation == "trip_basics":
            return cast(type[PrepareDecision], _TripBasicsDateFollowupModelDecision)
        return cast(type[PrepareDecision], _DateRangeFollowupDecision)
    if required_post_update_action != "none":
        contracts = {
            "reply_only": _AfterTaskBookConfirmationDecision,
            "show_preference_card": _AfterUpdatePreferenceCardDecision,
            "show_specific_card": _AfterUpdateSpecificCardDecision,
            "final_supplement": _AfterUpdateFinalSupplementDecision,
        }
        return cast(type[PrepareDecision], contracts[required_post_update_action])
    if required_semantic_operation == "trip_basics_with_additions":
        if not tools_allowed:
            return cast(type[PrepareDecision], _CompoundTripBasicsNoToolDecision)
        if capability is None:
            return cast(type[PrepareDecision], _CompoundTripBasicsDecision)
        compound_contract = _COMPOUND_BASICS_CAPABILITY_DECISIONS.get(capability)
        if compound_contract is None:
            raise ValueError("unsupported guarded Prepare capability")
        return cast(type[PrepareDecision], compound_contract)
    if not tools_allowed:
        if no_tool_semantics == "none":
            return _NoSemanticNoToolPrepareDecision
        if no_tool_semantics == "concrete_entity":
            if required_concrete_choice != "none":
                return cast(
                    type[PrepareDecision],
                    _REQUIRED_CONCRETE_DECISIONS[required_concrete_choice],
                )
            return _ConcreteEntityNoToolPrepareDecision
        if no_tool_semantics == "lodging_booking":
            return _LodgingBookingNoToolPrepareDecision
        return _NoToolPrepareDecision
    if capability is None and required_semantic_operation == "trip_basics":
        if trip_intake_transition_action == "ask_optional_preferences":
            return cast(type[PrepareDecision], _TripBasicsOptionalFollowupModelDecision)
        return cast(
            type[PrepareDecision],
            (
                _TripBasicsCompleteModelDecision
                if trip_intake_transition_action == "show_attraction_preferences"
                else _TripBasicsModelDecision
            ),
        )
    if capability is None and required_semantic_operation == "dining_requirement":
        return _DiningRequirementPrepareDecision
    if capability is None and required_semantic_operation == "lodging_booking":
        return _LodgingBookingPrepareDecision
    if capability is None and required_semantic_operation == "final_supplement":
        return cast(type[PrepareDecision], _FinalSupplementCompleteModelDecision)
    if capability is None and required_semantic_operation == "pace_requirement":
        return cast(type[PrepareDecision], _PaceRequirementPrepareDecision)
    if capability is None:
        return _RuntimePrepareDecision
    if continuation and no_tool_semantics == "concrete_entity" and not forbid_semantic_operations:
        concrete_contract = _CONCRETE_ENTITY_CAPABILITY_DECISIONS.get(capability)
        if concrete_contract is None:
            raise ValueError("unsupported guarded Prepare capability")
        return cast(type[PrepareDecision], concrete_contract)
    if required_semantic_operation == "lodging_booking" and not continuation:
        contracts = _LODGING_BOOKING_CAPABILITY_DECISIONS
    elif required_semantic_operation == "dining_requirement" and not continuation:
        contracts = _DINING_REQUIREMENT_CAPABILITY_DECISIONS
    else:
        contracts = (
            _CAPABILITY_CONTINUATION_DECISIONS
            if (continuation and no_tool_semantics == "none") or forbid_semantic_operations
            else _CAPABILITY_DECISIONS
        )
    contract = contracts.get(capability)
    if contract is None:
        raise ValueError("unsupported guarded Prepare capability")
    return cast(type[PrepareDecision], contract)


__all__ = [
    "CardActionObservation",
    "CompoundTripIntakeExtraction",
    "FinalSupplementAssessment",
    "GroundedPlaceReference",
    "PaceModificationAssessment",
    "PaceRequirementExtraction",
    "TaskBookReviewModificationAssessment",
    "TaskBookActionObservation",
    "TripBasicsAssessment",
    "TripDateRangeAssessment",
    "decision_contract_for_capability",
]
