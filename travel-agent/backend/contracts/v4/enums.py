"""Single-source enums for the V4 Prepare and Planner protocol."""

from __future__ import annotations

from enum import StrEnum


class DiscoverySection(StrEnum):
    OTHER = "other"
    ATTRACTION_PREFERENCE = "attraction_preference"
    ATTRACTION_SPECIFIC = "attraction_specific"
    DINING_PREFERENCE = "dining_preference"
    DINING_SPECIFIC = "dining_specific"
    LODGING_AREA_PREFERENCE = "lodging_area_preference"
    LODGING_CLASS_PREFERENCE = "lodging_class_preference"
    FINAL_SUPPLEMENT = "final_supplement"
    TASK_BOOK_REVIEW = "task_book_review"


class CoverageStatus(StrEnum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    REOPENED = "reopened"
    NOT_APPLICABLE = "not_applicable"


class CompletionMode(StrEnum):
    SELECTED = "selected"
    EXPLICITLY_NONE = "explicitly_none"
    EXISTING_BOOKING = "existing_booking"
    DELEGATED = "delegated"
    NOT_APPLICABLE = "not_applicable"


class PendingInteractionKind(StrEnum):
    PREFERENCE_CARD = "preference_card"
    SPECIFIC_CARD = "specific_card"
    FREE_TEXT_QUESTION = "free_text_question"
    CONFIRMATION = "confirmation"


class InteractionStatus(StrEnum):
    ACTIVE = "active"
    ANSWERED = "answered"
    SUPERSEDED = "superseded"
    EXPIRED = "expired"


class PrepareActionKind(StrEnum):
    REPLY_ONLY = "reply_only"
    USE_TOOL = "use_tool"
    SHOW_PREFERENCE_CARD = "show_preference_card"
    SHOW_SPECIFIC_CARD = "show_specific_card"
    ASK_CLARIFICATION = "ask_clarification"
    FINAL_SUPPLEMENT = "final_supplement"
    GENERATE_TASK_BOOK = "generate_task_book"


class PrepareDomain(StrEnum):
    ATTRACTION = "attraction"
    DINING = "dining"
    LODGING = "lodging"
    GENERAL = "general"


class SectionProposalKind(StrEnum):
    STAY = "stay"
    COMPLETE_AND_ADVANCE = "complete_and_advance"
    REOPEN = "reopen"


class ConfidenceLevel(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class SemanticOperationStatus(StrEnum):
    PROPOSED = "proposed"
    PENDING_RESOLUTION = "pending_resolution"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class ToolObservationStatus(StrEnum):
    SUCCESS = "success"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"
    INVALID_REQUEST = "invalid_request"


class ToolRequestPurpose(StrEnum):
    ANSWER_USER = "answer_user"
    VALIDATE_OPERATION = "validate_operation"
    BUILD_CARD = "build_card"


class CardDomain(StrEnum):
    ATTRACTION = "attraction"
    DINING = "dining"
    LODGING_AREA = "lodging_area"
    LODGING_CLASS = "lodging_class"


class CardKind(StrEnum):
    PREFERENCE_CARD = "preference_card"
    SPECIFIC_CARD = "specific_card"
    TASK_BOOK = "task_book"


class CardStatus(StrEnum):
    ACTIVE = "active"
    COMPLETED = "completed"
    SUPERSEDED = "superseded"
    PARTIAL_AVAILABILITY = "partial_availability"


class SelectionState(StrEnum):
    AVAILABLE = "available"
    SELECTED = "selected"
    EXCLUDED = "excluded"
    UNAVAILABLE = "unavailable"


class CompositionRole(StrEnum):
    PERSONALIZED_TOP = "personalized_top"
    REPRESENTATIVE_EXTRA = "representative_extra"


class TaskBookStatus(StrEnum):
    DRAFT = "draft"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    CONFIRMED = "confirmed"
    SUPERSEDED = "superseded"


class CommitmentLevel(StrEnum):
    IMMUTABLE = "immutable"
    STRONG = "strong"
    SOFT = "soft"
    FILLER = "filler"
    NEUTRAL = "neutral"
    FORBIDDEN = "forbidden"


class CandidateEntityKind(StrEnum):
    ATTRACTION = "attraction"
    RESTAURANT = "restaurant"
    ACTIVITY = "activity"


class CrossClusterReasonCode(StrEnum):
    RESERVATION_OR_FIXED_COMMITMENT = "reservation_or_fixed_commitment"
    DATE_SPECIFIC_AVAILABILITY = "date_specific_availability"
    STRONG_USER_INTENT = "strong_user_intent"
    LODGING_OR_TRANSPORT_ANCHOR = "lodging_or_transport_anchor"
    VERIFIED_GLOBAL_ROUTE_IMPROVEMENT = "verified_global_route_improvement"


class PlannerCapability(StrEnum):
    CANDIDATE_RECALL = "candidate_recall"
    PLACE_FACTS = "place_facts"
    OPENING_HOURS = "opening_hours"
    TICKET_AVAILABILITY = "ticket_availability"
    WEATHER_FORECAST = "weather_forecast"
    SPATIAL_ROUTES = "spatial_routes"
    HOTEL_SEARCH = "hotel_search"
    HOTEL_OFFER_REFRESH = "hotel_offer_refresh"


class PlannerStatus(StrEnum):
    PLANNING = "planning"
    AWAITING_USER = "awaiting_user"
    DRAFT_READY = "draft_ready"
    READY_TO_PUBLISH = "ready_to_publish"
    FAILED = "failed"
    CANCELLED = "cancelled"
    STALE = "stale"


class AskUserReasonCode(StrEnum):
    INCOMPATIBLE_STRONG_COMMITMENTS = "incompatible_strong_commitments"
    FIXED_BOOKING_CONFLICT = "fixed_booking_conflict"
    PERMISSION_TO_RELAX_CONSTRAINT = "permission_to_relax_constraint"
    MATERIAL_TRADEOFF_OUTSIDE_DELEGATION = "material_tradeoff_outside_delegation"
    MISSING_USER_OWNED_BOOKING_DETAIL = "missing_user_owned_booking_detail"
    REVISION_BUDGET_EXHAUSTED = "revision_budget_exhausted"


V4_SHARED_ENUMS = (
    DiscoverySection,
    CoverageStatus,
    CompletionMode,
    PendingInteractionKind,
    InteractionStatus,
    PrepareActionKind,
    PrepareDomain,
    SectionProposalKind,
    ConfidenceLevel,
    SemanticOperationStatus,
    ToolRequestPurpose,
    ToolObservationStatus,
    CardDomain,
    CardKind,
    CardStatus,
    SelectionState,
    CompositionRole,
    TaskBookStatus,
    CommitmentLevel,
    CandidateEntityKind,
    CrossClusterReasonCode,
    PlannerCapability,
    PlannerStatus,
    AskUserReasonCode,
)
