"""Protocol enums. Values are stable and user-facing copy lives elsewhere."""

from enum import IntEnum, StrEnum


class CityCode(StrEnum):
    BEIJING = "beijing"
    NANJING = "nanjing"


class TripPhase(StrEnum):
    COLD_START = "cold_start"
    CITY_SELECTION = "city_selection"
    CITY_BRIEF = "city_brief"
    TRIP_SETUP = "trip_setup"
    INTEREST_SELECTION = "interest_selection"
    ATTRACTION_SELECTION = "attraction_selection"
    DINING_SELECTION = "dining_selection"
    LODGING_SELECTION = "lodging_selection"
    TASK_REFLECTION = "task_reflection"
    PLANNING = "planning"
    DRAFT_READY = "draft_ready"
    REVISING = "revising"
    CONFIRMED = "confirmed"


class DayStart(StrEnum):
    BEFORE_07 = "before_07"
    AROUND_08 = "around_08"
    AROUND_09 = "around_09"
    AROUND_10 = "around_10"
    AFTER_11 = "after_11"
    FLEXIBLE = "flexible"


class DayReturn(StrEnum):
    BEFORE_20 = "before_20"
    AROUND_21 = "around_21"
    AFTER_22 = "after_22"
    FLEXIBLE = "flexible"


class MobilityTolerance(StrEnum):
    NEVER = "never"
    WITHIN_5 = "within_5"
    AROUND_10 = "around_10"
    FIFTEEN_PLUS = "15_plus"


class FiveLevel(IntEnum):
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5


class PriorityGoal(StrEnum):
    MUST_SEE_PLACES = "must_see_places"
    COMFORTABLE_STAY = "comfortable_stay"
    SATISFYING_FOOD = "satisfying_food"
    SMOOTH_ROUTES = "smooth_routes"
    GOOD_VALUE = "good_value"


class PreferenceDecisionMode(StrEnum):
    USE_DEFAULTS = "use_defaults"
    ADJUST_FOR_TRIP = "adjust_for_trip"
    USE_NEUTRAL = "use_neutral"


class CityBriefStatus(StrEnum):
    VIEWED = "viewed"
    SKIPPED_INTRO = "skipped_intro"


class CityThemeMode(StrEnum):
    SELECTED = "selected"
    OPEN_TO_ANY = "open_to_any"


class AttractionIntent(StrEnum):
    MUST = "must"
    WANT = "want"
    IF_CONVENIENT = "if_convenient"
    AVOID = "avoid"


class RestaurantIntent(StrEnum):
    DESTINATION = "destination"
    IF_CONVENIENT = "if_convenient"
    AVOID = "avoid"


class RecommendationIntent(StrEnum):
    MUST = "must"
    WANT = "want"
    IF_CONVENIENT = "if_convenient"
    AVOID = "avoid"


class FeedbackSource(StrEnum):
    CARD = "card"
    DIALOGUE = "dialogue"
    SYSTEM_DEFAULT = "system_default"


class DiningOptionKind(StrEnum):
    LOCAL_CUISINE = "local_cuisine"
    POPULAR_EATERY = "popular_eatery"


class CoordinateSystem(StrEnum):
    GCJ_02 = "gcj_02"
    WGS_84 = "wgs_84"
    BD_09 = "bd_09"


class ProviderCode(StrEnum):
    AMAP = "amap"
    BAIDU = "baidu"
    FLYAI = "flyai"
    WEATHER = "weather"
    CITY_CONTENT = "city_content"
    OFFICIAL = "official"
    MANUAL = "manual"


class PlaceCategory(StrEnum):
    ATTRACTION = "attraction"
    RESTAURANT = "restaurant"
    HOTEL = "hotel"
    TRANSPORT = "transport"
    ACTIVITY = "activity"
    OTHER = "other"


class DataAvailability(StrEnum):
    AVAILABLE = "available"
    PARTIAL = "partial"
    MISSING = "missing"


class PlaceFactKind(StrEnum):
    ADDRESS = "address"
    RATING = "rating"
    POPULARITY = "popularity"
    REVIEW_COUNT = "review_count"
    REGULAR_HOURS = "regular_hours"
    PRICE = "price"
    PHONE = "phone"
    WEBSITE = "website"
    OTHER = "other"


class AnchorRole(StrEnum):
    FIXED_EVENT = "fixed_event"
    FIXED_HOTEL = "fixed_hotel"
    MUST_ATTRACTION = "must_attraction"
    WANT_ATTRACTION = "want_attraction"
    DESTINATION_RESTAURANT = "destination_restaurant"
    CONVENIENT_ATTRACTION = "convenient_attraction"
    CONVENIENT_RESTAURANT = "convenient_restaurant"


class LodgingCenterKind(StrEnum):
    AREA = "area"
    TRANSIT_STATION = "transit_station"
    WALKING_RADIUS = "walking_radius"


class HotelSelectionSource(StrEnum):
    PREBOOKED = "prebooked"
    USER_FAVORITES = "user_favorites"
    SYSTEM_CANDIDATES = "system_candidates"


class LodgingAnchorDecision(StrEnum):
    NO_PREBOOKED_HOTEL = "no_prebooked_hotel"
    SINGLE_BASE_READY = "single_base_ready"
    REQUIRES_SINGLE_BASE_CLARIFICATION = "requires_single_base_clarification"


class ItineraryEntryKind(StrEnum):
    ATTRACTION = "attraction"
    RESTAURANT = "restaurant"
    ACTIVITY = "activity"
    HOTEL = "hotel"
    REST = "rest"
    BUFFER = "buffer"


class TransportMode(StrEnum):
    WALK = "walk"
    BICYCLE = "bicycle"
    PUBLIC_TRANSIT = "public_transit"
    TAXI = "taxi"


class CostCategory(StrEnum):
    LODGING = "lodging"
    DINING = "dining"
    ATTRACTION_TICKETS = "attraction_tickets"
    LOCAL_TRANSPORT = "local_transport"


class ExcludedCostKind(StrEnum):
    AIRFARE = "airfare"
    RAIL = "rail"
    INTERCITY_TRANSPORT = "intercity_transport"


class PlanningIssueCode(StrEnum):
    DATE_CONFLICT = "date_conflict"
    OPENING_HOURS_CONFLICT = "opening_hours_conflict"
    FIXED_EVENT_CONFLICT = "fixed_event_conflict"
    INSUFFICIENT_TRAVEL_TIME = "insufficient_travel_time"
    TIME_OVERLAP = "time_overlap"
    WALKING_LIMIT = "walking_limit"
    DAILY_LOAD = "daily_load"
    LODGING_COVERAGE = "lodging_coverage"
    REPEATED_EXPERIENCE = "repeated_experience"
    CROSS_AREA_BACKTRACKING = "cross_area_backtracking"
    MEAL_TIMING = "meal_timing"
    INSUFFICIENT_BUFFER = "insufficient_buffer"
    WEATHER_RISK = "weather_risk"
    MISSING_DATA = "missing_data"


class AssumptionKind(StrEnum):
    USER_OVERRIDE = "user_override"
    SYSTEM_DEFAULT = "system_default"
    DATA_MISSING = "data_missing"
    PLANNING_TRADEOFF = "planning_tradeoff"


class EvidenceSource(StrEnum):
    COLD_START = "cold_start"
    CARD = "card"
    DIALOGUE = "dialogue"
    AGENT_INFERENCE = "agent_inference"
    SYSTEM_DEFAULT = "system_default"
    PROVIDER = "provider"


class ConfirmationStatus(StrEnum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"


class EvidenceStatus(StrEnum):
    CONFIRMED = "confirmed"
    INFERRED = "inferred"
    ASSUMED = "assumed"
    UNKNOWN = "unknown"


class Confidence(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class IssueSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    HARD = "hard"


class VersionStatus(StrEnum):
    DRAFT = "draft"
    CONFIRMED = "confirmed"
    SUPERSEDED = "superseded"


class OwnerType(StrEnum):
    ANONYMOUS = "anonymous"
    USER = "user"


class DateInputMode(StrEnum):
    SUGGESTION = "suggestion"
    NATURAL_LANGUAGE = "natural_language"


class DateSuggestionKind(StrEnum):
    NEAREST_WEEKEND = "nearest_weekend"
    NEARBY_HOLIDAY = "nearby_holiday"


class FixedEventKind(StrEnum):
    ATTRACTION = "attraction"
    RESTAURANT = "restaurant"
    HOTEL = "hotel"
    ACTIVITY = "activity"
    OTHER = "other"


class ConstraintKind(StrEnum):
    MOBILITY = "mobility"
    SCHEDULE = "schedule"
    DIETARY = "dietary"
    ACCESSIBILITY = "accessibility"
    COMPANION = "companion"
    OTHER = "other"


class CommandType(StrEnum):
    COLD_START_SUBMIT = "cold_start_submit"
    CITY_SELECT = "city_select"
    CITY_BRIEF_ACKNOWLEDGE = "city_brief_acknowledge"
    PERSONAL_DEFAULTS_CONFIRM = "personal_defaults_confirm"
    PERSONAL_DEFAULTS_ADJUST = "personal_defaults_adjust"
    PERSONAL_DEFAULTS_NOT_USE = "personal_defaults_not_use"
    TRIP_SETUP_SUBMIT = "trip_setup_submit"
    CITY_THEMES_SUBMIT = "city_themes_submit"
    ATTRACTION_FEEDBACK_SUBMIT = "attraction_feedback_submit"
    DINING_PREFERENCES_SUBMIT = "dining_preferences_submit"
    RESTAURANT_FEEDBACK_SUBMIT = "restaurant_feedback_submit"
    HOTEL_FAVORITES_SUBMIT = "hotel_favorites_submit"
    TASK_BOOK_CONFIRM = "task_book_confirm"
    USER_MESSAGE = "user_message"
    ATTACHMENT_ANSWER = "attachment_answer"
    ACCEPT_PLAN = "accept_plan"
    CANCEL_GENERATION = "cancel_generation"
    RESET_TRIP = "reset_trip"


class EventType(StrEnum):
    STATE_PATCH = "state_patch"
    STREAM_TOKEN = "stream_token"
    GESTURE_READY = "gesture_ready"
    MAP_UPDATE = "map_update"
    ITINERARY_UPDATE = "itinerary_update"
    GENERATION_STATUS = "generation_status"
    ISSUE = "issue"
    ERROR = "error"


class GestureKind(StrEnum):
    CONVERSATION = "conversation"
    CITY_BRIEF = "city_brief"
    PERSONAL_PROFILE = "personal_profile"
    TRIP_SETUP = "trip_setup"
    CITY_THEMES = "city_themes"
    ATTRACTIONS = "attractions"
    DINING_OPTIONS = "dining_options"
    RESTAURANTS = "restaurants"
    LODGING = "lodging"
    TASK_BOOK = "task_book"
    PLAN_READY = "plan_ready"
    EXPORT_READY = "export_ready"


class GenerationStatus(StrEnum):
    STARTED = "started"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class HttpMethod(StrEnum):
    GET = "GET"
    POST = "POST"
    PATCH = "PATCH"
    DELETE = "DELETE"


class RestAuthMode(StrEnum):
    PUBLIC = "public"
    SESSION = "session"


class RestResourceKind(StrEnum):
    ACCOUNT = "account"
    ANONYMOUS_SESSION = "anonymous_session"
    TRIP = "trip"
    PREFERENCE = "preference"
    EXPORT = "export"


class RestOwnershipPolicy(StrEnum):
    NONE = "none"
    SAME_OWNER = "same_owner"
    ANONYMOUS_HANDOFF = "anonymous_handoff"


class ArtifactStatus(StrEnum):
    PENDING = "pending"
    READY = "ready"
    FAILED = "failed"


class CityHistoryStatus(StrEnum):
    AVAILABLE = "available"
    NOT_APPLICABLE = "not_applicable"
    UNAVAILABLE = "unavailable"


class AssetKind(StrEnum):
    ILLUSTRATION = "illustration"
    PLACEHOLDER = "placeholder"
    PHOTO = "photo"


class AssetRightsStatus(StrEnum):
    OWNED = "owned"
    PLACEHOLDER = "placeholder"
    LICENSED = "licensed"


SHARED_ENUMS = (
    CityCode,
    TripPhase,
    DayStart,
    DayReturn,
    MobilityTolerance,
    FiveLevel,
    PriorityGoal,
    PreferenceDecisionMode,
    CityBriefStatus,
    CityThemeMode,
    AttractionIntent,
    RestaurantIntent,
    RecommendationIntent,
    FeedbackSource,
    DiningOptionKind,
    CoordinateSystem,
    ProviderCode,
    PlaceCategory,
    DataAvailability,
    PlaceFactKind,
    AnchorRole,
    LodgingCenterKind,
    HotelSelectionSource,
    LodgingAnchorDecision,
    ItineraryEntryKind,
    TransportMode,
    CostCategory,
    ExcludedCostKind,
    PlanningIssueCode,
    AssumptionKind,
    EvidenceSource,
    ConfirmationStatus,
    EvidenceStatus,
    Confidence,
    IssueSeverity,
    VersionStatus,
    OwnerType,
    DateInputMode,
    DateSuggestionKind,
    FixedEventKind,
    ConstraintKind,
    CommandType,
    EventType,
    GestureKind,
    GenerationStatus,
    HttpMethod,
    RestAuthMode,
    RestResourceKind,
    RestOwnershipPolicy,
    ArtifactStatus,
    CityHistoryStatus,
    AssetKind,
    AssetRightsStatus,
)
