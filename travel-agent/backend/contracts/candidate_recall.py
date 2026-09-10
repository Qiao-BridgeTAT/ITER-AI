"""V3-30 contracts for bounded, source-backed candidate recall."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from enum import StrEnum
from typing import Any, Literal
from uuid import UUID

from pydantic import AwareDatetime, ConfigDict, Field, HttpUrl, model_validator

from backend.contracts.base import ContractModel
from backend.contracts.common import CnyAmountRange, NonEmptyText, ShortText
from backend.contracts.enums import DataAvailability, PlaceCategory, ProviderCode
from backend.contracts.places import Gcj02Coordinates

RECALL_ATTEMPT_SCHEMA_RULE: dict[str, Any] = {
    "allOf": [
        {
            "if": {
                "properties": {"status": {"const": "partial"}},
                "required": ["status"],
            },
            "then": {
                "anyOf": [
                    {
                        "properties": {"missing_fields": {"type": "array", "minItems": 1}},
                        "required": ["missing_fields"],
                    },
                    {
                        "properties": {"provider_notice": {"type": "string"}},
                        "required": ["provider_notice"],
                    },
                    {
                        "properties": {"failure_reason": {"type": "string"}},
                        "required": ["failure_reason"],
                    },
                ]
            },
        }
    ]
}


class ImmutableRecallModel(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CandidateDomain(StrEnum):
    ATTRACTION = "attraction"
    RESTAURANT = "restaurant"
    HOTEL = "hotel"


class RecallChannel(StrEnum):
    USER_NAMED = "user_named"
    SELECTED_THEME = "selected_theme"
    CITY_LANDMARK = "city_landmark"
    CITY_FEATURE = "city_feature"
    EXPLORATION = "exploration"
    NEARBY_ANCHOR = "nearby_anchor"


class RecallThemeMode(StrEnum):
    SELECTED = "selected"
    OPEN_TO_ANY = "open_to_any"
    UNSPECIFIED = "unspecified"


class LandmarkRecallPolicy(StrEnum):
    INCLUDE = "include"
    NEUTRAL = "neutral"
    EXCLUDE = "exclude"


class RecallThemeSource(StrEnum):
    CITY_CONTENT = "city_content"
    CURRENT_TRIP = "current_trip"
    PERSONAL_DEFAULT = "personal_default"
    GENERIC = "generic"


class NamedPlacePriority(StrEnum):
    MUST = "must"
    WANT = "want"
    MENTIONED = "mentioned"


class RecallSourceKind(StrEnum):
    USER_EVIDENCE = "user_evidence"
    CITY_CONTENT = "city_content"
    PROVIDER = "provider"


class RecallAttemptStatus(StrEnum):
    AVAILABLE = "available"
    PARTIAL = "partial"
    EMPTY = "empty"
    FAILED = "failed"


class RecallFailureCode(StrEnum):
    INVALID_REQUEST = "invalid_request"
    AUTHENTICATION_FAILED = "authentication_failed"
    PERMISSION_DENIED = "permission_denied"
    RATE_LIMITED = "rate_limited"
    TIMEOUT = "timeout"
    UNAVAILABLE = "unavailable"
    MALFORMED_RESPONSE = "malformed_response"
    UPSTREAM_ERROR = "upstream_error"


class RecallThemeInput(ImmutableRecallModel):
    theme_id: NonEmptyText
    label: NonEmptyText
    source: RecallThemeSource
    source_reference_ids: tuple[NonEmptyText, ...] = ()

    @model_validator(mode="after")
    def source_references_match_source(self) -> RecallThemeInput:
        if len(set(self.source_reference_ids)) != len(self.source_reference_ids):
            raise ValueError("theme source references must be unique")
        if self.source is RecallThemeSource.CITY_CONTENT and not self.source_reference_ids:
            raise ValueError("city-content themes require source references")
        return self


class NamedPlaceClue(ImmutableRecallModel):
    clue_id: UUID
    name: NonEmptyText
    domain: CandidateDomain
    priority: NamedPlacePriority
    source_message_id: UUID
    known_place_id: UUID | None = None


class ExcludedPlaceClue(ImmutableRecallModel):
    clue_id: UUID
    name: NonEmptyText
    source_message_id: UUID
    domain: CandidateDomain | None = None
    known_place_id: UUID | None = None


class RecallAnchor(ImmutableRecallModel):
    anchor_id: UUID
    name: NonEmptyText
    domain: CandidateDomain
    coordinates: Gcj02Coordinates


class DomainRecallBudget(ImmutableRecallModel):
    domain: CandidateDomain
    max_candidates: int = Field(ge=1, le=60, strict=True)
    max_provider_calls: int = Field(ge=0, le=12, strict=True)


class CandidateRecallBudget(ImmutableRecallModel):
    max_total_candidates: int = Field(ge=1, le=60, strict=True)
    max_total_provider_calls: int = Field(ge=0, le=12, strict=True)
    domains: tuple[DomainRecallBudget, ...] = Field(min_length=1, max_length=3)

    @model_validator(mode="after")
    def domain_limits_fit_total_limits(self) -> CandidateRecallBudget:
        domain_ids = [item.domain for item in self.domains]
        if len(set(domain_ids)) != len(domain_ids):
            raise ValueError("recall domain budgets must be unique")
        if sum(item.max_candidates for item in self.domains) > self.max_total_candidates:
            raise ValueError("domain candidate budgets exceed the total candidate budget")
        if sum(item.max_provider_calls for item in self.domains) > self.max_total_provider_calls:
            raise ValueError("domain provider-call budgets exceed the total call budget")
        return self


class CandidateCompositionFeedback(ImmutableRecallModel):
    """Server-observed shortfall for the one bounded discovery supplement."""

    domain: CandidateDomain
    missing_personalized_count: int = Field(ge=0, le=14, strict=True)
    missing_representative_count: int = Field(ge=0, le=12, strict=True)
    frozen_top_names: tuple[ShortText, ...] = Field(default=(), max_length=14)
    observed_alternative_names: tuple[ShortText, ...] = Field(default=(), max_length=40)
    excluded_restaurant_brands: tuple[ShortText, ...] = Field(default=(), max_length=40)
    failed_named_queries: tuple[ShortText, ...] = Field(default=(), max_length=8)

    @model_validator(mode="after")
    def requires_a_real_shortfall(self) -> CandidateCompositionFeedback:
        maximum = 6 if self.domain is CandidateDomain.RESTAURANT else 2
        if (
            self.domain is not CandidateDomain.ATTRACTION
            and self.missing_representative_count > maximum
        ):
            raise ValueError(f"representative shortfall cannot exceed {maximum} for this domain")
        if self.missing_personalized_count + self.missing_representative_count == 0:
            raise ValueError("supplemental composition feedback requires a candidate shortfall")
        return self


class DiscoveryPreferenceDirection(ImmutableRecallModel):
    direction_id: NonEmptyText
    label: NonEmptyText
    description: NonEmptyText | None = None
    tags: tuple[NonEmptyText, ...] = ()
    search_query: NonEmptyText | None = None
    selected: bool
    source_reference_ids: tuple[NonEmptyText, ...] = ()


class AttractionDiscoveryContext(ImmutableRecallModel):
    """Optional V4 semantic context; V3 callers retain their existing policy."""

    strategy_version: Literal["attraction-v2"] = "attraction-v2"
    destination_name: NonEmptyText | None = None
    directions: tuple[DiscoveryPreferenceDirection, ...] = ()
    explicit_place_intents: tuple[NonEmptyText, ...] = ()
    travelers: tuple[NonEmptyText, ...] = ()
    trip_goals: tuple[NonEmptyText, ...] = ()
    cold_start_defaults: tuple[NonEmptyText, ...] = ()
    pace_preferences: tuple[NonEmptyText, ...] = ()
    transport_preferences: tuple[NonEmptyText, ...] = ()
    minimum_target: int = Field(ge=1, le=12)
    maximum_target: int = Field(ge=1, le=12)
    city_target: int = Field(ge=0, le=12)
    personalized_target: int = Field(ge=0, le=12)
    screening_feedback: tuple[NonEmptyText, ...] = Field(default=(), max_length=40)


class CandidateRecallRequest(ImmutableRecallModel):
    request_id: UUID
    trip_id: UUID
    semantic_state_version: int = Field(default=0, ge=0, strict=True)
    task_book_id: UUID | None = None
    task_book_revision: int | None = Field(default=None, ge=1, strict=True)
    city_id: NonEmptyText
    start_date: date | None = None
    end_date: date | None = None
    duration_days: int | None = Field(default=None, ge=1, le=5, strict=True)
    theme_mode: RecallThemeMode = RecallThemeMode.UNSPECIFIED
    themes: tuple[RecallThemeInput, ...] = ()
    free_text_clues: tuple[ShortText, ...] = ()
    named_places: tuple[NamedPlaceClue, ...] = ()
    excluded_places: tuple[ExcludedPlaceClue, ...] = ()
    personal_preference_clues: tuple[ShortText, ...] = ()
    special_constraints: tuple[ShortText, ...] = ()
    anchors: tuple[RecallAnchor, ...] = ()
    landmark_policy: LandmarkRecallPolicy = LandmarkRecallPolicy.NEUTRAL
    composition_feedback: CandidateCompositionFeedback | None = None
    attraction_discovery: AttractionDiscoveryContext | None = None
    dining_city_target: int | None = Field(default=None, ge=1, le=6, strict=True)
    budget: CandidateRecallBudget

    @model_validator(mode="after")
    def dates_references_and_budgets_are_consistent(self) -> CandidateRecallRequest:
        if (self.task_book_id is None) != (self.task_book_revision is None):
            raise ValueError("task-book ID and revision must be supplied together")
        if (self.start_date is None) != (self.end_date is None):
            raise ValueError("candidate recall dates must be supplied together")
        if self.start_date is not None and self.end_date is not None:
            day_count = (self.end_date - self.start_date).days + 1
            if not 1 <= day_count <= 5:
                raise ValueError("candidate recall date range must contain 1 to 5 days")
            if self.duration_days is not None and self.duration_days != day_count:
                raise ValueError("duration_days must match the candidate recall date range")
        elif self.duration_days is None:
            raise ValueError("candidate recall requires dates or duration_days")
        _unique([theme.theme_id for theme in self.themes], "theme IDs")
        _unique([clue.clue_id for clue in self.named_places], "named-place clue IDs")
        _unique([clue.clue_id for clue in self.excluded_places], "excluded-place clue IDs")
        _unique([anchor.anchor_id for anchor in self.anchors], "anchor IDs")
        _unique(self.free_text_clues, "free-text clues")
        _unique(self.personal_preference_clues, "personal preference clues")
        _unique(self.special_constraints, "special constraints")
        if self.theme_mode is RecallThemeMode.SELECTED and not self.themes:
            raise ValueError("selected theme mode requires at least one theme")
        if self.theme_mode is RecallThemeMode.OPEN_TO_ANY and self.themes:
            raise ValueError("open-to-any theme mode cannot contain selected themes")
        requested_domains = {item.domain for item in self.budget.domains}
        if self.composition_feedback is not None and (
            self.composition_feedback.domain not in requested_domains
        ):
            raise ValueError("composition feedback must belong to a budgeted domain")
        referenced_domains = {
            *(clue.domain for clue in self.named_places),
            *(anchor.domain for anchor in self.anchors),
            *(clue.domain for clue in self.excluded_places if clue.domain is not None),
        }
        if not referenced_domains <= requested_domains:
            raise ValueError("recall clues and anchors must belong to budgeted domains")
        if len(self.named_places) > self.budget.max_total_provider_calls:
            raise ValueError("provider-call budget cannot cover all named places")
        return self

    @property
    def day_count(self) -> int:
        if self.duration_days is not None:
            return self.duration_days
        assert self.start_date is not None and self.end_date is not None
        return (self.end_date - self.start_date).days + 1


class ContentSeedRecallQuery(ImmutableRecallModel):
    query_id: NonEmptyText
    channel: RecallChannel
    theme_ids: tuple[NonEmptyText, ...] = ()
    max_results: int = Field(ge=1, le=20, strict=True)
    reason: ShortText

    @model_validator(mode="after")
    def channel_can_use_city_content(self) -> ContentSeedRecallQuery:
        if self.channel not in {
            RecallChannel.SELECTED_THEME,
            RecallChannel.CITY_LANDMARK,
            RecallChannel.CITY_FEATURE,
            RecallChannel.EXPLORATION,
        }:
            raise ValueError("content recall query uses an unsupported channel")
        return self


class ProviderRecallQuery(ImmutableRecallModel):
    query_id: NonEmptyText
    channel: RecallChannel
    domain: CandidateDomain
    keyword: NonEmptyText
    exact_name_match: bool = False
    theme_ids: tuple[NonEmptyText, ...] = ()
    typecodes: tuple[NonEmptyText, ...] = ()
    named_clue_id: UUID | None = None
    anchor_id: UUID | None = None
    radius_m: int | None = Field(default=None, ge=100, le=50_000, strict=True)
    max_results: int = Field(ge=1, le=25, strict=True)
    reason: ShortText

    @model_validator(mode="after")
    def references_match_channel(self) -> ProviderRecallQuery:
        _unique(self.theme_ids, "provider query theme IDs")
        _unique(self.typecodes, "provider query typecodes")
        if any(not code.isdigit() or len(code) not in {4, 6} for code in self.typecodes):
            raise ValueError("provider typecodes must contain 4 or 6 digits")
        if self.channel is RecallChannel.USER_NAMED:
            if self.named_clue_id is None or self.anchor_id is not None:
                raise ValueError("user-named queries require only named_clue_id")
        elif self.named_clue_id is not None:
            raise ValueError("only user-named queries may reference a named clue")
        if self.channel is RecallChannel.NEARBY_ANCHOR:
            if self.anchor_id is None or self.radius_m is None:
                raise ValueError("nearby queries require an anchor and radius")
        elif self.anchor_id is not None or self.radius_m is not None:
            raise ValueError("only nearby queries may reference an anchor or radius")
        return self


class RecallPlan(ImmutableRecallModel):
    plan_version: str = Field(pattern=r"^[1-9]\d*\.\d+\.\d+$")
    request_id: UUID
    city_id: NonEmptyText
    content_queries: tuple[ContentSeedRecallQuery, ...] = ()
    provider_queries: tuple[ProviderRecallQuery, ...] = ()

    @model_validator(mode="after")
    def query_ids_are_unique(self) -> RecallPlan:
        query_ids = [
            *(query.query_id for query in self.content_queries),
            *(query.query_id for query in self.provider_queries),
        ]
        _unique(query_ids, "recall query IDs")
        if not query_ids:
            raise ValueError("recall plan requires at least one query")
        return self


class CandidateSourceReference(ImmutableRecallModel):
    kind: RecallSourceKind
    source_record_id: NonEmptyText
    provider: ProviderCode | None = None
    source_place_id: NonEmptyText | None = None
    source_url: HttpUrl | None = None
    fetched_at: AwareDatetime | None = None
    content_version: NonEmptyText | None = None

    @model_validator(mode="after")
    def fields_match_source_kind(self) -> CandidateSourceReference:
        if self.kind is RecallSourceKind.PROVIDER:
            if self.provider is None or self.source_place_id is None or self.fetched_at is None:
                raise ValueError("provider source requires provider, place ID and fetched_at")
            if self.content_version is not None:
                raise ValueError("provider source cannot contain a content version")
        elif self.kind is RecallSourceKind.CITY_CONTENT:
            if self.content_version is None or self.provider is not None:
                raise ValueError("city-content source requires only a content version")
        elif self.provider is not None or self.source_place_id is not None:
            raise ValueError("user evidence cannot contain provider identity")
        return self


class RecalledPlace(ImmutableRecallModel):
    place_id: UUID
    city_id: NonEmptyText
    category: PlaceCategory
    name: NonEmptyText
    address: NonEmptyText | None = None
    short_description: ShortText | None = None
    cuisine: ShortText | None = None
    coordinates: Gcj02Coordinates | None = None
    image_url: HttpUrl | None = None
    provider_typecode: NonEmptyText | None = None
    provider_parent_place_id: NonEmptyText | None = None
    rating: float | None = Field(default=None, ge=0, le=5, allow_inf_nan=False)
    average_cost: CnyAmountRange | None = None


class RecalledCandidate(ImmutableRecallModel):
    representative_identity_verified: bool = False
    candidate_id: UUID
    domain: CandidateDomain
    place: RecalledPlace
    channels: tuple[RecallChannel, ...] = Field(min_length=1)
    theme_ids: tuple[NonEmptyText, ...] = ()
    reasons: tuple[ShortText, ...] = Field(min_length=1)
    named_evidence_ids: tuple[UUID, ...] = ()
    sources: tuple[CandidateSourceReference, ...] = Field(min_length=1)
    availability: DataAvailability
    missing_fields: tuple[NonEmptyText, ...] = ()
    missing_reason: ShortText | None = None

    @model_validator(mode="after")
    def availability_and_references_are_consistent(self) -> RecalledCandidate:
        _unique(self.channels, "candidate channels")
        _unique(self.theme_ids, "candidate theme IDs")
        _unique(self.reasons, "candidate reasons")
        _unique(self.named_evidence_ids, "candidate named evidence IDs")
        _unique(self.missing_fields, "candidate missing fields")
        if self.representative_identity_verified and (
            RecallChannel.CITY_LANDMARK not in self.channels
            or not any(source.kind is RecallSourceKind.PROVIDER for source in self.sources)
        ):
            raise ValueError("verified representative requires a named landmark Provider source")
        if self.availability is DataAvailability.MISSING:
            raise ValueError("missing recall attempts must not be emitted as candidates")
        if self.availability is DataAvailability.AVAILABLE:
            if self.place.coordinates is None or self.missing_fields or self.missing_reason:
                raise ValueError("available candidate requires coordinates and no missing data")
        elif not self.missing_fields or self.missing_reason is None:
            raise ValueError("partial candidate requires missing fields and reason")
        expected_category = {
            CandidateDomain.ATTRACTION: PlaceCategory.ATTRACTION,
            CandidateDomain.RESTAURANT: PlaceCategory.RESTAURANT,
            CandidateDomain.HOTEL: PlaceCategory.HOTEL,
        }[self.domain]
        if self.place.category is not expected_category:
            raise ValueError("candidate category must match its recall domain")
        return self


class RecallAttempt(ImmutableRecallModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, json_schema_extra=RECALL_ATTEMPT_SCHEMA_RULE
    )

    query_id: NonEmptyText
    channel: RecallChannel
    domain: CandidateDomain
    status: RecallAttemptStatus
    query_keyword: ShortText | None = None
    exact_name_match: bool = False
    provider: ProviderCode | None = None
    requested_limit: int = Field(ge=1, le=25, strict=True)
    returned_count: int = Field(ge=0, strict=True)
    accepted_count: int = Field(ge=0, strict=True)
    missing_fields: tuple[NonEmptyText, ...] = Field(
        default=(), json_schema_extra={"uniqueItems": True}
    )
    provider_notice: ShortText | None = None
    failure_code: RecallFailureCode | None = None
    failure_reason: ShortText | None = None

    @model_validator(mode="after")
    def status_matches_counts_and_failure(self) -> RecallAttempt:
        if self.accepted_count > self.returned_count:
            raise ValueError("accepted recall count cannot exceed returned count")
        _unique(self.missing_fields, "recall attempt missing fields")
        if self.status is RecallAttemptStatus.FAILED:
            if self.failure_code is None or self.failure_reason is None:
                raise ValueError("failed recall attempt requires safe failure details")
        elif self.failure_code is not None:
            raise ValueError("non-failed recall attempt cannot contain a failure code")
        if self.status is RecallAttemptStatus.PARTIAL and not (
            self.missing_fields or self.provider_notice or self.failure_reason
        ):
            raise ValueError("partial recall attempt requires an explicit reason")
        if self.status is RecallAttemptStatus.EMPTY and self.returned_count != 0:
            raise ValueError("empty recall attempt cannot contain returned results")
        if self.status is RecallAttemptStatus.AVAILABLE and self.accepted_count == 0:
            raise ValueError("available recall attempt requires accepted results")
        return self


class CandidateRecallResult(ImmutableRecallModel):
    result_version: str = Field(pattern=r"^[1-9]\d*\.\d+\.\d+$")
    request_id: UUID
    trip_id: UUID
    semantic_state_version: int = Field(default=0, ge=0, strict=True)
    task_book_id: UUID | None = None
    task_book_revision: int | None = Field(default=None, ge=1, strict=True)
    city_id: NonEmptyText
    status: DataAvailability
    candidates: tuple[RecalledCandidate, ...] = ()
    attempts: tuple[RecallAttempt, ...] = Field(min_length=1)
    provider_call_count: int = Field(ge=0, le=12, strict=True)
    content_package_version: NonEmptyText | None = None
    degradation_reasons: tuple[ShortText, ...] = ()
    generated_at: AwareDatetime

    @model_validator(mode="after")
    def result_status_and_ownership_are_consistent(self) -> CandidateRecallResult:
        _unique([item.candidate_id for item in self.candidates], "candidate IDs")
        _unique(self.degradation_reasons, "recall degradation reasons")
        if any(item.place.city_id != self.city_id for item in self.candidates):
            raise ValueError("recall candidates cannot cross cities")
        if self.status is DataAvailability.MISSING:
            if self.candidates or not self.degradation_reasons:
                raise ValueError("missing recall result requires no candidates and a reason")
        elif not self.candidates:
            raise ValueError("available or partial recall result requires candidates")
        if self.status is DataAvailability.AVAILABLE and self.degradation_reasons:
            raise ValueError("available recall result cannot contain degradation reasons")
        return self


def _unique(values: Sequence[object], label: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must be unique")


V3_CANDIDATE_RECALL_CONTRACTS: tuple[type[ContractModel], ...] = (
    CandidateRecallRequest,
    RecallPlan,
    CandidateRecallResult,
)
