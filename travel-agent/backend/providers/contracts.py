"""Vendor-neutral request, result, and failure contracts for travel providers."""

from __future__ import annotations

import re
from datetime import date, time
from enum import StrEnum
from typing import Any, Generic, Literal, TypeVar
from uuid import UUID
from zoneinfo import ZoneInfo

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, HttpUrl, model_validator

from backend.contracts.common import CnyAmountRange, NonEmptyText, ShortText
from backend.contracts.enums import PlaceCategory, ProviderCode
from backend.contracts.places import Coordinates, Gcj02Coordinates, WeeklyHoursPeriod


class ProviderModel(BaseModel):
    """Strict internal model; raw vendor shapes stop at the adapter boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ProviderResultStatus(StrEnum):
    SUCCESS = "success"
    EMPTY = "empty"
    PARTIAL = "partial"


class ProviderFailureCode(StrEnum):
    INVALID_REQUEST = "invalid_request"
    AUTHENTICATION_FAILED = "authentication_failed"
    PERMISSION_DENIED = "permission_denied"
    RATE_LIMITED = "rate_limited"
    TIMEOUT = "timeout"
    UNAVAILABLE = "unavailable"
    MALFORMED_RESPONSE = "malformed_response"
    UPSTREAM_ERROR = "upstream_error"


class ProviderError(RuntimeError):
    """Safe classified failure which never includes credentials or raw payloads."""

    def __init__(
        self,
        provider: ProviderCode,
        code: ProviderFailureCode,
        operation: str,
        *,
        retryable: bool,
        upstream_code: str | None = None,
        failures: dict[str, ProviderFailureDetail] | None = None,
    ) -> None:
        self.provider = provider
        self.code = code
        self.operation = operation
        self.retryable = retryable
        # Never retain vendor messages/URLs here: only bounded numeric error codes.
        self.upstream_code = (
            upstream_code if upstream_code and re.fullmatch(r"[0-9]{3,5}", upstream_code) else None
        )
        self.failures = failures or {}
        self.attempts = 1
        self.retry_after_seconds: float | None = None
        super().__init__(f"{provider.value} provider {operation} failed: {code.value}")


class ProviderFailureDetail(ProviderModel):
    """Safe per-operation diagnostics, including when sibling operations succeed."""

    code: ProviderFailureCode
    retryable: bool
    upstream_code: str | None = Field(default=None, pattern=r"^[0-9]{3,5}$")
    attempts: int = Field(default=1, ge=1, le=2, strict=True)

    @classmethod
    def from_error(cls, error: ProviderError) -> ProviderFailureDetail:
        return cls(
            code=error.code,
            retryable=error.retryable,
            upstream_code=error.upstream_code,
            attempts=error.attempts,
        )


ResultItem = TypeVar("ResultItem")


class ProviderResponse(ProviderModel, Generic[ResultItem]):
    provider: ProviderCode
    status: ProviderResultStatus
    items: list[ResultItem] = Field(default_factory=list)
    fetched_at: AwareDatetime
    missing_fields: list[NonEmptyText] = Field(default_factory=list)
    source_request_id: NonEmptyText | None = None
    provider_notice: NonEmptyText | None = None
    failures: dict[NonEmptyText, ProviderFailureDetail] = Field(
        default_factory=dict, exclude_if=lambda value: not value
    )

    @model_validator(mode="after")
    def status_matches_items(self) -> ProviderResponse[ResultItem]:
        if len(set(self.missing_fields)) != len(self.missing_fields):
            raise ValueError("provider missing_fields must be unique")
        if self.status is ProviderResultStatus.SUCCESS:
            if not self.items or self.missing_fields or self.failures:
                raise ValueError(
                    "successful provider response requires items and no missing fields"
                )
        elif self.status is ProviderResultStatus.EMPTY:
            if self.items or self.missing_fields or self.failures:
                raise ValueError("empty provider response cannot contain items or missing fields")
        elif not self.items or not self.missing_fields:
            raise ValueError("partial provider response requires items and missing fields")
        for item in self.items:
            item_provider = getattr(item, "provider", self.provider)
            if item_provider != self.provider:
                raise ValueError("provider response items must match the envelope provider")
        return self


class ProviderCityScope(ProviderModel):
    """Application city identity plus the code expected by one provider.

    `city_id` is intentionally not the stage-0 CityCode enum, so adding nationwide city
    registry rows does not require changing provider interfaces.
    """

    city_id: NonEmptyText
    provider_city_code: NonEmptyText
    provider_transit_city_code: NonEmptyText | None = None
    display_name: NonEmptyText | None = None


class KeywordPlaceSearchRequest(ProviderModel):
    city: ProviderCityScope
    query: NonEmptyText
    category_hint: PlaceCategory | None = None
    typecodes: list[NonEmptyText] = Field(default_factory=list)
    page: int = Field(default=1, ge=1, le=100, strict=True)
    page_size: int = Field(default=20, ge=1, le=25, strict=True)


class NearbyPlaceSearchRequest(ProviderModel):
    city: ProviderCityScope
    center: Gcj02Coordinates
    radius_m: int = Field(default=5_000, ge=1, le=50_000, strict=True)
    query: NonEmptyText | None = None
    category_hint: PlaceCategory | None = None
    typecodes: list[NonEmptyText] = Field(default_factory=list)
    page: int = Field(default=1, ge=1, le=100, strict=True)
    page_size: int = Field(default=20, ge=1, le=25, strict=True)

    @model_validator(mode="after")
    def requires_a_filter(self) -> NearbyPlaceSearchRequest:
        if self.query is None and not self.typecodes:
            raise ValueError("nearby search requires a query or at least one typecode")
        return self


def validate_search_polygon(polygon: tuple[Gcj02Coordinates, ...]) -> None:
    points = [(point.longitude, point.latitude) for point in polygon]
    if points[0] != points[-1] or len(set(points[:-1])) != len(points) - 1:
        raise ValueError("polygon must be closed with unique vertices")
    origin_x, origin_y = points[0]
    area = sum(
        (left[0] - origin_x) * (right[1] - origin_y) - (right[0] - origin_x) * (left[1] - origin_y)
        for left, right in zip(points, points[1:], strict=False)
    )
    if abs(area) < 1e-12:
        raise ValueError("polygon must enclose a non-zero area")


class PolygonPlaceSearchRequest(ProviderModel):
    """Attraction or dining corridor recall; the city identifies normalized facts."""

    city: ProviderCityScope
    polygon: tuple[Gcj02Coordinates, ...] = Field(min_length=4)
    query: str | None = Field(default=None, min_length=1, max_length=100)
    category_hint: Literal[PlaceCategory.RESTAURANT, PlaceCategory.ATTRACTION] = (
        PlaceCategory.RESTAURANT
    )
    typecodes: tuple[Literal["050000", "110000"], ...] = Field(
        default=("050000",), min_length=1, max_length=1
    )
    page: int = Field(default=1, ge=1, le=100, strict=True)
    page_size: int = Field(default=20, ge=1, le=25, strict=True)

    @model_validator(mode="after")
    def requires_closed_polygon(self) -> PolygonPlaceSearchRequest:
        expected = "110000" if self.category_hint == PlaceCategory.ATTRACTION else "050000"
        if self.typecodes != (expected,):
            raise ValueError("polygon category and typecodes must match")
        validate_search_polygon(self.polygon)
        return self


class PlaceDetailRequest(ProviderModel):
    city: ProviderCityScope
    source_place_id: NonEmptyText
    category_hint: PlaceCategory | None = None


class GeocodeRequest(ProviderModel):
    city: ProviderCityScope
    address: NonEmptyText


class ProviderPlace(ProviderModel):
    provider: ProviderCode
    source_place_id: NonEmptyText
    city_id: NonEmptyText
    provider_city_code: NonEmptyText | None = None
    name: NonEmptyText
    category: PlaceCategory
    address: NonEmptyText | None = None
    coordinates: Gcj02Coordinates
    entrance_coordinates: Gcj02Coordinates | None = None
    exit_coordinates: Gcj02Coordinates | None = None
    provider_typecode: NonEmptyText | None = None
    provider_parent_place_id: NonEmptyText | None = None
    image_url: HttpUrl | None = None
    rating: float | None = Field(default=None, ge=0, le=5, allow_inf_nan=False)
    average_cost: CnyAmountRange | None = None
    fetched_at: AwareDatetime
    raw_payload: dict[str, Any]


class GeocodedAddress(ProviderModel):
    provider: ProviderCode
    city_id: NonEmptyText
    provider_city_code: NonEmptyText | None = None
    formatted_address: NonEmptyText
    coordinates: Gcj02Coordinates
    match_level: NonEmptyText | None = None
    fetched_at: AwareDatetime
    raw_payload: dict[str, Any]


class HoursRequest(ProviderModel):
    """A canonical place projection suitable for provider-side identity matching."""

    place_id: UUID
    city: ProviderCityScope
    name: NonEmptyText
    address: NonEmptyText | None = None
    coordinates: Gcj02Coordinates
    source_place_ids: dict[ProviderCode, NonEmptyText] = Field(default_factory=dict)
    service_dates: list[date] = Field(default_factory=list, max_length=31)

    @model_validator(mode="after")
    def dates_are_unique(self) -> HoursRequest:
        if len(set(self.service_dates)) != len(self.service_dates):
            raise ValueError("hours service_dates must be unique")
        return self


class HoursDayStatus(StrEnum):
    OPEN = "open"
    CLOSED = "closed"
    UNKNOWN = "unknown"
    CONFLICT = "conflict"


class HoursInterval(ProviderModel):
    opens_at: time
    closes_at: time
    last_entry_at: time | None = None

    @model_validator(mode="after")
    def interval_is_ordered(self) -> HoursInterval:
        if self.opens_at.tzinfo or self.closes_at.tzinfo:
            raise ValueError("hours must use destination local wall time")
        if self.closes_at <= self.opens_at:
            raise ValueError("overnight hours require separate date evidence")
        if self.last_entry_at is not None and (
            self.last_entry_at.tzinfo or not self.opens_at <= self.last_entry_at <= self.closes_at
        ):
            raise ValueError("last entry must be within the opening interval")
        return self


class ProviderDateHours(ProviderModel):
    service_date: date
    status: HoursDayStatus
    intervals: list[HoursInterval] = Field(default_factory=list, max_length=10)
    basis: Literal["weekly", "dated_exception", "today", "unverified", "conflicting"]
    reason: NonEmptyText

    @model_validator(mode="after")
    def status_matches_intervals(self) -> ProviderDateHours:
        if (self.status is HoursDayStatus.OPEN) != bool(self.intervals):
            raise ValueError("only verified open days have usable intervals")
        ordered = sorted(self.intervals, key=lambda value: value.opens_at)
        if any(
            left.closes_at > right.opens_at
            for left, right in zip(ordered, ordered[1:], strict=False)
        ):
            raise ValueError("opening intervals must not overlap")
        return self


class ProviderRegularHours(ProviderModel):
    provider: ProviderCode
    source_place_id: NonEmptyText
    display_text: NonEmptyText | None = None
    weekly_periods: list[WeeklyHoursPeriod] = Field(default_factory=list)
    weekly_text: NonEmptyText | None = None
    today_text: NonEmptyText | None = None
    today_date: date | None = None
    date_hours: list[ProviderDateHours] = Field(default_factory=list, max_length=31)
    fetched_at: AwareDatetime
    missing_reason: ShortText | None = None
    raw_payload: dict[str, Any]

    @model_validator(mode="after")
    def hours_or_missing_reason(self) -> ProviderRegularHours:
        if (self.today_text is None) != (self.today_date is None):
            raise ValueError("today hours require their observed local date")
        if (
            self.today_date is not None
            and self.today_date != self.fetched_at.astimezone(ZoneInfo("Asia/Shanghai")).date()
        ):
            raise ValueError("today hours must match the fetched destination date")
        if len({item.service_date for item in self.date_hours}) != len(self.date_hours):
            raise ValueError("date hours must have unique dates")
        has_hours = (
            self.display_text is not None
            or bool(self.weekly_periods)
            or self.weekly_text is not None
            or self.today_text is not None
        )
        if self.missing_reason is not None:
            if has_hours:
                raise ValueError("missing provider hours cannot also contain hours")
            if any(
                day.status in (HoursDayStatus.OPEN, HoursDayStatus.CLOSED)
                for day in self.date_hours
            ):
                raise ValueError("missing provider hours cannot claim known date hours")
        elif not has_hours:
            raise ValueError("provider hours require a value or a missing reason")
        return self


class RouteMode(StrEnum):
    WALKING = "walking"
    CYCLING = "cycling"
    TRANSIT = "transit"
    DRIVING = "driving"


class RouteRequest(ProviderModel):
    city: ProviderCityScope
    origin: Gcj02Coordinates
    destination: Gcj02Coordinates
    modes: list[RouteMode]

    @model_validator(mode="after")
    def modes_are_unique(self) -> RouteRequest:
        if not self.modes or len(set(self.modes)) != len(self.modes):
            raise ValueError("route modes must be non-empty and unique")
        if RouteMode.TRANSIT in self.modes and self.city.provider_transit_city_code is None:
            raise ValueError("transit routes require a provider transit city code")
        return self


class ProviderRoute(ProviderModel):
    provider: ProviderCode
    mode: RouteMode
    source_route_index: int = Field(ge=0, strict=True)
    distance_m: int = Field(ge=0, strict=True)
    duration_seconds: int = Field(ge=0, strict=True)
    walking_distance_m: int | None = Field(default=None, ge=0, strict=True)
    transfer_count: int | None = Field(default=None, ge=0, strict=True)
    fare: CnyAmountRange | None = None
    polyline: list[Gcj02Coordinates] = Field(default_factory=list)
    fetched_at: AwareDatetime
    raw_payload: dict[str, Any]


class HotelSearchRequest(ProviderModel):
    hotel_stars: list[Literal[1, 2, 3, 4, 5]] = Field(default_factory=list, max_length=5)
    hotel_types: list[Literal["酒店", "民宿", "客栈"]] = Field(default_factory=list, max_length=3)
    sort: Literal["distance_asc", "rate_desc", "price_asc", "price_desc", "no_rank"] | None = None
    city: ProviderCityScope
    check_in: date
    check_out: date
    anchor: Gcj02Coordinates | None = None
    anchor_name: NonEmptyText | None = None
    query: NonEmptyText | None = None

    @model_validator(mode="after")
    def dates_are_ordered(self) -> HotelSearchRequest:
        if self.check_out <= self.check_in:
            raise ValueError("hotel check_out must be after check_in")
        return self


class ProductSearchRequest(ProviderModel):
    city: ProviderCityScope
    visit_date: date | None = None
    source_place_id: NonEmptyText | None = None
    query: NonEmptyText | None = None

    @model_validator(mode="after")
    def identifies_a_product_target(self) -> ProductSearchRequest:
        if self.source_place_id is None and self.query is None:
            raise ValueError("product search requires a source place or query")
        return self


class ProviderHotelOffer(ProviderModel):
    provider: ProviderCode
    source_offer_id: NonEmptyText
    source_hotel_id: NonEmptyText
    name: NonEmptyText
    hotel_type: NonEmptyText | None = None
    brand_name: NonEmptyText | None = None
    address: NonEmptyText | None = None
    raw_coordinates: Coordinates | None = None
    room_price: CnyAmountRange | None = None
    rating: float | None = Field(default=None, ge=0, le=5, allow_inf_nan=False)
    image_urls: list[HttpUrl] = Field(default_factory=list)
    detail_url: HttpUrl | None = None
    missing_fields: list[NonEmptyText] = Field(default_factory=list)
    fetched_at: AwareDatetime
    raw_payload: dict[str, Any]

    @model_validator(mode="after")
    def missing_fields_are_unique(self) -> ProviderHotelOffer:
        if len(set(self.missing_fields)) != len(self.missing_fields):
            raise ValueError("hotel offer missing_fields must be unique")
        return self


class ProviderTicketOffer(ProviderModel):
    provider: ProviderCode
    source_offer_id: NonEmptyText
    source_place_id: NonEmptyText
    name: NonEmptyText
    place_name: NonEmptyText | None = None
    image_url: HttpUrl | None = None
    price_date: date | None = None
    price: CnyAmountRange | None = None
    admission_status: Literal["free", "paid", "unknown"] = "unknown"
    detail_url: HttpUrl | None = None
    missing_fields: list[NonEmptyText] = Field(default_factory=list)
    fetched_at: AwareDatetime
    raw_payload: dict[str, Any]

    @model_validator(mode="after")
    def missing_fields_are_unique(self) -> ProviderTicketOffer:
        if len(set(self.missing_fields)) != len(self.missing_fields):
            raise ValueError("ticket offer missing_fields must be unique")
        return self


class WeatherRequest(ProviderModel):
    city: ProviderCityScope
    start_date: date
    end_date: date

    @model_validator(mode="after")
    def dates_are_ordered(self) -> WeatherRequest:
        if self.end_date < self.start_date:
            raise ValueError("weather end_date must not precede start_date")
        if (self.end_date - self.start_date).days > 4:
            raise ValueError("weather request must cover between one and five days")
        return self


class ProviderForecastDay(ProviderModel):
    provider: ProviderCode
    forecast_date: date
    condition_day: NonEmptyText | None = None
    condition_night: NonEmptyText | None = None
    low_celsius: int | None = None
    high_celsius: int | None = None
    source_name: NonEmptyText | None = None
    forecast_kind: Literal["forecast", "outlook"] = "forecast"
    fetched_at: AwareDatetime
    raw_payload: dict[str, Any]

    @model_validator(mode="after")
    def contains_at_least_one_forecast_value(self) -> ProviderForecastDay:
        if all(
            value is None
            for value in (
                self.condition_day,
                self.condition_night,
                self.low_celsius,
                self.high_celsius,
            )
        ):
            raise ValueError("forecast day requires at least one weather value")
        if (
            self.low_celsius is not None
            and self.high_celsius is not None
            and self.high_celsius < self.low_celsius
        ):
            raise ValueError("forecast high temperature cannot be below low temperature")
        return self


class ProviderTimeWindow(ProviderModel):
    start: time
    end: time
