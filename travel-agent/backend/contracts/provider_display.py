"""Provider-neutral display projection for the V2 conversation workspace."""

from __future__ import annotations

from datetime import date
from typing import Any, Literal
from uuid import UUID

from pydantic import AwareDatetime, ConfigDict, Field, model_validator

from backend.contracts.base import ContractModel
from backend.contracts.common import CnyAmountRange, NonEmptyText, ShortText
from backend.contracts.enums import (
    DataAvailability,
    PlaceCategory,
    ProviderCode,
    TransportMode,
)
from backend.contracts.places import Gcj02Coordinates

PROVIDER_DISPLAY_SCHEMA_RULE: dict[str, Any] = {
    "x-travel-provider-display": {
        "sourcesField": "sources",
        "placesField": "places",
        "routesField": "routes",
        "factsField": "facts",
    }
}


class ProviderDisplaySource(ContractModel):
    fact_id: NonEmptyText
    provider: ProviderCode
    source_record_id: NonEmptyText
    fetched_at: AwareDatetime


class ProviderDisplayPlace(ContractModel):
    place_id: UUID
    name: NonEmptyText
    category: PlaceCategory
    address: NonEmptyText | None = None
    coordinates: Gcj02Coordinates
    availability: DataAvailability
    source_fact_ids: list[NonEmptyText] = Field(
        default_factory=list,
        json_schema_extra={"uniqueItems": True},
    )
    missing_reason: ShortText | None = None

    @model_validator(mode="after")
    def availability_is_explicit(self) -> ProviderDisplayPlace:
        if len(set(self.source_fact_ids)) != len(self.source_fact_ids):
            raise ValueError("source_fact_ids must not contain duplicates")
        if self.availability is DataAvailability.AVAILABLE:
            if not self.source_fact_ids or self.missing_reason is not None:
                raise ValueError("available places require sources and no missing reason")
        elif self.missing_reason is None:
            raise ValueError("partial or missing places require a reason")
        return self


class ProviderDisplayRoute(ContractModel):
    route_id: NonEmptyText
    from_place_id: UUID
    to_place_id: UUID
    mode: TransportMode
    availability: DataAvailability
    distance_m: int | None = Field(default=None, ge=0, strict=True)
    duration_minutes: int | None = Field(default=None, ge=0, strict=True)
    walking_m: int | None = Field(default=None, ge=0, strict=True)
    fare: CnyAmountRange | None = None
    polyline: list[Gcj02Coordinates] = Field(default_factory=list)
    source_fact_ids: list[NonEmptyText] = Field(
        default_factory=list,
        json_schema_extra={"uniqueItems": True},
    )
    missing_reason: ShortText | None = None

    @model_validator(mode="after")
    def route_is_traceable(self) -> ProviderDisplayRoute:
        if self.from_place_id == self.to_place_id:
            raise ValueError("route endpoints must differ")
        if len(set(self.source_fact_ids)) != len(self.source_fact_ids):
            raise ValueError("source_fact_ids must not contain duplicates")
        if self.polyline and len(self.polyline) < 2:
            raise ValueError("route polyline requires at least two points")
        if self.availability is DataAvailability.MISSING:
            if (
                any(
                    value is not None
                    for value in (self.distance_m, self.duration_minutes, self.walking_m, self.fare)
                )
                or self.polyline
            ):
                raise ValueError("missing routes cannot contain route facts")
            if self.missing_reason is None:
                raise ValueError("missing routes require a reason")
        elif self.distance_m is None or self.duration_minutes is None or not self.source_fact_ids:
            raise ValueError("available or partial routes require distance, duration, and source")
        elif self.availability is DataAvailability.PARTIAL and self.missing_reason is None:
            raise ValueError("partial routes require a missing-data reason")
        elif self.availability is DataAvailability.AVAILABLE and self.missing_reason is not None:
            raise ValueError("available routes cannot declare a missing reason")
        return self


class ProviderDisplayFact(ContractModel):
    fact_id: NonEmptyText
    kind: Literal["regular_hours", "ticket_price", "hotel_price", "weather"]
    label: NonEmptyText
    availability: DataAvailability
    provider: ProviderCode | None = None
    fetched_at: AwareDatetime | None = None
    place_id: UUID | None = None
    forecast_date: date | None = None
    display_text: NonEmptyText | None = None
    amount: CnyAmountRange | None = None
    minimum_celsius: float | None = Field(default=None, ge=-90, le=70, allow_inf_nan=False)
    maximum_celsius: float | None = Field(default=None, ge=-90, le=70, allow_inf_nan=False)
    source_fact_ids: list[NonEmptyText] = Field(
        default_factory=list,
        json_schema_extra={"uniqueItems": True},
    )
    missing_reason: ShortText | None = None

    @model_validator(mode="after")
    def value_matches_kind_and_availability(self) -> ProviderDisplayFact:
        if len(set(self.source_fact_ids)) != len(self.source_fact_ids):
            raise ValueError("source_fact_ids must not contain duplicates")
        has_value = any(
            value is not None
            for value in (
                self.display_text,
                self.amount,
                self.minimum_celsius,
                self.maximum_celsius,
            )
        )
        if self.availability is DataAvailability.MISSING:
            if has_value or self.missing_reason is None:
                raise ValueError("missing facts require no value and an explicit reason")
            has_any_trace = (
                bool(self.source_fact_ids)
                or self.provider is not None
                or self.fetched_at is not None
            )
            has_complete_trace = (
                bool(self.source_fact_ids)
                and self.provider is not None
                and self.fetched_at is not None
            )
            if has_any_trace and not has_complete_trace:
                raise ValueError("missing fact retrieval trace must be complete when present")
            return self
        if (
            not has_value
            or not self.source_fact_ids
            or self.provider is None
            or self.fetched_at is None
        ):
            raise ValueError("available or partial facts require value, source, and retrieval time")
        if self.availability is DataAvailability.PARTIAL and self.missing_reason is None:
            raise ValueError("partial facts require a missing-data reason")
        if self.availability is DataAvailability.AVAILABLE and self.missing_reason is not None:
            raise ValueError("available facts cannot declare a missing reason")
        if self.kind == "weather":
            if self.forecast_date is None:
                raise ValueError("weather facts require a forecast date")
            if (
                self.minimum_celsius is not None
                and self.maximum_celsius is not None
                and self.maximum_celsius < self.minimum_celsius
            ):
                raise ValueError("weather facts require an ordered temperature range")
            if self.availability is DataAvailability.AVAILABLE and (
                self.minimum_celsius is None or self.maximum_celsius is None
            ):
                raise ValueError("available weather facts require a temperature range")
        elif self.forecast_date is not None:
            raise ValueError("only weather facts may contain forecast_date")
        return self


class ProviderDisplayProjection(ContractModel):
    model_config = ConfigDict(json_schema_extra=PROVIDER_DISPLAY_SCHEMA_RULE)
    sources: list[ProviderDisplaySource] = Field(default_factory=list)
    places: list[ProviderDisplayPlace] = Field(default_factory=list)
    routes: list[ProviderDisplayRoute] = Field(default_factory=list)
    facts: list[ProviderDisplayFact] = Field(default_factory=list)

    @model_validator(mode="after")
    def references_are_closed(self) -> ProviderDisplayProjection:
        source_ids = [source.fact_id for source in self.sources]
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("provider display source fact_id must be unique")
        place_ids = [place.place_id for place in self.places]
        if len(set(place_ids)) != len(place_ids):
            raise ValueError("provider display place_id must be unique")
        route_ids = [route.route_id for route in self.routes]
        if len(set(route_ids)) != len(route_ids):
            raise ValueError("provider display route_id must be unique")
        fact_ids = [fact.fact_id for fact in self.facts]
        if len(set(fact_ids)) != len(fact_ids):
            raise ValueError("provider display fact_id must be unique")
        declared_sources = set(source_ids)
        referenced_sources: set[str] = set()
        for place in self.places:
            referenced_sources.update(place.source_fact_ids)
        for route in self.routes:
            referenced_sources.update(route.source_fact_ids)
        for fact in self.facts:
            referenced_sources.update(fact.source_fact_ids)
        if not referenced_sources <= declared_sources:
            raise ValueError("provider display values may only reference declared sources")
        declared_places = set(place_ids)
        for route in self.routes:
            if (
                route.from_place_id not in declared_places
                or route.to_place_id not in declared_places
            ):
                raise ValueError("provider display routes must reference declared places")
        if any(
            fact.place_id is not None and fact.place_id not in declared_places
            for fact in self.facts
        ):
            raise ValueError("provider display facts must reference declared places")
        return self


V2_PROVIDER_DISPLAY_CONTRACTS: tuple[type[ContractModel], ...] = (ProviderDisplayProjection,)
