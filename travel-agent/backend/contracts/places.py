"""P0-06 canonical place, evidence, hours, and travel product contracts."""

from __future__ import annotations

from datetime import date, time
from typing import Any, Literal
from uuid import UUID

from pydantic import AwareDatetime, ConfigDict, Field, HttpUrl, JsonValue, model_validator

from backend.contracts.base import ContractModel
from backend.contracts.common import CnyAmountRange, NonEmptyText, ShortText
from backend.contracts.enums import (
    CityCode,
    CoordinateSystem,
    DataAvailability,
    EvidenceStatus,
    PlaceCategory,
    PlaceFactKind,
    ProviderCode,
)

AVAILABILITY_SCHEMA_RULE: dict[str, Any] = {
    "allOf": [
        {
            "if": {
                "properties": {"availability": {"const": "missing"}},
                "required": ["availability"],
            },
            "then": {
                "properties": {
                    "value": {"type": "null"},
                    "missing_reason": {
                        "minLength": 1,
                        "pattern": r"\S",
                        "type": "string",
                    },
                },
                "required": ["missing_reason"],
            },
            "else": {"not": {"properties": {"value": {"type": "null"}}, "required": ["value"]}},
        }
    ]
}

HOTEL_OFFER_AVAILABILITY_SCHEMA_RULE: dict[str, Any] = {
    "allOf": [
        {
            "if": {
                "properties": {"availability": {"const": "available"}},
                "required": ["availability"],
            },
            "then": {
                "properties": {
                    "room_price": {"not": {"type": "null"}},
                    "missing_fields": {"type": "array", "maxItems": 0},
                    "missing_reason": {"type": "null"},
                },
                "required": ["room_price"],
            },
        },
        {
            "if": {
                "properties": {"availability": {"const": "partial"}},
                "required": ["availability"],
            },
            "then": {
                "anyOf": [
                    {
                        "properties": {"room_price": {"not": {"type": "null"}}},
                        "required": ["room_price"],
                    },
                    {
                        "properties": {"rating": {"not": {"type": "null"}}},
                        "required": ["rating"],
                    },
                    {
                        "properties": {"image_urls": {"type": "array", "minItems": 1}},
                        "required": ["image_urls"],
                    },
                    {
                        "properties": {"detail_url": {"not": {"type": "null"}}},
                        "required": ["detail_url"],
                    },
                ],
                "properties": {
                    "missing_fields": {"type": "array", "minItems": 1},
                    "missing_reason": {
                        "minLength": 1,
                        "pattern": r"\S",
                        "type": "string",
                    },
                },
                "required": ["missing_fields", "missing_reason"],
            },
        },
        {
            "if": {
                "properties": {"availability": {"const": "missing"}},
                "required": ["availability"],
            },
            "then": {
                "properties": {
                    "room_price": {"type": "null"},
                    "missing_reason": {
                        "minLength": 1,
                        "pattern": r"\S",
                        "type": "string",
                    },
                },
                "required": ["missing_reason"],
            },
        },
        {
            "if": {
                "properties": {
                    "missing_fields": {
                        "type": "array",
                        "contains": {"const": "room_price"},
                    }
                },
                "required": ["missing_fields"],
            },
            "then": {"properties": {"room_price": {"type": "null"}}},
        },
        {
            "if": {
                "properties": {
                    "missing_fields": {"type": "array", "contains": {"const": "rating"}}
                },
                "required": ["missing_fields"],
            },
            "then": {"properties": {"rating": {"type": "null"}}},
        },
        {
            "if": {
                "properties": {
                    "missing_fields": {
                        "type": "array",
                        "contains": {"const": "image_urls"},
                    }
                },
                "required": ["missing_fields"],
            },
            "then": {"properties": {"image_urls": {"type": "array", "maxItems": 0}}},
        },
        {
            "if": {
                "properties": {
                    "missing_fields": {
                        "type": "array",
                        "contains": {"const": "detail_url"},
                    }
                },
                "required": ["missing_fields"],
            },
            "then": {"properties": {"detail_url": {"type": "null"}}},
        },
    ]
}

TICKET_OFFER_AVAILABILITY_SCHEMA_RULE: dict[str, Any] = {
    "allOf": [
        {
            "if": {
                "properties": {"availability": {"const": "available"}},
                "required": ["availability"],
            },
            "then": {
                "properties": {
                    "price": {"not": {"type": "null"}},
                    "missing_fields": {"type": "array", "maxItems": 0},
                    "missing_reason": {"type": "null"},
                },
                "required": ["price"],
            },
        },
        {
            "if": {
                "properties": {"availability": {"const": "partial"}},
                "required": ["availability"],
            },
            "then": {
                "anyOf": [
                    {
                        "properties": {"price": {"not": {"type": "null"}}},
                        "required": ["price"],
                    },
                    {
                        "properties": {"detail_url": {"not": {"type": "null"}}},
                        "required": ["detail_url"],
                    },
                ],
                "properties": {
                    "missing_fields": {"type": "array", "minItems": 1},
                    "missing_reason": {
                        "minLength": 1,
                        "pattern": r"\S",
                        "type": "string",
                    },
                },
                "required": ["missing_fields", "missing_reason"],
            },
        },
        {
            "if": {
                "properties": {"availability": {"const": "missing"}},
                "required": ["availability"],
            },
            "then": {
                "properties": {
                    "price": {"type": "null"},
                    "missing_reason": {
                        "minLength": 1,
                        "pattern": r"\S",
                        "type": "string",
                    },
                },
                "required": ["missing_reason"],
            },
        },
        {
            "if": {
                "properties": {"missing_fields": {"type": "array", "contains": {"const": "price"}}},
                "required": ["missing_fields"],
            },
            "then": {"properties": {"price": {"type": "null"}}},
        },
        {
            "if": {
                "properties": {
                    "missing_fields": {
                        "type": "array",
                        "contains": {"const": "detail_url"},
                    }
                },
                "required": ["missing_fields"],
            },
            "then": {"properties": {"detail_url": {"type": "null"}}},
        },
    ]
}


class Coordinates(ContractModel):
    latitude: float = Field(ge=-90, le=90, allow_inf_nan=False)
    longitude: float = Field(ge=-180, le=180, allow_inf_nan=False)
    coord_system: CoordinateSystem


class Gcj02Coordinates(Coordinates):
    """Canonical coordinates used by every internal planning contract."""

    coord_system: Literal[CoordinateSystem.GCJ_02]


class PlaceIdentityEvidence(ContractModel):
    """Evidence that may support a source record merge.

    Name similarity is useful but never sufficient. Address agreement or an
    already-evaluated coordinate-proximity signal is mandatory.
    """

    city_match: bool
    name_similarity: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    normalized_address_match: bool = False
    coordinate_proximity_match: bool = False
    coordinate_distance_m: int | None = Field(default=None, ge=0, strict=True)

    @model_validator(mode="after")
    def requires_non_name_identity_evidence(self) -> PlaceIdentityEvidence:
        if not self.city_match:
            raise ValueError("places in different cities cannot be merged")
        if not (self.normalized_address_match or self.coordinate_proximity_match):
            raise ValueError("name alone is not sufficient to merge places")
        if self.coordinate_proximity_match and self.coordinate_distance_m is None:
            raise ValueError("coordinate proximity requires a measured distance")
        return self


class PlaceSourceMapping(ContractModel):
    provider: ProviderCode
    source_place_id: NonEmptyText
    raw_name: NonEmptyText
    raw_address: NonEmptyText | None = None
    raw_coordinates: Coordinates | None = None
    fetched_at: AwareDatetime


class PlaceMatchProposal(ContractModel):
    existing_place_id: UUID
    incoming: PlaceSourceMapping
    evidence: PlaceIdentityEvidence


class CanonicalPlace(ContractModel):
    place_id: UUID
    city: CityCode
    category: PlaceCategory
    name: NonEmptyText
    address: NonEmptyText | None = None
    coordinates: Gcj02Coordinates
    source_mappings: list[PlaceSourceMapping] = Field(default_factory=list)

    @model_validator(mode="after")
    def canonical_coordinates_and_mappings_are_valid(self) -> CanonicalPlace:
        identities = [
            (mapping.provider, mapping.source_place_id) for mapping in self.source_mappings
        ]
        if len(set(identities)) != len(identities):
            raise ValueError("provider source mappings must be unique")
        return self


class PlaceFact(ContractModel):
    model_config = ConfigDict(json_schema_extra=AVAILABILITY_SCHEMA_RULE)

    fact_id: NonEmptyText
    place_id: UUID
    kind: PlaceFactKind
    provider: ProviderCode
    source_record_id: NonEmptyText
    availability: DataAvailability
    value: JsonValue | None = None
    raw_value: JsonValue | None = None
    evidence_status: EvidenceStatus
    fetched_at: AwareDatetime
    missing_reason: ShortText | None = None

    @model_validator(mode="after")
    def availability_matches_value(self) -> PlaceFact:
        if self.availability is DataAvailability.MISSING:
            if self.value is not None or self.missing_reason is None:
                raise ValueError("missing fact requires no value and an explicit reason")
        elif self.value is None:
            raise ValueError("available or partial fact requires a value")
        return self


class WeeklyHoursPeriod(ContractModel):
    weekday: int = Field(ge=1, le=7, strict=True)
    opens_at: time
    closes_at: time


class RegularHours(ContractModel):
    place_id: UUID
    provider: ProviderCode
    availability: DataAvailability
    display_text: NonEmptyText | None = None
    weekly_periods: list[WeeklyHoursPeriod] = Field(default_factory=list)
    source_record_id: NonEmptyText
    fetched_at: AwareDatetime
    missing_reason: ShortText | None = None

    @model_validator(mode="after")
    def availability_matches_hours(self) -> RegularHours:
        has_hours = self.display_text is not None or bool(self.weekly_periods)
        if self.availability is DataAvailability.MISSING:
            if has_hours or self.missing_reason is None:
                raise ValueError("missing hours require no hours value and an explicit reason")
        elif not has_hours:
            raise ValueError("available or partial hours require a display value or periods")
        return self


class HotelOffer(ContractModel):
    model_config = ConfigDict(json_schema_extra=HOTEL_OFFER_AVAILABILITY_SCHEMA_RULE)

    offer_id: NonEmptyText
    hotel_place_id: UUID
    provider: ProviderCode
    source_offer_id: NonEmptyText
    check_in: date
    check_out: date
    availability: DataAvailability
    room_price: CnyAmountRange | None = None
    rating: float | None = Field(default=None, ge=0, le=5, allow_inf_nan=False)
    image_urls: list[HttpUrl] = Field(default_factory=list)
    detail_url: HttpUrl | None = None
    missing_fields: list[Literal["room_price", "rating", "image_urls", "detail_url"]] = Field(
        default_factory=list,
        json_schema_extra={"uniqueItems": True},
    )
    missing_reason: ShortText | None = None
    raw_price: JsonValue | None = None
    fetched_at: AwareDatetime

    @model_validator(mode="after")
    def dates_and_availability_are_valid(self) -> HotelOffer:
        if self.check_out <= self.check_in:
            raise ValueError("hotel check_out must be after check_in")
        if len(set(self.missing_fields)) != len(self.missing_fields):
            raise ValueError("missing_fields must not contain duplicates")
        present_by_field = {
            "room_price": self.room_price is not None,
            "rating": self.rating is not None,
            "image_urls": bool(self.image_urls),
            "detail_url": self.detail_url is not None,
        }
        for field_name in self.missing_fields:
            if present_by_field.get(field_name, False):
                raise ValueError(f"{field_name} cannot be both present and declared missing")
        if self.availability is DataAvailability.AVAILABLE:
            if self.room_price is None:
                raise ValueError("available hotel offer requires room_price")
            if self.missing_fields or self.missing_reason is not None:
                raise ValueError("available hotel offer cannot declare missing data")
        elif self.availability is DataAvailability.PARTIAL:
            has_data = any(
                value is not None and value != []
                for value in (self.room_price, self.rating, self.image_urls, self.detail_url)
            )
            if not has_data or not self.missing_fields or self.missing_reason is None:
                raise ValueError(
                    "partial hotel offer requires returned data and "
                    "an explicit missing-data description"
                )
        elif self.room_price is not None or self.missing_reason is None:
            raise ValueError("missing hotel offer requires no price and an explicit reason")
        return self


class TicketOffer(ContractModel):
    model_config = ConfigDict(json_schema_extra=TICKET_OFFER_AVAILABILITY_SCHEMA_RULE)

    offer_id: NonEmptyText
    place_id: UUID
    provider: ProviderCode
    source_offer_id: NonEmptyText
    visit_date: date | None = None
    availability: DataAvailability
    price: CnyAmountRange | None = None
    detail_url: HttpUrl | None = None
    missing_fields: list[Literal["price", "detail_url"]] = Field(
        default_factory=list,
        json_schema_extra={"uniqueItems": True},
    )
    missing_reason: ShortText | None = None
    raw_price: JsonValue | None = None
    fetched_at: AwareDatetime

    @model_validator(mode="after")
    def availability_is_valid(self) -> TicketOffer:
        if len(set(self.missing_fields)) != len(self.missing_fields):
            raise ValueError("missing_fields must not contain duplicates")
        values_by_field = {"price": self.price, "detail_url": self.detail_url}
        for field_name in self.missing_fields:
            if field_name in values_by_field and values_by_field[field_name] is not None:
                raise ValueError(f"{field_name} cannot be both present and declared missing")
        if self.availability is DataAvailability.AVAILABLE:
            if self.price is None:
                raise ValueError("available ticket offer requires price")
            if self.missing_fields or self.missing_reason is not None:
                raise ValueError("available ticket offer cannot declare missing data")
        elif self.availability is DataAvailability.PARTIAL:
            if (
                self.price is None
                and self.detail_url is None
                or not self.missing_fields
                or self.missing_reason is None
            ):
                raise ValueError(
                    "partial ticket offer requires returned data and "
                    "an explicit missing-data description"
                )
        elif self.price is not None or self.missing_reason is None:
            raise ValueError("missing ticket offer requires no price and an explicit reason")
        return self


P0_PLACE_CONTRACTS: tuple[type[ContractModel], ...] = (
    CanonicalPlace,
    PlaceMatchProposal,
    PlaceFact,
    RegularHours,
    HotelOffer,
    TicketOffer,
)
