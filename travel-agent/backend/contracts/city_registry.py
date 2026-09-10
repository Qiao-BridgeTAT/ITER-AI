"""Versioned, provider-neutral city registry contracts."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from datetime import date
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Literal, TypeVar
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from backend.contracts.common import NonEmptyText
from backend.contracts.enums import CityCode, CoordinateSystem

SEMANTIC_VERSION_PATTERN = r"^[1-9]\d*\.\d+\.\d+$"
CITY_ID_PATTERN = r"^[a-z]{2}-[0-9]{6}$"
ADMINISTRATIVE_CODE_PATTERN = r"^[0-9]{6}$"
RegistryEntry = TypeVar("RegistryEntry")


def normalize_city_reference(value: str) -> str:
    """Normalize human and provider city references without erasing geography."""

    normalized = unicodedata.normalize("NFKC", value).strip().casefold()
    return re.sub(r"[\s·•._-]+", "", normalized)


class CityRegistryModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CityContentReviewStatus(StrEnum):
    UNREVIEWED = "unreviewed"
    AI_PRE_REVIEWED = "ai_pre_reviewed"
    USER_ACCEPTED = "user_accepted"


class CityContentCapability(StrEnum):
    CITY_BRIEF = "city_brief"
    ATTRACTION_SEEDS = "attraction_seeds"


class CityProviderCapability(StrEnum):
    AMAP_PLACES = "amap_places"
    AMAP_ROUTES = "amap_routes"
    AMAP_HOURS = "amap_hours"
    BAIDU_HOURS = "baidu_hours"
    FLYAI_PRODUCTS = "flyai_products"
    WEATHER_FORECAST = "weather_forecast"


class CityAdministrativeCodes(CityRegistryModel):
    country: str = Field(pattern=r"^[A-Z]{2}$")
    province: str = Field(pattern=ADMINISTRATIVE_CODE_PATTERN)
    prefecture: str = Field(pattern=ADMINISTRATIVE_CODE_PATTERN)


class CityProviderCodes(CityRegistryModel):
    amap: NonEmptyText | None = None
    amap_transit: NonEmptyText | None = None
    baidu: NonEmptyText | None = None
    flyai: NonEmptyText | None = None
    weather: NonEmptyText | None = None

    @model_validator(mode="after")
    def transit_requires_amap(self) -> CityProviderCodes:
        if self.amap_transit is not None and self.amap is None:
            raise ValueError("amap_transit requires an amap city code")
        if all(code is None for code in (self.amap, self.baidu, self.flyai, self.weather)):
            raise ValueError("a city requires at least one provider city code")
        return self


class CityContentRegistration(CityRegistryModel):
    content_version: str = Field(pattern=SEMANTIC_VERSION_PATTERN)
    relative_path: NonEmptyText
    review_status: CityContentReviewStatus
    capabilities: tuple[CityContentCapability, ...] = Field(min_length=1)

    @field_validator("relative_path")
    @classmethod
    def relative_path_stays_inside_project(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or path.suffix != ".json":
            raise ValueError("content package path must be a project-relative JSON path")
        return value

    @field_validator("capabilities")
    @classmethod
    def capabilities_are_unique(
        cls, value: tuple[CityContentCapability, ...]
    ) -> tuple[CityContentCapability, ...]:
        if len(set(value)) != len(value):
            raise ValueError("content capabilities must be unique")
        return value


class CityRegistration(CityRegistryModel):
    city_id: str = Field(pattern=CITY_ID_PATTERN)
    display_name: NonEmptyText
    aliases: tuple[NonEmptyText, ...] = Field(min_length=1)
    country_name: NonEmptyText
    province_name: NonEmptyText
    administrative_codes: CityAdministrativeCodes
    timezone: NonEmptyText
    coordinate_system: CoordinateSystem
    provider_codes: CityProviderCodes
    enabled_provider_capabilities: tuple[CityProviderCapability, ...] = Field(min_length=1)
    content_package: CityContentRegistration | None = None
    legacy_city_code: CityCode | None = None

    @field_validator("aliases")
    @classmethod
    def aliases_are_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = [normalize_city_reference(alias) for alias in value]
        if len(set(normalized)) != len(normalized):
            raise ValueError("city aliases must be unique after normalization")
        return value

    @field_validator("timezone")
    @classmethod
    def timezone_is_valid(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as exc:
            raise ValueError("timezone must be a valid IANA timezone") from exc
        return value

    @field_validator("coordinate_system")
    @classmethod
    def internal_coordinates_are_gcj02(cls, value: CoordinateSystem) -> CoordinateSystem:
        if value is not CoordinateSystem.GCJ_02:
            raise ValueError("registered city coordinates must use gcj_02")
        return value

    @field_validator("enabled_provider_capabilities")
    @classmethod
    def provider_capabilities_are_unique(
        cls, value: tuple[CityProviderCapability, ...]
    ) -> tuple[CityProviderCapability, ...]:
        if len(set(value)) != len(value):
            raise ValueError("provider capabilities must be unique")
        return value

    @model_validator(mode="after")
    def capabilities_have_provider_codes(self) -> CityRegistration:
        requirements = {
            CityProviderCapability.AMAP_PLACES: self.provider_codes.amap,
            CityProviderCapability.AMAP_ROUTES: self.provider_codes.amap_transit,
            CityProviderCapability.AMAP_HOURS: self.provider_codes.amap,
            CityProviderCapability.BAIDU_HOURS: self.provider_codes.baidu,
            CityProviderCapability.FLYAI_PRODUCTS: self.provider_codes.flyai,
            CityProviderCapability.WEATHER_FORECAST: self.provider_codes.weather,
        }
        missing = [
            capability.value
            for capability in self.enabled_provider_capabilities
            if requirements[capability] is None
        ]
        if missing:
            raise ValueError("provider capabilities require matching codes: " + ", ".join(missing))
        return self


class CityRegistryCatalog(CityRegistryModel):
    registry_version: str = Field(pattern=SEMANTIC_VERSION_PATTERN)
    cities: tuple[CityRegistration, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def registry_keys_are_consistent(self) -> CityRegistryCatalog:
        _require_unique(self.cities, "city_id", lambda city: city.city_id)
        _require_unique(
            self.cities,
            "prefecture administrative code",
            lambda city: (
                city.administrative_codes.country,
                city.administrative_codes.prefecture,
            ),
        )
        legacy = [
            city.legacy_city_code for city in self.cities if city.legacy_city_code is not None
        ]
        if len(set(legacy)) != len(legacy):
            raise ValueError("legacy city codes must be unique")
        # AMap's transit citycode belongs to the prefecture-level transit area.
        # County-level cities under the same prefecture therefore legitimately
        # share it, while the destination adcode used by place lookup stays
        # unique.
        for provider_field in ("amap", "baidu", "flyai", "weather"):
            values = [
                (normalize_city_reference(code), city.city_id)
                for city in self.cities
                if (code := getattr(city.provider_codes, provider_field)) is not None
            ]
            seen: dict[str, str] = {}
            for code, city_id in values:
                previous = seen.get(code)
                if previous is not None and previous != city_id:
                    raise ValueError(f"duplicate {provider_field} provider city code")
                seen[code] = city_id
        return self


class MainlandCityCatalogSource(CityRegistryModel):
    provider: Literal["amap"]
    download_url: NonEmptyText
    workbook_name: Literal["AMap_adcode_citycode.xlsx"]
    published_on: date
    workbook_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class MainlandCityCatalogEntry(CityRegistryModel):
    official_name: NonEmptyText
    adcode: str = Field(pattern=ADMINISTRATIVE_CODE_PATTERN)
    citycode: NonEmptyText
    province_name: NonEmptyText
    province_adcode: str = Field(pattern=ADMINISTRATIVE_CODE_PATTERN)

    @model_validator(mode="after")
    def province_matches_destination(self) -> MainlandCityCatalogEntry:
        if self.adcode[:2] != self.province_adcode[:2] or not self.province_adcode.endswith("0000"):
            raise ValueError("destination adcode must belong to its province")
        return self


class MainlandCityCatalog(CityRegistryModel):
    catalog_version: str = Field(pattern=SEMANTIC_VERSION_PATTERN)
    scope: Literal["mainland_city_like_destinations"]
    source: MainlandCityCatalogSource
    cities: tuple[MainlandCityCatalogEntry, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def destination_keys_are_unique(self) -> MainlandCityCatalog:
        _require_unique(self.cities, "mainland destination adcode", lambda city: city.adcode)
        _require_unique(
            self.cities,
            "mainland destination official name",
            lambda city: normalize_city_reference(city.official_name),
        )
        return self


def _require_unique(
    cities: tuple[RegistryEntry, ...],
    label: str,
    value_for: Callable[[RegistryEntry], object],
) -> None:
    values = [value_for(city) for city in cities]
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must be unique")


V3_CITY_REGISTRY_CONTRACTS: tuple[type[BaseModel], ...] = (CityRegistryCatalog,)
