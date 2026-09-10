"""P0-14 versioned city brief, theme, attraction, source, and asset format."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Self
from uuid import UUID

from pydantic import AwareDatetime, ConfigDict, Field, HttpUrl, model_validator

from backend.contracts.base import ContractModel
from backend.contracts.city_registry import CITY_ID_PATTERN
from backend.contracts.common import NonEmptyText, ShortText
from backend.contracts.enums import (
    AssetKind,
    AssetRightsStatus,
    CityCode,
    CityHistoryStatus,
)

HISTORY_SCHEMA_RULE: dict[str, Any] = {
    "allOf": [
        {
            "if": {
                "properties": {"history_status": {"const": "available"}},
                "required": ["history_status"],
            },
            "then": {
                "properties": {
                    "history_nodes": {"type": "array", "minItems": 1, "maxItems": 4},
                    "history_degradation_note": {"type": "null"},
                }
            },
            "else": {
                "properties": {
                    "history_nodes": {"type": "array", "maxItems": 0},
                    "history_degradation_note": {"not": {"type": "null"}},
                },
                "required": ["history_degradation_note"],
            },
        }
    ]
}

ASSET_SCHEMA_RULE: dict[str, Any] = {
    "allOf": [
        {
            "if": {
                "properties": {"rights_status": {"const": "licensed"}},
                "required": ["rights_status"],
            },
            "then": {
                "properties": {
                    "license_name": {"not": {"type": "null"}},
                    "attribution": {"not": {"type": "null"}},
                    "source_url": {"not": {"type": "null"}},
                },
                "required": ["license_name", "attribution", "source_url"],
            },
        }
    ]
}

CITY_CONTENT_SCHEMA_RULE: dict[str, Any] = {
    "x-travel-city-content": {
        "createdField": "created_at",
        "updatedField": "updated_at",
        "sourcesField": "sources",
        "sourceIdField": "source_id",
        "assetsField": "assets",
        "assetIdField": "asset_id",
        "briefField": "city_brief",
        "themesField": "themes",
        "themeIdField": "theme_id",
        "attractionsField": "core_attractions",
        "placeIdField": "place_id",
    }
}


def _unique(values: Sequence[str | UUID], field_name: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} must not contain duplicates")


class CityContentSource(ContractModel):
    source_id: NonEmptyText
    title: NonEmptyText
    publisher: NonEmptyText
    url: HttpUrl
    retrieved_at: AwareDatetime


class CityAsset(ContractModel):
    model_config = ConfigDict(json_schema_extra=ASSET_SCHEMA_RULE)

    asset_id: NonEmptyText
    kind: AssetKind
    uri: NonEmptyText
    rights_status: AssetRightsStatus
    license_name: NonEmptyText | None = None
    attribution: ShortText | None = None
    source_url: HttpUrl | None = None

    @model_validator(mode="after")
    def licensed_assets_are_traceable(self) -> CityAsset:
        if self.rights_status is AssetRightsStatus.LICENSED and (
            self.license_name is None or self.attribution is None or self.source_url is None
        ):
            raise ValueError("licensed assets require license, attribution, and source URL")
        if self.kind is AssetKind.PLACEHOLDER and (
            self.rights_status is not AssetRightsStatus.PLACEHOLDER
        ):
            raise ValueError("placeholder assets must use placeholder rights status")
        return self


class CityHistoryNode(ContractModel):
    period: NonEmptyText
    title: NonEmptyText
    summary: ShortText
    source_ids: list[NonEmptyText] = Field(min_length=1)


class TypicalPlace(ContractModel):
    place_id: UUID
    name: NonEmptyText
    source_ids: list[NonEmptyText] = Field(min_length=1)


class CityExperienceDirection(ContractModel):
    direction_id: NonEmptyText
    label: NonEmptyText
    summary: ShortText
    typical_places: list[TypicalPlace] = Field(min_length=1, max_length=2)
    source_ids: list[NonEmptyText] = Field(min_length=1)


class CitySpatialRelationship(ContractModel):
    relationship_id: NonEmptyText
    description: ShortText
    related_place_ids: list[UUID] = Field(min_length=2)
    source_ids: list[NonEmptyText] = Field(min_length=1)


class CityBriefContent(ContractModel):
    model_config = ConfigDict(json_schema_extra=HISTORY_SCHEMA_RULE)

    one_sentence: ShortText
    history_status: CityHistoryStatus
    history_nodes: list[CityHistoryNode] = Field(default_factory=list, max_length=4)
    history_degradation_note: ShortText | None = None
    experience_directions: list[CityExperienceDirection] = Field(min_length=4, max_length=6)
    spatial_relationships: list[CitySpatialRelationship] = Field(min_length=1)
    planning_tradeoff: ShortText

    @model_validator(mode="after")
    def history_is_present_or_explicitly_degraded(self) -> CityBriefContent:
        if self.history_status is CityHistoryStatus.AVAILABLE:
            if not self.history_nodes or self.history_degradation_note is not None:
                raise ValueError("available history requires nodes and no degradation note")
        elif self.history_nodes or self.history_degradation_note is None:
            raise ValueError("missing history requires an explicit degradation note and no nodes")
        return self


class CityThemeContent(ContractModel):
    theme_id: NonEmptyText
    label: NonEmptyText
    summary: ShortText
    source_ids: list[NonEmptyText] = Field(min_length=1)


class CoreAttractionContent(ContractModel):
    place_id: UUID
    name: NonEmptyText
    theme_ids: list[NonEmptyText] = Field(min_length=1)
    city_importance: ShortText
    visitor_experience: ShortText
    recommended_duration_minutes: int = Field(ge=30, le=720, strict=True)
    tradeoffs: list[ShortText] = Field(min_length=1)
    source_ids: list[NonEmptyText] = Field(min_length=1)
    asset_id: NonEmptyText


class _CityContentBody(ContractModel):
    model_config = ConfigDict(json_schema_extra=CITY_CONTENT_SCHEMA_RULE)

    content_version: NonEmptyText = Field(pattern=r"^\d+\.\d+\.\d+$")
    created_at: AwareDatetime
    updated_at: AwareDatetime
    sources: list[CityContentSource] = Field(min_length=1)
    assets: list[CityAsset] = Field(min_length=1)
    city_brief: CityBriefContent
    themes: list[CityThemeContent] = Field(min_length=6, max_length=8)
    core_attractions: list[CoreAttractionContent] = Field(min_length=30, max_length=40)

    @model_validator(mode="after")
    def references_and_versions_are_consistent(self) -> Self:
        if self.updated_at < self.created_at:
            raise ValueError("updated_at cannot be before created_at")
        _unique([item.source_id for item in self.sources], "source_id")
        _unique([item.asset_id for item in self.assets], "asset_id")
        _unique([item.theme_id for item in self.themes], "theme_id")
        _unique([item.place_id for item in self.core_attractions], "place_id")

        source_ids = {item.source_id for item in self.sources}
        asset_ids = {item.asset_id for item in self.assets}
        theme_ids = {item.theme_id for item in self.themes}
        place_ids = {item.place_id for item in self.core_attractions}
        for references in self._all_source_references():
            if not set(references) <= source_ids:
                raise ValueError("city content may only reference declared sources")
        for direction in self.city_brief.experience_directions:
            if not {item.place_id for item in direction.typical_places} <= place_ids:
                raise ValueError("typical places must reference core attractions")
        for relationship in self.city_brief.spatial_relationships:
            if not set(relationship.related_place_ids) <= place_ids:
                raise ValueError("spatial relationships must reference core attractions")
        for attraction in self.core_attractions:
            if attraction.asset_id not in asset_ids:
                raise ValueError("core attractions must reference declared assets")
            if not set(attraction.theme_ids) <= theme_ids:
                raise ValueError("core attractions must reference declared themes")
        return self

    def _all_source_references(self) -> Iterable[list[str]]:
        for node in self.city_brief.history_nodes:
            yield node.source_ids
        for direction in self.city_brief.experience_directions:
            yield direction.source_ids
            for place in direction.typical_places:
                yield place.source_ids
        for relationship in self.city_brief.spatial_relationships:
            yield relationship.source_ids
        for theme in self.themes:
            yield theme.source_ids
        for attraction in self.core_attractions:
            yield attraction.source_ids


class CityContentPackage(_CityContentBody):
    """Legacy Beijing/Nanjing package retained for frozen UI fixtures."""

    city: CityCode


class RegisteredCityContentPackage(_CityContentBody):
    """V3 package keyed only by the nationwide registry city identity."""

    city_id: str = Field(pattern=CITY_ID_PATTERN)

    @classmethod
    def from_legacy(
        cls,
        city_id: str,
        legacy: CityContentPackage,
    ) -> RegisteredCityContentPackage:
        payload = legacy.model_dump(mode="python")
        payload.pop("city")
        payload["city_id"] = city_id
        return cls.model_validate(payload)


P0_CITY_CONTENT_CONTRACTS: tuple[type[Any], ...] = (CityContentPackage,)
V3_CITY_CONTENT_CONTRACTS: tuple[type[Any], ...] = (RegisteredCityContentPackage,)
