"""P0-04 and P0-05 theme, attraction, dining, and restaurant contracts."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from uuid import UUID

from pydantic import ConfigDict, Field, model_validator

from backend.contracts.base import ContractModel
from backend.contracts.common import NonEmptyText, ShortText
from backend.contracts.enums import (
    AttractionIntent,
    CityThemeMode,
    DiningOptionKind,
    FeedbackSource,
    RestaurantIntent,
)

THEME_SELECTION_SCHEMA_RULE: dict[str, Any] = {
    "allOf": [
        {
            "if": {
                "properties": {"mode": {"const": "selected"}},
                "required": ["mode"],
            },
            "then": {"properties": {"selected_theme_ids": {"minItems": 1, "type": "array"}}},
            "else": {"properties": {"selected_theme_ids": {"maxItems": 0, "type": "array"}}},
        }
    ]
}

DEFAULT_ATTRACTION_SCHEMA_RULE: dict[str, Any] = {
    "allOf": [
        {
            "if": {
                "properties": {"source": {"const": "system_default"}},
                "required": ["source"],
            },
            "then": {"properties": {"intent": {"const": "if_convenient", "type": "string"}}},
        }
    ]
}

POPULAR_EATERY_SCHEMA_RULE: dict[str, Any] = {
    "allOf": [
        {
            "if": {
                "properties": {"kind": {"const": "popular_eatery"}},
                "required": ["kind"],
            },
            "then": {
                "properties": {
                    "place_id": {"format": "uuid", "type": "string"},
                    "source_fact_ids": {"minItems": 1, "type": "array"},
                },
                "required": ["place_id", "source_fact_ids"],
            },
        }
    ]
}

DINING_SELECTION_SCHEMA_RULE: dict[str, Any] = {
    "allOf": [
        {
            "if": {
                "properties": {"open_to_any": {"const": True}},
                "required": ["open_to_any"],
            },
            "then": {"properties": {"selected_option_ids": {"maxItems": 0, "type": "array"}}},
            "else": {
                "anyOf": [
                    {
                        "properties": {"selected_option_ids": {"minItems": 1, "type": "array"}},
                        "required": ["selected_option_ids"],
                    },
                    {
                        "properties": {"free_text": {"minLength": 1, "type": "string"}},
                        "required": ["free_text"],
                    },
                ]
            },
        }
    ]
}

DEFAULT_RESTAURANT_SCHEMA_RULE: dict[str, Any] = {
    "allOf": [
        {
            "if": {
                "properties": {"source": {"const": "system_default"}},
                "required": ["source"],
            },
            "then": {"properties": {"intent": {"const": "if_convenient", "type": "string"}}},
        }
    ]
}


def _ensure_unique(values: Sequence[str | UUID], field_name: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} must not contain duplicates")


class CityThemeSelection(ContractModel):
    model_config = ConfigDict(json_schema_extra=THEME_SELECTION_SCHEMA_RULE)

    mode: CityThemeMode
    selected_theme_ids: list[NonEmptyText] = Field(
        default_factory=list,
        json_schema_extra={"uniqueItems": True},
    )
    free_text: ShortText | None = None

    @model_validator(mode="after")
    def selected_mode_matches_values(self) -> CityThemeSelection:
        _ensure_unique(self.selected_theme_ids, "selected_theme_ids")
        if self.mode is CityThemeMode.SELECTED and not self.selected_theme_ids:
            raise ValueError("selected mode requires at least one theme")
        if self.mode is CityThemeMode.OPEN_TO_ANY and self.selected_theme_ids:
            raise ValueError("open_to_any cannot include selected themes")
        return self


class AttractionFeedback(ContractModel):
    model_config = ConfigDict(json_schema_extra=DEFAULT_ATTRACTION_SCHEMA_RULE)

    place_id: UUID
    intent: AttractionIntent = AttractionIntent.IF_CONVENIENT
    source: FeedbackSource = FeedbackSource.SYSTEM_DEFAULT

    @model_validator(mode="after")
    def system_default_is_only_the_neutral_intent(self) -> AttractionFeedback:
        if (
            self.source is FeedbackSource.SYSTEM_DEFAULT
            and self.intent is not AttractionIntent.IF_CONVENIENT
        ):
            raise ValueError("system-default attraction feedback must be if_convenient")
        return self

    @property
    def is_explicit(self) -> bool:
        return self.source is not FeedbackSource.SYSTEM_DEFAULT


class AttractionFeedbackSubmission(ContractModel):
    model_config = ConfigDict(
        json_schema_extra={
            "x-travel-unique-by": {"arrayField": "feedback", "itemField": "place_id"}
        }
    )

    feedback: list[AttractionFeedback] = Field(default_factory=list)

    @model_validator(mode="after")
    def place_ids_are_unique(self) -> AttractionFeedbackSubmission:
        _ensure_unique([item.place_id for item in self.feedback], "attraction place_id")
        return self


class DiningOption(ContractModel):
    model_config = ConfigDict(json_schema_extra=POPULAR_EATERY_SCHEMA_RULE)

    option_id: NonEmptyText
    kind: DiningOptionKind
    label: NonEmptyText
    summary: ShortText
    place_id: UUID | None = None
    source_fact_ids: list[NonEmptyText] = Field(
        default_factory=list,
        json_schema_extra={"uniqueItems": True},
    )

    @model_validator(mode="after")
    def popular_eatery_has_traceable_evidence(self) -> DiningOption:
        _ensure_unique(self.source_fact_ids, "source_fact_ids")
        if self.kind is DiningOptionKind.POPULAR_EATERY and (
            self.place_id is None or not self.source_fact_ids
        ):
            raise ValueError("popular_eatery requires a canonical place and source facts")
        return self


class DiningOptionCard(ContractModel):
    model_config = ConfigDict(
        json_schema_extra={
            "x-travel-unique-by": {"arrayField": "options", "itemField": "option_id"}
        }
    )

    options: list[DiningOption] = Field(min_length=1)

    @model_validator(mode="after")
    def option_ids_are_unique(self) -> DiningOptionCard:
        _ensure_unique([item.option_id for item in self.options], "option_id")
        return self


class DiningPreferenceSelection(ContractModel):
    model_config = ConfigDict(json_schema_extra=DINING_SELECTION_SCHEMA_RULE)

    selected_option_ids: list[NonEmptyText] = Field(
        default_factory=list,
        json_schema_extra={"uniqueItems": True},
    )
    open_to_any: bool = False
    free_text: ShortText | None = None

    @model_validator(mode="after")
    def selection_is_explicit(self) -> DiningPreferenceSelection:
        _ensure_unique(self.selected_option_ids, "selected_option_ids")
        if self.open_to_any and self.selected_option_ids:
            raise ValueError("open_to_any cannot be combined with selected options")
        if not self.open_to_any and not self.selected_option_ids and self.free_text is None:
            raise ValueError("select a dining option, add free text, or use open_to_any")
        return self


class RestaurantFeedback(ContractModel):
    model_config = ConfigDict(json_schema_extra=DEFAULT_RESTAURANT_SCHEMA_RULE)

    place_id: UUID
    intent: RestaurantIntent = RestaurantIntent.IF_CONVENIENT
    source: FeedbackSource = FeedbackSource.SYSTEM_DEFAULT

    @model_validator(mode="after")
    def system_default_is_only_the_neutral_intent(self) -> RestaurantFeedback:
        if (
            self.source is FeedbackSource.SYSTEM_DEFAULT
            and self.intent is not RestaurantIntent.IF_CONVENIENT
        ):
            raise ValueError("system-default restaurant feedback must be if_convenient")
        return self

    @property
    def is_explicit(self) -> bool:
        return self.source is not FeedbackSource.SYSTEM_DEFAULT


class RestaurantFeedbackSubmission(ContractModel):
    model_config = ConfigDict(
        json_schema_extra={
            "x-travel-unique-by": {"arrayField": "feedback", "itemField": "place_id"}
        }
    )

    feedback: list[RestaurantFeedback] = Field(default_factory=list)

    @model_validator(mode="after")
    def place_ids_are_unique(self) -> RestaurantFeedbackSubmission:
        _ensure_unique([item.place_id for item in self.feedback], "restaurant place_id")
        return self


P0_FEEDBACK_CONTRACTS: tuple[type[ContractModel], ...] = (
    CityThemeSelection,
    AttractionFeedbackSubmission,
    DiningOptionCard,
    DiningPreferenceSelection,
    RestaurantFeedbackSubmission,
)
