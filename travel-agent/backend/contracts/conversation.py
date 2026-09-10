"""V2 conversation messages and typed attachment contracts."""

from __future__ import annotations

from datetime import date
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import AwareDatetime, ConfigDict, Field, HttpUrl, RootModel, model_validator

from backend.contracts.base import ContractModel
from backend.contracts.commands import AttachmentAnswerValue
from backend.contracts.common import NonEmptyText, ShortText
from backend.contracts.enums import DataAvailability, ProviderCode

ATTACHMENT_AVAILABILITY_SCHEMA_RULE: dict[str, Any] = {
    "allOf": [
        {
            "if": {
                "properties": {"availability": {"const": "available"}},
                "required": ["availability"],
            },
            "then": {"properties": {"missing_reason": {"type": "null"}}},
            "else": {
                "properties": {
                    "missing_reason": {
                        "minLength": 1,
                        "pattern": r"\S",
                        "type": "string",
                    }
                },
                "required": ["missing_reason"],
            },
        },
        {
            "if": {
                "properties": {"availability": {"const": "missing"}},
                "required": ["availability"],
            },
            "then": {"properties": {"external_fact_ids": {"type": "array", "maxItems": 0}}},
        },
        {
            "if": {
                "properties": {"status": {"const": "superseded"}},
                "required": ["status"],
            },
            "then": {"properties": {"editable": {"const": False}}},
        },
    ]
}


def _selection_schema_rule(
    item_field: str,
    item_id_field: str,
    exclusive_field: str | None = None,
) -> dict[str, Any]:
    selection_rule = {
        "itemsField": item_field,
        "itemIdField": item_id_field,
        "minimumField": "minimum_selections",
        "maximumField": "maximum_selections",
    }
    if exclusive_field is not None:
        selection_rule["exclusiveField"] = exclusive_field
    return {
        **ATTACHMENT_AVAILABILITY_SCHEMA_RULE,
        "x-travel-selection-bounds": selection_rule,
    }


MESSAGE_SCHEMA_RULE: dict[str, Any] = {
    "x-travel-conversation-message": {
        "messageIdField": "message_id",
        "textField": "text",
        "attachmentsField": "attachments",
        "attachmentAnswersField": "attachment_answers",
        "attachmentIdField": "attachment_id",
        "sourceMessageIdField": "source_message_id",
        "stateVersionField": "state_version",
        "generationIdField": "generation_id",
        "factsField": "external_facts",
        "factIdField": "fact_id",
        "attachmentFactIdsField": "external_fact_ids",
        "textFactIdsField": "text_external_fact_ids",
    }
}


class ExternalFactReference(ContractModel):
    fact_id: UUID
    provider: ProviderCode
    source_record_id: NonEmptyText
    retrieved_at: AwareDatetime


class ChoiceSemanticValue(ContractModel):
    """Typed server meaning for one reviewed exploration option."""

    domain: Literal["dining_direction", "city_theme"]
    kind: Literal[
        "theme",
        "cuisine",
        "dietary_requirement",
        "specific_restaurant",
        "open_to_any",
    ]
    value: ShortText | None = None
    place_id: UUID | None = None
    source_fact_ids: list[UUID] = Field(
        default_factory=list,
        json_schema_extra={"uniqueItems": True},
    )

    @model_validator(mode="after")
    def fields_match_domain_and_kind(self) -> ChoiceSemanticValue:
        if len(set(self.source_fact_ids)) != len(self.source_fact_ids):
            raise ValueError("choice semantic fact references must be unique")
        if self.kind == "open_to_any":
            if self.value is not None or self.place_id is not None or self.source_fact_ids:
                raise ValueError("open dining direction cannot contain a value, place or facts")
            return self
        if self.domain == "city_theme":
            if self.kind != "theme":
                raise ValueError("city theme choices require theme or open-to-any meaning")
            if self.value is None or self.place_id is not None or not self.source_fact_ids:
                raise ValueError("a city theme requires a value and reviewed source facts")
            return self
        if self.kind == "theme":
            raise ValueError("dining directions cannot use a city theme meaning")
        if self.value is None or not self.source_fact_ids:
            raise ValueError("a concrete dining direction requires a value and source facts")
        if self.kind == "specific_restaurant":
            if self.place_id is None:
                raise ValueError("a restaurant dining direction requires a real place_id")
        elif self.place_id is not None:
            raise ValueError("only a restaurant dining direction can reference a place")
        return self


class ChoiceOption(ContractModel):
    option_id: NonEmptyText
    label: NonEmptyText
    description: ShortText | None = None
    semantic_value: ChoiceSemanticValue | None = None


class RecommendationItem(ContractModel):
    recommendation_id: UUID
    title: NonEmptyText
    summary: ShortText
    image_url: HttpUrl | None = None
    place_id: UUID | None = None
    city_significance: ShortText | None = None
    experience_summary: ShortText | None = None
    time_cost: ShortText | None = None
    physical_cost: ShortText | None = None
    match_reason: ShortText | None = None
    main_cost: ShortText | None = None
    source_fact_ids: list[UUID] = Field(
        default_factory=list,
        json_schema_extra={"uniqueItems": True},
    )


class WeatherDaySummary(ContractModel):
    date: date
    daytime_condition: NonEmptyText
    nighttime_condition: NonEmptyText | None = None
    minimum_celsius: float = Field(ge=-90, le=70, allow_inf_nan=False)
    maximum_celsius: float = Field(ge=-90, le=70, allow_inf_nan=False)

    @model_validator(mode="after")
    def maximum_is_not_below_minimum(self) -> WeatherDaySummary:
        if self.maximum_celsius < self.minimum_celsius:
            raise ValueError("maximum_celsius cannot be below minimum_celsius")
        return self


class AttachmentBase(ContractModel):
    attachment_id: UUID
    source_message_id: UUID
    created_at: AwareDatetime
    state_version: int = Field(ge=0, strict=True)
    generation_id: UUID | None = None
    status: Literal["active", "completed", "superseded"] = "active"
    editable: bool = True
    availability: DataAvailability = DataAvailability.AVAILABLE
    missing_reason: ShortText | None = None
    external_fact_ids: list[UUID] = Field(
        default_factory=list,
        json_schema_extra={"uniqueItems": True},
    )
    prompt: NonEmptyText

    @model_validator(mode="after")
    def availability_and_editability_are_consistent(self) -> AttachmentBase:
        if self.availability is DataAvailability.AVAILABLE:
            if self.missing_reason is not None:
                raise ValueError("available attachments cannot declare a missing reason")
        elif self.missing_reason is None:
            raise ValueError("partial or missing attachments require a missing reason")
        if self.availability is DataAvailability.MISSING and self.external_fact_ids:
            raise ValueError("missing attachments cannot reference available external facts")
        if self.status == "superseded" and self.editable:
            raise ValueError("superseded attachments cannot remain editable")
        if len(set(self.external_fact_ids)) != len(self.external_fact_ids):
            raise ValueError("external_fact_ids must not contain duplicates")
        return self


class SelectableAttachmentBase(AttachmentBase):
    minimum_selections: int = Field(ge=0, strict=True)
    maximum_selections: int = Field(ge=0, strict=True)

    def validate_selection_bounds(self, item_count: int) -> None:
        if self.minimum_selections > self.maximum_selections:
            raise ValueError("minimum_selections cannot exceed maximum_selections")
        if self.maximum_selections > item_count:
            raise ValueError("maximum_selections cannot exceed the number of choices")


class CompactChoiceAttachment(SelectableAttachmentBase):
    model_config = ConfigDict(json_schema_extra=_selection_schema_rule("options", "option_id"))

    kind: Literal["compact_choice"]
    minimum_selections: Literal[1] = 1
    maximum_selections: Literal[1] = 1
    options: list[ChoiceOption] = Field(min_length=2, max_length=4)

    @model_validator(mode="after")
    def options_are_unique(self) -> CompactChoiceAttachment:
        _validate_unique_option_ids(self.options)
        self.validate_selection_bounds(len(self.options))
        return self


class CompactMultiAttachment(SelectableAttachmentBase):
    model_config = ConfigDict(json_schema_extra=_selection_schema_rule("options", "option_id"))

    kind: Literal["compact_multi"]
    minimum_selections: int = Field(default=1, ge=1, le=8, strict=True)
    maximum_selections: int = Field(ge=1, le=8, strict=True)
    options: list[ChoiceOption] = Field(min_length=2, max_length=8)

    @model_validator(mode="after")
    def options_and_bounds_are_valid(self) -> CompactMultiAttachment:
        _validate_unique_option_ids(self.options)
        self.validate_selection_bounds(len(self.options))
        return self


class DetailedChoiceAttachment(SelectableAttachmentBase):
    model_config = ConfigDict(json_schema_extra=_selection_schema_rule("options", "option_id"))

    kind: Literal["detailed_choice"]
    minimum_selections: Literal[1] = 1
    maximum_selections: Literal[1] = 1
    options: list[ChoiceOption] = Field(min_length=2, max_length=4)

    @model_validator(mode="after")
    def options_are_unique(self) -> DetailedChoiceAttachment:
        _validate_unique_option_ids(self.options)
        self.validate_selection_bounds(len(self.options))
        return self


class PreferenceSliderAttachment(AttachmentBase):
    model_config = ConfigDict(
        json_schema_extra={
            **ATTACHMENT_AVAILABILITY_SCHEMA_RULE,
            "x-travel-slider-bounds": {
                "minimumField": "minimum_value",
                "maximumField": "maximum_value",
                "stepField": "step",
            },
        }
    )

    kind: Literal["preference_slider"]
    minimum_value: float = Field(allow_inf_nan=False)
    maximum_value: float = Field(allow_inf_nan=False)
    step: float = Field(gt=0, allow_inf_nan=False)
    minimum_label: NonEmptyText
    maximum_label: NonEmptyText
    allow_no_preference: bool = True

    @model_validator(mode="after")
    def slider_bounds_are_valid(self) -> PreferenceSliderAttachment:
        if self.maximum_value <= self.minimum_value:
            raise ValueError("maximum_value must be greater than minimum_value")
        if self.step > self.maximum_value - self.minimum_value:
            raise ValueError("step cannot exceed the slider range")
        return self


class TextMultiChoiceAttachment(SelectableAttachmentBase):
    model_config = ConfigDict(
        json_schema_extra={
            **_selection_schema_rule("options", "option_id", "exclusive_option_id"),
            "x-travel-semantic-choice": {
                "domainField": "interaction_domain",
                "optionsField": "options",
                "exclusiveField": "exclusive_option_id",
                "attachmentFactIdsField": "external_fact_ids",
            },
        }
    )

    kind: Literal["text_multi_choice"]
    interaction_domain: Literal["generic", "dining_direction", "city_theme"] = "generic"
    context_label: ShortText | None = None
    minimum_selections: int = Field(default=1, ge=1, le=12, strict=True)
    maximum_selections: int = Field(ge=1, le=12, strict=True)
    options: list[ChoiceOption] = Field(min_length=2, max_length=12)
    exclusive_option_id: NonEmptyText | None = None

    @model_validator(mode="after")
    def options_and_bounds_are_valid(self) -> TextMultiChoiceAttachment:
        _validate_unique_option_ids(self.options)
        self.validate_selection_bounds(len(self.options))
        option_ids = {option.option_id for option in self.options}
        if self.exclusive_option_id is not None and self.exclusive_option_id not in option_ids:
            raise ValueError("exclusive_option_id must reference a declared option")
        semantic_values = [option.semantic_value for option in self.options]
        if self.interaction_domain == "generic":
            if any(value is not None for value in semantic_values):
                raise ValueError("generic text choices cannot contain dining semantic values")
            return self
        if self.interaction_domain == "city_theme" and self.context_label is None:
            raise ValueError("city theme choices require a city context label")
        if self.interaction_domain != "city_theme" and self.context_label is not None:
            raise ValueError("only city theme choices may contain a context label")
        if any(value is None for value in semantic_values):
            raise ValueError("semantic choice options require typed semantic values")
        if any(
            value is not None and value.domain != self.interaction_domain
            for value in semantic_values
        ):
            raise ValueError("choice semantic domains must match the attachment domain")
        open_options = [
            option.option_id
            for option in self.options
            if option.semantic_value is not None and option.semantic_value.kind == "open_to_any"
        ]
        if len(open_options) != 1 or self.exclusive_option_id != open_options[0]:
            raise ValueError("semantic choices require one exclusive open-to-any option")
        option_fact_ids = {
            fact_id
            for option in self.options
            if option.semantic_value is not None
            for fact_id in option.semantic_value.source_fact_ids
        }
        if not option_fact_ids <= set(self.external_fact_ids):
            raise ValueError("choice option facts must belong to the attachment")
        return self


class RecommendationSetAttachment(SelectableAttachmentBase):
    model_config = ConfigDict(
        json_schema_extra={
            **_selection_schema_rule("items", "recommendation_id"),
            "x-travel-recommendation-set": {
                "domainField": "recommendation_domain",
                "itemsField": "items",
                "placeIdField": "place_id",
                "itemFactIdsField": "source_fact_ids",
                "attachmentFactIdsField": "external_fact_ids",
            },
            "allOf": [
                *ATTACHMENT_AVAILABILITY_SCHEMA_RULE["allOf"],
                {
                    "if": {
                        "properties": {"availability": {"const": "missing"}},
                        "required": ["availability"],
                    },
                    "then": {"properties": {"items": {"type": "array", "maxItems": 0}}},
                    "else": {"properties": {"items": {"type": "array", "minItems": 1}}},
                },
                {
                    "if": {
                        "properties": {"availability": {"const": "missing"}},
                        "required": ["availability"],
                    },
                    "else": {
                        "properties": {"external_fact_ids": {"type": "array", "minItems": 1}},
                        "required": ["external_fact_ids"],
                    },
                },
                {
                    "if": {
                        "properties": {"recommendation_domain": {"const": "attraction"}},
                        "required": ["recommendation_domain"],
                    },
                    "then": {
                        "properties": {
                            "items": {
                                "type": "array",
                                "maxItems": 10,
                                "items": {
                                    "type": "object",
                                    "required": [
                                        "place_id",
                                        "city_significance",
                                        "experience_summary",
                                        "time_cost",
                                        "physical_cost",
                                        "source_fact_ids",
                                    ],
                                    "properties": {
                                        "place_id": {"type": "string"},
                                        "city_significance": {"type": "string"},
                                        "experience_summary": {"type": "string"},
                                        "time_cost": {"type": "string"},
                                        "physical_cost": {"type": "string"},
                                        "source_fact_ids": {
                                            "type": "array",
                                            "minItems": 1,
                                            "uniqueItems": True,
                                        },
                                    },
                                },
                            }
                        }
                    },
                },
                {
                    "if": {
                        "properties": {
                            "recommendation_domain": {"const": "attraction"},
                            "availability": {"const": "available"},
                        },
                        "required": ["recommendation_domain", "availability"],
                    },
                    "then": {
                        "properties": {"items": {"type": "array", "minItems": 5, "maxItems": 10}}
                    },
                },
            ],
        }
    )

    kind: Literal["recommendation_set"]
    recommendation_domain: Literal["generic", "attraction", "restaurant", "hotel"] = "generic"
    minimum_selections: int = Field(default=0, ge=0, le=20, strict=True)
    maximum_selections: int = Field(ge=0, le=20, strict=True)
    items: list[RecommendationItem] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def items_and_bounds_are_valid(self) -> RecommendationSetAttachment:
        item_ids = [item.recommendation_id for item in self.items]
        if len(set(item_ids)) != len(item_ids):
            raise ValueError("recommendation_id must not contain duplicates")
        self.validate_selection_bounds(len(self.items))
        if self.availability is DataAvailability.MISSING:
            if self.items:
                raise ValueError("missing recommendation sets cannot contain items")
        elif not self.items:
            raise ValueError("available or partial recommendation sets require items")
        if self.availability is not DataAvailability.MISSING and not self.external_fact_ids:
            raise ValueError("recommendation sets require at least one external fact source")
        for item in self.items:
            if len(set(item.source_fact_ids)) != len(item.source_fact_ids):
                raise ValueError("recommendation item fact references must be unique")
        item_fact_ids = [fact_id for item in self.items for fact_id in item.source_fact_ids]
        if not set(item_fact_ids) <= set(self.external_fact_ids):
            raise ValueError("recommendation items may only reference attachment facts")
        if self.recommendation_domain == "attraction":
            if self.availability is DataAvailability.AVAILABLE and not 5 <= len(self.items) <= 10:
                raise ValueError("available attraction recommendations require 5 to 10 items")
            if self.availability is DataAvailability.PARTIAL and not 1 <= len(self.items) <= 10:
                raise ValueError("partial attraction recommendations require 1 to 10 items")
            place_ids = [item.place_id for item in self.items]
            if any(place_id is None for place_id in place_ids):
                raise ValueError("attraction recommendations require a real place_id")
            if len(set(place_ids)) != len(place_ids):
                raise ValueError("attraction recommendation place_id must be unique")
            for item in self.items:
                if not all(
                    (
                        item.city_significance,
                        item.experience_summary,
                        item.time_cost,
                        item.physical_cost,
                        item.source_fact_ids,
                    )
                ):
                    raise ValueError(
                        "attraction recommendations require significance, experience, "
                        "cost and sources"
                    )
        if self.recommendation_domain == "restaurant":
            if self.availability is DataAvailability.AVAILABLE and not 1 <= len(self.items) <= 6:
                raise ValueError("available restaurant recommendations require 1 to 6 items")
            if self.availability is DataAvailability.PARTIAL and not 1 <= len(self.items) <= 6:
                raise ValueError("partial restaurant recommendations require 1 to 6 items")
            place_ids = [item.place_id for item in self.items]
            if any(place_id is None for place_id in place_ids):
                raise ValueError("restaurant recommendations require a real place_id")
            if len(set(place_ids)) != len(place_ids):
                raise ValueError("restaurant recommendation place_id must be unique")
            for item in self.items:
                if not item.match_reason or not item.main_cost or not item.source_fact_ids:
                    raise ValueError(
                        "restaurant recommendations require match reason, main cost and sources"
                    )
        return self


class TaskBookReferenceAttachment(AttachmentBase):
    model_config = ConfigDict(
        json_schema_extra={
            **ATTACHMENT_AVAILABILITY_SCHEMA_RULE,
            "allOf": [
                *ATTACHMENT_AVAILABILITY_SCHEMA_RULE["allOf"],
                {
                    "if": {
                        "properties": {"availability": {"const": "missing"}},
                        "required": ["availability"],
                    },
                    "then": {"properties": {"task_book_id": {"type": "null"}}},
                    "else": {
                        "properties": {"task_book_id": {"not": {"type": "null"}}},
                        "required": ["task_book_id"],
                    },
                },
            ],
        }
    )

    kind: Literal["task_book_reference"]
    task_book_id: UUID | None
    label: NonEmptyText

    @model_validator(mode="after")
    def task_book_matches_availability(self) -> TaskBookReferenceAttachment:
        if self.availability is DataAvailability.MISSING:
            if self.task_book_id is not None:
                raise ValueError("missing task-book references cannot contain task_book_id")
        elif self.task_book_id is None:
            raise ValueError("available task-book references require task_book_id")
        return self


class WeatherAttachment(AttachmentBase):
    model_config = ConfigDict(
        json_schema_extra={
            **ATTACHMENT_AVAILABILITY_SCHEMA_RULE,
            "x-travel-unique-by": {"arrayField": "days", "itemField": "date"},
            "x-travel-weather-days": {
                "daysField": "days",
                "minimumField": "minimum_celsius",
                "maximumField": "maximum_celsius",
            },
            "allOf": [
                *ATTACHMENT_AVAILABILITY_SCHEMA_RULE["allOf"],
                {
                    "if": {
                        "properties": {"availability": {"const": "missing"}},
                        "required": ["availability"],
                    },
                    "then": {"properties": {"days": {"type": "array", "maxItems": 0}}},
                    "else": {"properties": {"days": {"type": "array", "minItems": 1}}},
                },
                {
                    "if": {
                        "properties": {"availability": {"const": "missing"}},
                        "required": ["availability"],
                    },
                    "else": {
                        "properties": {"external_fact_ids": {"type": "array", "minItems": 1}},
                        "required": ["external_fact_ids"],
                    },
                },
            ],
        }
    )

    kind: Literal["weather"]
    days: list[WeatherDaySummary] = Field(default_factory=list, max_length=5)

    @model_validator(mode="after")
    def days_match_availability(self) -> WeatherAttachment:
        day_values = [day.date for day in self.days]
        if len(set(day_values)) != len(day_values):
            raise ValueError("weather dates must not contain duplicates")
        if self.availability is DataAvailability.MISSING:
            if self.days:
                raise ValueError("missing weather attachments cannot contain days")
        elif not self.days:
            raise ValueError("available or partial weather attachments require days")
        if self.availability is not DataAvailability.MISSING and not self.external_fact_ids:
            raise ValueError("weather attachments require at least one external fact source")
        return self


ConversationAttachmentValue = Annotated[
    CompactChoiceAttachment
    | CompactMultiAttachment
    | DetailedChoiceAttachment
    | PreferenceSliderAttachment
    | TextMultiChoiceAttachment
    | RecommendationSetAttachment
    | TaskBookReferenceAttachment
    | WeatherAttachment,
    Field(discriminator="kind"),
]


class ConversationAttachment(RootModel[ConversationAttachmentValue]):
    """Discriminated union for every V2 message attachment."""


class AttachmentAnswerRecord(ContractModel):
    """Stable server-confirmed answer for one message attachment."""

    attachment_id: UUID
    answer: AttachmentAnswerValue
    answered_at: AwareDatetime
    state_version: int = Field(ge=0, strict=True)


class ConversationMessage(ContractModel):
    model_config = ConfigDict(json_schema_extra=MESSAGE_SCHEMA_RULE)

    message_id: UUID
    role: Literal["user", "assistant", "system"]
    text: NonEmptyText | None = None
    attachments: list[ConversationAttachment] = Field(default_factory=list, max_length=8)
    attachment_answers: list[AttachmentAnswerRecord] = Field(default_factory=list, max_length=8)
    text_external_fact_ids: list[UUID] = Field(
        default_factory=list,
        json_schema_extra={"uniqueItems": True},
    )
    external_facts: list[ExternalFactReference] = Field(default_factory=list)
    created_at: AwareDatetime
    state_version: int = Field(ge=0, strict=True)
    generation_id: UUID | None = None

    @model_validator(mode="after")
    def references_are_consistent(self) -> ConversationMessage:
        if self.text is None and not self.attachments:
            raise ValueError("conversation messages require text or at least one attachment")

        attachment_values = [attachment.root for attachment in self.attachments]
        attachment_ids = [attachment.attachment_id for attachment in attachment_values]
        if len(set(attachment_ids)) != len(attachment_ids):
            raise ValueError("attachment_id must not contain duplicates")

        answer_attachment_ids = [answer.attachment_id for answer in self.attachment_answers]
        if len(set(answer_attachment_ids)) != len(answer_attachment_ids):
            raise ValueError("attachment answers must not contain duplicate attachment_id values")
        if not set(answer_attachment_ids) <= set(attachment_ids):
            raise ValueError("attachment answers must reference an attachment in the message")
        if any(answer.state_version != self.state_version for answer in self.attachment_answers):
            raise ValueError("attachment answer state_version must match its containing message")

        fact_ids = [fact.fact_id for fact in self.external_facts]
        if len(set(fact_ids)) != len(fact_ids):
            raise ValueError("fact_id must not contain duplicates")
        if len(set(self.text_external_fact_ids)) != len(self.text_external_fact_ids):
            raise ValueError("text_external_fact_ids must not contain duplicates")

        referenced_fact_ids = set(self.text_external_fact_ids)
        for attachment in attachment_values:
            if attachment.source_message_id != self.message_id:
                raise ValueError("attachments must reference their containing message")
            if attachment.state_version != self.state_version:
                raise ValueError("attachment state_version must match its containing message")
            if attachment.generation_id != self.generation_id:
                raise ValueError("attachment generation_id must match its containing message")
            referenced_fact_ids.update(attachment.external_fact_ids)

        declared_fact_ids = set(fact_ids)
        if not referenced_fact_ids <= declared_fact_ids:
            raise ValueError("messages may only reference declared external facts")
        if declared_fact_ids != referenced_fact_ids:
            raise ValueError("declared external facts must be referenced by message content")
        return self


def _validate_unique_option_ids(options: list[ChoiceOption]) -> None:
    option_ids = [option.option_id for option in options]
    if len(set(option_ids)) != len(option_ids):
        raise ValueError("option_id must not contain duplicates")


V2_CONVERSATION_CONTRACTS: tuple[type[Any], ...] = (
    AttachmentAnswerRecord,
    ConversationAttachment,
    ConversationMessage,
)
