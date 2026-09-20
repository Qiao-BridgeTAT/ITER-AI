"""Server-signed V4 discovery card contracts."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import AwareDatetime, Field, HttpUrl, RootModel, model_validator

from backend.contracts.places import Gcj02Coordinates
from backend.contracts.v4.attraction_policy import ATTRACTION_TARGETS
from backend.contracts.v4.attraction_search import AttractionSearchHints
from backend.contracts.v4.base import DisplayText, Identifier, V4ContractModel, require_unique
from backend.contracts.v4.content_quality import (
    attraction_direction_quality_issue,
    require_meaningful_description,
    require_meaningful_label,
    require_visible_text,
)
from backend.contracts.v4.dining_display import DiningDisplayFacts
from backend.contracts.v4.dining_search import DiningSearchHints
from backend.contracts.v4.enums import (
    CardDomain,
    CardKind,
    CardStatus,
    CompositionRole,
    DiscoverySection,
    SelectionState,
)
from backend.contracts.v4.lodging_preferences import (
    LodgingAreaCopy,
    LodgingExample,
)
from backend.contracts.v4.visit_duration import VisitDurationRange

_ATTRACTION_COUNTS = {1: 5, 2: 7, 3: 9, 4: 12, 5: 14}
_DINING_COUNTS = {1: 3, 2: 4, 3: 8, 4: 10, 5: 12}


class CardGenerationMetadata(V4ContractModel):
    generated_at: AwareDatetime
    source_refs: list[Identifier] = Field(default_factory=list)
    model_plan_id: Identifier | None = None
    generation_mode: Literal["qwen", "safe_seed_fallback", "provider_composed", "static"]
    strategy_version: Literal[
        "legacy", "attraction-v2", "attraction-v3", "dining-v2", "dining-v3", "lodging-v2"
    ] = "legacy"


class CardEntityRef(V4ContractModel):
    canonical_entity_id: Identifier
    entity_kind: Literal["attraction", "restaurant", "area", "transit_node"]
    provider_entity_refs: list[Identifier] = Field(min_length=1)


class DirectionSemanticValue(V4ContractModel):
    kind: Literal["direction"]
    direction_id: Identifier
    direction_kind: Literal[
        "local_representative",
        "personalized",
        "hybrid",
        "area_strategy",
        "hotel_class",
    ]
    tags: list[Identifier] = Field(default_factory=list)
    search_query: DisplayText | None = None
    attraction_search_hints: AttractionSearchHints | None = None
    dining_search_hints: DiningSearchHints | None = None
    lodging_examples: list[LodgingExample] = Field(default_factory=list, max_length=3)
    hotel_quality_tier: Literal["economy", "comfort", "upscale", "luxury"] | None = None
    property_type: DisplayText | None = None
    nightly_budget_minimum_minor: int | None = Field(default=None, ge=0, strict=True)
    nightly_budget_maximum_minor: int | None = Field(default=None, ge=0, strict=True)

    @model_validator(mode="after")
    def lodging_fields_match_direction_kind(self) -> DirectionSemanticValue:
        lodging_values = (
            self.hotel_quality_tier,
            self.property_type,
            self.nightly_budget_minimum_minor,
            self.nightly_budget_maximum_minor,
        )
        if self.direction_kind != "hotel_class" and any(
            item is not None for item in lodging_values
        ):
            raise ValueError("lodging class fields require direction_kind=hotel_class")
        if (
            self.nightly_budget_minimum_minor is not None
            and self.nightly_budget_maximum_minor is not None
            and self.nightly_budget_maximum_minor < self.nightly_budget_minimum_minor
        ):
            raise ValueError("lodging budget maximum cannot be below minimum")
        return self


class EntitySemanticValue(V4ContractModel):
    kind: Literal["entity_disposition"]
    canonical_entity_id: Identifier
    allowed_dispositions: list[Literal["must", "want", "destination", "if_convenient", "avoid"]] = (
        Field(min_length=2)
    )


CardSemanticValueType = Annotated[
    DirectionSemanticValue | EntitySemanticValue,
    Field(discriminator="kind"),
]


class CardSemanticValue(RootModel[CardSemanticValueType]):
    pass


class CardOption(V4ContractModel):
    option_id: Identifier
    label: DisplayText
    description: DisplayText | None = None
    entity_ref: CardEntityRef | None = None
    semantic_value: CardSemanticValue
    signed_operation_ref: Identifier
    selection_state: SelectionState = SelectionState.AVAILABLE
    source_refs: list[Identifier] = Field(default_factory=list)
    observed_at: AwareDatetime | None = None
    composition_role: CompositionRole | None = None
    image_url: HttpUrl | None = None
    image_source_ref: Identifier | None = None
    dining_details: DiningDisplayFacts | None = None
    lodging_area_copy: LodgingAreaCopy | None = None
    selection_group: Literal["quality", "property_type"] | None = None
    coordinates: Gcj02Coordinates | None = None
    suggested_visit_duration: VisitDurationRange | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="模型估计的基本到深度游览分钟范围，不含交通、排队和用餐，不是营业事实。",
    )

    @model_validator(mode="after")
    def entity_and_observation_are_consistent(self) -> CardOption:
        if self.suggested_visit_duration is not None and (
            self.entity_ref is None or self.entity_ref.entity_kind != "attraction"
        ):
            raise ValueError("suggested visit duration requires an attraction entity")
        if self.coordinates is not None and (self.entity_ref is None or not self.source_refs):
            raise ValueError("card coordinates require a sourced entity")
        if self.dining_details is not None and (
            self.entity_ref is None
            or self.entity_ref.entity_kind != "restaurant"
            or self.dining_details.source_ref not in self.source_refs
        ):
            raise ValueError("dining facts require a restaurant and matching source reference")
        if self.image_url is not None and (
            self.image_source_ref is None or self.image_source_ref not in self.source_refs
        ):
            raise ValueError("card image requires a matching source reference")
        if self.image_url is None and self.image_source_ref is not None:
            raise ValueError("image source reference requires an image")
        value = self.semantic_value.root
        if isinstance(value, EntitySemanticValue):
            require_meaningful_label(self.label, "entity option label")
            if self.description is not None:
                require_visible_text(
                    self.description,
                    "entity option description",
                    minimum_units=2,
                )
            if self.entity_ref is None:
                raise ValueError("entity option requires entity_ref")
            if value.canonical_entity_id != self.entity_ref.canonical_entity_id:
                raise ValueError("entity semantic value must match entity_ref")
            if not self.source_refs or self.observed_at is None:
                raise ValueError("entity option requires sourced, observed facts")
        elif self.entity_ref is not None and self.entity_ref.entity_kind not in {
            "area",
            "transit_node",
        }:
            raise ValueError("direction option can only reference a real area or transit node")
        else:
            require_meaningful_label(self.label, "direction option label")
            if self.description is None:
                raise ValueError("direction option requires a complete description")
            require_meaningful_description(
                self.description,
                "direction option description",
            )
        return self


class CardControlAction(V4ContractModel):
    control_id: Identifier
    kind: Literal[
        "no_preference",
        "delegate",
        "existing_booking",
        "not_applicable",
        "refresh",
        "free_text",
    ]
    label: DisplayText
    signed_operation_ref: Identifier | None = None


class CardBase(V4ContractModel):
    attachment_id: Identifier
    interaction_id: Identifier
    kind: CardKind
    domain: CardDomain
    section: DiscoverySection
    based_on_state_version: int = Field(ge=0, strict=True)
    dependency_fingerprint: Identifier
    status: CardStatus
    options: list[CardOption]
    control_actions: list[CardControlAction] = Field(default_factory=list)
    generation_metadata: CardGenerationMetadata
    prompt: DisplayText

    @model_validator(mode="after")
    def option_and_control_ids_are_unique(self) -> CardBase:
        require_unique((option.option_id for option in self.options), "card option_id")
        require_unique(
            ("".join(option.label.casefold().split()) for option in self.options),
            "card option label",
        )
        require_unique((item.control_id for item in self.control_actions), "control_id")
        return self


class AttractionPreferenceCard(CardBase):
    kind: Literal[CardKind.PREFERENCE_CARD] = CardKind.PREFERENCE_CARD
    domain: Literal[CardDomain.ATTRACTION] = CardDomain.ATTRACTION
    section: Literal[DiscoverySection.ATTRACTION_PREFERENCE] = (
        DiscoverySection.ATTRACTION_PREFERENCE
    )
    options: list[CardOption] = Field(min_length=6, max_length=7)

    @model_validator(mode="after")
    def composition_has_city_and_personalized_directions(self) -> AttractionPreferenceCard:
        for option in self.options:
            issue = attraction_direction_quality_issue(option.label)
            if issue is not None:
                raise ValueError(issue)
        representative_count = sum(
            option.composition_role is CompositionRole.REPRESENTATIVE_EXTRA
            for option in self.options
        )
        if self.generation_metadata.strategy_version in {"attraction-v2", "attraction-v3"}:
            if representative_count not in {3, 4} or len(
                self.options
            ) - representative_count not in {2, 3}:
                raise ValueError(
                    "attraction directions require 3-4 city and 2-3 personalized options"
                )
        elif representative_count not in {2, 3}:
            raise ValueError(
                "legacy attraction preference requires 2-3 city representative options"
            )
        if representative_count == len(self.options):
            raise ValueError("attraction preference card must include personalized options")
        return self


class DiningPreferenceCard(CardBase):
    kind: Literal[CardKind.PREFERENCE_CARD] = CardKind.PREFERENCE_CARD
    domain: Literal[CardDomain.DINING] = CardDomain.DINING
    section: Literal[DiscoverySection.DINING_PREFERENCE] = DiscoverySection.DINING_PREFERENCE
    options: list[CardOption] = Field(min_length=1)

    @model_validator(mode="after")
    def composition_has_two_local_directions(self) -> DiningPreferenceCard:
        representative_count = sum(
            option.composition_role is CompositionRole.REPRESENTATIVE_EXTRA
            for option in self.options
        )
        if self.generation_metadata.strategy_version != "dining-v3" and (
            len(self.options) != 5 or representative_count != 2
        ):
            raise ValueError("legacy dining preference card requires five directions, two local")
        return self


class LodgingAreaPreferenceCard(CardBase):
    kind: Literal[CardKind.PREFERENCE_CARD] = CardKind.PREFERENCE_CARD
    domain: Literal[CardDomain.LODGING_AREA] = CardDomain.LODGING_AREA
    section: Literal[DiscoverySection.LODGING_AREA_PREFERENCE] = (
        DiscoverySection.LODGING_AREA_PREFERENCE
    )
    options: list[CardOption] = Field(min_length=3, max_length=6)

    @model_validator(mode="after")
    def areas_are_real_and_no_hotels_are_named(self) -> LodgingAreaPreferenceCard:
        if self.generation_metadata.strategy_version == "lodging-v2":
            if len(self.options) != 3 or any(
                option.lodging_area_copy is None for option in self.options
            ):
                raise ValueError("lodging-v2 requires three area groups with search leads")
            for option, (key, minimum, maximum) in zip(
                self.options,
                (("transit", 1, 2), ("attraction", 2, 3), ("commercial", 1, 2)),
                strict=True,
            ):
                value = option.semantic_value.root
                copy = option.lodging_area_copy
                if (
                    not isinstance(value, DirectionSemanticValue)
                    or value.direction_id != key
                    or copy is None
                    or not minimum <= len(copy.examples) <= maximum
                    or value.lodging_examples != copy.examples
                    or option.entity_ref is not None
                ):
                    raise ValueError("lodging-v2 group examples must match signed search leads")
            return self
        for option in self.options:
            if option.entity_ref is not None and option.entity_ref.entity_kind not in {
                "area",
                "transit_node",
            }:
                raise ValueError("lodging area card cannot contain a hotel or place candidate")
            if not option.source_refs:
                raise ValueError("lodging area option requires city/spatial evidence")
        return self


class LodgingClassPreferenceCard(CardBase):
    kind: Literal[CardKind.PREFERENCE_CARD] = CardKind.PREFERENCE_CARD
    domain: Literal[CardDomain.LODGING_CLASS] = CardDomain.LODGING_CLASS
    section: Literal[DiscoverySection.LODGING_CLASS_PREFERENCE] = (
        DiscoverySection.LODGING_CLASS_PREFERENCE
    )
    options: list[CardOption] = Field(min_length=5, max_length=6)

    @model_validator(mode="after")
    def class_card_never_contains_named_hotels(self) -> LodgingClassPreferenceCard:
        if self.generation_metadata.strategy_version == "lodging-v2":
            if [option.selection_group for option in self.options].count("quality") != 4 or [
                option.selection_group for option in self.options
            ].count("property_type") != 2:
                raise ValueError("lodging-v2 requires four quality and two property options")
        elif len(self.options) != 5:
            raise ValueError("legacy lodging class requires five options")
        if any(option.entity_ref is not None for option in self.options):
            raise ValueError("lodging class card cannot contain named hotel entities")
        return self


class SpecificCandidateCard(CardBase):
    kind: Literal[CardKind.SPECIFIC_CARD] = CardKind.SPECIFIC_CARD
    domain: Literal[CardDomain.ATTRACTION, CardDomain.DINING]
    section: Literal[
        DiscoverySection.ATTRACTION_SPECIFIC,
        DiscoverySection.DINING_SPECIFIC,
    ]
    duration_days: int = Field(ge=1, le=5, strict=True)
    options: list[CardOption] = Field(min_length=1, max_length=14)

    @model_validator(mode="after")
    def count_composition_and_dispositions_match_domain(self) -> SpecificCandidateCard:
        city_led = (
            self.domain is CardDomain.ATTRACTION
            and self.generation_metadata.strategy_version in {"attraction-v2", "attraction-v3"}
        )
        expected = (
            _ATTRACTION_COUNTS[self.duration_days]
            if self.domain is CardDomain.ATTRACTION
            else _DINING_COUNTS[self.duration_days]
        )
        if city_led and len(self.options) > ATTRACTION_TARGETS[self.duration_days].maximum:
            raise ValueError("attraction card exceeds the duration's maximum target")
        if (
            not city_led
            and self.status is not CardStatus.PARTIAL_AVAILABILITY
            and len(self.options) != expected
        ):
            raise ValueError("specific candidate card count does not match trip duration")
        extra_count = 1 if self.duration_days == 1 else 2
        representative_count = sum(
            option.composition_role is CompositionRole.REPRESENTATIVE_EXTRA
            for option in self.options
        )
        if (
            not city_led
            and not (
                self.domain is CardDomain.DINING
                and self.generation_metadata.strategy_version in {"dining-v2", "dining-v3"}
            )
            and self.status is not CardStatus.PARTIAL_AVAILABILITY
            and representative_count < extra_count
        ):
            raise ValueError(
                "specific card is missing representative candidates outside the TOP set"
            )
        expected_section = (
            DiscoverySection.ATTRACTION_SPECIFIC
            if self.domain is CardDomain.ATTRACTION
            else DiscoverySection.DINING_SPECIFIC
        )
        if self.section is not expected_section:
            raise ValueError("specific candidate card section does not match its domain")
        allowed = (
            {"must", "want", "if_convenient", "avoid"}
            if self.domain is CardDomain.ATTRACTION
            else {"destination", "if_convenient", "avoid"}
        )
        expected_entity_kind = (
            "attraction" if self.domain is CardDomain.ATTRACTION else "restaurant"
        )
        for option in self.options:
            value = option.semantic_value.root
            if not isinstance(value, EntitySemanticValue):
                raise ValueError("specific candidate card options must be concrete entities")
            if option.entity_ref is None or option.entity_ref.entity_kind != expected_entity_kind:
                raise ValueError("candidate entity kind does not match card domain")
            if not set(value.allowed_dispositions).issubset(allowed):
                raise ValueError("candidate dispositions do not match card domain")
        return self


PreferenceDirectionCardValue = Annotated[
    AttractionPreferenceCard
    | DiningPreferenceCard
    | LodgingAreaPreferenceCard
    | LodgingClassPreferenceCard,
    Field(discriminator="domain"),
]


class PreferenceDirectionCard(RootModel[PreferenceDirectionCardValue]):
    """One strict union for every pre-task-book preference-direction card."""


V4_CARD_CONTRACTS = (
    PreferenceDirectionCard,
    AttractionPreferenceCard,
    DiningPreferenceCard,
    LodgingAreaPreferenceCard,
    LodgingClassPreferenceCard,
    SpecificCandidateCard,
)
