"""V3-33 source-backed dining directions, restaurant projection and feedback."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime
from enum import StrEnum
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import ConfigDict, Field, model_validator

from backend.agent.model_gateway import ModelCancellation
from backend.agent.response_selection import ResponseActionKind, ResponseSelection
from backend.agent.semantic_operations import (
    DateRangeOperation,
    DestinationOperation,
    DiningPreferenceKind,
    DiningPreferenceOperation,
    DiningPreferenceValue,
    OperationEvidence,
    SemanticImpactKind,
    SemanticImpactScope,
    SemanticOperationBatch,
    SemanticOperationKind,
    SemanticPersistenceScope,
    SemanticTarget,
    whole_trip_impact,
)
from backend.agent.state_merge import SemanticTripState
from backend.contracts.base import ContractModel
from backend.contracts.candidate_ranking import (
    CandidateRankingPreferences,
    CandidateRankingRequest,
    CandidateRankingResult,
    RankedCandidate,
    RankingSelectionStatus,
)
from backend.contracts.candidate_recall import (
    CandidateDomain,
    CandidateRecallBudget,
    CandidateRecallRequest,
    DomainRecallBudget,
    ExcludedPlaceClue,
    LandmarkRecallPolicy,
    NamedPlaceClue,
    NamedPlacePriority,
    RecalledCandidate,
    RecallSourceKind,
    RecallThemeInput,
    RecallThemeMode,
    RecallThemeSource,
)
from backend.contracts.commands import MultiChoiceAnswer, RecommendationFeedbackAnswer
from backend.contracts.common import NonEmptyText, ShortText
from backend.contracts.conversation import (
    ChoiceOption,
    ChoiceSemanticValue,
    ExternalFactReference,
    RecommendationItem,
    RecommendationSetAttachment,
    TextMultiChoiceAttachment,
)
from backend.contracts.enums import (
    Confidence,
    DataAvailability,
    EvidenceSource,
    ProviderCode,
    RecommendationIntent,
    RestaurantIntent,
)
from backend.planning.candidate_ranking import CandidateRankingService
from backend.planning.candidate_recall import CandidateRecallService
from backend.planning.city_registry import CityRegistry


class ImmutableDiningModel(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DiningDirectionKind(StrEnum):
    CUISINE = "cuisine"
    DIETARY_REQUIREMENT = "dietary_requirement"
    SPECIFIC_RESTAURANT = "specific_restaurant"


class DiningDirectionSeed(ImmutableDiningModel):
    direction_id: NonEmptyText
    kind: DiningDirectionKind
    label: NonEmptyText
    summary: ShortText
    value: ShortText
    place_id: UUID | None = None
    source_fact_ids: tuple[UUID, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def place_and_sources_match_kind(self) -> DiningDirectionSeed:
        if len(set(self.source_fact_ids)) != len(self.source_fact_ids):
            raise ValueError("dining direction source facts must be unique")
        if self.kind is DiningDirectionKind.SPECIFIC_RESTAURANT:
            if self.place_id is None:
                raise ValueError("a restaurant direction requires a real place_id")
        elif self.place_id is not None:
            raise ValueError("only a restaurant direction may reference a place")
        return self


class DiningDirectionProjection(ImmutableDiningModel):
    attachment: TextMultiChoiceAttachment
    external_facts: tuple[ExternalFactReference, ...]

    @model_validator(mode="after")
    def facts_match_attachment(self) -> DiningDirectionProjection:
        if self.attachment.interaction_domain != "dining_direction":
            raise ValueError("dining projection requires a dining-direction attachment")
        if {fact.fact_id for fact in self.external_facts} != set(self.attachment.external_fact_ids):
            raise ValueError("dining direction facts must match attachment references")
        return self


class RestaurantCandidateDiningFacts(ImmutableDiningModel):
    candidate_id: UUID
    place_id: UUID
    cuisine_labels: tuple[NonEmptyText, ...] = ()
    dietary_labels: tuple[NonEmptyText, ...] = ()
    search_match_labels: tuple[NonEmptyText, ...] = ()
    ingredients_present: tuple[NonEmptyText, ...] = ()
    ingredients_absent: tuple[NonEmptyText, ...] = ()
    allergens_present: tuple[NonEmptyText, ...] = ()
    allergens_absent: tuple[NonEmptyText, ...] = ()
    source_fact_ids: tuple[UUID, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def facts_are_unique_and_non_contradictory(self) -> RestaurantCandidateDiningFacts:
        for label, values in {
            "cuisine labels": self.cuisine_labels,
            "dietary labels": self.dietary_labels,
            "search match labels": self.search_match_labels,
            "ingredients present": self.ingredients_present,
            "ingredients absent": self.ingredients_absent,
            "allergens present": self.allergens_present,
            "allergens absent": self.allergens_absent,
            "source facts": self.source_fact_ids,
        }.items():
            if len(set(values)) != len(values):
                raise ValueError(f"restaurant {label} must be unique")
        if _normalized_set(self.ingredients_present) & _normalized_set(self.ingredients_absent):
            raise ValueError("restaurant ingredient facts cannot contradict each other")
        if _normalized_set(self.allergens_present) & _normalized_set(self.allergens_absent):
            raise ValueError("restaurant allergen facts cannot contradict each other")
        return self


class RestaurantRecommendationProjection(ImmutableDiningModel):
    attachment: RecommendationSetAttachment
    external_facts: tuple[ExternalFactReference, ...] = ()
    excluded_candidate_ids: tuple[UUID, ...] = ()
    exclusion_reasons: tuple[ShortText, ...] = ()

    @model_validator(mode="after")
    def projection_is_owned_and_sourced(self) -> RestaurantRecommendationProjection:
        if self.attachment.recommendation_domain != "restaurant":
            raise ValueError("restaurant projection requires a restaurant recommendation")
        if {fact.fact_id for fact in self.external_facts} != set(self.attachment.external_fact_ids):
            raise ValueError("restaurant facts must match attachment references")
        if len(set(self.excluded_candidate_ids)) != len(self.excluded_candidate_ids):
            raise ValueError("excluded restaurant candidate IDs must be unique")
        if len(self.excluded_candidate_ids) != len(self.exclusion_reasons):
            raise ValueError("every excluded restaurant requires one safe reason")
        return self


class DiningDirectionProjector:
    def project(
        self,
        seeds: tuple[DiningDirectionSeed, ...],
        external_facts: tuple[ExternalFactReference, ...],
        *,
        attachment_id: UUID,
        source_message_id: UUID,
        created_at: datetime,
        state_version: int,
        generation_id: UUID,
    ) -> DiningDirectionProjection:
        if not 1 <= len(seeds) <= 5:
            raise ValueError("dining direction projection requires 1 to 5 sourced directions")
        fact_by_id = {fact.fact_id: fact for fact in external_facts}
        if len(fact_by_id) != len(external_facts):
            raise ValueError("dining direction facts must be unique")
        referenced_fact_ids = {fact_id for seed in seeds for fact_id in seed.source_fact_ids}
        if not referenced_fact_ids <= set(fact_by_id):
            raise ValueError("dining direction references an unknown fact")
        options = [
            ChoiceOption(
                option_id=seed.direction_id,
                label=seed.label,
                description=seed.summary,
                semantic_value=ChoiceSemanticValue(
                    domain="dining_direction",
                    kind=seed.kind.value,
                    value=seed.value,
                    place_id=seed.place_id,
                    source_fact_ids=list(seed.source_fact_ids),
                ),
            )
            for seed in seeds
        ]
        open_option_id = "dining:open_to_any"
        options.append(
            ChoiceOption(
                option_id=open_option_id,
                label="都可以，你来安排",
                description="保留当地代表味道，再结合路线选择。",
                semantic_value=ChoiceSemanticValue(
                    domain="dining_direction",
                    kind="open_to_any",
                ),
            )
        )
        attachment = TextMultiChoiceAttachment.model_validate(
            {
                "kind": "text_multi_choice",
                "interaction_domain": "dining_direction",
                "attachment_id": str(attachment_id),
                "source_message_id": str(source_message_id),
                "created_at": created_at,
                "state_version": state_version,
                "generation_id": str(generation_id),
                "prompt": "这次更想尝试哪些味道？也可以直接用文字告诉我。",
                "external_fact_ids": [str(item) for item in sorted(referenced_fact_ids, key=str)],
                "minimum_selections": 1,
                "maximum_selections": len(seeds),
                "exclusive_option_id": open_option_id,
                "options": [option.model_dump(mode="json") for option in options],
            }
        )
        return DiningDirectionProjection(
            attachment=attachment,
            external_facts=tuple(
                fact_by_id[fact_id] for fact_id in sorted(referenced_fact_ids, key=str)
            ),
        )


class DiningExplorationService:
    """Run restaurant recall, deterministic safety filtering and card projection."""

    def __init__(
        self,
        *,
        registry: CityRegistry,
        recall: CandidateRecallService,
        ranking: CandidateRankingService,
        recommendation_projector: RestaurantRecommendationProjector | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._registry = registry
        self._recall = recall
        self._ranking = ranking
        self._recommendation_projector = (
            recommendation_projector or RestaurantRecommendationProjector()
        )
        self._clock = clock

    async def recommendation_response(
        self,
        state: SemanticTripState,
        *,
        request_id: UUID,
        source_message_id: UUID,
        generation_id: UUID,
        cancellation: ModelCancellation | None = None,
    ) -> ResponseSelection | None:
        recall_request, preferences = _restaurant_recall_request(state, request_id=request_id)
        if recall_request is None or not preferences:
            return None
        # Resolve here as well as inside CandidateRecallService so an unsupported city
        # fails before any provider work and never falls back to a different city.
        self._registry.resolve(recall_request.city_id)
        recall_result = await self._recall.recall(recall_request, cancellation=cancellation)
        ranking = self._ranking.rank(
            CandidateRankingRequest(
                ranking_request_id=uuid5(NAMESPACE_URL, f"iter:restaurant-ranking:{request_id}"),
                recall_request=recall_request,
                recall_result=recall_result,
                preferences=CandidateRankingPreferences(
                    selected_theme_ids=tuple(theme.theme_id for theme in recall_request.themes)
                ),
                recommendation_limit=6,
            )
        )
        dining_facts, external_facts = _dining_facts_from_recall(
            ranking,
            preferences=preferences,
            themes=recall_request.themes,
        )
        projection = self._recommendation_projector.project(
            ranking,
            preferences=preferences,
            dining_facts=dining_facts,
            external_facts=external_facts,
            attachment_id=uuid5(NAMESPACE_URL, f"iter:restaurant-recommendation:{request_id}"),
            source_message_id=source_message_id,
            state_version=state.state_version,
            generation_id=generation_id,
        )
        return ResponseSelection(
            action=ResponseActionKind.CANDIDATE_RECOMMENDATION,
            reason="formal dining preferences drove restaurant recall, filtering and projection",
            recommendation=projection.attachment,
            recommendation_external_facts=projection.external_facts,
        )

    def _aware_now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("dining exploration clock must be timezone-aware")
        return value.astimezone(UTC)


class RestaurantRecommendationProjector:
    def project(
        self,
        ranking: CandidateRankingResult,
        *,
        preferences: tuple[DiningPreferenceValue, ...],
        dining_facts: tuple[RestaurantCandidateDiningFacts, ...],
        external_facts: tuple[ExternalFactReference, ...],
        attachment_id: UUID,
        source_message_id: UUID,
        state_version: int,
        generation_id: UUID,
    ) -> RestaurantRecommendationProjection:
        ranking = CandidateRankingResult.model_validate(ranking.model_dump(mode="json"))
        _validate_preferences(preferences)
        facts_by_candidate = {item.candidate_id: item for item in dining_facts}
        if len(facts_by_candidate) != len(dining_facts):
            raise ValueError("restaurant dining facts must be unique per candidate")
        supplied_facts = {fact.fact_id: fact for fact in external_facts}
        if len(supplied_facts) != len(external_facts):
            raise ValueError("restaurant external facts must be unique")

        selected = [
            item
            for item in ranking.candidates
            if item.selection_status is RankingSelectionStatus.RECOMMENDED
            and item.candidate.domain is CandidateDomain.RESTAURANT
        ]
        selected.sort(key=lambda item: _restaurant_priority(item, preferences))
        projected: list[RecommendationItem] = []
        projection_facts: dict[UUID, ExternalFactReference] = {}
        excluded_ids: list[UUID] = []
        exclusion_reasons: list[str] = []
        partial = ranking.status is not DataAvailability.AVAILABLE

        for item in selected:
            candidate = item.candidate
            profile = facts_by_candidate.get(candidate.candidate_id)
            if profile is not None and profile.place_id != candidate.place.place_id:
                raise ValueError("restaurant dining facts reference the wrong place")
            eligible, match_reason, risk = _restaurant_match(item, profile, preferences)
            if not eligible:
                excluded_ids.append(candidate.candidate_id)
                exclusion_reasons.append(risk or "餐厅与当前饮食要求冲突")
                continue
            item_fact_ids = _candidate_source_facts(item, ranking, projection_facts)
            if profile is not None:
                if not set(profile.source_fact_ids) <= set(supplied_facts):
                    raise ValueError("restaurant dining profile references an unknown fact")
                for fact_id in profile.source_fact_ids:
                    projection_facts.setdefault(fact_id, supplied_facts[fact_id])
                    item_fact_ids.append(fact_id)
            if not item_fact_ids:
                excluded_ids.append(candidate.candidate_id)
                exclusion_reasons.append("餐厅缺少可追溯来源")
                continue
            main_cost = _main_cost(item)
            if risk is not None:
                partial = True
                main_cost = f"{main_cost}；{risk}"
            projected.append(
                RecommendationItem(
                    recommendation_id=candidate.candidate_id,
                    place_id=candidate.place.place_id,
                    title=candidate.place.name,
                    summary=item.explanations[0],
                    image_url=candidate.place.image_url,
                    experience_summary=_restaurant_experience(candidate.reasons, profile),
                    match_reason=match_reason,
                    main_cost=main_cost,
                    source_fact_ids=list(dict.fromkeys(item_fact_ids)),
                )
            )
            if len(projected) == 6:
                break

        if not projected:
            reason = _empty_restaurant_reason(preferences, exclusion_reasons)
            return RestaurantRecommendationProjection(
                attachment=RecommendationSetAttachment(
                    kind="recommendation_set",
                    recommendation_domain="restaurant",
                    attachment_id=attachment_id,
                    source_message_id=source_message_id,
                    created_at=ranking.generated_at,
                    state_version=state_version,
                    generation_id=generation_id,
                    prompt="目前没有找到能安全满足这些饮食要求的餐厅。",
                    editable=True,
                    availability=DataAvailability.MISSING,
                    missing_reason=reason,
                    minimum_selections=0,
                    maximum_selections=0,
                    items=[],
                ),
                excluded_candidate_ids=tuple(excluded_ids),
                exclusion_reasons=tuple(exclusion_reasons),
            )

        availability = DataAvailability.PARTIAL if partial else DataAvailability.AVAILABLE
        attachment = RecommendationSetAttachment(
            kind="recommendation_set",
            recommendation_domain="restaurant",
            attachment_id=attachment_id,
            source_message_id=source_message_id,
            created_at=ranking.generated_at,
            state_version=state_version,
            generation_id=generation_id,
            prompt="这些餐厅里，有没有你想专程去的？不必全部回答。",
            editable=True,
            availability=availability,
            missing_reason=(
                "部分餐厅的饮食适配或位置代价资料不完整，已保留可核实结果"
                if availability is DataAvailability.PARTIAL
                else None
            ),
            external_fact_ids=list(projection_facts),
            minimum_selections=0,
            maximum_selections=len(projected),
            items=projected,
        )
        return RestaurantRecommendationProjection(
            attachment=attachment,
            external_facts=tuple(projection_facts.values()),
            excluded_candidate_ids=tuple(excluded_ids),
            exclusion_reasons=tuple(exclusion_reasons),
        )


def dining_direction_answer_operations(
    *,
    trip_id: UUID,
    request_id: UUID,
    source_message_id: UUID,
    attachment: TextMultiChoiceAttachment,
    answer: MultiChoiceAnswer,
    business_date: date,
) -> SemanticOperationBatch:
    if attachment.interaction_domain != "dining_direction":
        raise ValueError("only dining direction attachments use dining semantic mapping")
    options = {option.option_id: option for option in attachment.options}
    operations: list[DiningPreferenceOperation] = []
    for option_id in answer.option_ids:
        option = options.get(option_id)
        if option is None or option.semantic_value is None:
            raise ValueError("dining answer references an unknown option")
        semantic = option.semantic_value
        kind = DiningPreferenceKind(semantic.kind)
        value = DiningPreferenceValue(
            kind=kind,
            value=semantic.value,
            place_id=semantic.place_id,
            restaurant_intent=(
                RestaurantIntent.IF_CONVENIENT
                if kind is DiningPreferenceKind.SPECIFIC_RESTAURANT
                else None
            ),
        )
        impact = (
            SemanticImpactScope(
                kind=SemanticImpactKind.SPECIFIC_ITEM,
                item_id=semantic.place_id,
            )
            if semantic.place_id is not None
            else whole_trip_impact()
        )
        operations.append(
            _dining_operation(
                trip_id=trip_id,
                request_id=request_id,
                source_message_id=source_message_id,
                attachment_id=attachment.attachment_id,
                value=value,
                operation=SemanticOperationKind.APPEND,
                impact=impact,
                discriminator=option_id,
            )
        )
    return _validated_batch(trip_id, operations, business_date)


def restaurant_feedback_operations(
    *,
    trip_id: UUID,
    request_id: UUID,
    source_message_id: UUID,
    attachment: RecommendationSetAttachment,
    answer: RecommendationFeedbackAnswer,
    business_date: date,
) -> SemanticOperationBatch:
    if attachment.recommendation_domain != "restaurant":
        raise ValueError("only restaurant recommendations use restaurant feedback mapping")
    items = {item.recommendation_id: item for item in attachment.items}
    operations: list[DiningPreferenceOperation] = []
    for feedback in answer.feedback:
        item = items.get(feedback.recommendation_id)
        if item is None or item.place_id is None:
            raise ValueError("restaurant feedback references an unknown restaurant")
        restaurant_intent = {
            RecommendationIntent.MUST: RestaurantIntent.DESTINATION,
            RecommendationIntent.WANT: RestaurantIntent.DESTINATION,
            RecommendationIntent.IF_CONVENIENT: RestaurantIntent.IF_CONVENIENT,
            RecommendationIntent.AVOID: RestaurantIntent.AVOID,
        }[feedback.intent]
        operations.append(
            _dining_operation(
                trip_id=trip_id,
                request_id=request_id,
                source_message_id=source_message_id,
                attachment_id=attachment.attachment_id,
                value=DiningPreferenceValue(
                    kind=DiningPreferenceKind.SPECIFIC_RESTAURANT,
                    value=item.title,
                    place_id=item.place_id,
                    restaurant_intent=restaurant_intent,
                ),
                operation=(
                    SemanticOperationKind.NEGATE
                    if restaurant_intent is RestaurantIntent.AVOID
                    else SemanticOperationKind.OVERRIDE
                ),
                impact=SemanticImpactScope(
                    kind=SemanticImpactKind.SPECIFIC_ITEM,
                    item_id=item.place_id,
                ),
                discriminator=f"{item.place_id}:{restaurant_intent.value}",
            )
        )
    return _validated_batch(trip_id, operations, business_date)


def is_strong_restaurant_anchor(operation: DiningPreferenceOperation) -> bool:
    return (
        operation.value.kind is DiningPreferenceKind.SPECIFIC_RESTAURANT
        and operation.value.place_id is not None
        and operation.value.restaurant_intent is RestaurantIntent.DESTINATION
        and operation.operation is not SemanticOperationKind.NEGATE
    )


def _dining_operation(
    *,
    trip_id: UUID,
    request_id: UUID,
    source_message_id: UUID,
    attachment_id: UUID,
    value: DiningPreferenceValue,
    operation: SemanticOperationKind,
    impact: SemanticImpactScope,
    discriminator: str,
) -> DiningPreferenceOperation:
    return DiningPreferenceOperation(
        operation_id=uuid5(
            NAMESPACE_URL,
            f"iter:v3-dining:{request_id}:{attachment_id}:{discriminator}",
        ),
        trip_id=trip_id,
        operation=operation,
        target=SemanticTarget.DINING_PREFERENCES,
        value=value,
        evidence=OperationEvidence(
            source=EvidenceSource.CARD,
            source_trip_id=trip_id,
            source_message_id=source_message_id,
            source_attachment_id=attachment_id,
        ),
        confidence=Confidence.HIGH,
        persistence_scope=SemanticPersistenceScope.CURRENT_TRIP,
        impact_scope=impact,
    )


def _validated_batch(
    trip_id: UUID,
    operations: list[DiningPreferenceOperation],
    business_date: date,
) -> SemanticOperationBatch:
    return SemanticOperationBatch.model_validate(
        {
            "trip_id": str(trip_id),
            "operations": [item.model_dump(mode="json") for item in operations],
        },
        context={"today": business_date},
    )


def _validate_preferences(preferences: tuple[DiningPreferenceValue, ...]) -> None:
    if (
        any(item.kind is DiningPreferenceKind.OPEN_TO_ANY for item in preferences)
        and len(preferences) > 1
    ):
        raise ValueError("open-to-any dining cannot be combined with concrete requirements")


def _restaurant_priority(
    item: RankedCandidate,
    preferences: tuple[DiningPreferenceValue, ...],
) -> tuple[int, int, str]:
    place_id = item.candidate.place.place_id
    named = any(
        preference.kind is DiningPreferenceKind.SPECIFIC_RESTAURANT
        and preference.place_id == place_id
        for preference in preferences
    )
    return (0 if named else 1, item.rank, str(item.candidate.candidate_id))


def _restaurant_recall_request(
    state: SemanticTripState,
    *,
    request_id: UUID,
) -> tuple[CandidateRecallRequest | None, tuple[DiningPreferenceValue, ...]]:
    destination = next(
        (
            entry.operation.value
            for entry in state.entries
            if isinstance(entry.operation, DestinationOperation)
            and entry.operation.value is not None
        ),
        None,
    )
    date_range = next(
        (
            entry.operation.value
            for entry in state.entries
            if isinstance(entry.operation, DateRangeOperation) and entry.operation.value is not None
        ),
        None,
    )
    preference_operations = tuple(
        entry.operation
        for entry in state.entries
        if isinstance(entry.operation, DiningPreferenceOperation)
    )
    preferences = tuple(operation.value for operation in preference_operations)
    if destination is None or date_range is None or not preferences:
        return None, preferences

    themes: list[RecallThemeInput] = []
    named: list[NamedPlaceClue] = []
    excluded: list[ExcludedPlaceClue] = []
    free_text: list[str] = []
    special_constraints: list[str] = []
    open_to_any = False
    for operation in preference_operations:
        value = operation.value
        if value.kind is DiningPreferenceKind.OPEN_TO_ANY:
            open_to_any = True
            continue
        assert value.value is not None
        free_text.append(value.value)
        if value.kind in {
            DiningPreferenceKind.CUISINE,
            DiningPreferenceKind.DIETARY_REQUIREMENT,
        }:
            themes.append(
                RecallThemeInput(
                    theme_id=_dining_theme_id(value),
                    label=value.value,
                    source=RecallThemeSource.CURRENT_TRIP,
                    source_reference_ids=(str(operation.operation_id),),
                )
            )
        elif value.kind in {
            DiningPreferenceKind.ALLERGY,
            DiningPreferenceKind.AVOIDANCE,
        }:
            special_constraints.append(f"{value.kind.value}:{value.value}")
        elif value.kind is DiningPreferenceKind.SPECIFIC_RESTAURANT:
            if value.restaurant_intent is RestaurantIntent.AVOID:
                excluded.append(
                    ExcludedPlaceClue(
                        clue_id=operation.operation_id,
                        name=value.value,
                        source_message_id=(
                            operation.evidence.source_message_id or operation.operation_id
                        ),
                        domain=CandidateDomain.RESTAURANT,
                        known_place_id=value.place_id,
                    )
                )
            else:
                named.append(
                    NamedPlaceClue(
                        clue_id=operation.operation_id,
                        name=value.value,
                        domain=CandidateDomain.RESTAURANT,
                        priority=(
                            NamedPlacePriority.MUST
                            if value.restaurant_intent is RestaurantIntent.DESTINATION
                            else NamedPlacePriority.MENTIONED
                        ),
                        source_message_id=(
                            operation.evidence.source_message_id or operation.operation_id
                        ),
                        known_place_id=value.place_id,
                    )
                )
    task_book = state.task_book
    return (
        CandidateRecallRequest(
            request_id=uuid5(NAMESPACE_URL, f"iter:restaurant-recall:{request_id}"),
            trip_id=state.trip_id,
            semantic_state_version=state.state_version,
            task_book_id=task_book.task_book_id if task_book is not None else None,
            task_book_revision=task_book.revision if task_book is not None else None,
            city_id=destination.city_id,
            start_date=date_range.start_date,
            end_date=date_range.end_date,
            theme_mode=(
                RecallThemeMode.SELECTED
                if themes
                else RecallThemeMode.OPEN_TO_ANY
                if open_to_any
                else RecallThemeMode.UNSPECIFIED
            ),
            themes=tuple(themes),
            free_text_clues=tuple(dict.fromkeys(free_text)),
            named_places=tuple(named),
            excluded_places=tuple(excluded),
            special_constraints=tuple(dict.fromkeys(special_constraints)),
            landmark_policy=LandmarkRecallPolicy.NEUTRAL,
            budget=CandidateRecallBudget(
                max_total_candidates=20,
                max_total_provider_calls=4,
                domains=(
                    DomainRecallBudget(
                        domain=CandidateDomain.RESTAURANT,
                        max_candidates=20,
                        max_provider_calls=4,
                    ),
                ),
            ),
        ),
        preferences,
    )


def _dining_theme_id(value: DiningPreferenceValue) -> str:
    assert value.value is not None
    digest = uuid5(NAMESPACE_URL, f"iter:dining-theme:{value.kind.value}:{value.value}").hex[:12]
    return f"dining:{value.kind.value}:{digest}"


def _dining_facts_from_recall(
    ranking: CandidateRankingResult,
    *,
    preferences: tuple[DiningPreferenceValue, ...],
    themes: tuple[RecallThemeInput, ...],
) -> tuple[tuple[RestaurantCandidateDiningFacts, ...], tuple[ExternalFactReference, ...]]:
    preference_by_theme = {
        _dining_theme_id(preference): preference
        for preference in preferences
        if preference.kind
        in {DiningPreferenceKind.CUISINE, DiningPreferenceKind.DIETARY_REQUIREMENT}
    }
    known_theme_ids = {theme.theme_id for theme in themes}
    if set(preference_by_theme) != known_theme_ids:
        raise ValueError("restaurant recall themes do not match active dining preferences")
    external_facts: dict[UUID, ExternalFactReference] = {}
    profiles: list[RestaurantCandidateDiningFacts] = []
    for ranked in ranking.candidates:
        candidate = ranked.candidate
        if candidate.domain is not CandidateDomain.RESTAURANT:
            continue
        source_fact_ids = _source_fact_ids(candidate, ranking.generated_at, external_facts)
        matched = [
            preference_by_theme[theme_id]
            for theme_id in candidate.theme_ids
            if theme_id in preference_by_theme
        ]
        candidate_text = _normalize(" ".join((candidate.place.name, *candidate.reasons)))
        ingredients_present: list[str] = []
        allergens_present: list[str] = []
        for preference in preferences:
            if preference.value is None:
                continue
            if _normalize(preference.value) not in candidate_text:
                continue
            if preference.kind is DiningPreferenceKind.ALLERGY:
                allergens_present.append(preference.value)
            elif preference.kind is DiningPreferenceKind.AVOIDANCE:
                ingredients_present.append(preference.value)
        if matched or ingredients_present or allergens_present:
            profiles.append(
                RestaurantCandidateDiningFacts(
                    candidate_id=candidate.candidate_id,
                    place_id=candidate.place.place_id,
                    cuisine_labels=tuple(
                        item.value or ""
                        for item in matched
                        if item.kind is DiningPreferenceKind.CUISINE
                    ),
                    dietary_labels=(),
                    search_match_labels=tuple(
                        item.value or ""
                        for item in matched
                        if item.kind is DiningPreferenceKind.DIETARY_REQUIREMENT
                    ),
                    ingredients_present=tuple(dict.fromkeys(ingredients_present)),
                    allergens_present=tuple(dict.fromkeys(allergens_present)),
                    source_fact_ids=source_fact_ids,
                )
            )
    return tuple(profiles), tuple(external_facts.values())


def _source_fact_ids(
    candidate: RecalledCandidate,
    generated_at: datetime,
    facts: dict[UUID, ExternalFactReference],
) -> tuple[UUID, ...]:
    fact_ids: list[UUID] = []
    for source in candidate.sources:
        provider = _source_provider(source.kind, source.provider)
        fact_id = uuid5(
            NAMESPACE_URL,
            "iter:v3-restaurant-fact:"
            f"{provider.value}:{source.source_record_id}:{candidate.place.place_id}",
        )
        facts.setdefault(
            fact_id,
            ExternalFactReference(
                fact_id=fact_id,
                provider=provider,
                source_record_id=source.source_record_id,
                retrieved_at=source.fetched_at or generated_at,
            ),
        )
        fact_ids.append(fact_id)
    return tuple(fact_ids)


def _restaurant_match(
    item: RankedCandidate,
    profile: RestaurantCandidateDiningFacts | None,
    preferences: tuple[DiningPreferenceValue, ...],
) -> tuple[bool, str, str | None]:
    if not preferences or all(
        preference.kind is DiningPreferenceKind.OPEN_TO_ANY for preference in preferences
    ):
        return True, "符合开放推荐，保留真实来源较完整的本地餐厅", None
    candidate_text = _normalize(" ".join((item.candidate.place.name, *item.candidate.reasons)))
    matches: list[str] = []
    risks: list[str] = []
    for preference in preferences:
        value = preference.value or ""
        normalized = _normalize(value)
        if preference.kind is DiningPreferenceKind.CUISINE:
            cuisine_values = _normalized_set(profile.cuisine_labels if profile else ())
            if normalized not in cuisine_values and normalized not in candidate_text:
                return False, "", f"不匹配已选菜系：{value}"
            matches.append(f"匹配想尝试的{value}")
        elif preference.kind is DiningPreferenceKind.DIETARY_REQUIREMENT:
            dietary_values = _normalized_set(profile.dietary_labels if profile else ())
            search_values = _normalized_set(profile.search_match_labels if profile else ())
            if normalized in dietary_values:
                matches.append(f"已核实支持{value}")
            elif normalized in search_values:
                matches.append(f"来源检索匹配{value}方向")
                risks.append(f"具体菜单是否满足{value}仍需向餐厅确认")
            else:
                return False, "", f"无法核实满足饮食要求：{value}"
        elif preference.kind is DiningPreferenceKind.ALLERGY:
            present = _normalized_set(profile.allergens_present if profile else ())
            absent = _normalized_set(profile.allergens_absent if profile else ())
            if normalized in present:
                return False, "", f"已知存在过敏原：{value}"
            if normalized not in absent:
                return False, "", f"无法核实不含过敏原：{value}"
            matches.append(f"已核实不含过敏原{value}")
        elif preference.kind is DiningPreferenceKind.AVOIDANCE:
            present = _normalized_set(
                (
                    *(profile.ingredients_present if profile else ()),
                    *(profile.allergens_present if profile else ()),
                )
            )
            absent = _normalized_set(
                (
                    *(profile.ingredients_absent if profile else ()),
                    *(profile.allergens_absent if profile else ()),
                )
            )
            if normalized in present or normalized in candidate_text:
                return False, "", f"与避免食材冲突：{value}"
            if normalized not in absent:
                risks.append(f"尚未核实是否含{value}")
            else:
                matches.append(f"已核实不含{value}")
        elif preference.kind is DiningPreferenceKind.SPECIFIC_RESTAURANT:
            same_place = preference.place_id == item.candidate.place.place_id
            same_name = normalized and normalized in _normalize(item.candidate.place.name)
            if same_place or (preference.place_id is None and same_name):
                matches.append("这是你主动点名的餐厅")
        elif preference.kind is DiningPreferenceKind.OPEN_TO_ANY:
            continue
    reason = "；".join(matches) or "符合当前餐饮方向和候选排序"
    return True, reason, "；".join(risks) or None


def _candidate_source_facts(
    item: RankedCandidate,
    ranking: CandidateRankingResult,
    facts: dict[UUID, ExternalFactReference],
) -> list[UUID]:
    result: list[UUID] = []
    for source in item.candidate.sources:
        provider = _source_provider(source.kind, source.provider)
        fact_id = uuid5(
            NAMESPACE_URL,
            "iter:v3-restaurant-fact:"
            f"{provider.value}:{source.source_record_id}:{item.candidate.place.place_id}",
        )
        facts.setdefault(
            fact_id,
            ExternalFactReference(
                fact_id=fact_id,
                provider=provider,
                source_record_id=source.source_record_id,
                retrieved_at=source.fetched_at or ranking.generated_at,
            ),
        )
        result.append(fact_id)
    return result


def _source_provider(kind: RecallSourceKind, provider: ProviderCode | None) -> ProviderCode:
    if kind is RecallSourceKind.PROVIDER:
        assert provider is not None
        return provider
    if kind is RecallSourceKind.CITY_CONTENT:
        return ProviderCode.CITY_CONTENT
    return ProviderCode.MANUAL


def _main_cost(item: RankedCandidate) -> str:
    return next(
        (
            explanation
            for explanation in item.explanations
            if "交通约" in explanation or "距离" in explanation or "绕行" in explanation
        ),
        "位置、绕行和排队代价暂无完整事实",
    )


def _restaurant_experience(
    reasons: tuple[str, ...], profile: RestaurantCandidateDiningFacts | None
) -> str:
    labels = (
        (*profile.cuisine_labels, *profile.dietary_labels, *profile.search_match_labels)
        if profile is not None
        else ()
    )
    return "、".join(labels[:3]) if labels else "；".join(reasons[:2])


def _empty_restaurant_reason(
    preferences: tuple[DiningPreferenceValue, ...], reasons: list[str]
) -> str:
    requirements = "、".join(
        item.value or "都可以"
        for item in preferences
        if item.kind is not DiningPreferenceKind.SPECIFIC_RESTAURANT
    )
    if requirements:
        return f"没有可核实满足“{requirements}”的餐厅；不会用冲突候选补足数量"
    if reasons:
        return reasons[0]
    return "当前没有可追溯且符合要求的餐厅候选"


def _normalized_set(values: tuple[str, ...]) -> set[str]:
    return {_normalize(value) for value in values}


def _normalize(value: str) -> str:
    return "".join(character for character in value.casefold() if not character.isspace())
