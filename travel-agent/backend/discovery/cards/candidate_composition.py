"""Compose V4 specific cards from the V3 live recall and ranking services."""

from __future__ import annotations

import math
import re
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from typing import Literal, cast
from uuid import NAMESPACE_URL, UUID, uuid5

from backend.agent.model_gateway import ModelCancellation
from backend.contracts.candidate_ranking import (
    CandidateRankingPreferences,
    CandidateRankingRequest,
    RankedCandidate,
)
from backend.contracts.candidate_recall import (
    AttractionDiscoveryContext,
    CandidateCompositionFeedback,
    CandidateDomain,
    CandidateRecallBudget,
    CandidateRecallRequest,
    CandidateRecallResult,
    DiscoveryPreferenceDirection,
    DomainRecallBudget,
    ExcludedPlaceClue,
    LandmarkRecallPolicy,
    RecallAttempt,
    RecallChannel,
    RecalledCandidate,
    RecallSourceKind,
    RecallThemeInput,
    RecallThemeMode,
    RecallThemeSource,
)
from backend.contracts.enums import DataAvailability, PlaceCategory, ProviderCode
from backend.contracts.v4.attraction_policy import ATTRACTION_TARGETS
from backend.contracts.v4.cards import (
    CardEntityRef,
    CardGenerationMetadata,
    CardOption,
    CardSemanticValue,
    EntitySemanticValue,
    SpecificCandidateCard,
)
from backend.contracts.v4.content_quality import normalized_visible_text
from backend.contracts.v4.dining_display import DINING_CITY_TARGETS, DiningDisplayFacts
from backend.contracts.v4.enums import (
    CardDomain,
    CardKind,
    CardStatus,
    CompositionRole,
    DiscoverySection,
)
from backend.contracts.v4.state import ConcreteIntentState, TripSemanticState
from backend.discovery.cards.attraction_media import AttractionMediaResolver
from backend.discovery.cards.attraction_selection import (
    AttractionCandidateSelector,
    AttractionSelection,
)
from backend.domain.discovery.cold_start import cold_start_default_notes
from backend.planning.candidate_ranking import CandidateRankingService
from backend.planning.candidate_recall import CandidateRecallService
from backend.planning.recall_plan import RecallPlanError
from backend.providers.place_taxonomy import category_from_original_typecodes, is_shopping_complex

_COUNTS = {
    CardDomain.ATTRACTION: {
        1: (4, 1, 5),
        2: (5, 2, 7),
        3: (7, 2, 9),
        4: (10, 2, 12),
        5: (12, 2, 14),
    },
    CardDomain.DINING: {
        1: (1, 2, 3),
        2: (2, 2, 4),
        3: (4, 4, 8),
        4: (5, 5, 10),
        5: (6, 6, 12),
    },
}
_INITIAL_RECALL_PROVIDER_CALLS = 6
_INITIAL_RECALL_CANDIDATES = 30
_SUPPLEMENTAL_RECALL_PROVIDER_CALLS = 2
_SUPPLEMENTAL_RECALL_CANDIDATES = 10
_ATTRACTION_SUPPLEMENTAL_CALLS = 6
_ATTRACTION_SUPPLEMENTAL_CANDIDATES = 30


class CardGenerationError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        code: str = "card_generation_failed",
        recoverable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.recoverable = recoverable


class CardGenerationContextRequired(CardGenerationError):
    """A safe, user-supplied context gap that Prepare may ask Qwen to resolve."""

    def __init__(self, *missing_fields: Literal["destination", "duration_days"]) -> None:
        if not missing_fields:
            raise ValueError("card context error requires at least one missing field")
        super().__init__(
            "specific card is missing user-supplied context",
            code="card_context_required",
            recoverable=True,
        )
        self.missing_fields = tuple(dict.fromkeys(missing_fields))


@dataclass(frozen=True, slots=True)
class CandidateCompositionResult:
    card: SpecificCandidateCard
    recall_request_id: UUID
    recall_request_ids: tuple[UUID, ...]
    provider_candidate_count: int
    provider_call_count: int
    supplemental_recall_used: bool
    selected_provider_candidates: tuple[RecalledCandidate, ...]
    recall_attempts: tuple[RecallAttempt, ...]
    attraction_selection: AttractionSelection | None = None
    reconsideration_used: bool = False


class CandidateCompositionService:
    """Qwen chooses recall angles; real Providers and code own selectable entities."""

    def __init__(
        self,
        *,
        recall: CandidateRecallService,
        ranking: CandidateRankingService,
        attraction_selector: AttractionCandidateSelector | None = None,
        attraction_media: AttractionMediaResolver | None = None,
    ) -> None:
        self._recall = recall
        self._ranking = ranking
        self._attraction_selector = attraction_selector
        self._attraction_media = attraction_media

    async def compose(
        self,
        state: TripSemanticState,
        *,
        domain: CardDomain,
        interaction_id: UUID,
        attachment_id: UUID,
        cancellation: ModelCancellation | None = None,
    ) -> CandidateCompositionResult:
        if domain not in {CardDomain.ATTRACTION, CardDomain.DINING}:
            raise CardGenerationError("specific cards only support attractions and dining")
        request = _recall_request(
            state,
            domain,
            interaction_id,
            city_led=domain is CardDomain.ATTRACTION and self._attraction_selector is not None,
        )
        if request.attraction_discovery is not None:
            return await self._compose_attractions(
                state,
                request,
                interaction_id=interaction_id,
                attachment_id=attachment_id,
                cancellation=cancellation,
            )
        initial_request = _with_recall_budget(
            request,
            provider_calls=_INITIAL_RECALL_PROVIDER_CALLS,
            candidates=_INITIAL_RECALL_CANDIDATES,
        )
        result = await self._recall.recall(initial_request, cancellation=cancellation)
        recall_request_ids = [initial_request.request_id]
        provider_candidates, live_result = _select_live_provider_candidates(
            state,
            domain,
            result,
        )
        options, complete, frozen_top = self._rank_options(
            request,
            live_result,
            domain=domain,
            interaction_id=interaction_id,
            city_name=state.trip_basics.destination_name,
        )

        # V4 strategy allows exactly one observation-driven supplemental plan.
        # It reuses only the remaining 2 calls / 10 candidates from the original
        # 8 / 40 budget. Already displayed entities remain excluded; unused,
        # verified alternatives are observations for Qwen's next search plan.
        if not complete:
            supplemental_request = _supplemental_recall_request(
                request,
                live_result,
                interaction_id=interaction_id,
                options=options,
                domain=domain,
                city_name=state.trip_basics.destination_name,
            )
            recall_request_ids.append(supplemental_request.request_id)
            try:
                supplemental_result = await self._recall.recall(
                    supplemental_request,
                    cancellation=cancellation,
                )
            except RecallPlanError:
                # Optional recall is not a transaction over the already ranked
                # live results. A rejected supplement must not erase this card.
                pass
            else:
                merged = _merge_recall_results(request, result, supplemental_result)
                new_candidates, new_live_result = _select_live_provider_candidates(
                    state, domain, merged
                )
                new_options, complete, _ = self._rank_options(
                    request,
                    new_live_result,
                    domain=domain,
                    interaction_id=interaction_id,
                    city_name=state.trip_basics.destination_name,
                    frozen_top=frozen_top,
                )
                result, provider_candidates, options = merged, new_candidates, new_options

        if not provider_candidates:
            raise CardGenerationError(
                "live Provider did not return a selectable candidate",
                code="candidate_provider_unavailable",
                recoverable=True,
            )
        if not options:
            raise CardGenerationError(
                "ranking did not retain a selectable candidate",
                code="candidate_selection_empty",
                recoverable=True,
            )
        source_refs = list(
            dict.fromkeys(source for option in options for source in option.source_refs)
        )
        card_domain = cast(
            Literal[CardDomain.ATTRACTION, CardDomain.DINING],
            domain,
        )
        card = SpecificCandidateCard(
            attachment_id=str(attachment_id),
            interaction_id=str(interaction_id),
            kind=CardKind.SPECIFIC_CARD,
            domain=card_domain,
            section=(
                DiscoverySection.ATTRACTION_SPECIFIC
                if domain is CardDomain.ATTRACTION
                else DiscoverySection.DINING_SPECIFIC
            ),
            based_on_state_version=state.state_version,
            dependency_fingerprint=specific_dependency_fingerprint(state, domain),
            status=CardStatus.ACTIVE if complete else CardStatus.PARTIAL_AVAILABILITY,
            options=options,
            control_actions=[],
            generation_metadata=CardGenerationMetadata(
                generated_at=result.generated_at,
                source_refs=source_refs,
                model_plan_id=str(request.request_id),
                generation_mode="provider_composed",
                strategy_version="dining-v2" if domain is CardDomain.DINING else "legacy",
            ),
            prompt=(
                "选一选你想去的地方。"
                if domain is CardDomain.ATTRACTION
                else "看看哪些店让你想尝一尝？"
            ),
            duration_days=request.day_count,
        )
        return CandidateCompositionResult(
            card=card,
            recall_request_id=request.request_id,
            recall_request_ids=tuple(recall_request_ids),
            provider_candidate_count=len(provider_candidates),
            provider_call_count=result.provider_call_count,
            supplemental_recall_used=len(recall_request_ids) == 2,
            selected_provider_candidates=tuple(
                item
                for item in provider_candidates
                if str(item.place.place_id)
                in {
                    option.entity_ref.canonical_entity_id
                    for option in options
                    if option.entity_ref is not None
                }
            ),
            recall_attempts=result.attempts,
        )

    async def _compose_attractions(
        self,
        state: TripSemanticState,
        request: CandidateRecallRequest,
        *,
        interaction_id: UUID,
        attachment_id: UUID,
        cancellation: ModelCancellation | None,
    ) -> CandidateCompositionResult:
        assert self._attraction_selector is not None
        assert request.attraction_discovery is not None
        initial = _with_recall_budget(
            request,
            provider_calls=_INITIAL_RECALL_PROVIDER_CALLS,
            candidates=_INITIAL_RECALL_CANDIDATES,
        )
        try:
            result = await self._recall.recall(initial, cancellation=cancellation)
        except RecallPlanError as error:
            raise CardGenerationError(
                "attraction recall plan did not complete",
                code="candidate_recall_unavailable",
                recoverable=True,
            ) from error
        request_ids = [initial.request_id]
        candidates, _ = _select_live_provider_candidates(
            state,
            CardDomain.ATTRACTION,
            result,
            semantic_selection=True,
        )
        selection = await self._attraction_selector.select(
            request,
            candidates,
            cancellation=cancellation,
        )
        target = request.attraction_discovery.minimum_target
        if len(selection.selected) < target:
            supplemental = _attraction_supplement_request(request, candidates, selection, result)
            request_ids.append(supplemental.request_id)
            try:
                extra = await self._recall.recall(supplemental, cancellation=cancellation)
                result = _merge_recall_results(request, result, extra)
                candidates, _ = _select_live_provider_candidates(
                    state,
                    CardDomain.ATTRACTION,
                    result,
                    semantic_selection=True,
                )
                # Reconsider the merged pool, including the first selected set.
                # Never append an unscreened second batch to the visible card.
                selection = await self._attraction_selector.select(
                    request,
                    candidates,
                    cancellation=cancellation,
                    previous_selection=selection,
                )
            except (RecallPlanError, CardGenerationError):
                # Keep the first screened result; still allow a bounded look
                # back over verified candidates, even if supplementation fails.
                pass
        reconsideration_used = len(selection.selected) < target and bool(candidates)
        if reconsideration_used:
            # A failed look-back cannot append unprocessed records.
            with suppress(CardGenerationError):
                selection = await self._attraction_selector.reconsider(
                    request, candidates, selection, cancellation=cancellation
                )
        if not selection.selected:
            raise CardGenerationError(
                "no attraction remained after semantic selection",
                code="candidate_selection_empty",
                recoverable=True,
            )
        by_key = {str(item.place.place_id): item for item in candidates}
        selected_candidates = tuple(by_key[item.candidate_key] for item in selection.selected)
        options = [
            _entity_option(
                by_key[item.candidate_key],
                domain=CardDomain.ATTRACTION,
                interaction_id=interaction_id,
                role=item.composition_role,
                description=item.experience,
            )
            for item in selection.selected
        ]
        if self._attraction_media is not None:
            options = await self._attraction_media.enrich(
                options,
                request.city_id,
                cancellation=cancellation,
            )
        card = SpecificCandidateCard(
            attachment_id=str(attachment_id),
            interaction_id=str(interaction_id),
            kind=CardKind.SPECIFIC_CARD,
            domain=CardDomain.ATTRACTION,
            section=DiscoverySection.ATTRACTION_SPECIFIC,
            based_on_state_version=state.state_version,
            dependency_fingerprint=specific_dependency_fingerprint(state, CardDomain.ATTRACTION),
            status=(
                CardStatus.ACTIVE if len(options) >= target else CardStatus.PARTIAL_AVAILABILITY
            ),
            options=options,
            control_actions=[],
            generation_metadata=CardGenerationMetadata(
                generated_at=result.generated_at,
                source_refs=list(
                    dict.fromkeys(ref for item in options for ref in item.source_refs)
                ),
                model_plan_id=str(request.request_id),
                generation_mode="provider_composed",
                strategy_version="attraction-v2",
            ),
            prompt="选一选你想去的地方。",
            duration_days=request.day_count,
        )
        return CandidateCompositionResult(
            card=card,
            recall_request_id=request.request_id,
            recall_request_ids=tuple(request_ids),
            provider_candidate_count=len(candidates),
            provider_call_count=result.provider_call_count,
            supplemental_recall_used=len(request_ids) == 2,
            selected_provider_candidates=selected_candidates,
            recall_attempts=result.attempts,
            attraction_selection=selection,
            reconsideration_used=reconsideration_used,
        )

    def _rank_options(
        self,
        request: CandidateRecallRequest,
        result: CandidateRecallResult,
        *,
        domain: CardDomain,
        interaction_id: UUID,
        city_name: str | None = None,
        frozen_top: tuple[RankedCandidate, ...] = (),
    ) -> tuple[list[CardOption], bool, tuple[RankedCandidate, ...]]:
        _, _, total = _COUNTS[domain][request.day_count]
        ranking = self._ranking.rank(
            CandidateRankingRequest(
                ranking_request_id=uuid5(
                    NAMESPACE_URL,
                    f"v4-card-ranking:{interaction_id}:{domain.value}",
                ),
                recall_request=request,
                recall_result=result,
                preferences=CandidateRankingPreferences(selected_theme_ids=()),
                recommendation_limit=total,
            )
        )
        options, complete = _compose_options(
            ranking.candidates,
            domain=domain,
            duration_days=request.day_count,
            interaction_id=interaction_id,
            city_name=city_name,
            frozen_top=frozen_top,
        )
        candidates_by_id = {
            str(item.candidate.place.place_id): item for item in (*ranking.candidates, *frozen_top)
        }
        top = tuple(
            candidates_by_id[option.entity_ref.canonical_entity_id]
            for option in options
            if option.composition_role is CompositionRole.PERSONALIZED_TOP
            and option.entity_ref is not None
        )
        return options, complete, top


def _with_recall_budget(
    request: CandidateRecallRequest,
    *,
    provider_calls: int,
    candidates: int,
) -> CandidateRecallRequest:
    if len(request.budget.domains) != 1:
        raise CardGenerationError("specific card recall requires exactly one domain")
    domain = request.budget.domains[0].domain
    updated = request.model_copy(
        update={
            "budget": CandidateRecallBudget(
                max_total_candidates=candidates,
                max_total_provider_calls=provider_calls,
                domains=(
                    DomainRecallBudget(
                        domain=domain,
                        max_candidates=candidates,
                        max_provider_calls=provider_calls,
                    ),
                ),
            )
        },
        deep=True,
    )
    return CandidateRecallRequest.model_validate(updated.model_dump(mode="json"))


def _attraction_supplement_request(
    request: CandidateRecallRequest,
    candidates: tuple[RecalledCandidate, ...],
    selection: AttractionSelection,
    observed: CandidateRecallResult,
) -> CandidateRecallRequest:
    assert request.attraction_discovery is not None
    by_key = {str(item.place.place_id): item.place.name for item in candidates}
    feedback = tuple(
        [
            *selection.search_feedback,
            *(f"已保留，不重复搜索：{by_key[item.candidate_key]}" for item in selection.selected),
            *(
                f"上轮未选，仅作搜索反馈：{by_key[item.candidate_key]}；{item.reason}"
                for item in selection.rejected
            ),
        ][:40]
    )
    supplemental = _with_recall_budget(
        request,
        provider_calls=_ATTRACTION_SUPPLEMENTAL_CALLS,
        candidates=_ATTRACTION_SUPPLEMENTAL_CANDIDATES,
    ).model_copy(
        update={
            "request_id": uuid5(NAMESPACE_URL, f"{request.request_id}:supplement"),
            "attraction_discovery": request.attraction_discovery.model_copy(
                update={"screening_feedback": feedback},
            ),
            "composition_feedback": CandidateCompositionFeedback(
                domain=CandidateDomain.ATTRACTION,
                missing_personalized_count=max(
                    0,
                    request.attraction_discovery.personalized_target
                    - sum(
                        item.composition_role is CompositionRole.PERSONALIZED_TOP
                        for item in selection.selected
                    ),
                ),
                missing_representative_count=max(
                    0,
                    request.attraction_discovery.city_target
                    - sum(
                        item.composition_role is CompositionRole.REPRESENTATIVE_EXTRA
                        for item in selection.selected
                    ),
                ),
                failed_named_queries=tuple(
                    dict.fromkeys(
                        item.query_keyword
                        for item in observed.attempts
                        if item.exact_name_match and item.accepted_count == 0 and item.query_keyword
                    )
                )[:8],
            ),
        },
        deep=True,
    )
    return CandidateRecallRequest.model_validate(supplemental.model_dump(mode="json"))


def _supplemental_recall_request(
    request: CandidateRecallRequest,
    observed: CandidateRecallResult,
    *,
    interaction_id: UUID,
    options: list[CardOption],
    domain: CardDomain,
    city_name: str | None = None,
) -> CandidateRecallRequest:
    supplemental = _with_recall_budget(
        request,
        provider_calls=_SUPPLEMENTAL_RECALL_PROVIDER_CALLS,
        candidates=_SUPPLEMENTAL_RECALL_CANDIDATES,
    )
    displayed_ids = {
        option.entity_ref.canonical_entity_id for option in options if option.entity_ref is not None
    }
    displayed = tuple(
        candidate
        for candidate in observed.candidates
        if str(candidate.place.place_id) in displayed_ids
    )
    alternatives: list[RecalledCandidate] = []
    for candidate in observed.candidates:
        if not any(
            _same_recalled_entity(candidate, kept, domain=domain, city_name=city_name)
            for kept in (*displayed, *alternatives)
        ):
            alternatives.append(candidate)
    candidate_domain = request.budget.domains[0].domain
    top_count, representative_count, _ = _COUNTS[domain][request.day_count]
    feedback = CandidateCompositionFeedback(
        domain=candidate_domain,
        missing_personalized_count=max(
            0,
            top_count
            - sum(item.composition_role is CompositionRole.PERSONALIZED_TOP for item in options),
        ),
        missing_representative_count=max(
            0,
            representative_count
            - sum(
                item.composition_role is CompositionRole.REPRESENTATIVE_EXTRA for item in options
            ),
        ),
        frozen_top_names=tuple(
            item.label
            for item in options
            if item.composition_role is CompositionRole.PERSONALIZED_TOP
        ),
        observed_alternative_names=tuple(item.place.name for item in alternatives),
        excluded_restaurant_brands=(
            tuple(
                dict.fromkeys(
                    _restaurant_brand_key(item.place.name, city_name=city_name)
                    for item in displayed
                    if item.domain is CandidateDomain.RESTAURANT
                )
            )
            if domain is CardDomain.DINING
            else ()
        ),
        failed_named_queries=tuple(
            dict.fromkeys(
                item.query_keyword
                for item in observed.attempts
                if item.exact_name_match and item.accepted_count == 0 and item.query_keyword
            )
        ),
    )
    supplemental_exclusions = tuple(
        ExcludedPlaceClue(
            clue_id=uuid5(
                NAMESPACE_URL,
                f"v4-card-supplement-exclusion:{request.request_id}:{candidate.place.place_id}",
            ),
            name=candidate.place.name,
            source_message_id=interaction_id,
            domain=candidate_domain,
            known_place_id=candidate.place.place_id,
        )
        for candidate in displayed
    )
    updated = supplemental.model_copy(
        update={
            "request_id": uuid5(
                NAMESPACE_URL,
                f"v4-card-supplement:{request.request_id}",
            ),
            "excluded_places": tuple(
                dict.fromkeys((*request.excluded_places, *supplemental_exclusions))
            ),
            "composition_feedback": feedback,
        },
        deep=True,
    )
    return CandidateRecallRequest.model_validate(updated.model_dump(mode="json"))


def _merge_recall_results(
    request: CandidateRecallRequest,
    initial: CandidateRecallResult,
    supplemental: CandidateRecallResult,
) -> CandidateRecallResult:
    # One entity may be observed through several query channels. Preserve these
    # records until ranking merges their source and representative evidence.
    # Ignore ordinary repeated results, but retain a genuinely new named-search
    # verification for an unused entity that was only an exploration candidate.
    initial_place_ids = {item.place.place_id for item in initial.candidates}
    verified_ids = {
        item.place.place_id for item in initial.candidates if item.representative_identity_verified
    }
    candidates = (
        *initial.candidates,
        *(
            item
            for item in supplemental.candidates
            if item.place.place_id not in initial_place_ids
            or (item.representative_identity_verified and item.place.place_id not in verified_ids)
        ),
    )
    if len(candidates) > request.budget.max_total_candidates:
        raise CardGenerationError("supplemental recall exceeded the candidate budget")
    provider_call_count = initial.provider_call_count + supplemental.provider_call_count
    if provider_call_count > request.budget.max_total_provider_calls:
        raise CardGenerationError("supplemental recall exceeded the Provider-call budget")
    degradation = tuple(
        dict.fromkeys((*initial.degradation_reasons, *supplemental.degradation_reasons))
    )
    if not candidates:
        status = DataAvailability.MISSING
        if not degradation:
            degradation = ("no verified candidates were returned after supplemental recall",)
    elif (
        initial.status is DataAvailability.AVAILABLE
        and supplemental.status is DataAvailability.AVAILABLE
        and not degradation
    ):
        status = DataAvailability.AVAILABLE
    else:
        status = DataAvailability.PARTIAL
    return CandidateRecallResult(
        result_version=initial.result_version,
        request_id=request.request_id,
        trip_id=request.trip_id,
        semantic_state_version=request.semantic_state_version,
        task_book_id=request.task_book_id,
        task_book_revision=request.task_book_revision,
        city_id=request.city_id,
        status=status,
        candidates=candidates,
        attempts=(*initial.attempts, *supplemental.attempts),
        provider_call_count=provider_call_count,
        content_package_version=(
            initial.content_package_version or supplemental.content_package_version
        ),
        degradation_reasons=degradation,
        generated_at=max(initial.generated_at, supplemental.generated_at),
    )


def _select_live_provider_candidates(
    state: TripSemanticState,
    domain: CardDomain,
    result: CandidateRecallResult,
    *,
    semantic_selection: bool = False,
) -> tuple[tuple[RecalledCandidate, ...], CandidateRecallResult]:
    provider_candidates = tuple(
        item
        for item in result.candidates
        if any(source.kind is RecallSourceKind.PROVIDER for source in item.sources)
        and _candidate_category_matches(domain, item.place.category)
        and (
            _provider_type_matches(domain, item.place.provider_typecode)
            or (semantic_selection and is_shopping_complex(item.place.provider_typecode))
        )
        and (semantic_selection or not _hard_filtered(state, domain, item.place.name))
        and (
            semantic_selection
            or domain is not CardDomain.ATTRACTION
            or independent_attraction_issue(
                item.place.name,
                item.place.address,
                item.place.provider_parent_place_id,
            )
            is None
        )
    )
    removed = len(provider_candidates) != len(result.candidates)
    degradation = tuple(
        dict.fromkeys(
            (
                *result.degradation_reasons,
                *(("non-provider or hard-conflict candidates were removed",) if removed else ()),
                *(
                    ("live Provider did not return a selectable candidate",)
                    if not provider_candidates
                    else ()
                ),
            )
        )
    )
    if not provider_candidates:
        status = DataAvailability.MISSING
    elif not removed and result.status is DataAvailability.AVAILABLE:
        status = DataAvailability.AVAILABLE
    else:
        status = DataAvailability.PARTIAL
    live_result = result.model_copy(
        update={
            "candidates": provider_candidates,
            "status": status,
            "degradation_reasons": degradation,
        }
    )
    # Revalidate after the deterministic filter so ranking cannot accidentally
    # consume city-content-only or hard-conflicting candidates.
    return provider_candidates, CandidateRecallResult.model_validate(
        live_result.model_dump(mode="json")
    )


def _recall_request(
    state: TripSemanticState,
    domain: CardDomain,
    interaction_id: UUID,
    *,
    city_led: bool = False,
) -> CandidateRecallRequest:
    basics = state.trip_basics
    missing: list[Literal["destination", "duration_days"]] = []
    if basics.destination_canonical_id is None:
        missing.append("destination")
    if basics.duration_days is None:
        missing.append("duration_days")
    if missing:
        raise CardGenerationContextRequired(*missing)
    assert basics.destination_canonical_id is not None
    assert basics.duration_days is not None
    candidate_domain = (
        CandidateDomain.ATTRACTION
        if domain is CardDomain.ATTRACTION
        else CandidateDomain.RESTAURANT
    )
    preferences = (
        state.attractions.preference_directions
        if domain is CardDomain.ATTRACTION
        else state.dining.preference_directions
    )
    exclusions: list[ConcreteIntentState] = (
        state.attractions.exclusions if domain is CardDomain.ATTRACTION else state.dining.exclusions
    )
    selected_themes = tuple(
        RecallThemeInput(
            theme_id=item.direction_id,
            label=item.label,
            source=RecallThemeSource.CURRENT_TRIP,
            source_reference_ids=tuple(item.source_operation_refs),
        )
        for item in preferences
        if item.selected
    )
    target = ATTRACTION_TARGETS[basics.duration_days]
    attraction_context = (
        AttractionDiscoveryContext(
            destination_name=basics.destination_name,
            directions=tuple(
                DiscoveryPreferenceDirection(
                    direction_id=item.direction_id,
                    label=item.label,
                    description=item.description,
                    tags=tuple(item.tags),
                    search_query=item.search_query,
                    selected=item.selected,
                    source_reference_ids=tuple(item.source_operation_refs),
                )
                for item in preferences
            ),
            explicit_place_intents=tuple(
                f"{item.disposition}: {item.display_name}"
                for item in state.attractions.concrete_intents
            ),
            travelers=tuple(basics.travelers),
            trip_goals=tuple(basics.trip_goals),
            cold_start_defaults=tuple(
                item.value for item in cold_start_default_notes(state.cold_start_profile_snapshot)
            ),
            pace_preferences=tuple(state.transport_and_pace.pace_preferences),
            transport_preferences=tuple(state.transport_and_pace.transport_preferences),
            minimum_target=target.minimum_target,
            maximum_target=target.maximum,
            city_target=target.city_target,
            personalized_target=target.personalized_target,
        )
        if city_led
        else None
    )
    return CandidateRecallRequest(
        request_id=uuid5(
            NAMESPACE_URL,
            f"v4-card-recall:{interaction_id}:{domain.value}",
        ),
        trip_id=UUID(state.trip_id),
        semantic_state_version=state.state_version,
        city_id=basics.destination_canonical_id,
        start_date=basics.start_date,
        end_date=basics.end_date,
        duration_days=basics.duration_days,
        theme_mode=RecallThemeMode.SELECTED if selected_themes else RecallThemeMode.UNSPECIFIED,
        themes=selected_themes,
        attraction_discovery=attraction_context,
        dining_city_target=(
            DINING_CITY_TARGETS[basics.duration_days] if domain is CardDomain.DINING else None
        ),
        personal_preference_clues=tuple(item.label for item in preferences if item.selected),
        special_constraints=tuple(
            state.constraints
            if domain is CardDomain.ATTRACTION
            else [
                *state.dining.requirements,
                *state.dining.allergies,
                *state.dining.avoidances,
            ]
        ),
        excluded_places=tuple(
            ExcludedPlaceClue(
                clue_id=uuid5(
                    NAMESPACE_URL,
                    f"v4-exclusion:{item.canonical_entity_id}",
                ),
                name=item.display_name,
                source_message_id=interaction_id,
                domain=candidate_domain,
                known_place_id=_uuid_or_none(item.canonical_entity_id),
            )
            for item in exclusions
        ),
        landmark_policy=LandmarkRecallPolicy.INCLUDE,
        budget=CandidateRecallBudget(
            max_total_candidates=60 if city_led else 40,
            max_total_provider_calls=12 if city_led else 8,
            domains=(
                DomainRecallBudget(
                    domain=candidate_domain,
                    max_candidates=60 if city_led else 40,
                    max_provider_calls=12 if city_led else 8,
                ),
            ),
        ),
    )


def _compose_options(
    ranked: tuple[RankedCandidate, ...],
    *,
    domain: CardDomain,
    duration_days: int,
    interaction_id: UUID,
    city_name: str | None = None,
    frozen_top: tuple[RankedCandidate, ...] = (),
) -> tuple[list[CardOption], bool]:
    ranked = _deduplicate_card_candidates(ranked, domain=domain, city_name=city_name)
    top_count, representative_count, total = _COUNTS[domain][duration_days]
    if domain is CardDomain.DINING:
        # Reserve verified city places first. A representative already near the
        # top is still a city option, not consumed by the old frozen TOP quota.
        city = [item for item in ranked if _is_representative(item)][:representative_count]
        city_ids = {item.candidate.place.place_id for item in city}
        personalized = [item for item in ranked if item.candidate.place.place_id not in city_ids][
            : total - len(city)
        ]
        options = [
            _option(
                item,
                domain=domain,
                interaction_id=interaction_id,
                role=(
                    CompositionRole.REPRESENTATIVE_EXTRA
                    if item.candidate.place.place_id in city_ids
                    else CompositionRole.PERSONALIZED_TOP
                ),
            )
            for item in (*city, *personalized)
        ]
        return options, len(options) == total and len(city) == representative_count
    # Freeze the personalized TOP before selecting extras. A representative that
    # naturally ranks inside TOP cannot consume the separate extra quota.
    top = list(frozen_top[:top_count])
    for item in ranked:
        if len(top) >= top_count:
            break
        if not any(
            _same_visible_entity(item, kept, domain=domain, city_name=city_name) for kept in top
        ):
            top.append(item)
    representatives = [
        item
        for item in ranked
        if _is_representative(item)
        and not any(
            _same_visible_entity(item, kept, domain=domain, city_name=city_name) for kept in top
        )
    ]
    extras = representatives[:representative_count]
    extra_ids = {item.candidate.place.place_id for item in extras}
    selected = [*top, *extras]
    options = [
        _option(
            item,
            domain=domain,
            interaction_id=interaction_id,
            role=(
                CompositionRole.REPRESENTATIVE_EXTRA
                if item.candidate.place.place_id in extra_ids
                else CompositionRole.PERSONALIZED_TOP
            ),
        )
        for item in selected
    ]
    complete = len(options) == total and len(extras) == representative_count
    return options, complete


def _deduplicate_card_candidates(
    ranked: tuple[RankedCandidate, ...],
    *,
    domain: CardDomain,
    city_name: str | None = None,
) -> tuple[RankedCandidate, ...]:
    """Keep one visible entity per provider identity, attraction, or restaurant brand."""

    unique: list[RankedCandidate] = []
    for item in ranked:
        duplicate_index = next(
            (
                index
                for index, kept in enumerate(unique)
                if _same_visible_entity(item, kept, domain=domain, city_name=city_name)
            ),
            None,
        )
        if duplicate_index is not None:
            kept = unique[duplicate_index]
            # Retain the actually name-verified entity, not an unverified branch
            # of the same brand. Never transfer its evidence to another POI ID.
            if _is_representative(item) and not _is_representative(kept):
                unique[duplicate_index] = item
            continue
        unique.append(item)
    return tuple(unique)


def _same_visible_entity(
    left: RankedCandidate,
    right: RankedCandidate,
    *,
    domain: CardDomain,
    city_name: str | None = None,
) -> bool:
    return _same_recalled_entity(
        left.candidate, right.candidate, domain=domain, city_name=city_name
    )


def _same_recalled_entity(
    left_candidate: RecalledCandidate,
    right_candidate: RecalledCandidate,
    *,
    domain: CardDomain,
    city_name: str | None = None,
) -> bool:
    if left_candidate.place.place_id == right_candidate.place.place_id:
        return True
    left_provider_refs = {
        (source.provider, source.source_place_id)
        for source in left_candidate.sources
        if source.kind is RecallSourceKind.PROVIDER
    }
    right_provider_refs = {
        (source.provider, source.source_place_id)
        for source in right_candidate.sources
        if source.kind is RecallSourceKind.PROVIDER
    }
    if left_provider_refs & right_provider_refs:
        return True
    if domain is CardDomain.ATTRACTION:
        left_parent = left_candidate.place.provider_parent_place_id
        right_parent = right_candidate.place.provider_parent_place_id
        if (left_parent and left_parent in {ref[1] for ref in right_provider_refs}) or (
            right_parent and right_parent in {ref[1] for ref in left_provider_refs}
        ):
            return True
    left_name = normalized_visible_text(left_candidate.place.name)
    right_name = normalized_visible_text(right_candidate.place.name)
    if left_name == right_name:
        return True
    if domain is CardDomain.DINING:
        return _restaurant_brand_key(
            left_candidate.place.name, city_name=city_name
        ) == _restaurant_brand_key(right_candidate.place.name, city_name=city_name)
    left_root = _attraction_identity_root(left_candidate.place.name, city_name=city_name)
    right_root = _attraction_identity_root(right_candidate.place.name, city_name=city_name)
    if left_root == right_root:
        return True
    # A parent site's named sub-area is not an additional independent choice,
    # even when its entrance coordinates are more than 500 metres apart.
    if min(len(left_name), len(right_name)) >= 3 and (
        left_name in right_name or right_name in left_name
    ):
        return True
    generic_roots = {"博物馆", "博物院", "美术馆", "科技馆", "动物园", "植物园"}
    if (
        min(len(left_root), len(right_root)) >= 3
        and left_root not in generic_roots
        and right_root not in generic_roots
        and (left_root in right_root or right_root in left_root)
    ):
        return True
    distance = _coordinate_distance_m(
        left_candidate.place.coordinates,
        right_candidate.place.coordinates,
    )
    return bool(
        distance is not None
        and distance <= 500
        and min(len(left_root), len(right_root)) >= 3
        and (
            left_root in right_root
            or right_root in left_root
            or SequenceMatcher(None, left_root, right_root).ratio() >= 0.76
        )
    )


def _restaurant_brand_key(value: str, *, city_name: str | None = None) -> str:
    # Provider names mix branch addresses, cuisine taglines and city prefixes.
    # All belong to one visible brand, even when a suffix does not say "店".
    without_branch = re.sub(r"[（(][^）)]*[）)]", "", value.casefold().strip())
    without_branch = re.sub(r"^(?:清真|老字号)[·•・]", "", without_branch)
    without_branch = re.split(r"[·•・]", without_branch, maxsplit=1)[0]
    without_branch = re.sub(
        r"(?:旗舰店|总店|分店|直营店|概念店|体验店|门店|店)$",
        "",
        without_branch,
        flags=re.IGNORECASE,
    )
    normalized = normalized_visible_text(without_branch)
    city = normalized_visible_text(city_name or "").removesuffix("市")
    if city and normalized.startswith(city) and len(normalized) > len(city) + 1:
        normalized = normalized[len(city) :].removeprefix("市")
    return normalized


def _attraction_identity_root(value: str, *, city_name: str | None = None) -> str:
    without_section = re.sub(r"[（(][^）)]*[）)]", "", value)
    normalized = normalized_visible_text(without_section)
    city = normalized_visible_text(city_name or "").removesuffix("市")
    if city and normalized.startswith(city):
        normalized = normalized[len(city) :].removeprefix("市")
    if "城墙" in normalized:
        # A city's named gates/segments share the same linear attraction.
        # Removing the short city prefix must not make that family invisible
        # to the length-based near-duplicate guards below.
        return normalized.split("城墙", maxsplit=1)[0] + "城墙"
    # A main site and its named internal sections remain one visible choice.
    if "景区" in normalized:
        normalized = normalized.split("景区", maxsplit=1)[0]
    return re.sub(
        r"(?:国家)?(?:风景名胜区|风景区|旅游区|景区|公园)$",
        "",
        normalized,
    )


def _provider_type_matches(domain: CardDomain, typecode: str | None) -> bool:
    return _candidate_category_matches(domain, category_from_original_typecodes(typecode))


def independent_attraction_issue(
    name: str,
    address: str | None,
    parent_place_id: str | None = None,
) -> str | None:
    """Conservative independent-visit guard, even when a Provider calls it scenic."""

    museum = r"(?:博物馆|博物院|美术馆|科技馆|纪念馆|展览馆)"
    normalized_name = name.strip()
    site_parts = re.split(r"[-—－]", normalized_name, maxsplit=1)
    if len(site_parts) == 2 and (
        parent_place_id or re.search(r"(?:景区|公园|博物馆|博物院|寺|宫|陵|园)$", site_parts[0])
    ):
        return "site_internal_facility"
    if re.search(
        r"(?:府|寺|宫|园|陵|馆)(?:前|后|东|西|南|北|外|内|西侧|东侧)?广场$", normalized_name
    ):
        return "site_internal_facility"
    if re.search(r"(?:胡同|路|街)\d+号(?:四合院|院落)$", normalized_name):
        return "unverified_residential_site"
    if re.fullmatch(r"(?:传统|老北京)?四合院", normalized_name):
        return "unverified_residential_site"
    if re.search(r"(?:校园|校区|大学).*(?:绿化|休闲区|草坪)", normalized_name):
        return "site_internal_facility"
    if re.search(
        r"(?:景区|园区|艺术区|公园|博物馆)[A-Z0-9一二三四五六七八九十东南西北]+区$", normalized_name
    ):
        return "site_internal_facility"
    if parent_place_id and re.search(r"(?:石|亭|碑|廊|殿)$", normalized_name):
        return "site_internal_facility"
    if (
        parent_place_id
        and address
        and "故居" in address
        and not re.search(r"故居|博物馆|纪念馆", normalized_name)
    ):
        return "site_internal_facility"
    if re.search(
        r"(?:出入口|入口|出口|售票处|检票口|停车场|游客中心|会客厅)[）)]?$", normalized_name
    ):
        return "site_internal_facility"
    if re.search(
        r"[-—－·](?:.*)(?:观光台|观景台|桥|展区|展厅|陈列室|办公室|会客厅)$", normalized_name
    ):
        return "site_internal_facility"
    if address and re.search(museum, address) and not re.search(museum + r"$", normalized_name):
        return "internal_museum_exhibit"
    if re.search(
        r"(?:展厅|展项|陈列室|临湖厅|秘书处|办公厅|服务部|售票厅|会客厅)$", normalized_name
    ):
        return "site_internal_facility"
    if re.search(r"(?:分会场|展位)$", normalized_name):
        return "temporary_event_venue_not_verified"
    if parent_place_id and re.search(
        r"(?:碑亭|祭堂|南广场|北广场|东广场|西广场|入口|出口|打卡点[）)]?)$",
        normalized_name,
    ):
        return "site_internal_facility"
    if parent_place_id:
        # A child POI marked "scenic" can still be a room/object in a larger
        # attraction. Require a whole-venue identity, not just a scenic type.
        # Parentage alone is not disqualifying: museums inside malls and
        # independently visitable venues inside large scenic areas remain legal.
        venue_name = re.sub(r"[（(][^）)]*[）)]", "", normalized_name).strip()
        if not re.search(
            r"(?:博物院|博物馆|美术馆|科技馆|纪念馆|天文馆|艺术馆|展览馆|会馆"
            r"|公园|植物园|动物园|故居|旧居|公馆|景区|旅游区|风景名胜区|古镇|古村"
            r"|历史文化街区|历史街区|文化街区|步行街|艺术区|建筑群|文化园|创意园"
            r"|古城|城|园|寺|庙|宫|观|祠|陵|府|教堂|清真寺|山|湖|岛|瀑布|峡谷|广场|塔|桥)$",
            venue_name,
        ):
            return "unverified_independent_child_site"
    return None


def _coordinate_distance_m(left: object | None, right: object | None) -> float | None:
    if left is None or right is None:
        return None
    left_latitude = getattr(left, "latitude", None)
    left_longitude = getattr(left, "longitude", None)
    right_latitude = getattr(right, "latitude", None)
    right_longitude = getattr(right, "longitude", None)
    if (
        not isinstance(left_latitude, (int, float))
        or not isinstance(left_longitude, (int, float))
        or not isinstance(right_latitude, (int, float))
        or not isinstance(right_longitude, (int, float))
    ):
        return None
    latitude_1 = math.radians(float(left_latitude))
    latitude_2 = math.radians(float(right_latitude))
    delta_latitude = latitude_2 - latitude_1
    delta_longitude = math.radians(float(right_longitude) - float(left_longitude))
    haversine = (
        math.sin(delta_latitude / 2) ** 2
        + math.cos(latitude_1) * math.cos(latitude_2) * math.sin(delta_longitude / 2) ** 2
    )
    return 6_371_000 * 2 * math.asin(min(1.0, math.sqrt(haversine)))


def _option(
    ranked: RankedCandidate,
    *,
    domain: CardDomain,
    interaction_id: UUID,
    role: CompositionRole,
) -> CardOption:
    candidate = ranked.candidate
    if domain is CardDomain.DINING:
        return _entity_option(
            candidate,
            domain=domain,
            interaction_id=interaction_id,
            role=role,
            description=candidate.place.short_description,
        )
    explanations = "；".join(ranked.explanations[:2])
    description = " · ".join(
        item
        for item in (
            candidate.place.address,
            explanations,
            "城市代表性补位" if role is CompositionRole.REPRESENTATIVE_EXTRA else None,
        )
        if item
    )
    return _entity_option(
        candidate,
        domain=domain,
        interaction_id=interaction_id,
        role=role,
        description=description or "来自当前城市真实地点候选。",
    )


def _entity_option(
    candidate: RecalledCandidate,
    *,
    domain: CardDomain,
    interaction_id: UUID,
    role: CompositionRole,
    description: str | None,
) -> CardOption:
    provider_sources = [
        source for source in candidate.sources if source.kind is RecallSourceKind.PROVIDER
    ]
    if not provider_sources:
        raise CardGenerationError("specific option requires a Provider source")
    observed_at = max(
        source.fetched_at for source in provider_sources if source.fetched_at is not None
    )
    assert isinstance(observed_at, datetime)
    source_refs = [
        f"provider:{source.provider.value}:{source.source_place_id}"
        for source in provider_sources
        if source.provider is not None and source.source_place_id is not None
    ]
    place_id = str(candidate.place.place_id)
    option_id = str(uuid5(NAMESPACE_URL, f"v4-card-option:{interaction_id}:{place_id}"))
    allowed = (
        ["must", "want", "if_convenient", "avoid"]
        if domain is CardDomain.ATTRACTION
        else ["destination", "if_convenient", "avoid"]
    )
    return CardOption(
        option_id=option_id,
        label=candidate.place.name,
        description=description,
        entity_ref=CardEntityRef(
            canonical_entity_id=place_id,
            entity_kind=("attraction" if domain is CardDomain.ATTRACTION else "restaurant"),
            provider_entity_refs=source_refs,
        ),
        semantic_value=CardSemanticValue(
            root=EntitySemanticValue(
                kind="entity_disposition",
                canonical_entity_id=place_id,
                allowed_dispositions=allowed,  # type: ignore[arg-type]
            )
        ),
        signed_operation_ref=str(
            uuid5(NAMESPACE_URL, f"v4-card-operation:{interaction_id}:{option_id}")
        ),
        source_refs=source_refs,
        observed_at=observed_at,
        composition_role=role,
        image_url=candidate.place.image_url,
        image_source_ref=source_refs[0] if candidate.place.image_url else None,
        dining_details=(
            DiningDisplayFacts(
                cuisine=candidate.place.cuisine,
                rating=candidate.place.rating or None,
                average_cost=candidate.place.average_cost,
                source_ref=source_refs[0],
                source_name=(
                    "高德"
                    if provider_sources[0].provider is ProviderCode.AMAP
                    else (
                        provider_sources[0].provider.value
                        if provider_sources[0].provider is not None
                        else "地点服务"
                    )
                ),
                observed_at=observed_at,
            )
            if domain is CardDomain.DINING
            and (candidate.place.cuisine or candidate.place.rating or candidate.place.average_cost)
            else None
        ),
    )


def _is_representative(item: RankedCandidate) -> bool:
    return item.candidate.representative_identity_verified and bool(
        set(item.candidate.channels) & {RecallChannel.CITY_LANDMARK, RecallChannel.CITY_FEATURE}
    )


def _hard_filtered(state: TripSemanticState, domain: CardDomain, name: str) -> bool:
    explicit = (
        state.attractions.exclusions if domain is CardDomain.ATTRACTION else state.dining.exclusions
    )
    normalized = "".join(name.casefold().split())
    if re.search(
        r"暂停营业|暂停开放|暂不营业|暂未开放|已关闭|已停业|歇业|永久关闭|已搬迁"
        r"|temporarilyclosed|permanentlyclosed",
        normalized,
    ):
        return True
    if any("".join(item.display_name.casefold().split()) in normalized for item in explicit):
        return True
    if domain is CardDomain.ATTRACTION:
        invalid_attraction_tokens = (
            "酒店",
            "宾馆",
            "饭店",
            "客栈",
            "民宿",
            "旅馆",
            "hotel",
            "餐厅",
            "餐馆",
            "炒货",
            "便利店",
            "超市",
            "商场",
            "购物中心",
            "珠宝",
            "黄金",
            "汽车",
            "4s店",
            "公司",
            "写字楼",
            "大厦",
            "停车场",
            "售票处",
            "游客中心",
        )
        auxiliary_suffixes = ("入口", "出口", "东门", "西门", "南门", "北门")
        if any(token in normalized for token in invalid_attraction_tokens) or normalized.endswith(
            auxiliary_suffixes
        ):
            return True
    if domain is CardDomain.DINING:
        # Only reject what the available place identity proves. Menu-level
        # allergies remain a declared verification need instead of being guessed.
        for avoidance in state.dining.avoidances:
            token = "".join(avoidance.casefold().replace("不吃", "").replace("不要", "").split())
            if len(token) >= 2 and token in normalized:
                return True
    return False


def _candidate_category_matches(domain: CardDomain, category: PlaceCategory) -> bool:
    expected = (
        PlaceCategory.ATTRACTION if domain is CardDomain.ATTRACTION else PlaceCategory.RESTAURANT
    )
    return category is expected


def specific_dependency_fingerprint(state: TripSemanticState, domain: CardDomain) -> str:
    from backend.persistence.outbox_repository import canonical_json_hash

    payload: dict[str, object] = {
        "destination": state.trip_basics.destination_canonical_id,
        "dates": [
            (
                state.trip_basics.start_date.isoformat()
                if state.trip_basics.start_date is not None
                else None
            ),
            (
                state.trip_basics.end_date.isoformat()
                if state.trip_basics.end_date is not None
                else None
            ),
        ],
        "duration_days": state.trip_basics.duration_days,
        "travelers": state.trip_basics.travelers,
        "constraints": state.constraints,
    }
    if domain is CardDomain.ATTRACTION:
        payload["goals"] = state.trip_basics.trip_goals
        payload["pace"] = state.transport_and_pace.pace_preferences
        payload["transport"] = state.transport_and_pace.transport_preferences
        payload["cold_start"] = (
            state.cold_start_profile_snapshot.model_dump(mode="json")
            if state.cold_start_profile_snapshot is not None
            else None
        )
        payload["preferences"] = [
            item.model_dump(mode="json") for item in state.attractions.preference_directions
        ]
        payload["exclusions"] = [
            item.model_dump(mode="json") for item in state.attractions.exclusions
        ]
    else:
        payload["preferences"] = [
            item.model_dump(mode="json") for item in state.dining.preference_directions
        ]
        payload["requirements"] = [
            *state.dining.requirements,
            *state.dining.allergies,
            *state.dining.avoidances,
        ]
    return canonical_json_hash(payload)


def _uuid_or_none(value: str) -> UUID | None:
    try:
        return UUID(value)
    except ValueError:
        return None


__all__ = [
    "CandidateCompositionResult",
    "CandidateCompositionService",
    "CardGenerationContextRequired",
    "CardGenerationError",
    "specific_dependency_fingerprint",
]
