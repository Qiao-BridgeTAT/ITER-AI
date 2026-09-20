"""Formal city-theme and attraction discovery orchestration for Agent turns."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from uuid import NAMESPACE_URL, UUID, uuid5

from backend.agent.model_gateway import ModelCancellation
from backend.agent.response_selection import ResponseActionKind, ResponseSelection
from backend.agent.semantic_operations import (
    AttractionIntentOperation,
    DateRangeOperation,
    DestinationOperation,
    ExperiencePreferenceKind,
    ExperiencePreferenceOperation,
)
from backend.agent.state_merge import SemanticTripState
from backend.contracts.candidate_ranking import (
    CandidateRankingPreferences,
    CandidateRankingRequest,
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
    RecallThemeInput,
    RecallThemeMode,
    RecallThemeSource,
)
from backend.contracts.conversation import ConversationAttachment
from backend.contracts.enums import AttractionIntent
from backend.planning.attraction_recommendation import AttractionRecommendationProjector
from backend.planning.candidate_ranking import CandidateRankingService
from backend.planning.candidate_recall import CandidateRecallService
from backend.planning.city_registry import CityRegistry
from backend.planning.city_theme import CityThemeProjector


class AttractionExplorationService:
    """Compose reviewed themes and the real recall/rank/project pipeline."""

    def __init__(
        self,
        *,
        registry: CityRegistry,
        recall: CandidateRecallService,
        ranking: CandidateRankingService,
        theme_projector: CityThemeProjector | None = None,
        recommendation_projector: AttractionRecommendationProjector | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._registry = registry
        self._recall = recall
        self._ranking = ranking
        self._theme_projector = theme_projector or CityThemeProjector()
        self._recommendation_projector = (
            recommendation_projector or AttractionRecommendationProjector()
        )
        self._clock = clock

    def city_theme_response(
        self,
        state: SemanticTripState,
        *,
        request_id: UUID,
        source_message_id: UUID,
        generation_id: UUID,
    ) -> ResponseSelection | None:
        destination = _destination(state)
        if destination is None or _theme_preferences(state):
            return None
        content = self._registry.load_content_package(destination.city_id)
        if content is None:
            return None
        projection = self._theme_projector.project(
            content,
            city_name=destination.display_name,
            attachment_id=uuid5(NAMESPACE_URL, f"iter:city-theme:{request_id}"),
            source_message_id=source_message_id,
            created_at=self._aware_now(),
            state_version=state.state_version,
            generation_id=generation_id,
        )
        return ResponseSelection(
            action=ResponseActionKind.STRUCTURED_ATTACHMENT,
            reason="reviewed city themes are the lowest-cost way to shape first recall",
            attachment=ConversationAttachment(root=projection.attachment),
            attachment_external_facts=projection.external_facts,
            acknowledgement="先选几个更吸引你的方向；它们只影响首轮候选，不会锁死玩法。",
        )

    async def recommendation_response(
        self,
        state: SemanticTripState,
        *,
        request_id: UUID,
        source_message_id: UUID,
        generation_id: UUID,
        cancellation: ModelCancellation | None = None,
    ) -> ResponseSelection | None:
        recall_request = _recall_request(state, request_id=request_id)
        if recall_request is None:
            return None
        recall_result = await self._recall.recall(recall_request, cancellation=cancellation)
        ranking_request = CandidateRankingRequest(
            ranking_request_id=uuid5(NAMESPACE_URL, f"iter:attraction-ranking:{request_id}"),
            recall_request=recall_request,
            recall_result=recall_result,
            preferences=CandidateRankingPreferences(
                selected_theme_ids=tuple(theme.theme_id for theme in recall_request.themes)
            ),
            recommendation_limit=_recommendation_count(recall_request.day_count),
        )
        ranking = self._ranking.rank(ranking_request)
        projection = self._recommendation_projector.project(
            ranking,
            attachment_id=uuid5(NAMESPACE_URL, f"iter:attraction-recommendation:{request_id}"),
            source_message_id=source_message_id,
            state_version=state.state_version,
            generation_id=generation_id,
        )
        return ResponseSelection(
            action=ResponseActionKind.CANDIDATE_RECOMMENDATION,
            reason="the formal recall, ranking and projection pipeline produced attractions",
            recommendation=projection.attachment,
            recommendation_external_facts=projection.external_facts,
        )

    def _aware_now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("attraction exploration clock must be timezone-aware")
        return value.astimezone(UTC)


def _destination(state: SemanticTripState):  # type: ignore[no-untyped-def]
    return next(
        (
            entry.operation.value
            for entry in state.entries
            if isinstance(entry.operation, DestinationOperation)
            and entry.operation.value is not None
        ),
        None,
    )


def _date_range(state: SemanticTripState):  # type: ignore[no-untyped-def]
    return next(
        (
            entry.operation.value
            for entry in state.entries
            if isinstance(entry.operation, DateRangeOperation) and entry.operation.value is not None
        ),
        None,
    )


def _theme_preferences(
    state: SemanticTripState,
) -> tuple[ExperiencePreferenceOperation, ...]:
    return tuple(
        entry.operation
        for entry in state.entries
        if isinstance(entry.operation, ExperiencePreferenceOperation)
        and entry.operation.value.kind is ExperiencePreferenceKind.CITY_THEME
    )


def _recall_request(
    state: SemanticTripState,
    *,
    request_id: UUID,
) -> CandidateRecallRequest | None:
    destination = _destination(state)
    date_range = _date_range(state)
    if destination is None or date_range is None:
        return None
    theme_operations = _theme_preferences(state)
    concrete_themes = [item for item in theme_operations if item.value.theme_id != "*"]
    open_to_any = any(item.value.theme_id == "*" for item in theme_operations)
    themes = tuple(
        RecallThemeInput(
            theme_id=item.value.theme_id or "",
            label=item.value.note or item.value.theme_id or "本次兴趣",
            source=RecallThemeSource.CURRENT_TRIP,
        )
        for item in concrete_themes
    )
    named: list[NamedPlaceClue] = []
    excluded: list[ExcludedPlaceClue] = []
    for entry in state.entries:
        operation = entry.operation
        if not isinstance(operation, AttractionIntentOperation):
            continue
        if operation.value.intent is AttractionIntent.AVOID:
            excluded.append(
                ExcludedPlaceClue(
                    clue_id=operation.operation_id,
                    name=operation.value.place_name or str(operation.value.place_id),
                    source_message_id=(
                        operation.evidence.source_message_id or operation.operation_id
                    ),
                    domain=CandidateDomain.ATTRACTION,
                    known_place_id=operation.value.place_id,
                )
            )
        elif operation.value.place_name:
            named.append(
                NamedPlaceClue(
                    clue_id=operation.operation_id,
                    name=operation.value.place_name,
                    domain=CandidateDomain.ATTRACTION,
                    priority=(
                        NamedPlacePriority.MUST
                        if operation.value.intent is AttractionIntent.MUST
                        else NamedPlacePriority.WANT
                    ),
                    source_message_id=operation.evidence.source_message_id
                    or operation.operation_id,
                    known_place_id=operation.value.place_id,
                )
            )
    task_book = state.task_book
    return CandidateRecallRequest(
        request_id=uuid5(NAMESPACE_URL, f"iter:attraction-recall:{request_id}"),
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
            if open_to_any or not theme_operations
            else RecallThemeMode.UNSPECIFIED
        ),
        themes=themes,
        free_text_clues=tuple(
            item.value.note or ""
            for item in concrete_themes
            if (item.value.theme_id or "").startswith("free:")
        ),
        named_places=tuple(named),
        excluded_places=tuple(excluded),
        landmark_policy=LandmarkRecallPolicy.NEUTRAL,
        budget=CandidateRecallBudget(
            max_total_candidates=30,
            max_total_provider_calls=6,
            domains=(
                DomainRecallBudget(
                    domain=CandidateDomain.ATTRACTION,
                    max_candidates=30,
                    max_provider_calls=6,
                ),
            ),
        ),
    )


def _recommendation_count(day_count: int) -> int:
    return {1: 5, 2: 6, 3: 7, 4: 10, 5: 10}[day_count]
