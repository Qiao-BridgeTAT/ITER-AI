"""Production adapter from the V3 planning graph to deterministic planning services."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from typing import TypeVar
from uuid import NAMESPACE_URL, UUID, uuid5

from backend.agent.model_gateway import ModelCancellation
from backend.agent.planning_graph import PlanningGraphError, PlanningGraphRequest
from backend.agent.task_book_state import (
    SemanticTaskBook,
    SemanticTaskBookStatus,
    TaskBookAttractionItem,
    TaskBookRestaurantItem,
)
from backend.contracts.candidate_ranking import (
    CandidateRankingPreferences,
    CandidateRankingRequest,
    CandidateRankingResult,
)
from backend.contracts.candidate_recall import (
    CandidateDomain,
    CandidateRecallBudget,
    CandidateRecallRequest,
    CandidateRecallResult,
    DomainRecallBudget,
    LandmarkRecallPolicy,
    NamedPlaceClue,
    NamedPlacePriority,
    ProviderRecallQuery,
    RecallChannel,
    RecalledCandidate,
    RecalledPlace,
    RecallPlan,
    RecallThemeInput,
    RecallThemeMode,
    RecallThemeSource,
)
from backend.contracts.common import CnyAmountRange
from backend.contracts.cost_estimation import (
    CostEstimationRequest,
    CostPriceBasis,
    CostPriceFact,
    CostSubjectKind,
    OriginalMoneyRange,
    TripCostEstimate,
)
from backend.contracts.daily_scheduling import (
    DailyScheduleResult,
    DailyScheduleWindow,
    DailySchedulingRequest,
    OpeningDateStatus,
    OpeningWindow,
    ScheduleActivityKind,
    SchedulePlaceFact,
    SchedulePreferences,
    ScheduleRouteFact,
)
from backend.contracts.enums import (
    AnchorRole,
    AttractionIntent,
    CostCategory,
    DataAvailability,
    DayReturn,
    DayStart,
    MobilityTolerance,
    PlaceCategory,
    ProviderCode,
    RestaurantIntent,
    TransportMode,
)
from backend.contracts.hotel_selection import (
    FixedHotelInput,
    HotelDecisionMode,
    HotelSelectionRequest,
    HotelSelectionResult,
)
from backend.contracts.itinerary_draft import CostValidationDraft, ScheduleValidationDraft
from backend.contracts.itinerary_repair import ItineraryRepairRequest, ItineraryRepairResult
from backend.contracts.itinerary_validation import (
    DailyWeatherCoverage,
    ItineraryValidationRequest,
    ItineraryValidationResult,
)
from backend.contracts.lodging_strategy import (
    LodgingClusterAccess,
    LodgingStrategyPreferences,
    LodgingStrategyRequest,
    LodgingStrategyResult,
)
from backend.contracts.places import CanonicalPlace
from backend.contracts.plan_publication import PlanPublicationRequest
from backend.contracts.spatial_planning import (
    SpatialNodeInput,
    SpatialPlanningRequest,
    SpatialPlanningResult,
    SpatialRouteOption,
)
from backend.contracts.state import TripState
from backend.planning.candidate_ranking import CandidateRankingService
from backend.planning.candidate_recall import CandidateRecallService
from backend.planning.city_registry import (
    CityProviderUnavailableError,
    CityRegistry,
    default_city_registry,
)
from backend.planning.cost_estimation import CostEstimationService
from backend.planning.daily_scheduling import DailySchedulingService
from backend.planning.hotel_selection import HotelSelectionService
from backend.planning.itinerary_repair import ItineraryRepairService
from backend.planning.itinerary_validation import ItineraryValidationService
from backend.planning.lodging_strategy import LodgingStrategyService
from backend.planning.map_projection import MapProjectionService
from backend.planning.plan_publication import PlanPublicationService
from backend.planning.recall_plan import RecallPlanGenerator
from backend.planning.spatial_planning import SpatialPlanningService
from backend.providers.contracts import (
    HoursDayStatus,
    HoursRequest,
    ProductSearchRequest,
    ProviderForecastDay,
    ProviderRegularHours,
    ProviderResultStatus,
    ProviderTicketOffer,
    RouteMode,
    WeatherRequest,
)
from backend.providers.hours_rules import describe_date_hours, evaluate_regular_hours
from backend.providers.interfaces import (
    HoursProvider,
    PlaceProvider,
    RouteProvider,
    TravelProductProvider,
    WeatherProvider,
)


def _id(*parts: object) -> UUID:
    return uuid5(NAMESPACE_URL, "iter:v3-runtime:" + ":".join(str(part) for part in parts))


@dataclass(frozen=True)
class PlanningProviderSet:
    places: PlaceProvider
    routes: RouteProvider
    hours: HoursProvider
    products: TravelProductProvider
    weather: WeatherProvider | None


@dataclass(frozen=True)
class RecallArtifact:
    request: CandidateRecallRequest
    result: CandidateRecallResult


@dataclass(frozen=True)
class RankingArtifact:
    request: CandidateRankingRequest
    result: CandidateRankingResult


@dataclass(frozen=True)
class SpatialArtifact:
    request: SpatialPlanningRequest
    result: SpatialPlanningResult


@dataclass(frozen=True)
class LodgingArtifact:
    request: LodgingStrategyRequest
    result: LodgingStrategyResult


@dataclass(frozen=True)
class HotelArtifact:
    request: HotelSelectionRequest
    result: HotelSelectionResult


@dataclass(frozen=True)
class ScheduleArtifact:
    request: DailySchedulingRequest
    result: DailyScheduleResult


@dataclass(frozen=True)
class CostArtifact:
    request: CostEstimationRequest
    result: TripCostEstimate


@dataclass(frozen=True)
class ValidationArtifact:
    request: ItineraryValidationRequest
    result: ItineraryValidationResult


ArtifactT = TypeVar("ArtifactT")


class DeterministicRecallPlanGenerator:
    """Replay/fake plan generator that still uses the formal recall service."""

    async def create_plan(
        self,
        request: CandidateRecallRequest,
        city: object,
        content: object,
        *,
        cancellation: ModelCancellation | None = None,
    ) -> RecallPlan:
        del city, content
        if cancellation is not None:
            cancellation.raise_if_cancelled("candidate_recall_plan")
        queries: list[ProviderRecallQuery] = []
        for clue in request.named_places:
            queries.append(
                ProviderRecallQuery(
                    query_id=f"named:{clue.clue_id}",
                    channel=RecallChannel.USER_NAMED,
                    domain=clue.domain,
                    keyword=clue.name,
                    named_clue_id=clue.clue_id,
                    max_results=1,
                    reason="精确核对任务书中的用户已选地点。",
                )
            )
        for theme in request.themes:
            queries.append(
                ProviderRecallQuery(
                    query_id=f"theme:{theme.theme_id}",
                    channel=RecallChannel.SELECTED_THEME,
                    domain=CandidateDomain.ATTRACTION,
                    keyword=theme.label,
                    theme_ids=(theme.theme_id,),
                    max_results=1,
                    reason="按用户已确认的本次体验方向召回地点。",
                )
            )
        domains = {item.domain for item in request.budget.domains}
        named_attraction = next(
            (clue for clue in request.named_places if clue.domain is CandidateDomain.ATTRACTION),
            None,
        )
        if CandidateDomain.ATTRACTION in domains and named_attraction is None:
            queries.append(
                ProviderRecallQuery(
                    query_id="landmark:attraction",
                    channel=RecallChannel.CITY_LANDMARK,
                    domain=CandidateDomain.ATTRACTION,
                    keyword=(request.themes[0].label if request.themes else "景点"),
                    max_results=max(
                        1,
                        min(
                            5,
                            request.budget.max_total_candidates - len(request.themes),
                        ),
                    ),
                    reason="在当前城市补充代表性景点候选。",
                )
            )
        elif CandidateDomain.ATTRACTION in domains:
            # The recall contract requires an explicit landmark channel. Re-check the first
            # selected attraction through that channel without inventing another identity.
            assert named_attraction is not None
            queries.append(
                ProviderRecallQuery(
                    query_id="landmark:selected",
                    channel=RecallChannel.CITY_LANDMARK,
                    domain=CandidateDomain.ATTRACTION,
                    keyword=named_attraction.name,
                    max_results=1,
                    reason="保留一个经过核对的城市代表地点。",
                )
            )
        return RecallPlan(
            plan_version="1.0.0",
            request_id=request.request_id,
            city_id=request.city_id,
            provider_queries=tuple(queries),
        )


class FormalPlanningBackend:
    """Invoke V3-29..43 services with one trip-bound artifact chain."""

    def __init__(
        self,
        *,
        registry: CityRegistry,
        providers: PlanningProviderSet | None,
        recall_plan_generator: RecallPlanGenerator,
        business_date: Callable[[], date] = date.today,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._registry = registry
        self._providers = providers
        self._recall_plan_generator = recall_plan_generator
        self._business_date = business_date
        self._clock = clock
        self._ranking = CandidateRankingService(clock=clock)
        self._lodging = LodgingStrategyService(clock=clock)
        self._scheduling = DailySchedulingService(clock=clock)
        self._cost = CostEstimationService(clock=clock)
        self._validation = ItineraryValidationService(clock=clock)
        self._repair = ItineraryRepairService(
            validation_service=self._validation,
            scheduling_service=self._scheduling,
            clock=clock,
        )
        self._publication = PlanPublicationService(
            validation_service=self._validation,
            city_registry=registry,
            clock=clock,
        )

    def _provider_set(self, state: TripState) -> PlanningProviderSet:
        if self._providers is not None:
            return self._providers
        return stored_planning_providers(state, clock=self._clock)

    async def candidate_recall(
        self,
        request: PlanningGraphRequest,
        artifacts: Mapping[str, object],
        cancellation: ModelCancellation,
    ) -> RecallArtifact:
        del artifacts
        recall_request = _recall_request(request.state, request.semantic_task_book)
        providers = self._provider_set(request.state)
        result = await CandidateRecallService(
            registry=self._registry,
            places=providers.places,
            plan_generator=self._recall_plan_generator,
            clock=self._clock,
        ).recall(recall_request, cancellation=cancellation)
        if not result.candidates:
            raise PlanningGraphError("candidate_recall", "no executable candidates were recalled")
        return RecallArtifact(recall_request, result)

    async def candidate_ranking(
        self,
        request: PlanningGraphRequest,
        artifacts: Mapping[str, object],
        cancellation: ModelCancellation,
    ) -> RankingArtifact:
        cancellation.raise_if_cancelled("candidate_ranking")
        recall = _artifact(artifacts, "recall", RecallArtifact)
        preferences = request.state.resolved_preferences
        ranking_request = CandidateRankingRequest(
            ranking_request_id=_id(request.generation_id, "ranking"),
            recall_request=recall.request,
            recall_result=recall.result,
            preferences=CandidateRankingPreferences(
                pace_level=preferences.pace_level if preferences else 3,
                classic_niche_level=preferences.classic_niche_level if preferences else 3,
                transit_taxi_level=preferences.transit_taxi_level if preferences else 3,
                selected_theme_ids=tuple(theme.theme_id for theme in recall.request.themes),
            ),
        )
        result = self._ranking.rank(ranking_request)
        if not result.candidates:
            raise PlanningGraphError("candidate_ranking", "no executable candidates remain")
        return RankingArtifact(ranking_request, result)

    async def spatial_planning(
        self,
        request: PlanningGraphRequest,
        artifacts: Mapping[str, object],
        cancellation: ModelCancellation,
    ) -> SpatialArtifact:
        ranking = _artifact(artifacts, "ranking", RankingArtifact)
        spatial_request = _spatial_request(
            request.state,
            request.semantic_task_book,
            ranking.result,
            request.generation_id,
        )
        result = await SpatialPlanningService(
            registry=self._registry,
            routes=self._provider_set(request.state).routes,
            clock=self._clock,
        ).build(spatial_request, cancellation=cancellation)
        return SpatialArtifact(spatial_request, result)

    async def lodging_strategy(
        self,
        request: PlanningGraphRequest,
        artifacts: Mapping[str, object],
        cancellation: ModelCancellation,
    ) -> LodgingArtifact:
        cancellation.raise_if_cancelled("lodging_strategy")
        spatial = _artifact(artifacts, "spatial", SpatialArtifact)
        lodging_request = _lodging_request(request.state, spatial.result, request.generation_id)
        return LodgingArtifact(lodging_request, self._lodging.build(lodging_request))

    async def hotel_selection(
        self,
        request: PlanningGraphRequest,
        artifacts: Mapping[str, object],
        cancellation: ModelCancellation,
    ) -> HotelArtifact:
        lodging = _artifact(artifacts, "lodging", LodgingArtifact)
        hotel_request = _hotel_request(request.state, lodging.result, request.generation_id)
        result = await HotelSelectionService(
            registry=self._registry,
            products=self._provider_set(request.state).products,
            clock=self._clock,
        ).select(hotel_request)
        cancellation.raise_if_cancelled("hotel_selection")
        if request.state.night_count and result.selected_hotel_place_id is None:
            raise PlanningGraphError(
                "hotel_selection", "the confirmed hotel could not be preserved"
            )
        return HotelArtifact(hotel_request, result)

    async def daily_scheduling(
        self,
        request: PlanningGraphRequest,
        artifacts: Mapping[str, object],
        cancellation: ModelCancellation,
    ) -> ScheduleArtifact:
        spatial = _artifact(artifacts, "spatial", SpatialArtifact)
        hotel = _artifact(artifacts, "hotel", HotelArtifact)
        providers = self._provider_set(request.state)
        scheduling_request = await _scheduling_request(
            request.state,
            request.semantic_task_book,
            spatial.result,
            hotel.result,
            providers.hours,
            cancellation,
            business_date=self._business_date(),
            generation_id=request.generation_id,
        )
        result = self._scheduling.build(scheduling_request)
        return ScheduleArtifact(scheduling_request, result)

    async def cost_estimation(
        self,
        request: PlanningGraphRequest,
        artifacts: Mapping[str, object],
        cancellation: ModelCancellation,
    ) -> CostArtifact:
        schedule = _artifact(artifacts, "schedule", ScheduleArtifact)
        hotel = _artifact(artifacts, "hotel", HotelArtifact)
        providers = self._provider_set(request.state)
        cost_request = await _cost_request(
            request.state,
            schedule,
            hotel.result,
            providers.products,
            cancellation,
            business_time=self._aware_now(),
            generation_id=request.generation_id,
        )
        return CostArtifact(cost_request, self._cost.estimate(cost_request))

    async def itinerary_validation(
        self,
        request: PlanningGraphRequest,
        artifacts: Mapping[str, object],
        cancellation: ModelCancellation,
    ) -> ValidationArtifact:
        schedule = _artifact(artifacts, "schedule", ScheduleArtifact)
        cost = _artifact(artifacts, "cost", CostArtifact)
        weather = await _weather_coverage(
            request.state,
            self._provider_set(request.state).weather,
            self._registry,
            cancellation,
        )
        validation_request = ItineraryValidationRequest(
            request_id=_id(request.generation_id, "validation"),
            trip_id=request.state.trip_id,
            input_state_version=request.state.state_version,
            scheduling_request=schedule.request,
            schedule_draft=ScheduleValidationDraft.from_result(schedule.result),
            cost_draft=CostValidationDraft.from_estimate(cost.result),
            weather=weather,
        )
        return ValidationArtifact(
            validation_request,
            self._validation.validate(validation_request),
        )

    async def itinerary_repair(
        self,
        request: PlanningGraphRequest,
        artifacts: Mapping[str, object],
        cancellation: ModelCancellation,
    ) -> ItineraryRepairResult:
        validation = _artifact(artifacts, "validation", ValidationArtifact)
        result = self._repair.repair(
            ItineraryRepairRequest(
                request_id=_id(request.generation_id, "repair"),
                trip_id=request.state.trip_id,
                input_state_version=request.state.state_version,
                generation_id=request.generation_id,
                validation_request=validation.request,
                validation_result=validation.result,
            ),
            is_generation_active=lambda generation_id: (
                generation_id == request.generation_id and not cancellation.is_cancelled
            ),
        )
        if not result.strict_ready:
            raise PlanningGraphError("itinerary_repair", "hard conflicts remain after repair")
        return result

    async def plan_publication(
        self,
        request: PlanningGraphRequest,
        artifacts: Mapping[str, object],
        cancellation: ModelCancellation,
    ) -> TripState:
        ranking = _artifact(artifacts, "ranking", RankingArtifact)
        validation = _artifact(artifacts, "validation", ValidationArtifact)
        repair = _artifact(artifacts, "repair", ItineraryRepairResult)
        schedule = _artifact(artifacts, "schedule", ScheduleArtifact)
        scheduled_place_ids = {
            activity.place_id
            for day in repair.best_schedule_draft.days
            for activity in day.activities
        }
        selected_candidates: tuple[RecalledCandidate, ...] = tuple(
            ranked.candidate
            for ranked in ranking.result.candidates
            if ranked.candidate.place.place_id in scheduled_place_ids
        )
        publication = PlanPublicationRequest(
            request_id=_id(request.generation_id, "publication"),
            publication_key=f"v3-plan:{request.state.trip_id}:{request.generation_id}",
            trip_id=request.state.trip_id,
            generation_id=request.generation_id,
            expected_state_version=request.state.state_version,
            plan_version_id=_id(request.state.trip_id, request.generation_id, "plan-version"),
            parent_version_id=request.state.current_plan_version_id,
            base_confirmed_version_id=request.state.base_confirmed_version_id,
            validation_request=validation.request,
            repair_result=repair,
            selected_candidates=selected_candidates,
            assumptions=tuple(request.state.task_book.assumptions),  # type: ignore[union-attr]
            map_projection=MapProjectionService().build(schedule.request, schedule.result),
        )
        return await self._publication.prepare_stable_state(
            request.state,
            publication,
            is_generation_active=lambda trip_id, generation_id: _active(
                request, cancellation, trip_id, generation_id
            ),
        )

    def _aware_now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("planning clock must return an aware datetime")
        return value.astimezone(UTC)


def _artifact(artifacts: Mapping[str, object], key: str, expected: type[ArtifactT]) -> ArtifactT:
    value = artifacts.get(key)
    if not isinstance(value, expected):
        raise PlanningGraphError(key, f"planning artifact {key} has the wrong type")
    return value


async def _active(
    request: PlanningGraphRequest,
    cancellation: ModelCancellation,
    trip_id: UUID,
    generation_id: UUID,
) -> bool:
    return (
        not cancellation.is_cancelled
        and trip_id == request.state.trip_id
        and generation_id == request.generation_id
    )


def _city_id(state: TripState) -> str:
    value = state.city_id or (state.city.value if state.city is not None else None)
    if value is None:
        raise PlanningGraphError("planning_gate", "planning state has no city_id")
    return default_city_registry().resolve(value).city_id


def _trip_dates(state: TripState) -> tuple[date, ...]:
    if state.date_range is None:
        raise PlanningGraphError("planning_gate", "planning state has no date range")
    return tuple(
        state.date_range.start_date.fromordinal(day)
        for day in range(
            state.date_range.start_date.toordinal(),
            state.date_range.end_date.toordinal() + 1,
        )
    )


def semantic_task_book_from_state(state: TripState) -> SemanticTaskBook:
    """Compatibility adapter for reviewed Replay snapshots without Agent checkpoints."""

    task = state.task_book
    if task is None:
        raise PlanningGraphError("planning_gate", "task book is missing")
    source_ids = [_id(state.trip_id, "destination"), _id(state.trip_id, "dates")]
    places = _all_places(state)
    attraction_items: list[TaskBookAttractionItem] = []
    for place_id in task.strong_attraction_ids:
        operation_id = _id(state.trip_id, "task-attraction", place_id)
        source_ids.append(operation_id)
        attraction_items.append(
            TaskBookAttractionItem(
                source_operation_id=operation_id,
                place_id=place_id,
                place_name=(places[place_id].name if place_id in places else None),
                intent=AttractionIntent.MUST,
            )
        )
    restaurant_items: list[TaskBookRestaurantItem] = []
    for place_id in task.important_restaurant_ids:
        operation_id = _id(state.trip_id, "task-restaurant", place_id)
        source_ids.append(operation_id)
        restaurant_items.append(
            TaskBookRestaurantItem(
                source_operation_id=operation_id,
                name=places[place_id].name if place_id in places else str(place_id),
                place_id=place_id,
                intent=RestaurantIntent.DESTINATION,
            )
        )
    return SemanticTaskBook.model_validate(
        {
            "task_book_id": str(_id(state.trip_id, "runtime-task-book", state.state_version)),
            "trip_id": str(state.trip_id),
            "revision": max(2, state.state_version),
            "source_state_version": max(1, state.state_version - 1),
            "published_state_version": max(2, state.state_version),
            "destination": {"city_id": _city_id(state), "display_name": _city_id(state)},
            "date_range": task.model_dump(mode="json", include={"start_date", "end_date"}),
            "attraction_intents": [item.model_dump(mode="json") for item in attraction_items],
            "important_restaurants": [item.model_dump(mode="json") for item in restaurant_items],
            "tradeoffs": task.tradeoffs,
            "omitted_strong_desires": task.omitted_strong_desires,
            "source_operation_ids": [str(item) for item in source_ids],
            "status": SemanticTaskBookStatus.CONFIRMED.value,
            "confirmation_operation_id": str(_id(state.trip_id, "task-confirmation")),
            "confirmed_state_version": max(3, state.state_version + 1),
        },
        context={"today": task.start_date},
    )


def _all_places(state: TripState) -> dict[UUID, RecalledPlace]:
    city_id = _city_id(state)
    values: dict[UUID, RecalledPlace] = {}
    for canonical in state.candidate_places:
        values[canonical.place_id] = RecalledPlace(
            place_id=canonical.place_id,
            city_id=city_id,
            category=canonical.category,
            name=canonical.name,
            address=canonical.address,
            coordinates=canonical.coordinates,
        )
    for projected in state.provider_display.places:
        values[projected.place_id] = RecalledPlace(
            place_id=projected.place_id,
            city_id=city_id,
            category=projected.category,
            name=projected.name,
            address=projected.address,
            coordinates=projected.coordinates,
        )
    return values


def _recall_request(
    state: TripState, semantic_task_book: SemanticTaskBook
) -> CandidateRecallRequest:
    if state.task_book is None or state.date_range is None:
        raise PlanningGraphError("candidate_recall", "confirmed task book is incomplete")
    places = _all_places(state)
    named: list[NamedPlaceClue] = []
    source_message = (
        state.conversation_messages[-1].message_id if state.conversation_messages else state.trip_id
    )
    for attraction in semantic_task_book.attraction_intents:
        place_id = attraction.place_id
        place = places.get(place_id)
        if place is None or attraction.intent is AttractionIntent.AVOID:
            continue
        named.append(
            NamedPlaceClue(
                clue_id=_id(state.trip_id, "attraction-clue", place_id),
                name=place.name,
                domain=CandidateDomain.ATTRACTION,
                priority=(
                    NamedPlacePriority.MUST
                    if attraction.intent is AttractionIntent.MUST
                    else NamedPlacePriority.WANT
                ),
                source_message_id=source_message,
                known_place_id=place_id,
            )
        )
    for restaurant in semantic_task_book.important_restaurants:
        restaurant_place_id = restaurant.place_id
        if restaurant_place_id is None:
            continue
        place = places.get(restaurant_place_id)
        if place is None or restaurant.intent is RestaurantIntent.AVOID:
            continue
        named.append(
            NamedPlaceClue(
                clue_id=_id(state.trip_id, "restaurant-clue", restaurant_place_id),
                name=place.name,
                domain=CandidateDomain.RESTAURANT,
                priority=(
                    NamedPlacePriority.MUST
                    if restaurant.intent is RestaurantIntent.DESTINATION
                    else NamedPlacePriority.WANT
                ),
                source_message_id=source_message,
                known_place_id=restaurant_place_id,
            )
        )
    raw_themes = tuple(
        RecallThemeInput(
            theme_id=theme_id,
            label=theme_id,
            source=RecallThemeSource.CURRENT_TRIP,
        )
        for theme_id in (
            state.city_theme_selection.selected_theme_ids if state.city_theme_selection else ()
        )
    )
    if len(named) > 11:
        raise PlanningGraphError(
            "candidate_recall",
            "the confirmed task book contains more named places than the recall budget can verify",
        )
    # Named places are confirmed user intent and must be verified first. Theme expansion uses
    # only the remaining provider-call budget after reserving one landmark query.
    themes = raw_themes[: max(0, 11 - len(named))]
    domains = [CandidateDomain.ATTRACTION]
    if any(item.domain is CandidateDomain.RESTAURANT for item in named):
        domains.append(CandidateDomain.RESTAURANT)
    provider_calls = len(named) + len(themes) + 1
    total_candidates = min(40, max(12, provider_calls * 2))
    per_domain_candidates = max(1, total_candidates // len(domains))
    per_domain_calls = {domain: 0 for domain in domains}
    for clue in named:
        per_domain_calls[clue.domain] += 1
    per_domain_calls[CandidateDomain.ATTRACTION] += len(themes) + 1
    return CandidateRecallRequest(
        request_id=_id(state.trip_id, state.state_version, "recall"),
        trip_id=state.trip_id,
        semantic_state_version=state.state_version,
        task_book_id=semantic_task_book.task_book_id,
        task_book_revision=semantic_task_book.revision,
        city_id=_city_id(state),
        start_date=state.date_range.start_date,
        end_date=state.date_range.end_date,
        theme_mode=(RecallThemeMode.SELECTED if themes else RecallThemeMode.UNSPECIFIED),
        themes=themes,
        free_text_clues=tuple(
            value
            for value in (
                state.city_theme_selection.free_text if state.city_theme_selection else None,
                state.dining_preferences.free_text if state.dining_preferences else None,
            )
            if value
        ),
        named_places=tuple(named),
        landmark_policy=LandmarkRecallPolicy.NEUTRAL,
        budget=CandidateRecallBudget(
            max_total_candidates=total_candidates,
            max_total_provider_calls=provider_calls,
            domains=tuple(
                DomainRecallBudget(
                    domain=domain,
                    max_candidates=(
                        total_candidates - per_domain_candidates * (len(domains) - 1)
                        if index == 0
                        else per_domain_candidates
                    ),
                    max_provider_calls=per_domain_calls[domain],
                )
                for index, domain in enumerate(domains)
            ),
        ),
    )


def _spatial_request(
    state: TripState,
    semantic_task_book: SemanticTaskBook,
    ranking: CandidateRankingResult,
    generation_id: UUID,
) -> SpatialPlanningRequest:
    if state.task_book is None or state.date_range is None:
        raise PlanningGraphError("spatial_planning", "planning boundary is incomplete")
    dates = _trip_dates(state)
    feedback = {item.place_id: item.intent for item in state.attraction_feedback}
    restaurant_feedback = {item.place_id: item.intent for item in state.restaurant_feedback}
    candidates = [item.candidate for item in ranking.candidates]
    places = {item.place.place_id: item.place for item in candidates}
    nodes: list[SpatialNodeInput] = []
    for item in candidates:
        place_id = item.place.place_id
        if item.domain is CandidateDomain.ATTRACTION:
            intent = feedback.get(place_id)
            role = (
                AnchorRole.MUST_ATTRACTION
                if place_id in state.task_book.strong_attraction_ids
                or intent is AttractionIntent.MUST
                else AnchorRole.WANT_ATTRACTION
                if intent is AttractionIntent.WANT
                else AnchorRole.CONVENIENT_ATTRACTION
            )
        elif item.domain is CandidateDomain.RESTAURANT:
            restaurant_intent = restaurant_feedback.get(place_id)
            role = (
                AnchorRole.DESTINATION_RESTAURANT
                if place_id in state.task_book.important_restaurant_ids
                or restaurant_intent is RestaurantIntent.DESTINATION
                else AnchorRole.CONVENIENT_RESTAURANT
            )
        else:
            continue
        nodes.append(
            SpatialNodeInput(
                node_id=_id(generation_id, "node", place_id),
                place_id=place_id,
                candidate_id=item.candidate_id,
                role=role,
                available_dates=dates,
                source_reference_ids=tuple(source.source_record_id for source in item.sources),
            )
        )
    if state.selected_hotel is not None:
        selected = _all_places(state).get(state.selected_hotel.place_id)
        if selected is None or selected.coordinates is None:
            raise PlanningGraphError(
                "spatial_planning", "selected hotel has no verified coordinates"
            )
        places[selected.place_id] = selected
        nodes.append(
            SpatialNodeInput(
                node_id=_id(generation_id, "fixed-hotel", selected.place_id),
                place_id=selected.place_id,
                role=AnchorRole.FIXED_HOTEL,
                available_dates=dates,
                source_reference_ids=(f"trip:selected-hotel:{selected.place_id}",),
            )
        )
    if not nodes:
        raise PlanningGraphError("spatial_planning", "no selected places can form a plan")
    return SpatialPlanningRequest(
        request_id=_id(generation_id, "spatial"),
        trip_id=state.trip_id,
        input_state_version=state.state_version,
        task_book_id=semantic_task_book.task_book_id,
        task_book_revision=semantic_task_book.revision,
        city_id=_city_id(state),
        start_date=state.date_range.start_date,
        end_date=state.date_range.end_date,
        places=tuple(places[node.place_id] for node in nodes),
        nodes=tuple(nodes),
        route_modes=_allowed_modes(state),
    )


def _lodging_preferences(state: TripState) -> LodgingStrategyPreferences:
    preferences = state.resolved_preferences
    return LodgingStrategyPreferences(
        transit_taxi_level=preferences.transit_taxi_level if preferences else 3,
        walking_tolerance=preferences.walking_tolerance
        if preferences
        else MobilityTolerance.AROUND_10,
        cycling_tolerance=preferences.bike_tolerance if preferences else MobilityTolerance.NEVER,
        pace_level=preferences.pace_level if preferences else 3,
        quality_level=4,
        value_priority_level=3,
    )


def _lodging_request(
    state: TripState,
    spatial: SpatialPlanningResult,
    generation_id: UUID,
) -> LodgingStrategyRequest:
    if state.date_range is None:
        raise PlanningGraphError("lodging_strategy", "trip dates are missing")
    if not state.night_count:
        return LodgingStrategyRequest(
            request_id=_id(generation_id, "lodging"),
            trip_id=state.trip_id,
            input_state_version=state.state_version,
            city_id=_city_id(state),
            start_date=state.date_range.start_date,
            end_date=state.date_range.end_date,
            spatial_result=spatial,
            preferences=_lodging_preferences(state),
        )
    hotel = next((item for item in spatial.anchors if item.role is AnchorRole.FIXED_HOTEL), None)
    if hotel is None:
        raise PlanningGraphError("lodging_strategy", "overnight plan has no confirmed hotel anchor")
    edge_by_pair = {
        frozenset((edge.origin_node_id, edge.destination_node_id)): edge
        for edge in spatial.route_edges
    }
    accesses: list[LodgingClusterAccess] = []
    for cluster in spatial.clusters:
        if hotel.node_id in cluster.member_node_ids:
            accesses.append(
                LodgingClusterAccess(
                    cluster_id=cluster.cluster_id,
                    status=DataAvailability.AVAILABLE,
                    transit_minutes=0,
                    transit_transfer_count=0,
                    transit_last_mile_walk_m=0,
                    taxi_minutes=0,
                    cycling_minutes=0,
                    source_reference_ids=(f"trip:selected-hotel:{hotel.place_id}",),
                )
            )
            continue
        edges = [
            edge_by_pair.get(frozenset((hotel.node_id, member)))
            for member in cluster.member_node_ids
        ]
        options = [route for edge in edges if edge is not None for route in edge.routes]
        usable = [route for route in options if route.mode in _allowed_modes(state)]
        if not usable:
            accesses.append(
                LodgingClusterAccess(
                    cluster_id=cluster.cluster_id,
                    status=DataAvailability.MISSING,
                    source_reference_ids=(
                        f"amap:route-missing:{hotel.node_id}:{cluster.cluster_id}",
                    ),
                    missing_fields=("route",),
                    missing_reason="没有取得用户允许交通方式的酒店通勤路线。",
                )
            )
            continue
        by_mode = {mode: [item for item in usable if item.mode is mode] for mode in RouteMode}

        transit = _best_route(by_mode, RouteMode.TRANSIT)
        driving = _best_route(by_mode, RouteMode.DRIVING)
        cycling = _best_route(by_mode, RouteMode.CYCLING)
        accesses.append(
            LodgingClusterAccess(
                cluster_id=cluster.cluster_id,
                status=DataAvailability.AVAILABLE,
                transit_minutes=_route_minutes(transit),
                transit_transfer_count=(transit.transfer_count if transit else None),
                transit_last_mile_walk_m=(transit.walking_distance_m if transit else None),
                taxi_minutes=_route_minutes(driving),
                cycling_minutes=_route_minutes(cycling),
                source_reference_ids=tuple(
                    sorted(
                        f"amap:spatial-edge:{edge.edge_id}"
                        for member in cluster.member_node_ids
                        if (edge := edge_by_pair.get(frozenset((hotel.node_id, member))))
                        is not None
                    )
                ),
            )
        )
    return LodgingStrategyRequest(
        request_id=_id(generation_id, "lodging"),
        trip_id=state.trip_id,
        input_state_version=state.state_version,
        city_id=_city_id(state),
        start_date=state.date_range.start_date,
        end_date=state.date_range.end_date,
        spatial_result=spatial,
        preferences=_lodging_preferences(state),
        fixed_hotel_node_id=hotel.node_id,
        fixed_hotel_accesses=tuple(accesses),
    )


def _hotel_request(
    state: TripState,
    lodging: LodgingStrategyResult,
    generation_id: UUID,
) -> HotelSelectionRequest:
    if state.date_range is None:
        raise PlanningGraphError("hotel_selection", "trip dates are missing")
    if not state.night_count:
        return HotelSelectionRequest(
            request_id=_id(generation_id, "hotel"),
            trip_id=state.trip_id,
            input_state_version=state.state_version,
            city_id=_city_id(state),
            check_in=state.date_range.start_date,
            check_out=state.date_range.end_date,
            lodging_result=lodging,
            preferences=_lodging_preferences(state),
            decision_mode=HotelDecisionMode.NOT_REQUIRED,
        )
    selected_hotel = state.selected_hotel
    if selected_hotel is None:
        raise PlanningGraphError("hotel_selection", "overnight trip has no selected hotel")
    place = _all_places(state).get(selected_hotel.place_id)
    fixed_anchor = next(item for item in lodging.strategies if item.fixed_hotel_node_id is not None)
    if place is None or place.coordinates is None or fixed_anchor.fixed_hotel_node_id is None:
        raise PlanningGraphError("hotel_selection", "selected hotel identity is incomplete")
    return HotelSelectionRequest(
        request_id=_id(generation_id, "hotel"),
        trip_id=state.trip_id,
        input_state_version=state.state_version,
        city_id=_city_id(state),
        check_in=state.date_range.start_date,
        check_out=state.date_range.end_date,
        lodging_result=lodging,
        preferences=_lodging_preferences(state),
        fixed_hotel=FixedHotelInput(
            node_id=fixed_anchor.fixed_hotel_node_id,
            place_id=place.place_id,
            name=place.name,
            coordinates=place.coordinates,
            source_reference_ids=(f"trip:selected-hotel:{place.place_id}",),
        ),
        decision_mode=HotelDecisionMode.PREBOOKED,
    )


async def _scheduling_request(
    state: TripState,
    task_book: SemanticTaskBook,
    spatial: SpatialPlanningResult,
    hotel: HotelSelectionResult,
    hours: HoursProvider,
    cancellation: ModelCancellation,
    *,
    business_date: date,
    generation_id: UUID,
) -> DailySchedulingRequest:
    dates = _trip_dates(state)
    places: list[SchedulePlaceFact] = []
    for anchor in spatial.anchors:
        if anchor.role is AnchorRole.FIXED_HOTEL:
            places.append(
                SchedulePlaceFact(
                    place_id=anchor.place_id,
                    city_id=_city_id(state),
                    name=anchor.name,
                    category=PlaceCategory.HOTEL,
                    coordinates=anchor.coordinates,
                    recommended_duration_minutes=15,
                    opening_availability=DataAvailability.MISSING,
                    opening_missing_reason="酒店作为每日边界，不需要景点营业时段。",
                    source_reference_ids=anchor.source_reference_ids,
                )
            )
            continue
        category = _role_category(anchor.role)
        if category is None:
            raise PlanningGraphError("daily_scheduling", "fixed events require a mapped place")
        hours_scope = _optional_provider_scope(state, ProviderCode.AMAP)
        if hours_scope is None:
            places.append(
                SchedulePlaceFact(
                    place_id=anchor.place_id,
                    city_id=_city_id(state),
                    name=anchor.name,
                    category=category,
                    coordinates=anchor.coordinates,
                    recommended_duration_minutes=(
                        75 if category is PlaceCategory.ATTRACTION else 45
                    ),
                    opening_availability=DataAvailability.MISSING,
                    opening_missing_reason="该城市尚未配置营业时间 Provider。",
                    source_reference_ids=anchor.source_reference_ids,
                )
            )
            continue
        cancellation.raise_if_cancelled("hours_provider")
        canonical = _canonical_place(state, anchor.place_id)
        response = await hours.get_regular_hours(
            HoursRequest(
                place_id=anchor.place_id,
                city=hours_scope,
                name=anchor.name,
                address=canonical.address if canonical else None,
                coordinates=anchor.coordinates,
                service_dates=list(dates),
                source_place_ids=(
                    {item.provider: item.source_place_id for item in canonical.source_mappings}
                    if canonical
                    else {}
                ),
            )
        )
        regular = response.items[0] if response.items else None
        windows = _opening_windows(regular, dates) if regular else ()
        date_hours = evaluate_regular_hours(regular, dates) if regular else []
        known_hours = any(
            day.status in (HoursDayStatus.OPEN, HoursDayStatus.CLOSED) for day in date_hours
        )
        uncertain_hours = [
            day
            for day in date_hours
            if day.status in (HoursDayStatus.UNKNOWN, HoursDayStatus.CONFLICT)
        ]
        availability = (
            DataAvailability.PARTIAL
            if known_hours and uncertain_hours
            else DataAvailability.AVAILABLE
            if known_hours
            else DataAvailability.MISSING
        )
        sources = tuple(
            sorted(
                {
                    *anchor.source_reference_ids,
                    *(
                        (f"{regular.provider.value}:hours:{regular.source_place_id}",)
                        if regular is not None
                        else ()
                    ),
                }
            )
        )
        places.append(
            SchedulePlaceFact(
                place_id=anchor.place_id,
                city_id=_city_id(state),
                name=anchor.name,
                category=category,
                coordinates=anchor.coordinates,
                recommended_duration_minutes=75 if category is PlaceCategory.ATTRACTION else 45,
                opening_availability=availability,
                opening_windows=windows,
                opening_dates=tuple(
                    OpeningDateStatus(
                        service_date=day.service_date, status=day.status, reason=day.reason
                    )
                    for day in date_hours
                ),
                opening_missing_reason=(
                    None
                    if availability is DataAvailability.AVAILABLE
                    else (
                        "；".join(describe_date_hours(day) for day in uncertain_hours)[:500]
                        or "没有取得完整的逐日营业证据。"
                    )
                    if regular
                    else "没有取得可验证的逐日营业时段。"
                ),
                source_reference_ids=sources,
            )
        )
    routes = _schedule_routes(spatial)
    boundary = hotel.selected_hotel_place_id
    if boundary is None:
        boundary = spatial.anchors[0].place_id
    preferences = state.resolved_preferences
    return DailySchedulingRequest(
        request_id=_id(generation_id, "schedule"),
        trip_id=state.trip_id,
        input_state_version=state.state_version,
        business_date=business_date,
        city_id=_city_id(state),
        task_book=task_book,
        spatial_result=spatial,
        hotel_result=hotel,
        places=tuple(places),
        routes=routes,
        daily_windows=tuple(
            DailyScheduleWindow(
                service_date=service_date,
                start_time=_day_start(preferences.day_start if preferences else DayStart.AROUND_09),
                end_time=_day_return(
                    preferences.day_return if preferences else DayReturn.AROUND_21
                ),
                start_place_id=boundary,
                end_place_id=boundary,
            )
            for service_date in dates
        ),
        preferences=SchedulePreferences(
            pace_level=preferences.pace_level if preferences else 3,
            allowed_modes=_allowed_modes(state),
            maximum_walking_m_per_leg=_walking_limit(
                preferences.walking_tolerance if preferences else MobilityTolerance.AROUND_10
            ),
        ),
    )


def _schedule_routes(spatial: SpatialPlanningResult) -> tuple[ScheduleRouteFact, ...]:
    anchors = {item.node_id: item for item in spatial.anchors}
    values: list[ScheduleRouteFact] = []
    for edge in spatial.route_edges:
        origin = anchors[edge.origin_node_id].place_id
        destination = anchors[edge.destination_node_id].place_id
        for option in edge.routes:
            for left, right in ((origin, destination), (destination, origin)):
                values.append(
                    ScheduleRouteFact(
                        route_fact_id=f"amap:{edge.edge_id}:{left}:{right}:{option.mode.value}",
                        origin_place_id=left,
                        destination_place_id=right,
                        mode=option.mode,
                        availability=edge.status,
                        distance_m=option.distance_m,
                        duration_minutes=max(1, (option.duration_seconds + 59) // 60),
                        walking_m=option.walking_distance_m or 0,
                        polyline=(
                            option.polyline if left == origin else tuple(reversed(option.polyline))
                        ),
                        provider=option.provider,
                        source_reference_ids=(f"amap:spatial-edge:{edge.edge_id}",),
                        missing_reason=edge.missing_reason,
                    )
                )
    return tuple(values)


async def _cost_request(
    state: TripState,
    schedule: ScheduleArtifact,
    hotel: HotelSelectionResult,
    products: TravelProductProvider,
    cancellation: ModelCancellation,
    *,
    business_time: datetime,
    generation_id: UUID,
) -> CostEstimationRequest:
    facts: list[CostPriceFact] = []
    place_by_id = _all_places(state)
    ticket_cache: dict[tuple[UUID, date], ProviderTicketOffer | None] = {}
    for day in schedule.result.days:
        for activity in day.activities:
            if activity.kind is ScheduleActivityKind.ATTRACTION:
                key = (activity.place_id, day.service_date)
                if key not in ticket_cache:
                    cancellation.raise_if_cancelled("ticket_provider")
                    place = place_by_id.get(activity.place_id)
                    product_scope = _optional_provider_scope(state, ProviderCode.FLYAI)
                    if product_scope is None:
                        ticket_cache[key] = None
                    else:
                        response = await products.search_place_products(
                            ProductSearchRequest(
                                city=product_scope,
                                visit_date=day.service_date,
                                query=place.name if place else activity.title,
                            )
                        )
                        ticket_cache[key] = response.items[0] if response.items else None
                offer = ticket_cache[key]
                facts.append(
                    _price_fact(
                        activity.activity_id,
                        day.service_date,
                        CostCategory.ATTRACTION_TICKETS,
                        offer.price if offer else None,
                        offer.fetched_at if offer else business_time,
                        (
                            f"flyai:ticket:{offer.source_offer_id}"
                            if offer
                            else f"flyai:ticket-missing:{activity.place_id}"
                        ),
                    )
                )
            elif activity.kind is ScheduleActivityKind.RESTAURANT:
                facts.append(
                    _price_fact(
                        activity.activity_id,
                        day.service_date,
                        CostCategory.DINING,
                        None,
                        business_time,
                        f"dining:price-missing:{activity.place_id}",
                    )
                )
        for leg in day.transport_legs:
            facts.append(
                CostPriceFact(
                    price_fact_id=f"route-price:{leg.leg_id}",
                    subject_kind=CostSubjectKind.TRANSPORT_LEG,
                    subject_id=leg.leg_id,
                    service_date=day.service_date,
                    category=CostCategory.LOCAL_TRANSPORT,
                    basis=CostPriceBasis.PER_VEHICLE,
                    availability=DataAvailability.MISSING,
                    source_reference_ids=leg.source_reference_ids,
                    fetched_at=business_time,
                    missing_reason="路线 Provider 未返回可追溯票价。",
                )
            )
    if hotel.selected_hotel_place_id is not None:
        selected = next(
            item
            for item in hotel.candidates
            if item.hotel_place_id == hotel.selected_hotel_place_id
        )
        for service_date in _trip_dates(state)[:-1]:
            facts.append(
                CostPriceFact(
                    price_fact_id=f"hotel-price:{selected.hotel_place_id}:{service_date}",
                    subject_kind=CostSubjectKind.HOTEL_NIGHT,
                    subject_id=selected.hotel_place_id,
                    service_date=service_date,
                    category=CostCategory.LODGING,
                    basis=CostPriceBasis.PER_ROOM_NIGHT,
                    availability=(
                        DataAvailability.AVAILABLE
                        if selected.room_price is not None
                        else DataAvailability.MISSING
                    ),
                    original_amount=(
                        OriginalMoneyRange(
                            currency="CNY",
                            minimum_minor=selected.room_price.minimum_fen,
                            maximum_minor=selected.room_price.maximum_fen,
                        )
                        if selected.room_price is not None
                        else None
                    ),
                    source_reference_ids=selected.source_reference_ids,
                    fetched_at=selected.fetched_at,
                    missing_reason=(None if selected.room_price is not None else "酒店价格缺失。"),
                )
            )
    return CostEstimationRequest(
        request_id=_id(generation_id, "cost"),
        trip_id=state.trip_id,
        input_state_version=state.state_version,
        business_time=business_time,
        party_size=2,
        scheduling_request=schedule.request,
        schedule_result=schedule.result,
        price_facts=tuple(facts),
    )


def _price_fact(
    subject_id: UUID,
    service_date: date,
    category: CostCategory,
    amount: CnyAmountRange | None,
    fetched_at: datetime,
    source: str,
) -> CostPriceFact:
    usable = amount is not None
    original_amount = None
    if amount is not None:
        original_amount = OriginalMoneyRange(
            currency="CNY",
            minimum_minor=amount.minimum_fen,
            maximum_minor=amount.maximum_fen,
        )
    return CostPriceFact(
        price_fact_id=f"price:{subject_id}:{category.value}",
        subject_kind=CostSubjectKind.ACTIVITY,
        subject_id=subject_id,
        service_date=service_date,
        category=category,
        basis=CostPriceBasis.PER_PERSON,
        availability=DataAvailability.AVAILABLE if usable else DataAvailability.MISSING,
        original_amount=original_amount,
        source_reference_ids=(source,),
        fetched_at=fetched_at,
        missing_reason=None if usable else "当前没有可追溯的人均价格。",
    )


async def _weather_coverage(
    state: TripState,
    weather: WeatherProvider | None,
    registry: CityRegistry,
    cancellation: ModelCancellation,
) -> tuple[DailyWeatherCoverage, ...]:
    dates = _trip_dates(state)
    if weather is None:
        return tuple(
            DailyWeatherCoverage(
                service_date=value,
                availability=DataAvailability.MISSING,
                night_condition_available=False,
                missing_reason="当前运行模式没有天气 Provider。",
            )
            for value in dates
        )
    cancellation.raise_if_cancelled("weather_provider")
    weather_scope = _optional_provider_scope(state, ProviderCode.WEATHER)
    if weather_scope is None:
        return tuple(
            DailyWeatherCoverage(
                service_date=value,
                availability=DataAvailability.MISSING,
                night_condition_available=False,
                missing_reason="该城市尚未配置天气 Provider。",
            )
            for value in dates
        )
    response = await weather.get_forecast(
        WeatherRequest(
            city=weather_scope,
            start_date=dates[0],
            end_date=dates[-1],
        )
    )
    by_date = {item.forecast_date: item for item in response.items}
    return tuple(_weather_day(value, by_date.get(value), response.status) for value in dates)


def _weather_day(
    service_date: date,
    forecast: ProviderForecastDay | None,
    status: ProviderResultStatus,
) -> DailyWeatherCoverage:
    if forecast is None:
        return DailyWeatherCoverage(
            service_date=service_date,
            availability=DataAvailability.MISSING,
            night_condition_available=False,
            missing_reason="该日期没有天气预报。",
        )
    complete = forecast.condition_night is not None and status is ProviderResultStatus.SUCCESS
    return DailyWeatherCoverage(
        service_date=service_date,
        availability=DataAvailability.AVAILABLE if complete else DataAvailability.PARTIAL,
        night_condition_available=forecast.condition_night is not None,
        condition_day=forecast.condition_day,
        condition_night=forecast.condition_night,
        low_celsius=forecast.low_celsius,
        high_celsius=forecast.high_celsius,
        source_reference_ids=(f"{forecast.provider.value}:weather:{service_date}",),
        missing_reason=None if complete else "夜间天气或部分预报字段缺失。",
        fetched_at=forecast.fetched_at,
    )


def _opening_windows(
    regular: ProviderRegularHours,
    dates: tuple[date, ...],
) -> tuple[OpeningWindow, ...]:
    return tuple(
        OpeningWindow(
            service_date=day.service_date,
            start_time=period.opens_at,
            end_time=period.closes_at,
            last_entry_at=period.last_entry_at,
            source_reference_ids=(f"{regular.provider.value}:hours:{regular.source_place_id}",),
        )
        for day in evaluate_regular_hours(regular, dates)
        for period in day.intervals
    )


def _allowed_modes(state: TripState) -> tuple[RouteMode, ...]:
    preferences = state.resolved_preferences
    modes = [RouteMode.TRANSIT, RouteMode.DRIVING]
    if preferences is None or preferences.walking_tolerance is not MobilityTolerance.NEVER:
        modes.append(RouteMode.WALKING)
    if preferences is not None and preferences.bike_tolerance is not MobilityTolerance.NEVER:
        modes.append(RouteMode.CYCLING)
    return tuple(modes)


def _route_minutes(route: SpatialRouteOption | None) -> int | None:
    return max(1, (route.duration_seconds + 59) // 60) if route is not None else None


def _best_route(
    by_mode: Mapping[RouteMode, list[SpatialRouteOption]], mode: RouteMode
) -> SpatialRouteOption | None:
    values = by_mode[mode]
    return min(values, key=lambda item: item.duration_seconds) if values else None


def _role_category(role: AnchorRole) -> PlaceCategory | None:
    if role in {
        AnchorRole.MUST_ATTRACTION,
        AnchorRole.WANT_ATTRACTION,
        AnchorRole.CONVENIENT_ATTRACTION,
    }:
        return PlaceCategory.ATTRACTION
    if role in {AnchorRole.DESTINATION_RESTAURANT, AnchorRole.CONVENIENT_RESTAURANT}:
        return PlaceCategory.RESTAURANT
    return None


def _canonical_place(state: TripState, place_id: UUID) -> CanonicalPlace | None:
    return next((item for item in state.candidate_places if item.place_id == place_id), None)


def _provider_scope(state: TripState, provider: ProviderCode):  # type: ignore[no-untyped-def]
    from backend.planning.city_registry import default_city_registry

    return default_city_registry().provider_scope(_city_id(state), provider)


def _optional_provider_scope(state: TripState, provider: ProviderCode):  # type: ignore[no-untyped-def]
    try:
        return _provider_scope(state, provider)
    except CityProviderUnavailableError:
        return None


def _walking_limit(tolerance: MobilityTolerance) -> int:
    return {
        MobilityTolerance.NEVER: 0,
        MobilityTolerance.WITHIN_5: 400,
        MobilityTolerance.AROUND_10: 800,
        MobilityTolerance.FIFTEEN_PLUS: 1_500,
    }[tolerance]


def _day_start(value: DayStart) -> time:
    return {
        DayStart.BEFORE_07: time(7, 0),
        DayStart.AROUND_08: time(8, 0),
        DayStart.AROUND_09: time(9, 0),
        DayStart.AROUND_10: time(10, 0),
        DayStart.AFTER_11: time(11, 0),
        DayStart.FLEXIBLE: time(9, 0),
    }[value]


def _day_return(value: DayReturn) -> time:
    return {
        DayReturn.BEFORE_20: time(20, 0),
        DayReturn.AROUND_21: time(21, 0),
        DayReturn.AFTER_22: time(22, 0),
        DayReturn.FLEXIBLE: time(21, 0),
    }[value]


def _transport_mode(value: RouteMode) -> TransportMode:
    return {
        RouteMode.WALKING: TransportMode.WALK,
        RouteMode.CYCLING: TransportMode.BICYCLE,
        RouteMode.TRANSIT: TransportMode.PUBLIC_TRANSIT,
        RouteMode.DRIVING: TransportMode.TAXI,
    }[value]


# Replay/fake providers are assembled from the stable normalized state, never from a fixed story.
def stored_planning_providers(
    state: TripState,
    *,
    clock: Callable[[], datetime],
) -> PlanningProviderSet:
    from backend.providers.stored_planning import (
        StoredHoursProvider,
        StoredPlaceProvider,
        StoredProductProvider,
        StoredRouteProvider,
        StoredWeatherProvider,
    )

    return PlanningProviderSet(
        places=StoredPlaceProvider(state, clock=clock),
        routes=StoredRouteProvider(state, clock=clock),
        hours=StoredHoursProvider(state, clock=clock),
        products=StoredProductProvider(state, clock=clock),
        weather=StoredWeatherProvider(state, clock=clock),
    )
