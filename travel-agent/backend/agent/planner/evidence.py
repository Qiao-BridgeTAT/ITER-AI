"""Live Provider capability adapters. No itinerary selection or materialization."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable
from datetime import UTC, date, datetime, timedelta
from typing import Literal
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from backend.agent.model_gateway import ModelCancellation, ModelGateway
from backend.agent.planner.dependencies import (
    prepare_workspace_for_evidence_refresh,
    rebind_semantic_artifacts_after_evidence,
)
from backend.agent.planner.dining_context import deduplicate_inherited_dining, dining_place_blocked
from backend.agent.planner.dining_recall import recall_initial_dining
from backend.agent.planner.materializer import _refresh_route_mode, _select_route
from backend.agent.planner.observation_router import EvidenceUpdate, observe_capability_results
from backend.agent.planner.route_diagnostics import route_needs_retry
from backend.agent.planner.spatial import build_spatial_observation
from backend.agent.planner.ticket_prices import lookup_ticket_prices
from backend.agent.planner.timing_quality import (
    afternoon_activity_opportunities,
    evening_activity_opportunities,
    minutes,
    schedule_coverage_issues,
    schedule_quality_gaps,
)
from backend.agent.planner.visit_identity import is_explicit_internal_subsite, is_named_subsite
from backend.agent.planner.workspace import (
    PlannerGuardError,
    advance,
    entity_intents,
    readiness,
    refresh_pool,
    server_id,
    service_dates,
    task_book_references,
)
from backend.contracts.candidate_ranking import CandidateRankingRequest
from backend.contracts.candidate_recall import (
    CandidateDomain,
    CandidateRecallBudget,
    CandidateRecallRequest,
    DomainRecallBudget,
    RecallAnchor,
    RecallSourceKind,
)
from backend.contracts.enums import PlaceCategory, ProviderCode
from backend.contracts.v4.enums import CandidateEntityKind, CommitmentLevel, PlannerCapability
from backend.contracts.v4.planner_decision import RequestEvidencePayload
from backend.contracts.v4.planner_dining import PlannerDiningState
from backend.contracts.v4.planner_draft import DraftItem
from backend.contracts.v4.planner_evidence import (
    PlannerCapabilityObservation,
    PlannerHoursEvidence,
    PlannerPlaceEvidence,
    PlannerTicketEvidence,
    PlannerWeatherEvidence,
)
from backend.contracts.v4.planner_observations import (
    CandidateRecallArguments,
    HotelSearchArguments,
    OpeningHoursArguments,
    PlaceFactsArguments,
    PlannerCapabilityRequest,
    SpatialRouteEndpoint,
    SpatialRoutePair,
    SpatialRoutesArguments,
    TicketAvailabilityArguments,
    WeatherForecastArguments,
)
from backend.contracts.v4.planner_refs import (
    CandidateRef,
    FixedCommitmentRef,
    require_same_scope_ownership,
)
from backend.contracts.v4.planner_strategy import CandidatePoolEntry
from backend.contracts.v4.planner_workspace import PlannerWorkspaceState, VerifiedFactSummary
from backend.contracts.v4.task_book import TaskBookEntityIntent, TaskBookV4
from backend.persistence.outbox_repository import canonical_json_hash
from backend.planning.candidate_ranking import CandidateRankingService
from backend.planning.candidate_recall import CandidateRecallService
from backend.planning.city_registry import (
    CityProviderUnavailableError,
    CityRegistry,
    default_city_registry,
)
from backend.planning.recall_plan import ModelRecallPlanGenerator
from backend.planning.runtime_backend import PlanningProviderSet
from backend.providers.contracts import (
    HoursRequest,
    KeywordPlaceSearchRequest,
    NearbyPlaceSearchRequest,
    PlaceDetailRequest,
    ProviderCityScope,
    ProviderError,
    ProviderPlace,
    WeatherRequest,
)
from backend.providers.hours_rules import evaluate_regular_hours
from backend.providers.place_copy import provider_place_cuisine
from backend.providers.place_taxonomy import category_from_original_typecodes
from backend.providers.request_budget import RequestBudgetExceeded

MAX_SELECTED_ROUTE_PAIRS_PER_REQUEST = 40
MAX_SELECTED_ROUTE_REQUESTS = 4
RoutePreference = Literal["public_transit", "taxi", "walking", "driving"]


def _guard_repeated_evidence_request(
    request: PlannerCapabilityRequest, workspace: PlannerWorkspaceState
) -> None:
    """Unknown Provider data is an observation, not permission for a retry loop."""

    arguments = request.arguments
    if isinstance(arguments, OpeningHoursArguments):
        attempted_pairs = {
            (reference.candidate_id, service_date)
            for observation in workspace.capability_observations
            if observation.capability is PlannerCapability.OPENING_HOURS
            for reference in observation.candidate_refs
            for service_date in observation.service_dates
        }
        requested_pairs = {
            (reference.candidate_id, service_date)
            for reference in arguments.candidate_refs
            for service_date in arguments.service_dates
        }
        if requested_pairs and requested_pairs <= attempted_pairs:
            raise PlannerGuardError("planner_opening_hours_already_observed")

    if isinstance(arguments, HotelSearchArguments):
        observation = workspace.hotel_observation
        if observation is None or observation.mode != "search":
            return
        if observation.query_origin == "prepare_handoff":
            # Prepare searched its chosen area keywords, not these activity anchors.
            return
        if arguments.search_keyword != observation.search_keyword:
            return
        requested_clusters = set(arguments.activity_cluster_refs)
        if any(
            segment.check_in_date == arguments.check_in_date
            and segment.check_out_date == arguments.check_out_date
            and set(segment.activity_cluster_ids) == requested_clusters
            for segment in observation.stay_segments
        ):
            raise PlannerGuardError("planner_hotel_search_already_observed")


def capability_request_fingerprint(request: PlannerCapabilityRequest) -> str:
    """Canonical semantic identity, excluding server request IDs and workspace revisions."""

    return canonical_json_hash(
        {
            "capability": request.capability.value,
            "purpose": request.purpose,
            "service_dates": [item.isoformat() for item in request.service_dates],
            "based_on_issue_ids": request.based_on_issue_ids,
            "blocking": request.blocking,
            "arguments": request.arguments.model_dump(mode="json"),
        }
    )


class PlannerEvidenceBackend:
    def __init__(
        self,
        providers: PlanningProviderSet,
        gateway: ModelGateway,
        *,
        registry: CityRegistry | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        dining_review_gateway: ModelGateway | None = None,
    ) -> None:
        self.providers = providers
        self.registry = registry or default_city_registry()
        self.clock = clock
        self.gateway = gateway
        self.dining_review_gateway = dining_review_gateway
        self.recall = CandidateRecallService(
            registry=self.registry,
            places=providers.places,
            # Planner candidate hypotheses must remain usable for every
            # registered city. Most cities do not have a reviewed content
            # package, so expose the same Provider-only, server-bounded search
            # contract used by V4 discovery instead of allowing content queries
            # that cannot be executed for that destination.
            plan_generator=ModelRecallPlanGenerator(gateway, provider_only=True),
        )
        self.ranking = CandidateRankingService()

    async def initialize(
        self,
        workspace: PlannerWorkspaceState,
        book: TaskBookV4,
        cancellation: ModelCancellation,
        *,
        checkpoint: Callable[[PlannerWorkspaceState], Awaitable[None]] | None = None,
        agent_driven: bool = False,
    ) -> PlannerWorkspaceState:
        city = self.registry.provider_scope(
            book.destination_and_dates.destination_name, ProviderCode.AMAP
        )
        intents = entity_intents(book)
        origins = {item.canonical_entity_id: item for item in workspace.candidate_origins}
        seeds: dict[str, tuple[TaskBookEntityIntent | str, CandidateEntityKind, str | None]] = {
            key: (item, kind, origins[key].provider_entity_id if key in origins else None)
            for key, (item, kind) in intents.items()
        }
        for origin in workspace.candidate_origins:
            if (
                origin.inherit_as_neutral
                and origin.display_name
                and origin.canonical_entity_id not in seeds
            ):
                seeds[origin.canonical_entity_id] = (
                    origin.display_name,
                    origin.entity_kind,
                    origin.provider_entity_id,
                )
        if workspace.dining_state is not None:
            for place in workspace.place_evidence:
                if (
                    place.entity_kind is CandidateEntityKind.RESTAURANT
                    and place.canonical_entity_id in workspace.dining_state.admitted_canonical_ids
                ):
                    seeds.setdefault(
                        place.canonical_entity_id,
                        (place.display_name, place.entity_kind, place.provider_entity_id),
                    )
        semaphore = asyncio.Semaphore(3)
        previous_places = {p.canonical_entity_id: p for p in workspace.place_evidence}

        async def resolve(key: str) -> tuple[PlannerPlaceEvidence | None, str]:
            item, kind, provider_id = seeds[key]
            if isinstance(item, TaskBookEntityIntent) and item.disposition.value == "avoid":
                return None, "excluded"
            async with semaphore:
                cancellation.raise_if_cancelled("planner_initial_place")
                previous = previous_places.get(key)
                if (
                    agent_driven
                    and previous is not None
                    and previous.provider_entity_id == provider_id
                    and previous.city_id == city.city_id
                    and previous.entity_kind is kind
                    and category_from_original_typecodes(previous.provider_typecode)
                    is _category(kind)
                    and timedelta(0) <= self.clock() - previous.observed_at < timedelta(hours=24)
                    and not (
                        kind is CandidateEntityKind.RESTAURANT
                        and dining_place_blocked(previous, book)
                    )
                ):
                    return previous, "reused_fresh_prepared_identity"
                place, code = await self._resolve_selected_place(
                    key,
                    item,
                    kind,
                    city,
                    provider_id,
                    cancellation,
                )
                if (
                    place is not None
                    and kind is CandidateEntityKind.RESTAURANT
                    and dining_place_blocked(place, book)
                ):
                    return None, "excluded_or_known_unavailable"
                previous = previous_places.get(key)
                if (
                    place is None
                    and kind is CandidateEntityKind.RESTAURANT
                    and previous is not None
                    and previous.city_id == city.city_id
                    and "known_identity_invalid" not in code
                    and not dining_place_blocked(previous, book)
                ):
                    return previous, f"retained_previous_evidence:{code}"
                return place, code

        resolved = await asyncio.gather(*(resolve(key) for key in seeds))
        places = tuple(place for place, _ in resolved if place is not None)
        resolved_identities = {
            (place.canonical_entity_id, place.provider_entity_id) for place in places
        }
        observations = tuple(
            PlannerCapabilityObservation(
                observation_id=str(uuid4()),
                scope=workspace.current_scope,
                request_id=server_id(workspace.generation_id, "initial-place", key),
                capability=PlannerCapability.PLACE_FACTS,
                status="complete" if place else "unavailable",
                fact_reference_ids=(place.fact_reference_id,) if place else (),
                reason_summary=f"任务书实体身份核验：{code}；仅精确实体、原始类型与城市一致时接受。",
                observed_at=self.clock(),
            )
            for key, (place, code) in zip(seeds, resolved, strict=True)
            if code != "excluded"
        )
        workspace = advance(
            workspace,
            place_evidence=places,
            dining_state=(workspace.dining_state or PlannerDiningState()).model_copy(
                update={
                    "initial_threshold": 3 * len(service_dates(book)) + 2,
                    "inherited_canonical_ids": tuple(
                        p.canonical_entity_id
                        for p in places
                        if p.entity_kind is CandidateEntityKind.RESTAURANT
                    ),
                    "admitted_canonical_ids": tuple(
                        p.canonical_entity_id
                        for p in places
                        if p.entity_kind is CandidateEntityKind.RESTAURANT
                    ),
                }
            ),
            # A full replan replaces the initial identity set. Discard hours
            # for removed supplementary POIs or a changed provider identity in
            # the SAME atomic update, before workspace ownership validation.
            # Kept identities retain only fresh, date-compatible evidence; new
            # selected places are queried below / by ensure_selected_hours.
            hours_evidence=tuple(
                hours
                for hours in workspace.hours_evidence
                if (hours.canonical_entity_id, hours.provider_entity_id) in resolved_identities
                and hours.expires_at > self.clock()
                and {d.service_date for d in hours.days} == set(service_dates(book))
            ),
            capability_observations=(*workspace.capability_observations, *observations),
        )
        workspace = deduplicate_inherited_dining(workspace, book)
        workspace = refresh_pool(workspace, book, self.clock())
        if checkpoint is not None:
            await checkpoint(workspace)
        if agent_driven:
            workspace = await build_spatial_observation(
                workspace,
                book,
                routes=self.providers.routes,
                city=city,
                cancellation=cancellation,
                now=self.clock(),
                query_routes=False,
            )
            return advance(workspace, initial_evidence_ready=True)
        # Initial recall fills an observation pool, not a daily plan. All selected
        # identities remain protected; the model can request further gaps later.
        recall_snapshot = workspace
        recall_requests: list[PlannerCapabilityRequest] = []
        for kind, minimum in (
            (CandidateEntityKind.ATTRACTION, min(14, 3 * len(service_dates(book)))),
        ):
            count = sum(place.entity_kind is kind for place in recall_snapshot.place_evidence)
            if count < minimum:
                args = CandidateRecallArguments(
                    domain=kind,
                    gap_code="initial_evidence",
                    task_book_preference_refs=(
                        "goal:0" if "goal:0" in task_book_references(book) else "destination",
                    ),
                    limit=min(14, minimum + 2),
                )
                request = PlannerCapabilityRequest(
                    request_id=str(uuid4()),
                    scope=recall_snapshot.current_scope,
                    capability=PlannerCapability.CANDIDATE_RECALL,
                    purpose="complete_initial_evidence",
                    blocking=True,
                    arguments=args,
                )
                recall_requests.append(request)
        if recall_requests:
            # Both domain gaps are independent and must observe the same pool
            # snapshot. asyncio.gather preserves request order even when the
            # Provider calls complete in the opposite order, so one merge gives
            # deterministic observations and one checkpoint.
            recall_updates, workspace = await asyncio.gather(
                asyncio.gather(
                    *(
                        self._execute(request, recall_snapshot, book, cancellation)
                        for request in recall_requests
                    )
                ),
                self._initialize_dining(recall_snapshot, book, cancellation, checkpoint=checkpoint),
            )
            workspace = self._merge_updates(workspace, book, tuple(recall_updates))
            if checkpoint is not None:
                await checkpoint(workspace)
        else:
            workspace = await self._initialize_dining(
                workspace, book, cancellation, checkpoint=checkpoint
            )
        calendar_targets = tuple(
            entry.candidate_ref.candidate_id
            for entry in workspace.candidate_pool.candidates
            if entry.entity_kind is CandidateEntityKind.ATTRACTION
            or entry.commitment_level is CommitmentLevel.STRONG
        )
        calendar_requests = []
        current_candidates = workspace.candidate_pool.candidate_by_id()
        for offset in range(0, len(calendar_targets), 20):
            # Every independent lookup observes the same immutable pool. Merge
            # only after all responses arrive so concurrent writes cannot erase
            # another batch or leave references on different pool revisions.
            targets = tuple(
                current_candidates[key].candidate_ref
                for key in calendar_targets[offset : offset + 20]
            )
            request = PlannerCapabilityRequest(
                request_id=str(uuid4()),
                scope=workspace.current_scope,
                capability=PlannerCapability.OPENING_HOURS,
                purpose="complete_initial_evidence",
                service_dates=service_dates(book),
                blocking=True,
                arguments=OpeningHoursArguments(
                    candidate_refs=targets, service_dates=service_dates(book)
                ),
            )
            calendar_requests.append(request)
        trip_dates = service_dates(book)
        weather_requests = [
            PlannerCapabilityRequest(
                request_id=server_id(
                    workspace.generation_id,
                    "initial-weather",
                    dates[0].isoformat(),
                    dates[-1].isoformat(),
                ),
                scope=workspace.current_scope,
                capability=PlannerCapability.WEATHER_FORECAST,
                purpose="complete_initial_evidence",
                service_dates=dates,
                blocking=False,
                arguments=WeatherForecastArguments(
                    destination_ref="destination",
                    service_dates=dates,
                    weather_fields=(
                        "condition_day",
                        "condition_night",
                        "high_celsius",
                        "low_celsius",
                    ),
                ),
            )
            for offset in range(0, len(trip_dates), 5)
            if (dates := trip_dates[offset : offset + 5])
        ]
        evidence_updates, spatial_workspace = await asyncio.gather(
            asyncio.gather(
                *(
                    self._execute(request, workspace, book, cancellation)
                    for request in (*calendar_requests, *weather_requests)
                )
            ),
            self._spatial(workspace, book, cancellation),
        )
        # Opening dates do not change geometry. The calendar-only merge rebinds
        # the real route clusters to the refreshed pool without querying again.
        workspace = spatial_workspace
        if evidence_updates:
            workspace = self._merge_updates(workspace, book, tuple(evidence_updates))
            if checkpoint is not None:
                await checkpoint(workspace)
        return advance(
            workspace,
            readiness_observation=readiness(workspace, book, self.clock()),
            initial_evidence_ready=True,
        )

    async def _initialize_dining(
        self,
        workspace: PlannerWorkspaceState,
        book: TaskBookV4,
        cancellation: ModelCancellation,
        *,
        checkpoint: Callable[[PlannerWorkspaceState], Awaitable[None]] | None = None,
    ) -> PlannerWorkspaceState:
        return await recall_initial_dining(
            workspace,
            book,
            cancellation,
            gateway=self.gateway,
            review_gateway=self.dining_review_gateway,
            search_places=self.providers.places.search_places,
            city=self.registry.provider_scope(
                book.destination_and_dates.destination_name, ProviderCode.AMAP
            ),
            normalize=lambda place: _place_evidence(place, CandidateEntityKind.RESTAURANT),
            clock=self.clock,
            checkpoint=checkpoint,
        )

    async def _resolve_selected_place(
        self,
        canonical_id: str,
        intent: TaskBookEntityIntent | str,
        kind: CandidateEntityKind,
        city: ProviderCityScope,
        provider_id: str | None,
        cancellation: ModelCancellation,
    ) -> tuple[PlannerPlaceEvidence | None, str]:
        failures = []
        # Some POIs are unavailable from detail while text search still supplies
        # the exact identity. Neither source may substitute a same-name entity.
        for mode in ("detail", "search") if provider_id else ("search",):
            cancellation.raise_if_cancelled("planner_resolve_selected_place")
            try:
                response = (
                    await self.providers.places.get_place(
                        PlaceDetailRequest(
                            city=city,
                            source_place_id=provider_id,
                        )
                    )
                    if mode == "detail" and provider_id
                    else await self.providers.places.search_places(
                        KeywordPlaceSearchRequest(
                            city=city,
                            query=intent if isinstance(intent, str) else intent.display_name,
                            category_hint=_category(kind),
                            page_size=25,
                        )
                    )
                )
            except ProviderError as error:
                failures.append(f"{mode}:{error.code.value}")
                continue
            for place in response.items:
                if str(uuid5(NAMESPACE_URL, f"amap:{place.source_place_id}")) == canonical_id:
                    if place.city_id == city.city_id and _real_category(place, kind):
                        return _place_evidence(place, kind), f"verified_by_{mode}"
                    return None, "known_identity_invalid"
            failures.append(f"{mode}:exact_identity_or_type_missing")
        return None, ";".join(failures)

    async def execute_batch(
        self,
        requests: tuple[PlannerCapabilityRequest, ...],
        workspace: PlannerWorkspaceState,
        book: TaskBookV4,
        cancellation: ModelCancellation,
    ) -> PlannerWorkspaceState:
        self.validate_batch(requests, workspace, book)
        semaphore = asyncio.Semaphore(4)

        async def execute(request: PlannerCapabilityRequest) -> EvidenceUpdate:
            cached = self._cached_update(request, workspace)
            if cached is not None:
                return cached
            async with semaphore:
                return await self._execute(request, workspace, book, cancellation)

        updates = await asyncio.gather(*(execute(request) for request in requests))
        stable_updates = tuple(
            update
            for _, update in sorted(
                zip(requests, updates, strict=True),
                key=lambda pair: (
                    pair[0].capability.value,
                    capability_request_fingerprint(pair[0]),
                    pair[0].request_id,
                ),
            )
        )
        result = self._merge_updates(workspace, book, stable_updates)
        if result.spatial_observation is None:
            result = await self._spatial(result, book, cancellation)
        return advance(
            result,
            readiness_observation=readiness(result, book, self.clock()),
            segment_evidence_count=workspace.segment_evidence_count + 1,
        )

    async def ensure_selected_itinerary_routes(
        self,
        workspace: PlannerWorkspaceState,
        book: TaskBookV4,
        cancellation: ModelCancellation,
    ) -> PlannerWorkspaceState:
        """Observe only the exact route legs selected by the compiled itinerary.

        Initial spatial evidence is deliberately sparse. This second pass runs
        after the semantic plan and single lodging baseline are fixed, so hotel
        boundaries and activity-to-activity legs are queried using their exact
        endpoints rather than cluster representatives.
        """

        refresh_baseline = (
            workspace.materialized_schedule
            if workspace.react_state and workspace.react_state.route_refresh_only
            else None
        )
        for _ in range(4 if workspace.react_state is not None else 1):
            requests = selected_itinerary_route_requests(workspace, book, now=self.clock())
            if not requests:
                break
            for request in requests:
                # Follow the declared preference order. Only failed/unusable
                # primary routes require the next permitted mode. Independent
                # legs remain parallel inside the provider operation.
                self._validate_request(request, workspace, book)
                update = await self._execute(request, workspace, book, cancellation)
                workspace = self._merge_updates(workspace, book, (update,))
                if refresh_baseline is not None:
                    workspace = workspace.model_copy(
                        update={"materialized_schedule": refresh_baseline}
                    )
                workspace = advance(
                    workspace,
                    readiness_observation=readiness(workspace, book, self.clock()),
                )
        return workspace

    async def ensure_selected_hours(
        self,
        workspace: PlannerWorkspaceState,
        book: TaskBookV4,
        cancellation: ModelCancellation,
    ) -> PlannerWorkspaceState:
        """Check newly selected places, not only the initial must-visit pool.

        One bounded lookup per identity/date; unknown and failed lookups are
        completed observations, never a reason to retry the same request.
        """
        if workspace.working_itinerary is None:
            return workspace
        targets: dict[CandidateRef, set[date]] = {}
        for day in workspace.working_itinerary.days:
            for item in day.ordered_items:
                if isinstance(item.object_ref, CandidateRef):
                    targets.setdefault(item.object_ref, set()).add(day.service_date)
        return await self.ensure_candidate_hours(workspace, book, cancellation, targets)

    async def ensure_candidate_hours(
        self,
        workspace: PlannerWorkspaceState,
        book: TaskBookV4,
        cancellation: ModelCancellation,
        requested: dict[CandidateRef, set[date]],
    ) -> PlannerWorkspaceState:
        """Verify a bounded candidate/date set before choosing optional gap visits.

        This only supplies facts; it never selects a place or writes a schedule.
        Both selected-path and pre-choice lookups share completion/de-duplication.
        """
        observed = {
            (ref.canonical_entity_id, day)
            for observation in workspace.capability_observations
            if observation.capability is PlannerCapability.OPENING_HOURS
            for ref in observation.candidate_refs
            for day in observation.service_dates
        }
        observed.update(
            (hours.canonical_entity_id, day.service_date)
            for hours in workspace.hours_evidence
            if hours.expires_at > self.clock()
            for day in hours.days
        )
        targets = {
            ref: {day for day in dates if (ref.canonical_entity_id, day) not in observed}
            for ref, dates in requested.items()
        }
        targets = {ref: dates for ref, dates in targets.items() if dates}
        semaphore = asyncio.Semaphore(4)

        async def lookup(ref: CandidateRef, dates: set[date]) -> EvidenceUpdate | None:
            request_dates = tuple(sorted(dates))
            request = PlannerCapabilityRequest(
                request_id=server_id(
                    workspace.generation_id, ref.candidate_id, *request_dates, "selected-hours-v1"
                ),
                scope=workspace.current_scope,
                capability=PlannerCapability.OPENING_HOURS,
                purpose="complete_initial_evidence",
                service_dates=request_dates,
                blocking=False,
                arguments=OpeningHoursArguments(candidate_refs=(ref,), service_dates=request_dates),
            )
            async with semaphore:
                self._validate_request(request, workspace, book)
                try:
                    return await self._execute(request, workspace, book, cancellation)
                except RequestBudgetExceeded:
                    if workspace.react_state is None:
                        raise
                    # Preserve completed siblings. An unexecuted lookup gets no
                    # receipt or fact, so it cannot masquerade as an unknown result.
                    return None

        updates = await asyncio.gather(*(lookup(ref, dates) for ref, dates in targets.items()))
        completed = tuple(update for update in updates if update is not None)
        return self._merge_updates(workspace, book, completed) if completed else workspace

    async def ensure_selected_prices(
        self,
        workspace: PlannerWorkspaceState,
        book: TaskBookV4,
        cancellation: ModelCancellation,
    ) -> PlannerWorkspaceState:
        """One bounded, parallel price lookup per selected attraction/date.

        Failed lookups are receipts too, so missing prices never drive a retry loop.
        Names are bound by the adapter; the model cannot supply amounts.
        """
        draft = workspace.working_itinerary
        if draft is None:
            return workspace
        observed = {
            (ref.canonical_entity_id, day)
            for observation in workspace.capability_observations
            if observation.capability is PlannerCapability.TICKET_AVAILABILITY
            for ref in observation.candidate_refs
            for day in observation.service_dates
        }
        requests = []
        for day in draft.days:
            for item in day.ordered_items:
                ref = item.object_ref
                if (
                    not isinstance(ref, CandidateRef)
                    or ref.entity_kind is not CandidateEntityKind.ATTRACTION
                    or (ref.canonical_entity_id, day.service_date) in observed
                ):
                    continue
                observed.add((ref.canonical_entity_id, day.service_date))
                requests.append(
                    PlannerCapabilityRequest(
                        request_id=server_id(
                            workspace.generation_id, ref.candidate_id, day.service_date, "price"
                        ),
                        scope=workspace.current_scope,
                        capability=PlannerCapability.TICKET_AVAILABILITY,
                        purpose="complete_initial_evidence",
                        service_dates=(day.service_date,),
                        blocking=False,
                        arguments=TicketAvailabilityArguments(
                            party_size_ref="party",
                            candidate_refs=(ref,),
                            service_dates=(day.service_date,),
                        ),
                    )
                )
        semaphore = asyncio.Semaphore(4)

        async def execute(request: PlannerCapabilityRequest) -> EvidenceUpdate:
            async with semaphore:
                self._validate_request(request, workspace, book)
                return await self._execute(request, workspace, book, cancellation)

        updates, workspace = await asyncio.gather(
            asyncio.gather(*(execute(request) for request in requests)),
            self._ensure_selected_business_facts(workspace, book, cancellation),
        )
        return self._merge_updates(workspace, book, tuple(updates)) if updates else workspace

    async def _ensure_selected_business_facts(
        self, workspace: PlannerWorkspaceState, book: TaskBookV4, cancellation: ModelCancellation
    ) -> PlannerWorkspaceState:
        """Enrich selected identities once, without invalidating unchanged geometry.

        A missing price is a completed lookup, not a retry request. Detail prices
        have their own provenance; recall coordinates and route refs stay intact.
        """
        assert workspace.working_itinerary is not None
        selected = {
            item.object_ref.canonical_entity_id: item.object_ref
            for day in workspace.working_itinerary.days
            for item in day.ordered_items
            if isinstance(item.object_ref, CandidateRef)
        }
        attempted = {item.request_id for item in workspace.capability_observations}
        missing = [
            place
            for place in workspace.place_evidence
            if place.canonical_entity_id in selected
            and (place.average_cost is None or place.rating is None)
            and server_id(workspace.generation_id, place.canonical_entity_id, "business-price-v1")
            not in attempted
        ]
        if not missing:
            return workspace
        city = self.registry.provider_scope(
            book.destination_and_dates.destination_name, ProviderCode.AMAP
        )
        semaphore = asyncio.Semaphore(4)

        async def lookup(
            expected: PlannerPlaceEvidence,
        ) -> tuple[PlannerPlaceEvidence, ProviderPlace | None]:
            async with semaphore:
                cancellation.raise_if_cancelled("planner_selected_business_facts")
                try:
                    response = await self.providers.places.get_place(
                        PlaceDetailRequest(city=city, source_place_id=expected.provider_entity_id)
                    )
                except ProviderError:
                    return expected, None
                exact = next(
                    (
                        place
                        for place in response.items
                        if place.source_place_id == expected.provider_entity_id
                        and place.city_id == expected.city_id
                        and _real_category(place, expected.entity_kind)
                    ),
                    None,
                )
                return expected, exact

        results = await asyncio.gather(*(lookup(place) for place in missing))
        places = {place.canonical_entity_id: place for place in workspace.place_evidence}
        facts = {fact.fact_reference_id: fact for fact in workspace.verified_facts}
        observations = list(workspace.capability_observations)
        for expected, detail in results:
            fact_id = None
            if detail is not None and (
                detail.average_cost is not None or detail.rating is not None
            ):
                fact_id = server_id(
                    "place-business", detail.source_place_id, detail.fetched_at.isoformat()
                )
                replaces_cost = detail.average_cost is not None or expected.average_cost is None
                places[expected.canonical_entity_id] = expected.model_copy(
                    update={
                        "average_cost": detail.average_cost or expected.average_cost,
                        "rating": detail.rating if detail.rating is not None else expected.rating,
                        "business_fact_reference_id": fact_id
                        if replaces_cost
                        else expected.business_fact_reference_id,
                        "business_observed_at": detail.fetched_at
                        if replaces_cost
                        else expected.business_observed_at,
                    }
                )
                facts[fact_id] = VerifiedFactSummary(
                    fact_reference_id=fact_id,
                    fact_kind="place_business",
                    safe_summary="同一高德地点的评分与人均参考消费；消费不等于门票或成交价。",
                    observed_at=detail.fetched_at,
                    source_reference_ids=(f"amap:{detail.source_place_id}",),
                )
            request_id = server_id(
                workspace.generation_id, expected.canonical_entity_id, "business-price-v1"
            )
            observations.append(
                PlannerCapabilityObservation(
                    observation_id=server_id(request_id, "receipt"),
                    scope=workspace.current_scope,
                    request_id=request_id,
                    capability=PlannerCapability.PLACE_FACTS,
                    status="complete"
                    if detail is not None and detail.average_cost is not None
                    else "partial",
                    candidate_refs=(selected[expected.canonical_entity_id],),
                    fact_reference_ids=(fact_id,) if fact_id else (),
                    reason_summary="已完成一次地点消费与评分补查；未返回价格的项目保持未知。",
                    observed_at=self.clock(),
                )
            )
        return advance(
            workspace,
            place_evidence=tuple(places.values()),
            verified_facts=tuple(facts.values()),
            capability_observations=tuple(observations),
        )

    def validate_batch(
        self,
        requests: tuple[PlannerCapabilityRequest, ...],
        workspace: PlannerWorkspaceState,
        book: TaskBookV4,
    ) -> None:
        """Validate before accepting/checkpointing a tool plan, and again before execution."""
        if not 1 <= len(requests) <= 4 or len({item.request_id for item in requests}) != len(
            requests
        ):
            raise PlannerGuardError("planner_invalid_capability_batch")
        fingerprints = [capability_request_fingerprint(item) for item in requests]
        if len(set(fingerprints)) != len(fingerprints):
            raise PlannerGuardError("planner_duplicate_capability_request")
        if workspace.segment_evidence_count >= 4:
            raise PlannerGuardError("planner_evidence_budget_exhausted")
        hotel_requests = [
            request
            for request in requests
            if request.capability
            in {PlannerCapability.HOTEL_SEARCH, PlannerCapability.HOTEL_OFFER_REFRESH}
        ]
        if len(hotel_requests) > 1:
            raise PlannerGuardError("planner_hotel_requests_must_not_overwrite_same_observation")
        # Validate every request against one snapshot before any side effect.
        for request in requests:
            self._validate_request(request, workspace, book)

    def _validate_request(
        self,
        request: PlannerCapabilityRequest,
        workspace: PlannerWorkspaceState,
        book: TaskBookV4,
        *,
        cached_observation: bool = False,
    ) -> None:
        from backend.agent.planner.location_capabilities import validate_location_request

        if not cached_observation:
            _guard_repeated_evidence_request(request, workspace)
        validate_location_request(request, workspace, book)
        require_same_scope_ownership(request.scope, workspace.current_scope)
        if request.scope.workspace_revision > workspace.workspace_revision:
            raise PlannerGuardError("planner_future_request_scope")
        if not set(request.service_dates) <= set(service_dates(book)):
            raise PlannerGuardError("planner_request_outside_trip_dates")
        args = request.arguments
        if isinstance(
            args, (PlaceFactsArguments, OpeningHoursArguments, TicketAvailabilityArguments)
        ):
            for reference in args.candidate_refs:
                self._entry(reference, workspace)
        if isinstance(args, CandidateRecallArguments):
            if args.domain is CandidateEntityKind.ACTIVITY:
                raise PlannerGuardError("planner_activity_recall_not_supported")
            if not set(args.task_book_preference_refs) <= task_book_references(book).keys():
                raise PlannerGuardError("planner_unknown_preference_reference")
            for reference in args.nearby_candidate_refs:
                self._entry(reference, workspace)
            known_clusters = (
                {cluster.cluster_id for cluster in workspace.spatial_observation.clusters}
                if workspace.spatial_observation
                else set()
            )
            if set(args.nearby_cluster_refs) - known_clusters:
                raise PlannerGuardError("planner_unknown_cluster")
        if isinstance(args, WeatherForecastArguments) and args.destination_ref != "destination":
            raise PlannerGuardError("planner_unknown_destination_reference")
        if isinstance(args, TicketAvailabilityArguments) and args.party_size_ref != "party":
            raise PlannerGuardError("planner_unknown_party_reference")
        if request.purpose == "resolve_validation_issue":
            issues = (
                {item.issue_id for item in workspace.validation_observation.issues}
                if workspace.validation_observation
                else set()
            )
            if workspace.readiness_observation is not None:
                issues.update(item.issue_id for item in workspace.readiness_observation.issues)
            if not set(request.based_on_issue_ids) <= issues:
                raise PlannerGuardError("planner_unknown_validation_issue")

    def _cached_update(
        self,
        request: PlannerCapabilityRequest,
        workspace: PlannerWorkspaceState,
    ) -> EvidenceUpdate | None:
        """Reuse a fresh normalized receipt; never replay a Provider payload."""

        target = capability_request_fingerprint(request)
        requests_by_id = {
            candidate.request_id: candidate
            for decision in workspace.decision_trace
            if isinstance(decision.payload, RequestEvidencePayload)
            for candidate in decision.payload.capability_requests
            if candidate.request_id != request.request_id
        }
        observations = {
            item.request_id: item
            for item in workspace.capability_observations
            if item.status in {"complete", "partial"}
            and self.clock() - item.observed_at <= timedelta(minutes=15)
        }
        for request_id, previous in requests_by_id.items():
            observation = observations.get(request_id)
            if observation is None or capability_request_fingerprint(previous) != target:
                continue
            return EvidenceUpdate(
                observation=observation.model_copy(
                    update={
                        "observation_id": server_id(request.request_id, "cache-reuse"),
                        "scope": request.scope,
                        "request_id": request.request_id,
                        "reason_summary": (
                            "复用本 generation freshness 窗口内的规范化 Observation；"
                            "未再次调用 Provider。"
                        ),
                    }
                )
            )
        return None

    @staticmethod
    def _entry(reference: CandidateRef, workspace: PlannerWorkspaceState) -> CandidatePoolEntry:
        entry = workspace.candidate_pool.candidate_by_id().get(reference.candidate_id)
        if (
            entry is None
            or entry.candidate_ref != reference
            or entry.commitment_level is CommitmentLevel.FORBIDDEN
        ):
            raise PlannerGuardError("planner_candidate_reference_rejected")
        return entry

    async def _execute(
        self,
        request: PlannerCapabilityRequest,
        workspace: PlannerWorkspaceState,
        book: TaskBookV4,
        cancellation: ModelCancellation,
    ) -> EvidenceUpdate:
        cancellation.raise_if_cancelled("planner_capability")
        arguments = request.arguments
        places: tuple[PlannerPlaceEvidence, ...] = ()
        hours: tuple[PlannerHoursEvidence, ...] = ()
        weather: tuple[PlannerWeatherEvidence, ...] = ()
        tickets: tuple[PlannerTicketEvidence, ...] = ()
        status: Literal["complete", "partial", "unavailable", "invalid_request"] = "complete"
        reason = "已取得实时 Provider 证据。"
        try:
            if isinstance(arguments, CandidateRecallArguments):
                places = await self._recall(arguments, workspace, book, cancellation)
            elif isinstance(arguments, OpeningHoursArguments):
                hours = await self._hours(arguments, workspace, book, cancellation)
            elif isinstance(arguments, PlaceFactsArguments):
                places = await self._place_facts(arguments, workspace, book)
            elif isinstance(arguments, WeatherForecastArguments):
                weather = await self._weather(arguments, book)
            elif isinstance(arguments, TicketAvailabilityArguments):
                tickets = await self._tickets(arguments, workspace, book)
                status = "partial"
                reason = "已查询地点商品与参考票价；仅用于费用估算，本次不核验用户预约或预约余量。"
            else:
                # Spatial/hotel adapters are implemented separately, sharing this
                # request and observation boundary rather than arbitrary tool names.
                return await self._execute_location_capability(
                    request, workspace, book, cancellation
                )
        except ProviderError as error:
            status = "unavailable"
            reason = (
                f"本次 {error.provider.value} 能力返回 {error.code.value}；保留缺失，不补造事实。"
            )
        except CityProviderUnavailableError:
            status = "unavailable"
            reason = (
                "目的地已进入规划，但本次运行没有对应 Provider 城市绑定；"
                "保留缺失并记录本次尝试，不将配置问题交给模型反复重试。"
            )
        cancellation.raise_if_cancelled("planner_observe_capability")
        values: tuple[
            PlannerPlaceEvidence
            | PlannerHoursEvidence
            | PlannerWeatherEvidence
            | PlannerTicketEvidence,
            ...,
        ] = (*places, *hours, *weather, *tickets)
        fact_ids = tuple(item.fact_reference_id for item in values)
        if not values and status == "complete":
            status = "unavailable"
            reason = "本次真实查询没有取得可绑定当前实体的有效证据。"
        if hours and (
            len(hours) != len(getattr(arguments, "candidate_refs", ()))
            or any(day.status in {"unknown", "conflict"} for item in hours for day in item.days)
        ):
            status = "partial"
            reason = "部分地点的营业信息缺失，不能默认为全天开放。"
        if (
            weather
            and isinstance(arguments, WeatherForecastArguments)
            and ({item.service_date for item in weather} != set(arguments.service_dates))
        ):
            status = "partial"
            reason = "天气来源只返回了部分旅行日期；缺失日期保持未知。"
        if (
            places
            and isinstance(arguments, PlaceFactsArguments)
            and (
                len(places) != len(arguments.candidate_refs)
                or not set(arguments.fact_kinds)
                <= {"identity", "coordinates", "category", "address", "name"}
            )
        ):
            status = "partial"
            reason = "已核验实体、类别、地址与坐标；其他请求事实不在本次来源能力内，仍未知。"
        observation = PlannerCapabilityObservation(
            observation_id=str(uuid4()),
            scope=request.scope,
            request_id=request.request_id,
            capability=request.capability,
            status=status,
            candidate_refs=getattr(arguments, "candidate_refs", ()),
            service_dates=request.service_dates,
            fact_reference_ids=fact_ids,
            reason_summary=reason,
            observed_at=self.clock(),
        )
        return EvidenceUpdate(
            observation=observation, places=places, hours=hours, weather=weather, tickets=tickets
        )

    async def supplement_schedule_choices(
        self,
        workspace: PlannerWorkspaceState,
        book: TaskBookV4,
        cancellation: ModelCancellation,
        *,
        attraction_dates: set[date],
        dining_dates: set[date],
    ) -> PlannerWorkspaceState:
        """Distinct, coordinate-bound recall batches within the shared turn budget.

        This supplies choices, never assigns visits. No invented POIs, child POIs
        inside an already selected attraction, or excluded places may fill a gap.
        """
        if workspace.working_itinerary is None:
            return workspace
        # A global two-batch cap starved the fourth day. The repair ledger now
        # owns attempts/fairness; identical anchor queries remain deduplicated below.
        places = {item.canonical_entity_id: item for item in workspace.place_evidence}
        excluded = {
            key
            for key, (intent, _) in entity_intents(book).items()
            if intent.disposition.value == "avoid"
        }
        selected_provider_ids = {
            places[item.object_ref.canonical_entity_id].provider_entity_id
            for day in workspace.working_itinerary.days
            for item in day.ordered_items
            if isinstance(item.object_ref, CandidateRef)
            and item.object_ref.canonical_entity_id in places
        }
        searches: list[tuple[CandidateEntityKind, PlannerPlaceEvidence]] = []
        for kind, dates in (
            (CandidateEntityKind.ATTRACTION, attraction_dates),
            (CandidateEntityKind.RESTAURANT, dining_dates),
        ):
            if kind is CandidateEntityKind.ATTRACTION:
                searches.extend(
                    (kind, anchor)
                    for anchor in _schedule_attraction_anchors(workspace, book, dates)
                )
                continue
            anchors = dict.fromkeys(
                item.object_ref.canonical_entity_id
                for day in workspace.working_itinerary.days
                if day.service_date in dates
                for item in day.ordered_items
                if isinstance(item.object_ref, CandidateRef)
                and item.item_kind == "visit"
                and item.object_ref.canonical_entity_id in places
            )
            if not anchors and dates:
                anchors = dict.fromkeys(
                    item.canonical_entity_id
                    for item in places.values()
                    if item.entity_kind is CandidateEntityKind.ATTRACTION
                )
            searches.extend((kind, places[key]) for key in list(anchors)[:3])
        if not searches:
            return workspace
        request_id = server_id(
            workspace.generation_id,
            "schedule-nearby-choices",
            tuple(sorted((kind.value, anchor.canonical_entity_id) for kind, anchor in searches)),
        )
        if any(item.request_id == request_id for item in workspace.capability_observations):
            return workspace
        scope = self.registry.provider_scope(
            book.destination_and_dates.destination_name, ProviderCode.AMAP
        )
        semaphore = asyncio.Semaphore(3)

        async def search(
            kind: CandidateEntityKind, anchor: PlannerPlaceEvidence
        ) -> tuple[PlannerPlaceEvidence, ...]:
            provider_unavailable = False

            async def query(keyword: str | None = None) -> tuple[PlannerPlaceEvidence, ...]:
                nonlocal provider_unavailable
                cancellation.raise_if_cancelled("planner_schedule_recall")
                try:
                    async with semaphore:
                        response = await self.providers.places.search_nearby(
                            NearbyPlaceSearchRequest(
                                city=scope,
                                center=anchor.coordinates,
                                radius_m=8000 if kind is CandidateEntityKind.ATTRACTION else 1800,
                                query=keyword,
                                category_hint=_category(kind),
                                typecodes=[]
                                if keyword
                                else ["110000", "140100"]
                                if kind is CandidateEntityKind.ATTRACTION
                                else ["050100", "050200", "050300"],
                                page_size=20,
                            )
                        )
                except ProviderError:
                    provider_unavailable = True
                    return ()
                return tuple(
                    _place_evidence(item, kind)
                    for item in response.items
                    if _real_category(item, kind)
                    and item.city_id == scope.city_id
                    and str(uuid5(NAMESPACE_URL, f"amap:{item.source_place_id}"))
                    not in (excluded | places.keys())
                    and not (
                        kind is CandidateEntityKind.RESTAURANT
                        and re.search(
                            r"coffee|咖啡|奶茶|果饮|糖水|糕点|茶室|茶座|文创|冰淇淋",
                            item.name,
                            re.I,
                        )
                    )
                    and not (
                        kind is CandidateEntityKind.ATTRACTION
                        and (
                            item.provider_parent_place_id in selected_provider_ids
                            or is_explicit_internal_subsite(
                                item.name, item.provider_parent_place_id
                            )
                            or any(
                                is_named_subsite(item.name, site.display_name)
                                for site in places.values()
                                if site.entity_kind is CandidateEntityKind.ATTRACTION
                            )
                            or re.search(r"售票处|检票口|出入口|停车场|游客中心", item.name)
                        )
                    )
                )[:6]

            initial = await query()
            if provider_unavailable or kind is not CandidateEntityKind.ATTRACTION:
                return initial
            # A nearby page can contain many legitimate but tiny photo stops.
            # Raw hit count is not evidence of enough content to fill a half-day.
            # Diversify once, within the existing four-query/eight-result cap.
            focused = await asyncio.gather(
                *(query(word) for word in ("历史街区", "博物馆", "公园"))
            )
            diversified: dict[str, PlannerPlaceEvidence] = {}
            for index in range(6):
                for group in focused:
                    if index < len(group):
                        item = group[index]
                        diversified.setdefault(item.canonical_entity_id, item)
            for item in initial:
                diversified.setdefault(item.canonical_entity_id, item)
            return tuple(diversified.values())[:8]

        groups = await asyncio.gather(*(search(kind, anchor) for kind, anchor in searches))
        additions = {
            item.canonical_entity_id: item
            for group in groups
            for item in group
            if item.canonical_entity_id not in places
        }
        attraction_count = sum(
            x.entity_kind is CandidateEntityKind.ATTRACTION for x in additions.values()
        )
        dining_count = sum(
            x.entity_kind is CandidateEntityKind.RESTAURANT for x in additions.values()
        )
        update = EvidenceUpdate(
            places=tuple(additions.values()),
            observation=PlannerCapabilityObservation(
                observation_id=server_id(request_id, "result"),
                scope=workspace.current_scope,
                request_id=request_id,
                capability=PlannerCapability.CANDIDATE_RECALL,
                status="complete"
                if all(
                    any(item.entity_kind is kind for item in additions.values())
                    for kind in {kind for kind, _ in searches}
                )
                else "partial",
                fact_reference_ids=tuple(item.fact_reference_id for item in additions.values()),
                reason_summary=(
                    "排程补查：优先按实际空档附近的真实坐标补查景点和餐厅；"
                    f"新增独立景点{attraction_count}个，餐厅{dining_count}个；"
                    "缺失候选不虚构，本批次结束后不重复查询。"
                ),
                observed_at=self.clock(),
            ),
        )
        # Complete geometry before restoring the old semantic choices. Restoring
        # a draft immediately after a pool refresh would leave stale cluster refs.
        staged = prepare_workspace_for_evidence_refresh(workspace)
        expanded = self._merge_updates(staged, book, (update,))
        if expanded.spatial_observation is None:
            expanded = await self._spatial(expanded, book, cancellation)
            # Same-generation exact endpoints have not moved. Keep their real
            # observations when adding the new sparse candidate graph.
            expanded = advance(
                expanded,
                route_evidence=tuple(
                    {
                        edge.route_edge_id: edge
                        for edge in (*workspace.route_evidence, *expanded.route_evidence)
                    }.values()
                ),
            )
        return rebind_semantic_artifacts_after_evidence(
            workspace, expanded, refresh_clusters=bool(additions)
        )

    async def _recall(
        self,
        args: CandidateRecallArguments,
        workspace: PlannerWorkspaceState,
        book: TaskBookV4,
        cancellation: ModelCancellation,
    ) -> tuple[PlannerPlaceEvidence, ...]:
        domain = (
            CandidateDomain.ATTRACTION
            if args.domain is CandidateEntityKind.ATTRACTION
            else CandidateDomain.RESTAURANT
        )
        refs = task_book_references(book)
        preferences = (
            book.attraction_direction.preferences
            if domain is CandidateDomain.ATTRACTION
            else book.dining_direction.preferences
        )
        clues = tuple(
            dict.fromkeys(
                [
                    *(
                        item.display_name
                        for key, (item, kind) in entity_intents(book).items()
                        if key in workspace.candidate_pool.missing_required_candidate_refs
                        and kind is args.domain
                    ),
                    *(item.value[:240] for item in preferences),
                    *(refs[key][:240] for key in args.task_book_preference_refs),
                ]
            )
        )
        place_by_id = {place.canonical_entity_id: place for place in workspace.place_evidence}
        anchor_refs = list(args.nearby_candidate_refs)
        if workspace.spatial_observation:
            for cluster in workspace.spatial_observation.clusters:
                if cluster.cluster_id in args.nearby_cluster_refs:
                    anchor_refs.extend(cluster.candidate_refs[:1])
        anchors = {
            reference.canonical_entity_id: RecallAnchor(
                anchor_id=UUID(reference.canonical_entity_id),
                name=place_by_id[reference.canonical_entity_id].display_name,
                domain=CandidateDomain.ATTRACTION
                if reference.entity_kind is CandidateEntityKind.ATTRACTION
                else CandidateDomain.RESTAURANT,
                coordinates=place_by_id[reference.canonical_entity_id].coordinates,
            )
            for reference in anchor_refs
        }
        request = CandidateRecallRequest(
            request_id=uuid4(),
            trip_id=UUID(workspace.trip_id),
            semantic_state_version=book.based_on_state_version,
            task_book_id=UUID(book.task_book_id),
            task_book_revision=book.version,
            city_id=self.registry.resolve(book.destination_and_dates.destination_name).city_id,
            start_date=book.destination_and_dates.start_date,
            end_date=book.destination_and_dates.end_date,
            free_text_clues=clues,
            anchors=tuple(anchors.values()),
            special_constraints=tuple(item.value[:240] for item in book.hard_constraints),
            budget=CandidateRecallBudget(
                max_total_candidates=min(20, args.limit * 2),
                max_total_provider_calls=3,
                domains=(
                    DomainRecallBudget(
                        domain=domain, max_candidates=min(20, args.limit * 2), max_provider_calls=3
                    ),
                ),
            ),
        )
        recalled = await self.recall.recall(request, cancellation=cancellation)
        ranked = self.ranking.rank(
            CandidateRankingRequest(
                ranking_request_id=uuid4(),
                recall_request=request,
                recall_result=recalled,
                recommendation_limit=min(14, args.limit),
            )
        )
        excluded = {
            key
            for key, (item, _) in entity_intents(book).items()
            if item.disposition.value == "avoid"
        }
        results = []
        # Ranking only limits/annotates the evidence pool. Planner alone chooses days.
        for item in ranked.candidates:
            candidate = item.candidate
            place = candidate.place
            source = next(
                (
                    source
                    for source in candidate.sources
                    if source.kind is RecallSourceKind.PROVIDER
                    and source.provider is ProviderCode.AMAP
                ),
                None,
            )
            if (
                source is None
                or source.source_place_id is None
                or source.fetched_at is None
                or place.coordinates is None
                or not place.provider_typecode
            ):
                continue
            canonical = str(uuid5(NAMESPACE_URL, f"amap:{source.source_place_id}"))
            if (
                canonical != str(place.place_id)
                or canonical in excluded
                or category_from_original_typecodes(place.provider_typecode)
                is not _category(args.domain)
            ):
                continue
            results.append(
                PlannerPlaceEvidence(
                    canonical_entity_id=canonical,
                    entity_kind=args.domain,
                    display_name=place.name,
                    city_id=place.city_id,
                    coordinates=place.coordinates,
                    provider="amap",
                    provider_entity_id=source.source_place_id,
                    provider_parent_place_id=place.provider_parent_place_id,
                    provider_typecode=place.provider_typecode,
                    address=place.address,
                    rating=place.rating,
                    average_cost=place.average_cost,
                    fact_reference_id=server_id(
                        "place", source.source_place_id, source.fetched_at.isoformat()
                    ),
                    observed_at=source.fetched_at,
                )
            )
            if len(results) >= args.limit:
                break
        return tuple(results)

    async def _hours(
        self,
        args: OpeningHoursArguments,
        workspace: PlannerWorkspaceState,
        book: TaskBookV4,
        cancellation: ModelCancellation,
    ) -> tuple[PlannerHoursEvidence, ...]:
        places = {place.canonical_entity_id: place for place in workspace.place_evidence}
        city = self.registry.provider_scope(
            book.destination_and_dates.destination_name, ProviderCode.AMAP
        )
        results = []
        for reference in args.candidate_refs:
            cancellation.raise_if_cancelled("planner_opening_hours")
            place = places[reference.canonical_entity_id]
            cached = next(
                (
                    hours
                    for hours in workspace.hours_evidence
                    if workspace.react_state is not None
                    and hours.canonical_entity_id == reference.canonical_entity_id
                    and hours.provider_entity_id == place.provider_entity_id
                    and hours.observed_at <= self.clock() < hours.expires_at
                    and set(args.service_dates) <= {day.service_date for day in hours.days}
                ),
                None,
            )
            if cached is not None:
                results.append(cached)
                continue
            try:
                response = await self.providers.hours.get_regular_hours(
                    HoursRequest(
                        place_id=UUID(place.canonical_entity_id),
                        city=city,
                        name=place.display_name,
                        address=place.address,
                        coordinates=place.coordinates,
                        source_place_ids={ProviderCode.AMAP: place.provider_entity_id},
                        service_dates=list(args.service_dates),
                    )
                )
            except ProviderError:
                continue
            for hours in response.items:
                if (
                    hours.provider is not ProviderCode.AMAP
                    or hours.source_place_id != place.provider_entity_id
                ):
                    continue
                days = tuple(evaluate_regular_hours(hours, args.service_dates))
                results.append(
                    PlannerHoursEvidence(
                        canonical_entity_id=place.canonical_entity_id,
                        provider_entity_id=place.provider_entity_id,
                        fact_reference_id=server_id(
                            "hours",
                            place.provider_entity_id,
                            hours.fetched_at.isoformat(),
                            *args.service_dates,
                        ),
                        observed_at=hours.fetched_at,
                        expires_at=hours.fetched_at + timedelta(hours=6),
                        days=days,
                    )
                )
                break
        return tuple(results)

    async def _place_facts(
        self, args: PlaceFactsArguments, workspace: PlannerWorkspaceState, book: TaskBookV4
    ) -> tuple[PlannerPlaceEvidence, ...]:
        city = self.registry.provider_scope(
            book.destination_and_dates.destination_name, ProviderCode.AMAP
        )
        places = {place.canonical_entity_id: place for place in workspace.place_evidence}
        result = []
        for reference in args.candidate_refs:
            expected = places[reference.canonical_entity_id]
            response = await self.providers.places.get_place(
                PlaceDetailRequest(city=city, source_place_id=expected.provider_entity_id)
            )
            for place in response.items:
                if (
                    place.source_place_id == expected.provider_entity_id
                    and place.city_id == city.city_id
                    and _real_category(place, reference.entity_kind)
                ):
                    result.append(_place_evidence(place, reference.entity_kind))
                    break
        return tuple(result)

    async def _weather(
        self, args: WeatherForecastArguments, book: TaskBookV4
    ) -> tuple[PlannerWeatherEvidence, ...]:
        if self.providers.weather is None:
            return ()
        city = self.registry.provider_scope(
            book.destination_and_dates.destination_name, ProviderCode.WEATHER
        )
        response = await self.providers.weather.get_forecast(
            WeatherRequest(
                city=city, start_date=min(args.service_dates), end_date=max(args.service_dates)
            )
        )
        return tuple(
            PlannerWeatherEvidence(
                service_date=item.forecast_date,
                condition_day=item.condition_day,
                condition_night=item.condition_night,
                high_celsius=item.high_celsius,
                low_celsius=item.low_celsius,
                source_name=item.source_name,
                forecast_kind=item.forecast_kind,
                observed_at=item.fetched_at,
                fact_reference_id=server_id(
                    "weather", city.city_id, item.forecast_date, item.fetched_at.isoformat()
                ),
            )
            for item in response.items
            if item.forecast_date in args.service_dates
        )

    def fresh_ticket_facts(
        self, workspace: PlannerWorkspaceState
    ) -> dict[tuple[str, date], PlannerTicketEvidence]:
        return {
            (ticket.canonical_entity_id, ticket.service_date): ticket
            for ticket in workspace.ticket_evidence
            if timedelta(0) <= self.clock() - ticket.observed_at <= timedelta(minutes=15)
        }

    async def _tickets(
        self, args: TicketAvailabilityArguments, workspace: PlannerWorkspaceState, book: TaskBookV4
    ) -> tuple[PlannerTicketEvidence, ...]:
        city = self.registry.provider_scope(
            book.destination_and_dates.destination_name, ProviderCode.FLYAI
        )
        entries = workspace.candidate_pool.candidate_by_id()
        cached = self.fresh_ticket_facts(workspace) if workspace.react_state is not None else {}
        results = []
        for reference in args.candidate_refs:
            entry = entries[reference.candidate_id]
            missing = []
            for day in args.service_dates:
                if existing := cached.get((reference.canonical_entity_id, day)):
                    results.append(existing)
                    continue
                missing.append(day)
            if missing:
                results.extend(
                    await lookup_ticket_prices(
                        self.providers.products,
                        city=city,
                        canonical_id=reference.canonical_entity_id,
                        name=entry.display_name,
                        days=tuple(missing),
                    )
                )
        return tuple(results)

    async def _spatial(
        self, workspace: PlannerWorkspaceState, book: TaskBookV4, cancellation: ModelCancellation
    ) -> PlannerWorkspaceState:
        return await build_spatial_observation(
            workspace,
            book,
            routes=self.providers.routes,
            city=self.registry.provider_scope(
                book.destination_and_dates.destination_name, ProviderCode.AMAP
            ),
            cancellation=cancellation,
            now=self.clock(),
        )

    def _merge_updates(
        self,
        workspace: PlannerWorkspaceState,
        book: TaskBookV4,
        updates: tuple[EvidenceUpdate, ...],
    ) -> PlannerWorkspaceState:
        return observe_capability_results(workspace, book, updates, self.clock())

    async def _execute_location_capability(
        self,
        request: PlannerCapabilityRequest,
        workspace: PlannerWorkspaceState,
        book: TaskBookV4,
        cancellation: ModelCancellation,
    ) -> EvidenceUpdate:
        from backend.agent.planner.location_capabilities import execute_location_capability

        return await execute_location_capability(
            request,
            workspace,
            book,
            providers=self.providers,
            registry=self.registry,
            cancellation=cancellation,
            now=self.clock(),
        )


def selected_itinerary_route_requests(
    workspace: PlannerWorkspaceState,
    book: TaskBookV4,
    *,
    now: datetime | None = None,
) -> tuple[PlannerCapabilityRequest, ...]:
    """Build deterministic requests for currently selected adjacent exact endpoints."""

    now = now or datetime.now(UTC)
    draft = workspace.working_itinerary
    if draft is None:
        return ()

    boundary: SpatialRouteEndpoint | None = None
    boundary_property_id: str | None = None
    if draft.lodging_baseline.selected_offer_ref is not None:
        offer = draft.lodging_baseline.selected_offer_ref
        boundary = SpatialRouteEndpoint(kind="hotel_offer", reference_id=offer.offer_id)
        boundary_property_id = offer.property_id
    elif draft.lodging_baseline.fixed_commitment_ref is not None:
        fixed = draft.lodging_baseline.fixed_commitment_ref
        boundary = SpatialRouteEndpoint(
            kind="fixed_commitment",
            reference_id=fixed.commitment_id,
        )
        boundary_property_id = fixed.commitment_id

    located_hotels = {item.property_id for item in workspace.hotel_location_evidence}
    if boundary is not None and boundary_property_id not in located_hotels:
        boundary = None

    located_candidates = {item.canonical_entity_id for item in workspace.place_evidence}
    candidate_entries = workspace.candidate_pool.candidate_by_id()

    def item_endpoint(item: DraftItem) -> SpatialRouteEndpoint | None:
        reference = item.object_ref
        if isinstance(reference, CandidateRef):
            entry = candidate_entries.get(reference.candidate_id)
            if (
                entry is None
                or entry.candidate_ref != reference
                or reference.canonical_entity_id not in located_candidates
            ):
                return None
            return SpatialRouteEndpoint(
                kind="candidate",
                reference_id=reference.candidate_id,
            )
        if isinstance(reference, FixedCommitmentRef):
            fixed_hotel = book.lodging_direction.existing_booking
            if (
                fixed_hotel is not None
                and reference.commitment_id == fixed_hotel.booking_id
                and reference.commitment_id in located_hotels
            ):
                return SpatialRouteEndpoint(
                    kind="fixed_commitment",
                    reference_id=reference.commitment_id,
                )
        return None

    ordered_pairs: list[SpatialRoutePair] = []
    seen_pairs: set[tuple[str, str, str, str]] = set()
    pair_modes: dict[tuple[str, str, str, str], list[RoutePreference]] = {}
    modes: list[RoutePreference] = (
        [] if workspace.react_state is not None else ["taxi", "public_transit", "walking"]
    )
    observed_edges = {
        (
            edge.origin.kind,
            edge.origin.reference_id,
            edge.destination.kind,
            edge.destination.reference_id,
            edge.transport_mode,
        ): edge
        for edge in (
            *(workspace.spatial_observation.route_edges if workspace.spatial_observation else ()),
            *workspace.route_evidence,
        )
    }
    if workspace.react_state is not None:
        observed_edges = {
            key: edge for key, edge in observed_edges.items() if not route_needs_retry(edge, now)
        }
    for day in draft.days:
        walked_m = 0
        for mode in day.transport_preferences:
            if mode not in modes:
                modes.append(mode)
        endpoints = [item_endpoint(item) for item in day.ordered_items]
        if boundary is not None:
            endpoints = [boundary, *endpoints, boundary]
        for origin, destination in zip(endpoints, endpoints[1:], strict=False):
            if origin is None or destination is None or origin == destination:
                continue
            identity = (
                origin.kind,
                origin.reference_id,
                destination.kind,
                destination.reference_id,
            )
            selected_modes = next(
                (
                    (selection.transport_mode,)
                    for selection in day.route_mode_selections
                    if selection.origin == origin and selection.destination == destination
                ),
                day.transport_preferences,
            )
            if old_mode := _refresh_route_mode(origin, destination, workspace, day):
                selected_modes = (old_mode,)
            if workspace.react_state is not None:
                needed: tuple[RoutePreference, ...] = ()
                for index, mode in enumerate(selected_modes):
                    if (*identity, mode) not in observed_edges:
                        needed = (mode,)
                        break
                    selected = _select_route(
                        origin,
                        destination,
                        selected_modes[: index + 1],
                        workspace,
                        day=day,
                        book=book,
                        walked_m=walked_m,
                    )
                    if selected is not None:
                        if selected.mode.value == "walking":
                            walked_m += selected.edge.distance_meters or 0
                        break
                selected_modes = needed
            for mode in selected_modes:
                if mode not in pair_modes.setdefault(identity, []):
                    pair_modes[identity].append(mode)
            if identity in seen_pairs:
                continue
            seen_pairs.add(identity)
            ordered_pairs.append(SpatialRoutePair(origin=origin, destination=destination))

    if not ordered_pairs or not modes:
        return ()

    observed = {
        (
            edge.origin.kind,
            edge.origin.reference_id,
            edge.destination.kind,
            edge.destination.reference_id,
            edge.transport_mode,
        )
        for edge in (
            *workspace.route_evidence,
            *(workspace.spatial_observation.route_edges if workspace.spatial_observation else ()),
        )
    }
    if workspace.react_state is not None:
        observed = set(observed_edges)
    missing_pairs = tuple(
        pair
        for pair in ordered_pairs
        if any(
            (
                pair.origin.kind,
                pair.origin.reference_id,
                pair.destination.kind,
                pair.destination.reference_id,
                mode,
            )
            not in observed
            for mode in modes
        )
    )
    if not missing_pairs and workspace.react_state is None:
        return ()

    grouped: dict[tuple[RoutePreference, ...], list[SpatialRoutePair]] = {}
    if workspace.react_state is not None:
        # Different days can choose different modes. An observed mode for an
        # exact pair is reusable even when another mode still needs a lookup.
        for pair in ordered_pairs:
            identity = (
                pair.origin.kind,
                pair.origin.reference_id,
                pair.destination.kind,
                pair.destination.reference_id,
            )
            missing_modes = tuple(
                mode for mode in pair_modes.get(identity, ()) if (*identity, mode) not in observed
            )
            if missing_modes:
                grouped.setdefault(missing_modes, []).append(pair)
    else:
        grouped[tuple(modes)] = list(missing_pairs)

    requests = []
    batches = [
        (requested_modes, tuple(pairs[offset : offset + MAX_SELECTED_ROUTE_PAIRS_PER_REQUEST]))
        for requested_modes, pairs in grouped.items()
        for offset in range(0, len(pairs), MAX_SELECTED_ROUTE_PAIRS_PER_REQUEST)
    ]
    for request_index, (requested_modes, pairs) in enumerate(batches):
        if workspace.react_state is None and request_index >= MAX_SELECTED_ROUTE_REQUESTS:
            raise PlannerGuardError("planner_selected_route_request_budget_exceeded")
        departure_date = draft.days[0].service_date
        requests.append(
            PlannerCapabilityRequest(
                request_id=server_id(
                    workspace.generation_id,
                    "selected-itinerary-routes",
                    draft.draft_id,
                    draft.draft_revision,
                    request_index,
                    *(
                        f"{pair.origin.kind}:{pair.origin.reference_id}>"
                        f"{pair.destination.kind}:{pair.destination.reference_id}"
                        for pair in pairs
                    ),
                    *requested_modes,
                ),
                scope=workspace.current_scope,
                capability=PlannerCapability.SPATIAL_ROUTES,
                purpose="complete_initial_evidence",
                service_dates=(departure_date,),
                blocking=False,
                arguments=SpatialRoutesArguments(
                    endpoint_pairs=pairs,
                    transport_modes=requested_modes,
                    departure_service_date=departure_date,
                ),
            )
        )
    return tuple(requests)


def _schedule_attraction_anchors(
    workspace: PlannerWorkspaceState, book: TaskBookV4, dates: set[date]
) -> tuple[PlannerPlaceEvidence, ...]:
    """Search where a real gap exists, not only around the first day's stops.

    One anchor per affected date, at most three. This only chooses query centers;
    the Agent still selects venues and real routes determine whether they fit.
    """
    draft, schedule = workspace.working_itinerary, workspace.materialized_schedule
    if draft is None or not dates:
        return ()
    places = {item.canonical_entity_id: item for item in workspace.place_evidence}
    selected: dict[str, PlannerPlaceEvidence] = {}
    covered_dates: set[date] = set()
    if schedule is not None:
        days = {str(day.service_date): day for day in schedule.days}
        for gap in sorted(
            [
                *schedule_quality_gaps(workspace, book),
                *afternoon_activity_opportunities(workspace, book),
                *evening_activity_opportunities(workspace, book),
            ],
            key=lambda x: (x["start"] >= "18:00", -x["minutes"]),
        ):
            day = days[gap["date"]]
            if day.service_date not in dates or day.service_date in covered_dates:
                continue
            start = int(gap["start"][:2]) * 60 + int(gap["start"][3:])
            end = int(gap["end"][:2]) * 60 + int(gap["end"][3:])
            adjacent = [
                *reversed([item for item in day.activities if minutes(item.end_time) <= start]),
                *(item for item in day.activities if minutes(item.start_time) >= end),
            ]
            anchor = next(
                (places[str(item.place_id)] for item in adjacent if str(item.place_id) in places),
                None,
            )
            if anchor is not None:
                selected.setdefault(anchor.canonical_entity_id, anchor)
                covered_dates.add(day.service_date)
        for issue in schedule_coverage_issues(workspace, book):
            day = days[str(issue["date"])]
            if day.service_date not in dates or day.service_date in covered_dates:
                continue
            ordered = (
                day.activities if issue["period"] == "morning" else tuple(reversed(day.activities))
            )
            anchor = next(
                (places[str(item.place_id)] for item in ordered if str(item.place_id) in places),
                None,
            )
            if anchor is not None:
                selected.setdefault(anchor.canonical_entity_id, anchor)
                covered_dates.add(day.service_date)
    if selected:
        return tuple(selected.values())[:3]
    # A long restaurant commute can expose a gap after reordering. In that case
    # retain a small fallback covering separate dates, instead of three day-one POIs.
    for semantic_day in draft.days:
        if semantic_day.service_date not in dates:
            continue
        anchor = next(
            (
                places[item.object_ref.canonical_entity_id]
                for item in semantic_day.ordered_items
                if item.item_kind == "visit"
                and isinstance(item.object_ref, CandidateRef)
                and item.object_ref.canonical_entity_id in places
            ),
            None,
        )
        if anchor is not None:
            selected.setdefault(anchor.canonical_entity_id, anchor)
    if not selected:
        selected = {
            key: value
            for key, value in places.items()
            if value.entity_kind is CandidateEntityKind.ATTRACTION
        }
    return tuple(selected.values())[:3]


def _category(kind: CandidateEntityKind) -> PlaceCategory:
    return (
        PlaceCategory.ATTRACTION
        if kind is CandidateEntityKind.ATTRACTION
        else PlaceCategory.RESTAURANT
    )


def _real_category(place: ProviderPlace, kind: CandidateEntityKind) -> bool:
    return (
        place.provider is ProviderCode.AMAP
        and place.category is _category(kind)
        and category_from_original_typecodes(place.provider_typecode or "") is _category(kind)
    )


def _place_evidence(place: ProviderPlace, kind: CandidateEntityKind) -> PlannerPlaceEvidence:
    return PlannerPlaceEvidence(
        canonical_entity_id=str(uuid5(NAMESPACE_URL, f"amap:{place.source_place_id}")),
        entity_kind=kind,
        display_name=place.name,
        city_id=place.city_id,
        coordinates=place.coordinates,
        provider="amap",
        provider_entity_id=place.source_place_id,
        provider_parent_place_id=place.provider_parent_place_id,
        provider_typecode=place.provider_typecode or "unknown",
        address=place.address,
        rating=place.rating,
        average_cost=place.average_cost,
        cuisine=provider_place_cuisine(place) if kind is CandidateEntityKind.RESTAURANT else None,
        fact_reference_id=server_id("place", place.source_place_id, place.fetched_at.isoformat()),
        observed_at=place.fetched_at,
    )
