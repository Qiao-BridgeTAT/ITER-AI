"""Collect private candidate facts while the user continues Prepare."""

from __future__ import annotations

import asyncio
import logging
from datetime import timedelta
from uuid import NAMESPACE_URL, UUID, uuid5

from backend.agent.model_gateway import ModelCancellation
from backend.agent.planner.ticket_prices import lookup_ticket_prices
from backend.agent.planner.workspace import server_id
from backend.contracts.candidate_recall import RecalledCandidate
from backend.contracts.enums import ProviderCode
from backend.contracts.prepared_evidence import PreparedCandidateEvidence
from backend.contracts.v4.cards import SpecificCandidateCard
from backend.contracts.v4.enums import CandidateEntityKind
from backend.contracts.v4.planner_evidence import (
    PlannerHoursEvidence,
    PlannerPlaceEvidence,
    PlannerTicketEvidence,
)
from backend.contracts.v4.state import TripSemanticState
from backend.persistence.prepared_evidence_repository import PreparedEvidenceRepository
from backend.planning.city_registry import CityRegistry
from backend.providers.contracts import HoursRequest, ProviderError
from backend.providers.hours_rules import evaluate_regular_hours
from backend.providers.interfaces import HoursProvider, TravelProductProvider

_LOGGER = logging.getLogger(__name__)


class PreparedEvidenceCollector:
    def __init__(
        self,
        repository: PreparedEvidenceRepository,
        registry: CityRegistry,
        hours: HoursProvider | None,
        products: TravelProductProvider | None,
    ) -> None:
        self.repository, self.registry = repository, registry
        self.hours, self.products = hours, products
        self._tasks: dict[tuple[UUID, str], asyncio.Task[None]] = {}
        self._semaphore = asyncio.Semaphore(3)

    async def remember(
        self,
        state: TripSemanticState,
        turn_id: UUID,
        card: SpecificCandidateCard,
        candidates: tuple[RecalledCandidate, ...],
        cancellation: ModelCancellation | None,
    ) -> None:
        basics = state.trip_basics
        if not basics.start_date or not basics.end_date:
            return
        dates = tuple(
            basics.start_date + timedelta(days=i)
            for i in range((basics.end_date - basics.start_date).days + 1)
        )
        chosen = {
            option.entity_ref.canonical_entity_id for option in card.options if option.entity_ref
        }
        bundles = []
        for candidate in candidates:
            p = candidate.place
            source = next((s for s in candidate.sources if s.provider is ProviderCode.AMAP), None)
            if (
                source is None
                or source.source_place_id is None
                or source.fetched_at is None
                or p.coordinates is None
                or p.provider_typecode is None
                or str(p.place_id) not in chosen
                or p.place_id != uuid5(NAMESPACE_URL, f"amap:{source.source_place_id}")
            ):
                continue
            place = PlannerPlaceEvidence(
                canonical_entity_id=str(p.place_id),
                entity_kind=CandidateEntityKind.ATTRACTION
                if candidate.domain.value == "attraction"
                else CandidateEntityKind.RESTAURANT,
                display_name=p.name,
                city_id=p.city_id,
                coordinates=p.coordinates,
                provider="amap",
                provider_entity_id=source.source_place_id,
                provider_typecode=p.provider_typecode,
                provider_parent_place_id=p.provider_parent_place_id,
                address=p.address,
                rating=p.rating,
                average_cost=p.average_cost,
                cuisine=p.cuisine,
                fact_reference_id=server_id(
                    "place", source.source_place_id, source.fetched_at.isoformat()
                ),
                observed_at=source.fetched_at,
            )
            bundle = PreparedCandidateEvidence(
                trip_id=UUID(state.trip_id),
                source_attachment_id=card.attachment_id,
                dependency_fingerprint=card.dependency_fingerprint,
                service_dates=dates,
                place=place,
            )
            await self.repository.save(turn_id, bundle)
            bundles.append(bundle)
        key = (UUID(state.trip_id), card.attachment_id)
        if bundles and key not in self._tasks:
            task = asyncio.create_task(self._collect(turn_id, tuple(bundles), cancellation))
            self._tasks[key] = task
            task.add_done_callback(lambda completed: self._done(key, completed))

    def _done(self, key: tuple[UUID, str], task: asyncio.Task[None]) -> None:
        self._tasks.pop(key, None)
        if not task.cancelled() and (error := task.exception()) is not None:
            _LOGGER.warning("prepare fact collection stopped: %s", type(error).__name__)

    async def remember_supplement(self, turn_id: UUID, bundle: PreparedCandidateEvidence) -> None:
        """An internal source can collect facts without creating a user-visible card."""
        await self.repository.save(turn_id, bundle)
        key = (
            bundle.trip_id,
            f"pool:{bundle.source_attachment_id}:{bundle.place.canonical_entity_id}",
        )
        if key not in self._tasks:
            task = asyncio.create_task(self._collect(turn_id, (bundle,), None))
            self._tasks[key] = task
            task.add_done_callback(lambda completed: self._done(key, completed))

    async def _collect(
        self,
        turn_id: UUID,
        bundles: tuple[PreparedCandidateEvidence, ...],
        cancellation: ModelCancellation | None,
    ) -> None:
        async def one(bundle: PreparedCandidateEvidence) -> None:
            async with self._semaphore:
                if cancellation:
                    cancellation.raise_if_cancelled("prepare_candidate_facts")
                place = bundle.place
                hours = None
                tickets: tuple[PlannerTicketEvidence, ...] = ()
                if self.hours:
                    try:
                        async with asyncio.timeout(12):
                            response = await self.hours.get_regular_hours(
                                HoursRequest(
                                    place_id=UUID(place.canonical_entity_id),
                                    city=self.registry.provider_scope(
                                        place.city_id, ProviderCode.AMAP
                                    ),
                                    name=place.display_name,
                                    address=place.address,
                                    coordinates=place.coordinates,
                                    source_place_ids={ProviderCode.AMAP: place.provider_entity_id},
                                    service_dates=list(bundle.service_dates),
                                )
                            )
                        original = next(
                            (
                                h
                                for h in response.items
                                if h.provider is ProviderCode.AMAP
                                and h.source_place_id == place.provider_entity_id
                            ),
                            None,
                        )
                        if original:
                            hours = PlannerHoursEvidence(
                                canonical_entity_id=place.canonical_entity_id,
                                provider_entity_id=place.provider_entity_id,
                                fact_reference_id=server_id(
                                    "hours",
                                    place.provider_entity_id,
                                    original.fetched_at.isoformat(),
                                    *bundle.service_dates,
                                ),
                                observed_at=original.fetched_at,
                                expires_at=original.fetched_at + timedelta(hours=6),
                                days=tuple(evaluate_regular_hours(original, bundle.service_dates)),
                            )
                    except (ProviderError, TimeoutError):
                        pass
                # Save independently: slow ticket lookup cannot discard completed hours.
                if hours:
                    bundle = bundle.model_copy(update={"hours": hours})
                    await self.repository.save(turn_id, bundle)
                if self.products and place.entity_kind is CandidateEntityKind.ATTRACTION:
                    try:
                        async with asyncio.timeout(16):
                            tickets = await lookup_ticket_prices(
                                self.products,
                                city=self.registry.provider_scope(
                                    place.city_id, ProviderCode.FLYAI
                                ),
                                canonical_id=place.canonical_entity_id,
                                name=place.display_name,
                                days=bundle.service_dates,
                            )
                    except (ProviderError, TimeoutError):
                        pass
                if tickets:
                    await self.repository.save(
                        turn_id, bundle.model_copy(update={"tickets": tickets})
                    )

        async with asyncio.timeout(45):
            async with asyncio.TaskGroup() as group:
                for bundle in bundles:
                    group.create_task(one(bundle))

    async def wait(self, trip_id: UUID, *, timeout: float = 2) -> None:
        """A short optional join avoids racing a just-finishing Prepare lookup."""
        tasks = [t for (trip, _), t in self._tasks.items() if trip == trip_id]
        if tasks:
            await asyncio.wait(tasks, timeout=timeout)

    async def aclose(self) -> None:
        tasks = tuple(self._tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
