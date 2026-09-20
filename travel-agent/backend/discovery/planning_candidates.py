"""Prepare a larger private pool while the conversation continues."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Literal
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import Field

from backend.agent.model_gateway import (
    ModelAuditMetadata,
    ModelCancellation,
    ModelContract,
    ModelGateway,
    ModelGatewayError,
    ModelMessage,
    ModelRequest,
    ModelRole,
)
from backend.agent.planner.evidence import _place_evidence
from backend.agent.planner.workspace import server_id
from backend.contracts.enums import PlaceCategory, ProviderCode
from backend.contracts.prepared_candidates import PreparedPlanningCandidate, PreparedPlanningPool
from backend.contracts.prepared_evidence import PreparedCandidateEvidence
from backend.contracts.v4.cards import SpecificCandidateCard
from backend.contracts.v4.enums import CandidateEntityKind, CardDomain, DiscoverySection
from backend.contracts.v4.planner_evidence import PlannerCandidateOrigin, PlannerPlaceEvidence
from backend.contracts.v4.state import TripSemanticState
from backend.contracts.v4.visit_duration import VisitDurationRange
from backend.discovery.cards.attraction_schema import attraction_schema
from backend.discovery.cards.candidate_composition import specific_dependency_fingerprint
from backend.discovery.prepared_evidence import PreparedEvidenceCollector
from backend.persistence.outbox_repository import canonical_json_hash
from backend.persistence.prepared_candidates_repository import PreparedCandidatesRepository
from backend.persistence.turn_repository import TurnRepository
from backend.planning.city_registry import CityRegistry
from backend.providers.contracts import KeywordPlaceSearchRequest, ProviderError
from backend.providers.interfaces import PlaceProvider
from backend.providers.place_matching import matches_place_search_identity
from backend.providers.place_taxonomy import category_from_original_typecodes

_LOG = logging.getLogger(__name__)
Domain = Literal["attraction", "dining"]
SECTIONS: dict[Domain, DiscoverySection] = {
    "attraction": DiscoverySection.ATTRACTION_SPECIFIC,
    "dining": DiscoverySection.DINING_SPECIFIC,
}
SUPPLEMENT_PROMPT = """为旅行规划补充备选地点，不修改用户意愿或安排日程。
程序提供当前清单、偏好、排除项和 missing_count。只补 missing_count 个互不重复的新地点，
已有地点、别名、同一景区内的子点和排除项都不能再次推荐；避开本次核验失败的线索。
景点兼顾兴趣、大小和区域分布，方便通常每天安排 3 个、根据规模安排 2—4 个。
餐厅给出具体店铺，尽量注明分店，照顾餐饮偏好和景点区域，不用菜名代替店名。
这里只提出搜索线索，后续由地图核验；不编造价格、营业时间或已核验的说法。
景点同时给出基本到深游的建议分钟范围；餐厅的建议游览时长填 null。"""


class SupplementClue(ModelContract):
    name: str = Field(min_length=2, max_length=100)
    suggested_visit_duration: VisitDurationRange | None = None


class CandidateSupplement(ModelContract):
    candidates: tuple[SupplementClue, ...] = Field(max_length=15)


def context_fingerprint(state: TripSemanticState, domain: Domain) -> str:
    # Exclusions change the retained list, not the validity of its sibling identities.
    clean = state.model_copy(
        update={
            "attractions": state.attractions.model_copy(update={"exclusions": []}),
            "dining": state.dining.model_copy(update={"exclusions": []}),
        }
    )
    return specific_dependency_fingerprint(clean, CardDomain(domain))


def selection_fingerprint(state: TripSemanticState, domain: Domain) -> str:
    projection = state.attractions if domain == "attraction" else state.dining
    return canonical_json_hash(
        {
            "context": context_fingerprint(state, domain),
            "selection": projection.model_dump(mode="json"),
        }
    )


def sightseeing_days(state: TripSemanticState) -> int:
    days = state.trip_basics.duration_days
    if days is None:
        raise ValueError("candidate pool requires confirmed travel duration")
    return days


def normalized_name(name: str) -> str:
    return re.sub(r"[\W_]+", "", name.casefold())


def same_place(left: PlannerPlaceEvidence, right: PlannerPlaceEvidence) -> bool:
    if left.provider_entity_id == right.provider_entity_id:
        return True
    if normalized_name(left.display_name) == normalized_name(right.display_name):
        return True
    # Entrances and nested attractions must not inflate the pool's count.
    if left.entity_kind is CandidateEntityKind.ATTRACTION:
        return bool(
            left.provider_parent_place_id == right.provider_entity_id
            or right.provider_parent_place_id == left.provider_entity_id
            or (
                left.provider_parent_place_id
                and left.provider_parent_place_id == right.provider_parent_place_id
            )
        )
    return False


def retained_candidates(
    values: tuple[PreparedPlanningCandidate, ...],
    state: TripSemanticState,
    domain: Domain,
    now: datetime,
) -> tuple[PreparedPlanningCandidate, ...]:
    projection = state.attractions if domain == "attraction" else state.dining
    intents = (
        state.attractions.concrete_intents
        if domain == "attraction"
        else state.dining.concrete_restaurant_intents
    )
    excluded = [*projection.exclusions, *(i for i in intents if i.disposition == "avoid")]
    excluded_ids = {i.canonical_entity_id for i in excluded}
    excluded_names = {normalized_name(i.display_name) for i in excluded}
    priorities = {"must": 0, "destination": 0, "want": 1, "if_convenient": 2}
    priority = {i.canonical_entity_id: priorities.get(i.disposition, 3) for i in intents}
    category = PlaceCategory.ATTRACTION if domain == "attraction" else PlaceCategory.RESTAURANT
    kept: list[PreparedPlanningCandidate] = []
    for value in sorted(values, key=lambda v: priority.get(v.place.canonical_entity_id, 3)):
        place = value.place
        if (
            place.canonical_entity_id in excluded_ids
            or normalized_name(place.display_name) in excluded_names
            or (
                place.provider_parent_place_id
                and str(uuid5(NAMESPACE_URL, f"amap:{place.provider_parent_place_id}"))
                in excluded_ids
            )
            or (
                domain == "attraction"
                and any(
                    matches_place_search_identity(
                        item.display_name,
                        place.display_name,
                        city_name=state.trip_basics.destination_name or "",
                        category=category,
                    )
                    for item in excluded
                )
            )
            or place.city_id != state.trip_basics.destination_canonical_id
            or category_from_original_typecodes(place.provider_typecode) is not category
            or not timedelta(0) <= now - place.observed_at < timedelta(hours=24)
            or any(same_place(place, old.place) for old in kept)
        ):
            continue
        kept.append(value)
    return tuple(kept)


class PreparedPlanningPoolService:
    def __init__(
        self,
        *,
        repository: PreparedCandidatesRepository,
        turns: TurnRepository,
        facts: PreparedEvidenceCollector,
        gateway: ModelGateway,
        places: PlaceProvider,
        registry: CityRegistry,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.repository, self.turns, self.facts = repository, turns, facts
        self.gateway, self.places, self.registry, self.clock = gateway, places, registry, clock
        self._tasks: dict[tuple[UUID, Domain], tuple[str, asyncio.Task[PreparedPlanningPool]]] = {}
        self._queries = asyncio.Semaphore(4)

    async def on_commit(self, owner: UUID, trip: UUID, version: int) -> None:
        """Only small local writes are awaited; all LLM/provider work is background."""
        snapshot = await self.turns.load_conversation_snapshot(owner, trip)
        state = snapshot.trip_state.semantic_state
        if state.state_version != version or not state.trip_basics.duration_days:
            return
        for message in snapshot.messages:
            if message.state_version != version or message.role != "assistant":
                continue
            for attachment in message.attachments:
                card = attachment.root
                if not isinstance(card, SpecificCandidateCard) or card.domain.value not in SECTIONS:
                    continue
                domain = card.domain.value
                origins = []
                for option in card.options:
                    ref = option.entity_ref
                    if ref is None:
                        continue
                    for source in ref.provider_entity_refs:
                        if source.startswith("provider:amap:"):
                            origins.append(
                                PlannerCandidateOrigin(
                                    canonical_entity_id=ref.canonical_entity_id,
                                    provider_entity_id=source.removeprefix("provider:amap:"),
                                    entity_kind=CandidateEntityKind(ref.entity_kind),
                                    source_message_id=message.message_id,
                                    source_option_id=option.option_id,
                                    display_name=option.label,
                                    inherit_as_neutral=True,
                                    source_attachment_id=card.attachment_id,
                                    dependency_fingerprint=card.dependency_fingerprint,
                                    suggested_visit_duration=option.suggested_visit_duration,
                                )
                            )
                bundles = await self.facts.repository.load(
                    owner,
                    trip,
                    origins=tuple(origins),
                    based_on_state_version=version,
                    now=self.clock(),
                )
                origin_map = {o.canonical_entity_id: o for o in origins}
                values = tuple(
                    PreparedPlanningCandidate(
                        origin=origin_map[b.place.canonical_entity_id], place=b.place
                    )
                    for b in bundles
                )
                seed = PreparedPlanningPool(
                    trip_id=trip,
                    domain=domain,
                    context_fingerprint=context_fingerprint(state, domain),
                    selection_fingerprint=selection_fingerprint(state, domain),
                    source_turn_id=UUID(message.turn_id),
                    source_state_version=version,
                    target_count=3 * sightseeing_days(state),
                    candidates=retained_candidates(values, state, domain, self.clock()),
                    updated_at=self.clock(),
                )
                await self.repository.save(owner, seed)
        for domain, section in SECTIONS.items():
            coverage = snapshot.trip_state.discovery_runtime_state.section_coverage[section]
            if coverage.status.value in {"complete", "not_applicable"}:
                confirmed = state.confirmed_task_book_ref
                self._start(
                    owner,
                    state,
                    domain,
                    min(version, confirmed.based_on_state_version) if confirmed else version,
                )

    def _start(
        self,
        owner: UUID,
        state: TripSemanticState,
        domain: Domain,
        version: int,
    ) -> asyncio.Task[PreparedPlanningPool]:
        key = (UUID(state.trip_id), domain)
        fingerprint = selection_fingerprint(state, domain)
        current = self._tasks.get(key)
        if current and current[0] == fingerprint:
            return current[1]
        if current and not current[1].done():
            current[1].cancel()
        task = asyncio.create_task(self._prepare(owner, state, domain, version))
        self._tasks[key] = (fingerprint, task)

        def done(completed: asyncio.Task[PreparedPlanningPool]) -> None:
            self._observe(completed)
            if self._tasks.get(key, (None, None))[1] is completed:
                self._tasks.pop(key, None)

        task.add_done_callback(done)
        return task

    @staticmethod
    def _observe(task: asyncio.Task[PreparedPlanningPool]) -> None:
        if not task.cancelled() and (error := task.exception()) is not None:
            _LOG.warning("prepare candidate pool stopped: %s", type(error).__name__)

    async def ensure(
        self,
        owner: UUID,
        state: TripSemanticState,
        based_on_state_version: int,
    ) -> tuple[PreparedPlanningPool, ...]:
        pools = tuple(
            await asyncio.gather(
                *(
                    asyncio.shield(self._start(owner, state, domain, based_on_state_version))
                    for domain in SECTIONS
                )
            )
        )
        return pools

    async def _prepare(
        self,
        owner: UUID,
        state: TripSemanticState,
        domain: Domain,
        version: int,
    ) -> PreparedPlanningPool:
        trip = UUID(state.trip_id)
        context, selection = (
            context_fingerprint(state, domain),
            selection_fingerprint(state, domain),
        )
        old = await self.repository.load(
            owner,
            trip,
            domain,
            context,
            selection_fingerprint=selection,
            based_on_state_version=version,
            now=self.clock(),
        )
        if old and (old.status == "ready" or old.model_rounds >= 3):
            valid = retained_candidates(old.candidates, state, domain, self.clock())
            if len(valid) == len(old.candidates):
                return old
        seed = await self.repository.load(
            owner,
            trip,
            domain,
            context,
            seed=True,
            based_on_state_version=version,
            now=self.clock(),
        )
        previous = old or await self.repository.load(
            owner, trip, domain, context, based_on_state_version=version, now=self.clock()
        )
        values = (*(seed.candidates if seed else ()), *(previous.candidates if previous else ()))
        source_turn, source_version = await self.repository.source_turn(owner, trip, version)
        pool = PreparedPlanningPool(
            trip_id=trip,
            domain=domain,
            context_fingerprint=context,
            selection_fingerprint=selection,
            source_turn_id=source_turn,
            source_state_version=source_version,
            target_count=3 * sightseeing_days(state),
            candidates=retained_candidates(values, state, domain, self.clock()),
            status="building",
            model_rounds=old.model_rounds if old else 0,
            updated_at=self.clock(),
        )
        await self.repository.save(owner, pool)
        rejected: list[str] = []
        failures: list[str] = []
        while pool.missing_count and pool.model_rounds < 3:
            pool = pool.model_copy(update={"model_rounds": pool.model_rounds + 1})
            await self.repository.save(owner, pool)
            try:
                clues = await self._suggest(state, pool, rejected)
                results = await asyncio.gather(
                    *(self._verify(clue, state, pool, seed) for clue in clues)
                )
                for clue, candidate in zip(clues, results, strict=True):
                    retained = retained_candidates(
                        (*pool.candidates, *((candidate,) if candidate else ())),
                        state,
                        domain,
                        self.clock(),
                    )
                    if len(retained) == len(pool.candidates):
                        rejected.append(clue.name)
                    else:
                        assert candidate is not None
                        pool = pool.model_copy(
                            update={"candidates": retained, "updated_at": self.clock()}
                        )
                        await self.repository.save(owner, pool)
                        await self._remember(pool, candidate, state)
            except (ModelGatewayError, ProviderError, TimeoutError) as error:
                failures.append(type(error).__name__)
                # A provider outage does not justify repeating the same whole batch.
                break
        pool = pool.model_copy(
            update={
                "status": "partial" if pool.missing_count else "ready",
                "failures": tuple(
                    failures + (["unverified_or_duplicate_candidates"] if rejected else [])
                ),
                "updated_at": self.clock(),
            }
        )
        await self.repository.save(owner, pool)
        return pool

    async def _suggest(
        self,
        state: TripSemanticState,
        pool: PreparedPlanningPool,
        rejected: list[str],
    ) -> tuple[SupplementClue, ...]:
        schema = attraction_schema(CandidateSupplement)
        schema["properties"]["candidates"].update(
            minItems=pool.missing_count, maxItems=pool.missing_count
        )
        projection = state.attractions if pool.domain == "attraction" else state.dining
        payload = {
            "destination": state.trip_basics.destination_name,
            "domain": pool.domain,
            "sightseeing_days": state.trip_basics.duration_days,
            "missing_count": pool.missing_count,
            "preferences_and_exclusions": projection.model_dump(mode="json"),
            "existing_candidates": [
                {"name": c.place.display_name, "address": c.place.address} for c in pool.candidates
            ],
            "attraction_intents": [
                i.model_dump(mode="json") for i in state.attractions.concrete_intents
            ],
            "previously_rejected_clues": rejected,
        }
        response = await self.gateway.generate_structured(
            ModelRequest(
                messages=[
                    ModelMessage(role=ModelRole.SYSTEM, content=SUPPLEMENT_PROMPT),
                    ModelMessage(
                        role=ModelRole.USER, content=json.dumps(payload, ensure_ascii=False)
                    ),
                ],
                structured_output_mode="json_schema",
                output_schema_override=schema,
                max_output_tokens=3072,
                request_timeout_seconds=45,
                audit=ModelAuditMetadata(
                    stage="prepare_planning_candidates",
                    node=pool.domain,
                    contract_version="prepare-pool-v1",
                    attempt=pool.model_rounds,
                ),
            ),
            CandidateSupplement,
            cancellation=ModelCancellation(),
        )
        return response.value.candidates[: pool.missing_count]

    async def _verify(
        self,
        clue: SupplementClue,
        state: TripSemanticState,
        pool: PreparedPlanningPool,
        seed: PreparedPlanningPool | None,
    ) -> PreparedPlanningCandidate | None:
        category = (
            PlaceCategory.ATTRACTION if pool.domain == "attraction" else PlaceCategory.RESTAURANT
        )
        kind = (
            CandidateEntityKind.ATTRACTION
            if pool.domain == "attraction"
            else CandidateEntityKind.RESTAURANT
        )
        city_id = state.trip_basics.destination_canonical_id
        if city_id is None:
            raise ValueError("candidate pool requires a confirmed destination")
        scope = self.registry.provider_scope(city_id, ProviderCode.AMAP)
        if any(
            normalized_name(clue.name) == normalized_name(c.place.display_name)
            for c in pool.candidates
        ):
            return None
        try:
            async with self._queries, asyncio.timeout(15):
                response = await self.places.search_places(
                    KeywordPlaceSearchRequest(
                        city=scope,
                        query=clue.name,
                        category_hint=category,
                        page_size=5,
                    )
                )
        except (ProviderError, TimeoutError):
            return None
        for place in response.items:
            if (
                place.provider is not ProviderCode.AMAP
                or place.city_id != scope.city_id
                or category_from_original_typecodes(place.provider_typecode) is not category
                or not matches_place_search_identity(
                    clue.name,
                    place.name,
                    city_name=scope.display_name or scope.city_id,
                    category=category,
                )
            ):
                continue
            # A specified restaurant branch must not silently resolve to a sibling branch.
            branch = re.findall(r"[（(]([^）)]+)[）)]", clue.name)
            if (
                category is PlaceCategory.RESTAURANT
                and branch
                and not all(normalized_name(b) in normalized_name(place.name) for b in branch)
            ):
                continue
            evidence = _place_evidence(place, kind)
            origin = PlannerCandidateOrigin(
                canonical_entity_id=evidence.canonical_entity_id,
                provider_entity_id=place.source_place_id,
                entity_kind=kind,
                source_kind="prepare_pool",
                source_message_id=(
                    seed.candidates[0].origin.source_message_id
                    if seed and seed.candidates
                    else f"turn:{pool.source_turn_id}"
                ),
                source_option_id=server_id(
                    "prepared-option", pool.selection_fingerprint, place.source_place_id
                ),
                display_name=place.name,
                inherit_as_neutral=True,
                source_attachment_id=server_id(
                    "prepared-pool", pool.trip_id, pool.selection_fingerprint
                ),
                dependency_fingerprint=pool.selection_fingerprint,
                suggested_visit_duration=clue.suggested_visit_duration
                if pool.domain == "attraction"
                else None,
            )
            return PreparedPlanningCandidate(origin=origin, place=evidence)
        return None

    async def _remember(
        self,
        pool: PreparedPlanningPool,
        candidate: PreparedPlanningCandidate,
        state: TripSemanticState,
    ) -> None:
        start = state.trip_basics.start_date
        if start is None:
            return
        assert candidate.origin.source_attachment_id and candidate.origin.dependency_fingerprint
        await self.facts.remember_supplement(
            pool.source_turn_id,
            PreparedCandidateEvidence(
                trip_id=pool.trip_id,
                source_attachment_id=candidate.origin.source_attachment_id,
                dependency_fingerprint=candidate.origin.dependency_fingerprint,
                service_dates=tuple(
                    start + timedelta(days=i) for i in range(sightseeing_days(state))
                ),
                place=candidate.place,
            ),
        )

    async def aclose(self) -> None:
        tasks = [task for _, task in self._tasks.values()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
