"""Parallel city prefetch, signed personal hints, and bounded free-text supplementation."""

from __future__ import annotations

import asyncio
import json
import math
import re
import time
from datetime import UTC, datetime
from typing import Literal
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5
from weakref import WeakValueDictionary

from pydantic import Field
from redis.exceptions import RedisError

from backend.agent.model_audit import current_model_audit_execution, record_execution_event
from backend.agent.model_gateway import (
    ModelAuditMetadata,
    ModelCancellation,
    ModelFailureCode,
    ModelGateway,
    ModelGatewayError,
    ModelMessage,
    ModelRequest,
    ModelRole,
)
from backend.agent.prepare.progress import AttractionProgress, report_attraction_progress
from backend.contracts.candidate_recall import (
    CandidateDomain,
    CandidateRecallRequest,
    CandidateSourceReference,
    RecallAttempt,
    RecallAttemptStatus,
    RecallChannel,
    RecalledCandidate,
    RecalledPlace,
    RecallFailureCode,
    RecallSourceKind,
)
from backend.contracts.enums import DataAvailability, PlaceCategory, ProviderCode
from backend.contracts.v4.base import DisplayText, Identifier, V4ContractModel
from backend.discovery.cards.attraction_recall_prompts import (
    CITY_REPRESENTATIVE_PROMPT,
    SUPPLEMENT_PROMPT,
)
from backend.discovery.cards.attraction_schema import attraction_schema
from backend.persistence.redis_temporary import RedisTemporaryStore
from backend.planning.city_registry import CityRegistry
from backend.providers.contracts import KeywordPlaceSearchRequest, ProviderError, ProviderPlace
from backend.providers.interfaces import PlaceProvider
from backend.providers.place_matching import matches_place_search_identity
from backend.providers.place_taxonomy import original_typecodes

VERSION = "attraction-parallel-v3-20260916-flash-schema-broad-types-v2"
MAX_SUPPLEMENT_ROUNDS = 2
QUERY_LIMIT = 8
CACHE_SECONDS = 1_800


class CityRepresentative(V4ContractModel):
    name: DisplayText
    category: DisplayText
    local_significance: DisplayText


class CityRepresentatives(V4ContractModel):
    places: list[CityRepresentative] = Field(max_length=15)
    limitations: DisplayText | None = None


class AttractionQuery(V4ContractModel):
    kind: Literal["named_place", "category"]
    keyword: str = Field(min_length=2, max_length=80)
    related_direction_ids: list[Identifier] = Field(default_factory=list, max_length=7)
    reason: str = Field(min_length=6, max_length=160)

    @property
    def limit(self) -> int:
        return 1 if self.kind == "named_place" else 2

    @property
    def identity(self) -> str:
        return f"{self.kind}:{''.join(self.keyword.casefold().split())}"


class CoverageGap(V4ContractModel):
    kind: Literal["count", "personal_preference", "city_representation", "diversity", "identity"]
    detail: DisplayText


class SupplementPlan(V4ContractModel):
    coverage_gaps: list[CoverageGap] = Field(default_factory=list, max_length=8)
    queries: list[AttractionQuery] = Field(max_length=QUERY_LIMIT)


class AttractionSearchObservation(V4ContractModel):
    query: AttractionQuery
    branch: Literal["city", "personalized", "supplement"]
    attempt: RecallAttempt
    candidates: list[RecalledCandidate] = Field(default_factory=list)
    skipped: list[dict[str, str | int]] = Field(default_factory=list)
    started_at: datetime
    duration_seconds: float
    cache_hit: bool = False


class AttractionPool(V4ContractModel):
    observations: list[AttractionSearchObservation] = Field(default_factory=list)
    candidates: tuple[RecalledCandidate, ...] = ()
    supplemental_rounds: int = 0
    limitations: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def attempts(self) -> tuple[RecallAttempt, ...]:
        return tuple(item.attempt for item in self.observations)

    @property
    def provider_call_count(self) -> int:
        return sum(not item.cache_hit for item in self.observations)


def pool_targets(request: CandidateRecallRequest) -> tuple[int, int]:
    assert request.attraction_discovery is not None
    preferred = 10 if request.day_count == 1 else 20 if request.day_count <= 3 else 25
    return preferred, math.ceil(request.attraction_discovery.maximum_target * 1.5)


def candidate_projection(candidate: RecalledCandidate) -> dict[str, object]:
    place = candidate.place
    return {
        "candidate_key": str(place.place_id),
        "name": place.name,
        "city_id": place.city_id,
        "address": place.address,
        "coordinates": place.coordinates.model_dump(mode="json") if place.coordinates else None,
        "typecode": place.provider_typecode,
        "parent_id": place.provider_parent_place_id,
        "related_direction_ids": list(candidate.theme_ids),
        "recall_channels": [item.value for item in candidate.channels],
        "search_intent_hints": list(candidate.reasons),
        "named_identity_matched": candidate.representative_identity_verified,
    }


def preference_projection(request: CandidateRecallRequest) -> dict[str, object]:
    context = request.attraction_discovery
    assert context is not None
    return {
        "liked_directions": [
            item.model_dump(mode="json") for item in context.directions if item.selected
        ],
        "excluded_directions": [
            item.model_dump(mode="json") for item in context.directions if not item.selected
        ],
        "free_text_preferences": list(request.free_text_clues),
        "explicit_constraints": list(request.special_constraints),
        "explicit_place_intents": list(context.explicit_place_intents),
        "excluded_places": [item.model_dump(mode="json") for item in request.excluded_places],
        "cold_start": list(context.cold_start_defaults),
        "travelers": list(context.travelers),
        "trip_goals": list(context.trip_goals),
    }


def merge_candidates(
    observations: list[AttractionSearchObservation],
) -> tuple[RecalledCandidate, ...]:
    """POI identity only; shared people, categories and parents do not imply duplicates."""
    by_id: dict[UUID, RecalledCandidate] = {}
    for observation in observations:
        for candidate in observation.candidates:
            key = candidate.place.place_id
            previous = by_id.get(key)
            if previous is None:
                by_id[key] = candidate
                continue
            sources = {
                source.model_dump_json(): source
                for source in (*previous.sources, *candidate.sources)
            }
            by_id[key] = previous.model_copy(
                update={
                    "channels": tuple(dict.fromkeys((*previous.channels, *candidate.channels))),
                    "theme_ids": tuple(dict.fromkeys((*previous.theme_ids, *candidate.theme_ids))),
                    "reasons": tuple(dict.fromkeys((*previous.reasons, *candidate.reasons))),
                    "sources": tuple(sources.values()),
                    "representative_identity_verified": previous.representative_identity_verified
                    or candidate.representative_identity_verified,
                }
            )
    return tuple(by_id.values())


def candidates_for_request(
    request: CandidateRecallRequest, observations: list[AttractionSearchObservation]
) -> tuple[RecalledCandidate, ...]:
    excluded_ids = {item.known_place_id for item in request.excluded_places if item.known_place_id}
    excluded_names = {item.name.casefold().strip() for item in request.excluded_places}
    return tuple(
        item
        for item in merge_candidates(observations)
        if item.place.place_id not in excluded_ids
        and item.place.name.casefold().strip() not in excluded_names
    )


# Broad searchable domains, independent of the user's preference-card selections.
ATTRACTION_SEARCH_TYPECODES = ("110000", "140000", "080000", "060000")


def irrelevant_poi_type(place: ProviderPlace) -> str | None:
    codes = original_typecodes(place.provider_typecode)
    # An industrial tourism site may carry both company and scenic tags.
    # Reject only unambiguous unwanted types, not legitimate mixed-use landmarks.
    if codes and all(code.startswith(("17", "1203", "0602")) for code in codes):
        return "non_visitable_poi_type"
    return None


def closed_evidence(place: ProviderPlace) -> str | None:
    pattern = (
        r"暂停营业|暂停开放|暂不营业|暂未开放|已关闭|已停业|歇业|永久关闭"
        r"|temporarilyclosed|permanentlyclosed"
    )
    if re.search(pattern, "".join(place.name.casefold().split())):
        return "provider_name_closed_marker"
    # Do not interpret empty hours, ordinary off-hours or unknown numeric codes as closure.
    for field in ("business_status", "operating_status"):
        status = place.raw_payload.get(field)
        if isinstance(status, str) and status.strip() in {
            "停业",
            "已停业",
            "暂停营业",
            "暂停开放",
            "永久关闭",
        }:
            return f"provider_{field}_closed"
    return None


async def _audit(event: str, payload: dict[str, object]) -> None:
    context = current_model_audit_execution()
    if context is not None:
        await record_execution_event(context, event, payload)


def _observe_task(task: asyncio.Task[AttractionPool]) -> None:
    # Detached prefetch may outlive its original turn or be replaced by a new city.
    # Retrieving the exception prevents orphan warnings without changing await semantics.
    if not task.cancelled():
        task.exception()


class AttractionRecallService:
    """One managed task per trip/city; reusable facts live in the existing Redis cache."""

    def __init__(
        self,
        *,
        gateway: ModelGateway,
        places: PlaceProvider,
        registry: CityRegistry,
        store: RedisTemporaryStore | None = None,
        concurrency: int = 6,
    ) -> None:
        self._gateway = gateway
        self._places = places
        self._registry = registry
        self._store = store
        self._semaphore = asyncio.Semaphore(concurrency)
        self._city_tasks: dict[tuple[str, str], asyncio.Task[AttractionPool]] = {}
        self._city_tokens: dict[tuple[str, str], ModelCancellation] = {}
        self._query_locks: WeakValueDictionary[str, asyncio.Lock] = WeakValueDictionary()

    def start_city(self, trip_id: str, city_id: str) -> asyncio.Task[AttractionPool]:
        self._registry.resolve(city_id)
        key = (trip_id, city_id)
        for old_key, task in list(self._city_tasks.items()):
            if old_key[0] == trip_id and old_key != key:
                self._city_tokens[old_key].cancel()
                task.cancel()
                del self._city_tasks[old_key]
                del self._city_tokens[old_key]
            elif len(self._city_tasks) > 128 and task.done() and old_key != key:
                del self._city_tasks[old_key]
                del self._city_tokens[old_key]
        existing = self._city_tasks.get(key)
        if (
            existing is not None
            and existing.done()
            and (
                existing.cancelled()
                or existing.exception() is not None
                or not existing.result().candidates
            )
        ):
            del self._city_tasks[key]
            self._city_tokens.pop(key, None)
        if key not in self._city_tasks:
            token = ModelCancellation()
            self._city_tokens[key] = token
            task = asyncio.create_task(self._city_pool(trip_id, city_id, token))
            task.add_done_callback(_observe_task)
            self._city_tasks[key] = task
        return self._city_tasks[key]

    async def close(self) -> None:
        tasks = list(self._city_tasks.values())
        for token in self._city_tokens.values():
            token.cancel()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._city_tasks.clear()
        self._city_tokens.clear()

    async def _get(self, kind: str, params: dict[str, object]) -> object:
        if self._store is None:
            return None
        return await self._store.get_cache(f"attraction-{kind}", VERSION, params)

    async def _put(self, kind: str, params: dict[str, object], value: object) -> None:
        if self._store is not None:
            await self._store.put_cache(f"attraction-{kind}", VERSION, params, value, CACHE_SECONDS)

    async def _city_pool(
        self, trip_id: str, city_id: str, token: ModelCancellation
    ) -> AttractionPool:
        params: dict[str, object] = {"trip_id": trip_id, "city_id": city_id}
        lease_key = f"attraction-prefetch:{trip_id}:{city_id}:{VERSION}"
        lease = str(uuid4())
        acquired = False
        try:
            cached = await self._get("city-pool", params)
            if isinstance(cached, dict):
                return AttractionPool.model_validate(cached)
            if self._store is not None:
                # Separate from the conversation lease: the next foreground turn must not cancel it.
                for _ in range(181):
                    token.raise_if_cancelled("attraction_city_lease")
                    acquired = await self._store.acquire_trip_lock(lease_key, lease, 180)
                    if acquired:
                        break
                    cached = await self._get("city-pool", params)
                    if isinstance(cached, dict):
                        return AttractionPool.model_validate(cached)
                    await asyncio.sleep(1)
                if not acquired:
                    return AttractionPool(limitations=["city_prefetch_lease_timeout"])
            city = self._registry.resolve(city_id)
            cached_seeds = await self._get("city-seeds", params)
            if isinstance(cached_seeds, dict):
                seeds = CityRepresentatives.model_validate(cached_seeds)
            else:
                result = await self._gateway.generate_structured(
                    ModelRequest(
                        messages=[
                            ModelMessage(role=ModelRole.SYSTEM, content=CITY_REPRESENTATIVE_PROMPT),
                            ModelMessage(
                                role=ModelRole.USER,
                                content=json.dumps(
                                    {
                                        "destination": {
                                            "city_id": city_id,
                                            "name": city.display_name,
                                        },
                                        "evidence": [],
                                    },
                                    ensure_ascii=False,
                                ),
                            ),
                        ],
                        structured_output_mode="json_schema",
                        output_schema_override=attraction_schema(CityRepresentatives),
                        max_output_tokens=3_072,
                        audit=ModelAuditMetadata(
                            stage="prepare_attraction_city_representatives",
                            contract_version=VERSION,
                        ),
                    ),
                    CityRepresentatives,
                    cancellation=token,
                )
                seeds = result.value
                await self._put("city-seeds", params, seeds.model_dump(mode="json"))
            queries = [
                AttractionQuery(
                    kind="named_place",
                    keyword=place.name,
                    reason=f"城市代表搜索线索：{place.local_significance}"[:160],
                )
                for place in seeds.places
            ]
            observations = await self._search_many(trip_id, city_id, queries, "city", token)
            pool = AttractionPool(
                observations=observations,
                candidates=merge_candidates(observations),
                limitations=[seeds.limitations] if seeds.limitations else [],
            )
            token.raise_if_cancelled("attraction_city_cache")
            await self._put("city-pool", params, pool.model_dump(mode="json"))
            await _audit(
                "attraction_city_prefetch_completed",
                {
                    "city_id": city_id,
                    "seed_count": len(seeds.places),
                    "candidate_count": len(pool.candidates),
                    "query_count": len(observations),
                },
            )
            return pool
        except ModelGatewayError as error:
            if error.code is ModelFailureCode.CANCELLED:
                raise
            await _audit(
                "attraction_city_prefetch_failed",
                {"city_id": city_id, "failure_code": error.code.value},
            )
            return AttractionPool(limitations=[f"city_prefetch_{error.code.value}"])
        except (ValueError, ProviderError, RedisError) as error:
            await _audit(
                "attraction_city_prefetch_failed",
                {"city_id": city_id, "failure_code": type(error).__name__},
            )
            return AttractionPool(limitations=["city_prefetch_invalid_response"])
        finally:
            if acquired and self._store is not None:
                try:
                    await self._store.release_trip_lock(lease_key, lease)
                except RedisError:
                    # The lease expires; a cache outage must not hide the computed result.
                    await _audit("attraction_city_lease_release_failed", {"city_id": city_id})

    async def recall(
        self, request: CandidateRecallRequest, *, cancellation: ModelCancellation | None = None
    ) -> AttractionPool:
        assert request.attraction_discovery is not None
        await report_attraction_progress(AttractionProgress.SPECIFIC_SEARCH)
        city_task = self.start_city(str(request.trip_id), request.city_id)
        queries: list[AttractionQuery] = []
        for direction in request.attraction_discovery.directions:
            if not direction.selected:
                continue
            hints = direction.attraction_search_hints
            if hints is not None:
                queries.extend(
                    AttractionQuery(
                        kind="named_place",
                        keyword=place.name,
                        related_direction_ids=[direction.direction_id],
                        reason=f"用户明确喜欢的方向：{direction.label}",
                    )
                    for place in hints.representative_places
                )
                words = hints.search_queries
            else:
                # Old signed cards keep their selections and their already generated clue.
                words = [direction.search_query] if direction.search_query else []
            queries.extend(
                AttractionQuery(
                    kind="category",
                    keyword=word,
                    related_direction_ids=[direction.direction_id],
                    reason=f"用户明确喜欢的方向：{direction.label}",
                )
                for word in words
            )
        personal_task = asyncio.create_task(
            self._search_many(
                str(request.trip_id), request.city_id, queries, "personalized", cancellation
            )
        )
        cancellation_wait = (
            asyncio.create_task(cancellation.wait_cancelled()) if cancellation is not None else None
        )
        try:
            wait_for: set[asyncio.Task[AttractionPool] | asyncio.Task[None]] = {city_task}
            if cancellation_wait is not None:
                wait_for.add(cancellation_wait)
            done, _ = await asyncio.wait(wait_for, timeout=60, return_when=asyncio.FIRST_COMPLETED)
            if cancellation is not None:
                cancellation.raise_if_cancelled("attraction_city_wait")
            city_pool = (
                city_task.result()
                if city_task in done
                else AttractionPool(limitations=["city_prefetch_still_running"])
            )
            personal = await personal_task
        except BaseException:
            # Personal queries belong to this turn. The city task has its own lifecycle.
            personal_task.cancel()
            await asyncio.gather(personal_task, return_exceptions=True)
            raise
        finally:
            if cancellation_wait is not None:
                cancellation_wait.cancel()
                await asyncio.gather(cancellation_wait, return_exceptions=True)
        if cancellation is not None:
            cancellation.raise_if_cancelled("attraction_pool_merge")
        observations = [*city_pool.observations, *personal]
        pool = AttractionPool(
            observations=observations,
            candidates=candidates_for_request(request, observations),
            limitations=city_pool.limitations,
        )
        return await self.supplement(request, pool, cancellation=cancellation)

    async def supplement(
        self,
        request: CandidateRecallRequest,
        pool: AttractionPool,
        *,
        cancellation: ModelCancellation | None = None,
        selection_feedback: list[str] | None = None,
    ) -> AttractionPool:
        context = request.attraction_discovery
        assert context is not None
        preferred, minimum = pool_targets(request)
        known_direction_ids = {
            direction.direction_id for direction in context.directions if direction.selected
        }
        while pool.supplemental_rounds < MAX_SUPPLEMENT_ROUNDS:
            await report_attraction_progress(AttractionProgress.SPECIFIC_SUPPLEMENT)
            if cancellation is not None:
                cancellation.raise_if_cancelled("attraction_supplement")
            n = len(pool.candidates)
            payload: dict[str, object] = {
                "destination": {"city_id": request.city_id, "name": context.destination_name},
                "preferences": preference_projection(request),
                "limits": {
                    "duration_days": request.day_count,
                    "final_minimum": context.minimum_target,
                    "final_maximum": context.maximum_target,
                    "preferred_pool_size": preferred,
                    "minimum_pool_size": minimum,
                    "current_pool_size": n,
                    "minimum_shortfall": max(0, minimum - n),
                    "preferred_shortfall": max(0, preferred - n),
                    "query_limit": QUERY_LIMIT,
                    "remaining_query_budget": QUERY_LIMIT
                    * (MAX_SUPPLEMENT_ROUNDS - pool.supplemental_rounds),
                },
                "candidates": [candidate_projection(item) for item in pool.candidates],
                "query_history": [
                    {
                        "query": item.query.model_dump(mode="json"),
                        "status": item.attempt.status.value,
                        "accepted_count": item.attempt.accepted_count,
                        "candidate_keys": [str(c.place.place_id) for c in item.candidates],
                        "skipped": item.skipped,
                    }
                    for item in pool.observations
                ],
                "selection_feedback": selection_feedback or [],
                "limitations": pool.limitations,
            }
            previous_ids = {
                item.query.identity
                for item in pool.observations
                if item.attempt.status is not RecallAttemptStatus.FAILED
            }
            plan: SupplementPlan | None = None
            for attempt in range(2):
                try:
                    response = await self._gateway.generate_structured(
                        ModelRequest(
                            messages=[
                                ModelMessage(role=ModelRole.SYSTEM, content=SUPPLEMENT_PROMPT),
                                ModelMessage(
                                    role=ModelRole.USER,
                                    content=json.dumps(payload, ensure_ascii=False),
                                ),
                            ],
                            max_output_tokens=3_072,
                            structured_output_mode="json_schema",
                            output_schema_override=attraction_schema(SupplementPlan),
                            audit=ModelAuditMetadata(
                                stage="prepare_attraction_supplement_plan",
                                contract_version=VERSION,
                                attempt=attempt + 1,
                                repair=attempt > 0,
                            ),
                        ),
                        SupplementPlan,
                        cancellation=cancellation,
                    )
                    proposals = [
                        query
                        for query in response.value.queries
                        if query.identity not in previous_ids
                    ]
                    if any(
                        set(query.related_direction_ids) - known_direction_ids
                        for query in proposals
                    ):
                        raise ValueError("query references an unknown or excluded direction")
                    if n < minimum and not proposals:
                        raise ValueError(
                            "pool below minimum; provide new queries "
                            "instead of empty or repeated queries"
                        )
                    plan = response.value.model_copy(update={"queries": proposals})
                    break
                except ModelGatewayError as error:
                    if error.code is ModelFailureCode.CANCELLED or error.requires_runtime_recovery:
                        raise
                    payload["repair"] = {"code": error.code.value}
                except ValueError as error:
                    payload["repair"] = {"instruction": str(error)}
            if plan is None:
                return pool.model_copy(
                    update={"limitations": [*pool.limitations, "supplement_plan_unavailable"]}
                )
            if not plan.queries:
                return pool
            extra = await self._search_many(
                str(request.trip_id), request.city_id, plan.queries, "supplement", cancellation
            )
            observations = [*pool.observations, *extra]
            pool = pool.model_copy(
                update={
                    "observations": observations,
                    "candidates": candidates_for_request(request, observations),
                    "supplemental_rounds": pool.supplemental_rounds + 1,
                    "generated_at": datetime.now(UTC),
                }
            )
            await _audit(
                "attraction_pool_supplemented",
                {
                    "candidate_count": len(pool.candidates),
                    "minimum_pool_size": minimum,
                    "preferred_pool_size": preferred,
                    "round": pool.supplemental_rounds,
                    "coverage_gaps": [gap.model_dump(mode="json") for gap in plan.coverage_gaps],
                },
            )
            # Once the margin exists, another round solely for the soft target is optional.
            if len(pool.candidates) >= minimum:
                break
        if len(pool.candidates) < minimum:
            pool = pool.model_copy(
                update={"limitations": [*pool.limitations, "candidate_pool_below_1_5_margin"]}
            )
        return pool

    async def _search_many(
        self,
        trip_id: str,
        city_id: str,
        queries: list[AttractionQuery],
        branch: Literal["city", "personalized", "supplement"],
        cancellation: ModelCancellation | None,
    ) -> list[AttractionSearchObservation]:
        unique: dict[str, AttractionQuery] = {}
        for query in queries:
            previous = unique.get(query.identity)
            unique[query.identity] = (
                query
                if previous is None
                else previous.model_copy(
                    update={
                        "related_direction_ids": list(
                            dict.fromkeys(
                                [*previous.related_direction_ids, *query.related_direction_ids]
                            )
                        ),
                    }
                )
            )
        return list(
            await asyncio.gather(
                *(
                    self._search(trip_id, city_id, query, branch, cancellation)
                    for query in unique.values()
                )
            )
        )

    async def _search(
        self,
        trip_id: str,
        city_id: str,
        query: AttractionQuery,
        branch: Literal["city", "personalized", "supplement"],
        cancellation: ModelCancellation | None,
    ) -> AttractionSearchObservation:
        params: dict[str, object] = {
            "trip_id": trip_id,
            "city_id": city_id,
            "query": query.identity,
        }
        lock_key = json.dumps(params, sort_keys=True)
        lock = self._query_locks.setdefault(lock_key, asyncio.Lock())
        async with lock:
            if cancellation is not None:
                cancellation.raise_if_cancelled("attraction_search")
            cached = await self._get("query", params)
            if isinstance(cached, dict):
                previous = AttractionSearchObservation.model_validate(cached)
                channel = (
                    RecallChannel.CITY_LANDMARK
                    if branch == "city"
                    else RecallChannel.SELECTED_THEME
                )
                candidates = [
                    candidate.model_copy(
                        update={
                            "theme_ids": tuple(query.related_direction_ids),
                            "reasons": (query.reason,),
                            "channels": (channel,),
                            "representative_identity_verified": (
                                candidate.representative_identity_verified and branch == "city"
                            ),
                        }
                    )
                    for candidate in previous.candidates
                ]
                return previous.model_copy(
                    update={
                        "query": query,
                        "branch": branch,
                        "candidates": candidates,
                        "cache_hit": True,
                    }
                )
            async with self._semaphore:
                observation = await self._execute(city_id, query, branch, cancellation)
            if observation.attempt.status is not RecallAttemptStatus.FAILED:
                await self._put("query", params, observation.model_dump(mode="json"))
            await _audit("attraction_provider_query_completed", observation.model_dump(mode="json"))
            return observation

    async def _execute(
        self,
        city_id: str,
        query: AttractionQuery,
        branch: Literal["city", "personalized", "supplement"],
        cancellation: ModelCancellation | None,
    ) -> AttractionSearchObservation:
        if cancellation is not None:
            cancellation.raise_if_cancelled("attraction_provider_query")
        scope = self._registry.provider_scope(city_id, ProviderCode.AMAP)
        started_at, started = datetime.now(UTC), time.monotonic()
        query_id = str(uuid5(NAMESPACE_URL, f"{city_id}:{query.identity}"))
        channel = RecallChannel.CITY_LANDMARK if branch == "city" else RecallChannel.SELECTED_THEME
        candidates: list[RecalledCandidate] = []
        skipped: list[dict[str, str | int]] = []
        try:
            response = await self._places.search_places(
                KeywordPlaceSearchRequest(
                    city=scope,
                    query=query.keyword,
                    category_hint=PlaceCategory.ATTRACTION,
                    page_size=query.limit,
                    typecodes=list(ATTRACTION_SEARCH_TYPECODES),
                )
            )
            for rank, place in enumerate(response.items[: query.limit], 1):
                issue = (
                    "cross_city"
                    if place.city_id != city_id
                    else closed_evidence(place) or irrelevant_poi_type(place)
                )
                if issue is not None:
                    skipped.append(
                        {
                            "rank": rank,
                            "name": place.name,
                            "source_place_id": place.source_place_id,
                            "reason": issue,
                        }
                    )
                    continue
                matched = query.kind == "named_place" and matches_place_search_identity(
                    query.keyword,
                    place.name,
                    city_name=scope.display_name or city_id,
                    category=PlaceCategory.ATTRACTION,
                )
                place_id = uuid5(
                    NAMESPACE_URL, f"{response.provider.value}:{place.source_place_id}"
                )
                candidates.append(
                    RecalledCandidate(
                        candidate_id=place_id,
                        domain=CandidateDomain.ATTRACTION,
                        place=RecalledPlace(
                            place_id=place_id,
                            city_id=city_id,
                            category=PlaceCategory.ATTRACTION,
                            name=place.name,
                            address=place.address,
                            coordinates=place.coordinates,
                            image_url=place.image_url,
                            provider_typecode=place.provider_typecode,
                            provider_parent_place_id=place.provider_parent_place_id,
                            rating=place.rating,
                            average_cost=place.average_cost,
                        ),
                        channels=(channel,),
                        theme_ids=tuple(query.related_direction_ids),
                        reasons=(query.reason,),
                        representative_identity_verified=matched and branch == "city",
                        sources=(
                            CandidateSourceReference(
                                kind=RecallSourceKind.PROVIDER,
                                source_record_id=response.source_request_id or query_id,
                                provider=response.provider,
                                source_place_id=place.source_place_id,
                                fetched_at=place.fetched_at,
                            ),
                        ),
                        availability=DataAvailability.AVAILABLE,
                    )
                )
            returned = len(response.items[: query.limit])
            status = (
                RecallAttemptStatus.AVAILABLE
                if candidates
                else RecallAttemptStatus.PARTIAL
                if returned
                else RecallAttemptStatus.EMPTY
            )
            if skipped:
                status = RecallAttemptStatus.PARTIAL
            attempt = RecallAttempt(
                query_id=query_id,
                channel=channel,
                domain=CandidateDomain.ATTRACTION,
                query_keyword=query.keyword,
                provider=response.provider,
                requested_limit=query.limit,
                returned_count=returned,
                accepted_count=len(candidates),
                status=status,
                failure_reason="closed, cross-city or irrelevant POI results removed"
                if skipped
                else None,
            )
        except ProviderError as error:
            attempt = RecallAttempt(
                query_id=query_id,
                channel=channel,
                domain=CandidateDomain.ATTRACTION,
                query_keyword=query.keyword,
                provider=error.provider,
                requested_limit=query.limit,
                returned_count=0,
                accepted_count=0,
                status=RecallAttemptStatus.FAILED,
                failure_code=RecallFailureCode(error.code.value),
                failure_reason=error.code.value,
            )
        return AttractionSearchObservation(
            query=query,
            branch=branch,
            attempt=attempt,
            candidates=candidates,
            skipped=skipped,
            started_at=started_at,
            duration_seconds=time.monotonic() - started,
        )
