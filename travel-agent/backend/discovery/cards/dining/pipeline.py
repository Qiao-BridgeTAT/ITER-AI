"""Real restaurant discovery: prefetch, liked hints, parallel analysis and final selection."""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import Any
from uuid import NAMESPACE_URL, uuid5
from weakref import WeakValueDictionary

from redis.exceptions import RedisError

from backend.agent.model_gateway import ModelCancellation, ModelGatewayError
from backend.agent.prepare.progress import DiningProgress, report_dining_progress
from backend.contracts.candidate_recall import (
    CandidateDomain,
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
from backend.discovery.cards.attraction_recall import closed_evidence
from backend.discovery.cards.dining.logic import (
    FIXED_KEYWORDS,
    TYPE_FIELDS,
    compile_pool,
    merge_roles,
    minimum_for_days,
    validate_brand_groups,
    validate_types,
)
from backend.discovery.cards.dining.model import VERSION, audit, run_task
from backend.discovery.cards.dining.validation import validate_final
from backend.providers.contracts import KeywordPlaceSearchRequest, ProviderError, ProviderPlace
from backend.providers.place_copy import provider_place_cuisine

CACHE_SECONDS = 1800
FINAL_COUNTS = {1: 3, 2: 4, 3: 8, 4: 10, 5: 12}


class DiningDiscovery:
    def __init__(self, *, gateway: Any, places: Any, registry: Any, store: Any = None) -> None:
        self.gateway, self.places, self.registry, self.store = gateway, places, registry, store
        self.tasks: dict[tuple[str, str, str], tuple[float, asyncio.Task[Any]]] = {}
        self.tokens: dict[tuple[str, str, str], ModelCancellation] = {}
        self.semaphore = asyncio.Semaphore(4)
        self.query_locks: WeakValueDictionary[tuple[str, str, str, int], asyncio.Lock] = (
            WeakValueDictionary()
        )

    async def close(self) -> None:
        tasks = list(self.tasks.values())
        for token in self.tokens.values():
            token.cancel()
        for _, task in tasks:
            task.cancel()
        await asyncio.gather(*(t for _, t in tasks), return_exceptions=True)
        self.tasks.clear()
        self.tokens.clear()

    def start(self, trip_id: Any, city_id: Any, kind: Any = "city") -> Any:
        self.registry.resolve(city_id)
        key = (trip_id, city_id, kind)
        now = time.monotonic()
        for old, (created, task) in list(self.tasks.items()):
            if (old[0] == trip_id and old[1] != city_id) or (
                task.done() and (now - created > CACHE_SECONDS or len(self.tasks) > 128)
            ):
                self.tokens.pop(old).cancel()
                task.cancel()
                self.tasks.pop(old)
        previous = self.tasks.get(key)
        if (
            previous
            and previous[1].done()
            and (previous[1].cancelled() or previous[1].exception() or not previous[1].result())
        ):
            self.tasks.pop(key)
        if key not in self.tasks:
            token = ModelCancellation()
            self.tokens[key] = token
            task = asyncio.create_task(self._prefetch(trip_id, city_id, kind, token))
            task.add_done_callback(lambda done: None if done.cancelled() else done.exception())
            self.tasks[key] = (now, task)
        return self.tasks[key][1]

    async def _get(self, kind: Any, params: Any) -> Any:
        if self.store:
            try:
                return await self.store.get_cache("dining-" + kind, VERSION, params)
            except RedisError:
                pass
        return None

    async def _put(self, kind: Any, params: Any, value: Any) -> None:
        if self.store:
            with contextlib.suppress(RedisError):
                await self.store.put_cache("dining-" + kind, VERSION, params, value, CACHE_SECONDS)

    async def _prefetch(self, trip_id: Any, city_id: Any, kind: Any, token: Any) -> Any:
        params = {"trip_id": trip_id, "city_id": city_id}
        cached = await self._get(kind, params)
        if isinstance(cached, list) and cached:
            return cached
        if kind == "city":
            try:
                seed = await run_task(
                    self.gateway,
                    "A",
                    {
                        "destination": {
                            "name": self.registry.resolve(city_id).display_name,
                            "city_id": city_id,
                        },
                        "evidence": [],
                    },
                    cancellation=token,
                )
            except (ModelGatewayError, ValueError):
                await audit("dining_prefetch_failed", {"kind": kind, "city_id": city_id})
                return []
            queries = [
                (
                    r["name"],
                    {
                        "source_kind": "local_specialty",
                        "query_kind": "named_restaurant",
                        "source_stage": "city_background",
                        "top_k": 3,
                    },
                )
                for r in seed["restaurants"]
            ]
        else:
            queries = [
                (
                    k,
                    {
                        "source_kind": "regular",
                        "query_kind": "keyword",
                        "source_stage": "fixed_reserve",
                        "top_k": 1,
                    },
                )
                for k in FIXED_KEYWORDS
            ]
        entries = await self.search_many(trip_id, city_id, queries, token)
        token.raise_if_cancelled("dining_prefetch")
        # Never cache upstream failures as an authoritative empty result.
        if entries and all(not e.get("failed") for e in entries):
            await self._put(kind, params, entries)
        await audit(
            "dining_prefetch_completed",
            {"kind": kind, "query_count": len(entries), "city_id": city_id},
        )
        return entries

    async def search_many(self, trip_id: Any, city_id: Any, queries: Any, token: Any) -> Any:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for query, origin in queries:
            query = " ".join(query.split())
            if query:
                grouped.setdefault(query, []).append(origin)
        return list(
            await asyncio.gather(
                *(
                    self._search(trip_id, city_id, q, origins, token)
                    for q, origins in grouped.items()
                )
            )
        )

    async def _search(
        self, trip_id: Any, city_id: Any, query: Any, origins: Any, token: Any
    ) -> Any:
        limit = max(o["top_k"] for o in origins)
        params = {"trip_id": trip_id, "city_id": city_id, "query": query, "limit": limit}
        key = (trip_id, city_id, query, limit)
        lock = self.query_locks.setdefault(key, asyncio.Lock())
        start = time.monotonic()
        async with lock:
            if token:
                token.raise_if_cancelled("dining_search")
            cached = await self._get("query", params)
            if isinstance(cached, list):
                return {"query": query, "origins": origins, "items": cached, "cache_hit": True}
            async with self.semaphore:
                try:
                    response = await self.places.search_places(
                        KeywordPlaceSearchRequest(
                            city=self.registry.provider_scope(city_id, ProviderCode.AMAP),
                            query=query,
                            category_hint=PlaceCategory.RESTAURANT,
                            typecodes=["050000"],
                            page_size=limit,
                        )
                    )
                    items = [p.model_dump(mode="json") for p in response.items[:limit]]
                except ProviderError as error:
                    await audit(
                        "dining_search_failed",
                        {"city_id": city_id, "failure_code": error.code.value},
                    )
                    return {
                        "query": query,
                        "origins": origins,
                        "items": [],
                        "failed": error.code.value,
                    }
            await self._put("query", params, items)
            await audit(
                "dining_search_completed",
                {
                    "query": query,
                    "top_k": limit,
                    "count": len(items),
                    "seconds": time.monotonic() - start,
                },
            )
            return {"query": query, "origins": origins, "items": items, "cache_hit": False}

    async def select(self, state: Any, *, cancellation: Any = None) -> Any:
        await report_dining_progress(DiningProgress.SPECIFIC_SEARCH)
        trip_id, city_id = state.trip_id, state.trip_basics.destination_canonical_id
        days = state.trip_basics.duration_days
        city = self.start(trip_id, city_id)
        reserve = self.start(trip_id, city_id, "fixed")
        queries: list[tuple[str, dict[str, Any]]] = []
        for direction in state.dining.preference_directions:
            if not direction.selected:
                continue
            hints = direction.dining_search_hints
            kind = hints.kind if hints else "regular"
            origin = {
                "source_kind": kind,
                "source_stage": "liked_direction",
                "direction_id": direction.direction_id,
                "top_k": 3 if kind == "local_specialty" else 2,
            }
            if hints:
                queries.extend(
                    (name, {**origin, "query_kind": "named_restaurant"})
                    for name in hints.representative_restaurants
                )
                words = hints.search_keywords
            else:
                words = [direction.search_query or direction.label]
            queries.extend((word, {**origin, "query_kind": "keyword"}) for word in words)
        # Explicitly named restaurants remain user intent, independent of a direction toggle.
        queries.extend(
            (
                intent.display_name,
                {
                    "source_kind": "regular",
                    "source_stage": "explicit_intent",
                    "query_kind": "named_restaurant",
                    "top_k": 3,
                },
            )
            for intent in state.dining.concrete_restaurant_intents
            if intent.disposition != "avoid"
        )
        foreground = asyncio.create_task(self.search_many(trip_id, city_id, queries, cancellation))
        try:
            initial, background, fixed = await asyncio.gather(
                foreground, asyncio.shield(city), asyncio.shield(reserve)
            )
        except BaseException:
            foreground.cancel()
            await asyncio.gather(foreground, return_exceptions=True)
            raise
        if cancellation:
            cancellation.raise_if_cancelled("dining_pool")
        entries = initial + background + fixed
        rows, recalled, attempts = collect(entries, state)
        if not rows:
            raise ValueError("dining_provider_pool_empty")
        brands_user = {
            "candidates": [
                {
                    k: row.get(k)
                    for k in (
                        "candidate_key",
                        "name",
                        "address",
                        "provider_alias",
                        "explicit_brand_evidence",
                    )
                }
                for row in rows
            ]
        }
        types_user = {
            "candidates": [{k: row.get(k) for k in ("candidate_key", *TYPE_FIELDS)} for row in rows]
        }
        keys = [r["candidate_key"] for r in rows]
        await report_dining_progress(DiningProgress.SPECIFIC_CLASSIFY)
        brands, types = await asyncio.gather(
            run_task(
                self.gateway,
                "F",
                brands_user,
                cancellation=cancellation,
                candidate_keys=keys,
                validator=lambda result: validate_brand_groups(result, rows),
            ),
            run_task(
                self.gateway,
                "G",
                types_user,
                cancellation=cancellation,
                candidate_keys=keys,
                validator=lambda result: validate_types(result, rows),
            ),
        )
        groups = merge_roles(rows, brands, types)
        base_keys = {
            r["candidate_key"]
            for r in rows
            if any(o["source_stage"] != "fixed_reserve" for o in r["origins"])
        }
        reserve_keys = {
            r["candidate_key"]
            for r in rows
            if any(o["source_stage"] == "fixed_reserve" for o in r["origins"])
        }
        pool, metadata = compile_pool(
            groups, base_keys, reserve_keys, minimum_for_days(days), f"{trip_id}:{city_id}"
        )
        if not pool:
            raise ValueError("dining_eligible_pool_empty")
        final_count = FINAL_COUNTS[days]
        preferences = {
            "liked_directions": [
                d.model_dump(mode="json") for d in state.dining.preference_directions if d.selected
            ],
            "excluded_directions": [
                d.model_dump(mode="json")
                for d in state.dining.preference_directions
                if not d.selected
            ],
            "named_restaurant_intents": [
                i.model_dump(mode="json") for i in state.dining.concrete_restaurant_intents
            ],
            "hard_constraints": {
                "allergies": state.dining.allergies,
                "avoidances": state.dining.avoidances,
                "requirements": state.dining.requirements,
            },
        }
        user = {
            "destination": {"name": state.trip_basics.destination_name, "city_id": city_id},
            "user_preferences": preferences,
            "shortlist": pool,
            "selection_policy": {"final_card_count": final_count, "max_per_type": 3},
        }
        await report_dining_progress(DiningProgress.SPECIFIC_READY)
        output = await run_task(
            self.gateway,
            "D",
            user,
            cancellation=cancellation,
            candidate_keys=[m["candidate_key"] for g in pool for m in g["members"]],
            anchors=[g["anchor_candidate_key"] for g in pool],
            validator=lambda result: validate_final(result, pool, final_count),
        )
        selected = [recalled[r["candidate_key"]] for r in output["selected"]]
        await audit(
            "dining_selection_completed",
            {
                **metadata,
                "provider_candidate_count": len(rows),
                "selected_count": len(selected),
                "final_card_count": final_count,
                "shortfall": output["shortfall"],
                "candidates": rows,
                "brand_groups": brands,
                "primary_types": types,
                "selection": output,
            },
        )
        return selected, rows, attempts, entries


def collect(entries: Any, state: Any) -> Any:
    from backend.contracts.v4.enums import CardDomain
    from backend.discovery.cards.candidate_composition import _hard_filtered

    rows: dict[str, dict[str, Any]] = {}
    recalled: dict[str, RecalledCandidate] = {}
    attempts: list[RecallAttempt] = []
    for entry in entries:
        accepted = 0
        for rank, item in enumerate(entry["items"], 1):
            place = ProviderPlace.model_validate(item)
            origins = [
                {**o, "query_text": entry["query"], "provider_rank": rank}
                for o in entry["origins"]
                if rank <= o["top_k"]
            ]
            raw_adcode = str(place.raw_payload.get("adcode") or "")
            expected_code = (state.trip_basics.destination_canonical_id or "").removeprefix("cn-")
            outside = bool(
                raw_adcode and expected_code.isdigit() and raw_adcode[:4] != expected_code[:4]
            )
            if (
                outside
                or not origins
                or place.city_id != state.trip_basics.destination_canonical_id
                or place.category is not PlaceCategory.RESTAURANT
                or bool(
                    place.provider_typecode
                    and not any(
                        code.startswith("05")
                        for code in place.provider_typecode.replace("|", ";").split(";")
                    )
                )
                or closed_evidence(place)
                or _hard_filtered(state, CardDomain.DINING, place.name)
            ):
                continue
            accepted += 1
            place_id = uuid5(NAMESPACE_URL, f"{place.provider.value}:{place.source_place_id}")
            key = str(place_id)
            raw = place.raw_payload
            business = raw.get("business") or raw.get("biz_ext") or {}
            if key not in rows:
                rows[key] = {
                    "candidate_key": key,
                    "name": place.name,
                    "address": place.address,
                    "provider_category": raw.get("type"),
                    "provider_food_tags": raw.get("tag") or business.get("tag"),
                    "provider_alias": raw.get("alias"),
                    "explicit_brand_evidence": [],
                    "known_conflicts": [],
                    "coordinates": place.coordinates.model_dump(mode="json"),
                    "origins": [],
                }
                source = CandidateSourceReference(
                    kind=RecallSourceKind.PROVIDER,
                    source_record_id=f"dining:{place_id}",
                    provider=place.provider,
                    source_place_id=place.source_place_id,
                    fetched_at=place.fetched_at,
                )
                recalled[key] = RecalledCandidate(
                    candidate_id=place_id,
                    domain=CandidateDomain.RESTAURANT,
                    place=RecalledPlace(
                        place_id=place_id,
                        city_id=place.city_id,
                        category=PlaceCategory.RESTAURANT,
                        name=place.name,
                        address=place.address,
                        coordinates=place.coordinates,
                        provider_typecode=place.provider_typecode,
                        provider_parent_place_id=place.provider_parent_place_id,
                        image_url=place.image_url,
                        cuisine=provider_place_cuisine(place),
                        rating=place.rating,
                        average_cost=place.average_cost,
                    ),
                    channels=(RecallChannel.SELECTED_THEME,),
                    reasons=("来自实际餐饮检索",),
                    sources=(source,),
                    availability=DataAvailability.AVAILABLE,
                )
            rows[key]["origins"].extend(o for o in origins if o not in rows[key]["origins"])
        attempts.append(
            RecallAttempt(
                query_id=str(uuid5(NAMESPACE_URL, entry["query"])),
                channel=RecallChannel.SELECTED_THEME,
                domain=CandidateDomain.RESTAURANT,
                query_keyword=entry["query"],
                provider=ProviderCode.AMAP,
                requested_limit=max(o["top_k"] for o in entry["origins"]),
                returned_count=len(entry["items"]),
                accepted_count=accepted,
                status=RecallAttemptStatus.FAILED
                if entry.get("failed")
                else RecallAttemptStatus.PARTIAL
                if accepted < len(entry["items"])
                else RecallAttemptStatus.AVAILABLE
                if accepted
                else RecallAttemptStatus.EMPTY,
                failure_code=RecallFailureCode(entry["failed"]) if entry.get("failed") else None,
                failure_reason=entry.get("failed")
                or (
                    "non-dining, closed, cross-city or excluded POIs removed"
                    if accepted < len(entry["items"])
                    else None
                ),
            )
        )
    return list(rows.values()), recalled, tuple(attempts)
