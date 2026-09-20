"""Non-blocking lodging prefetch; normalized provider cache is shared with Planner.

Tasks never write trip state. Exact request keys isolate city/date/preferences;
ProviderRuntime owns TTL and durable Redis cache. Restart or failure is requeryable.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Literal, cast

from backend.contracts.v4.lodging_preferences import HOTEL_STARS, LodgingExample
from backend.providers.contracts import (
    HotelSearchRequest,
    ProviderCityScope,
    ProviderError,
    ProviderHotelOffer,
    ProviderResponse,
)
from backend.providers.interfaces import TravelProductProvider

logger = logging.getLogger(__name__)
_tasks: dict[tuple[int, str], asyncio.Task[ProviderResponse[ProviderHotelOffer]]] = {}
_semaphore = asyncio.Semaphore(3)


def hotel_requests(
    *,
    city: ProviderCityScope,
    check_in: date,
    check_out: date,
    examples: Sequence[LodgingExample],
    tiers: Sequence[str],
    types: Sequence[str],
) -> list[HotelSearchRequest]:
    return [
        HotelSearchRequest(
            city=city,
            check_in=check_in,
            check_out=check_out,
            query=example.search_keyword,
            anchor_name=example.name,
            hotel_stars=[HOTEL_STARS[tier] for tier in dict.fromkeys(tiers) if tier in HOTEL_STARS],
            hotel_types=cast(
                list[Literal["酒店", "民宿", "客栈"]],
                [item for item in dict.fromkeys(types) if item in {"酒店", "民宿", "客栈"}],
            ),
            sort="rate_desc",
        )
        for example in {item.search_keyword: item for item in examples}.values()
    ]


def _start(
    provider: TravelProductProvider, request: HotelSearchRequest
) -> asyncio.Task[ProviderResponse[ProviderHotelOffer]]:
    key = (id(provider), hashlib.sha256(request.model_dump_json().encode()).hexdigest())
    existing = _tasks.get(key)
    if existing is not None:
        return existing

    async def run() -> ProviderResponse[ProviderHotelOffer]:
        async with _semaphore:
            return await provider.search_hotels(request)

    task = asyncio.create_task(run(), name="lodging-search")
    _tasks[key] = task

    def finish(done: asyncio.Task[ProviderResponse[ProviderHotelOffer]]) -> None:
        _tasks.pop(key, None)
        if not done.cancelled() and done.exception() is not None:
            logger.warning("lodging_prefetch_unavailable")

    task.add_done_callback(finish)
    return task


def start_prefetch(provider: TravelProductProvider, requests: Sequence[HotelSearchRequest]) -> None:
    for request in requests:
        _start(provider, request)


@dataclass(frozen=True)
class LodgingSearchResult:
    request: HotelSearchRequest
    response: ProviderResponse[ProviderHotelOffer] | None = None
    error: ProviderError | None = None
    internal_error: bool = False


async def search_lodging_results(
    provider: TravelProductProvider, requests: Sequence[HotelSearchRequest]
) -> list[LodgingSearchResult]:
    from backend.agent.model_gateway import ModelGatewayError
    from backend.providers.request_budget import RequestBudgetExceeded

    results = await asyncio.gather(
        *(asyncio.shield(_start(provider, request)) for request in requests), return_exceptions=True
    )
    collected = []
    for request, result in zip(requests, results, strict=True):
        if isinstance(
            result, (asyncio.CancelledError, ModelGatewayError, RequestBudgetExceeded, TimeoutError)
        ):
            raise result
        if isinstance(result, ProviderResponse):
            collected.append(LodgingSearchResult(request, response=result))
        elif isinstance(result, ProviderError):
            collected.append(LodgingSearchResult(request, error=result))
        elif isinstance(result, Exception):
            collected.append(LodgingSearchResult(request, internal_error=True))
        else:
            raise result
    return collected


async def search_lodging(
    provider: TravelProductProvider, requests: Sequence[HotelSearchRequest]
) -> list[ProviderResponse[ProviderHotelOffer]]:
    results = await search_lodging_results(provider, requests)
    return [
        result.response.model_copy(update={"items": top_hotels(result.response.items)})
        for result in results
        if result.response is not None
    ]


def top_hotels(items: Sequence[ProviderHotelOffer]) -> list[ProviderHotelOffer]:
    # rate_desc is requested from FlyAI; returned numeric scores refine ties/order.
    # Unknown ratings stay unknown and retain stable provider order after known scores.
    ranked = sorted(items, key=lambda item: (item.rating is None, -(item.rating or 0)))
    unique: dict[str, ProviderHotelOffer] = {}
    for item in ranked:
        unique.setdefault(item.source_hotel_id, item)
    return list(unique.values())[:3]
