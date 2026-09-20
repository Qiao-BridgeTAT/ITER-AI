"""Bounded ticket-price recheck, retaining exact place/date/admission provenance."""

from __future__ import annotations

import asyncio
import re
from datetime import date, datetime

from backend.agent.planner.workspace import server_id
from backend.contracts.common import CnyAmountRange
from backend.contracts.v4.planner_evidence import PlannerTicketEvidence
from backend.providers.contracts import (
    ProductSearchRequest,
    ProviderCityScope,
    ProviderError,
    ProviderTicketOffer,
)
from backend.providers.interfaces import TravelProductProvider


def _place_name(value: str) -> str:
    return re.sub(r"[\s·•（）()]", "", value).removesuffix("门票")


def ticket_fact(
    offers: list[ProviderTicketOffer],
    *,
    canonical_id: str,
    name: str,
    day: date,
    observed_at: datetime,
    rechecked: bool = False,
) -> PlannerTicketEvidence:
    exact = [
        offer
        for offer in offers
        if offer.place_name and _place_name(offer.place_name) == _place_name(name)
    ]
    states = {offer.admission_status for offer in exact} - {"unknown"}
    admission = next(iter(states)) if len(states) == 1 else "unknown"
    admission_sources = tuple(
        dict.fromkeys(
            f"{offer.source_place_id}:admission"
            for offer in exact
            if admission != "unknown" and offer.admission_status == admission
        )
    )
    priced = [
        offer
        for offer in exact
        if offer.price is not None
        and (offer.price.maximum_fen > 0 or offer.admission_status == "free")
        and offer.price_date in {None, day}
        and not re.search(r"儿童|学生|老人|优待|讲解|导览|接送|酒店|套票|联票", offer.name)
    ]
    prices = [offer.price for offer in priced if offer.price is not None]
    if admission == "free":
        prices = [CnyAmountRange(minimum_fen=0, maximum_fen=0)]
    reason = (
        "来源标记基础参观免费；不含收费特展或讲解，如需预约请提前办理。"
        if admission == "free"
        else "来源标记非免费，但未返回可用门票价格。"
        if admission == "paid" and not prices
        else "参考票价仅用于费用估算；如需预约请提前办理，本次不核验预约状态或余量。"
    )
    if rechecked:
        reason += (
            "已按成人基础门票复查一次。" if prices else "已按成人基础门票复查一次，仍无可靠报价。"
        )
    return PlannerTicketEvidence(
        canonical_entity_id=canonical_id,
        service_date=day,
        product_count=len({offer.source_offer_id for offer in offers}),
        reference_price=CnyAmountRange(
            minimum_fen=min(price.minimum_fen for price in prices),
            maximum_fen=max(price.maximum_fen for price in prices),
        )
        if prices
        else None,
        source_offer_ids=admission_sources
        if admission == "free"
        else tuple(dict.fromkeys(offer.source_offer_id for offer in priced)),
        admission_status=admission,
        admission_source_ids=admission_sources,
        reason_summary=reason,
        observed_at=observed_at,
        fact_reference_id=server_id(
            "ticket",
            canonical_id,
            day,
            observed_at.isoformat(),
            "rechecked" if rechecked else "initial",
        ),
    )


async def lookup_ticket_price(
    provider: TravelProductProvider,
    *,
    city: ProviderCityScope,
    canonical_id: str,
    name: str,
    day: date,
) -> PlannerTicketEvidence:
    return (
        await lookup_ticket_prices(
            provider, city=city, canonical_id=canonical_id, name=name, days=(day,)
        )
    )[0]


async def lookup_ticket_prices(
    provider: TravelProductProvider,
    *,
    city: ProviderCityScope,
    canonical_id: str,
    name: str,
    days: tuple[date, ...],
) -> tuple[PlannerTicketEvidence, ...]:
    """FlyAI searches a product keyword, not dated inventory. Share that observation.

    Keep each product's actual price_date when projecting to travel dates; an
    undated reference price never proves availability for any of those dates.
    """
    if not days:
        return ()
    response = await provider.search_place_products(ProductSearchRequest(city=city, query=name))
    offers = list(response.items)
    initial = tuple(
        ticket_fact(
            offers, canonical_id=canonical_id, name=name, day=day, observed_at=response.fetched_at
        )
        for day in days
    )
    if all(fact.reference_price is not None for fact in initial):
        return initial
    # A different, explicit product query, not repeated model calls or an
    # indefinite retry of the same failed capability. Missing facts stay unknown.
    observed_at = response.fetched_at
    try:
        async with asyncio.timeout(12):
            recheck = await provider.search_place_products(
                ProductSearchRequest(city=city, query=f"{name} 成人门票")
            )
        offers.extend(recheck.items)
        observed_at = max(observed_at, recheck.fetched_at)
    except (ProviderError, TimeoutError):
        # Keep the original paid/free evidence even if the optional recheck fails.
        pass
    return tuple(
        ticket_fact(
            offers,
            canonical_id=canonical_id,
            name=name,
            day=day,
            observed_at=observed_at,
            rechecked=True,
        )
        for day in days
    )
