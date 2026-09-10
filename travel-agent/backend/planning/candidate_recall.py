"""Bounded candidate recall from reviewed city content and PlaceProvider."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from datetime import UTC, datetime
from uuid import NAMESPACE_URL, UUID, uuid5

from backend.agent.model_gateway import ModelCancellation
from backend.contracts.candidate_recall import (
    CandidateDomain,
    CandidateRecallRequest,
    CandidateRecallResult,
    CandidateSourceReference,
    ContentSeedRecallQuery,
    ExcludedPlaceClue,
    ProviderRecallQuery,
    RecallAttempt,
    RecallAttemptStatus,
    RecallChannel,
    RecalledCandidate,
    RecalledPlace,
    RecallFailureCode,
    RecallSourceKind,
)
from backend.contracts.city_content import CoreAttractionContent, RegisteredCityContentPackage
from backend.contracts.enums import DataAvailability, PlaceCategory, ProviderCode
from backend.planning.city_registry import CityRegistry
from backend.planning.recall_plan import RecallPlanGenerator, validate_recall_plan
from backend.providers.contracts import (
    KeywordPlaceSearchRequest,
    NearbyPlaceSearchRequest,
    ProviderError,
    ProviderFailureCode,
    ProviderPlace,
    ProviderResponse,
    ProviderResultStatus,
)
from backend.providers.interfaces import PlaceProvider
from backend.providers.place_copy import provider_place_cuisine, provider_place_intro
from backend.providers.place_matching import matches_place_search_identity

RECALL_RESULT_VERSION = "1.0.0"

_NOTICE_CREDENTIAL_PATTERN = re.compile(
    r"(?i)\b(?:api[_-]?key|apikey|access[_-]?token|token|secret|authorization|ak)"
    r"\s*[:=]\s*[^\s,;，；]+"
)
_NOTICE_BEARER_PATTERN = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}")
_NOTICE_PHONE_PATTERN = re.compile(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)")
_NOTICE_EMAIL_PATTERN = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
_NOTICE_URL_PATTERN = re.compile(r"https?://\S+", re.IGNORECASE)


class CandidateRecallService:
    def __init__(
        self,
        *,
        registry: CityRegistry,
        places: PlaceProvider,
        plan_generator: RecallPlanGenerator,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._registry = registry
        self._places = places
        self._plan_generator = plan_generator
        self._clock = clock

    async def recall(
        self,
        request: CandidateRecallRequest,
        *,
        cancellation: ModelCancellation | None = None,
    ) -> CandidateRecallResult:
        city = self._registry.resolve(request.city_id)
        if city.city_id != request.city_id:
            raise ValueError("candidate recall requires the canonical registered city_id")
        content = self._registry.load_content_package(city.city_id)
        plan = await self._plan_generator.create_plan(
            request,
            city,
            content,
            cancellation=cancellation,
        )
        plan = validate_recall_plan(request, plan, content)
        excluded = tuple(request.excluded_places)

        candidates: list[RecalledCandidate] = []
        attempts: list[RecallAttempt] = []
        for query in plan.content_queries:
            content_candidates, attempt = _recall_content(
                request,
                query,
                content,
                excluded,
            )
            candidates.extend(content_candidates)
            attempts.append(attempt)

        provider_results = await asyncio.gather(
            *(
                self._execute_provider_query(request, query, cancellation=cancellation)
                for query in plan.provider_queries
            )
        )
        for provider_candidates, attempt in provider_results:
            candidates.extend(
                candidate
                for candidate in provider_candidates
                if not _is_excluded(candidate, excluded)
            )
            accepted_count = sum(
                1 for candidate in provider_candidates if not _is_excluded(candidate, excluded)
            )
            if accepted_count != attempt.accepted_count:
                attempt = attempt.model_copy(
                    update={
                        "accepted_count": accepted_count,
                        "status": (
                            RecallAttemptStatus.PARTIAL
                            if attempt.returned_count > 0
                            else attempt.status
                        ),
                        "failure_reason": (
                            "explicitly excluded results were removed"
                            if accepted_count < attempt.returned_count
                            else attempt.failure_reason
                        ),
                    }
                )
                attempt = RecallAttempt.model_validate(attempt.model_dump(mode="json"))
            attempts.append(attempt)

        candidates = _apply_candidate_budgets(request, candidates)
        degradation = _degradation_reasons(content, attempts, candidates)
        status = _overall_status(candidates, attempts, degradation)
        generated_at = self._clock()
        if generated_at.tzinfo is None or generated_at.utcoffset() is None:
            raise ValueError("candidate recall clock must return an aware datetime")
        return CandidateRecallResult(
            result_version=RECALL_RESULT_VERSION,
            request_id=request.request_id,
            trip_id=request.trip_id,
            semantic_state_version=request.semantic_state_version,
            task_book_id=request.task_book_id,
            task_book_revision=request.task_book_revision,
            city_id=request.city_id,
            status=status,
            candidates=tuple(candidates),
            attempts=tuple(attempts),
            provider_call_count=len(plan.provider_queries),
            content_package_version=content.content_version if content else None,
            degradation_reasons=tuple(degradation),
            generated_at=generated_at.astimezone(UTC),
        )

    async def _execute_provider_query(
        self,
        request: CandidateRecallRequest,
        query: ProviderRecallQuery,
        *,
        cancellation: ModelCancellation | None,
    ) -> tuple[list[RecalledCandidate], RecallAttempt]:
        if cancellation is not None:
            cancellation.raise_if_cancelled("candidate_recall_provider")
        scope = self._registry.provider_scope(request.city_id, ProviderCode.AMAP)
        try:
            if query.channel is RecallChannel.NEARBY_ANCHOR:
                anchor = next(item for item in request.anchors if item.anchor_id == query.anchor_id)
                response = await self._places.search_nearby(
                    NearbyPlaceSearchRequest(
                        city=scope,
                        center=anchor.coordinates,
                        radius_m=query.radius_m or 5_000,
                        query=query.keyword,
                        category_hint=_category_for(query.domain),
                        typecodes=list(query.typecodes),
                        page_size=query.max_results,
                    )
                )
            else:
                response = await self._places.search_places(
                    KeywordPlaceSearchRequest(
                        city=scope,
                        query=query.keyword,
                        category_hint=_category_for(query.domain),
                        typecodes=list(query.typecodes),
                        page_size=query.max_results,
                    )
                )
        except ProviderError as error:
            return [], RecallAttempt(
                query_id=query.query_id,
                channel=query.channel,
                domain=query.domain,
                status=RecallAttemptStatus.FAILED,
                query_keyword=query.keyword,
                exact_name_match=query.exact_name_match,
                provider=error.provider,
                requested_limit=query.max_results,
                returned_count=0,
                accepted_count=0,
                failure_code=_failure_code(error.code),
                failure_reason=(
                    f"{error.provider.value} candidate recall failed: {error.code.value}"
                ),
            )
        if cancellation is not None:
            cancellation.raise_if_cancelled("candidate_recall_provider")
        return _provider_candidates(
            request,
            query,
            response,
            city_name=self._registry.resolve(request.city_id).display_name,
        )


def _recall_content(
    request: CandidateRecallRequest,
    query: ContentSeedRecallQuery,
    content: RegisteredCityContentPackage | None,
    excluded: tuple[ExcludedPlaceClue, ...],
) -> tuple[list[RecalledCandidate], RecallAttempt]:
    if content is None:
        raise ValueError("validated recall plan cannot query missing city content")
    source_by_id = {source.source_id: source for source in content.sources}
    matches = [
        item
        for item in content.core_attractions
        if (not query.theme_ids or set(item.theme_ids) & set(query.theme_ids))
        and not _content_item_is_excluded(item, excluded)
    ][: query.max_results]
    candidates = [
        RecalledCandidate(
            candidate_id=_candidate_id(query.query_id, "content", str(item.place_id)),
            domain=CandidateDomain.ATTRACTION,
            place=RecalledPlace(
                place_id=item.place_id,
                city_id=request.city_id,
                category=PlaceCategory.ATTRACTION,
                name=item.name,
            ),
            channels=(query.channel,),
            theme_ids=tuple(
                theme_id
                for theme_id in item.theme_ids
                if not query.theme_ids or theme_id in query.theme_ids
            ),
            reasons=(query.reason, item.city_importance),
            sources=tuple(
                CandidateSourceReference(
                    kind=RecallSourceKind.CITY_CONTENT,
                    source_record_id=source_id,
                    source_url=source_by_id[source_id].url,
                    fetched_at=source_by_id[source_id].retrieved_at,
                    content_version=content.content_version,
                )
                for source_id in item.source_ids
            ),
            availability=DataAvailability.PARTIAL,
            missing_fields=("address", "coordinates", "provider_place_id"),
            missing_reason="reviewed content seed still requires provider place resolution",
        )
        for item in matches
    ]
    status = RecallAttemptStatus.AVAILABLE if candidates else RecallAttemptStatus.EMPTY
    return candidates, RecallAttempt(
        query_id=query.query_id,
        channel=query.channel,
        domain=CandidateDomain.ATTRACTION,
        status=status,
        requested_limit=query.max_results,
        returned_count=len(candidates),
        accepted_count=len(candidates),
    )


def _provider_candidates(
    request: CandidateRecallRequest,
    query: ProviderRecallQuery,
    response: ProviderResponse[ProviderPlace],
    *,
    city_name: str = "",
) -> tuple[list[RecalledCandidate], RecallAttempt]:
    candidates: list[RecalledCandidate] = []
    cross_city_count = 0
    wrong_domain_count = 0
    wrong_identity_count = 0
    named_evidence = (query.named_clue_id,) if query.named_clue_id is not None else ()
    named_clue = next(
        (item for item in request.named_places if item.clue_id == query.named_clue_id),
        None,
    )
    for place in response.items[: query.max_results]:
        if place.city_id != request.city_id:
            cross_city_count += 1
            continue
        from backend.providers.place_taxonomy import is_shopping_complex

        shopping_experience = (
            request.attraction_discovery is not None
            and query.domain is CandidateDomain.ATTRACTION
            and is_shopping_complex(place.provider_typecode)
        )
        if place.category is not _category_for(query.domain) and not shopping_experience:
            wrong_domain_count += 1
            continue
        if query.exact_name_match and not matches_place_search_identity(
            query.keyword, place.name, city_name=city_name, category=place.category
        ):
            wrong_identity_count += 1
            continue
        missing_fields = tuple(
            field
            for field, missing in (
                ("address", place.address is None),
                ("provider_city_code", place.provider_city_code is None),
                ("provider_typecode", place.provider_typecode is None),
            )
            if missing
        )
        availability = DataAvailability.PARTIAL if missing_fields else DataAvailability.AVAILABLE
        candidates.append(
            RecalledCandidate(
                candidate_id=_candidate_id(
                    query.query_id,
                    response.provider.value,
                    place.source_place_id,
                ),
                domain=query.domain,
                place=RecalledPlace(
                    place_id=(
                        named_clue.known_place_id
                        if named_clue is not None and named_clue.known_place_id is not None
                        else uuid5(
                            NAMESPACE_URL,
                            f"{response.provider.value}:{place.source_place_id}",
                        )
                    ),
                    city_id=place.city_id,
                    # Business projection only; preserve original provider
                    # taxonomy for the semantic selector and all other callers.
                    category=(PlaceCategory.ATTRACTION if shopping_experience else place.category),
                    name=place.name,
                    address=place.address,
                    short_description=(
                        provider_place_intro(place)
                        if query.domain is CandidateDomain.RESTAURANT
                        else None
                    ),
                    cuisine=provider_place_cuisine(place),
                    coordinates=place.coordinates,
                    image_url=place.image_url,
                    provider_typecode=place.provider_typecode,
                    provider_parent_place_id=place.provider_parent_place_id,
                    rating=place.rating,
                    average_cost=place.average_cost,
                ),
                channels=(query.channel,),
                representative_identity_verified=(
                    query.exact_name_match and query.channel is RecallChannel.CITY_LANDMARK
                ),
                theme_ids=query.theme_ids,
                reasons=(query.reason,),
                named_evidence_ids=named_evidence,
                sources=(
                    CandidateSourceReference(
                        kind=RecallSourceKind.PROVIDER,
                        source_record_id=response.source_request_id
                        or f"{response.provider.value}:{query.query_id}",
                        provider=response.provider,
                        source_place_id=place.source_place_id,
                        fetched_at=place.fetched_at,
                    ),
                ),
                availability=availability,
                missing_fields=missing_fields,
                missing_reason=(
                    "provider returned incomplete optional place fields" if missing_fields else None
                ),
            )
        )
    rejected = cross_city_count + wrong_domain_count + wrong_identity_count
    accepted = len(candidates)
    if response.status is ProviderResultStatus.EMPTY:
        status = RecallAttemptStatus.EMPTY
    elif response.status is ProviderResultStatus.PARTIAL or rejected:
        status = RecallAttemptStatus.PARTIAL
    else:
        status = RecallAttemptStatus.AVAILABLE
    reason_parts = []
    if cross_city_count:
        reason_parts.append(f"rejected {cross_city_count} cross-city result(s)")
    if wrong_domain_count:
        reason_parts.append(f"rejected {wrong_domain_count} wrong-domain result(s)")
    if wrong_identity_count:
        reason_parts.append(f"rejected {wrong_identity_count} mismatched named-target result(s)")
    return candidates, RecallAttempt(
        query_id=query.query_id,
        channel=query.channel,
        domain=query.domain,
        status=status,
        provider=response.provider,
        query_keyword=query.keyword,
        exact_name_match=query.exact_name_match,
        requested_limit=query.max_results,
        returned_count=len(response.items[: query.max_results]),
        accepted_count=accepted,
        missing_fields=tuple(response.missing_fields),
        provider_notice=_safe_provider_notice(response.provider_notice),
        failure_reason="; ".join(reason_parts) or None,
    )


def _apply_candidate_budgets(
    request: CandidateRecallRequest,
    candidates: list[RecalledCandidate],
) -> list[RecalledCandidate]:
    limits = {item.domain: item.max_candidates for item in request.budget.domains}
    counts = {domain: 0 for domain in limits}
    selected: list[RecalledCandidate] = []
    for candidate in candidates:
        if len(selected) >= request.budget.max_total_candidates:
            break
        if counts.get(candidate.domain, 0) >= limits.get(candidate.domain, 0):
            continue
        selected.append(candidate)
        counts[candidate.domain] += 1
    return selected


def _degradation_reasons(
    content: RegisteredCityContentPackage | None,
    attempts: list[RecallAttempt],
    candidates: list[RecalledCandidate],
) -> list[str]:
    reasons: list[str] = []
    if content is None:
        reasons.append("reviewed city content package is unavailable; provider-only recall used")
    if any(attempt.status is RecallAttemptStatus.FAILED for attempt in attempts):
        reasons.append("one or more provider recall channels failed")
    if any(attempt.status is RecallAttemptStatus.PARTIAL for attempt in attempts):
        reasons.append("one or more recall channels returned partial results")
    if not candidates:
        reasons.append("no verified candidates were returned")
    return list(dict.fromkeys(reasons))


def _overall_status(
    candidates: list[RecalledCandidate],
    attempts: list[RecallAttempt],
    degradation: list[str],
) -> DataAvailability:
    if not candidates:
        return DataAvailability.MISSING
    if degradation or any(
        candidate.availability is DataAvailability.PARTIAL for candidate in candidates
    ):
        return DataAvailability.PARTIAL
    if any(attempt.status is not RecallAttemptStatus.AVAILABLE for attempt in attempts):
        return DataAvailability.PARTIAL
    return DataAvailability.AVAILABLE


def _is_excluded(
    candidate: RecalledCandidate,
    exclusions: tuple[ExcludedPlaceClue, ...],
) -> bool:
    return any(
        (clue.domain is None or clue.domain is candidate.domain)
        and (
            clue.known_place_id == candidate.place.place_id
            if clue.known_place_id is not None
            else _normalize(clue.name) == _normalize(candidate.place.name)
        )
        for clue in exclusions
    )


def _content_item_is_excluded(
    item: CoreAttractionContent,
    exclusions: tuple[ExcludedPlaceClue, ...],
) -> bool:
    return any(
        clue.domain in {None, CandidateDomain.ATTRACTION}
        and (
            clue.known_place_id == item.place_id
            if clue.known_place_id is not None
            else _normalize(clue.name) == _normalize(item.name)
        )
        for clue in exclusions
    )


def _candidate_id(query_id: str, source: str, source_place_id: str) -> UUID:
    return uuid5(NAMESPACE_URL, f"candidate-recall:{query_id}:{source}:{source_place_id}")


def _category_for(domain: CandidateDomain) -> PlaceCategory:
    return {
        CandidateDomain.ATTRACTION: PlaceCategory.ATTRACTION,
        CandidateDomain.RESTAURANT: PlaceCategory.RESTAURANT,
        CandidateDomain.HOTEL: PlaceCategory.HOTEL,
    }[domain]


def _failure_code(code: ProviderFailureCode) -> RecallFailureCode:
    return RecallFailureCode(code.value)


def _safe_provider_notice(notice: str | None) -> str | None:
    """Keep a short user-safe Provider explanation without leaking upstream data."""

    if notice is None:
        return None
    safe = " ".join(notice.split())
    for pattern in (
        _NOTICE_CREDENTIAL_PATTERN,
        _NOTICE_BEARER_PATTERN,
        _NOTICE_PHONE_PATTERN,
        _NOTICE_EMAIL_PATTERN,
        _NOTICE_URL_PATTERN,
    ):
        safe = pattern.sub("[redacted]", safe)
    safe = safe[:500].rstrip()
    return safe or None


def _normalize(value: str) -> str:
    return "".join(value.casefold().split())
