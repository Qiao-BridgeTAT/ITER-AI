"""Real route and hotel evidence; never selects the Planner's itinerary or hotel."""

from __future__ import annotations

import asyncio
import re
import unicodedata
from datetime import datetime, timedelta
from math import ceil, cos, radians, sqrt
from typing import Literal, cast
from uuid import NAMESPACE_URL, uuid4, uuid5

from backend.agent.model_gateway import ModelCancellation
from backend.agent.planner.observation_router import EvidenceUpdate
from backend.agent.planner.route_comparison import compare_observed_routes
from backend.agent.planner.route_diagnostics import missing_route_reason, route_query_failure
from backend.agent.planner.workspace import (
    PlannerGuardError,
    server_id,
    service_dates,
    task_book_references,
)
from backend.contracts.enums import PlaceCategory, ProviderCode
from backend.contracts.places import Gcj02Coordinates
from backend.contracts.v4.lodging_preferences import HOTEL_STARS
from backend.contracts.v4.planner_evidence import (
    PlannerCapabilityObservation,
    PlannerHotelLocationEvidence,
)
from backend.contracts.v4.planner_observations import (
    AppliedHotelConstraints,
    FixedBookingObservation,
    HotelClusterCommute,
    HotelObservation,
    HotelOfferObservation,
    HotelOfferRefreshArguments,
    HotelQueryAttempt,
    HotelSearchArguments,
    HotelStaySegment,
    MoneyAmountRange,
    PlannerCapabilityRequest,
    SpatialRouteEdge,
    SpatialRouteEndpoint,
    SpatialRoutePair,
    SpatialRoutesArguments,
)
from backend.contracts.v4.planner_refs import HotelOfferRef
from backend.contracts.v4.planner_workspace import PlannerWorkspaceState, VerifiedFactSummary
from backend.contracts.v4.task_book import TaskBookV4
from backend.persistence.outbox_repository import canonical_json_hash
from backend.planning.city_registry import CityProviderUnavailableError, CityRegistry
from backend.planning.runtime_backend import PlanningProviderSet
from backend.providers.contracts import (
    HotelSearchRequest,
    KeywordPlaceSearchRequest,
    ProviderError,
    ProviderHotelOffer,
    ProviderPlace,
    ProviderResponse,
    ProviderRoute,
    RouteMode,
    RouteRequest,
)
from backend.providers.place_matching import coordinates_to_gcj02
from backend.providers.place_taxonomy import category_from_original_typecodes
from backend.providers.request_budget import RequestBudgetExceeded

_HOTEL_QUALITY_QUERY_LABELS = {
    "economy": "经济实用",
    "comfort": "舒适中档",
    "upscale": "高档品质",
    "luxury": "豪华享受",
}
MAX_PARALLEL_ROUTE_CALLS = 6
MAX_PARALLEL_HOTEL_SEARCHES = 3
MAX_PARALLEL_HOTEL_IDENTITIES = 4
MAX_PARALLEL_HOTEL_COMMUTES = 6
MAX_HOTEL_IDENTITY_ATTEMPTS = 12
MAX_HOTEL_OFFERS = 5
MAX_HOTEL_COMMUTE_CLUSTERS = 3


def _minor_amount_query_text(currency: str, amount_minor: int) -> str:
    if currency == "CNY":
        yuan, fen = divmod(amount_minor, 100)
        return f"{yuan}元" if fen == 0 else f"{yuan}.{fen:02d}元"
    return f"{amount_minor}{currency}最小货币单位"


def _hotel_constraint_query(book: TaskBookV4) -> str | None:
    """Project confirmed lodging constraints into FlyAI's free-text query field."""

    lodging = book.lodging_direction
    parts: list[str] = []
    if lodging.area_preferences:
        parts.append("住宿区域：" + "、".join(item.value for item in lodging.area_preferences))
    if lodging.hotel_quality_tier is not None:
        parts.append(
            "酒店档次/星级偏好：" + _HOTEL_QUALITY_QUERY_LABELS[lodging.hotel_quality_tier]
        )
    if lodging.property_type_preferences:
        parts.append(
            "房型或住宿类型：" + "、".join(item.value for item in lodging.property_type_preferences)
        )
    budget = lodging.nightly_budget
    if budget is not None:
        minimum = (
            _minor_amount_query_text(budget.currency, budget.minimum_minor)
            if budget.minimum_minor is not None
            else None
        )
        maximum = (
            _minor_amount_query_text(budget.currency, budget.maximum_minor)
            if budget.maximum_minor is not None
            else None
        )
        if minimum is not None and maximum is not None:
            amount = minimum if minimum == maximum else f"{minimum}至{maximum}"
            parts.append(f"每晚预算：{amount}")
        elif minimum is not None:
            parts.append(f"每晚预算：不低于{minimum}")
        elif maximum is not None:
            parts.append(f"每晚预算：不高于{maximum}")
    if lodging.facility_requirements:
        parts.append("设施要求：" + "、".join(item.value for item in lodging.facility_requirements))
    return "；".join(parts) or None


def validate_location_request(
    request: PlannerCapabilityRequest, workspace: PlannerWorkspaceState, book: TaskBookV4
) -> None:
    args = request.arguments
    dates = book.destination_and_dates
    known = task_book_references(book)
    clusters = (
        {cluster.cluster_id for cluster in workspace.spatial_observation.clusters}
        if workspace.spatial_observation
        else set()
    )
    if isinstance(args, HotelSearchArguments):
        if book.lodging_direction.not_applicable:
            raise PlannerGuardError("planner_hotel_not_applicable")
        if (args.check_in_date, args.check_out_date) != (dates.start_date, dates.end_date):
            raise PlannerGuardError(
                "planner_hotel_dates_do_not_cover_trip:"
                f"expected_check_in={dates.start_date}:expected_check_out={dates.end_date}"
            )
        if args.party_size_ref != "party" or not set(args.activity_cluster_refs) <= clusters:
            raise PlannerGuardError("planner_hotel_reference_invalid")
        requested_refs = (
            *args.lodging_preference_refs,
            *args.facility_constraint_refs,
            *((args.budget_constraint_ref,) if args.budget_constraint_ref else ()),
        )
        if not set(requested_refs) <= known.keys():
            raise PlannerGuardError("planner_hotel_constraint_reference_invalid")
    elif isinstance(args, HotelOfferRefreshArguments):
        if (args.check_in_date, args.check_out_date) != (dates.start_date, dates.end_date):
            raise PlannerGuardError(
                "planner_hotel_refresh_dates_invalid:"
                f"expected_check_in={dates.start_date}:expected_check_out={dates.end_date}"
            )
        if workspace.hotel_observation is None or args.offer_ref not in tuple(
            offer.offer_ref for offer in workspace.hotel_observation.offers
        ):
            raise PlannerGuardError("planner_hotel_refresh_reference_invalid")
    elif isinstance(args, SpatialRoutesArguments):
        if args.departure_service_date not in service_dates(book):
            raise PlannerGuardError("planner_route_date_invalid")
        for pair in args.endpoint_pairs:
            for endpoint in (pair.origin, pair.destination):
                endpoint_coordinates(endpoint, workspace, book)
        if args.comparison:
            if tuple(day.service_date for day in args.comparison.proposed_days) != service_dates(
                book
            ):
                raise PlannerGuardError("planner_route_comparison_must_cover_all_trip_dates")
            for day in (*args.comparison.baseline_days, *args.comparison.proposed_days):
                for endpoint in day.ordered_endpoints:
                    endpoint_coordinates(endpoint, workspace, book)


def endpoint_coordinates(
    endpoint: SpatialRouteEndpoint, workspace: PlannerWorkspaceState, book: TaskBookV4
) -> Gcj02Coordinates:
    places = {item.canonical_entity_id: item for item in workspace.place_evidence}
    if endpoint.kind == "candidate":
        entry = workspace.candidate_pool.candidate_by_id().get(endpoint.reference_id)
        if entry is None or entry.selection_permission == "forbidden":
            raise PlannerGuardError("planner_route_candidate_not_current")
        return places[entry.candidate_ref.canonical_entity_id].coordinates
    if endpoint.kind == "cluster":
        cluster = (
            next(
                (
                    item
                    for item in workspace.spatial_observation.clusters
                    if item.cluster_id == endpoint.reference_id
                ),
                None,
            )
            if workspace.spatial_observation
            else None
        )
        if cluster is None:
            raise PlannerGuardError("planner_route_cluster_not_current")
        # A cluster route explicitly uses a sourced representative, not its centroid.
        return places[cluster.candidate_refs[0].canonical_entity_id].coordinates
    property_id = None
    if endpoint.kind == "hotel_offer" and workspace.hotel_observation:
        offer = next(
            (
                item
                for item in workspace.hotel_observation.offers
                if item.offer_ref.offer_id == endpoint.reference_id
            ),
            None,
        )
        if offer:
            property_id = offer.offer_ref.property_id
    elif endpoint.kind == "fixed_commitment":
        fixed = book.lodging_direction.existing_booking
        if fixed and fixed.booking_id == endpoint.reference_id:
            property_id = fixed.booking_id
    location = next(
        (item for item in workspace.hotel_location_evidence if item.property_id == property_id),
        None,
    )
    if location is None:
        raise PlannerGuardError("planner_route_endpoint_location_unverified")
    return location.coordinates


async def execute_location_capability(
    request: PlannerCapabilityRequest,
    workspace: PlannerWorkspaceState,
    book: TaskBookV4,
    *,
    providers: PlanningProviderSet,
    registry: CityRegistry,
    cancellation: ModelCancellation,
    now: datetime,
    reuse_prepare: bool = False,
) -> EvidenceUpdate:
    validate_location_request(request, workspace, book)
    if isinstance(request.arguments, SpatialRoutesArguments):
        return await _routes(request, workspace, book, providers, registry, cancellation, now)
    try:
        return await _hotels(
            request,
            workspace,
            book,
            providers,
            registry,
            cancellation,
            now,
            reuse_prepare=reuse_prepare,
        )
    except CityProviderUnavailableError:
        # A missing city/provider binding is a real negative observation. It
        # must be checkpointed so the Planner cannot retry the same impossible
        # request until the whole execution times out.
        return _unavailable_hotel_update(request, workspace, book, now, reuse_prepare=reuse_prepare)


def _unavailable_hotel_update(
    request: PlannerCapabilityRequest,
    workspace: PlannerWorkspaceState,
    book: TaskBookV4,
    now: datetime,
    *,
    reuse_prepare: bool = False,
) -> EvidenceUpdate:
    args = request.arguments
    assert isinstance(args, (HotelSearchArguments, HotelOfferRefreshArguments))
    spatial = workspace.spatial_observation
    assert spatial is not None
    cluster_ids = (
        args.activity_cluster_refs
        if isinstance(args, HotelSearchArguments)
        else tuple(cluster.cluster_id for cluster in spatial.clusters)
    )
    refs = task_book_references(book)
    applied = AppliedHotelConstraints(
        area_refs=tuple(key for key in refs if key.startswith("area:")),
        quality_tier=book.lodging_direction.hotel_quality_tier,
        quality_tiers=tuple(book.lodging_direction.hotel_quality_tiers),
        nightly_budget_ref="lodging_budget" if book.lodging_direction.nightly_budget else None,
        property_types=tuple(
            item.value for item in book.lodging_direction.property_type_preferences
        ),
        facility_requirement_refs=tuple(key for key in refs if key.startswith("facility:")),
        source_task_book_refs=("lodging",),
    )
    fixed = book.lodging_direction.existing_booking
    fixed_observation = None
    if fixed is not None:
        fixed_ref = next(
            item
            for item in workspace.candidate_pool.fixed_commitments
            if item.commitment_id == fixed.booking_id
        )
        fixed_observation = FixedBookingObservation(
            commitment_ref=fixed_ref,
            verification_status="unavailable",
        )
    observation_id = str(uuid4())
    hotel = HotelObservation(
        query_status="failed",
        query_origin="prepare_handoff" if reuse_prepare else "planner_query",
        query_attempts=(
            HotelQueryAttempt(
                outcome="failed",
                error_code="provider_city_binding",
                observed_at=now,
            ),
        ),
        hotel_observation_id=observation_id,
        scope=request.scope,
        request_id=request.request_id,
        mode="fixed_booking_verification" if fixed else "search",
        status="unavailable",
        observed_at=now,
        expires_at=now + timedelta(minutes=20),
        stay_segments=(
            HotelStaySegment(
                check_in_date=args.check_in_date,
                check_out_date=args.check_out_date,
                nights=(args.check_out_date - args.check_in_date).days,
                activity_cluster_ids=cluster_ids,
            ),
        ),
        applied_constraints=applied,
        fixed_booking=fixed_observation,
        missing_fact_kinds=("provider_city_binding", "verified_hotel_entity"),
    )
    receipt = PlannerCapabilityObservation(
        observation_id=str(uuid4()),
        scope=request.scope,
        request_id=request.request_id,
        capability=request.capability,
        status="unavailable",
        artifact_reference_ids=(hotel.hotel_observation_id,),
        reason_summary=(
            "目的地已进入规划，但本次运行没有可用的酒店 Provider 城市绑定；"
            "已记录不可用结果，不能在同一执行段原样重试。"
        ),
        observed_at=now,
    )
    return EvidenceUpdate(observation=receipt, hotel=hotel)


async def _routes(
    request: PlannerCapabilityRequest,
    workspace: PlannerWorkspaceState,
    book: TaskBookV4,
    providers: PlanningProviderSet,
    registry: CityRegistry,
    cancellation: ModelCancellation,
    now: datetime,
) -> EvidenceUpdate:
    args = request.arguments
    assert isinstance(args, SpatialRoutesArguments)
    city = registry.provider_scope(book.destination_and_dates.destination_name, ProviderCode.AMAP)
    modes = {
        "public_transit": RouteMode.TRANSIT,
        "taxi": RouteMode.DRIVING,
        "driving": RouteMode.DRIVING,
        "walking": RouteMode.WALKING,
    }
    semaphore = asyncio.Semaphore(MAX_PARALLEL_ROUTE_CALLS)
    previous_edges = {edge.route_edge_id: edge for edge in workspace.route_evidence}

    async def query_pair(
        pair: SpatialRoutePair,
    ) -> tuple[tuple[SpatialRouteEdge, ...], tuple[VerifiedFactSummary, ...]]:
        observations: dict[
            RouteMode, tuple[ProviderResponse[ProviderRoute] | None, ProviderError | None, bool]
        ] = {}
        requested_modes = list(dict.fromkeys(modes[mode] for mode in args.transport_modes))
        groups = (
            [[mode] for mode in requested_modes] if workspace.react_state else [requested_modes]
        )
        async with semaphore:
            for group in groups:
                cancellation.raise_if_cancelled("planner_requested_routes")
                failure: ProviderError | None = None
                skipped = False
                try:
                    response = await providers.routes.get_routes(
                        RouteRequest(
                            city=city,
                            origin=endpoint_coordinates(pair.origin, workspace, book),
                            destination=endpoint_coordinates(pair.destination, workspace, book),
                            modes=group,
                        )
                    )
                except ProviderError as error:
                    response = None
                    failure = error
                except RequestBudgetExceeded:
                    if workspace.react_state is None:
                        raise
                    response = None
                    skipped = True
                cancellation.raise_if_cancelled("planner_requested_routes_observe")
                for provider_mode in group:
                    observations[provider_mode] = response, failure, skipped
        pair_edges: list[SpatialRouteEdge] = []
        pair_facts: list[VerifiedFactSummary] = []
        for mode in args.transport_modes:
            response, failure, skipped = observations[modes[mode]]
            edge_id = server_id(
                workspace.generation_id,
                pair.origin.kind,
                pair.origin.reference_id,
                pair.destination.kind,
                pair.destination.reference_id,
                mode,
            )
            choices = (
                [item for item in response.items if item.mode is modes[mode]]
                if response is not None
                else []
            )
            if not choices:
                pair_edges.append(
                    SpatialRouteEdge(
                        route_edge_id=edge_id,
                        origin=pair.origin,
                        destination=pair.destination,
                        transport_mode=mode,
                        status="missing",
                        query_failure=route_query_failure(
                            modes[mode],
                            response,
                            failure,
                            previous_edges.get(edge_id),
                            now,
                            skipped=skipped,
                        ),
                        missing_reason=(
                            "本次规划的外部请求预算已用完或收尾时间已到，未取得该方式的路线证据；不能认定没有路线。"
                            if skipped
                            else missing_route_reason(modes[mode], response, failure)
                        ),
                    )
                )
                continue
            route = min(choices, key=lambda item: item.duration_seconds)
            fact_id = server_id(edge_id, route.fetched_at.isoformat())
            pair_facts.append(
                VerifiedFactSummary(
                    fact_reference_id=fact_id,
                    fact_kind="route_cost",
                    safe_summary=(
                        f"{pair.origin.kind}:{pair.origin.reference_id} → "
                        f"{pair.destination.kind}:{pair.destination.reference_id}，{mode}，"
                        f"查询时估算{ceil(route.duration_seconds / 60)}分钟、{route.distance_m}米；"
                        "非未来出发时间预测。"
                    ),
                    observed_at=route.fetched_at,
                    expires_at=route.fetched_at + timedelta(hours=2),
                    source_reference_ids=(f"provider:amap:route:{edge_id}",),
                )
            )
            pair_edges.append(
                SpatialRouteEdge(
                    route_edge_id=edge_id,
                    origin=pair.origin,
                    destination=pair.destination,
                    transport_mode=mode,
                    status="available",
                    duration_minutes=ceil(route.duration_seconds / 60),
                    distance_meters=route.distance_m,
                    transfer_count=route.transfer_count,
                    fare=route.fare,
                    polyline=tuple(route.polyline),
                    fact_reference_ids=(fact_id,),
                )
            )
        return tuple(pair_edges), tuple(pair_facts)

    batches = await asyncio.gather(*(query_pair(pair) for pair in args.endpoint_pairs))
    edges = [edge for pair_edges, _ in batches for edge in pair_edges]
    facts = [fact for _, pair_facts in batches for fact in pair_facts]
    usable = [edge for edge in edges if edge.status == "available"]
    comparison = (
        compare_observed_routes(
            args.comparison,
            tuple(edges),
            tuple(facts),
            scope=request.scope,
            pool_revision=workspace.candidate_pool.revision,
            request_id=request.request_id,
            now=now,
        )
        if args.comparison
        else None
    )
    if comparison:
        facts.append(
            VerifiedFactSummary(
                fact_reference_id=comparison.observation_id,
                fact_kind="route_comparison",
                safe_summary=(
                    f"同一地点集合与旅行日期的路线对照：基线{comparison.baseline_duration_minutes}分钟，"
                    f"提议{comparison.proposed_duration_minutes}分钟。仅表示相对该基线的查询时估算，"
                    "不代表所有方案中的全局最优或未来实时路况。"
                ),
                observed_at=now,
                expires_at=comparison.expires_at,
                source_reference_ids=tuple(
                    dict.fromkeys(
                        ref for edge in comparison.route_edges for ref in edge.fact_reference_ids
                    )
                ),
            )
        )
    observation = PlannerCapabilityObservation(
        observation_id=str(uuid4()),
        scope=request.scope,
        request_id=request.request_id,
        capability=request.capability,
        status="complete"
        if usable and len(usable) == len(edges)
        else "partial"
        if usable
        else "unavailable",
        service_dates=(args.departure_service_date,),
        fact_reference_ids=tuple(fact.fact_reference_id for fact in facts),
        artifact_reference_ids=tuple(edge.route_edge_id for edge in edges)
        + ((comparison.observation_id,) if comparison else ()),
        reason_summary=(
            "按 Planner 指定端点和交通方式查询真实路线；"
            "簇端点为实际代表地点，出发时刻预测未由当前接口保证。"
        ),
        observed_at=now,
    )
    return EvidenceUpdate(
        observation=observation,
        route_edges=tuple(edges),
        facts=tuple(facts),
        route_comparisons=(comparison,) if comparison else (),
    )


async def _verified_hotel_place(
    name: str,
    expected_coordinates: Gcj02Coordinates | None,
    providers: PlanningProviderSet,
    registry: CityRegistry,
    book: TaskBookV4,
    expected_canonical_id: str | None = None,
) -> ProviderPlace | None:
    city = registry.provider_scope(book.destination_and_dates.destination_name, ProviderCode.AMAP)
    response = await providers.places.search_places(
        KeywordPlaceSearchRequest(
            city=city, query=name, category_hint=PlaceCategory.HOTEL, page_size=25
        )
    )

    def normalized(value: str) -> str:
        text = re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", value)).casefold()
        return text.removeprefix(book.destination_and_dates.destination_name.removesuffix("市"))

    matches = []
    for place in response.items:
        if (
            place.city_id != city.city_id
            or category_from_original_typecodes(place.provider_typecode) is not PlaceCategory.HOTEL
        ):
            continue
        if expected_canonical_id:
            if str(uuid5(NAMESPACE_URL, f"amap:{place.source_place_id}")) != expected_canonical_id:
                continue
        elif normalized(place.name) != normalized(name):
            continue
        if expected_coordinates is not None:
            dx = (
                (place.coordinates.longitude - expected_coordinates.longitude)
                * cos(radians(expected_coordinates.latitude))
                * 111_320
            )
            dy = (place.coordinates.latitude - expected_coordinates.latitude) * 110_540
            if sqrt(dx * dx + dy * dy) > 350:
                continue
        matches.append(place)
    return matches[0] if len(matches) == 1 else None


async def _hotels(
    request: PlannerCapabilityRequest,
    workspace: PlannerWorkspaceState,
    book: TaskBookV4,
    providers: PlanningProviderSet,
    registry: CityRegistry,
    cancellation: ModelCancellation,
    now: datetime,
    *,
    reuse_prepare: bool = False,
) -> EvidenceUpdate:
    args = request.arguments
    assert isinstance(args, (HotelSearchArguments, HotelOfferRefreshArguments))
    spatial = workspace.spatial_observation
    if spatial is None or not spatial.clusters:
        raise PlannerGuardError("planner_hotel_requires_activity_clusters")
    cluster_ids = (
        args.activity_cluster_refs
        if isinstance(args, HotelSearchArguments)
        else tuple(cluster.cluster_id for cluster in spatial.clusters)
    )
    refs = task_book_references(book)
    applied = AppliedHotelConstraints(
        area_refs=tuple(key for key in refs if key.startswith("area:")),
        quality_tier=book.lodging_direction.hotel_quality_tier,
        quality_tiers=tuple(book.lodging_direction.hotel_quality_tiers),
        nightly_budget_ref="lodging_budget" if book.lodging_direction.nightly_budget else None,
        property_types=tuple(
            item.value for item in book.lodging_direction.property_type_preferences
        ),
        facility_requirement_refs=tuple(key for key in refs if key.startswith("facility:")),
        source_task_book_refs=("lodging",),
    )
    stays = (
        HotelStaySegment(
            check_in_date=args.check_in_date,
            check_out_date=args.check_out_date,
            nights=(args.check_out_date - args.check_in_date).days,
            activity_cluster_ids=cluster_ids,
        ),
    )
    observation_id = str(uuid4())
    expires = now + timedelta(minutes=20)
    fixed = book.lodging_direction.existing_booking
    locations = []
    facts = []
    offers: list[HotelOfferObservation] = []
    fixed_observation = None
    attempts: list[HotelQueryAttempt] = []

    def record_search(
        hotel_request: HotelSearchRequest,
        response: ProviderResponse[ProviderHotelOffer] | None = None,
        error: ProviderError | None = None,
    ) -> None:
        attempts.append(
            HotelQueryAttempt(
                anchor_name=hotel_request.anchor_name,
                search_keyword=hotel_request.query,
                outcome="failed"
                if error
                else "results"
                if response and response.items
                else "empty",
                result_count=len(response.items) if response else 0,
                error_code=error.code.value if error else None,
                retryable=error.retryable if error else False,
                attempts=error.attempts if error else 1,
                observed_at=response.fetched_at if response else now,
            )
        )
        if response:
            for failure in response.failures.values():
                attempts.append(
                    HotelQueryAttempt(
                        anchor_name=hotel_request.anchor_name,
                        search_keyword=hotel_request.query,
                        outcome="failed",
                        error_code=failure.code.value,
                        retryable=failure.retryable,
                        attempts=failure.attempts,
                        observed_at=response.fetched_at,
                    )
                )

    if fixed:
        fixed_ref = next(
            item
            for item in workspace.candidate_pool.fixed_commitments
            if item.commitment_id == fixed.booking_id
        )
        try:
            place = await _verified_hotel_place(
                fixed.user_description, None, providers, registry, book, fixed.canonical_entity_id
            )
        except ProviderError as error:
            attempts.append(
                HotelQueryAttempt(
                    stage="identity",
                    outcome="failed",
                    error_code=error.code.value,
                    retryable=error.retryable,
                    attempts=error.attempts,
                    observed_at=now,
                )
            )
            place = None
        if place:
            fact_id = server_id("fixed_hotel", place.source_place_id, place.fetched_at.isoformat())
            locations.append(
                PlannerHotelLocationEvidence(
                    property_id=fixed.booking_id,
                    provider_entity_id=place.source_place_id,
                    coordinates=place.coordinates,
                    display_name=place.name,
                    address=place.address,
                    rating=place.rating,
                    fact_reference_id=fact_id,
                    observed_at=place.fetched_at,
                )
            )
            facts.append(
                VerifiedFactSummary(
                    fact_reference_id=fact_id,
                    fact_kind="fixed_hotel_identity",
                    safe_summary=f"已核验用户预订酒店的同城实体：{place.name}；预订房型与入住凭证不是地图能够核验的事实。",
                    observed_at=place.fetched_at,
                    source_reference_ids=(f"provider:amap:{place.source_place_id}",),
                )
            )
        fixed_observation = FixedBookingObservation(
            commitment_ref=fixed_ref,
            property_id=fixed.booking_id if place else None,
            verification_status="verified" if place else "unavailable",
        )
    else:
        city = registry.provider_scope(
            book.destination_and_dates.destination_name, ProviderCode.FLYAI
        )
        amap_city = registry.provider_scope(
            book.destination_and_dates.destination_name, ProviderCode.AMAP
        )
        selected_keyword = args.search_keyword if isinstance(args, HotelSearchArguments) else None
        constraint_query = selected_keyword or (
            _hotel_constraint_query(book) if workspace.react_state is None else None
        )
        places = {place.canonical_entity_id: place for place in workspace.place_evidence}
        clusters = {cluster.cluster_id: cluster for cluster in spatial.clusters}
        search_names = [
            places[clusters[key].candidate_refs[0].canonical_entity_id].display_name
            for key in cluster_ids[:3]
        ]
        refresh_property = (
            args.offer_ref.property_id if isinstance(args, HotelOfferRefreshArguments) else None
        )
        search_semaphore = asyncio.Semaphore(MAX_PARALLEL_HOTEL_SEARCHES)
        relaxed_keyword_searches: set[str] = set()

        async def search(anchor_name: str) -> ProviderResponse[ProviderHotelOffer] | None:
            async with search_semaphore:
                cancellation.raise_if_cancelled("planner_hotel_search")
                try:
                    hotel_request = HotelSearchRequest(
                        city=city,
                        check_in=args.check_in_date,
                        check_out=args.check_out_date,
                        anchor_name=anchor_name,
                        query=constraint_query,
                        hotel_stars=[
                            HOTEL_STARS[tier] for tier in book.lodging_direction.hotel_quality_tiers
                        ],
                        hotel_types=[
                            cast(Literal["酒店", "民宿", "客栈"], item.value)
                            for item in book.lodging_direction.property_type_preferences
                            if item.value in {"酒店", "民宿"}
                        ],
                        sort="rate_desc" if book.lodging_direction.hotel_quality_tiers else None,
                    )
                    response = await providers.products.search_hotels(hotel_request)
                    record_search(hotel_request, response=response)
                    if (
                        not response.items
                        and constraint_query is not None
                        and workspace.react_state is None
                        and selected_keyword is None
                    ):
                        # FlyAI treats key-words as a search term, not a structured
                        # budget/facility filter. Keep the dates and real area anchor,
                        # then verify constraints separately on returned facts.
                        cancellation.raise_if_cancelled("planner_hotel_keyword_recovery")
                        relaxed_keyword_searches.add(anchor_name)
                        hotel_request = hotel_request.model_copy(update={"query": None})
                        response = await providers.products.search_hotels(hotel_request)
                        record_search(hotel_request, response=response)
                    return response
                except ProviderError as error:
                    record_search(hotel_request, error=error)
                    return None

        use_prepare_examples = (
            bool(book.lodging_direction.search_examples)
            and (workspace.react_state is None or reuse_prepare)
            and selected_keyword is None
        )
        if use_prepare_examples:
            from backend.discovery.lodging_search import (
                hotel_requests,
                search_lodging_results,
                top_hotels,
            )

            lodging = book.lodging_direction
            requests = hotel_requests(
                city=city,
                check_in=args.check_in_date,
                check_out=args.check_out_date,
                examples=lodging.search_examples,
                tiers=lodging.hotel_quality_tiers
                or ([lodging.hotel_quality_tier] if lodging.hotel_quality_tier else []),
                types=[item.value for item in lodging.property_type_preferences],
            )
            cancellation.raise_if_cancelled("planner_hotel_prepare_handoff")
            results = await search_lodging_results(providers.products, requests)
            cancellation.raise_if_cancelled("planner_hotel_prepare_handoff")
            responses = []
            for result in results:
                if result.internal_error:
                    attempts.append(
                        HotelQueryAttempt(
                            anchor_name=result.request.anchor_name,
                            search_keyword=result.request.query,
                            outcome="failed",
                            error_code="internal_error",
                            observed_at=now,
                        )
                    )
                else:
                    record_search(result.request, response=result.response, error=result.error)
                if result.response is not None:
                    responses.append(
                        result.response.model_copy(
                            update={"items": top_hotels(result.response.items)}
                        )
                    )
        else:
            responses = [
                response
                for response in await asyncio.gather(*(search(anchor) for anchor in search_names))
                if response is not None
            ]
        identity_limit = (
            21
            if use_prepare_examples and workspace.react_state is None
            else MAX_HOTEL_IDENTITY_ATTEMPTS
        )
        offer_limit = (
            21 if use_prepare_examples and workspace.react_state is None else MAX_HOTEL_OFFERS
        )
        candidate_offers: list[ProviderHotelOffer] = []
        seen_properties: set[str] = set()
        max_result_count = max(
            (len(response.items) for response in responses if response is not None),
            default=0,
        )
        # Interleave anchors so one noisy search cannot consume every bounded
        # identity-verification attempt before another area is considered.
        for result_index in range(max_result_count):
            for response in responses:
                if response is None or result_index >= len(response.items):
                    continue
                offer = response.items[result_index]
                budget = book.lodging_direction.nightly_budget
                if (
                    budget is not None
                    and budget.currency == "CNY"
                    and budget.maximum_minor is not None
                    and offer.room_price is not None
                    and offer.room_price.minimum_fen > budget.maximum_minor
                ):
                    # A known over-budget offer must not consume the bounded
                    # identity checks; unknown prices stay explicitly unknown.
                    continue
                if offer.source_hotel_id in seen_properties or (
                    refresh_property is not None and refresh_property != offer.source_hotel_id
                ):
                    continue
                seen_properties.add(offer.source_hotel_id)
                candidate_offers.append(offer)
                if len(candidate_offers) >= identity_limit:
                    break
            if len(candidate_offers) >= identity_limit:
                break

        identity_semaphore = asyncio.Semaphore(MAX_PARALLEL_HOTEL_IDENTITIES)

        async def verify_identity(
            offer: ProviderHotelOffer,
        ) -> tuple[ProviderHotelOffer, ProviderPlace] | None:
            async with identity_semaphore:
                cancellation.raise_if_cancelled("planner_hotel_identity")
                try:
                    coordinates = (
                        coordinates_to_gcj02(offer.raw_coordinates)
                        if offer.raw_coordinates
                        else None
                    )
                    place = await _verified_hotel_place(
                        offer.name, coordinates, providers, registry, book
                    )
                except ProviderError as error:
                    attempts.append(
                        HotelQueryAttempt(
                            stage="identity",
                            anchor_name=offer.name,
                            outcome="failed",
                            error_code=error.code.value,
                            retryable=error.retryable,
                            attempts=error.attempts,
                            observed_at=now,
                        )
                    )
                    return None
                except ValueError:
                    return None
                return (offer, place) if place is not None else None

        if workspace.react_state is not None:
            # Preserve the same offer order and identity checks, but stop once
            # enough verified options exist instead of checking discarded rows.
            verified_results = []
            for offset in range(0, len(candidate_offers), MAX_PARALLEL_HOTEL_IDENTITIES):
                verified_results.extend(
                    await asyncio.gather(
                        *(
                            verify_identity(offer)
                            for offer in candidate_offers[
                                offset : offset + MAX_PARALLEL_HOTEL_IDENTITIES
                            ]
                        )
                    )
                )
                if sum(result is not None for result in verified_results) >= offer_limit:
                    break
        else:
            verified_results = await asyncio.gather(
                *(verify_identity(offer) for offer in candidate_offers)
            )
        verified_offers = tuple(result for result in verified_results if result is not None)[
            :offer_limit
        ]

        # The Agent can request comparisons through MCP. Automatic calculation
        # only needs the chosen hotel's actual adjacent itinerary legs.
        commute_cluster_ids = (
            ()
            if workspace.react_state is not None
            else tuple(cluster_ids[:MAX_HOTEL_COMMUTE_CLUSTERS])
        )
        commute_semaphore = asyncio.Semaphore(MAX_PARALLEL_HOTEL_COMMUTES)

        async def query_commute(
            offer: ProviderHotelOffer,
            place: ProviderPlace,
            cluster_id: str,
        ) -> tuple[HotelClusterCommute | None, VerifiedFactSummary | None]:
            target = places[clusters[cluster_id].candidate_refs[0].canonical_entity_id]
            async with commute_semaphore:
                cancellation.raise_if_cancelled("planner_hotel_commute")
                try:
                    response = await providers.routes.get_routes(
                        RouteRequest(
                            city=amap_city,
                            origin=place.coordinates,
                            destination=target.coordinates,
                            modes=[RouteMode.DRIVING],
                        )
                    )
                except ProviderError:
                    return None, None
            choices = [item for item in response.items if item.mode is RouteMode.DRIVING]
            if not choices:
                return None, None
            route = min(choices, key=lambda item: item.duration_seconds)
            route_fact = server_id(
                "hotel_commute",
                offer.source_hotel_id,
                cluster_id,
                route.fetched_at.isoformat(),
            )
            return (
                HotelClusterCommute(
                    cluster_id=cluster_id,
                    route_observation_ref=route_fact,
                    duration_minutes=ceil(route.duration_seconds / 60),
                ),
                VerifiedFactSummary(
                    fact_reference_id=route_fact,
                    fact_kind="hotel_commute",
                    safe_summary=(
                        f"{offer.name}至{target.display_name}，"
                        "当前驾车/打车路线估算"
                        f"{ceil(route.duration_seconds / 60)}分钟。"
                    ),
                    observed_at=route.fetched_at,
                    expires_at=expires,
                    source_reference_ids=(f"provider:amap:route:{route_fact}",),
                ),
            )

        commute_jobs = tuple(
            (offer, place, cluster_id)
            for offer, place in verified_offers
            for cluster_id in commute_cluster_ids
        )
        commute_results = await asyncio.gather(
            *(query_commute(offer, place, cluster_id) for offer, place, cluster_id in commute_jobs)
        )
        commutes_by_property: dict[str, list[HotelClusterCommute]] = {
            offer.source_hotel_id: [] for offer, _ in verified_offers
        }
        commute_facts_by_property: dict[str, list[VerifiedFactSummary]] = {
            offer.source_hotel_id: [] for offer, _ in verified_offers
        }
        for (offer, _place, _cluster_id), (commute, fact) in zip(
            commute_jobs, commute_results, strict=True
        ):
            if commute is not None:
                commutes_by_property[offer.source_hotel_id].append(commute)
            if fact is not None:
                commute_facts_by_property[offer.source_hotel_id].append(fact)

        for offer, place in verified_offers:
            cancellation.raise_if_cancelled("planner_hotel_observe")
            fact_id = server_id(
                "hotel_search",
                offer.source_hotel_id,
                offer.fetched_at.isoformat(),
                args.check_in_date,
                args.check_out_date,
            )
            reference_price = (
                MoneyAmountRange(
                    currency=offer.room_price.currency,
                    minimum_minor=offer.room_price.minimum_fen,
                    maximum_minor=offer.room_price.maximum_fen,
                )
                if offer.room_price is not None
                and offer.room_price.minimum_fen > 0
                and offer.room_price.maximum_fen > 0
                else None
            )
            reference_price_fact_id = (
                server_id(
                    "hotel_reference_price",
                    offer.source_offer_id,
                    offer.fetched_at.isoformat(),
                    args.check_in_date,
                    args.check_out_date,
                )
                if reference_price is not None
                else None
            )
            facts.append(
                VerifiedFactSummary(
                    fact_reference_id=fact_id,
                    fact_kind="hotel_search",
                    safe_summary=(
                        f"实时酒店搜索返回{offer.name}，已用高德核验同城位置。"
                        + ("列表参考价已另行记录；" if reference_price is not None else "")
                        + "候选返回不代表任务书住宿约束已逐项核验满足；"
                        + "当前接口未给出可核验的房型库存、入住总价与取消条款。"
                    ),
                    observed_at=offer.fetched_at,
                    expires_at=expires,
                    source_reference_ids=(
                        f"provider:flyai:{offer.source_hotel_id}",
                        f"provider:amap:{place.source_place_id}",
                    ),
                )
            )
            if reference_price is not None and reference_price_fact_id is not None:
                minimum_yuan = reference_price.minimum_minor / 100
                maximum_yuan = reference_price.maximum_minor / 100
                price_text = (
                    f"{minimum_yuan:.2f}元"
                    if minimum_yuan == maximum_yuan
                    else f"{minimum_yuan:.2f}元至{maximum_yuan:.2f}元"
                )
                facts.append(
                    VerifiedFactSummary(
                        fact_reference_id=reference_price_fact_id,
                        fact_kind="hotel_reference_price",
                        safe_summary=(
                            f"FlyAI酒店列表返回{offer.name}每间夜参考价{price_text}；"
                            "该区间不代表指定房型库存、含税入住总价或可预订承诺。"
                        ),
                        observed_at=offer.fetched_at,
                        expires_at=expires,
                        source_reference_ids=(f"provider:flyai:{offer.source_offer_id}",),
                    )
                )
            facts.extend(commute_facts_by_property[offer.source_hotel_id])
            locations.append(
                PlannerHotelLocationEvidence(
                    property_id=offer.source_hotel_id,
                    provider_entity_id=place.source_place_id,
                    coordinates=place.coordinates,
                    display_name=offer.name,
                    address=place.address or offer.address,
                    rating=offer.rating or place.rating,
                    fact_reference_id=fact_id,
                    observed_at=place.fetched_at,
                )
            )
            offers.append(
                HotelOfferObservation(
                    offer_ref=HotelOfferRef(
                        hotel_observation_id=observation_id,
                        offer_id=server_id(observation_id, offer.source_offer_id),
                        property_id=offer.source_hotel_id,
                        inventory_snapshot_id=canonical_json_hash(
                            {
                                "offer": offer.source_offer_id,
                                "observed_at": offer.fetched_at.isoformat(),
                                "check_in": args.check_in_date.isoformat(),
                                "check_out": args.check_out_date.isoformat(),
                            }
                        ),
                    ),
                    property_name=offer.name,
                    area_ref=f"amap:{place.source_place_id}",
                    availability_status="unknown",
                    room_and_price_fact_refs=(),
                    reference_price=reference_price,
                    reference_price_fact_refs=(reference_price_fact_id,)
                    if reference_price_fact_id is not None
                    else (),
                    total_price=None,
                    price_missing=True,
                    commute_to_clusters=tuple(commutes_by_property[offer.source_hotel_id]),
                    source_reference_ids=(fact_id,),
                    observed_at=offer.fetched_at,
                    expires_at=expires,
                )
            )
    query_status: Literal["available", "failed", "unverified", "empty"] = (
        "available"
        if offers or (fixed_observation and fixed_observation.verification_status == "verified")
        else "failed"
        if any(item.outcome == "failed" for item in attempts)
        else "unverified"
        if fixed or any(item.result_count for item in attempts)
        else "empty"
    )
    hotel = HotelObservation(
        query_status=query_status,
        query_origin="prepare_handoff" if reuse_prepare else "planner_query",
        query_attempts=tuple(attempts),
        search_keyword=args.search_keyword if isinstance(args, HotelSearchArguments) else None,
        hotel_observation_id=observation_id,
        scope=request.scope,
        request_id=request.request_id,
        mode="fixed_booking_verification" if fixed else "search",
        status="complete"
        if fixed_observation and fixed_observation.verification_status == "verified"
        else "partial"
        if offers
        else "unavailable",
        observed_at=now,
        expires_at=expires,
        stay_segments=stays,
        applied_constraints=applied,
        fixed_booking=fixed_observation,
        offers=tuple(offers),
        missing_fact_kinds=()
        if fixed_observation and fixed_observation.verification_status == "verified"
        else ("room_inventory", "total_price", "cancellation_terms")
        if offers
        else ("verified_hotel_entity",),
        source_reference_ids=tuple(fact.fact_reference_id for fact in facts),
    )
    receipt = PlannerCapabilityObservation(
        observation_id=str(uuid4()),
        scope=request.scope,
        request_id=request.request_id,
        capability=request.capability,
        status=hotel.status,
        fact_reference_ids=hotel.source_reference_ids,
        artifact_reference_ids=(hotel.hotel_observation_id,),
        reason_summary=(
            (
                "已接收 Prepare 住宿预查条件对应的结果，复用有效缓存并核验酒店实体；"
                if reuse_prepare
                else "约束关键词没有匹配结果，已保留日期与活动区位进行一次扩展查询；"
                "已排除参考价明确超过每晚预算上限的候选；"
                if not fixed and relaxed_keyword_searches
                else "已按 Agent 选定的活动区域及可选关键词查询，保留任务书星级、类型和预算核验；"
                if not fixed and workspace.react_state is not None
                else "FlyAI查询已携带任务书中实际存在的住宿约束；"
                if not fixed and _hotel_constraint_query(book) is not None
                else "任务书没有可写入FlyAI查询的具体住宿约束；"
                if not fixed
                else "已按任务书核验用户提供的固定住宿；"
            )
            + "Provider返回候选不代表档次、预算、房型或设施已逐项核验满足；"
            + "仅同城酒店身份和已返回的通勤证据经过对应来源核验；"
            "列表参考价独立标注，不冒充可预订总价、房型库存或取消条款。"
        ),
        observed_at=now,
    )
    return EvidenceUpdate(
        observation=receipt, hotel=hotel, hotel_locations=tuple(locations), facts=tuple(facts)
    )
