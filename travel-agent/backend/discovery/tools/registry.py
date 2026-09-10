"""Bounded, provider-backed Prepare tool registry with normalized observations."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from uuid import NAMESPACE_URL, UUID, uuid5
from zoneinfo import ZoneInfo

from backend.contracts.enums import ProviderCode
from backend.contracts.v4.enums import ToolObservationStatus
from backend.contracts.v4.prepare import (
    HotelBookingFactsRequest,
    OpeningHoursRequest,
    PlaceFactsRequest,
    PlaceProductsRequest,
    PrepareToolCapability,
    ResolvePlaceRequest,
    SpatialRoutesRequest,
    TicketAvailabilityRequest,
    ToolFact,
    ToolObservation,
    ToolRequest,
    ToolRequestValue,
    WeatherForecastRequest,
)
from backend.contracts.v4.state import TripSemanticState
from backend.planning.city_registry import (
    CityRegistry,
    CityRegistryError,
    default_city_registry,
)
from backend.providers.contracts import (
    HotelSearchRequest,
    HoursDayStatus,
    HoursRequest,
    KeywordPlaceSearchRequest,
    ProductSearchRequest,
    ProviderError,
    ProviderPlace,
    ProviderResultStatus,
    RouteMode,
    RouteRequest,
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


@dataclass(frozen=True, slots=True)
class ExecutedToolObservation:
    observation: ToolObservation
    provider: str


@dataclass(frozen=True, slots=True)
class _ResolvedPlace:
    canonical_id: str
    place: ProviderPlace


class PrepareToolExecutor:
    """Execute only the frozen V4 Prepare capabilities, never arbitrary functions."""

    def __init__(
        self,
        *,
        places: PlaceProvider | None,
        hours: HoursProvider | None,
        routes: RouteProvider | None,
        products: TravelProductProvider | None,
        weather: WeatherProvider | None,
        city_registry: CityRegistry | None = None,
        business_date: date | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._places = places
        self._hours = hours
        self._routes = routes
        self._products = products
        self._weather = weather
        self._cities = city_registry or default_city_registry()
        self._business_date = business_date or date.today()
        self._clock = clock or (lambda: datetime.now(UTC))
        self._resolved: dict[str, _ResolvedPlace] = {}

    def supports(self, capability: str) -> bool:
        try:
            value = PrepareToolCapability(capability)
        except ValueError:
            return False
        return {
            PrepareToolCapability.RESOLVE_PLACE: self._places is not None,
            PrepareToolCapability.PLACE_FACTS: self._places is not None,
            PrepareToolCapability.OPENING_HOURS: self._hours is not None,
            PrepareToolCapability.TICKET_AVAILABILITY: self._products is not None,
            PrepareToolCapability.WEATHER_FORECAST: self._weather is not None,
            PrepareToolCapability.SPATIAL_ROUTES: self._routes is not None,
            PrepareToolCapability.HOTEL_BOOKING_FACTS: self._products is not None,
            PrepareToolCapability.PLACE_PRODUCTS: self._products is not None,
        }[value]

    async def execute_plan(
        self,
        requests: list[ToolRequest],
        semantic_state: TripSemanticState,
    ) -> list[ExecutedToolObservation]:
        """Run a validated two-layer DAG; independent requests share one provider round."""

        pending = {item.root.request_id: item for item in requests}
        completed: set[str] = set()
        results: list[ExecutedToolObservation] = []
        while pending:
            ready = [
                item for item in pending.values() if set(item.root.depends_on).issubset(completed)
            ]
            if not ready:
                raise ValueError("tool dependency plan has no executable request")
            observations = await asyncio.gather(
                *(self.execute(item, semantic_state) for item in ready)
            )
            results.extend(observations)
            for item in ready:
                completed.add(item.root.request_id)
                pending.pop(item.root.request_id)
        return results

    async def execute(
        self,
        request: ToolRequest,
        semantic_state: TripSemanticState,
    ) -> ExecutedToolObservation:
        value = request.root
        if not self.supports(value.capability.value):
            return self._unavailable(value, "该能力当前不可用。")
        try:
            if isinstance(value, ResolvePlaceRequest):
                return await self._resolve_place(value, semantic_state)
            if isinstance(value, PlaceFactsRequest):
                return await self._place_facts(value)
            if isinstance(value, OpeningHoursRequest):
                return await self._opening_hours(value)
            if isinstance(value, TicketAvailabilityRequest):
                return await self._ticket_availability(value, semantic_state)
            if isinstance(value, WeatherForecastRequest):
                return await self._weather_forecast(value, semantic_state)
            if isinstance(value, SpatialRoutesRequest):
                return await self._spatial_routes(value, semantic_state)
            if isinstance(value, HotelBookingFactsRequest):
                return await self._hotel_booking_facts(value, semantic_state)
            if isinstance(value, PlaceProductsRequest):
                return await self._place_products(value, semantic_state)
        except ProviderError as error:
            return self._unavailable(
                value,
                "外部数据暂时无法可靠返回。",
                error.provider.value,
            )
        except ValueError:
            return self._invalid(value, "工具参数无法映射到受支持的数据范围。")
        return self._invalid(value, "工具请求类型不受支持。")

    async def _resolve_place(
        self,
        request: ResolvePlaceRequest,
        semantic_state: TripSemanticState,
    ) -> ExecutedToolObservation:
        assert self._places is not None
        try:
            scope = self._cities.provider_scope(request.city, ProviderCode.AMAP)
        except CityRegistryError:
            basics = semantic_state.trip_basics
            authoritative_city = basics.destination_canonical_id or basics.destination_name
            if authoritative_city is None:
                raise
            scope = self._cities.provider_scope(authoritative_city, ProviderCode.AMAP)
        response = await self._places.search_places(
            KeywordPlaceSearchRequest(city=scope, query=request.query, page_size=10)
        )
        if response.status is ProviderResultStatus.EMPTY or not response.items:
            return self._unavailable(request, "没有找到可确认的地点实体。", "amap")
        place = _best_place_match(request.query, response.items)
        # Card selections and the formal Planner use the provider entity as the
        # stable cross-stage identity. Free-text resolution must produce the
        # same canonical ID or a confirmed task-book entity can never be
        # re-verified after the Prepare executor's in-memory cache is gone.
        canonical_id = str(uuid5(NAMESPACE_URL, f"{place.provider.value}:{place.source_place_id}"))
        self._resolved[canonical_id] = _ResolvedPlace(canonical_id, place)
        source_ref = f"{place.provider.value}:place:{place.source_place_id}"
        summary = _place_summary(place)
        return self._success(
            request,
            provider=place.provider.value,
            facts=[
                ToolFact(
                    fact_kind="canonical_place",
                    subject_ref=canonical_id,
                    value_summary=summary,
                    source_reference_ids=[source_ref],
                    observed_at=place.fetched_at,
                    expires_at=place.fetched_at + timedelta(days=30),
                )
            ],
            entity_refs=[canonical_id],
            source_refs=[source_ref],
            safe_summary=f"地点已解析为：{summary}",
        )

    async def _place_facts(self, request: PlaceFactsRequest) -> ExecutedToolObservation:
        facts: list[ToolFact] = []
        sources: list[str] = []
        missing: list[str] = []
        for canonical_id in request.canonical_entity_ids:
            resolved = self._resolved.get(canonical_id)
            if resolved is None:
                missing.append(canonical_id)
                continue
            place = resolved.place
            source_ref = f"{place.provider.value}:place:{place.source_place_id}"
            sources.append(source_ref)
            facts.append(
                ToolFact(
                    fact_kind="place_facts",
                    subject_ref=canonical_id,
                    value_summary=_place_summary(place),
                    source_reference_ids=[source_ref],
                    observed_at=place.fetched_at,
                    expires_at=place.fetched_at + timedelta(days=30),
                )
            )
        return self._facts_result(
            request,
            provider="amap",
            facts=facts,
            entity_refs=[item.subject_ref for item in facts],
            source_refs=sources,
            limitations=[f"未找到本轮解析记录：{item}" for item in missing],
            summary="已取得地点基础事实。",
        )

    async def _opening_hours(self, request: OpeningHoursRequest) -> ExecutedToolObservation:
        assert self._hours is not None
        facts: list[ToolFact] = []
        sources: list[str] = []
        limitations: list[str] = []
        for canonical_id in request.canonical_entity_ids:
            resolved = self._resolved.get(canonical_id)
            if resolved is None:
                limitations.append(f"地点未在本轮完成解析：{canonical_id}")
                continue
            place = resolved.place
            city = self._cities.provider_scope(place.city_id, ProviderCode.AMAP)
            response = await self._hours.get_regular_hours(
                HoursRequest(
                    place_id=UUID(canonical_id),
                    city=city,
                    name=place.name,
                    address=place.address,
                    coordinates=place.coordinates,
                    source_place_ids={place.provider: place.source_place_id},
                    service_dates=request.service_dates,
                )
            )
            if response.status is ProviderResultStatus.EMPTY or not response.items:
                limitations.append(f"{place.name} 暂无可验证营业时间")
                continue
            item = response.items[0]
            source_ref = f"{item.provider.value}:hours:{item.source_place_id}"
            sources.append(source_ref)
            days = evaluate_regular_hours(item, request.service_dates)
            text = "；".join(describe_date_hours(day) for day in days)
            limitations.extend(
                f"{place.name} {describe_date_hours(day)}"
                for day in days
                if day.status in (HoursDayStatus.UNKNOWN, HoursDayStatus.CONFLICT)
            )
            local_observed = item.fetched_at.astimezone(ZoneInfo("Asia/Shanghai"))
            expiry = (local_observed + timedelta(days=1)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            facts.append(
                ToolFact(
                    fact_kind="opening_hours",
                    subject_ref=canonical_id,
                    value_summary=f"{place.name}：{text}",
                    source_reference_ids=[source_ref],
                    observed_at=item.fetched_at,
                    expires_at=expiry,
                )
            )
        return self._facts_result(
            request,
            provider="amap",
            facts=facts,
            entity_refs=[item.subject_ref for item in facts],
            source_refs=sources,
            limitations=limitations,
            summary="已按实际日期核验高德营业规则；未知或冲突日期不能确认开放。",
        )

    async def _ticket_availability(
        self,
        request: TicketAvailabilityRequest,
        semantic_state: TripSemanticState,
    ) -> ExecutedToolObservation:
        synthetic = PlaceProductsRequest(
            request_id=request.request_id,
            depends_on=request.depends_on,
            purpose=request.purpose,
            required=request.required,
            capability=PrepareToolCapability.PLACE_PRODUCTS,
            canonical_entity_ids=request.canonical_entity_ids,
            service_dates=request.service_dates,
        )
        result = await self._place_products(synthetic, semantic_state)
        return ExecutedToolObservation(
            result.observation.model_copy(
                update={"capability": PrepareToolCapability.TICKET_AVAILABILITY}
            ),
            result.provider,
        )

    async def _weather_forecast(
        self,
        request: WeatherForecastRequest,
        semantic_state: TripSemanticState,
    ) -> ExecutedToolObservation:
        assert self._weather is not None
        basics = semantic_state.trip_basics
        city_ref = basics.destination_canonical_id or basics.destination_name
        if city_ref is None or not request.service_dates:
            return self._invalid(request, "天气查询缺少目的地或日期。")
        scope = self._cities.provider_scope(city_ref, ProviderCode.WEATHER)
        response = await self._weather.get_forecast(
            WeatherRequest(
                city=scope,
                start_date=min(request.service_dates),
                end_date=max(request.service_dates),
            )
        )
        facts: list[ToolFact] = []
        sources: list[str] = []
        for item in response.items:
            source_ref = f"{item.provider.value}:weather:{item.forecast_date.isoformat()}"
            sources.append(source_ref)
            facts.append(
                ToolFact(
                    fact_kind="weather_forecast",
                    subject_ref=item.forecast_date.isoformat(),
                    value_summary=(
                        ("长期趋势（非临近预报）· " if item.forecast_kind == "outlook" else "")
                        + f"{item.forecast_date.isoformat()}："
                        f"{item.condition_day or '天气待补充'}，"
                        f"{item.low_celsius if item.low_celsius is not None else '?'}～"
                        f"{item.high_celsius if item.high_celsius is not None else '?'}℃"
                    ),
                    source_reference_ids=[source_ref],
                    observed_at=item.fetched_at,
                    expires_at=item.fetched_at + timedelta(hours=6),
                )
            )
        return self._facts_result(
            request,
            provider="weather",
            facts=facts,
            entity_refs=[],
            source_refs=sources,
            limitations=list(response.missing_fields),
            summary="已查询旅行日期天气。",
        )

    async def _spatial_routes(
        self,
        request: SpatialRoutesRequest,
        semantic_state: TripSemanticState,
    ) -> ExecutedToolObservation:
        del semantic_state
        assert self._routes is not None
        origin = self._resolved.get(request.origin_ref)
        destination = self._resolved.get(request.destination_ref)
        if origin is None or destination is None:
            return self._invalid(request, "路线查询的起终点尚未完成地点解析。")
        scope = self._cities.provider_scope(origin.place.city_id, ProviderCode.AMAP)
        mode_map = {
            "public_transit": RouteMode.TRANSIT,
            "taxi": RouteMode.DRIVING,
            "walking": RouteMode.WALKING,
            "driving": RouteMode.DRIVING,
        }
        modes = list(dict.fromkeys(mode_map[item] for item in request.transport_modes))
        response = await self._routes.get_routes(
            RouteRequest(
                city=scope,
                origin=origin.place.coordinates,
                destination=destination.place.coordinates,
                modes=modes,
            )
        )
        facts: list[ToolFact] = []
        sources: list[str] = []
        for item in response.items:
            source_ref = f"{item.provider.value}:route:{item.mode.value}:{item.source_route_index}"
            sources.append(source_ref)
            facts.append(
                ToolFact(
                    fact_kind="spatial_route",
                    subject_ref=f"{request.origin_ref}->{request.destination_ref}",
                    value_summary=(
                        f"{item.mode.value}：约 {item.distance_m} 米，"
                        f"{round(item.duration_seconds / 60)} 分钟"
                    ),
                    source_reference_ids=[source_ref],
                    observed_at=item.fetched_at,
                    expires_at=item.fetched_at + timedelta(minutes=15),
                )
            )
        return self._facts_result(
            request,
            provider="amap",
            facts=facts,
            entity_refs=[request.origin_ref, request.destination_ref],
            source_refs=sources,
            limitations=list(response.missing_fields),
            summary="已比较地点之间的路线。",
        )

    async def _hotel_booking_facts(
        self,
        request: HotelBookingFactsRequest,
        semantic_state: TripSemanticState,
    ) -> ExecutedToolObservation:
        assert self._products is not None
        basics = semantic_state.trip_basics
        if basics.start_date is None or basics.end_date is None:
            return self._invalid(request, "酒店核验需要旅行日期。")
        scope = self._cities.provider_scope(request.city, ProviderCode.FLYAI)
        response = await self._products.search_hotels(
            HotelSearchRequest(
                city=scope,
                check_in=basics.start_date,
                check_out=basics.end_date,
                query=request.user_booking_ref.removeprefix("booking:"),
            )
        )
        facts: list[ToolFact] = []
        sources: list[str] = []
        for item in response.items[:3]:
            source_ref = f"{item.provider.value}:hotel:{item.source_hotel_id}"
            sources.append(source_ref)
            facts.append(
                ToolFact(
                    fact_kind="hotel_booking_match",
                    subject_ref=item.source_hotel_id,
                    value_summary=f"{item.name}，{item.address or '地址待补充'}",
                    source_reference_ids=[source_ref],
                    observed_at=item.fetched_at,
                    expires_at=item.fetched_at + timedelta(minutes=30),
                )
            )
        return self._facts_result(
            request,
            provider="flyai",
            facts=facts,
            entity_refs=[],
            source_refs=sources,
            limitations=list(response.missing_fields),
            summary="已核对可能匹配的酒店信息。",
        )

    async def _place_products(
        self,
        request: PlaceProductsRequest,
        semantic_state: TripSemanticState,
    ) -> ExecutedToolObservation:
        del semantic_state
        assert self._products is not None
        facts: list[ToolFact] = []
        sources: list[str] = []
        limitations: list[str] = []
        for canonical_id in request.canonical_entity_ids:
            resolved = self._resolved.get(canonical_id)
            if resolved is None:
                limitations.append(f"地点未在本轮完成解析：{canonical_id}")
                continue
            scope = self._cities.provider_scope(resolved.place.city_id, ProviderCode.FLYAI)
            visit_date = request.service_dates[0] if request.service_dates else None
            response = await self._products.search_place_products(
                ProductSearchRequest(
                    city=scope,
                    visit_date=visit_date,
                    query=resolved.place.name,
                )
            )
            for item in response.items[:3]:
                source_ref = f"{item.provider.value}:product:{item.source_offer_id}"
                sources.append(source_ref)
                price = item.price.model_dump(mode="json") if item.price is not None else None
                facts.append(
                    ToolFact(
                        fact_kind="place_product",
                        subject_ref=canonical_id,
                        value_summary=(
                            f"{item.name}，价格信息：{json.dumps(price, ensure_ascii=False)}"
                        ),
                        source_reference_ids=[source_ref],
                        observed_at=item.fetched_at,
                        expires_at=item.fetched_at + timedelta(minutes=30),
                    )
                )
        return self._facts_result(
            request,
            provider="flyai",
            facts=facts,
            entity_refs=[item.subject_ref for item in facts],
            source_refs=sources,
            limitations=limitations,
            summary="已查询地点相关产品信息。",
        )

    def _facts_result(
        self,
        request: ToolRequestValue,
        *,
        provider: str,
        facts: list[ToolFact],
        entity_refs: list[str],
        source_refs: list[str],
        limitations: list[str],
        summary: str,
    ) -> ExecutedToolObservation:
        if facts and limitations:
            return ExecutedToolObservation(
                ToolObservation(
                    request_id=request.request_id,
                    capability=request.capability,
                    status=ToolObservationStatus.PARTIAL,
                    facts=facts,
                    entity_refs=list(dict.fromkeys(entity_refs)),
                    source_refs=list(dict.fromkeys(source_refs)),
                    observed_at=self._clock(),
                    safe_summary=summary,
                    limitations=limitations,
                ),
                provider,
            )
        if facts:
            return self._success(
                request,
                provider=provider,
                facts=facts,
                entity_refs=entity_refs,
                source_refs=source_refs,
                safe_summary=summary,
            )
        return self._unavailable(
            request,
            limitations[0] if limitations else "没有取得可验证的事实。",
            provider,
        )

    def _success(
        self,
        request: ToolRequestValue,
        *,
        provider: str,
        facts: list[ToolFact],
        entity_refs: list[str],
        source_refs: list[str],
        safe_summary: str,
    ) -> ExecutedToolObservation:
        return ExecutedToolObservation(
            ToolObservation(
                request_id=request.request_id,
                capability=request.capability,
                status=ToolObservationStatus.SUCCESS,
                facts=facts,
                entity_refs=list(dict.fromkeys(entity_refs)),
                source_refs=list(dict.fromkeys(source_refs)),
                observed_at=self._clock(),
                safe_summary=safe_summary,
            ),
            provider,
        )

    def _unavailable(
        self, request: ToolRequestValue, limitation: str, provider: str = "unavailable"
    ) -> ExecutedToolObservation:
        return ExecutedToolObservation(
            ToolObservation(
                request_id=request.request_id,
                capability=request.capability,
                status=ToolObservationStatus.UNAVAILABLE,
                observed_at=self._clock(),
                safe_summary="工具没有返回可验证结果。",
                limitations=[limitation],
            ),
            provider,
        )

    def _invalid(self, request: ToolRequestValue, limitation: str) -> ExecutedToolObservation:
        return ExecutedToolObservation(
            ToolObservation(
                request_id=request.request_id,
                capability=request.capability,
                status=ToolObservationStatus.INVALID_REQUEST,
                observed_at=self._clock(),
                safe_summary="工具请求无法安全执行。",
                limitations=[limitation],
            ),
            "invalid_request",
        )


def _best_place_match(query: str, values: list[ProviderPlace]) -> ProviderPlace:
    normalized = _normalize_name(query)
    return min(
        values,
        key=lambda item: (
            0 if _normalize_name(item.name) == normalized else 1,
            0 if normalized in _normalize_name(item.name) else 1,
            len(item.name),
            item.source_place_id,
        ),
    )


def _normalize_name(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def _place_summary(place: ProviderPlace) -> str:
    address = f"，地址：{place.address}" if place.address else ""
    return f"{place.name}（{place.category.value}{address}）"


__all__ = ["ExecutedToolObservation", "PrepareToolExecutor"]
