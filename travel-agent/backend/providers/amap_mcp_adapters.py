"""Normalize official MCP search/routes into the existing Provider evidence contracts."""

from __future__ import annotations

import asyncio
from typing import Any

from backend.contracts.enums import PlaceCategory
from backend.providers.amap_mcp import McpGateway, _error
from backend.providers.amap_places import AmapPlaceProvider
from backend.providers.amap_routes import (
    AmapRouteProvider,
    _format_location,
    _required_nonnegative_int,
)
from backend.providers.contracts import (
    GeocodedAddress,
    GeocodeRequest,
    KeywordPlaceSearchRequest,
    NearbyPlaceSearchRequest,
    PlaceDetailRequest,
    PolygonPlaceSearchRequest,
    ProviderError,
    ProviderFailureCode,
    ProviderPlace,
    ProviderResponse,
    ProviderResultStatus,
    ProviderRoute,
    RouteMode,
    RouteRequest,
)
from backend.providers.request_budget import RequestBudgetExceeded


class AmapMcpPlaceProvider(AmapPlaceProvider):
    def __init__(self, mcp: McpGateway) -> None:
        super().__init__("mcp-transport-only", client=mcp._client)
        self.mcp = mcp

    async def search_places(
        self, request: KeywordPlaceSearchRequest
    ) -> ProviderResponse[ProviderPlace]:
        data = await self.mcp.call(
            "maps_text_search",
            {"keywords": request.query, "city": request.city.provider_city_code, "citylimit": True},
        )
        return await self._normalize(data, request.city.city_id, request.category_hint)

    async def search_nearby(
        self, request: NearbyPlaceSearchRequest
    ) -> ProviderResponse[ProviderPlace]:
        data = await self.mcp.call(
            "maps_around_search",
            {
                "keywords": request.query,
                "location": _format_location(request.center),
                "radius": str(request.radius_m),
            },
        )
        return await self._normalize(data, request.city.city_id, request.category_hint)

    async def get_place(self, request: PlaceDetailRequest) -> ProviderResponse[ProviderPlace]:
        data = await self.mcp.call("maps_search_detail", {"id": request.source_place_id})
        return await self._normalize(
            data, request.city.city_id, request.category_hint, hydrate=False
        )

    async def search_polygon(
        self, request: PolygonPlaceSearchRequest
    ) -> ProviderResponse[ProviderPlace]:
        data = await self.mcp.call(
            "maps_polygon_search",
            {
                "polygon": "|".join(_format_location(point) for point in request.polygon),
                "types": request.typecodes[0],
                "page_num": request.page,
                "page_size": request.page_size,
                **({"keywords": request.query} if request.query else {}),
            },
        )
        return await self._normalize(data, request.city.city_id, request.category_hint)

    async def geocode(self, request: GeocodeRequest) -> ProviderResponse[GeocodedAddress]:
        raise _error("mcp_geocode_not_exposed", ProviderFailureCode.INVALID_REQUEST)

    async def _normalize(
        self,
        data: dict[str, Any],
        city_id: str,
        kind: PlaceCategory | None,
        *,
        hydrate: bool = True,
    ) -> ProviderResponse[ProviderPlace]:
        records = data.get("pois")
        if records is None:
            records = [data] if data.get("id") else data.get("results")
        if not isinstance(records, list):
            raise _error("mcp_places_shape", ProviderFailureCode.MALFORMED_RESPONSE)
        semaphore = asyncio.Semaphore(4)

        async def verify(record: object) -> dict[str, Any]:
            if not isinstance(record, dict) or not record.get("id"):
                raise _error("mcp_place_identity", ProviderFailureCode.MALFORMED_RESPONSE)
            if hydrate and any(
                not record.get(field) for field in ("location", "adcode", "typecode")
            ):
                async with semaphore:
                    detail = await self.mcp.call("maps_search_detail", {"id": record["id"]})
                matches = detail.get("pois", [detail])
                if not isinstance(matches, list):
                    raise _error("mcp_place_detail", ProviderFailureCode.MALFORMED_RESPONSE)
                match = next(
                    (p for p in matches if isinstance(p, dict) and p.get("id") == record["id"]),
                    None,
                )
                if match is None:
                    raise _error("mcp_place_identity", ProviderFailureCode.MALFORMED_RESPONSE)
                record = {**record, **match}
            city_code = city_id.removeprefix("cn-")
            prefix = city_code[:2] if city_code[:2] in {"11", "12", "31", "50"} else city_code[:4]
            if not str(record.get("adcode", "")).startswith(prefix) or not record.get("typecode"):
                raise _error(
                    "mcp_place_city_or_type_unverified", ProviderFailureCode.MALFORMED_RESPONSE
                )
            return dict(record)

        # The search tool promises verified candidates: sparse search rows are hydrated by
        # exact POI id. Every detail request passes through the same I/O budget.
        async def one(record: object) -> dict[str, Any] | ProviderError | RequestBudgetExceeded:
            try:
                return await verify(record)
            except (ProviderError, RequestBudgetExceeded) as error:
                return error

        tasks = [asyncio.create_task(one(record)) for record in records[:20]]
        try:
            verified = await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        valid = [row for row in verified if isinstance(row, dict)]
        failures = [row for row in verified if not isinstance(row, dict)]
        if failures and not valid:
            raise failures[0]
        parsed = self._parse_places({"pois": valid}, city_id, kind)
        if failures:
            parsed = parsed.model_copy(
                update={
                    "status": ProviderResultStatus.PARTIAL,
                    "missing_fields": [*parsed.missing_fields, "items.unverified_search_rows"],
                    "provider_notice": "Some POIs could not be verified and were excluded.",
                }
            )
        return parsed


class AmapMcpRouteProvider(AmapRouteProvider):
    def __init__(self, mcp: McpGateway) -> None:
        super().__init__("mcp-transport-only", client=mcp._client)
        self.mcp = mcp

    async def _get_mode_routes(
        self, request: RouteRequest, mode: RouteMode
    ) -> ProviderResponse[ProviderRoute]:
        names = {
            RouteMode.WALKING: "maps_direction_walking",
            RouteMode.DRIVING: "maps_direction_driving",
            RouteMode.TRANSIT: "maps_direction_transit_integrated",
            RouteMode.CYCLING: "maps_direction_bicycling",
        }
        args = {
            "origin": _format_location(request.origin),
            "destination": _format_location(request.destination),
        }
        if mode is RouteMode.TRANSIT:
            assert request.city.provider_transit_city_code is not None
            args.update(
                city=request.city.provider_transit_city_code,
                cityd=request.city.provider_transit_city_code,
            )
        data = await self.mcp.call(names[mode], args)
        return self.normalize_result(data, mode)

    def normalize_result(
        self, data: dict[str, Any], mode: RouteMode
    ) -> ProviderResponse[ProviderRoute]:
        """Parse a native Agent query without issuing the same request again."""
        route = data.get("route", data)
        if not isinstance(route, dict):
            raise _error("mcp_route_shape", ProviderFailureCode.MALFORMED_RESPONSE)
        collection = "transits" if mode is RouteMode.TRANSIT else "paths"
        records = route.get(collection)
        if not isinstance(records, list):
            raise _error("mcp_route_shape", ProviderFailureCode.MALFORMED_RESPONSE)
        if not records:
            return self._parse_mode({"route": route}, mode, "mcp_route")
        responses = []
        failures = []
        for index, record in enumerate(records):
            try:
                if not isinstance(record, dict):
                    raise _error("mcp_route_shape", ProviderFailureCode.MALFORMED_RESPONSE)
                cost = record.get("cost")
                if not isinstance(cost, dict):
                    cost = {"duration": record.get("duration"), "transit_fee": cost}
                entry = {**record, "cost": cost}
                if mode is RouteMode.TRANSIT:
                    # Each alternative owns its distance. Ambiguous alternatives
                    # must not erase another independently valid real route.
                    entry = _normalize_transit(entry)
                parsed = self._parse_mode(
                    {"route": {**route, collection: [entry]}}, mode, "mcp_route"
                )
                responses.append(
                    parsed.model_copy(
                        update={
                            "items": [
                                item.model_copy(update={"source_route_index": index})
                                for item in parsed.items
                            ]
                        }
                    )
                )
            except ProviderError as error:
                failures.append(error)
        if not responses:
            raise failures[0]
        missing = {field for response in responses for field in response.missing_fields}
        if failures:
            missing.add("items.unparsed_route_alternatives")
        return responses[0].model_copy(
            update={
                "items": [item for response in responses for item in response.items],
                "status": ProviderResultStatus.PARTIAL if missing else ProviderResultStatus.SUCCESS,
                "missing_fields": sorted(missing),
                "provider_notice": "Unverified route alternatives were excluded."
                if failures
                else None,
            }
        )


def _normalize_transit(record: dict[str, Any]) -> dict[str, Any]:
    segments = record.get("segments")
    if not isinstance(segments, list):
        raise _error("mcp_transit_segments", ProviderFailureCode.MALFORMED_RESPONSE)
    normalized = []
    total, found = 0, False
    for raw in segments:
        if not isinstance(raw, dict):
            raise _error("mcp_transit_segments", ProviderFailureCode.MALFORMED_RESPONSE)
        segment = dict(raw)
        for name in ("walking", "railway", "taxi"):
            part = segment.get(name)
            if isinstance(part, dict) and not any(v not in (None, "", []) for v in part.values()):
                # MCP projects empty railway objects to {name: '', trip: ''};
                # they are not real rides and must not inflate transfer counts.
                segment.pop(name)
                continue
            if part not in (None, "", []):
                if not isinstance(part, dict):
                    raise _error("mcp_transit_segment", ProviderFailureCode.MALFORMED_RESPONSE)
                if record.get("distance") in (None, "", []):
                    total += _required_nonnegative_int(part.get("distance"), "mcp_transit_distance")
                    found = True
        bus = segment.get("bus")
        if bus not in (None, "", []):
            if not isinstance(bus, dict) or not isinstance(bus.get("buslines"), list):
                raise _error("mcp_transit_bus", ProviderFailureCode.MALFORMED_RESPONSE)
            lines = bus["buslines"]
            if lines and record.get("distance") in (None, "", []):
                distances = {
                    _required_nonnegative_int(line.get("distance"), "mcp_transit_distance")
                    for line in lines
                    if isinstance(line, dict)
                }
                # Alternatives of different lengths cannot be added together or
                # assigned the root distance. Retain an explicit parse failure.
                if len(distances) != 1 or any(not isinstance(line, dict) for line in lines):
                    raise _error("mcp_transit_distance", ProviderFailureCode.MALFORMED_RESPONSE)
                total += next(iter(distances))
                found = True
        normalized.append(segment)
    if record.get("distance") in (None, "", []):
        if not found:
            raise _error("mcp_transit_distance", ProviderFailureCode.MALFORMED_RESPONSE)
        record = {**record, "distance": total}
    return {**record, "segments": normalized}
