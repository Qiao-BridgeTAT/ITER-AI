"""AMap POI 2.0 business hours, bound to an exact POI and destination dates."""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo

import httpx

from backend.config.settings import AmapSearchProxySettings
from backend.contracts.enums import ProviderCode
from backend.providers.amap_http import AMAP_BASE_URL, amap_malformed, request_amap_json
from backend.providers.contracts import (
    HoursDayStatus,
    HoursRequest,
    ProviderRegularHours,
    ProviderResponse,
    ProviderResultStatus,
)
from backend.providers.hours_rules import evaluate_regular_hours

AMAP_HOURS_DETAIL_PATH = "/v5/place/detail"
AMAP_HOURS_SEARCH_PATH = "/v5/place/text"
DESTINATION_TIMEZONE = ZoneInfo("Asia/Shanghai")


class AmapHoursProvider:
    def __init__(
        self,
        api_key: str,
        *,
        client: httpx.AsyncClient | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        timeout_seconds: float = 10.0,
        search_proxy: AmapSearchProxySettings | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("AMAP_WEB_SERVICE_KEY must not be empty")
        self._api_key = api_key
        self._search_proxy = search_proxy
        self._client = client or httpx.AsyncClient(base_url=AMAP_BASE_URL, timeout=timeout_seconds)
        self._owns_client = client is None
        self._clock = clock

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def get_regular_hours(
        self, request: HoursRequest
    ) -> ProviderResponse[ProviderRegularHours]:
        source_id = request.source_place_ids.get(ProviderCode.AMAP)
        if source_id is None:
            payload = await self._get(
                AMAP_HOURS_SEARCH_PATH,
                {
                    "keywords": request.name,
                    "region": request.city.provider_city_code,
                    "city_limit": "true",
                    "page_size": "10",
                    "show_fields": "business",
                },
            )
            matches = [record for record in _records(payload) if _same_place(record, request)]
            if len(matches) != 1:
                return self._empty()
            source_id = _text(matches[0].get("id"))
            if source_id is None:
                raise amap_malformed("regular_hours")
        payload = await self._get(
            AMAP_HOURS_DETAIL_PATH, {"id": source_id, "show_fields": "business"}
        )
        matches = [record for record in _records(payload) if record.get("id") == source_id]
        if len(matches) != 1 or not _same_place(matches[0], request):
            return self._empty()
        business = matches[0].get("business")
        if not isinstance(business, dict):
            business = {}
        weekly = _text(business.get("opentime_week"))
        today = _text(business.get("opentime_today"))
        fetched_at = self._now()
        local_date = fetched_at.astimezone(DESTINATION_TIMEZONE).date()
        missing = weekly is None and today is None
        regular = ProviderRegularHours(
            provider=ProviderCode.AMAP,
            source_place_id=source_id,
            display_text=weekly
            or (f"仅{local_date.isoformat()}今日字段：{today}" if today else None),
            weekly_text=weekly,
            today_text=today,
            today_date=local_date if today else None,
            fetched_at=fetched_at,
            missing_reason="高德未提供可用的 business 营业时间字段。" if missing else None,
            # Never carry unrelated business data (including phone numbers) upstream.
            raw_payload={},
        )
        date_hours = evaluate_regular_hours(regular, request.service_dates or [local_date])
        regular = regular.model_copy(update={"date_hours": date_hours})
        unresolved = any(
            item.status in (HoursDayStatus.UNKNOWN, HoursDayStatus.CONFLICT) for item in date_hours
        )
        missing_fields = (["items.business_hours"] if missing else []) + (
            ["items.date_hours"] if unresolved else []
        )
        return ProviderResponse[ProviderRegularHours](
            provider=ProviderCode.AMAP,
            status=ProviderResultStatus.PARTIAL if missing_fields else ProviderResultStatus.SUCCESS,
            items=[regular],
            fetched_at=fetched_at,
            missing_fields=missing_fields,
        )

    async def _get(self, path: str, parameters: dict[str, str]) -> dict[str, Any]:
        return await request_amap_json(
            self._client,
            self._api_key,
            path,
            parameters,
            "regular_hours",
            search_proxy=self._search_proxy,
        )

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None:
            raise ValueError("hours clock must be timezone-aware")
        return value

    def _empty(self) -> ProviderResponse[ProviderRegularHours]:
        return ProviderResponse[ProviderRegularHours](
            provider=ProviderCode.AMAP,
            status=ProviderResultStatus.EMPTY,
            fetched_at=self._now(),
        )


def _records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    records = payload.get("pois")
    if not isinstance(records, list) or any(not isinstance(record, dict) for record in records):
        raise amap_malformed("regular_hours")
    return records


def _text(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip() if len(value) <= 4000 else None


def _normal(value: str) -> str:
    return "".join(unicodedata.normalize("NFKC", value).casefold().split())


def _same_place(record: dict[str, Any], request: HoursRequest) -> bool:
    name, city = _text(record.get("name")), _text(record.get("cityname"))
    if not name or _normal(name) != _normal(request.name):
        return False
    expected_city = request.city.display_name
    if expected_city and (
        not city or _normal(city).removesuffix("市") != _normal(expected_city).removesuffix("市")
    ):
        return False
    if not expected_city and record.get("adcode") != request.city.provider_city_code:
        return False
    location = record.get("location")
    if not isinstance(location, str):
        return False
    try:
        lon, lat = (float(value) for value in location.split(","))
    except ValueError:
        return False
    if not (math.isfinite(lon) and math.isfinite(lat) and -180 <= lon <= 180 and -90 <= lat <= 90):
        return False
    lat1, lat2 = math.radians(lat), math.radians(request.coordinates.latitude)
    dlat = lat1 - lat2
    dlon = math.radians(lon - request.coordinates.longitude)
    distance = (
        2
        * 6_371_000
        * math.asin(
            min(
                1.0,
                math.sqrt(
                    math.sin(dlat / 2) ** 2
                    + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
                ),
            )
        )
    )
    return distance <= 350
