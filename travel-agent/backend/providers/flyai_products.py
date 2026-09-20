"""FlyAI CLI transport and normalized Fliggy hotel/ticket product adapter."""

from __future__ import annotations

import asyncio
import json
import os
import re
import tempfile
from collections.abc import Callable, Sequence
from contextlib import suppress
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Literal, Protocol, cast
from urllib.parse import urlsplit
from uuid import UUID

from pydantic import HttpUrl, TypeAdapter, ValidationError

from backend.contracts.common import CnyAmountRange
from backend.contracts.enums import CoordinateSystem, DataAvailability, ProviderCode
from backend.contracts.places import Coordinates, HotelOffer, PlaceSourceMapping, TicketOffer
from backend.providers.contracts import (
    HotelSearchRequest,
    ProductSearchRequest,
    ProviderError,
    ProviderFailureCode,
    ProviderHotelOffer,
    ProviderResponse,
    ProviderResultStatus,
    ProviderTicketOffer,
)
from backend.providers.request_budget import budgeted_external_request

FLYAI_MAX_OUTPUT_BYTES = 2 * 1024 * 1024
FLYAI_PRICE_NOTICE = "价格和可售情况仅代表飞猪平台本次查询结果，以详情页实时展示为准。"
FLYAI_DETAIL_HOST_SUFFIXES = (
    ".fliggy.com",
    ".alitrip.com",
    ".taobao.com",
    ".tmall.com",
    ".tb.cn",
    ".fly.ai",
)
FLYAI_IMAGE_HOST_SUFFIXES = (
    ".alicdn.com",
    ".fliggy.com",
    ".taobao.com",
    ".tmall.com",
)
_HTTP_URL_ADAPTER = TypeAdapter(HttpUrl)
HotelMissingField = Literal["room_price", "rating", "image_urls", "detail_url"]
TicketMissingField = Literal["price", "detail_url"]


class FlyAiTransport(Protocol):
    async def execute(self, command: str, arguments: Sequence[str]) -> dict[str, Any]: ...


class FlyAiCliTransport:
    """Invoke the official CLI without putting the API key in process arguments."""

    def __init__(
        self,
        api_key: str,
        *,
        executable: str = "flyai",
        timeout_seconds: float = 12.0,
    ) -> None:
        if not api_key.strip():
            raise ValueError("FLYAI_API_KEY must not be empty")
        self._api_key = api_key
        self._executable = executable
        self._timeout_seconds = timeout_seconds

    @budgeted_external_request
    async def execute(self, command: str, arguments: Sequence[str]) -> dict[str, Any]:
        environment = os.environ.copy()
        environment["FLYAI_API_KEY"] = self._api_key
        # The official Node CLI calls process.exit immediately after console.log.
        # On POSIX pipes (notably uvloop's subprocess socket), pending writes can
        # be lost at 8192 bytes. Regular-file stdout is synchronous in Node.
        # TemporaryFile is private and unlinked on POSIX, never an audit artifact;
        # bounded reads and context cleanup apply on success, timeout and cancel.
        with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as errors:
            try:
                process = await asyncio.create_subprocess_exec(
                    self._executable,
                    command,
                    *arguments,
                    stdout=output,
                    stderr=errors,
                    env=environment,
                )
            except FileNotFoundError:
                raise ProviderError(
                    ProviderCode.FLYAI,
                    ProviderFailureCode.UNAVAILABLE,
                    command,
                    retryable=False,
                ) from None
            try:
                async with asyncio.timeout(self._timeout_seconds):
                    await process.wait()
            except TimeoutError:
                raise ProviderError(
                    ProviderCode.FLYAI,
                    ProviderFailureCode.TIMEOUT,
                    command,
                    retryable=True,
                ) from None
            finally:
                if process.returncode is None:
                    with suppress(ProcessLookupError):
                        process.kill()
                    await process.wait()
            output.seek(0)
            errors.seek(0)
            stdout = output.read(FLYAI_MAX_OUTPUT_BYTES + 1)
            stderr = errors.read(2_048)

        if process.returncode != 0:
            raise _flyai_process_error(command, stderr)
        if len(stdout) > FLYAI_MAX_OUTPUT_BYTES:
            raise _flyai_malformed(command)
        try:
            payload = json.loads(stdout)
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise _flyai_malformed(command) from None
        if not isinstance(payload, dict):
            raise _flyai_malformed(command)
        return cast(dict[str, Any], payload)


class FlyAiProductProvider:
    """Expose the documented FlyAI hotel and attraction-search subset."""

    def __init__(
        self,
        transport: FlyAiTransport,
        *,
        source_coordinate_system: CoordinateSystem | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._transport = transport
        self._source_coordinate_system = source_coordinate_system
        self._clock = clock

    async def search_hotels(
        self, request: HotelSearchRequest
    ) -> ProviderResponse[ProviderHotelOffer]:
        city_name = _required_city_name(request.city.display_name, "search_hotel")
        arguments = [
            "--dest-name",
            city_name,
            "--check-in-date",
            request.check_in.isoformat(),
            "--check-out-date",
            request.check_out.isoformat(),
        ]
        if request.query is not None:
            arguments.extend(["--key-words", request.query])
        if request.anchor_name is not None:
            arguments.extend(["--poi-name", request.anchor_name])
        if request.hotel_stars:
            arguments.extend(["--hotel-stars", ",".join(str(star) for star in request.hotel_stars)])
        if request.hotel_types:
            arguments.extend(["--hotel-types", ",".join(request.hotel_types)])
        if request.sort:
            arguments.extend(["--sort", request.sort])
        payload = await self._transport.execute("search-hotel", arguments)
        return self._parse_hotels(payload, request)

    async def search_place_products(
        self, request: ProductSearchRequest
    ) -> ProviderResponse[ProviderTicketOffer]:
        city_name = _required_city_name(request.city.display_name, "search_poi")
        if request.query is None:
            raise ProviderError(
                ProviderCode.FLYAI,
                ProviderFailureCode.INVALID_REQUEST,
                "search_poi",
                retryable=False,
            )
        payload = await self._transport.execute(
            "search-poi",
            ["--city-name", city_name, "--keyword", request.query],
        )
        return self._parse_tickets(payload, request)

    def _parse_hotels(
        self,
        payload: dict[str, Any],
        request: HotelSearchRequest,
    ) -> ProviderResponse[ProviderHotelOffer]:
        items, notice = _flyai_items(payload, "search_hotel")
        fetched_at = self._fetched_at()
        if not items:
            return ProviderResponse[ProviderHotelOffer](
                provider=ProviderCode.FLYAI,
                status=ProviderResultStatus.EMPTY,
                fetched_at=fetched_at,
                provider_notice=_flyai_notice(notice),
            )

        offers: list[ProviderHotelOffer] = []
        missing_fields: set[str] = set()
        for raw_item in items:
            item_missing_fields: set[str] = set()
            source_hotel_id = _required_text(raw_item.get("shId"), "search_hotel")
            name = _required_text(raw_item.get("name"), "search_hotel")
            hotel_type = _optional_text(raw_item.get("star") or raw_item.get("hotelType"))
            brand_name = _optional_text(raw_item.get("brandName"))
            price = _optional_cny_price(raw_item.get("price"))
            rating = _optional_rating(raw_item.get("score"))
            image = _safe_url(raw_item.get("mainPic"), FLYAI_IMAGE_HOST_SUFFIXES)
            detail_url = _safe_url(raw_item.get("detailUrl"), FLYAI_DETAIL_HOST_SUFFIXES)
            address = _optional_text(raw_item.get("address"))
            coordinates = _optional_source_coordinates(
                raw_item,
                self._source_coordinate_system,
            )
            if price is None:
                missing_fields.add("items.room_price")
                item_missing_fields.add("room_price")
            if rating is None:
                missing_fields.add("items.rating")
                item_missing_fields.add("rating")
            if image is None:
                missing_fields.add("items.image_urls")
                item_missing_fields.add("image_urls")
            if detail_url is None:
                missing_fields.add("items.detail_url")
                item_missing_fields.add("detail_url")
            if address is None:
                missing_fields.add("items.address")
                item_missing_fields.add("address")
            offers.append(
                ProviderHotelOffer(
                    provider=ProviderCode.FLYAI,
                    source_offer_id=source_hotel_id,
                    source_hotel_id=source_hotel_id,
                    name=name,
                    hotel_type=hotel_type,
                    brand_name=brand_name,
                    address=address,
                    raw_coordinates=coordinates,
                    room_price=price,
                    rating=rating,
                    image_urls=[image] if image is not None else [],
                    detail_url=detail_url,
                    missing_fields=sorted(item_missing_fields),
                    fetched_at=fetched_at,
                    raw_payload=raw_item,
                )
            )
        return ProviderResponse[ProviderHotelOffer](
            provider=ProviderCode.FLYAI,
            status=(
                ProviderResultStatus.PARTIAL if missing_fields else ProviderResultStatus.SUCCESS
            ),
            items=offers,
            fetched_at=fetched_at,
            missing_fields=sorted(missing_fields),
            provider_notice=_flyai_notice(notice),
        )

    def _parse_tickets(
        self,
        payload: dict[str, Any],
        request: ProductSearchRequest,
    ) -> ProviderResponse[ProviderTicketOffer]:
        items, notice = _flyai_items(payload, "search_poi")
        fetched_at = self._fetched_at()
        if not items:
            return ProviderResponse[ProviderTicketOffer](
                provider=ProviderCode.FLYAI,
                status=ProviderResultStatus.EMPTY,
                fetched_at=fetched_at,
                provider_notice=_flyai_notice(notice),
            )

        offers: list[ProviderTicketOffer] = []
        missing_fields: set[str] = set()
        for raw_item in items:
            item_missing_fields: set[str] = set()
            source_place_id = _required_text(raw_item.get("id"), "search_poi")
            place_name = _required_text(raw_item.get("name"), "search_poi")
            ticket_info_value = raw_item.get("ticketInfo")
            if ticket_info_value in (None, "", []):
                ticket_info: dict[str, Any] = {}
            elif isinstance(ticket_info_value, dict):
                ticket_info = cast(dict[str, Any], ticket_info_value)
            else:
                raise _flyai_malformed("search_poi")
            price_date = _optional_date(ticket_info.get("priceDate"))
            ticket_name = _optional_text(ticket_info.get("ticketName")) or place_name
            price = _optional_cny_price(ticket_info.get("price"))
            if (
                request.visit_date is not None
                and price_date is not None
                and price_date != request.visit_date
            ):
                price = None
                missing_fields.add("items.price_for_requested_date")
                item_missing_fields.add("price_for_requested_date")
            detail_url = _safe_url(raw_item.get("jumpUrl"), FLYAI_DETAIL_HOST_SUFFIXES)
            if price is None:
                missing_fields.add("items.price")
                item_missing_fields.add("price")
            if price_date is None:
                missing_fields.add("items.price_date")
                item_missing_fields.add("price_date")
            if detail_url is None:
                missing_fields.add("items.detail_url")
                item_missing_fields.add("detail_url")
            source_offer_id = f"{source_place_id}:{price_date or 'undated'}:{ticket_name}"
            offers.append(
                ProviderTicketOffer(
                    provider=ProviderCode.FLYAI,
                    source_offer_id=source_offer_id,
                    source_place_id=source_place_id,
                    name=ticket_name,
                    place_name=place_name,
                    admission_status=(
                        "free"
                        if raw_item.get("freePoiStatus") == "FREE"
                        else "paid"
                        if raw_item.get("freePoiStatus") == "NOT_FREE"
                        else "unknown"
                    ),
                    image_url=_safe_url(raw_item.get("mainPic"), FLYAI_IMAGE_HOST_SUFFIXES),
                    price_date=price_date,
                    price=price,
                    detail_url=detail_url,
                    missing_fields=sorted(item_missing_fields),
                    fetched_at=fetched_at,
                    raw_payload=raw_item,
                )
            )
        return ProviderResponse[ProviderTicketOffer](
            provider=ProviderCode.FLYAI,
            status=(
                ProviderResultStatus.PARTIAL if missing_fields else ProviderResultStatus.SUCCESS
            ),
            items=offers,
            fetched_at=fetched_at,
            missing_fields=sorted(missing_fields),
            provider_notice=_flyai_notice(notice),
        )

    def _fetched_at(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("provider clock must return an aware datetime")
        return value.astimezone(UTC)


def flyai_hotel_to_source_mapping(offer: ProviderHotelOffer) -> PlaceSourceMapping:
    if offer.provider is not ProviderCode.FLYAI:
        raise ValueError("only FlyAI hotels can use the FlyAI mapping helper")
    return PlaceSourceMapping(
        provider=ProviderCode.FLYAI,
        source_place_id=offer.source_hotel_id,
        raw_name=offer.name,
        raw_address=offer.address,
        raw_coordinates=offer.raw_coordinates,
        fetched_at=offer.fetched_at,
    )


def flyai_hotel_to_canonical_offer(
    offer: ProviderHotelOffer,
    *,
    offer_id: str,
    hotel_place_id: UUID,
    request: HotelSearchRequest,
) -> HotelOffer:
    canonical_missing: list[HotelMissingField] = []
    for field in offer.missing_fields:
        candidate = field
        if candidate in {"room_price", "rating", "image_urls", "detail_url"}:
            canonical_missing.append(cast(HotelMissingField, candidate))
    actual_missing: dict[HotelMissingField, bool] = {
        "room_price": offer.room_price is None,
        "rating": offer.rating is None,
        "image_urls": not offer.image_urls,
        "detail_url": offer.detail_url is None,
    }
    for field, is_missing in actual_missing.items():
        if is_missing and field not in canonical_missing:
            canonical_missing.append(field)
    has_supplemental_data = any(
        value is not None and value != []
        for value in (offer.rating, offer.image_urls, offer.detail_url)
    )
    if offer.room_price is not None and not canonical_missing:
        availability = DataAvailability.AVAILABLE
        missing_reason = None
    elif offer.room_price is not None or has_supplemental_data:
        availability = DataAvailability.PARTIAL
        missing_reason = "FlyAI did not return all hotel offer fields"
    else:
        availability = DataAvailability.MISSING
        missing_reason = "FlyAI did not return a usable hotel offer"
    return HotelOffer(
        offer_id=offer_id,
        hotel_place_id=hotel_place_id,
        provider=ProviderCode.FLYAI,
        source_offer_id=offer.source_offer_id,
        check_in=request.check_in,
        check_out=request.check_out,
        availability=availability,
        room_price=offer.room_price,
        rating=offer.rating,
        image_urls=offer.image_urls,
        detail_url=offer.detail_url,
        missing_fields=canonical_missing,
        missing_reason=missing_reason,
        raw_price=offer.raw_payload.get("price"),
        fetched_at=offer.fetched_at,
    )


def flyai_ticket_to_canonical_offer(
    offer: ProviderTicketOffer,
    *,
    offer_id: str,
    place_id: UUID,
) -> TicketOffer:
    canonical_missing: list[TicketMissingField] = []
    for field in offer.missing_fields:
        candidate = field
        if candidate in {"price", "detail_url"}:
            canonical_missing.append(cast(TicketMissingField, candidate))
    actual_missing: dict[TicketMissingField, bool] = {
        "price": offer.price is None,
        "detail_url": offer.detail_url is None,
    }
    for field, is_missing in actual_missing.items():
        if is_missing and field not in canonical_missing:
            canonical_missing.append(field)
    if offer.price is not None and not canonical_missing:
        availability = DataAvailability.AVAILABLE
        missing_reason = None
    elif offer.price is not None or offer.detail_url is not None:
        availability = DataAvailability.PARTIAL
        missing_reason = "FlyAI did not return all ticket offer fields"
    else:
        availability = DataAvailability.MISSING
        missing_reason = "FlyAI did not return a usable ticket offer"
    ticket_info = offer.raw_payload.get("ticketInfo")
    raw_price = ticket_info.get("price") if isinstance(ticket_info, dict) else None
    return TicketOffer(
        offer_id=offer_id,
        place_id=place_id,
        provider=ProviderCode.FLYAI,
        source_offer_id=offer.source_offer_id,
        visit_date=offer.price_date,
        availability=availability,
        price=offer.price,
        detail_url=offer.detail_url,
        missing_fields=canonical_missing,
        missing_reason=missing_reason,
        raw_price=raw_price,
        fetched_at=offer.fetched_at,
    )


def _flyai_items(
    payload: dict[str, Any], operation: str
) -> tuple[list[dict[str, Any]], str | None]:
    status = payload.get("status")
    if status != 0:
        raise _flyai_payload_error(payload, operation)
    data = payload.get("data")
    # The live search endpoint also represents an empty match as status=0,
    # data=null. This is absence of results, not an invented offer or schema retry.
    if data is None and "data" in payload:
        return [], _optional_text(payload.get("systemMessage"))
    if not isinstance(data, dict):
        raise _flyai_malformed(operation)
    item_list = data.get("itemList")
    if not isinstance(item_list, list):
        raise _flyai_malformed(operation)
    items: list[dict[str, Any]] = []
    for item in item_list:
        if not isinstance(item, dict):
            raise _flyai_malformed(operation)
        items.append(cast(dict[str, Any], item))
    return items, _optional_text(payload.get("systemMessage"))


def _flyai_payload_error(payload: dict[str, Any], operation: str) -> ProviderError:
    message = _optional_text(payload.get("message")) or ""
    return _classified_flyai_error(operation, message)


def _flyai_notice(source_notice: str | None) -> str:
    if source_notice is None:
        return FLYAI_PRICE_NOTICE
    return f"{source_notice} {FLYAI_PRICE_NOTICE}"


def _flyai_process_error(operation: str, stderr: bytes) -> ProviderError:
    text = stderr[:2_048].decode("utf-8", errors="ignore")
    return _classified_flyai_error(operation, text)


def _classified_flyai_error(operation: str, message: str) -> ProviderError:
    normalized = message.casefold()
    if any(token in normalized for token in ("api key", "apikey", "unauthorized", "token")):
        code = ProviderFailureCode.AUTHENTICATION_FAILED
    elif any(token in normalized for token in ("permission", "forbidden", "no access")):
        code = ProviderFailureCode.PERMISSION_DENIED
    elif any(token in normalized for token in ("quota", "rate limit", "too many", "qps")):
        code = ProviderFailureCode.RATE_LIMITED
    elif any(token in normalized for token in ("invalid", "parameter", "date")):
        code = ProviderFailureCode.INVALID_REQUEST
    else:
        code = ProviderFailureCode.UPSTREAM_ERROR
    return ProviderError(
        ProviderCode.FLYAI,
        code,
        operation,
        retryable=code in {ProviderFailureCode.RATE_LIMITED, ProviderFailureCode.UPSTREAM_ERROR},
    )


def _flyai_malformed(operation: str) -> ProviderError:
    return ProviderError(
        ProviderCode.FLYAI,
        ProviderFailureCode.MALFORMED_RESPONSE,
        operation,
        retryable=False,
    )


def _required_city_name(value: str | None, operation: str) -> str:
    if value is None:
        raise ProviderError(
            ProviderCode.FLYAI,
            ProviderFailureCode.INVALID_REQUEST,
            operation,
            retryable=False,
        )
    return value


def _required_text(value: Any, operation: str) -> str:
    parsed = _optional_text(value)
    if parsed is None:
        raise _flyai_malformed(operation)
    return parsed


def _optional_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    parsed = value.strip()
    return parsed or None


def _optional_date(value: Any) -> date | None:
    text = _optional_text(value)
    if text is None:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def _optional_rating(value: Any) -> float | None:
    if value in (None, "", []):
        return None
    try:
        rating = float(value)
    except (TypeError, ValueError):
        return None
    return rating if 0 <= rating <= 5 else None


def _optional_cny_price(value: Any) -> CnyAmountRange | None:
    if isinstance(value, (int, float, Decimal)) and not isinstance(value, bool):
        values = [str(value)]
        text = str(value)
    elif isinstance(value, str):
        text = value.strip()
        if any(marker in text for marker in ("起", "以上", "+")):
            return None
        values = re.findall(r"\d+(?:\.\d{1,2})?", text.replace(",", ""))
    else:
        return None
    if not values or len(values) > 2:
        return None
    try:
        amounts = [Decimal(item) * 100 for item in values]
    except InvalidOperation:
        return None
    if any(not amount.is_finite() or amount < 0 for amount in amounts):
        return None
    if any(amount != amount.to_integral_value() for amount in amounts):
        return None
    if len(amounts) == 1 and not re.fullmatch(r"[¥￥]?\s*\d+(?:\.\d{1,2})?", text):
        return None
    minimum = int(amounts[0])
    maximum = int(amounts[-1])
    if maximum < minimum:
        return None
    return CnyAmountRange(minimum_fen=minimum, maximum_fen=maximum)


def _safe_url(value: Any, allowed_suffixes: Sequence[str]) -> HttpUrl | None:
    text = _optional_text(value)
    if text is None:
        return None
    parsed = urlsplit(text)
    hostname = (parsed.hostname or "").casefold()
    if (
        parsed.scheme != "https"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port not in {None, 443}
        or not any(
            hostname == suffix[1:] or hostname.endswith(suffix) for suffix in allowed_suffixes
        )
    ):
        return None
    try:
        return _HTTP_URL_ADAPTER.validate_python(text)
    except ValidationError:
        return None


def _optional_source_coordinates(
    item: dict[str, Any],
    coordinate_system: CoordinateSystem | None,
) -> Coordinates | None:
    if coordinate_system is None:
        return None
    latitude = item.get("latitude")
    longitude = item.get("longitude")
    if latitude is None or longitude is None or latitude in ("", []) or longitude in ("", []):
        return None
    try:
        return Coordinates(
            latitude=float(latitude),
            longitude=float(longitude),
            coord_system=coordinate_system,
        )
    except (TypeError, ValueError):
        return None
