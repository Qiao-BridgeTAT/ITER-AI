"""Resolve authenticated and temporary V4 actors onto durable trip ownership."""

from __future__ import annotations

import secrets
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import cast
from uuid import UUID, uuid4

from pydantic import ValidationError
from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from backend.contracts.cold_start import ColdStartSubmission
from backend.contracts.enums import OwnerType, TripPhase
from backend.contracts.state import TripState
from backend.domain.anonymous_v4 import (
    ANONYMOUS_V4_OWNER_NICKNAME,
    anonymous_v4_owner_id,
)
from backend.domain.authorization import RequestActor
from backend.persistence.models import Trip, TripSnapshot, User
from backend.persistence.redis_temporary import RedisTemporaryStore


class V4OwnerResolutionError(RuntimeError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class V4OwnerResolver:
    """Keep V4 transactions durable while Redis proves temporary ownership."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        temporary: RedisTemporaryStore,
        *,
        anonymous_ttl_seconds: int,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._session_factory = session_factory
        self._temporary = temporary
        self._anonymous_ttl_seconds = anonymous_ttl_seconds
        self._clock = clock

    async def resolve(self, actor: RequestActor, trip_id: UUID) -> UUID:
        if actor.owner_type is OwnerType.USER:
            try:
                return UUID(actor.owner_id)
            except ValueError as error:
                raise V4OwnerResolutionError("invalid_user_session") from error
        if actor.owner_type is not OwnerType.ANONYMOUS:
            raise V4OwnerResolutionError("v4_session_required")

        session_payload = await self._temporary.load_anonymous_session(actor.owner_id)
        if session_payload is None:
            raise V4OwnerResolutionError("v4_anonymous_session_required")
        raw_state = await self._temporary.load_anonymous_state(actor.owner_id, trip_id)
        if raw_state is None:
            raise V4OwnerResolutionError("v4_trip_not_found")
        try:
            anonymous_state = TripState.model_validate(raw_state)
        except ValidationError as error:
            raise V4OwnerResolutionError("v4_anonymous_state_invalid") from error
        if (
            anonymous_state.owner_type is not OwnerType.ANONYMOUS
            or anonymous_state.owner_id != actor.owner_id
            or anonymous_state.trip_id != trip_id
        ):
            raise V4OwnerResolutionError("v4_trip_not_found")

        guest_owner_id = anonymous_v4_owner_id(actor.owner_id)
        existing_owner = await self._trip_owner(trip_id)
        if existing_owner is not None:
            if existing_owner != guest_owner_id:
                raise V4OwnerResolutionError("v4_trip_not_found")
            await self._renew_anonymous_owner(actor.owner_id, trip_id)
            await self._touch_guest_owner(guest_owner_id)
            return guest_owner_id

        token = secrets.token_urlsafe(24)
        if not await self._temporary.acquire_trip_lock(trip_id, token, 30):
            raise V4OwnerResolutionError("trip_busy")
        try:
            existing_owner = await self._trip_owner(trip_id)
            if existing_owner is not None:
                if existing_owner != guest_owner_id:
                    raise V4OwnerResolutionError("v4_trip_not_found")
                await self._renew_anonymous_owner(actor.owner_id, trip_id)
                await self._touch_guest_owner(guest_owner_id)
                return guest_owner_id
            state = _guest_legacy_state(
                anonymous_state,
                session_payload,
                guest_owner_id,
                clock=self._clock,
            )
            await self._create_guest_trip(guest_owner_id, state)
            await self._renew_anonymous_owner(actor.owner_id, trip_id)
            return guest_owner_id
        finally:
            await self._temporary.release_trip_lock(trip_id, token)

    async def _renew_anonymous_owner(self, session_id: str, trip_id: UUID) -> None:
        renewed = await self._temporary.renew_anonymous_session(
            session_id,
            expires_at=(self._clock() + timedelta(seconds=self._anonymous_ttl_seconds)).isoformat(),
            ttl_seconds=self._anonymous_ttl_seconds,
            trip_id=trip_id,
        )
        if renewed is None:
            raise V4OwnerResolutionError("v4_anonymous_session_required")

    async def _trip_owner(self, trip_id: UUID) -> UUID | None:
        async with self._session_factory() as session:
            return cast(
                UUID | None,
                await session.scalar(select(Trip.owner_user_id).where(Trip.id == trip_id)),
            )

    async def _create_guest_trip(self, owner_id: UUID, state: TripState) -> None:
        cutoff = self._clock() - timedelta(seconds=self._anonymous_ttl_seconds)
        async with self._session_factory() as session, session.begin():
            stale_guest_ids = select(User.id).where(
                User.nickname == ANONYMOUS_V4_OWNER_NICKNAME,
                User.updated_at <= cutoff,
                ~User.identities.any(),
            )
            await session.execute(delete(User).where(User.id.in_(stale_guest_ids)))
            if await session.get(User, owner_id) is None:
                session.add(
                    User(
                        id=owner_id,
                        status="active",
                        nickname=ANONYMOUS_V4_OWNER_NICKNAME,
                    )
                )
                await session.flush()
            session.add(
                Trip(
                    id=state.trip_id,
                    owner_user_id=owner_id,
                    phase=state.phase.value,
                    state_version=state.state_version,
                    schema_version=state.schema_version,
                    title=_trip_title(state),
                )
            )
            session.add(
                TripSnapshot(
                    id=uuid4(),
                    trip_id=state.trip_id,
                    state_version=state.state_version,
                    schema_version=state.schema_version,
                    snapshot=state.model_dump(mode="json"),
                )
            )

    async def _touch_guest_owner(self, owner_id: UUID) -> None:
        async with self._session_factory() as session, session.begin():
            await session.execute(
                update(User)
                .where(
                    User.id == owner_id,
                    User.nickname == ANONYMOUS_V4_OWNER_NICKNAME,
                )
                .values(updated_at=self._clock())
            )


async def user_only_v4_owner(actor: RequestActor, _trip_id: UUID) -> UUID:
    if actor.owner_type is not OwnerType.USER:
        raise V4OwnerResolutionError("v4_user_session_required")
    try:
        return UUID(actor.owner_id)
    except ValueError as error:
        raise V4OwnerResolutionError("invalid_user_session") from error


def _guest_legacy_state(
    state: TripState,
    session_payload: dict[str, object],
    guest_owner_id: UUID,
    *,
    clock: Callable[[], datetime],
) -> TripState:
    payload = state.model_dump(mode="json")
    raw_defaults = session_payload.get("personal_defaults")
    raw_completed_at = session_payload.get("cold_start_completed_at")
    if raw_defaults is not None:
        try:
            defaults = ColdStartSubmission.model_validate(raw_defaults)
        except ValidationError as error:
            raise V4OwnerResolutionError("v4_anonymous_state_invalid") from error
        payload["personal_defaults"] = defaults.model_dump(mode="json")
        payload["cold_start_completed_at"] = raw_completed_at or clock().isoformat()
        if state.phase is TripPhase.COLD_START:
            payload["phase"] = TripPhase.CITY_SELECTION.value
        if state.personal_defaults is None:
            payload["state_version"] = state.state_version + 1
    payload["owner_type"] = OwnerType.USER.value
    payload["owner_id"] = str(guest_owner_id)
    payload["active_generation_id"] = None
    try:
        return TripState.model_validate(payload)
    except ValidationError as error:
        raise V4OwnerResolutionError("v4_anonymous_state_invalid") from error


def _trip_title(state: TripState) -> str:
    if state.city is not None:
        return f"{state.city.value}旅行"
    if state.city_id:
        return f"{state.city_id}旅行"
    return "未命名旅行"
