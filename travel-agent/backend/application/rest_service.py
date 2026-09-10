"""M0-06 travel, preference, account, and export application use cases."""

from __future__ import annotations

import hashlib
import json
import secrets
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, TypeVar
from uuid import UUID, uuid4

from pydantic import BaseModel, ValidationError
from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from backend.contracts.cold_start import ColdStartSubmission
from backend.contracts.enums import ArtifactStatus, OwnerType, TripPhase
from backend.contracts.rest import (
    AccountView,
    AnonymousSessionView,
    AttachAnonymousTripRequest,
    ConfirmPreferenceCandidatesRequest,
    CreateExportRequest,
    CreateTripRequest,
    ExportView,
    PatchPreferencesRequest,
    PreferenceListView,
    PreferenceValue,
    TripListItem,
    TripListView,
    TripSnapshotView,
    UpdateAccountRequest,
)
from backend.contracts.state import TripState
from backend.contracts.trip_setup import TripShell
from backend.domain.account_profile import effective_nickname
from backend.domain.anonymous_v4 import (
    ANONYMOUS_V4_OWNER_NICKNAME,
    anonymous_v4_owner_id,
)
from backend.domain.authorization import RequestActor
from backend.persistence.models import (
    Artifact,
    AuthIdentity,
    Message,
    Trip,
    TripSnapshot,
    TripVersion,
    User,
    UserPreference,
)
from backend.persistence.redis_temporary import RedisTemporaryStore
from backend.persistence.trip_repository import TripNotFoundError, TripRepository

if TYPE_CHECKING:
    from backend.application.auth_service import PhoneProtector


class RestApplicationError(RuntimeError):
    pass


class RestResourceNotFoundError(RestApplicationError):
    pass


class RestConflictError(RestApplicationError):
    pass


class RestActorError(RestApplicationError):
    pass


ResponseModel = TypeVar("ResponseModel", bound=BaseModel)


def _aware(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value


class TravelRestService:
    """Application boundary shared by REST adapters and future command handlers."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        redis: RedisTemporaryStore,
        *,
        anonymous_ttl_seconds: int,
        idempotency_ttl_seconds: int = 86_400,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        phone_protector: PhoneProtector | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._trips = TripRepository(session_factory)
        self._redis = redis
        self._anonymous_ttl_seconds = anonymous_ttl_seconds
        self._idempotency_ttl_seconds = idempotency_ttl_seconds
        self._clock = clock
        self._phone_protector = phone_protector

    async def create_anonymous_session(
        self,
        idempotency_key: str,
        existing_session_id: str | None = None,
    ) -> AnonymousSessionView:
        # Reuse is an authenticated touch, not a cached creation response.
        if existing_session_id:
            renewed = await self._redis.renew_anonymous_session(
                existing_session_id,
                expires_at=(
                    self._clock() + timedelta(seconds=self._anonymous_ttl_seconds)
                ).isoformat(),
                ttl_seconds=self._anonymous_ttl_seconds,
            )
            if renewed is not None:
                return AnonymousSessionView.model_validate(
                    {"session_id": renewed["session_id"], "expires_at": renewed["expires_at"]}
                )

        async def create() -> AnonymousSessionView:
            session_id = secrets.token_urlsafe(32)
            expires_at = self._clock() + timedelta(seconds=self._anonymous_ttl_seconds)
            view = AnonymousSessionView(session_id=session_id, expires_at=expires_at)
            await self._redis.save_anonymous_session(
                session_id,
                view.model_dump(mode="json"),
                self._anonymous_ttl_seconds,
            )
            return view

        view = await self._idempotent_model(
            "create-anonymous-session",
            "public",
            idempotency_key,
            {},
            AnonymousSessionView,
            create,
        )
        # An old creation key cannot reissue an already deleted/expired credential.
        if await self._redis.load_anonymous_session(view.session_id) is None:
            raise RestResourceNotFoundError("anonymous session has expired")
        return view

    async def delete_anonymous_session(
        self, actor: RequestActor, session_id: str, idempotency_key: str
    ) -> None:
        self._require_actor(actor, OwnerType.ANONYMOUS)
        if actor.owner_id != session_id:
            raise RestResourceNotFoundError("anonymous session was not found")

        async def remove() -> None:
            guest_owner_id = anonymous_v4_owner_id(session_id)
            async with self._session_factory() as session, session.begin():
                await session.execute(
                    delete(User).where(
                        User.id == guest_owner_id,
                        User.nickname == ANONYMOUS_V4_OWNER_NICKNAME,
                    )
                )
            await self._redis.clear_anonymous_session(session_id)

        await self._idempotent_void(
            "delete-anonymous-session",
            actor.owner_id,
            idempotency_key,
            {"session_id": session_id},
            remove,
        )

    async def get_account(self, actor: RequestActor) -> AccountView:
        user_id = self._user_id(actor)
        async with self._session_factory() as session:
            user = await session.get(User, user_id)
            identity = await session.scalar(
                select(AuthIdentity)
                .where(AuthIdentity.user_id == user_id, AuthIdentity.provider == "phone")
                .order_by(AuthIdentity.created_at, AuthIdentity.id)
                .limit(1)
            )
        if user is None:
            raise RestResourceNotFoundError("account was not found")
        masked_phone = "***"
        if identity is not None and self._phone_protector is not None:
            phone = self._phone_protector.decrypt(identity.encrypted_identifier)
            masked_phone = self._phone_protector.mask(phone)
        return AccountView(
            user_id=user.id,
            masked_phone=masked_phone,
            nickname=effective_nickname(user.id, user.nickname),
            created_at=_aware(user.created_at),
        )

    async def update_account(
        self,
        actor: RequestActor,
        request: UpdateAccountRequest,
        idempotency_key: str,
    ) -> AccountView:
        user_id = self._user_id(actor)

        async def update_profile() -> AccountView:
            async with self._session_factory() as session, session.begin():
                user = await session.get(User, user_id)
                if user is None:
                    raise RestResourceNotFoundError("account was not found")
                user.nickname = request.nickname
            return await self.get_account(actor)

        return await self._idempotent_model(
            "update-account",
            actor.owner_id,
            idempotency_key,
            request.model_dump(mode="json"),
            AccountView,
            update_profile,
        )

    async def delete_account(self, actor: RequestActor, idempotency_key: str) -> None:
        user_id = self._user_id(actor)

        async def remove() -> None:
            async with self._session_factory() as session, session.begin():
                exists = await session.scalar(select(User.id).where(User.id == user_id))
                if exists is None:
                    raise RestResourceNotFoundError("account was not found")
                await session.execute(
                    update(Trip)
                    .where(Trip.owner_user_id == user_id)
                    .values(current_plan_version_id=None, base_confirmed_version_id=None)
                )
                await session.execute(delete(User).where(User.id == user_id))

        await self._idempotent_void("delete-account", actor.owner_id, idempotency_key, {}, remove)

    async def create_trip(
        self,
        actor: RequestActor,
        request: CreateTripRequest,
        idempotency_key: str,
    ) -> TripSnapshotView:
        async def create() -> TripSnapshotView:
            trip_id = request.trip_id or uuid4()
            if actor.owner_type is OwnerType.ANONYMOUS:
                if request.anonymous_session_id != actor.owner_id:
                    raise RestActorError("anonymous session proof does not match the actor")
                anonymous_session = await self._redis.load_anonymous_session(actor.owner_id)
                if anonymous_session is None:
                    raise RestResourceNotFoundError("anonymous session was not found")
                state = self._anonymous_initial_state(trip_id, actor, anonymous_session)
                await self._redis.save_anonymous_state(
                    actor.owner_id,
                    trip_id,
                    state.model_dump(mode="json"),
                    self._anonymous_ttl_seconds,
                )
                return TripSnapshotView(state=state)

            if request.anonymous_session_id is not None:
                raise RestActorError("user trips cannot use anonymous session ownership")
            user_id = self._user_id(actor)
            async with self._session_factory() as session:
                user_exists = await session.scalar(select(User.id).where(User.id == user_id))
            if user_exists is None:
                raise RestResourceNotFoundError("account was not found")
            state = await self._user_initial_state(trip_id, actor, user_id)
            await self._trips.create_user_trip(user_id, state)
            return TripSnapshotView(state=state)

        return await self._idempotent_model(
            "create-trip",
            actor.owner_id,
            idempotency_key,
            request.model_dump(mode="json"),
            TripSnapshotView,
            create,
        )

    async def attach_anonymous_trip(
        self,
        actor: RequestActor,
        request: AttachAnonymousTripRequest,
        idempotency_key: str,
    ) -> TripSnapshotView:
        user_id = self._user_id(actor)

        async def attach() -> TripSnapshotView:
            lock_token = secrets.token_urlsafe(24)
            if not await self._redis.acquire_trip_lock(request.trip_id, lock_token, 30):
                raise RestConflictError("trip migration is already in progress")
            try:
                anonymous_session = await self._redis.load_anonymous_session(
                    request.anonymous_session_id
                )
                raw_state = await self._redis.load_anonymous_state(
                    request.anonymous_session_id, request.trip_id
                )
                if anonymous_session is None or raw_state is None:
                    raise RestResourceNotFoundError("anonymous trip was not found")
                state = TripState.model_validate(raw_state)
                if (
                    state.owner_type is not OwnerType.ANONYMOUS
                    or state.owner_id != request.anonymous_session_id
                    or state.trip_id != request.trip_id
                ):
                    raise RestResourceNotFoundError("anonymous trip was not found")
                session_defaults = anonymous_session.get("personal_defaults")
                if session_defaults is not None and state.personal_defaults is None:
                    try:
                        defaults = ColdStartSubmission.model_validate(session_defaults)
                        state = TripState.model_validate(
                            {
                                **state.model_dump(mode="json"),
                                "personal_defaults": defaults.model_dump(mode="json"),
                                "cold_start_completed_at": anonymous_session.get(
                                    "cold_start_completed_at"
                                )
                                or self._clock().isoformat(),
                                "phase": (
                                    TripPhase.CITY_SELECTION.value
                                    if state.phase is TripPhase.COLD_START
                                    else state.phase.value
                                ),
                                "state_version": state.state_version + 1,
                            }
                        )
                    except ValidationError as exc:
                        raise RestConflictError(
                            "saved anonymous cold-start preference is invalid"
                        ) from exc
                persisted_owner = await self._trip_owner_projection(request.trip_id)
                guest_owner_id = anonymous_v4_owner_id(request.anonymous_session_id)
                if persisted_owner is not None and persisted_owner[0] == guest_owner_id:
                    migrated = TripState.model_validate(
                        {
                            **state.model_dump(mode="json"),
                            "owner_type": OwnerType.USER.value,
                            "owner_id": str(user_id),
                            "phase": persisted_owner[1],
                            "state_version": persisted_owner[2],
                            "active_generation_id": None,
                        }
                    )
                    await self._transfer_v4_guest_trip(
                        guest_owner_id,
                        user_id,
                        migrated,
                    )
                    await self._redis.clear_anonymous_session(request.anonymous_session_id)
                    return TripSnapshotView(state=migrated)
                if persisted_owner is not None:
                    if persisted_owner[0] == user_id:
                        existing = await self._load_owned_trip_if_present(user_id, request.trip_id)
                        if existing is not None:
                            await self._redis.clear_anonymous_session(request.anonymous_session_id)
                            return TripSnapshotView(state=existing)
                    raise RestResourceNotFoundError("anonymous trip was not found")
                migrated = TripState.model_validate(
                    {
                        **state.model_dump(mode="json"),
                        "owner_type": OwnerType.USER.value,
                        "owner_id": str(user_id),
                        "active_generation_id": None,
                    }
                )
                messages = await self._redis.load_anonymous_messages(
                    request.anonymous_session_id, request.trip_id
                )
                await self._persist_migrated_trip(user_id, migrated, messages)
                await self._redis.clear_anonymous_session(request.anonymous_session_id)
                return TripSnapshotView(state=migrated)
            finally:
                await self._redis.release_trip_lock(request.trip_id, lock_token)

        return await self._idempotent_model(
            "attach-anonymous-trip",
            actor.owner_id,
            idempotency_key,
            request.model_dump(mode="json"),
            TripSnapshotView,
            attach,
        )

    async def list_trips(self, actor: RequestActor) -> TripListView:
        user_id = self._user_id(actor)
        records = await self._trips.list_user_trips(user_id)
        return TripListView(
            trips=[
                TripListItem(
                    trip_id=item.trip_id,
                    phase=TripPhase(item.phase),
                    title=item.title,
                    updated_at=_aware(item.updated_at),
                )
                for item in records
            ]
        )

    async def get_trip_shell(self, actor: RequestActor, trip_id: UUID) -> TripShell:
        """Authorize navigation without decoding a V4 trip as legacy TripState."""
        if actor.owner_type is OwnerType.USER:
            async with self._session_factory() as session:
                row = (
                    await session.execute(
                        select(Trip, TripSnapshot.snapshot_kind)
                        .join(TripSnapshot, TripSnapshot.trip_id == Trip.id)
                        .where(Trip.id == trip_id, Trip.owner_user_id == self._user_id(actor))
                        .order_by(TripSnapshot.state_version.desc())
                        .limit(1)
                    )
                ).first()
            if row is None:
                raise RestResourceNotFoundError("trip was not found")
            trip, kind = row
            if kind == "v4":
                return TripShell(
                    trip_id=trip.id,
                    owner_type=OwnerType.USER,
                    owner_id=str(trip.owner_user_id),
                    phase=TripPhase(trip.phase),
                    state_version=trip.state_version,
                    protocol_version="v4",
                )
        state = (await self.get_trip(actor, trip_id)).state
        return TripShell.model_validate(
            state.model_dump(
                include={
                    "trip_id",
                    "owner_type",
                    "owner_id",
                    "city",
                    "phase",
                    "state_version",
                }
            )
        )

    async def get_trip(self, actor: RequestActor, trip_id: UUID) -> TripSnapshotView:
        if actor.owner_type is OwnerType.ANONYMOUS:
            async with self._session_factory() as session:
                transferred = await session.scalar(select(Trip.id).where(Trip.id == trip_id))
            if transferred is not None:
                raise RestResourceNotFoundError("trip was not found")
            raw = await self._redis.load_anonymous_state(actor.owner_id, trip_id)
            if raw is None:
                raise RestResourceNotFoundError("trip was not found")
            state = TripState.model_validate(raw)
            if state.owner_type is not OwnerType.ANONYMOUS or state.owner_id != actor.owner_id:
                raise RestResourceNotFoundError("trip was not found")
            return TripSnapshotView(state=state)
        try:
            state = await self._trips.load_user_trip(self._user_id(actor), trip_id)
        except ValidationError:
            state = await self._load_v4_shell_projection(self._user_id(actor), trip_id)
        except TripNotFoundError as exc:
            raise RestResourceNotFoundError("trip was not found") from exc
        return TripSnapshotView(state=state)

    async def delete_trip(self, actor: RequestActor, trip_id: UUID, idempotency_key: str) -> None:
        async def remove() -> None:
            await self.get_trip(actor, trip_id)
            if actor.owner_type is OwnerType.ANONYMOUS:
                await self._redis.delete_anonymous_trip(actor.owner_id, trip_id)
                return
            deleted = await self._trips.delete_user_trip(self._user_id(actor), trip_id)
            if not deleted:
                raise RestResourceNotFoundError("trip was not found")

        await self._idempotent_void(
            "delete-trip",
            actor.owner_id,
            idempotency_key,
            {"trip_id": trip_id},
            remove,
        )

    async def get_preferences(self, actor: RequestActor) -> PreferenceListView:
        return await self._preference_view(self._user_id(actor))

    async def patch_preferences(
        self,
        actor: RequestActor,
        request: PatchPreferencesRequest,
        idempotency_key: str,
    ) -> PreferenceListView:
        user_id = self._user_id(actor)

        async def patch() -> PreferenceListView:
            ids = [item.preference_id for item in request.preferences]
            async with self._session_factory() as session, session.begin():
                rows = (
                    await session.scalars(
                        select(UserPreference).where(
                            UserPreference.user_id == user_id,
                            UserPreference.id.in_(ids),
                        )
                    )
                ).all()
                if len(rows) != len(ids):
                    raise RestResourceNotFoundError("preference was not found")
                patches = {item.preference_id: item for item in request.preferences}
                for row in rows:
                    item = patches[row.id]
                    row.value = {"text": item.value}
                    row.active = item.active
            return await self._preference_view(user_id)

        return await self._idempotent_model(
            "patch-preferences",
            actor.owner_id,
            idempotency_key,
            request.model_dump(mode="json"),
            PreferenceListView,
            patch,
        )

    async def save_cold_start_preference(
        self,
        actor: RequestActor,
        request: ColdStartSubmission,
        idempotency_key: str,
    ) -> PreferenceListView:
        if actor.owner_type is OwnerType.ANONYMOUS:

            async def save_anonymous() -> PreferenceListView:
                session_payload = await self._redis.load_anonymous_session(actor.owner_id)
                if session_payload is None:
                    raise RestResourceNotFoundError("anonymous session was not found")
                session_payload["personal_defaults"] = request.model_dump(mode="json")
                session_payload["cold_start_completed_at"] = self._clock().isoformat()
                session_payload["expires_at"] = (
                    self._clock() + timedelta(seconds=self._anonymous_ttl_seconds)
                ).isoformat()
                await self._redis.save_anonymous_session(
                    actor.owner_id,
                    session_payload,
                    self._anonymous_ttl_seconds,
                )
                return PreferenceListView(preferences=[])

            return await self._idempotent_model(
                "save-anonymous-cold-start-preference",
                actor.owner_id,
                idempotency_key,
                request.model_dump(mode="json"),
                PreferenceListView,
                save_anonymous,
            )

        user_id = self._user_id(actor)

        async def save() -> PreferenceListView:
            async with self._session_factory() as session, session.begin():
                row = await session.scalar(
                    select(UserPreference)
                    .where(
                        UserPreference.user_id == user_id,
                        UserPreference.preference_key == "cold_start",
                        UserPreference.source == "cold_start",
                        UserPreference.active.is_(True),
                    )
                    .order_by(UserPreference.updated_at.desc(), UserPreference.id)
                    .limit(1)
                )
                if row is None:
                    session.add(
                        UserPreference(
                            id=uuid4(),
                            user_id=user_id,
                            preference_key="cold_start",
                            value=request.model_dump(mode="json"),
                            source="cold_start",
                            confidence="high",
                            active=True,
                            last_confirmed_at=self._clock(),
                        )
                    )
                else:
                    row.value = request.model_dump(mode="json")
                    row.confidence = "high"
                    row.last_confirmed_at = self._clock()
            return await self._preference_view(user_id)

        return await self._idempotent_model(
            "save-cold-start-preference",
            actor.owner_id,
            idempotency_key,
            request.model_dump(mode="json"),
            PreferenceListView,
            save,
        )

    async def delete_preference(
        self, actor: RequestActor, preference_id: UUID, idempotency_key: str
    ) -> None:
        user_id = self._user_id(actor)

        async def remove() -> None:
            async with self._session_factory() as session, session.begin():
                result = await session.execute(
                    delete(UserPreference).where(
                        UserPreference.id == preference_id,
                        UserPreference.user_id == user_id,
                    )
                )
                if getattr(result, "rowcount", 0) != 1:
                    raise RestResourceNotFoundError("preference was not found")

        await self._idempotent_void(
            "delete-preference",
            actor.owner_id,
            idempotency_key,
            {"preference_id": preference_id},
            remove,
        )

    async def confirm_preference_candidates(
        self,
        actor: RequestActor,
        trip_id: UUID,
        request: ConfirmPreferenceCandidatesRequest,
        idempotency_key: str,
    ) -> PreferenceListView:
        user_id = self._user_id(actor)

        async def confirm() -> PreferenceListView:
            await self.get_trip(actor, trip_id)
            async with self._session_factory() as session, session.begin():
                rows = (
                    await session.scalars(
                        select(UserPreference).where(
                            UserPreference.user_id == user_id,
                            UserPreference.id.in_(request.candidate_ids),
                            UserPreference.source == "user_confirmed_inference",
                            UserPreference.active.is_(False),
                        )
                    )
                ).all()
                if len(rows) != len(request.candidate_ids):
                    raise RestResourceNotFoundError("preference candidate was not found")
                for row in rows:
                    row.active = True
                    row.last_confirmed_at = self._clock()
            return await self._preference_view(user_id)

        return await self._idempotent_model(
            "confirm-preference-candidates",
            actor.owner_id,
            idempotency_key,
            {"trip_id": trip_id, **request.model_dump(mode="json")},
            PreferenceListView,
            confirm,
        )

    async def create_export(
        self,
        actor: RequestActor,
        trip_id: UUID,
        plan_version_id: UUID,
        request: CreateExportRequest,
        idempotency_key: str,
    ) -> ExportView:
        if request.plan_version_id != plan_version_id:
            raise RestConflictError("path and request plan version do not match")

        async def create() -> ExportView:
            trip = await self.get_trip(actor, trip_id)
            if actor.owner_type is OwnerType.ANONYMOUS:
                if (
                    trip.state.phase is not TripPhase.CONFIRMED
                    or trip.state.current_plan_version_id != plan_version_id
                ):
                    raise RestResourceNotFoundError("confirmed plan version was not found")
                artifact_id = uuid4()
                expires_at = self._clock() + timedelta(seconds=self._anonymous_ttl_seconds)
                view = ExportView(
                    artifact_id=artifact_id,
                    trip_id=trip_id,
                    plan_version_id=plan_version_id,
                    status=ArtifactStatus.PENDING,
                    expires_at=expires_at,
                )
                await self._redis.save_anonymous_artifact(
                    actor.owner_id,
                    trip_id,
                    artifact_id,
                    view.model_dump(mode="json"),
                    self._anonymous_ttl_seconds,
                )
                return view

            async with self._session_factory() as session, session.begin():
                version = await session.scalar(
                    select(TripVersion).where(
                        TripVersion.id == plan_version_id,
                        TripVersion.trip_id == trip_id,
                    )
                )
                if version is None or version.status != "confirmed":
                    raise RestResourceNotFoundError("confirmed plan version was not found")
                artifact = await session.scalar(
                    select(Artifact).where(
                        Artifact.trip_id == trip_id,
                        Artifact.plan_version_id == plan_version_id,
                        Artifact.artifact_type == "long_image",
                    )
                )
                if artifact is None:
                    artifact = Artifact(
                        id=uuid4(),
                        trip_id=trip_id,
                        plan_version_id=plan_version_id,
                        artifact_type="long_image",
                        status="pending",
                    )
                    session.add(artifact)
                    await session.flush()
                view = self._artifact_view(artifact)
            return view

        return await self._idempotent_model(
            "create-export",
            actor.owner_id,
            idempotency_key,
            {
                "trip_id": trip_id,
                "path_plan_version_id": plan_version_id,
                **request.model_dump(mode="json"),
            },
            ExportView,
            create,
        )

    async def get_export(self, actor: RequestActor, artifact_id: UUID) -> ExportView:
        if actor.owner_type is OwnerType.ANONYMOUS:
            raw = await self._redis.load_anonymous_artifact(actor.owner_id, artifact_id)
            if raw is None:
                raise RestResourceNotFoundError("export was not found")
            return ExportView.model_validate(raw)
        user_id = self._user_id(actor)
        async with self._session_factory() as session:
            artifact = await session.scalar(
                select(Artifact)
                .join(Trip, Trip.id == Artifact.trip_id)
                .where(Artifact.id == artifact_id, Trip.owner_user_id == user_id)
            )
        if artifact is None:
            raise RestResourceNotFoundError("export was not found")
        return self._artifact_view(artifact)

    async def _preference_view(self, user_id: UUID) -> PreferenceListView:
        async with self._session_factory() as session:
            rows = (
                await session.scalars(
                    select(UserPreference)
                    .where(UserPreference.user_id == user_id, UserPreference.active.is_(True))
                    .order_by(UserPreference.updated_at.desc(), UserPreference.id)
                )
            ).all()
        return PreferenceListView(
            preferences=[
                PreferenceValue(
                    preference_id=row.id,
                    key=row.preference_key,
                    value=self._preference_text(row.value),
                    active=row.active,
                    updated_at=_aware(row.updated_at),
                )
                for row in rows
            ]
        )

    async def _load_owned_trip_if_present(self, user_id: UUID, trip_id: UUID) -> TripState | None:
        async with self._session_factory() as session:
            owner_id = await session.scalar(select(Trip.owner_user_id).where(Trip.id == trip_id))
        if owner_id is None:
            return None
        if owner_id != user_id:
            raise RestResourceNotFoundError("anonymous trip was not found")
        try:
            return await self._trips.load_user_trip(user_id, trip_id)
        except ValidationError:
            return await self._load_v4_shell_projection(user_id, trip_id)
        except TripNotFoundError as exc:
            raise RestConflictError("migrated trip is missing its stable snapshot") from exc

    async def _load_v4_shell_projection(self, user_id: UUID, trip_id: UUID) -> TripState:
        async with self._session_factory() as session:
            trip = await session.scalar(
                select(Trip).where(
                    Trip.id == trip_id,
                    Trip.owner_user_id == user_id,
                )
            )
            legacy = await session.scalar(
                select(TripSnapshot)
                .where(
                    TripSnapshot.trip_id == trip_id,
                    TripSnapshot.snapshot_kind == "legacy",
                )
                .order_by(TripSnapshot.state_version.desc())
                .limit(1)
            )
        if trip is None:
            raise RestResourceNotFoundError("trip was not found")
        if legacy is None:
            raise RestConflictError("V4 trip is missing its shell projection")
        try:
            return TripState.model_validate(
                {
                    **legacy.snapshot,
                    "owner_type": OwnerType.USER.value,
                    "owner_id": str(user_id),
                    "phase": trip.phase,
                    "state_version": trip.state_version,
                    "current_plan_version_id": trip.current_plan_version_id,
                    "base_confirmed_version_id": trip.base_confirmed_version_id,
                    "active_generation_id": None,
                }
            )
        except ValidationError as error:
            raise RestConflictError("V4 trip shell projection is invalid") from error

    async def _trip_owner_projection(self, trip_id: UUID) -> tuple[UUID, str, int] | None:
        async with self._session_factory() as session:
            row = (
                await session.execute(
                    select(Trip.owner_user_id, Trip.phase, Trip.state_version).where(
                        Trip.id == trip_id
                    )
                )
            ).one_or_none()
        if row is None:
            return None
        return row[0], str(row[1]), int(row[2])

    async def _transfer_v4_guest_trip(
        self,
        guest_owner_id: UUID,
        user_id: UUID,
        state: TripState,
    ) -> None:
        async with self._session_factory() as session, session.begin():
            user_exists = await session.scalar(select(User.id).where(User.id == user_id))
            if user_exists is None:
                raise RestResourceNotFoundError("account was not found")
            transferred = await session.execute(
                update(Trip)
                .where(
                    Trip.id == state.trip_id,
                    Trip.owner_user_id == guest_owner_id,
                )
                .values(owner_user_id=user_id)
            )
            if getattr(transferred, "rowcount", 0) != 1:
                raise RestResourceNotFoundError("anonymous trip was not found")
            await session.flush()
            if state.personal_defaults is not None and state.cold_start_completed_at is not None:
                existing_preference = await session.scalar(
                    select(UserPreference.id).where(
                        UserPreference.user_id == user_id,
                        UserPreference.preference_key == "cold_start",
                        UserPreference.source == "cold_start",
                        UserPreference.active.is_(True),
                    )
                )
                if existing_preference is None:
                    session.add(
                        UserPreference(
                            id=uuid4(),
                            user_id=user_id,
                            preference_key="cold_start",
                            value=state.personal_defaults.model_dump(mode="json"),
                            source="cold_start",
                            confidence="high",
                            active=True,
                            last_confirmed_at=state.cold_start_completed_at,
                        )
                    )
            await session.execute(
                delete(User).where(
                    User.id == guest_owner_id,
                    User.nickname == ANONYMOUS_V4_OWNER_NICKNAME,
                )
            )

    async def _persist_migrated_trip(
        self,
        user_id: UUID,
        state: TripState,
        messages: list[dict[str, Any]],
    ) -> None:
        snapshot = state.model_dump(mode="json")
        async with self._session_factory() as session, session.begin():
            user_exists = await session.scalar(select(User.id).where(User.id == user_id))
            if user_exists is None:
                raise RestResourceNotFoundError("account was not found")
            trip_exists = await session.scalar(
                select(Trip.owner_user_id).where(Trip.id == state.trip_id)
            )
            if trip_exists is not None:
                if trip_exists != user_id:
                    raise RestResourceNotFoundError("anonymous trip was not found")
                return

            session.add(
                Trip(
                    id=state.trip_id,
                    owner_user_id=user_id,
                    phase=state.phase.value,
                    state_version=state.state_version,
                    schema_version=state.schema_version,
                    title=(f"{state.city.value}旅行" if state.city is not None else "未命名旅行"),
                    current_plan_version_id=state.current_plan_version_id,
                    base_confirmed_version_id=state.base_confirmed_version_id,
                )
            )
            self._add_migrated_versions(session, state, snapshot)
            session.add(
                TripSnapshot(
                    id=uuid4(),
                    trip_id=state.trip_id,
                    plan_version_id=state.current_plan_version_id,
                    state_version=state.state_version,
                    schema_version=state.schema_version,
                    snapshot=snapshot,
                )
            )
            for raw in messages:
                session.add(self._migrated_message(state.trip_id, raw))

            if state.personal_defaults is not None and state.cold_start_completed_at is not None:
                existing_preference = await session.scalar(
                    select(UserPreference.id).where(
                        UserPreference.user_id == user_id,
                        UserPreference.preference_key == "cold_start",
                        UserPreference.source == "cold_start",
                        UserPreference.active.is_(True),
                    )
                )
                if existing_preference is None:
                    session.add(
                        UserPreference(
                            id=uuid4(),
                            user_id=user_id,
                            preference_key="cold_start",
                            value=state.personal_defaults.model_dump(mode="json"),
                            source="cold_start",
                            confidence="high",
                            active=True,
                            last_confirmed_at=state.cold_start_completed_at,
                        )
                    )

    @staticmethod
    def _add_migrated_versions(
        session: AsyncSession, state: TripState, snapshot: dict[str, Any]
    ) -> None:
        version_ids = [
            value
            for value in (state.base_confirmed_version_id, state.current_plan_version_id)
            if value is not None
        ]
        unique_ids = list(dict.fromkeys(version_ids))
        for index, version_id in enumerate(unique_ids, start=1):
            session.add(
                TripVersion(
                    id=version_id,
                    trip_id=state.trip_id,
                    parent_version_id=(unique_ids[index - 2] if index > 1 else None),
                    version_number=index,
                    state_version=max(0, state.state_version - len(unique_ids) + index),
                    status=(
                        "confirmed"
                        if version_id == state.base_confirmed_version_id
                        or state.phase is TripPhase.CONFIRMED
                        else "draft"
                    ),
                    snapshot=snapshot,
                    schema_version=state.schema_version,
                    confirmed_at=(
                        datetime.now(UTC)
                        if version_id == state.base_confirmed_version_id
                        or state.phase is TripPhase.CONFIRMED
                        else None
                    ),
                )
            )

    @staticmethod
    def _migrated_message(trip_id: UUID, raw: dict[str, Any]) -> Message:
        role = str(raw.get("role", "system"))
        if role not in {"user", "assistant", "system", "tool"}:
            raise RestConflictError("anonymous message role is invalid")

        def optional_uuid(name: str) -> UUID | None:
            value = raw.get(name)
            if value in (None, ""):
                return None
            try:
                return UUID(str(value))
            except ValueError as exc:
                raise RestConflictError(f"anonymous message {name} is invalid") from exc

        message_id = optional_uuid("message_id") or optional_uuid("id") or uuid4()
        attachments = raw.get("attachments", [])
        metadata = raw.get("message_metadata", raw.get("metadata", {}))
        if not isinstance(attachments, list) or not isinstance(metadata, dict):
            raise RestConflictError("anonymous message payload is invalid")
        message = Message(
            id=message_id,
            trip_id=trip_id,
            client_message_id=optional_uuid("client_message_id"),
            role=role,
            message_type=str(raw.get("message_type", "text")),
            text=(str(raw["text"]) if raw.get("text") is not None else None),
            attachments=attachments,
            message_metadata=metadata,
            request_id=optional_uuid("request_id"),
            generation_id=optional_uuid("generation_id"),
        )
        created_at = raw.get("created_at")
        if created_at is not None:
            try:
                message.created_at = _aware(
                    datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
                )
            except ValueError as exc:
                raise RestConflictError("anonymous message created_at is invalid") from exc
        return message

    async def _idempotent_model(
        self,
        scope: str,
        owner_id: str,
        idempotency_key: str,
        request_data: Any,
        model: type[ResponseModel],
        operation: Callable[[], Awaitable[ResponseModel]],
    ) -> ResponseModel:
        fingerprint = self._request_fingerprint(request_data)
        existing = await self._redis.get_idempotency(scope, owner_id, idempotency_key)
        if existing is not None:
            if existing.get("request_fingerprint") != fingerprint:
                raise RestConflictError("idempotency key was already used for another request")
            if existing.get("status") == "complete":
                return model.model_validate(existing["response"])
            raise RestConflictError("an identical request is still being processed")
        claimed = await self._redis.claim_idempotency(
            scope,
            owner_id,
            idempotency_key,
            {"status": "processing", "request_fingerprint": fingerprint},
            self._idempotency_ttl_seconds,
        )
        if not claimed:
            raise RestConflictError("an identical request is still being processed")
        try:
            response = await operation()
        except Exception:
            await self._redis.clear_idempotency(scope, owner_id, idempotency_key)
            raise
        saved = await self._redis.save_idempotency_result(
            scope,
            owner_id,
            idempotency_key,
            {
                "status": "complete",
                "request_fingerprint": fingerprint,
                "response": response.model_dump(mode="json"),
            },
            self._idempotency_ttl_seconds,
        )
        if not saved:
            raise RestConflictError("idempotency record expired before completion")
        return response

    async def _idempotent_void(
        self,
        scope: str,
        owner_id: str,
        idempotency_key: str,
        request_data: Any,
        operation: Callable[[], Awaitable[None]],
    ) -> None:
        fingerprint = self._request_fingerprint(request_data)
        existing = await self._redis.get_idempotency(scope, owner_id, idempotency_key)
        if existing is not None:
            if existing.get("request_fingerprint") != fingerprint:
                raise RestConflictError("idempotency key was already used for another request")
            if existing.get("status") == "complete":
                return
            raise RestConflictError("an identical request is still being processed")
        claimed = await self._redis.claim_idempotency(
            scope,
            owner_id,
            idempotency_key,
            {"status": "processing", "request_fingerprint": fingerprint},
            self._idempotency_ttl_seconds,
        )
        if not claimed:
            raise RestConflictError("an identical request is still being processed")
        try:
            await operation()
        except Exception:
            await self._redis.clear_idempotency(scope, owner_id, idempotency_key)
            raise
        saved = await self._redis.save_idempotency_result(
            scope,
            owner_id,
            idempotency_key,
            {
                "status": "complete",
                "request_fingerprint": fingerprint,
                "response": None,
            },
            self._idempotency_ttl_seconds,
        )
        if not saved:
            raise RestConflictError("idempotency record expired before completion")

    async def _user_initial_state(
        self, trip_id: UUID, actor: RequestActor, user_id: UUID
    ) -> TripState:
        async with self._session_factory() as session:
            preference = await session.scalar(
                select(UserPreference)
                .where(
                    UserPreference.user_id == user_id,
                    UserPreference.preference_key == "cold_start",
                    UserPreference.source == "cold_start",
                    UserPreference.active.is_(True),
                )
                .order_by(UserPreference.updated_at.desc(), UserPreference.id)
                .limit(1)
            )
        if preference is None:
            return self._initial_state(trip_id, actor)
        try:
            defaults = ColdStartSubmission.model_validate(preference.value)
        except ValueError as exc:
            raise RestConflictError("saved cold-start preference is invalid") from exc
        return self._initial_state(
            trip_id,
            actor,
            personal_defaults=defaults,
            cold_start_completed_at=_aware(preference.last_confirmed_at or preference.updated_at),
        )

    def _anonymous_initial_state(
        self,
        trip_id: UUID,
        actor: RequestActor,
        anonymous_session: dict[str, Any],
    ) -> TripState:
        raw_defaults = anonymous_session.get("personal_defaults")
        if not isinstance(raw_defaults, dict):
            return self._initial_state(trip_id, actor)
        try:
            defaults = ColdStartSubmission.model_validate(raw_defaults)
            completed_at = datetime.fromisoformat(str(anonymous_session["cold_start_completed_at"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise RestConflictError("saved anonymous cold-start preference is invalid") from exc
        return self._initial_state(
            trip_id,
            actor,
            personal_defaults=defaults,
            cold_start_completed_at=_aware(completed_at),
        )

    @staticmethod
    def _initial_state(
        trip_id: UUID,
        actor: RequestActor,
        *,
        personal_defaults: ColdStartSubmission | None = None,
        cold_start_completed_at: datetime | None = None,
    ) -> TripState:
        return TripState(
            trip_id=trip_id,
            owner_type=actor.owner_type,
            owner_id=actor.owner_id,
            phase=(TripPhase.CITY_SELECTION if personal_defaults else TripPhase.COLD_START),
            state_version=0,
            schema_version="1.0.0",
            personal_defaults=personal_defaults,
            cold_start_completed_at=cold_start_completed_at,
        )

    @staticmethod
    def _request_fingerprint(request_data: Any) -> str:
        encoded = json.dumps(
            request_data,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _user_id(actor: RequestActor) -> UUID:
        TravelRestService._require_actor(actor, OwnerType.USER)
        try:
            return UUID(actor.owner_id)
        except ValueError as exc:
            raise RestActorError("user actor identifier is invalid") from exc

    @staticmethod
    def _require_actor(actor: RequestActor, owner_type: OwnerType) -> None:
        if actor.owner_type is not owner_type:
            raise RestActorError(f"this operation requires a {owner_type.value} actor")

    @staticmethod
    def _preference_text(value: dict[str, Any]) -> str:
        text = value.get("text")
        if isinstance(text, str) and text.strip():
            return text
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _artifact_view(artifact: Artifact) -> ExportView:
        return ExportView(
            artifact_id=artifact.id,
            trip_id=artifact.trip_id,
            plan_version_id=artifact.plan_version_id,
            status=ArtifactStatus(artifact.status),
            download_url=None,
            expires_at=_aware(artifact.expires_at) if artifact.expires_at else None,
        )
