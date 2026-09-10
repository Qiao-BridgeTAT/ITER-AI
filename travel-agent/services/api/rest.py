"""REST adapter for M0-06 application use cases."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Annotated
from urllib.parse import urlsplit
from uuid import UUID

from fastapi import APIRouter, Header, Request, Response, status
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from starlette.requests import HTTPConnection

from backend.application.auth_service import (
    AuthClientContext,
    AuthenticationConflictError,
    AuthenticationRateLimitError,
    AuthenticationService,
    InvalidVerificationCodeError,
    PhoneProtector,
)
from backend.application.rest_service import TravelRestService
from backend.config import ConfigurationError, Settings
from backend.contracts.cold_start import ColdStartSubmission
from backend.contracts.commands import IdempotencyKey
from backend.contracts.enums import OwnerType
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
    SmsSendRequest,
    SmsVerifyRequest,
    TripListView,
    TripSnapshotView,
    UpdateAccountRequest,
)
from backend.contracts.trip_setup import TripShell
from backend.domain.authorization import RequestActor
from backend.persistence.database import create_database_engine, create_session_factory
from backend.persistence.redis_temporary import RedisTemporaryStore
from backend.providers.sms import (
    AliyunPhoneVerificationSmsConfig,
    AliyunPhoneVerificationSmsProvider,
    SmsDeliveryError,
    SmsProvider,
    TencentCloudSmsConfig,
    TencentCloudSmsProvider,
    UnavailableSmsProvider,
)
from services.api.errors import ServiceError

ActorResolver = Callable[[HTTPConnection], Awaitable[RequestActor]]
IdempotencyHeader = Annotated[IdempotencyKey, Header(alias="Idempotency-Key")]
SESSION_COOKIE_NAME = "travel_agent_session"
ANONYMOUS_SESSION_COOKIE_NAME = "travel_agent_anonymous_session"


@dataclass(frozen=True)
class RestServiceRuntime:
    engine: AsyncEngine
    session_factory: async_sessionmaker[AsyncSession]
    redis_client: Redis
    redis_store: RedisTemporaryStore
    service: TravelRestService
    auth_service: AuthenticationService
    sms_provider: SmsProvider

    async def close(self) -> None:
        if isinstance(self.sms_provider, TencentCloudSmsProvider):
            await self.sms_provider.aclose()
        await self.redis_client.aclose()
        await self.engine.dispose()


def build_rest_runtime(
    settings: Settings, sms_provider: SmsProvider | None = None
) -> RestServiceRuntime:
    engine = create_database_engine(settings.values["DATABASE_URL"])
    session_factory = create_session_factory(engine)
    redis_client = Redis.from_url(settings.values["REDIS_URL"], decode_responses=True)
    redis_store = RedisTemporaryStore(redis_client)
    pii_secret = settings.values.get("PII_ENCRYPTION_KEY", "development-only-pii-key")
    session_secret = settings.values.get("SESSION_SECRET", "development-only-session-key")
    phone_protector = PhoneProtector(pii_secret)
    service = TravelRestService(
        session_factory,
        redis_store,
        anonymous_ttl_seconds=settings.anonymous_session_ttl_seconds,
        phone_protector=phone_protector,
    )
    configured_sms_provider = sms_provider or _build_configured_sms_provider(settings)
    auth_service = AuthenticationService(
        session_factory,
        redis_store,
        configured_sms_provider,
        phone_protector=phone_protector,
        session_secret=session_secret,
    )
    return RestServiceRuntime(
        engine=engine,
        session_factory=session_factory,
        redis_client=redis_client,
        redis_store=redis_store,
        service=service,
        auth_service=auth_service,
        sms_provider=configured_sms_provider,
    )


def _build_configured_sms_provider(settings: Settings) -> SmsProvider:
    provider_name = settings.values.get("SMS_PROVIDER", "unavailable")
    if provider_name in {"", "unavailable"}:
        return UnavailableSmsProvider()
    if provider_name == "tencent_cloud":
        required = {
            "SMS_API_KEY": "secret_id",
            "SMS_API_SECRET": "secret_key",
            "SMS_APP_ID": "sdk_app_id",
            "SMS_SIGN_NAME": "sign_name",
            "SMS_TEMPLATE_ID": "template_id",
            "SMS_REGION": "region",
        }
        values = _required_sms_values(settings, required)
        return TencentCloudSmsProvider(TencentCloudSmsConfig(**values))
    if provider_name == "aliyun_phone_verification":
        values = _required_sms_values(
            settings,
            {
                "SMS_API_KEY": "access_key_id",
                "SMS_API_SECRET": "access_key_secret",
                "SMS_SIGN_NAME": "sign_name",
                "SMS_TEMPLATE_ID": "template_code",
            },
        )
        return AliyunPhoneVerificationSmsProvider(
            AliyunPhoneVerificationSmsConfig(
                access_key_id=values["access_key_id"],
                access_key_secret=values["access_key_secret"],
                sign_name=values["sign_name"],
                template_code=values["template_code"],
                region=settings.values.get("SMS_REGION", "cn-hangzhou") or "cn-hangzhou",
            )
        )
    raise ConfigurationError("Unsupported SMS_PROVIDER configuration")


def _required_sms_values(settings: Settings, required: dict[str, str]) -> dict[str, str]:
    missing = sorted(name for name in required if not settings.values.get(name, "").strip())
    if missing:
        raise ConfigurationError("Missing server configuration: " + ", ".join(missing))
    return {field: settings.values[name] for name, field in required.items()}


async def default_actor_resolver(connection: HTTPConnection) -> RequestActor:
    anonymous_session_id = connection.headers.get("X-Anonymous-Session-ID", "").strip()
    if not anonymous_session_id:
        anonymous_session_id = connection.cookies.get(ANONYMOUS_SESSION_COOKIE_NAME, "").strip()
    if anonymous_session_id:
        return RequestActor(owner_type=OwnerType.ANONYMOUS, owner_id=anonymous_session_id)
    raise ServiceError(
        "authentication_required",
        "A valid session is required for this operation.",
        status.HTTP_401_UNAUTHORIZED,
    )


def build_actor_resolver(
    auth_service: AuthenticationService,
    *,
    public_app_url: str,
) -> ActorResolver:
    expected = urlsplit(public_app_url)
    expected_origin = f"{expected.scheme}://{expected.netloc}"

    async def resolve(connection: HTTPConnection) -> RequestActor:
        session_token = connection.cookies.get(SESSION_COOKIE_NAME, "").strip()
        if connection.scope.get("type") == "websocket" and session_token:
            origin = connection.headers.get("origin")
            if origin is None or origin.rstrip("/") != expected_origin.rstrip("/"):
                raise ServiceError(
                    "origin_not_allowed",
                    "The WebSocket origin is not allowed.",
                    status.HTTP_403_FORBIDDEN,
                )
        if session_token:
            user_id = await auth_service.resolve_user_id(session_token)
            if user_id is not None:
                return RequestActor(owner_type=OwnerType.USER, owner_id=str(user_id))
        return await default_actor_resolver(connection)

    return resolve


def create_rest_router(
    service: TravelRestService,
    actor_resolver: ActorResolver = default_actor_resolver,
    *,
    auth_service: AuthenticationService | None = None,
    secure_cookies: bool = False,
) -> APIRouter:
    router = APIRouter()

    async def actor(request: Request) -> RequestActor:
        return await actor_resolver(request)

    def auth() -> AuthenticationService:
        if auth_service is None:
            raise ServiceError(
                "authentication_not_available",
                "Authentication is not available.",
                status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        return auth_service

    @router.post("/auth/sms", status_code=status.HTTP_202_ACCEPTED)
    async def send_sms(
        payload: SmsSendRequest,
        request: Request,
        idempotency_key: IdempotencyHeader,
    ) -> Response:
        try:
            await auth().send_code(payload, _auth_context(request), idempotency_key)
        except AuthenticationRateLimitError as exc:
            raise ServiceError(
                "authentication_rate_limited",
                "Too many verification requests. Try again later.",
                status.HTTP_429_TOO_MANY_REQUESTS,
            ) from exc
        except AuthenticationConflictError as exc:
            raise ServiceError("request_conflict", str(exc), status.HTTP_409_CONFLICT) from exc
        except SmsDeliveryError as exc:
            raise ServiceError(
                "sms_delivery_unavailable",
                "The verification message could not be sent.",
                status.HTTP_503_SERVICE_UNAVAILABLE,
            ) from exc
        return Response(status_code=status.HTTP_202_ACCEPTED)

    @router.post("/auth/verify", response_model=AccountView)
    async def verify_sms(
        payload: SmsVerifyRequest,
        request: Request,
        response: Response,
        idempotency_key: IdempotencyHeader,
    ) -> AccountView:
        try:
            verified = await auth().verify_code(payload, _auth_context(request), idempotency_key)
        except InvalidVerificationCodeError as exc:
            raise ServiceError(
                "verification_code_invalid",
                "The verification code is invalid or expired.",
                status.HTTP_401_UNAUTHORIZED,
            ) from exc
        except AuthenticationRateLimitError as exc:
            raise ServiceError(
                "authentication_rate_limited",
                "Too many verification attempts. Try again later.",
                status.HTTP_429_TOO_MANY_REQUESTS,
            ) from exc
        except AuthenticationConflictError as exc:
            raise ServiceError("request_conflict", str(exc), status.HTTP_409_CONFLICT) from exc
        response.set_cookie(
            SESSION_COOKIE_NAME,
            verified.session_token,
            max_age=auth().session_ttl_seconds,
            httponly=True,
            secure=secure_cookies,
            samesite="lax",
            path="/",
        )
        return verified.account

    @router.post("/auth/logout", status_code=status.HTTP_204_NO_CONTENT)
    async def logout(
        request: Request,
        response: Response,
        idempotency_key: IdempotencyHeader,
    ) -> Response:
        del idempotency_key
        current_actor = await actor(request)
        if current_actor.owner_type is not OwnerType.USER:
            raise ServiceError(
                "authentication_required",
                "A user session is required.",
                status.HTTP_401_UNAUTHORIZED,
            )
        token = request.cookies.get(SESSION_COOKIE_NAME, "")
        if token:
            await auth().logout(token)
        response.delete_cookie(
            SESSION_COOKIE_NAME,
            httponly=True,
            secure=secure_cookies,
            samesite="lax",
            path="/",
        )
        response.status_code = status.HTTP_204_NO_CONTENT
        return response

    @router.post(
        "/anonymous-sessions",
        response_model=AnonymousSessionView,
        status_code=status.HTTP_201_CREATED,
    )
    async def create_anonymous_session(
        idempotency_key: IdempotencyHeader,
        request: Request,
        response: Response,
    ) -> AnonymousSessionView:
        existing_session_id = (
            request.headers.get("X-Anonymous-Session-ID", "").strip()
            or request.cookies.get(ANONYMOUS_SESSION_COOKIE_NAME, "").strip()
            or None
        )
        view = await service.create_anonymous_session(
            idempotency_key,
            existing_session_id,
        )
        response.set_cookie(
            ANONYMOUS_SESSION_COOKIE_NAME,
            view.session_id,
            httponly=True,
            secure=secure_cookies,
            samesite="lax",
            path="/",
        )
        return view

    @router.delete("/anonymous-sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
    async def delete_anonymous_session(
        session_id: str,
        request: Request,
        idempotency_key: IdempotencyHeader,
    ) -> Response:
        await service.delete_anonymous_session(await actor(request), session_id, idempotency_key)
        response = Response(status_code=status.HTTP_204_NO_CONTENT)
        response.delete_cookie(
            ANONYMOUS_SESSION_COOKIE_NAME,
            httponly=True,
            secure=secure_cookies,
            samesite="lax",
            path="/",
        )
        return response

    @router.get("/account", response_model=AccountView)
    async def get_account(request: Request) -> AccountView:
        return await service.get_account(await actor(request))

    @router.patch("/account", response_model=AccountView)
    async def update_account(
        payload: UpdateAccountRequest,
        request: Request,
        idempotency_key: IdempotencyHeader,
    ) -> AccountView:
        return await service.update_account(await actor(request), payload, idempotency_key)

    @router.delete("/account", status_code=status.HTTP_204_NO_CONTENT)
    async def delete_account(
        request: Request,
        idempotency_key: IdempotencyHeader,
    ) -> Response:
        current_actor = await actor(request)
        await service.delete_account(current_actor, idempotency_key)
        if auth_service is not None:
            await auth_service.logout_all(UUID(current_actor.owner_id))
        response = Response(status_code=status.HTTP_204_NO_CONTENT)
        response.delete_cookie(
            SESSION_COOKIE_NAME,
            httponly=True,
            secure=secure_cookies,
            samesite="lax",
            path="/",
        )
        return response

    @router.post("/auth/attach-trip", response_model=TripSnapshotView)
    async def attach_trip(
        payload: AttachAnonymousTripRequest,
        request: Request,
        response: Response,
        idempotency_key: IdempotencyHeader,
    ) -> TripSnapshotView:
        attached = await service.attach_anonymous_trip(
            await actor(request), payload, idempotency_key
        )
        response.delete_cookie(
            ANONYMOUS_SESSION_COOKIE_NAME,
            httponly=True,
            secure=secure_cookies,
            samesite="lax",
            path="/",
        )
        return attached

    @router.get("/trips", response_model=TripListView)
    async def list_trips(request: Request) -> TripListView:
        return await service.list_trips(await actor(request))

    @router.post("/trips", response_model=TripSnapshotView, status_code=status.HTTP_201_CREATED)
    async def create_trip(
        payload: CreateTripRequest,
        request: Request,
        idempotency_key: IdempotencyHeader,
    ) -> TripSnapshotView:
        return await service.create_trip(await actor(request), payload, idempotency_key)

    @router.get("/trips/{trip_id}/shell", response_model=TripShell)
    async def get_trip_shell(trip_id: UUID, request: Request) -> TripShell:
        return await service.get_trip_shell(await actor(request), trip_id)

    @router.get("/trips/{trip_id}", response_model=TripSnapshotView)
    async def get_trip(trip_id: UUID, request: Request) -> TripSnapshotView:
        return await service.get_trip(await actor(request), trip_id)

    @router.delete("/trips/{trip_id}", status_code=status.HTTP_204_NO_CONTENT)
    async def delete_trip(
        trip_id: UUID,
        request: Request,
        idempotency_key: IdempotencyHeader,
    ) -> Response:
        await service.delete_trip(await actor(request), trip_id, idempotency_key)
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @router.get("/preferences", response_model=PreferenceListView)
    async def get_preferences(request: Request) -> PreferenceListView:
        return await service.get_preferences(await actor(request))

    @router.patch("/preferences", response_model=PreferenceListView)
    async def patch_preferences(
        payload: PatchPreferencesRequest,
        request: Request,
        idempotency_key: IdempotencyHeader,
    ) -> PreferenceListView:
        return await service.patch_preferences(await actor(request), payload, idempotency_key)

    @router.post("/preferences/cold-start", response_model=PreferenceListView)
    async def save_cold_start_preference(
        payload: ColdStartSubmission,
        request: Request,
        idempotency_key: IdempotencyHeader,
    ) -> PreferenceListView:
        return await service.save_cold_start_preference(
            await actor(request), payload, idempotency_key
        )

    @router.delete("/preferences/{preference_id}", status_code=status.HTTP_204_NO_CONTENT)
    async def delete_preference(
        preference_id: UUID,
        request: Request,
        idempotency_key: IdempotencyHeader,
    ) -> Response:
        await service.delete_preference(await actor(request), preference_id, idempotency_key)
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @router.post(
        "/trips/{trip_id}/preference-candidates/confirm",
        response_model=PreferenceListView,
    )
    async def confirm_preference_candidates(
        trip_id: UUID,
        payload: ConfirmPreferenceCandidatesRequest,
        request: Request,
        idempotency_key: IdempotencyHeader,
    ) -> PreferenceListView:
        return await service.confirm_preference_candidates(
            await actor(request), trip_id, payload, idempotency_key
        )

    @router.post(
        "/trips/{trip_id}/versions/{plan_version_id}/exports",
        response_model=ExportView,
        status_code=status.HTTP_201_CREATED,
    )
    async def create_export(
        trip_id: UUID,
        plan_version_id: UUID,
        payload: CreateExportRequest,
        request: Request,
        idempotency_key: IdempotencyHeader,
    ) -> ExportView:
        return await service.create_export(
            await actor(request),
            trip_id,
            plan_version_id,
            payload,
            idempotency_key,
        )

    @router.get("/exports/{artifact_id}", response_model=ExportView)
    async def get_export(artifact_id: UUID, request: Request) -> ExportView:
        return await service.get_export(await actor(request), artifact_id)

    return router


def _auth_context(request: Request) -> AuthClientContext:
    ip_address = request.client.host if request.client is not None else "unknown"
    device_id = request.headers.get("X-Device-ID", "").strip()
    if not device_id:
        device_id = f"unidentified:{ip_address}:{request.headers.get('user-agent', '')}"
    return AuthClientContext(ip_address=ip_address, device_id=device_id)
