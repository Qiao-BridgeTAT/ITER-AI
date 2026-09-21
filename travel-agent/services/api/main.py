"""FastAPI process factory for the stage-0 modular monolith."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Literal
from uuid import UUID

from fastapi import FastAPI, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from starlette import status

from backend.agent.factory import ModelGatewayRuntime, build_model_gateway
from backend.agent.planner.evidence import PlannerEvidenceBackend
from backend.agent.planner.graph import PlannerAgentGraph
from backend.agent.prepare.graph import PrepareAgentGraph
from backend.agent.schema_gateway import SchemaModelGateway
from backend.application.auth_service import AuthenticationService
from backend.application.place_introduction_service import PlaceIntroductionService
from backend.application.plan_preview_service import PlanPreviewService
from backend.application.planner_journey_service import PlannerJourneyService
from backend.application.prepare_journey_service import (
    PrepareJourneyError,
    PrepareJourneyService,
)
from backend.application.published_plan_view import visible_published_plan
from backend.application.realtime_command_service import (
    RealtimeStateConflictError,
)
from backend.application.rest_service import TravelRestService
from backend.application.v4_owner_resolver import V4OwnerResolver
from backend.config import Settings
from backend.contracts.v4.conversation import (
    ConversationHistoryPage,
    ConversationMessageV4,
    ConversationSnapshotV4,
    ConversationView,
)
from backend.contracts.v4.place_introduction import PlaceIntroductionView
from backend.contracts.v4.plan_preview import PlannerPlanPreview
from backend.discovery.cards import PrepareCardService
from backend.discovery.cards.attraction_media import AttractionMediaResolver
from backend.discovery.cards.attraction_recall import AttractionRecallService
from backend.discovery.cards.attraction_selection import AttractionCandidateSelector
from backend.discovery.cards.candidate_composition import CandidateCompositionService
from backend.discovery.cards.dining.pipeline import DiningDiscovery
from backend.discovery.cards.preference_generation import PreferenceDirectionGenerator
from backend.discovery.tools.registry import PrepareToolExecutor
from backend.domain.authorization import RequestActor
from backend.persistence import DependencyRegistry, build_dependency_registry
from backend.persistence.checkpoint_repository import CheckpointRepository
from backend.persistence.outbox_repository import OutboxRepository
from backend.persistence.turn_repository import (
    InvalidTurnWriteError,
    TurnNotFoundError,
    TurnRepository,
)
from backend.planning.candidate_ranking import CandidateRankingService
from backend.planning.candidate_recall import CandidateRecallService
from backend.planning.city_registry import default_city_registry
from backend.planning.recall_plan import ModelRecallPlanGenerator
from backend.planning.runtime_backend import (
    PlanningProviderSet,
)
from backend.providers.factory import ProviderGatewayRuntime, build_provider_gateway
from services.api.errors import ServiceError, install_error_handlers
from services.api.middleware import RequestCorrelationMiddleware
from services.api.rest import (
    ActorResolver,
    RestServiceRuntime,
    build_actor_resolver,
    build_rest_runtime,
    create_rest_router,
    default_actor_resolver,
)

logger = logging.getLogger("uvicorn.error")


@dataclass
class _OutboundFrame:
    payload: dict[str, Any]
    delivered: asyncio.Future[bool]


def create_app(
    settings: Settings,
    dependencies: DependencyRegistry | None = None,
    *,
    rest_service: TravelRestService | None = None,
    actor_resolver: ActorResolver | None = None,
    auth_service: AuthenticationService | None = None,
    prepare_service: PrepareJourneyService | None = None,
    planner_service: PlannerJourneyService | None = None,
) -> FastAPI:
    registry = dependencies or build_dependency_registry(settings)
    rest_runtime: RestServiceRuntime | None = None
    provider_runtime: ProviderGatewayRuntime | None = None
    model_runtime: ModelGatewayRuntime = build_model_gateway(settings)
    prepared_evidence = None
    planning_pool = None
    if rest_service is None:
        rest_runtime = build_rest_runtime(settings)
        rest_service = rest_runtime.service
        auth_service = rest_runtime.auth_service
    if prepare_service is None and rest_runtime is not None:
        if provider_runtime is None:
            provider_runtime = build_provider_gateway(settings, rest_runtime.redis_store)

        def prepare_tools(business_date):  # type: ignore[no-untyped-def]
            assert provider_runtime is not None
            return PrepareToolExecutor(
                places=provider_runtime.places,
                hours=provider_runtime.hours,
                routes=provider_runtime.routes,
                products=provider_runtime.products,
                weather=provider_runtime.weather,
                business_date=business_date,
            )

        prepare_cards = None
        from backend.discovery.prepared_evidence import PreparedEvidenceCollector
        from backend.persistence.prepared_evidence_repository import PreparedEvidenceRepository

        prepared_evidence = PreparedEvidenceCollector(
            PreparedEvidenceRepository(rest_runtime.session_factory),
            default_city_registry(),
            provider_runtime.hours,
            provider_runtime.products,
        )
        provider_runtime.closeables = (prepared_evidence, *provider_runtime.closeables)
        if provider_runtime.places is not None:
            prepare_recall = CandidateRecallService(
                registry=default_city_registry(),
                places=provider_runtime.places,
                plan_generator=ModelRecallPlanGenerator(
                    model_runtime.gateway,
                    provider_only=True,
                ),
            )
            attraction_gateway = model_runtime.attraction_gateway or model_runtime.gateway
            from backend.discovery.planning_candidates import PreparedPlanningPoolService
            from backend.persistence.prepared_candidates_repository import (
                PreparedCandidatesRepository,
            )

            planning_pool = PreparedPlanningPoolService(
                repository=PreparedCandidatesRepository(rest_runtime.session_factory),
                turns=TurnRepository(rest_runtime.session_factory),
                facts=prepared_evidence,
                gateway=attraction_gateway,
                places=provider_runtime.places,
                registry=default_city_registry(),
            )
            provider_runtime.closeables = (planning_pool, *provider_runtime.closeables)
            attraction_recall = AttractionRecallService(
                gateway=attraction_gateway,
                places=provider_runtime.places,
                registry=default_city_registry(),
                store=rest_runtime.redis_store,
            )
            dining_discovery = DiningDiscovery(
                gateway=attraction_gateway,
                places=provider_runtime.places,
                registry=default_city_registry(),
                store=rest_runtime.redis_store,
            )
            prepare_cards = PrepareCardService(
                directions=PreferenceDirectionGenerator(
                    model_runtime.gateway,
                    attraction_gateway=attraction_gateway,
                    dining_gateway=attraction_gateway,
                ),
                candidates=CandidateCompositionService(
                    recall=prepare_recall,
                    ranking=CandidateRankingService(),
                    attraction_selector=AttractionCandidateSelector(
                        attraction_gateway, parallel_discovery=True
                    ),
                    attraction_recall=attraction_recall,
                    dining_discovery=dining_discovery,
                    attraction_media=AttractionMediaResolver(
                        registry=default_city_registry(),
                        places=provider_runtime.places,
                        products=provider_runtime.products,
                    ),
                ),
                registry=default_city_registry(),
                places=provider_runtime.places,
                products=provider_runtime.products,
                evidence_collector=prepared_evidence,
            )

        v4_owner_resolver = V4OwnerResolver(
            rest_runtime.session_factory,
            rest_runtime.redis_store,
            anonymous_ttl_seconds=settings.anonymous_session_ttl_seconds,
        )
        prepare_service = PrepareJourneyService(
            graph=PrepareAgentGraph(
                model_runtime.gateway,
                attraction_gateway=model_runtime.attraction_gateway,
                dining_gateway=SchemaModelGateway(
                    model_runtime.attraction_gateway or model_runtime.gateway
                ),
            ),
            turns=TurnRepository(rest_runtime.session_factory),
            outbox=OutboxRepository(rest_runtime.session_factory),
            temporary=rest_runtime.redis_store,
            tool_executor_factory=prepare_tools,
            card_service=prepare_cards,
            planning_pool=planning_pool,
            timezone=settings.values.get("DEFAULT_TIMEZONE", "Asia/Shanghai"),
            owner_resolver=v4_owner_resolver.resolve,
            model_audit=model_runtime.audit_recorder,
        )

    if (
        planner_service is None
        and prepare_service is not None
        and rest_runtime is not None
        and provider_runtime is not None
        and settings.provider_mode == "live"
        and settings.model.enabled
        and settings.v4_planner_enabled
        and provider_runtime.places is not None
        and provider_runtime.routes is not None
        and provider_runtime.hours is not None
        and provider_runtime.products is not None
    ):
        from backend.agent.planner.react_graph import PlannerEngineRouter
        from backend.providers.amap_mcp import AmapMcpClient, AmapMcpRouter
        from backend.providers.amap_mcp_adapters import AmapMcpPlaceProvider, AmapMcpRouteProvider
        from backend.providers.tavily_mcp import TavilyMcpClient

        mcp_client = AmapMcpClient(
            settings.values["AMAP_WEB_SERVICE_KEY"], rate_limiter=rest_runtime.redis_store
        )
        provider_runtime.closeables = (*provider_runtime.closeables, mcp_client)
        local_search = None
        if endpoint := settings.values.get("AMAP_SEARCH_MCP_URL", "").strip():
            local_search = AmapMcpClient("", endpoint=endpoint)
            provider_runtime.closeables = (*provider_runtime.closeables, local_search)
        mcp_router = AmapMcpRouter(mcp_client, local_search)
        legacy_planner = PlannerAgentGraph(
            model_runtime.gateway,
            PlannerEvidenceBackend(
                PlanningProviderSet(
                    places=provider_runtime.places,
                    routes=provider_runtime.routes,
                    hours=provider_runtime.hours,
                    products=provider_runtime.products,
                    weather=provider_runtime.weather,
                ),
                model_runtime.gateway,
                dining_review_gateway=model_runtime.dining_review_gateway,
            ),
            complete_plan=True,
            compact_planning=True,
            estimate_visit_durations=True,
            optimize_timing=True,
        )
        if model_runtime.planner_gateway is None:
            raise ValueError("Planner Flash gateway was not configured")
        planner_gateway = SchemaModelGateway(model_runtime.planner_gateway)
        react_evidence = PlannerEvidenceBackend(
            PlanningProviderSet(
                places=AmapMcpPlaceProvider(mcp_router),
                routes=AmapMcpRouteProvider(mcp_router),
                hours=provider_runtime.hours,
                products=provider_runtime.products,
                weather=provider_runtime.weather,
            ),
            planner_gateway,
        )
        planner_service = PlannerJourneyService(
            prepare=prepare_service,
            graph=PlannerEngineRouter(
                legacy_planner,
                react_evidence,
                mcp=mcp_router,
                react_gateway=planner_gateway,
                web_search=TavilyMcpClient(
                    api_key=settings.tavily_mcp.api_key,
                    auth_mode=settings.tavily_mcp.auth_mode,
                )
                if settings.tavily_mcp
                else None,
                new_engine=settings.values.get("V4_PLANNER_ENGINE", "legacy")
                == "langgraph-react-2",
                time_limit_enabled=settings.values.get("V4_PLANNER_TIME_LIMIT_ENABLED", "true")
                == "true",
                max_decisions=int(settings.values.get("V4_PLANNER_MAX_DECISIONS", "24")),
            ),
            prepared_evidence=prepared_evidence,
            planning_pool=planning_pool,
            turns=TurnRepository(rest_runtime.session_factory),
            checkpoints=CheckpointRepository(rest_runtime.session_factory),
            temporary=rest_runtime.redis_store,
            model_audit=model_runtime.audit_recorder,
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
        app.state.dependencies = registry
        app.state.model_gateway = model_runtime.gateway
        await registry.start()
        try:
            await model_runtime.start()
            yield
        finally:
            if provider_runtime is not None:
                await provider_runtime.close()
            await model_runtime.close()
            if rest_runtime is not None:
                await rest_runtime.close()
            await registry.close()

    app = FastAPI(title="Travel Agent API", version="2.0.0", lifespan=lifespan)
    app.state.prepare_agent_wired = prepare_service is not None
    app.state.planner_agent_wired = planner_service is not None
    plan_preview_service = (
        PlanPreviewService(
            provider_runtime.places, provider_runtime.products, provider_runtime.weather
        )
        if provider_runtime is not None and provider_runtime.places is not None
        else None
    )
    place_introductions = PlaceIntroductionService(
        model_runtime.gateway,
        card_gateway=model_runtime.attraction_gateway or model_runtime.gateway,
    )
    app.add_middleware(RequestCorrelationMiddleware)
    install_error_handlers(app)
    effective_actor_resolver = actor_resolver
    if effective_actor_resolver is None and auth_service is not None:
        effective_actor_resolver = build_actor_resolver(
            auth_service, public_app_url=settings.values["PUBLIC_APP_URL"]
        )
    router = create_rest_router(
        rest_service,
        actor_resolver=effective_actor_resolver or default_actor_resolver,
        auth_service=auth_service,
        secure_cookies=settings.values["PUBLIC_APP_URL"].startswith("https://"),
    )
    app.include_router(router)
    app.include_router(router, prefix="/api")

    @app.get("/health/live", tags=["health"])
    async def liveness() -> dict[str, str]:
        return {"status": "alive"}

    @app.get("/health/ready", tags=["health"])
    async def readiness(request: Request) -> JSONResponse:
        report = await request.app.state.dependencies.check()
        payload: dict[str, Any] = {
            "status": "ready" if report.ready else "unavailable",
            "dependencies": [
                {
                    "name": item.name,
                    "status": "available" if item.available else "unavailable",
                    "critical": item.critical,
                    **({"reason": item.reason} if item.reason else {}),
                }
                for item in report.dependencies
            ],
        }
        code = status.HTTP_200_OK if report.ready else status.HTTP_503_SERVICE_UNAVAILABLE
        return JSONResponse(status_code=code, content=payload)

    @app.get(
        "/v4/trips/{trip_id}",
        response_model=ConversationSnapshotV4,
        tags=["v4"],
    )
    @app.get(
        "/api/v4/trips/{trip_id}",
        response_model=ConversationSnapshotV4,
        tags=["v4"],
    )
    async def v4_trip_snapshot(
        request: Request,
        trip_id: UUID,
    ) -> ConversationSnapshotV4:
        if prepare_service is None:
            raise ServiceError(
                "prepare_agent_not_available",
                "The V4 Prepare Agent is not available.",
                status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        resolver = effective_actor_resolver or default_actor_resolver
        actor = await resolver(request)
        try:
            return await (planner_service or prepare_service).get_snapshot(actor, trip_id)
        except TurnNotFoundError as exc:
            raise ServiceError(
                "trip_not_found",
                "The trip was not found.",
                status.HTTP_404_NOT_FOUND,
            ) from exc
        except PrepareJourneyError as exc:
            raise _prepare_service_error(exc) from exc

    async def history_actor(request: Request) -> RequestActor:
        if prepare_service is None:
            raise ServiceError("prepare_agent_not_available", "V4 is not available.", 503)
        return await (effective_actor_resolver or default_actor_resolver)(request)

    def history_error(exc: Exception) -> ServiceError:
        if isinstance(exc, TurnNotFoundError):
            return ServiceError("trip_not_found", "The trip or message was not found.", 404)
        if isinstance(exc, PrepareJourneyError):
            return _prepare_service_error(exc)
        return ServiceError("v4_snapshot_incompatible", "Saved trip data cannot be restored.", 409)

    @app.get("/api/v4/trips/{trip_id}/view", response_model=ConversationView, tags=["v4"])
    async def v4_conversation_view(request: Request, trip_id: UUID) -> ConversationView:
        actor = await history_actor(request)
        assert prepare_service is not None
        try:
            return await (planner_service or prepare_service).get_conversation_view(actor, trip_id)
        except (TurnNotFoundError, PrepareJourneyError, InvalidTurnWriteError) as exc:
            raise history_error(exc) from exc

    @app.get("/api/v4/trips/{trip_id}/history", response_model=ConversationHistoryPage, tags=["v4"])
    async def v4_conversation_history(
        request: Request,
        trip_id: UUID,
        before_state_version: int = Query(ge=0),
        through_state_version: int = Query(ge=0),
    ) -> ConversationHistoryPage:
        actor = await history_actor(request)
        assert prepare_service is not None
        try:
            return await prepare_service.get_conversation_history(
                actor,
                trip_id,
                before_state_version=before_state_version,
                through_state_version=through_state_version,
            )
        except (TurnNotFoundError, PrepareJourneyError, InvalidTurnWriteError) as exc:
            raise history_error(exc) from exc

    @app.get(
        "/api/v4/trips/{trip_id}/messages/{message_id}",
        response_model=ConversationMessageV4,
        tags=["v4"],
    )
    async def v4_conversation_message(
        request: Request,
        trip_id: UUID,
        message_id: UUID,
    ) -> ConversationMessageV4:
        actor = await history_actor(request)
        assert prepare_service is not None
        try:
            return await prepare_service.get_conversation_message(actor, trip_id, message_id)
        except (TurnNotFoundError, PrepareJourneyError, InvalidTurnWriteError) as exc:
            raise history_error(exc) from exc

    @app.get(
        "/api/v4/trips/{trip_id}/plan-preview",
        response_model=PlannerPlanPreview,
        tags=["v4"],
    )
    async def v4_plan_preview(
        request: Request, trip_id: UUID, plan_version_id: UUID
    ) -> PlannerPlanPreview:
        # Reuse the authenticated snapshot boundary BEFORE inspecting/caching
        # any private trip data. This endpoint cannot trigger a Planner turn.
        snapshot = (await v4_conversation_view(request, trip_id)).snapshot
        plan = visible_published_plan(snapshot)
        if plan is None or plan.plan_version_id != plan_version_id:
            raise ServiceError(
                "plan_version_conflict",
                "The published plan has changed.",
                status.HTTP_409_CONFLICT,
            )
        if plan_preview_service is None:
            return PlannerPlanPreview(trip_id=trip_id, plan_version_id=plan_version_id)
        return await plan_preview_service.get_preview(snapshot)

    @app.get(
        "/api/v4/trips/{trip_id}/place-introductions",
        response_model=PlaceIntroductionView,
        tags=["v4"],
    )
    async def v4_place_introductions(
        request: Request,
        trip_id: UUID,
        scope_kind: Literal["card", "plan"],
        scope_id: UUID,
    ) -> PlaceIntroductionView:
        snapshot = (await v4_conversation_view(request, trip_id)).snapshot
        try:
            return await place_introductions.get_view(
                snapshot, trip_id=trip_id, scope_kind=scope_kind, scope_id=scope_id
            )
        except ValueError as exc:
            raise ServiceError(
                "introduction_scope_not_found",
                "This card or plan is no longer in the current view.",
                status.HTTP_404_NOT_FOUND,
            ) from exc

    @app.websocket("/trips/{trip_id}/stream")
    @app.websocket("/api/trips/{trip_id}/stream")
    async def trip_stream(websocket: WebSocket, trip_id: UUID) -> None:
        await websocket.accept()
        outbound: asyncio.Queue[_OutboundFrame | None] = asyncio.Queue()
        sender_finished = asyncio.Event()
        command_tasks: set[asyncio.Task[None]] = set()

        async def send_loop() -> None:
            try:
                while True:
                    queued = await outbound.get()
                    if queued is None:
                        return
                    await websocket.send_json(queued.payload)
                    if not queued.delivered.done():
                        queued.delivered.set_result(True)
            except (WebSocketDisconnect, RuntimeError):
                return
            finally:
                sender_finished.set()
                while not outbound.empty():
                    queued = outbound.get_nowait()
                    if queued is not None and not queued.delivered.done():
                        queued.delivered.set_result(False)

        sender_task = asyncio.create_task(send_loop())

        async def enqueue(
            payload: dict[str, Any],
        ) -> bool:
            if sender_finished.is_set():
                return False
            delivered: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
            await outbound.put(_OutboundFrame(payload, delivered))
            return await delivered

        async def run_command(
            actor: RequestActor,
            frame: Any,
            admitted: asyncio.Event,
        ) -> None:
            try:
                if isinstance(frame, dict) and frame.get("protocol_version") == "v4":
                    if prepare_service is None:
                        await enqueue(
                            {
                                "type": "transport.error",
                                "code": "prepare_agent_not_available",
                                "retryable": False,
                            }
                        )
                        return
                    await (planner_service or prepare_service).handle_stream(
                        actor,
                        trip_id,
                        frame,
                        enqueue,
                        on_admitted=admitted.set,
                    )
                    return
                await enqueue(
                    {
                        "type": "transport.error",
                        "code": "unsupported_protocol",
                        "retryable": False,
                    }
                )
            except ServiceError as exc:
                logger.warning(
                    "Realtime command rejected for trip %s: %s",
                    trip_id,
                    exc.code,
                )
                await enqueue(
                    {
                        "type": "transport.error",
                        "code": exc.code,
                        "retryable": False,
                    }
                )
            except PrepareJourneyError as exc:
                logger.warning(
                    "V4 realtime command rejected for trip %s: %s",
                    trip_id,
                    exc.code,
                )
                await enqueue(
                    {
                        "type": "transport.error",
                        "code": exc.code,
                        "retryable": exc.retryable,
                    }
                )
            except RealtimeStateConflictError:
                logger.warning("Realtime command conflicted for trip %s", trip_id)
                await enqueue(
                    {
                        "type": "transport.error",
                        "code": "trip_busy",
                        "retryable": True,
                    }
                )
            except Exception:
                logger.exception("Realtime command failed for trip %s", trip_id)
                await enqueue(
                    {
                        "type": "transport.error",
                        "code": "internal_error",
                        "retryable": True,
                    }
                )
            finally:
                admitted.set()

        try:
            while True:
                frame = await websocket.receive_json()
                logger.info(
                    "Realtime command received for trip %s: protocol=%s type=%s",
                    trip_id,
                    frame.get("protocol_version") if isinstance(frame, dict) else None,
                    frame.get("type") if isinstance(frame, dict) else None,
                )
                if isinstance(frame, dict) and frame.get("type") == "transport.ping":
                    await enqueue(
                        {
                            "type": "transport.pong",
                            "correlation_id": frame.get("correlation_id"),
                        }
                    )
                    continue
                try:
                    resolver = effective_actor_resolver or default_actor_resolver
                    actor = await resolver(websocket)
                except ServiceError as exc:
                    logger.warning(
                        "Realtime actor resolution failed for trip %s: %s",
                        trip_id,
                        exc.code,
                    )
                    await enqueue(
                        {
                            "type": "transport.error",
                            "code": exc.code,
                            "retryable": False,
                        }
                    )
                    continue
                admitted = asyncio.Event()
                task = asyncio.create_task(run_command(actor, frame, admitted))
                command_tasks.add(task)
                task.add_done_callback(command_tasks.discard)
                await admitted.wait()
        except WebSocketDisconnect:
            pass
        finally:
            for task in command_tasks:
                task.cancel()
            await asyncio.gather(*command_tasks, return_exceptions=True)
            if not sender_task.done():
                await outbound.put(None)
            await asyncio.gather(sender_task, return_exceptions=True)

    return app


def _prepare_service_error(error: PrepareJourneyError) -> ServiceError:
    if error.code in {
        "v4_user_session_required",
        "v4_anonymous_session_required",
        "v4_session_required",
        "invalid_user_session",
    }:
        http_status = status.HTTP_401_UNAUTHORIZED
    elif error.code == "v4_trip_not_found":
        http_status = status.HTTP_404_NOT_FOUND
    elif error.code in {
        "generation_conflict",
        "idempotent_turn_is_terminal",
        "trip_busy",
    }:
        http_status = status.HTTP_409_CONFLICT
    elif error.code == "v4_interaction_not_issued":
        http_status = status.HTTP_422_UNPROCESSABLE_ENTITY
    else:
        http_status = status.HTTP_400_BAD_REQUEST
    return ServiceError(error.code, error.code, http_status)


def create_default_app() -> FastAPI:
    return create_app(Settings.from_environment())
