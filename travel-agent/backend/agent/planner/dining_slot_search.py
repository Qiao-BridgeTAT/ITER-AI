"""Bounded polygon recall for one meal; unchosen POIs never enter the Planner pool."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import date

from backend.agent.model_gateway import ModelCancellation
from backend.agent.planner.daily_repair import execution_remaining
from backend.agent.planner.dining_context import dining_place_blocked, same_dining_entity
from backend.agent.planner.evidence import PlannerEvidenceBackend, _place_evidence, _real_category
from backend.agent.planner.workspace import advance, entity_intents, server_id
from backend.contracts.enums import ProviderCode
from backend.contracts.places import Gcj02Coordinates
from backend.contracts.v4.enums import CandidateEntityKind, PlannerCapability
from backend.contracts.v4.planner_evidence import (
    PlannerCapabilityObservation,
    PlannerGuardObservation,
    PlannerPlaceEvidence,
)
from backend.contracts.v4.planner_workspace import PlannerWorkspaceState
from backend.contracts.v4.task_book import TaskBookV4
from backend.planning.dining_geometry import corridor_polygon
from backend.providers.contracts import PolygonPlaceSearchRequest, ProviderError

MAX_SLOT_PAGES = 3
MAX_SLOT_CHOICES = 2


@dataclass(frozen=True)
class DiningSlotSearchResult:
    workspace: PlannerWorkspaceState
    candidates: tuple[PlannerPlaceEvidence, ...] = ()
    failure_code: str | None = None
    terminal: bool = False


def slot_search_id(workspace: PlannerWorkspaceState, slot_key: str, page: int) -> str:
    return server_id(workspace.generation_id, "dining-polygon", slot_key, page)


def slot_choice_id(workspace: PlannerWorkspaceState, slot_key: str, attempt: int) -> str:
    return server_id(workspace.generation_id, "dining-slot-choice", slot_key, attempt)


def rejected_slot_candidates(workspace: PlannerWorkspaceState, slot_key: str) -> set[str]:
    result: set[str] = set()
    for observation in workspace.guard_observations:
        if observation.code not in {
            "planner_dining_candidate_rejected",
            "planner_dining_batch_observed",
        }:
            continue
        try:
            value = json.loads(observation.message)
        except (ValueError, TypeError):
            continue
        if value.get("slot_key") == slot_key:
            result.update(value.get("candidate_ids", ()))
            if value.get("candidate_id"):
                result.add(value["candidate_id"])
    return result


async def supplement_dining_slot(
    backend: PlannerEvidenceBackend,
    workspace: PlannerWorkspaceState,
    book: TaskBookV4,
    cancellation: ModelCancellation,
    *,
    slot_key: str,
    service_date: date,
    before: Gcj02Coordinates | None,
    after: Gcj02Coordinates | None,
    page: int,
    excluded_ids: set[str] | None = None,
    checkpoint: Callable[[PlannerWorkspaceState], Awaitable[None]] | None = None,
) -> DiningSlotSearchResult:
    """One logical page in provider order, with classified failure and no pool mutation."""
    if execution_remaining(workspace, calls=True) <= 0:
        return DiningSlotSearchResult(workspace, failure_code="budget_exhausted", terminal=True)
    if not 1 <= page <= MAX_SLOT_PAGES:
        return DiningSlotSearchResult(
            workspace, failure_code="page_budget_exhausted", terminal=True
        )
    request_id = slot_search_id(workspace, slot_key, page)
    terminal_receipt = next(
        (
            item
            for item in workspace.capability_observations
            if item.request_id
            in {slot_search_id(workspace, slot_key, p) for p in range(1, MAX_SLOT_PAGES + 1)}
            and item.status == "unavailable"
        ),
        None,
    )
    if terminal_receipt is not None:
        return DiningSlotSearchResult(
            workspace,
            failure_code=terminal_receipt.reason_summary.rsplit("；", 1)[-1],
            terminal=True,
        )
    if any(item.request_id == request_id for item in workspace.capability_observations):
        return DiningSlotSearchResult(workspace, failure_code="page_already_attempted")
    try:
        polygon = corridor_polygon(before, after)
    except ValueError:
        return DiningSlotSearchResult(
            workspace, failure_code="endpoint_coordinates_missing", terminal=True
        )
    city = backend.registry.provider_scope(
        book.destination_and_dates.destination_name, ProviderCode.AMAP
    )
    excluded = (
        {
            identity
            for identity, (intent, _) in entity_intents(book).items()
            if intent.disposition.value == "avoid"
        }
        | (excluded_ids or set())
        | rejected_slot_candidates(workspace, slot_key)
    )
    excluded.update(
        entry.candidate_ref.canonical_entity_id
        for entry in workspace.candidate_pool.candidates
        if entry.selection_permission == "forbidden"
        or entry.eligibility in {"excluded", "unavailable"}
        or service_date in entry.infeasible_dates
        or (entry.feasible_dates and service_date not in entry.feasible_dates)
    )
    excluded_places = [
        place for place in workspace.place_evidence if place.canonical_entity_id in excluded
    ]
    candidates: dict[str, PlannerPlaceEvidence] = {}
    failure_code = None
    terminal = False
    started = PlannerCapabilityObservation(
        observation_id=request_id,
        scope=workspace.current_scope,
        request_id=request_id,
        capability=PlannerCapability.CANDIDATE_RECALL,
        status="partial",
        reason_summary=f"餐次矩形补查：{slot_key}；page={page}；started",
        observed_at=backend.clock(),
    )
    workspace = advance(
        workspace, capability_observations=(*workspace.capability_observations, started)
    )
    if checkpoint:
        await checkpoint(workspace)
    try:
        cancellation.raise_if_cancelled("planner_dining_polygon")
        remaining = execution_remaining(workspace, calls=True)
        if remaining <= 0:
            return DiningSlotSearchResult(workspace, failure_code="budget_exhausted", terminal=True)
        async with asyncio.timeout(min(15, remaining)):
            response = await backend.providers.places.search_polygon(
                PolygonPlaceSearchRequest(city=city, polygon=polygon, page=page, page_size=20)
            )
        cancellation.raise_if_cancelled("planner_dining_polygon_result")
        for item in response.items:
            if (
                not _real_category(item, CandidateEntityKind.RESTAURANT)
                or item.city_id != city.city_id
            ):
                continue
            if re.search(
                r"暂停营业|暂停开放|暂不营业|已关闭|已停业|歇业|永久关闭|已搬迁|temporarilyclosed|permanentlyclosed",
                "".join(item.name.casefold().split()),
            ):
                continue
            place = _place_evidence(item, CandidateEntityKind.RESTAURANT)
            if dining_place_blocked(place, book) or any(
                same_dining_entity(place, other)
                for other in (*excluded_places, *candidates.values())
            ):
                continue
            if place.canonical_entity_id not in excluded:
                candidates.setdefault(place.canonical_entity_id, place)
            if len(candidates) >= 20:
                break
        if not candidates:
            failure_code = "filtered_empty" if response.items else "provider_empty"
        if response.failures:
            failure_code = next(iter(response.failures.values())).code.value
    except ProviderError as error:
        failure_code = error.code.value
        terminal = not error.retryable or failure_code in {
            "authentication_failed",
            "permission_denied",
            "rate_limited",
            "invalid_request",
        }
    observation = PlannerCapabilityObservation(
        observation_id=request_id,
        scope=workspace.current_scope,
        request_id=request_id,
        capability=PlannerCapability.CANDIDATE_RECALL,
        # Temporary candidates are deliberately not asserted as workspace facts.
        status="unavailable" if terminal else "partial",
        reason_summary=(
            f"餐次矩形补查：{slot_key}；page={page}；count={len(candidates)}；"
            f"{failure_code or 'candidates_observed'}"
        ),
        observed_at=backend.clock(),
    )
    result = advance(
        workspace,
        capability_observations=tuple(
            observation if item.request_id == request_id else item
            for item in workspace.capability_observations
        ),
    )
    if candidates:
        result = advance(
            result,
            guard_observations=(
                *result.guard_observations,
                PlannerGuardObservation(
                    observation_id=server_id(request_id, "batch"),
                    attempted_action="dining_slot_search",
                    code="planner_dining_batch_observed",
                    message=json.dumps({"slot_key": slot_key, "candidate_ids": list(candidates)}),
                    based_on_workspace_revision=result.workspace_revision,
                ),
            ),
        )
    return DiningSlotSearchResult(result, tuple(candidates.values()), failure_code, terminal)
