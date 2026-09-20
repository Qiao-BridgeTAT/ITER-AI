"""Planner admission, authoritative identities, and evidence-derived permissions."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from backend.contracts.v4.enums import (
    AskUserReasonCode,
    CandidateEntityKind,
    CommitmentLevel,
    PlannerStatus,
    TaskBookStatus,
)
from backend.contracts.v4.planner_observations import (
    PlannerInteractionOption,
    PlannerReadinessIssue,
    PlannerReadinessObservation,
)
from backend.contracts.v4.planner_refs import CandidateRef, FixedCommitmentRef, PlannerScope
from backend.contracts.v4.planner_strategy import (
    CandidateAdvisoryFeatures,
    CandidatePoolEntry,
    CandidatePoolSummary,
)
from backend.contracts.v4.planner_workspace import PlannerWorkspaceState
from backend.contracts.v4.state import V4TripStateEnvelope
from backend.contracts.v4.task_book import TaskBookEntityIntent, TaskBookV4
from backend.persistence.outbox_repository import canonical_json_hash
from backend.providers.contracts import HoursDayStatus


class PlannerGuardError(ValueError):
    """Safe schema/permission code, never Provider payload or private reasoning."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def server_id(*parts: object) -> str:
    return str(uuid5(NAMESPACE_URL, "iter:v4-planner:" + ":".join(map(str, parts))))


def confirmed_task_book(state: V4TripStateEnvelope) -> TaskBookV4:
    """Revalidate the authoritative aggregate before any model or Provider call."""

    # Existing operations are immutable history, not newly submitted date edits.
    # Validate their shape/identity without requiring a new parser business date.
    state = V4TripStateEnvelope.model_validate(
        state.model_dump(mode="json"), context={"restore_historical_semantic_state": True}
    )
    reference = state.semantic_state.confirmed_task_book_ref
    candidate = state.discovery_runtime_state.task_book_candidate
    if reference is None or candidate is None:
        raise PlannerGuardError("planner_task_book_not_confirmed")
    book = candidate.value
    if (
        candidate.status is not TaskBookStatus.CONFIRMED
        or book.status is not TaskBookStatus.CONFIRMED
    ):
        raise PlannerGuardError("planner_task_book_not_confirmed")
    if (
        book.task_book_id != reference.task_book_id
        or book.version != reference.task_book_version
        or book.based_on_state_version != reference.based_on_state_version
        or book.based_on_state_version > state.semantic_state.state_version
    ):
        raise PlannerGuardError("planner_stale_task_book")
    if state.semantic_state.unresolved_conflicts:
        raise PlannerGuardError("planner_task_book_has_unresolved_conflicts")
    return book


def service_dates(book: TaskBookV4) -> tuple[date, ...]:
    dates = book.destination_and_dates
    return tuple(
        date.fromordinal(dates.start_date.toordinal() + day) for day in range(dates.duration_days)
    )


def entity_intents(book: TaskBookV4) -> dict[str, tuple[TaskBookEntityIntent, CandidateEntityKind]]:
    attractions = book.attraction_direction
    dining = book.dining_direction
    return {
        **{
            item.canonical_entity_id: (item, CandidateEntityKind.ATTRACTION)
            for item in (
                *attractions.must_visit,
                *attractions.wanted,
                *attractions.if_convenient,
                *attractions.exclusions,
            )
        },
        **{
            item.canonical_entity_id: (item, CandidateEntityKind.RESTAURANT)
            for item in (
                *dining.destination_restaurants,
                *dining.if_convenient_restaurants,
                *dining.excluded_restaurants,
            )
        },
    }


def task_book_references(book: TaskBookV4) -> dict[str, str]:
    """Stable read-only paths exposed to the model; no ad-hoc hard constraint keys."""

    references = {
        "destination": book.destination_and_dates.destination_name,
        "party": "、".join(book.travelers_and_trip_goal.travelers),
        "lodging": book.lodging_direction.model_dump_json(),
    }
    groups = {
        "goal": book.travelers_and_trip_goal.trip_goals,
        "pace": book.pace_and_transport.pace_preferences,
        "transport": book.pace_and_transport.transport_preferences,
        "attraction_preference": book.attraction_direction.preferences,
        "dining_preference": book.dining_direction.preferences,
        "dietary": book.dining_direction.hard_requirements,
        "area": book.lodging_direction.area_preferences,
        "facility": book.lodging_direction.facility_requirements,
        "hard": book.hard_constraints,
        "assumption": book.tradeoffs_and_assumptions,
    }
    for kind, values in groups.items():
        references.update({f"{kind}:{index}": item.value for index, item in enumerate(values)})
    if book.lodging_direction.nightly_budget is not None:
        references["lodging_budget"] = book.lodging_direction.nightly_budget.model_dump_json()
    return references


def fixed_commitments(book: TaskBookV4) -> tuple[FixedCommitmentRef, ...]:
    bookings = {item.booking_id: item for item in book.existing_bookings}
    hotel = book.lodging_direction.existing_booking
    if hotel is not None:
        bookings[hotel.booking_id] = hotel
    result = []
    for booking in bookings.values():
        kind = (
            "named_hotel"
            if hotel is not None and booking.booking_id == hotel.booking_id
            else booking.booking_kind
        )
        if kind not in {"reservation", "arrival", "departure", "named_hotel", "existing_booking"}:
            kind = "existing_booking"
        result.append(
            FixedCommitmentRef.model_validate(
                {
                    "task_book_id": book.task_book_id,
                    "task_book_version": book.version,
                    "commitment_id": booking.booking_id,
                    "commitment_kind": kind,
                }
            )
        )
    return tuple(result)


def initial_workspace(
    state: V4TripStateEnvelope, generation_id: UUID, now: datetime
) -> PlannerWorkspaceState:
    book = confirmed_task_book(state)
    scope = PlannerScope(
        trip_id=state.semantic_state.trip_id,
        generation_id=str(generation_id),
        task_book_id=book.task_book_id,
        task_book_version=book.version,
        task_book_state_version=book.based_on_state_version,
        workspace_revision=0,
    )
    return PlannerWorkspaceState(
        trip_id=scope.trip_id,
        generation_id=scope.generation_id,
        based_on_task_book_id=scope.task_book_id,
        based_on_task_book_version=scope.task_book_version,
        workspace_revision=0,
        candidate_pool=CandidatePoolSummary(
            candidate_pool_id=str(uuid4()),
            revision=1,
            scope=scope,
            pool_fingerprint=canonical_json_hash({"task_book": book.model_dump(mode="json")}),
            generated_at=now,
            source_reference_ids=(f"task_book:{book.task_book_id}:{book.version}",),
            candidates=(),
            fixed_commitments=fixed_commitments(book),
            missing_required_candidate_refs=tuple(
                key
                for key, (item, _) in entity_intents(book).items()
                if item.disposition.value != "avoid"
            ),
        ),
    )


def advance(workspace: PlannerWorkspaceState, **changes: Any) -> PlannerWorkspaceState:
    return PlannerWorkspaceState.model_validate(
        {
            **workspace.model_dump(mode="json"),
            **changes,
            "workspace_revision": workspace.workspace_revision + 1,
        }
    )


def refresh_pool(
    workspace: PlannerWorkspaceState, book: TaskBookV4, now: datetime
) -> PlannerWorkspaceState:
    """Changed evidence creates new references; old strategies are not silently rewritten."""

    old = workspace.candidate_pool
    existing_refs = {
        entry.candidate_ref.canonical_entity_id: entry.candidate_ref for entry in old.candidates
    }
    revision = old.revision + 1
    scope = workspace.current_scope.model_copy(
        update={"workspace_revision": workspace.workspace_revision + 1}
    )
    intents = entity_intents(book)
    hours = {item.canonical_entity_id: item for item in workspace.hours_evidence}
    entries = []
    levels = {
        "must": CommitmentLevel.STRONG,
        "destination": CommitmentLevel.STRONG,
        "want": CommitmentLevel.SOFT,
        "if_convenient": CommitmentLevel.FILLER,
        "avoid": CommitmentLevel.FORBIDDEN,
    }
    for place in workspace.place_evidence:
        if (
            place.entity_kind is CandidateEntityKind.RESTAURANT
            and workspace.dining_state is not None
            and place.canonical_entity_id not in workspace.dining_state.admitted_canonical_ids
        ):
            continue
        existing_ref = existing_refs.get(place.canonical_entity_id)
        intent = intents.get(place.canonical_entity_id)
        level = levels[intent[0].disposition.value] if intent else CommitmentLevel.NEUTRAL
        availability = hours.get(place.canonical_entity_id)
        known_days = (
            availability.days if availability is not None and availability.expires_at > now else ()
        )
        facts = (place.fact_reference_id,) + (
            (availability.fact_reference_id,) if known_days and availability is not None else ()
        )
        open_dates = tuple(
            day.service_date for day in known_days if day.status is HoursDayStatus.OPEN
        )
        closed_dates = tuple(
            day.service_date for day in known_days if day.status is HoursDayStatus.CLOSED
        )
        missing = {day.service_date for day in known_days} != set(service_dates(book)) or any(
            day.status in {HoursDayStatus.UNKNOWN, HoursDayStatus.CONFLICT} for day in known_days
        )
        entries.append(
            CandidatePoolEntry(
                candidate_ref=CandidateRef(
                    candidate_pool_id=old.candidate_pool_id,
                    candidate_pool_revision=revision,
                    # A new user turn is not a new real-world place. Pool
                    # revision changes express refreshed evidence; keep an
                    # existing identity stable across published-plan edits.
                    candidate_id=(
                        existing_ref.candidate_id
                        if existing_ref is not None
                        and existing_ref.entity_kind == place.entity_kind
                        else server_id(workspace.generation_id, place.canonical_entity_id)
                    ),
                    entity_kind=place.entity_kind,
                    canonical_entity_id=place.canonical_entity_id,
                ),
                display_name=place.display_name,
                entity_kind=place.entity_kind,
                commitment_level=level,
                source_intent_refs=tuple(intent[0].source_operation_refs) if intent else (),
                selection_permission="required"
                if level is CommitmentLevel.STRONG
                else "forbidden"
                if level is CommitmentLevel.FORBIDDEN
                else "filler_only"
                if level is CommitmentLevel.FILLER
                else "allowed",
                eligibility="excluded"
                if level is CommitmentLevel.FORBIDDEN
                else "needs_evidence"
                if missing or len(closed_dates) == len(service_dates(book))
                else "eligible",
                fact_reference_ids=facts,
                feasible_dates=open_dates if not missing else (),
                infeasible_dates=closed_dates,
                missing_fact_kinds=("opening_hours",) if missing else (),
                advisory_features=CandidateAdvisoryFeatures(
                    preference_fit="high" if intent else "unknown",
                    city_representativeness="unknown",
                    accessibility_fit="unknown",
                ),
            )
        )
    observed = {item.canonical_entity_id for item in workspace.place_evidence}
    missing_refs = tuple(
        key
        for key, (item, _) in intents.items()
        if item.disposition.value != "avoid" and key not in observed
    )
    pool = CandidatePoolSummary(
        candidate_pool_id=old.candidate_pool_id,
        revision=revision,
        scope=scope,
        pool_fingerprint=canonical_json_hash([item.model_dump(mode="json") for item in entries]),
        generated_at=now,
        source_reference_ids=tuple(
            dict.fromkeys(reference for entry in entries for reference in entry.fact_reference_ids)
        )
        or old.source_reference_ids,
        candidates=tuple(entries),
        fixed_commitments=old.fixed_commitments,
        missing_required_candidate_refs=missing_refs,
    )
    return advance(
        workspace,
        candidate_pool=pool,
        planning_strategy=None,
        spatial_observation=None,
        readiness_observation=None,
        working_itinerary=None,
        status=PlannerStatus.PLANNING,
        unresolved_decisions=(),
    )


def readiness(
    workspace: PlannerWorkspaceState, book: TaskBookV4, now: datetime
) -> PlannerReadinessObservation:
    issues: list[PlannerReadinessIssue] = []
    dates = service_dates(book)
    for reference in workspace.candidate_pool.missing_required_candidate_refs:
        issues.append(
            PlannerReadinessIssue(
                issue_id=server_id(
                    workspace.generation_id,
                    workspace.candidate_pool.revision,
                    "identity",
                    reference,
                ),
                code="required_entity_missing",
                affected_dates=dates,
                fact_reference_ids=(f"task_book:{book.task_book_id}:{book.version}",),
                reason_summary="尚未取得与任务书选择身份一致的实时实体，需要继续核验，不能替换为同名对象。",
                user_authority_required=False,
            )
        )
    for entry in workspace.candidate_pool.candidates:
        if entry.commitment_level is not CommitmentLevel.STRONG:
            continue
        if set(entry.infeasible_dates) == set(dates):
            issue_id = server_id(
                workspace.generation_id,
                workspace.candidate_pool.revision,
                "closed",
                entry.candidate_ref.candidate_id,
            )
            issues.append(
                PlannerReadinessIssue(
                    issue_id=issue_id,
                    code="strong_opening_conflict",
                    candidate_refs=(entry.candidate_ref,),
                    affected_dates=dates,
                    fact_reference_ids=entry.fact_reference_ids,
                    reason_summary=f"{entry.display_name}在全部旅行日期均有闭馆证据，不能静默删除这项强意愿。",
                    user_authority_required=True,
                    ask_user_reason=AskUserReasonCode.INCOMPATIBLE_STRONG_COMMITMENTS,
                    option_contracts=(
                        PlannerInteractionOption(
                            option_id=server_id(issue_id, "keep"),
                            semantic_action="keep_task_book",
                            affected_refs=(entry.candidate_ref.candidate_id,),
                            verified_impact_summary="保留原任务书并继续调整，先提供可用行程，明确说明未满足项。",
                        ),
                        PlannerInteractionOption(
                            option_id=server_id(issue_id, "revise"),
                            semantic_action="revise_task_book",
                            affected_refs=(entry.candidate_ref.candidate_id,),
                            verified_impact_summary="返回任务书调整旅行日期或这项意愿；修改后重新确认再规划。",
                        ),
                    ),
                )
            )
    hotel = book.lodging_direction.existing_booking
    if hotel is not None and (hotel.start_date is None or hotel.end_date is None):
        fixed = next(
            item
            for item in workspace.candidate_pool.fixed_commitments
            if item.commitment_id == hotel.booking_id
        )
        issue_id = server_id(workspace.generation_id, "booking_dates", hotel.booking_id)
        issues.append(
            PlannerReadinessIssue(
                issue_id=issue_id,
                code="missing_booking_detail",
                fixed_commitment_refs=(fixed,),
                affected_dates=dates,
                fact_reference_ids=tuple(hotel.source_operation_refs),
                reason_summary="已有预订缺少入住或退房日期；这是只有用户能确认的预订信息。",
                user_authority_required=True,
                ask_user_reason=AskUserReasonCode.MISSING_USER_OWNED_BOOKING_DETAIL,
                option_contracts=(
                    PlannerInteractionOption(
                        option_id=server_id(issue_id, "supply"),
                        semantic_action="supply_booking_detail",
                        affected_refs=(hotel.booking_id,),
                        verified_impact_summary="补充实际预订日期并重新确认任务书，不由系统猜测预订信息。",
                    ),
                ),
            )
        )
    for booking in book.existing_bookings:
        if (
            hotel is not None
            and booking.booking_id == hotel.booking_id
            or booking.start_date is not None
        ):
            continue
        fixed = next(
            item
            for item in workspace.candidate_pool.fixed_commitments
            if item.commitment_id == booking.booking_id
        )
        issue_id = server_id(workspace.generation_id, "booking_dates", booking.booking_id)
        issues.append(
            PlannerReadinessIssue(
                issue_id=issue_id,
                code="missing_booking_detail",
                fixed_commitment_refs=(fixed,),
                affected_dates=dates,
                fact_reference_ids=tuple(booking.source_operation_refs),
                reason_summary="已有固定事项缺少发生日期；这是只有用户能确认的信息。",
                user_authority_required=True,
                ask_user_reason=AskUserReasonCode.MISSING_USER_OWNED_BOOKING_DETAIL,
                option_contracts=(
                    PlannerInteractionOption(
                        option_id=server_id(issue_id, "supply"),
                        semantic_action="supply_booking_detail",
                        affected_refs=(booking.booking_id,),
                        verified_impact_summary="补充固定事项的实际日期并重新确认任务书，系统不会猜测日期。",
                    ),
                ),
            )
        )
    bookings = {booking.booking_id: booking for booking in book.existing_bookings}
    if hotel is not None:
        bookings[hotel.booking_id] = hotel
    for fixed in workspace.candidate_pool.fixed_commitments:
        booking = bookings[fixed.commitment_id]
        hotel_conflict = (
            hotel is not None
            and booking.booking_id == hotel.booking_id
            and booking.start_date is not None
            and booking.end_date is not None
            and (booking.start_date > dates[0] or booking.end_date < dates[-1])
        )
        event_conflict = fixed.commitment_kind != "named_hotel" and (
            booking.start_date is not None and booking.start_date not in dates
        )
        if not (hotel_conflict or event_conflict):
            continue
        issue_id = server_id(workspace.generation_id, "booking_conflict", booking.booking_id)
        issues.append(
            PlannerReadinessIssue(
                issue_id=issue_id,
                code="fixed_booking_conflict",
                fixed_commitment_refs=(fixed,),
                affected_dates=dates,
                fact_reference_ids=tuple(booking.source_operation_refs),
                reason_summary="已确认的预订日期与旅行日期不兼容；系统不能自行改动或放弃固定预订。",
                user_authority_required=True,
                ask_user_reason=AskUserReasonCode.FIXED_BOOKING_CONFLICT,
                option_contracts=(
                    PlannerInteractionOption(
                        option_id=server_id(issue_id, "keep"),
                        semantic_action="keep_task_book",
                        affected_refs=(booking.booking_id,),
                        verified_impact_summary="保留当前预订和旅行要求，暂停规划，不擅自修改预订。",
                    ),
                    PlannerInteractionOption(
                        option_id=server_id(issue_id, "revise"),
                        semantic_action="revise_task_book",
                        affected_refs=(booking.booking_id,),
                        verified_impact_summary="提供正确日期或调整旅行需求，重新确认任务书后继续。",
                    ),
                ),
            )
        )
    return PlannerReadinessObservation(
        observation_id=str(uuid4()),
        scope=workspace.current_scope,
        candidate_pool_revision=workspace.candidate_pool.revision,
        checked_at=now.astimezone(UTC),
        issues=tuple(issues),
    )
