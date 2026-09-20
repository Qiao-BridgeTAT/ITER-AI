"""Admit private Prepare facts only through the confirmed card provenance."""

from datetime import datetime, timedelta
from uuid import UUID

from backend.agent.planner.workspace import advance, entity_intents, service_dates
from backend.contracts.v4.enums import CandidateEntityKind
from backend.contracts.v4.planner_workspace import PlannerWorkspaceState
from backend.contracts.v4.task_book import TaskBookV4
from backend.persistence.prepared_evidence_repository import PreparedEvidenceRepository


def fully_cached_fact_tools(
    workspace: PlannerWorkspaceState, book: TaskBookV4, now: datetime
) -> set[str]:
    """Hide queries that can only return facts already in the model's current state.

    A still-valid unknown is a completed query, not a promise of availability.
    New candidates, new dates or expiry make the tool available again. Web
    search remains available to look for another source of a missing fact.
    """
    candidates = [
        c for c in workspace.candidate_pool.candidates if c.selection_permission != "forbidden"
    ]
    entities = {c.candidate_ref.canonical_entity_id for c in candidates}
    dates = set(service_dates(book))
    hours = {
        h.canonical_entity_id
        for h in workspace.hours_evidence
        if h.observed_at <= now < h.expires_at and dates <= {d.service_date for d in h.days}
    }
    covered = {"lookup_hours"} if entities and entities <= hours else set()
    tickets = {
        (t.canonical_entity_id, t.service_date)
        for t in workspace.ticket_evidence
        if timedelta(0) <= now - t.observed_at < timedelta(minutes=15)
    }
    needed = {
        (c.candidate_ref.canonical_entity_id, day)
        for c in candidates
        if c.entity_kind is CandidateEntityKind.ATTRACTION
        for day in dates
    }
    if needed and needed <= tickets:
        covered.add("lookup_tickets")
    return covered


async def inherit_prepared_facts(
    workspace: PlannerWorkspaceState,
    book: TaskBookV4,
    repository: PreparedEvidenceRepository,
    owner_id: UUID,
    now: datetime,
) -> PlannerWorkspaceState:
    if workspace.react_state is None or workspace.initial_evidence_ready:
        return workspace
    allowed = {
        key
        for key, (intent, _) in entity_intents(book).items()
        if intent.disposition.value != "avoid"
    } | {o.canonical_entity_id for o in workspace.candidate_origins if o.inherit_as_neutral}
    origins = tuple(o for o in workspace.candidate_origins if o.canonical_entity_id in allowed)
    bundles = await repository.load(
        owner_id,
        UUID(workspace.trip_id),
        origins=origins,
        based_on_state_version=book.based_on_state_version,
        now=now,
    )
    if not bundles:
        return workspace
    dates = set(service_dates(book))
    places = {p.canonical_entity_id: p for p in workspace.place_evidence}
    hours = {h.canonical_entity_id: h for h in workspace.hours_evidence}
    tickets = {(t.canonical_entity_id, t.service_date): t for t in workspace.ticket_evidence}
    for bundle in bundles:
        place = bundle.place
        if place.city_id != book.destination_and_dates.destination_canonical_id:
            continue
        places[place.canonical_entity_id] = place
        h = bundle.hours
        if h and h.observed_at <= now < h.expires_at and dates <= {d.service_date for d in h.days}:
            hours[place.canonical_entity_id] = h.model_copy(
                update={"days": tuple(d for d in h.days if d.service_date in dates)}
            )
        for ticket in bundle.tickets:
            if ticket.service_date in dates and timedelta(
                0
            ) <= now - ticket.observed_at < timedelta(minutes=15):
                tickets[(ticket.canonical_entity_id, ticket.service_date)] = ticket
    return advance(
        workspace,
        place_evidence=tuple(places.values()),
        hours_evidence=tuple(hours.values()),
        ticket_evidence=tuple(tickets.values()),
    )
