"""Pure P0-10 admission rules layered on top of the command Schema."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID

from backend.contracts.commands import CancelGenerationCommand, ClientCommandValue
from backend.contracts.enums import CommandType, TripPhase


class CommandAdmissionOutcome(StrEnum):
    ACCEPTED = "accepted"
    DUPLICATE = "duplicate"
    STATE_CONFLICT = "state_conflict"
    PHASE_CONFLICT = "phase_conflict"
    GENERATION_CONFLICT = "generation_conflict"


@dataclass(frozen=True)
class CommandAdmissionDecision:
    outcome: CommandAdmissionOutcome
    invalidate_active_generation: bool
    generation_id_to_invalidate: UUID | None = None


_PHASE_SCOPED_COMMANDS: dict[CommandType, frozenset[TripPhase]] = {
    CommandType.COLD_START_SUBMIT: frozenset({TripPhase.COLD_START}),
    CommandType.CITY_SELECT: frozenset({TripPhase.CITY_SELECTION}),
    CommandType.CITY_BRIEF_ACKNOWLEDGE: frozenset({TripPhase.CITY_BRIEF}),
    CommandType.PERSONAL_DEFAULTS_CONFIRM: frozenset({TripPhase.CITY_BRIEF}),
    CommandType.PERSONAL_DEFAULTS_ADJUST: frozenset({TripPhase.CITY_BRIEF}),
    CommandType.PERSONAL_DEFAULTS_NOT_USE: frozenset({TripPhase.CITY_BRIEF}),
    CommandType.TRIP_SETUP_SUBMIT: frozenset({TripPhase.TRIP_SETUP}),
    CommandType.CITY_THEMES_SUBMIT: frozenset({TripPhase.INTEREST_SELECTION}),
    CommandType.ATTRACTION_FEEDBACK_SUBMIT: frozenset({TripPhase.ATTRACTION_SELECTION}),
    CommandType.DINING_PREFERENCES_SUBMIT: frozenset({TripPhase.DINING_SELECTION}),
    CommandType.RESTAURANT_FEEDBACK_SUBMIT: frozenset({TripPhase.DINING_SELECTION}),
    CommandType.HOTEL_FAVORITES_SUBMIT: frozenset({TripPhase.LODGING_SELECTION}),
    CommandType.TASK_BOOK_CONFIRM: frozenset({TripPhase.TASK_REFLECTION}),
    CommandType.ACCEPT_PLAN: frozenset({TripPhase.DRAFT_READY}),
}


def command_allowed_in_phase(command_type: CommandType, phase: TripPhase) -> bool:
    """Return whether a command is legal in a phase without choosing the next phase."""

    allowed = _PHASE_SCOPED_COMMANDS.get(command_type)
    return allowed is None or phase in allowed


def evaluate_command_admission(
    command: ClientCommandValue,
    *,
    current_state_version: int,
    seen_idempotency_keys: Collection[str],
    active_generation_id: UUID | None,
    current_phase: TripPhase | None = None,
) -> CommandAdmissionDecision:
    """Classify a validated command without mutating state or persistence."""

    if command.idempotency_key in seen_idempotency_keys:
        return CommandAdmissionDecision(
            outcome=CommandAdmissionOutcome.DUPLICATE,
            invalidate_active_generation=False,
        )
    if command.expected_state_version != current_state_version:
        return CommandAdmissionDecision(
            outcome=CommandAdmissionOutcome.STATE_CONFLICT,
            invalidate_active_generation=False,
        )
    if current_phase is not None and not command_allowed_in_phase(command.type, current_phase):
        return CommandAdmissionDecision(
            outcome=CommandAdmissionOutcome.PHASE_CONFLICT,
            invalidate_active_generation=False,
        )
    if isinstance(command, CancelGenerationCommand):
        if active_generation_id is None or command.payload.generation_id != active_generation_id:
            return CommandAdmissionDecision(
                outcome=CommandAdmissionOutcome.GENERATION_CONFLICT,
                invalidate_active_generation=False,
            )
        return CommandAdmissionDecision(
            outcome=CommandAdmissionOutcome.ACCEPTED,
            invalidate_active_generation=True,
            generation_id_to_invalidate=active_generation_id,
        )
    invalidating_types = {
        CommandType.USER_MESSAGE,
        CommandType.ATTACHMENT_ANSWER,
        CommandType.RESET_TRIP,
    }
    should_invalidate = active_generation_id is not None and command.type in invalidating_types
    return CommandAdmissionDecision(
        outcome=CommandAdmissionOutcome.ACCEPTED,
        invalidate_active_generation=should_invalidate,
        generation_id_to_invalidate=(active_generation_id if should_invalidate else None),
    )
