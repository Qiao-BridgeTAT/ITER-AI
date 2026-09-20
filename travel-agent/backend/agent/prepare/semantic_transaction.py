"""Runtime-only compilation: one basics entry and per-request entity evidence."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from backend.contracts.v4.enums import ToolObservationStatus
from backend.contracts.v4.prepare import PrepareDecision, ResolvePlaceRequest, ToolObservation
from backend.contracts.v4.semantic_operations import SemanticOperationProposal


class SemanticCompilationError(ValueError):
    def __init__(self, code: str, path: str, *, recovery: str = "repair_fields") -> None:
        self.code = code
        self.path = path
        self.recovery = recovery
        super().__init__(f"{code}: {path}")


def normalize_compound_basics(value: Any) -> Any:
    """Accept redundant copies, never choose between conflicting user facts.

    The schema excludes basics from additions. This pre-validation adapter also
    handles a model ignoring that schema without spending another model call.
    """
    if not isinstance(value, dict) or not isinstance(value.get("trip_basics"), dict):
        return value
    payload = deepcopy(value)
    basics = payload["trip_basics"]
    additions = payload.get("semantic_operations", [])
    if not isinstance(additions, list):
        return value
    kept = []
    fields = (
        "destination_name",
        "destination_canonical_id",
        "start_date",
        "end_date",
        "duration_days",
        "travelers",
        "trip_goals",
    )
    for index, operation in enumerate(additions):
        if not isinstance(operation, dict) or operation.get("operation_type") != "set_trip_basics":
            kept.append(operation)
            continue
        for field in fields:
            incoming = operation.get(field)
            if incoming is None:
                continue
            current = basics.get(field)
            if current is not None and current != incoming:
                raise SemanticCompilationError(
                    "compound_basics_conflict", f"semantic_operations[{index}].{field}"
                )
            # Canonical identity remains server-owned, not part of the compact choice.
            if field != "destination_canonical_id":
                basics[field] = incoming
    payload["semantic_operations"] = kept
    return payload


@dataclass(frozen=True)
class ResolvedSelection:
    key: str
    query: str
    entity_refs: tuple[str, ...]
    source_refs: tuple[str, ...]

    def context(self) -> dict[str, object]:
        return {
            "resolution_key": self.key,
            "query": self.query,
            "status": "resolved" if len(self.entity_refs) == 1 else "needs_clarification",
        }


def resolution_choices(
    decisions: list[PrepareDecision], observations: list[ToolObservation]
) -> list[ResolvedSelection]:
    requests = {
        request.request_id: request
        for decision in decisions
        for wrapped in decision.tool_requests
        if isinstance((request := wrapped.root), ResolvePlaceRequest)
    }
    choices: list[ResolvedSelection] = []
    for observation in observations:
        resolved_request = requests.get(observation.request_id)
        if observation.capability.value != "resolve_place" or resolved_request is None:
            continue
        successful = observation.status in {
            ToolObservationStatus.SUCCESS,
            ToolObservationStatus.PARTIAL,
        }
        choices.append(
            ResolvedSelection(
                key=f"e{len(choices) + 1}",
                query=resolved_request.query,
                entity_refs=tuple(observation.entity_refs) if successful else (),
                source_refs=tuple(observation.source_refs) if successful else (),
            )
        )
    return choices


def bind_selection(
    operation: dict[str, object],
    choices: list[ResolvedSelection],
    *,
    index: int,
    allow_single_fallback: bool = True,
) -> ResolvedSelection:
    key = operation.pop("resolution_key", None)
    path = f"semantic_operations[{index}].resolution_key"
    if key is not None:
        matches = [item for item in choices if item.key == key]
    else:
        name = operation.get("display_name")
        matches = [item for item in choices if item.query == name]
        if not matches and len(choices) == 1 and allow_single_fallback:
            matches = choices  # Compatible with the former single-place compact contract.
    if len(matches) != 1:
        raise SemanticCompilationError("entity_resolution_key_required", path)
    choice = matches[0]
    if len(choice.entity_refs) != 1:
        raise SemanticCompilationError("entity_resolution_unavailable", path, recovery="clarify")
    return choice


def partition_pending_entities(
    decision: PrepareDecision,
    proposals: list[SemanticOperationProposal],
) -> tuple[list[SemanticOperationProposal], list[dict[str, object]]]:
    """A named intent waiting for this decision's query is not a database write.

    Retain the semantic choice, discard its provisional identity even if the
    model reused an unrelated known ID. The requested lookup has not run yet.
    Other refs still reach the strict merge guard.
    """
    queries = {
        request.query
        for wrapped in decision.tool_requests
        if isinstance((request := wrapped.root), ResolvePlaceRequest)
        and request.purpose.value == "validate_operation"
    }
    ready: list[SemanticOperationProposal] = []
    pending: list[dict[str, object]] = []
    for wrapped in proposals:
        operation = wrapped.root
        if (
            operation.operation_type in {"select_concrete_entity", "exclude_concrete_entity"}
            and getattr(operation, "display_name", None) in queries
        ):
            data = operation.model_dump(mode="json")
            pending.append(
                {
                    key: data[key]
                    for key in (
                        "operation_type",
                        "domain",
                        "display_name",
                        "disposition",
                        "confidence",
                    )
                    if key in data
                }
            )
        else:
            ready.append(wrapped)
    return ready, pending
