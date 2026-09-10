"""Domain-specific atomic patch contracts for V4 working itineraries."""

from __future__ import annotations

from collections import Counter
from datetime import date
from typing import Annotated, Literal, TypeAlias

from pydantic import Field, model_validator

from backend.contracts.v4.base import (
    Digest,
    DisplayText,
    Identifier,
    V4ContractModel,
    require_unique,
)
from backend.contracts.v4.planner_draft import (
    CrossClusterSegment,
    DraftItem,
    ExpectedWindow,
    LodgingBaseline,
    UnassignedIntent,
    WorkingItineraryDraft,
    canonical_planning_projection,
    planning_projection_digest,
    validate_draft_against_pool,
)
from backend.contracts.v4.planner_refs import (
    CandidateRef,
    PlannerObjectRef,
    PlannerScope,
    candidate_ref_key,
    planner_object_ref_key,
    require_same_scope_ownership,
)
from backend.contracts.v4.planner_strategy import CandidatePoolSummary


class ValidationIssuePatchAuthority(V4ContractModel):
    kind: Literal["validation_issue"] = "validation_issue"
    reference_id: Identifier
    issue_ids: tuple[Identifier, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def issues_are_unique(self) -> ValidationIssuePatchAuthority:
        require_unique(self.issue_ids, "Patch validation issue IDs")
        return self


class PlanChangeRequestPatchAuthority(V4ContractModel):
    kind: Literal["plan_change_request"] = "plan_change_request"
    reference_id: Identifier
    user_message_or_answer_ref: Identifier


class PlannerInteractionAnswerPatchAuthority(V4ContractModel):
    kind: Literal["planner_interaction_answer"] = "planner_interaction_answer"
    reference_id: Identifier
    user_message_or_answer_ref: Identifier


PatchAuthority: TypeAlias = Annotated[
    ValidationIssuePatchAuthority
    | PlanChangeRequestPatchAuthority
    | PlannerInteractionAnswerPatchAuthority,
    Field(discriminator="kind"),
]


class _AnchoredOperation(V4ContractModel):
    before_item_id: Identifier | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    after_item_id: Identifier | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    at_end: Literal[True] | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )

    @model_validator(mode="after")
    def placement_is_present_exactly_once(self) -> _AnchoredOperation:
        present = {
            name
            for name in ("before_item_id", "after_item_id", "at_end")
            if name in self.model_fields_set
        }
        if len(present) != 1:
            raise ValueError("exactly one before_item_id, after_item_id or at_end is required")
        selected = next(iter(present))
        if getattr(self, selected) is None:
            raise ValueError("the selected placement field cannot be null")
        return self


class InsertItemOperation(_AnchoredOperation):
    operation: Literal["insert_item"] = "insert_item"
    service_date: date
    item: DraftItem
    authority_item_refs: tuple[Identifier, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def authority_refs_are_unique(self) -> InsertItemOperation:
        require_unique(self.authority_item_refs, "insert authority references")
        return self


class RemoveItemOperation(V4ContractModel):
    operation: Literal["remove_item"] = "remove_item"
    draft_item_id: Identifier
    expected_object_ref: PlannerObjectRef
    authority_item_refs: tuple[Identifier, ...] = Field(min_length=1)
    removal_reason_code: Literal[
        "capacity_conflict",
        "route_conflict",
        "opening_conflict",
        "budget_conflict",
        "duplicate_experience",
        "user_requested",
        "interaction_choice",
    ]
    unscheduled_intent_record: UnassignedIntent | None = None

    @model_validator(mode="after")
    def authority_refs_are_unique(self) -> RemoveItemOperation:
        require_unique(self.authority_item_refs, "remove authority references")
        return self


class ReplaceItemOperation(V4ContractModel):
    operation: Literal["replace_item"] = "replace_item"
    draft_item_id: Identifier
    expected_old_ref: PlannerObjectRef
    new_candidate_ref: CandidateRef
    authority_item_refs: tuple[Identifier, ...] = Field(min_length=1)
    unscheduled_intent_record: UnassignedIntent | None = None

    @model_validator(mode="after")
    def authority_refs_are_unique(self) -> ReplaceItemOperation:
        require_unique(self.authority_item_refs, "replace authority references")
        return self


class MoveItemOperation(_AnchoredOperation):
    operation: Literal["move_item"] = "move_item"
    draft_item_id: Identifier
    from_date: date
    to_date: date
    authority_item_refs: tuple[Identifier, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def authority_refs_are_unique(self) -> MoveItemOperation:
        require_unique(self.authority_item_refs, "move authority references")
        return self


class ReorderItemOperation(_AnchoredOperation):
    operation: Literal["reorder_item"] = "reorder_item"
    service_date: date
    draft_item_id: Identifier
    authority_item_refs: tuple[Identifier, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def authority_refs_are_unique(self) -> ReorderItemOperation:
        require_unique(self.authority_item_refs, "reorder authority references")
        return self


class SetPreferredWindowOperation(V4ContractModel):
    operation: Literal["set_preferred_window"] = "set_preferred_window"
    draft_item_id: Identifier
    expected_window: ExpectedWindow
    authority_item_refs: tuple[Identifier, ...] = Field(min_length=1)


class RouteEndpointRef(V4ContractModel):
    kind: Literal["draft_item", "cluster"]
    reference_id: Identifier


class SetTransportPreferenceOperation(V4ContractModel):
    operation: Literal["set_transport_preference"] = "set_transport_preference"
    service_date: date
    from_endpoint: RouteEndpointRef
    to_endpoint: RouteEndpointRef
    allowed_transport_modes: tuple[Literal["public_transit", "taxi", "walking", "driving"], ...] = (
        Field(min_length=1)
    )
    authority_item_refs: tuple[Identifier, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def endpoints_modes_and_authority_are_valid(self) -> SetTransportPreferenceOperation:
        if self.from_endpoint == self.to_endpoint:
            raise ValueError("transport preference endpoints must be distinct")
        require_unique(self.allowed_transport_modes, "allowed transport modes")
        require_unique(self.authority_item_refs, "transport authority references")
        return self


class SetPrimaryClusterOperation(V4ContractModel):
    operation: Literal["set_primary_cluster"] = "set_primary_cluster"
    service_date: date
    cluster_id: Identifier
    authority_item_refs: tuple[Identifier, ...] = Field(min_length=1)


class SetCrossClusterSegmentOperation(V4ContractModel):
    operation: Literal["set_cross_cluster_segment"] = "set_cross_cluster_segment"
    service_date: date
    segment: CrossClusterSegment
    authority_item_refs: tuple[Identifier, ...] = Field(min_length=1)


class SetHotelBaselineOperation(V4ContractModel):
    operation: Literal["set_hotel_baseline"] = "set_hotel_baseline"
    expected_old_baseline: LodgingBaseline
    new_baseline: LodgingBaseline
    authority_item_refs: tuple[Identifier, ...] = Field(min_length=1)


class SetDiscardRankOperation(V4ContractModel):
    operation: Literal["set_discard_rank"] = "set_discard_rank"
    draft_item_id: Identifier
    discard_rank: int = Field(ge=0, strict=True)
    authority_item_refs: tuple[Identifier, ...] = Field(min_length=1)


ItineraryPatchOperation: TypeAlias = Annotated[
    InsertItemOperation
    | RemoveItemOperation
    | ReplaceItemOperation
    | MoveItemOperation
    | ReorderItemOperation
    | SetPreferredWindowOperation
    | SetTransportPreferenceOperation
    | SetPrimaryClusterOperation
    | SetCrossClusterSegmentOperation
    | SetHotelBaselineOperation
    | SetDiscardRankOperation,
    Field(discriminator="operation"),
]


class ItineraryPatch(V4ContractModel):
    """Authorized, domain-specific patch; deliberately not RFC 6902."""

    patch_id: Identifier
    scope: PlannerScope
    base_draft_id: Identifier
    base_draft_revision: int = Field(ge=1, strict=True)
    base_content_digest: Digest
    authority: PatchAuthority
    operations: tuple[ItineraryPatchOperation, ...] = Field(min_length=1, max_length=50)
    declared_affected_dates: tuple[date, ...] = Field(min_length=1, max_length=5)
    reason_summary: DisplayText

    @model_validator(mode="after")
    def operation_and_date_sets_are_well_formed(self) -> ItineraryPatch:
        require_unique(self.declared_affected_dates, "declared affected dates")
        if list(self.declared_affected_dates) != sorted(self.declared_affected_dates):
            raise ValueError("declared affected dates must be date ordered")
        return self


def _operation_authority_refs(operation: ItineraryPatchOperation) -> tuple[str, ...]:
    return operation.authority_item_refs


def _to_mutable(value: object) -> object:
    if isinstance(value, dict):
        return {key: _to_mutable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_mutable(child) for child in value]
    return value


def _find_day(days: list[dict[str, object]], service_date: date) -> dict[str, object]:
    for day in days:
        if day["service_date"] == service_date:
            return day
    raise ValueError("Patch references a date outside the working draft")


def _find_item(
    days: list[dict[str, object]], draft_item_id: str
) -> tuple[dict[str, object], int, dict[str, object]]:
    for day in days:
        items = day["ordered_items"]
        assert isinstance(items, list)
        for index, item in enumerate(items):
            assert isinstance(item, dict)
            if item["draft_item_id"] == draft_item_id:
                return day, index, item
    raise ValueError("Patch references an unknown draft_item_id")


def _placement_index(items: list[dict[str, object]], operation: _AnchoredOperation) -> int:
    if operation.at_end is True:
        return len(items)
    anchor_id = operation.before_item_id or operation.after_item_id
    for index, item in enumerate(items):
        if item["draft_item_id"] == anchor_id:
            return index if operation.before_item_id is not None else index + 1
    raise ValueError("Patch placement anchor does not exist in the target day")


def _renumber(items: list[dict[str, object]]) -> None:
    for index, item in enumerate(items):
        item["position"] = index


def _remove_item_from_segments(day: dict[str, object], item_id: str) -> None:
    segments = day["cross_cluster_segments"]
    assert isinstance(segments, list)
    retained: list[dict[str, object]] = []
    for raw_segment in segments:
        assert isinstance(raw_segment, dict)
        covered = [value for value in raw_segment["covered_item_ids"] if value != item_id]
        if covered:
            raw_segment["covered_item_ids"] = tuple(covered)
            retained.append(raw_segment)
    day["cross_cluster_segments"] = retained


def _append_unassigned(
    payload: dict[str, object], record: UnassignedIntent | None, *, required: bool
) -> None:
    if required and record is None:
        raise ValueError("removing or replacing a soft/want item requires an unassigned record")
    if record is None:
        return
    records = payload["unassigned_intents"]
    assert isinstance(records, list)
    records.append(record.model_dump(mode="python"))


def _require_expected_ref(item: dict[str, object], expected: PlannerObjectRef) -> None:
    current = DraftItem.model_validate(item).object_ref
    if planner_object_ref_key(current) != planner_object_ref_key(expected):
        raise ValueError("Patch expected object reference does not match the current draft")


def _operation_dates(
    current: WorkingItineraryDraft,
    operation: ItineraryPatchOperation,
) -> set[date]:
    if isinstance(
        operation,
        (
            InsertItemOperation,
            ReorderItemOperation,
            SetTransportPreferenceOperation,
            SetPrimaryClusterOperation,
            SetCrossClusterSegmentOperation,
        ),
    ):
        return {operation.service_date}
    if isinstance(operation, MoveItemOperation):
        return {operation.from_date, operation.to_date}
    if isinstance(operation, SetHotelBaselineOperation):
        return {day.service_date for day in current.days}
    item_id = operation.draft_item_id
    for day in current.days:
        if any(item.draft_item_id == item_id for item in day.ordered_items):
            return {day.service_date}
    raise ValueError("Patch references an unknown draft_item_id")


def atomic_apply_itinerary_patch(
    current: WorkingItineraryDraft,
    patch: ItineraryPatch,
    *,
    candidate_pool: CandidatePoolSummary | None = None,
    allowed_authority_refs: frozenset[str] | None = None,
) -> WorkingItineraryDraft:
    """Validate and atomically apply every operation, rejecting no-op patches."""

    require_same_scope_ownership(current.scope, patch.scope)
    if (
        patch.base_draft_id != current.draft_id
        or patch.base_draft_revision != current.draft_revision
        or patch.base_content_digest != current.content_digest
    ):
        raise ValueError("Patch base draft ID, revision and digest must all be current")

    computed_dates: set[date] = set()
    for operation in patch.operations:
        computed_dates.update(_operation_dates(current, operation))
        if (
            allowed_authority_refs is not None
            and not set(_operation_authority_refs(operation)) <= allowed_authority_refs
        ):
            raise ValueError("Patch operation uses an unrelated authority reference")
    if tuple(sorted(computed_dates)) != patch.declared_affected_dates:
        raise ValueError("declared_affected_dates must equal the server-computed minimum scope")

    payload = _to_mutable(current.model_dump(mode="python"))
    assert isinstance(payload, dict)
    days = payload["days"]
    assert isinstance(days, list)
    for operation in patch.operations:
        if isinstance(operation, InsertItemOperation):
            if any(
                item["draft_item_id"] == operation.item.draft_item_id
                for day in days
                for item in day["ordered_items"]
            ):
                raise ValueError("insert_item draft_item_id already exists")
            target_day = _find_day(days, operation.service_date)
            items = target_day["ordered_items"]
            assert isinstance(items, list)
            index = _placement_index(items, operation)
            inserted = operation.item.model_dump(mode="python")
            items.insert(index, inserted)
            _renumber(items)
        elif isinstance(operation, RemoveItemOperation):
            day, index, item = _find_item(days, operation.draft_item_id)
            _require_expected_ref(item, operation.expected_object_ref)
            commitment = item["commitment_level"]
            if commitment in {"immutable", "strong"}:
                raise ValueError("immutable and strong items cannot be removed by Planner Patch")
            if (
                operation.unscheduled_intent_record is not None
                and isinstance(operation.expected_object_ref, CandidateRef)
                and candidate_ref_key(operation.unscheduled_intent_record.candidate_ref)
                != candidate_ref_key(operation.expected_object_ref)
            ):
                raise ValueError("unassigned record must describe the removed candidate")
            _append_unassigned(
                payload,
                operation.unscheduled_intent_record,
                required=commitment == "soft",
            )
            items = day["ordered_items"]
            assert isinstance(items, list)
            items.pop(index)
            _renumber(items)
            _remove_item_from_segments(day, operation.draft_item_id)
            discardable = payload["discardable_objects"]
            assert isinstance(discardable, list)
            payload["discardable_objects"] = [
                value for value in discardable if value["draft_item_id"] != operation.draft_item_id
            ]
        elif isinstance(operation, ReplaceItemOperation):
            _, _, item = _find_item(days, operation.draft_item_id)
            _require_expected_ref(item, operation.expected_old_ref)
            commitment = item["commitment_level"]
            if commitment in {"immutable", "strong"}:
                raise ValueError("immutable and strong items cannot be replaced by Planner Patch")
            if (
                operation.unscheduled_intent_record is not None
                and isinstance(operation.expected_old_ref, CandidateRef)
                and candidate_ref_key(operation.unscheduled_intent_record.candidate_ref)
                != candidate_ref_key(operation.expected_old_ref)
            ):
                raise ValueError("unassigned record must describe the replaced candidate")
            _append_unassigned(
                payload,
                operation.unscheduled_intent_record,
                required=commitment == "soft",
            )
            item["object_ref"] = operation.new_candidate_ref.model_dump(mode="python")
            item["onsite_lunch"] = False
            discardable = payload["discardable_objects"]
            assert isinstance(discardable, list)
            payload["discardable_objects"] = [
                value for value in discardable if value["draft_item_id"] != operation.draft_item_id
            ]
            item["item_kind"] = (
                "dining"
                if operation.new_candidate_ref.entity_kind.value == "restaurant"
                else "visit"
            )
            if candidate_pool is not None:
                replacement_entry = candidate_pool.candidate_by_id().get(
                    operation.new_candidate_ref.candidate_id
                )
                if (
                    replacement_entry is None
                    or replacement_entry.candidate_ref != operation.new_candidate_ref
                    or not replacement_entry.cluster_ids
                ):
                    raise ValueError("replacement candidate is outside the current pool")
                item["cluster_id"] = replacement_entry.cluster_ids[0]
                item["commitment_level"] = replacement_entry.commitment_level.value
            if item["item_kind"] != "dining":
                item["meal_slot"] = None
        elif isinstance(operation, MoveItemOperation):
            source_day, index, item = _find_item(days, operation.draft_item_id)
            if item["commitment_level"] == "immutable":
                raise ValueError("immutable items cannot be moved by Planner Patch")
            if source_day["service_date"] != operation.from_date:
                raise ValueError("move_item from_date does not match the current item date")
            target_day = _find_day(days, operation.to_date)
            source_items = source_day["ordered_items"]
            assert isinstance(source_items, list)
            source_items.pop(index)
            _renumber(source_items)
            _remove_item_from_segments(source_day, operation.draft_item_id)
            target_items = target_day["ordered_items"]
            assert isinstance(target_items, list)
            target_index = _placement_index(target_items, operation)
            target_items.insert(target_index, item)
            _renumber(target_items)
        elif isinstance(operation, ReorderItemOperation):
            day, index, item = _find_item(days, operation.draft_item_id)
            if day["service_date"] != operation.service_date:
                raise ValueError("reorder_item service_date does not match the current item date")
            items = day["ordered_items"]
            assert isinstance(items, list)
            items.pop(index)
            target_index = _placement_index(items, operation)
            items.insert(target_index, item)
            _renumber(items)
        elif isinstance(operation, SetPreferredWindowOperation):
            _, _, item = _find_item(days, operation.draft_item_id)
            item["expected_window"] = operation.expected_window.model_dump(mode="python")
        elif isinstance(operation, SetTransportPreferenceOperation):
            day = _find_day(days, operation.service_date)
            day["transport_preferences"] = operation.allowed_transport_modes
        elif isinstance(operation, SetPrimaryClusterOperation):
            day = _find_day(days, operation.service_date)
            day["primary_cluster_id"] = operation.cluster_id
        elif isinstance(operation, SetCrossClusterSegmentOperation):
            day = _find_day(days, operation.service_date)
            segments = day["cross_cluster_segments"]
            assert isinstance(segments, list)
            incoming = operation.segment.model_dump(mode="python")
            covered = set(operation.segment.covered_item_ids)
            retained = [
                segment
                for segment in segments
                if not covered.intersection(segment["covered_item_ids"])
            ]
            retained.append(incoming)
            day["cross_cluster_segments"] = retained
        elif isinstance(operation, SetHotelBaselineOperation):
            if payload["lodging_baseline"] != operation.expected_old_baseline.model_dump(
                mode="python"
            ):
                raise ValueError("set_hotel_baseline expected baseline is stale")
            payload["lodging_baseline"] = operation.new_baseline.model_dump(mode="python")
            selected = operation.new_baseline.selected_offer_ref
            payload["hotel_observation_id"] = (
                selected.hotel_observation_id if selected is not None else None
            )
        else:
            _, _, _ = _find_item(days, operation.draft_item_id)
            discardable = payload["discardable_objects"]
            assert isinstance(discardable, list)
            matched = False
            for value in discardable:
                if value["draft_item_id"] == operation.draft_item_id:
                    value["discard_rank"] = operation.discard_rank
                    matched = True
            if not matched:
                raise ValueError("set_discard_rank is limited to an authorized discardable item")

    _refresh_changed_day_spatial_metadata(current, patch, days)
    payload["scope"] = patch.scope.model_dump(mode="python")
    payload["draft_revision"] = current.draft_revision + 1
    payload["content_digest"] = "0" * 64
    candidate = WorkingItineraryDraft.model_validate(payload)
    payload["content_digest"] = planning_projection_digest(candidate)
    candidate = WorkingItineraryDraft.model_validate(payload)

    if canonical_planning_projection(candidate) == canonical_planning_projection(current):
        raise ValueError("ItineraryPatch is a no-op under the canonical planning projection")
    if candidate_pool is not None:
        validate_draft_against_pool(candidate, candidate_pool)
    return candidate


def _refresh_changed_day_spatial_metadata(
    current: WorkingItineraryDraft,
    patch: ItineraryPatch,
    days: list[dict[str, object]],
) -> None:
    """Compile geometry metadata, never a new place, order, route or travel fact.

    Compact planning derives clusters from the chosen items. Apply that same
    boundary to old published drafts too; stale optional legacy segments cannot
    authorize the new route. The graph queries actual new adjacent endpoints.
    Explicit legacy spatial operations still receive their normal strict checks.
    """
    explicit_spatial_dates = {
        op.service_date
        for op in patch.operations
        if isinstance(op, (SetPrimaryClusterOperation, SetCrossClusterSegmentOperation))
    }
    old_days = {day.service_date: day for day in current.days}
    for day in days:
        previous = old_days[day["service_date"]]  # type: ignore[index]
        items = day["ordered_items"]
        assert isinstance(items, list)
        old_items = [item.model_dump(mode="python") for item in previous.ordered_items]
        old_path = [(item["object_ref"], item["cluster_id"]) for item in old_items]
        new_path = [(item["object_ref"], item["cluster_id"]) for item in items]
        if old_path == new_path:
            continue
        # Route-menu choices refer to the old adjacency. Do not silently apply
        # an old selection to a new leg with the same item position.
        day["route_mode_selections"] = []
        if day["service_date"] in explicit_spatial_dates:
            continue
        clusters = [item["cluster_id"] for item in items if item["cluster_id"] is not None]
        day["primary_cluster_id"] = Counter(clusters).most_common(1)[0][0] if clusters else None
        day["day_kind"] = "active" if clusters else "arrival_departure" if items else "rest"
        day["cross_cluster_segments"] = []


def is_effective_patch(
    current: WorkingItineraryDraft,
    patch: ItineraryPatch,
    *,
    candidate_pool: CandidatePoolSummary | None = None,
    allowed_authority_refs: frozenset[str] | None = None,
) -> bool:
    """Return whether a validated Patch produces an authorized semantic change."""

    try:
        atomic_apply_itinerary_patch(
            current,
            patch,
            candidate_pool=candidate_pool,
            allowed_authority_refs=allowed_authority_refs,
        )
    except ValueError:
        return False
    return True


V4_PLANNER_PATCH_CONTRACTS: tuple[type[V4ContractModel], ...] = (ItineraryPatch,)
