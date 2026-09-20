"""Shared dining projection, identity checks and persistent Planner admission."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from datetime import UTC, datetime
from difflib import SequenceMatcher

from backend.agent.planner.workspace import advance
from backend.contracts.v4.enums import CandidateEntityKind
from backend.contracts.v4.planner_dining import PlannerDiningState
from backend.contracts.v4.planner_evidence import PlannerPlaceEvidence
from backend.contracts.v4.planner_workspace import PlannerWorkspaceState
from backend.contracts.v4.task_book import TaskBookV4

DINING_SELECTION_REQUIREMENTS = (
    "午晚餐先遵守已确认的饮食及消费硬要求，再比较距离、评分和多样性。"
    "已知人均超过用户明确上限的候选不得为了凑数或评分而选。"
    "average_cost/reference_cost 的 minimum_fen 和 maximum_fen 单位是人民币分，100分=1元，"
    "属于人均参考值，不是菜单报价或保证实际花费。"
    "结合真实 cuisine/provider_typecode 及已有资料选择适合正餐的店；"
    "餐饮大类、店名或未知资料本身不能证明提供正餐、符合预算或饮食限制。"
    "未知价格和品类保留未知，不编造适配证明，优先选择已有事实支持的合适候选。"
)


def dining_candidate_facts(place: PlannerPlaceEvidence | None) -> dict[str, object]:
    """Project existing Provider business facts without classifying meal capability."""
    if place is None:
        return {}
    return {
        "cuisine": place.cuisine,
        "provider_typecode": (
            place.provider_typecode if place.provider_typecode != "unknown" else None
        ),
        "average_cost": place.average_cost.model_dump(mode="json") if place.average_cost else None,
    }


def build_dining_context(book: TaskBookV4, workspace: PlannerWorkspaceState) -> dict[str, object]:
    """Deliberately omit the trip/date/party envelope from every dining model call."""
    dining = book.dining_direction
    intents = {
        item.canonical_entity_id: item
        for item in (
            *dining.destination_restaurants,
            *dining.if_convenient_restaurants,
            *dining.excluded_restaurants,
        )
    }
    places = {
        place.canonical_entity_id: place
        for place in workspace.place_evidence
        if place.entity_kind is CandidateEntityKind.RESTAURANT
        and (
            workspace.dining_state is None
            or place.canonical_entity_id in workspace.dining_state.admitted_canonical_ids
        )
    }
    existing = [
        {
            "canonical_entity_id": key,
            "display_name": place.display_name if place else intents[key].display_name,
            "disposition": intents[key].disposition.value if key in intents else "neutral",
            "cuisine": place.cuisine if place else None,
        }
        for key in dict.fromkeys((*places, *intents))
        for place in [places.get(key)]
    ]
    return {
        "city": book.destination_and_dates.destination_name,
        "preferences": [item.value for item in dining.preferences],
        "hard_constraints": [item.value for item in dining.hard_requirements],
        "delegation_scopes": (
            {
                "domain": dining.delegated_scope.domain,
                "delegated_targets": dining.delegated_scope.delegated_targets,
                "boundary_refs": dining.delegated_scope.boundary_refs,
            }
            if dining.delegated_scope
            else None
        ),
        "existing_restaurants": existing,
    }


def mark_dining_admitted(
    workspace: PlannerWorkspaceState, canonical_ids: Iterable[str]
) -> PlannerWorkspaceState:
    state = workspace.dining_state
    if state is None:
        forbidden = {
            entry.candidate_ref.canonical_entity_id
            for entry in workspace.candidate_pool.candidates
            if entry.selection_permission == "forbidden"
        }
        inherited = tuple(
            place.canonical_entity_id
            for place in workspace.place_evidence
            if place.entity_kind is CandidateEntityKind.RESTAURANT
            and place.canonical_entity_id not in forbidden
        )
        state = PlannerDiningState(
            inherited_canonical_ids=inherited, admitted_canonical_ids=inherited
        )
    return advance(
        workspace,
        dining_state=state.model_copy(
            update={
                "admitted_canonical_ids": tuple(
                    dict.fromkeys((*state.admitted_canonical_ids, *canonical_ids))
                )
            }
        ),
    )


def deduplicate_inherited_dining(
    workspace: PlannerWorkspaceState, book: TaskBookV4
) -> PlannerWorkspaceState:
    """Choose inherited representatives without rebinding explicit user identities.

    Every normalized source remains in place_evidence/candidate_origins. Only
    admission changes, so an alias cannot donate its business facts or invent
    user intent on the chosen representative.
    """
    state = workspace.dining_state
    if state is None:
        return workspace
    explicit = {
        item.canonical_entity_id
        for item in (
            *book.dining_direction.destination_restaurants,
            *book.dining_direction.if_convenient_restaurants,
        )
    }
    old = [
        place
        for place in workspace.place_evidence
        if place.entity_kind is CandidateEntityKind.RESTAURANT
        and place.canonical_entity_id in state.inherited_canonical_ids
        and not dining_place_blocked(place, book)
    ]
    representatives: list[PlannerPlaceEvidence] = []
    for place in old:
        index = next(
            (
                index
                for index, previous in enumerate(representatives)
                if same_dining_entity(previous, place)
            ),
            None,
        )
        if index is None:
            representatives.append(place)
            continue
        previous = representatives[index]
        previous_explicit = previous.canonical_entity_id in explicit
        current_explicit = place.canonical_entity_id in explicit
        if previous_explicit and current_explicit:
            # Different explicit canonical IDs are not proven identical solely
            # by fuzzy evidence. Preserve both user records rather than remap one.
            representatives.append(place)
        elif current_explicit:
            representatives[index] = place
    kept = tuple(place.canonical_entity_id for place in representatives)
    inherited_set = set(state.inherited_canonical_ids)
    admitted = tuple(
        dict.fromkeys(
            (
                *(key for key in state.admitted_canonical_ids if key not in inherited_set),
                *kept,
            )
        )
    )
    if kept == state.inherited_canonical_ids and admitted == state.admitted_canonical_ids:
        return workspace
    return advance(
        workspace,
        dining_state=state.model_copy(
            update={
                "inherited_canonical_ids": kept,
                "admitted_canonical_ids": admitted,
            }
        ),
    )


def dining_call_timeout(workspace: PlannerWorkspaceState, maximum: float) -> float:
    """The initial phase reserves 35 seconds for its main Planner decision."""
    budget = workspace.schedule_repair_state
    if budget is None:
        return maximum
    now = datetime.now(UTC)
    remaining = maximum
    if budget.initial_deadline_at is not None:
        remaining = min(remaining, (budget.initial_deadline_at - now).total_seconds() - 35)
    if budget.call_cutoff_at is not None:
        remaining = min(remaining, (budget.call_cutoff_at - now).total_seconds())
    return max(0.0, remaining)


def straight_distance_km(left: PlannerPlaceEvidence, right: PlannerPlaceEvidence) -> float:
    lat1, lat2 = math.radians(left.coordinates.latitude), math.radians(right.coordinates.latitude)
    dlat = lat2 - lat1
    dlon = math.radians(right.coordinates.longitude - left.coordinates.longitude)
    return (
        6371.0088
        * 2
        * math.asin(
            min(
                1.0,
                math.sqrt(
                    math.sin(dlat / 2) ** 2
                    + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
                ),
            )
        )
    )


def same_dining_entity(left: PlannerPlaceEvidence, right: PlannerPlaceEvidence) -> bool:
    if left.city_id != right.city_id:
        return False
    if left.provider_entity_id == right.provider_entity_id:
        return True

    # Never collapse a brand's separate branches or copy one branch's rating.
    def normalize(value: str) -> str:
        return re.sub(r"[^\w]", "", value.casefold())

    return bool(
        left.address
        and right.address
        and normalize(left.address) == normalize(right.address)
        and SequenceMatcher(
            None, normalize(left.display_name), normalize(right.display_name)
        ).ratio()
        >= 0.9
        and straight_distance_km(left, right) <= 0.12
    )


def merge_exact_dining_evidence(
    previous: PlannerPlaceEvidence, incoming: PlannerPlaceEvidence
) -> PlannerPlaceEvidence:
    if previous.provider_entity_id != incoming.provider_entity_id:
        return previous
    if previous.observed_at > incoming.observed_at:
        previous, incoming = incoming, previous
    preserved_business = (incoming.rating is None and previous.rating is not None) or (
        incoming.average_cost is None and previous.average_cost is not None
    )
    return incoming.model_copy(
        update={
            "rating": incoming.rating if incoming.rating is not None else previous.rating,
            "average_cost": incoming.average_cost or previous.average_cost,
            "cuisine": incoming.cuisine or previous.cuisine,
            "business_fact_reference_id": (
                (previous.business_fact_reference_id or previous.fact_reference_id)
                if preserved_business
                else incoming.business_fact_reference_id
            ),
            "business_observed_at": (
                (previous.business_observed_at or previous.observed_at)
                if preserved_business
                else incoming.business_observed_at
            ),
        }
    )


def dining_place_blocked(place: PlannerPlaceEvidence, book: TaskBookV4) -> bool:
    name = re.sub(r"\s+", "", place.display_name.casefold())
    if re.search(
        r"暂停营业|暂不营业|已关闭|已停业|歇业|永久关闭|已搬迁|temporarilyclosed|permanentlyclosed",
        name,
    ):
        return True
    for excluded in book.dining_direction.excluded_restaurants:
        if place.canonical_entity_id == excluded.canonical_entity_id or name == re.sub(
            r"\s+", "", excluded.display_name.casefold()
        ):
            return True
    # Only an explicit ingredient/name contradiction is a hard filter. Unknown
    # menus must not be advertised as allergy-safe or treated as known unsafe.
    for requirement in book.dining_direction.hard_requirements:
        for token in re.findall(
            r"(?:不吃|不能吃|忌口|禁止食用)[：: ]?([\u4e00-\u9fff]{2,6})(?:[，。；、]|$)",
            requirement.value,
        ):
            if token in name:
                return True
    return False


def generic_fast_food(place: PlannerPlaceEvidence) -> bool:
    return bool(
        re.search(
            r"麦当劳|必胜客|肯德基|汉堡王|mcdonald|pizzahut|burgerking|\bkfc\b",
            re.sub(r"\s+", "", place.display_name.casefold()),
        )
    )
