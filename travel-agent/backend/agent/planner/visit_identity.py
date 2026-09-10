"""Group a venue and its known internal POIs without inventing geographic facts."""

import re
from collections.abc import Iterable

from backend.contracts.v4.planner_evidence import PlannerPlaceEvidence


def is_named_subsite(name: str, parent_name: str) -> bool:
    return len(parent_name) >= 3 and bool(re.match(re.escape(parent_name) + r"\s*[-—·/（(]", name))


def is_explicit_internal_subsite(name: str, provider_parent_id: str | None) -> bool:
    """A named park's internal stop is not an independent recovery option.

    Parent IDs alone are insufficient: whole gardens and museums can also have
    an AMap parent. Require the explicit venue-prefix/subsite naming as well.
    """
    return bool(
        provider_parent_id
        and re.match(
            r".+(?:风景名胜区|风景区|景区|公园|园林|园|历史文化街区|历史街区|步行街|古镇|旅游区)\s*[-—·/]\s*.+",
            name,
        )
    )


def visit_venue_groups(places: Iterable[PlannerPlaceEvidence]) -> dict[str, str]:
    sites = [item for item in places if item.entity_kind.value == "attraction"]
    by_provider = {item.provider_entity_id: item.canonical_entity_id for item in sites}
    parents = {}
    for site in sites:
        parent = by_provider.get(site.provider_parent_place_id or "")
        if parent is None:
            named = [
                item for item in sites if is_named_subsite(site.display_name, item.display_name)
            ]
            if named:
                parent = max(named, key=lambda item: len(item.display_name)).canonical_entity_id
        parents[site.canonical_entity_id] = parent or site.canonical_entity_id
    result = {}
    for key in parents:
        root = key
        seen = set()
        while root not in seen and parents.get(root, root) != root:
            seen.add(root)
            root = parents[root]
        result[key] = root if root not in seen else key
    return result


def overlapping_visits(places: Iterable[PlannerPlaceEvidence]) -> set[frozenset[str]]:
    """Duplicate identities or ancestor/descendant visits, not sibling venues.

    A district may contain several independently visitable museums. Loading
    that district later must not turn two previously selected museums into a
    duplicate. Direct provider ancestry works even before the parent is loaded.
    Grouping for presentation/coverage remains separate from overlap legality.
    """
    sites = [place for place in places if place.entity_kind.value == "attraction"]
    by_provider = {place.provider_entity_id: place for place in sites}

    def ancestors(place: PlannerPlaceEvidence) -> set[str]:
        result: set[str] = set()
        parent = place.provider_parent_place_id
        while parent and parent not in result and parent != place.provider_entity_id:
            result.add(parent)
            loaded = by_provider.get(parent)
            parent = loaded.provider_parent_place_id if loaded else None
        return result

    ancestry = {place.canonical_entity_id: ancestors(place) for place in sites}
    return {
        frozenset((first.canonical_entity_id, second.canonical_entity_id))
        for index, first in enumerate(sites)
        for second in sites[index + 1 :]
        if (
            first.provider_entity_id == second.provider_entity_id
            or first.provider_entity_id in ancestry[second.canonical_entity_id]
            or second.provider_entity_id in ancestry[first.canonical_entity_id]
            or is_named_subsite(first.display_name, second.display_name)
            or is_named_subsite(second.display_name, first.display_name)
        )
    }
