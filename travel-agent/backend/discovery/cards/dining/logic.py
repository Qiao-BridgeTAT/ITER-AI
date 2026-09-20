"""Program-owned merge/count/cache selection, independent of network and model clients."""

import copy
import random
from collections import Counter
from typing import Any

MINIMUM_GROUPS = {1: 5, 2: 8, 3: 10, 4: 12, 5: 15}
FIXED_KEYWORDS = ("异国料理", "火锅", "烤肉", "自助餐", "家常菜", "快餐简餐", "烧烤")
TYPE_FIELDS = ("name", "provider_category", "provider_food_tags", "provider_alias")


def minimum_for_days(days: Any) -> Any:
    if type(days) is not int or days not in MINIMUM_GROUPS:
        raise ValueError("Supported dining discovery duration is 1 to 5 days")
    return MINIMUM_GROUPS[days]


def validate_brand_groups(output: Any, candidates: Any) -> Any:
    lookup = {c["candidate_key"]: c for c in candidates}
    errors, seen = [], []
    for group in output["groups"]:
        keys = group["member_keys"]
        seen.extend(keys)
        if len(keys) < 2 or len(set(keys)) != len(keys):
            errors.append("brand_group_requires_two_distinct_members")
        if not group["brand_name"].strip():
            errors.append("empty_brand_name")
        if group["basis"] == "explicit_provider" and not all(
            lookup.get(k, {}).get("explicit_brand_evidence") for k in keys
        ):
            errors.append("explicit_brand_evidence_missing")
    counts = Counter(seen)
    errors.extend("unknown_brand_member:" + k for k in counts if k not in lookup)
    errors.extend(
        "brand_member_in_multiple_groups:" + k for k, count in counts.items() if count > 1
    )
    return errors


def validate_types(output: Any, candidates: Any) -> Any:
    lookup = {c["candidate_key"]: c for c in candidates}
    counts = Counter(row["candidate_key"] for row in output["items"])
    errors = ["type_item_must_appear_once:" + k for k in lookup if counts[k] != 1]
    errors.extend("unknown_type_member:" + k for k in counts if k not in lookup)
    for row in output["items"]:
        key = row["candidate_key"]
        for field in row["evidence_fields"]:
            if field not in TYPE_FIELDS or not lookup.get(key, {}).get(field):
                errors.append(f"invalid_type_evidence:{key}:{field}")
        if (row["primary_type"] or row["broad_type"]) and not row["evidence_fields"]:
            errors.append("missing_type_evidence:" + key)
    return errors


def merge_roles(candidates: Any, brands: Any, types: Any) -> Any:
    errors = validate_brand_groups(brands, candidates) + validate_types(types, candidates)
    if errors:
        raise ValueError(";".join(errors))
    lookup = {c["candidate_key"]: c for c in candidates}
    order = {c["candidate_key"]: i for i, c in enumerate(candidates)}
    classified = {r["candidate_key"]: r for r in types["items"]}
    assigned = set()
    groups = []

    def build(keys: Any, brand_name: Any, basis: Any) -> Any:
        members = []
        for key in sorted(keys, key=lambda key: order[key]):
            classification = classified[key]
            members.append(
                {
                    **copy.deepcopy(lookup[key]),
                    "model_classification": {
                        "primary_type": classification["primary_type"],
                        "broad_type": classification["broad_type"],
                        "type_evidence_fields": classification["evidence_fields"],
                    },
                }
            )
        return {
            "anchor_candidate_key": members[0]["candidate_key"],
            "brand_name": brand_name,
            "brand_basis": basis,
            "members": members,
        }

    for group in brands["groups"]:
        keys = group["member_keys"]
        assigned.update(keys)
        groups.append(build(keys, group["brand_name"], group["basis"]))
    for key in lookup:
        if key not in assigned:
            groups.append(build([key], None, "insufficient"))
    groups.sort(key=lambda g: order[g["anchor_candidate_key"]])
    return groups


def compile_pool(
    groups: Any, base_ids: Any, reserve_ids: Any, minimum: Any, seed: Any, excluded_ids: Any = None
) -> Any:
    """Keep all base groups; random fill only from unused eligible cache brands."""
    excluded_ids = set(excluded_ids or ())
    base_ids, reserve_ids = set(base_ids), set(reserve_ids)
    base_groups, reserve_groups = [], []

    def narrowed(group: Any, members: Any) -> Any:
        return {
            **copy.deepcopy(group),
            "anchor_candidate_key": members[0]["candidate_key"],
            "members": copy.deepcopy(members),
        }

    for group in groups:
        eligible = [
            m
            for m in group["members"]
            if m["candidate_key"] not in excluded_ids and not m.get("known_conflicts")
        ]
        base = [m for m in eligible if m["candidate_key"] in base_ids]
        if base:
            base_groups.append(narrowed(group, base))
            continue
        reserve = [m for m in eligible if m["candidate_key"] in reserve_ids]
        if reserve:
            reserve_groups.append(narrowed(group, reserve))
    gap = max(0, minimum - len(base_groups))
    rng = random.Random(seed)
    rng.shuffle(reserve_groups)
    selected = []
    for group in reserve_groups[:gap]:
        # A brand contributes one cached real store, never several threshold slots.
        member = rng.choice(group["members"])
        selected.append(narrowed(group, [member]))
    pool = base_groups + selected
    metadata = {
        "minimum_groups": minimum,
        "base_groups": len(base_groups),
        "eligible_reserve_groups": len(reserve_groups),
        "random_seed": seed,
        "added_cached_groups": len(selected),
        "actual_groups": len(pool),
        "shortfall_count": max(0, minimum - len(pool)),
        "added_cached_member_keys": [g["members"][0]["candidate_key"] for g in selected],
    }
    return pool, metadata
