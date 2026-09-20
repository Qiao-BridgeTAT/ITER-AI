"""Map validation findings to narrow, immutable write targets across all dining roles."""

import copy
import re
from collections import Counter
from typing import Any

from backend.discovery.cards.dining.local_repair import (
    LocalRepairError,
    RepairPlan,
    Target,
    array_target,
)


def schema_at(schema: Any, path: Any) -> Any:
    node = copy.deepcopy(schema)
    for part in path:
        node = node["items"] if isinstance(part, int) else node["properties"][part]
    return copy.deepcopy(node)


def parse_schema_path(text: Any) -> Any:
    if not text.startswith("$"):
        return None
    location = text.split(": ", 1)[0]
    pieces = re.findall(r"\.([^.[\]]+)|\[(\d+)\]", location[1:])
    return tuple(int(index) if index else name for name, index in pieces)


def minimal_context(user: Any, path: Any, original: Any) -> Any:
    # Single-entry repairs receive that entry's facts, not every candidate's facts.
    if len(path) >= 2 and isinstance(path[1], int):
        row = original.get(path[0], [])[path[1]]
        key = row.get("candidate_key") if isinstance(row, dict) else None
        if key:
            return {
                "record": next(
                    (c for c in user.get("candidates", []) if c["candidate_key"] == key), None
                ),
                "frozen_other_fields": {k: v for k, v in row.items() if k != path[-1]},
            }
        if path[0] == "groups":
            return {
                "affected_group": row,
                "candidate_evidence": user.get("candidates", []),
                "frozen_other_groups": [
                    g for i, g in enumerate(original["groups"]) if i != path[1]
                ],
            }
        if path[0] == "directions":
            return {
                "trip_context": user.get("trip_context"),
                "direction": row,
                "other_labels": [
                    d.get("label") for i, d in enumerate(original["directions"]) if i != path[1]
                ],
                "hard_constraints": user.get("hard_constraints", []),
            }
        return {
            "record": row,
            "destination": user.get("destination"),
            "trip_context": user.get("trip_context"),
        }
    return {"destination": user.get("destination"), "trip_context": user.get("trip_context")}


def selection_target(output: Any, user: Any, schema: Any, errors: Any) -> Any:
    selected = output["selected"]
    pool = user["shortlist"]
    facts = {m["candidate_key"]: (g["anchor_candidate_key"], m) for g in pool for m in g["members"]}
    target = user["selection_policy"]["final_card_count"]
    cap = user["selection_policy"].get("max_per_type", 3)
    selected_keys = [row["candidate_key"] for row in selected]
    if len(selected_keys) != len(set(selected_keys)):
        raise LocalRepairError("duplicate_selection_needs_index_scope")
    offenders = set()
    remove_count = 0
    if any(e.startswith("type_exceeds_three:") for e in errors):
        bad_types = {
            e.removeprefix("type_exceeds_three:").rsplit(":", 1)[0]
            for e in errors
            if e.startswith("type_exceeds_three:")
        }
        # Different excess categories need independent budgets; do not unlock a full list.
        if len(bad_types) != 1:
            raise LocalRepairError("multiple_category_conflicts_require_separate_patches")
        bad_type = next(iter(bad_types))
        offenders = {
            key
            for key in selected_keys
            if key in facts and facts[key][1]["model_classification"]["primary_type"] == bad_type
        }
        remove_count = len(offenders) - cap
    elif any(e == "too_many_final_cards" for e in errors):
        offenders = set(selected_keys)
        remove_count = len(selected) - target
    elif any(e.startswith("candidate_not_member_of_group:") for e in errors):
        offenders = {
            row["candidate_key"]
            for row in selected
            if row["candidate_key"] not in facts
            or facts[row["candidate_key"]][0] != row["group_anchor_candidate_key"]
        }
        remove_count = len(offenders)
    elif "duplicate_brand_group" in errors:
        groups = Counter(row["group_anchor_candidate_key"] for row in selected)
        duplicate_anchors = [key for key, n in groups.items() if n > 1]
        if len(duplicate_anchors) != 1:
            raise LocalRepairError("multiple_brand_conflicts_require_separate_patches")
        anchor = duplicate_anchors[0]
        offenders = {
            row["candidate_key"] for row in selected if row["group_anchor_candidate_key"] == anchor
        }
        remove_count = len(offenders) - 1
    elif not any(
        e in {"unnecessary_final_shortfall", "final_shortfall_not_declared"} for e in errors
    ):
        raise LocalRepairError("unmapped_selection_issue")
    add_count = max(0, target - (len(selected) - remove_count))
    locked = [row for row in selected if row["candidate_key"] not in offenders]
    selected_anchors = {row["group_anchor_candidate_key"] for row in selected}
    selected_types = Counter(
        facts[key][1]["model_classification"]["primary_type"]
        for key in selected_keys
        if key in facts
    )
    removable_types = Counter(
        facts[key][1]["model_classification"]["primary_type"] for key in offenders if key in facts
    )
    # A repair cannot add an item whose type is full even after every permitted
    # removal. In particular, an underfill patch cannot add a fourth noodle shop.
    minimum_retained = {
        kind: count - min(remove_count, removable_types[kind])
        for kind, count in selected_types.items()
    }
    available_type_slots = {
        kind: max(0, cap - count) for kind, count in minimum_retained.items() if kind is not None
    }
    # Replacement facts are necessary to decide which new restaurant fits. The
    # frozen selections are reduced to identities and types, never re-emitted.
    alternatives = [
        {
            "group_anchor_candidate_key": anchor,
            "candidate_key": key,
            "name": m["name"],
            "primary_type": m["model_classification"]["primary_type"],
            "provider_category": m.get("provider_category"),
            "provider_food_tags": m.get("provider_food_tags"),
            "known_conflicts": m.get("known_conflicts", []),
        }
        for key, (anchor, m) in facts.items()
        if anchor not in selected_anchors
        and key not in selected_keys
        and not m.get("known_conflicts")
        and (
            m["model_classification"]["primary_type"] is None
            or minimum_retained.get(m["model_classification"]["primary_type"], 0) < cap
        )
    ]
    if add_count and len({m["group_anchor_candidate_key"] for m in alternatives}) < add_count:
        raise LocalRepairError("insufficient_candidates_within_local_repair_scope")
    item_schema = schema_at(schema, ("selected", 0))
    if alternatives:
        item_schema["properties"]["candidate_key"]["enum"] = [
            m["candidate_key"] for m in alternatives
        ]
        item_schema["properties"]["group_anchor_candidate_key"]["enum"] = list(
            dict.fromkeys(m["group_anchor_candidate_key"] for m in alternatives)
        )
    context = {
        "required_final_count": target,
        "max_per_type": cap,
        "type_slots_after_permitted_removals": available_type_slots,
        "type_slot_note": (
            "这是每类最多可新增数量；若少移除该类型，额度相应减少。未知类型不受类型上限约束。"
        ),
        "user_preferences": user.get("user_preferences", {}),
        "frozen_selected": [
            {
                "candidate_key": row["candidate_key"],
                "group_anchor_candidate_key": row["group_anchor_candidate_key"],
                "primary_type": facts[row["candidate_key"]][1]["model_classification"][
                    "primary_type"
                ],
            }
            for row in locked
            if row["candidate_key"] in facts
        ],
        "replacement_candidates": alternatives,
    }
    return array_target(
        ("selected",),
        "; ".join(errors),
        key_field="candidate_key",
        remove_keys=[k for k in selected_keys if k in offenders],
        remove_count=remove_count,
        add_count=add_count,
        item_schema=item_schema,
        context=context,
    )


def build_plan(stage: Any, original: Any, user: Any, schema: Any, errors: Any) -> Any:
    targets = []
    remaining = list(errors)
    if stage == "G":
        coverage = [
            e
            for e in remaining
            if e.startswith(("type_item_must_appear_once:", "unknown_type_member:"))
        ]
        if coverage:
            missing = []
            removable: list[str] = []
            remove_count = 0
            duplicate_sets = []
            for error in coverage:
                key = error.split(":", 1)[1]
                matches = [
                    i for i, item in enumerate(original["items"]) if item["candidate_key"] == key
                ]
                if error.startswith("unknown_type_member:"):
                    removable.extend(f"row_{i}" for i in matches)
                    remove_count += len(matches)
                elif not matches:
                    missing.append(key)
                elif len(matches) > 1:
                    removable.extend(f"row_{i}" for i in matches)
                    remove_count += len(matches) - 1
                    duplicate_sets.append(
                        {
                            "candidate_key": key,
                            "rows": [f"row_{i}" for i in matches],
                            "keep_exactly": 1,
                        }
                    )
            item_schema = schema_at(schema, ("items", 0))
            if missing:
                item_schema["properties"]["candidate_key"]["enum"] = missing
            targets.append(
                array_target(
                    ("items",),
                    "; ".join(coverage),
                    key_field="$index",
                    remove_keys=removable,
                    remove_count=remove_count,
                    add_count=len(missing),
                    item_schema=item_schema,
                    context={
                        "missing_candidates": [
                            c for c in user["candidates"] if c["candidate_key"] in missing
                        ],
                        "duplicate_sets": duplicate_sets,
                    },
                )
            )
            remaining = [e for e in remaining if e not in coverage]
    if stage == "D" and any(not e.startswith("$") for e in errors):
        business = [e for e in errors if not e.startswith("$")]
        targets.append(selection_target(original, user, schema, business))
        if original.get("shortfall"):
            targets.append(
                Target(
                    ("shortfall",),
                    "数量恢复到目标后清除失效的不足说明",
                    {"type": "null", "enum": [None]},
                )
            )
        remaining = [e for e in errors if e.startswith("$")]
    for error in remaining:
        path = parse_schema_path(error)
        if path is not None:
            if not path:
                raise LocalRepairError("root_error_requires_explicit_failure")
            if "unexpected" in error:
                targets.append(
                    Target(path, error, {"type": "boolean", "enum": [True]}, mode="delete_field")
                )
                continue
            if (
                stage == "G"
                and len(path) >= 3
                and path[0] == "items"
                and path[2] == "candidate_key"
            ):
                index = path[1]
                occupied = {
                    row["candidate_key"] for i, row in enumerate(original["items"]) if i != index
                }
                missing = [c for c in user["candidates"] if c["candidate_key"] not in occupied]
                item_schema = schema_at(schema, ("items", 0))
                item_schema["properties"]["candidate_key"]["enum"] = [
                    c["candidate_key"] for c in missing
                ]
                targets.append(
                    Target(
                        ("items", index),
                        error,
                        item_schema,
                        {
                            "candidate_facts": missing,
                            "instruction": "修复这一条身份及依赖该身份的分类，其他条目冻结。",
                        },
                    )
                )
                continue
            if (
                stage == "D"
                and len(path) >= 2
                and path[0] == "selected"
                and isinstance(path[1], int)
            ):
                index = path[1]
                locked = [row for i, row in enumerate(original["selected"]) if i != index]
                alternatives = [
                    {
                        "group_anchor_candidate_key": g["anchor_candidate_key"],
                        "candidate_key": m["candidate_key"],
                        "name": m["name"],
                        "primary_type": m["model_classification"]["primary_type"],
                    }
                    for g in user["shortlist"]
                    for m in g["members"]
                ]
                targets.append(
                    Target(
                        ("selected", index),
                        error,
                        schema_at(schema, ("selected", 0)),
                        {
                            "frozen_selected": locked,
                            "allowed_pairs": alternatives,
                            "selection_policy": user["selection_policy"],
                            "user_preferences": user.get("user_preferences", {}),
                        },
                    )
                )
                continue
            value_schema = schema_at(schema, path)
            # Whole collections must use scoped edit operations, never replacement.
            if len(path) == 1 and value_schema.get("type") == "array":
                raise LocalRepairError("whole_collection_replacement_forbidden")
            targets.append(Target(path, error, value_schema, minimal_context(user, path, original)))
            continue
        if stage == "G" and error.startswith(("invalid_type_evidence:", "missing_type_evidence:")):
            if error.startswith("invalid_type_evidence:"):
                key = error[len("invalid_type_evidence:") :].rsplit(":", 1)[0]
            else:
                key = error[len("missing_type_evidence:") :]
            index = next(
                i for i, item in enumerate(original["items"]) if item["candidate_key"] == key
            )
            row = next(c for c in user["candidates"] if c["candidate_key"] == key)
            path = ("items", index, "evidence_fields")
            value_schema = schema_at(schema, path)
            available = [
                f
                for f in ("name", "provider_category", "provider_food_tags", "provider_alias")
                if row.get(f)
            ]
            if available:
                value_schema["items"]["enum"] = available
            targets.append(Target(path, error, value_schema, minimal_context(user, path, original)))
        elif stage == "G" and error.startswith("type_item_must_appear_once:"):
            key = error[len("type_item_must_appear_once:") :]
            matches = [
                i for i, item in enumerate(original["items"]) if item["candidate_key"] == key
            ]
            if matches:
                raise LocalRepairError("duplicate_classification_needs_index_scope")
            item_schema = schema_at(schema, ("items", 0))
            item_schema["properties"]["candidate_key"]["enum"] = [key]
            facts = next(c for c in user["candidates"] if c["candidate_key"] == key)
            targets.append(
                array_target(
                    ("items",),
                    error,
                    key_field="candidate_key",
                    remove_keys=[],
                    remove_count=0,
                    add_count=1,
                    item_schema=item_schema,
                    context={"missing_candidate": facts},
                )
            )
        elif stage == "F" and error.startswith(
            ("unknown_brand_member:", "brand_member_in_multiple_groups:")
        ):
            key = error.split(":", 1)[1]
            for i, group in enumerate(original["groups"]):
                if key in group["member_keys"]:
                    path = ("groups", i, "member_keys")
                    targets.append(
                        Target(
                            path,
                            error,
                            schema_at(schema, path),
                            {
                                "affected_group": group,
                                "candidates": user["candidates"],
                                "frozen_other_groups": [
                                    g for j, g in enumerate(original["groups"]) if i != j
                                ],
                            },
                        )
                    )
        elif stage == "F" and error in {
            "empty_brand_name",
            "explicit_brand_evidence_missing",
            "brand_group_requires_two_distinct_members",
        }:
            for i, group in enumerate(original["groups"]):
                if error == "empty_brand_name" and not group["brand_name"].strip():
                    field = "brand_name"
                elif (
                    error == "explicit_brand_evidence_missing"
                    and group["basis"] == "explicit_provider"
                ):
                    field = "basis"
                elif error == "brand_group_requires_two_distinct_members" and (
                    len(group["member_keys"]) < 2
                    or len(set(group["member_keys"])) != len(group["member_keys"])
                ):
                    field = "member_keys"
                else:
                    continue
                path = ("groups", i, field)
                targets.append(
                    Target(
                        path,
                        error,
                        schema_at(schema, path),
                        {
                            "group": group,
                            "candidates": [
                                c
                                for c in user["candidates"]
                                if c["candidate_key"] in group["member_keys"]
                            ],
                        },
                    )
                )
        elif stage == "B" and error.startswith(
            (
                "empty_preference_copy:",
                "regular_direction_has_named_restaurant:",
                "empty_search_keyword:",
            )
        ):
            index = int(error.rsplit(":", 1)[1])
            row = original["directions"][index]
            fields = []
            if error.startswith("empty_preference_copy:"):
                fields = [f for f in ("label", "description") if not row[f].strip()]
            elif error.startswith("regular_direction_has_named_restaurant:"):
                fields = ["representative_restaurants"]
            else:
                fields = ["search_keywords"]
            for field in fields:
                path = ("directions", index, field)
                value_schema = schema_at(schema, path)
                if field == "representative_restaurants":
                    value_schema["maxItems"] = 0
                targets.append(
                    Target(path, error, value_schema, minimal_context(user, path, original))
                )
        else:
            raise LocalRepairError("unmapped_issue_no_full_retry:" + error)
    return RepairPlan.build(original, targets)
