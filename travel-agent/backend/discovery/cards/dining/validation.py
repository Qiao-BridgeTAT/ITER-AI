"""Schema and final selection checks, independent of model output claims."""

import copy
from collections import Counter
from typing import Any


def schema_errors(value: Any, schema: Any, path: Any = "$") -> Any:
    """Validate every keyword actually used in these simple review schemas."""
    supported = {
        "type",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "enum",
        "minItems",
        "maxItems",
    }
    unsupported = set(schema) - supported
    if unsupported:
        raise ValueError("Unsupported schema keywords: " + ",".join(sorted(unsupported)))
    errors = []
    types = schema.get("type", [])
    types = [types] if isinstance(types, str) else types
    matches = {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "null": value is None,
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
    }
    if types and not any(matches.get(kind, False) for kind in types):
        return [f"{path}: wrong type, expected {types}"]
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: not in allowed enum")
    if isinstance(value, dict):
        props = schema.get("properties", {})
        errors += [
            f"{path}.{key}: missing" for key in schema.get("required", []) if key not in value
        ]
        if schema.get("additionalProperties") is False:
            errors += [f"{path}.{key}: unexpected" for key in value if key not in props]
        for key, child in value.items():
            if key in props:
                errors.extend(schema_errors(child, props[key], f"{path}.{key}"))
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append(f"{path}: fewer than minItems")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{path}: more than maxItems")
    if isinstance(value, list) and "items" in schema:
        for index, child in enumerate(value):
            errors.extend(schema_errors(child, schema["items"], f"{path}[{index}]"))
    return errors


def specialize(
    schema: Any, candidate_keys: Any = None, anchors: Any = None, direction_ids: Any = None
) -> Any:
    schema = copy.deepcopy(schema)

    def visit(node: Any, prop: Any = None) -> Any:
        if isinstance(node, dict):
            if prop in {"candidate_key", "anchor_candidate_key"} and candidate_keys:
                node["enum"] = candidate_keys
            if prop == "group_anchor_candidate_key" and anchors:
                node["enum"] = anchors
            if prop == "member_keys" and candidate_keys:
                node["items"]["enum"] = candidate_keys
            if prop == "shortlist_anchor_keys" and candidate_keys:
                node["items"]["enum"] = candidate_keys
            if prop == "related_direction_ids" and direction_ids:
                node["items"]["enum"] = direction_ids
            for key, child in node.items():
                if key == "properties":
                    for field, definition in child.items():
                        visit(definition, field)
                elif key != "enum":
                    visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(schema)
    return schema


def type_capacity(shortlist: Any) -> Any:
    """Maximum group count under type cap; unknowns remain unknown, not a cuisine."""
    occupied: dict[Any, Any] = {}
    edges = {}
    for group in shortlist:
        anchor = group["anchor_candidate_key"]
        types = {m["model_classification"]["primary_type"] for m in group["members"]}
        edges[anchor] = [(kind, i) for kind in types if kind for i in range(3)]
        if None in types:
            edges[anchor].append(("unknown:" + anchor, 0))

    def match(anchor: Any, visited: Any) -> Any:
        for slot in edges[anchor]:
            if slot in visited:
                continue
            visited.add(slot)
            if slot not in occupied or match(occupied[slot], visited):
                occupied[slot] = anchor
                return True
        return False

    return sum(match(anchor, set()) for anchor in edges)


def validate_final(output: Any, shortlist: Any, final_count: Any) -> Any:
    lookup = {
        g["anchor_candidate_key"]: {m["candidate_key"]: m for m in g["members"]} for g in shortlist
    }
    errors = []
    selected_anchors = []
    types: Counter[str] = Counter()
    for row in output["selected"]:
        anchor, key = row["group_anchor_candidate_key"], row["candidate_key"]
        selected_anchors.append(anchor)
        if key not in lookup.get(anchor, {}):
            errors.append(f"candidate_not_member_of_group:{anchor}:{key}")
            continue
        category = lookup[anchor][key]["model_classification"]["primary_type"]
        if category:
            types[category] += 1
    if len(selected_anchors) != len(set(selected_anchors)):
        errors.append("duplicate_brand_group")
    if len(selected_anchors) > final_count:
        errors.append("too_many_final_cards")
    if len(selected_anchors) < final_count:
        if type_capacity(shortlist) >= final_count:
            errors.append("unnecessary_final_shortfall")
        elif not output["shortfall"]:
            errors.append("final_shortfall_not_declared")
    errors += [f"type_exceeds_three:{kind}:{count}" for kind, count in types.items() if count > 3]
    return errors
