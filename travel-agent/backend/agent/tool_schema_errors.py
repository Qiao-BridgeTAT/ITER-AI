"""Small, field-specific repair feedback for original tool JSON Schemas."""

import re
from collections.abc import Iterable, Iterator
from typing import Any

from jsonschema.exceptions import best_match  # type: ignore[import-untyped]


def _resolve_local(schema: dict[str, Any], root: dict[str, Any]) -> dict[str, Any]:
    seen = set()
    while isinstance(schema.get("$ref"), str):
        ref = schema["$ref"]
        if not ref.startswith("#/") or ref in seen:
            break
        seen.add(ref)
        resolved: Any = root
        for part in ref[2:].split("/"):
            if not isinstance(resolved, dict):
                return schema
            resolved = resolved.get(part.replace("~1", "/").replace("~0", "~"))
        if not isinstance(resolved, dict):
            return schema
        schema = resolved
    return schema


def _branch(error: Any, root: dict[str, Any]) -> int | None:
    choices = error.schema.get(error.validator)
    if not isinstance(error.instance, dict) or not isinstance(choices, list):
        return None
    # Infer a discriminator only when every branch declares a closed domain.
    # For plan stops this is candidate_key; no Agent choice is changed here.
    for key, value in error.instance.items():
        domains = []
        for choice in choices:
            choice = _resolve_local(choice, root)
            field = choice.get("properties", {}).get(key, {})
            domains.append(field.get("enum", [field["const"]] if "const" in field else None))
        if all(isinstance(domain, list) for domain in domains):
            matches = [i for i, domain in enumerate(domains) if value in domain]
            if len(matches) == 1:
                return matches[0]
    return None


def _leaves(error: Any, root: dict[str, Any]) -> Iterator[Any]:
    if not error.context:
        yield error
        return
    branch = _branch(error, root)
    children = [
        child
        for child in error.context
        if branch is not None and child.schema_path and child.schema_path[0] == branch
    ]
    if children:
        for child in children:
            yield from _leaves(child, root)
    else:
        closest = best_match(error.context)
        if closest is not None:
            yield from _leaves(closest, root)


def tool_schema_error_details(
    errors: Iterable[Any], *, schema: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    for error in errors:
        for leaf in _leaves(error, schema or {}):
            path = list(leaf.absolute_path)
            rows: list[dict[str, Any]]
            if leaf.validator == "required" and isinstance(leaf.instance, dict):
                rows = [
                    {"loc": [*path, key], "type": "required", "expected": "required"}
                    for key in leaf.validator_value
                    if key not in leaf.instance
                ]
            elif leaf.validator == "additionalProperties" and isinstance(leaf.instance, dict):
                fields = list(leaf.schema.get("properties", {}))
                patterns = leaf.schema.get("patternProperties", {})
                rows = [
                    {
                        "loc": [*path, key],
                        "type": "additionalProperties",
                        "expected": False,
                        "allowed_fields": fields,
                    }
                    for key in leaf.instance
                    if key not in fields
                    and not any(re.search(pattern, key) for pattern in patterns)
                ]
            else:
                expected = leaf.validator_value
                if isinstance(expected, (dict, list)) and leaf.validator not in {"enum", "type"}:
                    expected = "must_match_tool_schema"
                row = {"loc": path, "type": leaf.validator, "expected": expected}
                rows = [row]
            for row in rows:
                if row not in details:
                    details.append(row)
                if len(details) >= 5:
                    return details
    return details
