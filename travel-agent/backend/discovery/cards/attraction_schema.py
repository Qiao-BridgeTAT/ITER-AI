"""Flash wire compatibility only; original Pydantic validators remain authoritative."""

from typing import Any

from pydantic import BaseModel

from backend.agent.json_schema import flash_wire_schema


def attraction_schema(
    model: type[BaseModel],
    *,
    candidate_keys: list[str] | None = None,
    maximum_target: int | None = None,
) -> dict[str, Any]:
    schema = flash_wire_schema(model.model_json_schema())
    if candidate_keys is not None:
        for name in ("SelectedAttraction", "RejectedAttraction"):
            if name in schema.get("$defs", {}):
                schema["$defs"][name]["properties"]["candidate_key"]["enum"] = candidate_keys
        schema["required"] = list(schema["properties"])
        selected = schema.get("$defs", {}).get("SelectedAttraction")
        if selected is not None:
            selected["properties"]["suggested_visit_duration"] = {
                "$ref": "#/$defs/VisitDurationRange"
            }
            selected["required"] = list(
                dict.fromkeys([*selected["required"], "suggested_visit_duration"])
            )
    if maximum_target is not None:
        schema["properties"]["selected"]["maxItems"] = maximum_target
    return schema
