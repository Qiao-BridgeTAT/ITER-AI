"""Flash wire compatibility; business validators keep the original schema."""

from copy import deepcopy
from typing import Any


def flash_wire_schema(schema: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(schema)

    def adapt(value: Any) -> None:
        if isinstance(value, dict):
            # Same compatibility rule as Prepare: this pattern has caused
            # Flash constrained decoding to emit one-character values.
            if value.get("pattern") == r"\S":
                value.pop("pattern")
            for child in value.values():
                adapt(child)
        elif isinstance(value, list):
            for child in value:
                adapt(child)

    adapt(result)
    return result
