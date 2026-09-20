"""Audited Flash tasks with API schemas and immutable, narrowly scoped repair."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import RootModel

from backend.agent.model_audit import current_model_audit_execution, record_execution_event
from backend.agent.model_gateway import ModelAuditMetadata, ModelMessage, ModelRequest, ModelRole
from backend.discovery.cards.dining.local_repair import PATCH_SYSTEM, LocalRepairError
from backend.discovery.cards.dining.prompts import (
    DINING_PREFERENCE_PROMPT_VERSION,
    MAIN_MEAL_DIRECTION_RULES,
    PROMPTS,
)
from backend.discovery.cards.dining.repair_plans import build_plan
from backend.discovery.cards.dining.validation import schema_errors, specialize

VERSION = "dining-v3-flash-schema-parallel-scoped-20260918"
SCHEMAS = {
    key[0]: value
    for key, value in json.loads(Path(__file__).with_name("schemas.json").read_text()).items()
}
STAGES = {
    "A": "city_representatives",
    "B": "preference",
    "F": "brand_groups",
    "G": "primary_types",
    "D": "selection",
}


class DiningModelOutput(RootModel[dict[str, Any]]):
    """Retain the parsed object for local validation instead of losing repairable fields."""


async def audit(event: Any, payload: Any) -> None:
    context = current_model_audit_execution()
    if context is not None:
        await record_execution_event(context, event, payload=payload)


async def run_task(
    gateway: Any,
    stage: Any,
    user: Any,
    *,
    cancellation: Any = None,
    validator: Any = None,
    **schema_args: Any,
) -> Any:
    schema = specialize(SCHEMAS[stage], **schema_args)

    async def call(
        system: Any, payload: Any, output_schema: Any, *, repair_of: Any = None, budget: Any = 8192
    ) -> Any:
        return await gateway.generate_structured(
            ModelRequest(
                messages=[
                    ModelMessage(role=ModelRole.SYSTEM, content=system),
                    ModelMessage(
                        role=ModelRole.USER,
                        content=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                    ),
                ],
                structured_output_mode="json_schema",
                output_schema_override=output_schema,
                max_output_tokens=budget,
                temperature_override=0.2,
                request_timeout_seconds=120,
                audit=ModelAuditMetadata(
                    stage="prepare_dining_" + STAGES[stage] + ("_patch" if repair_of else ""),
                    contract_version=DINING_PREFERENCE_PROMPT_VERSION if stage == "B" else VERSION,
                    repair=bool(repair_of),
                    attempt=2 if repair_of else 1,
                    repair_of_call_id=repair_of,
                ),
            ),
            DiningModelOutput,
            cancellation=cancellation,
        )

    result = await call(PROMPTS[stage], user, schema)
    original = result.value.root
    errors = schema_errors(original, schema)
    if not errors and validator:
        errors = validator(original)
    if not errors:
        await audit(
            "dining_task_validated",
            {"stage": STAGES[stage], "call_id": result.audit_call_id, "repaired": False},
        )
        return original
    plan = build_plan(stage, original, user, schema, errors)
    patch = await call(
        PATCH_SYSTEM
        + "\n保持原任务的身份、品牌与类型约束。"
        + ("\n" + MAIN_MEAL_DIRECTION_RULES if stage == "B" else ""),
        plan.model_input(original),
        plan.output_schema(),
        repair_of=result.audit_call_id or "local-validation",
        budget=min(4096, 256 + 512 * len(plan.targets)),
    )
    merged = plan.apply(original, patch.value.root, schema_errors)
    remaining = schema_errors(merged, schema)
    if not remaining and validator:
        remaining = validator(merged)
    if remaining:
        raise LocalRepairError("local_patch_failed_validation_no_full_retry")
    await audit(
        "dining_local_patch_applied",
        {
            "stage": STAGES[stage],
            "call_id": result.audit_call_id,
            "paths": [list(t.path) for t in plan.targets],
            "unaffected_content_frozen": True,
        },
    )
    return merged


def preference_errors(output: Any) -> Any:
    from backend.contracts.v4.content_quality import (
        require_meaningful_description,
        require_meaningful_label,
    )

    errors = []
    if not output["directions"]:
        # No original direction exists to repair locally; fail without regenerating.
        raise LocalRepairError("empty_directions_no_localizable_content")
    for i, row in enumerate(output["directions"]):
        for field, check in [
            ("label", require_meaningful_label),
            ("description", require_meaningful_description),
        ]:
            try:
                check(row[field], field)
            except ValueError:
                errors.append(f"$.directions[{i}].{field}: provide meaningful Chinese {field}")
        if row["kind"] == "regular" and row["representative_restaurants"]:
            errors.append(f"regular_direction_has_named_restaurant:{i}")
        if not all(k.strip() for k in row["search_keywords"]):
            errors.append(f"empty_search_keyword:{i}")
    return errors


def preference_context(state: Any) -> Any:
    dining = state.dining
    return {
        "trip_context": {
            "destination": state.trip_basics.destination_name,
            "duration_days": state.trip_basics.duration_days,
            "explicit_dining_preferences": dining.requirements,
            "selected_directions": [
                d.model_dump(mode="json") for d in dining.preference_directions if d.selected
            ],
        },
        "hard_constraints": {"allergies": dining.allergies, "avoidances": dining.avoidances},
        "explicit_exclusions": [
            d.model_dump(mode="json") for d in dining.preference_directions if not d.selected
        ],
        "evidence": [],
    }
