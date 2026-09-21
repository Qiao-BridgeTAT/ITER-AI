"""Schema-constrained Agent decisions over the same durable tool execution protocol."""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable, Collection
from copy import deepcopy
from typing import Any
from uuid import uuid4

from jsonschema import Draft202012Validator  # type: ignore[import-untyped]

from backend.agent.json_schema import flash_wire_schema
from backend.agent.model_audit import record_model_call_annotation
from backend.agent.model_gateway import (
    ModelCancellation,
    ModelFailureCode,
    ModelGateway,
    ModelGatewayError,
    ModelMessage,
    ModelRequest,
    ModelRole,
    ModelToolCall,
    ModelToolFunction,
)
from backend.agent.tool_dialogue import MAX_TOOL_BATCH_SIZE, ToolTurn
from backend.agent.tool_schema_errors import tool_schema_error_details

SCHEMA_TURN_INSTRUCTION = (
    "\n## 本轮输出\n"
    "按响应 Schema 输出一个对象。先在 public_summary 写给用户的一句行动或发现摘要，"
    "再在 tool_calls 选择工具并填写参数。工具用途与参数见本轮实际工具定义；"
    "不要在对象外输出文字。工具执行结果将在下一轮作为资料返回，提出调用不代表执行成功。"
    "历史 tool_result 记录已执行的请求及结果，不是本轮待执行列表。"
)


class SchemaDecisionError(ModelGatewayError):
    """Keep a complete rejected decision in private repair context, never in error text."""

    def __init__(self, issues: tuple[str, ...], decision: dict[str, Any], call_id: str | None):
        super().__init__(
            ModelFailureCode.MALFORMED_RESPONSE,
            "schema_tool_dialogue",
            retryable=False,
            validation_issues=issues,
            audit_call_id=call_id,
        )
        self.rejected_decision = decision


def tool_turn_schema(
    request: ModelRequest, *, parallel_tool_names: Collection[str] | None = None
) -> dict[str, Any]:
    """Use the allowed tools verbatim, relocating local refs without collisions."""
    definitions: dict[str, Any] = {}
    choices: list[dict[str, Any]] = []
    for index, tool in enumerate(request.tools):
        if request.forced_tool_name and tool.name != request.forced_tool_name:
            continue
        key = f"tool_{index}"
        parameters = deepcopy(tool.parameters)

        def relocate(value: Any, root: str = key) -> None:
            if isinstance(value, dict):
                ref = value.get("$ref")
                if isinstance(ref, str) and ref.startswith("#"):
                    value["$ref"] = f"#/$defs/{root}{ref[1:]}"
                for child in value.values():
                    relocate(child, root)
            elif isinstance(value, list):
                for child in value:
                    relocate(child, root)

        relocate(parameters)
        definitions[key] = parameters
        choices.append(
            {
                "type": "object",
                "description": tool.description,
                "properties": {
                    "name": {"type": "string", "enum": [tool.name]},
                    "arguments": {"$ref": f"#/$defs/{key}"},
                },
                "required": ["name", "arguments"],
                "additionalProperties": False,
            }
        )
    if not choices:
        raise ValueError("schema Agent requires at least one allowed tool")
    calls: dict[str, Any] = {
        "type": "array",
        "description": "独立查询最多四项；编辑、试算、评审或完成每次仅选一项。"
        "后续行动等待观察结果。",
        "minItems": 1 if request.forced_tool_name else 0,
        "maxItems": 1
        if request.forced_tool_name or not request.parallel_tool_calls
        else MAX_TOOL_BATCH_SIZE,
        "items": {"anyOf": choices},
    }
    # Qwen rejects uniqueItems on arrays (verified with a synthetic real request).
    # Duplicate calls still fail the runtime's canonical batch check before I/O.
    if parallel_tool_names is not None and calls["maxItems"] > 1:
        queries = [
            choice
            for choice in choices
            if choice["properties"]["name"]["enum"][0] in parallel_tool_names
        ]
        if queries:
            # Tool permissions are known by code. Constrain the turn grammar,
            # without rewriting any official MCP definition or choosing an action.
            calls["anyOf"] = [
                {"maxItems": 1},
                {"minItems": 2, "items": {"anyOf": queries}},
            ]
        else:
            calls["maxItems"] = 1
    return {
        "type": "object",
        "properties": {
            "public_summary": {
                "type": "string",
                "minLength": 1,
                "maxLength": 160,
                "description": "给用户的一句简短行动或发现摘要；不含私有推理、工具名或参数。",
            },
            "tool_calls": calls,
        },
        "required": ["public_summary", "tool_calls"],
        "additionalProperties": False,
        "$defs": definitions,
    }


def schema_dialogue(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Keep executed arguments and observations, not obsolete action narration.

    Replaying previous decision JSON as assistant output can prime the next
    constrained turn to copy it despite failure feedback. Preserve the actual
    action once in its identified result; durable native messages are unchanged.
    """
    result = []
    calls: dict[str, ModelToolCall] = {}
    for message in messages:
        if message.tool_calls:
            calls.update((call.id, call) for call in message.tool_calls)
        elif message.role is ModelRole.TOOL:
            assert message.tool_call_id is not None
            call = calls[message.tool_call_id]
            try:
                observation = json.loads(message.content)
            except ValueError:
                observation = message.content
            rejected_plan = (
                call.function.name in {"write_plan", "edit_plan", "patch_plan"}
                and isinstance(observation, dict)
                and observation.get("ok") is False
            )
            result.append(
                ModelMessage(
                    role=ModelRole.USER,
                    content=json.dumps(
                        {
                            "tool_result": {
                                "name": call.function.name,
                                **(
                                    {
                                        "argument_status": "rejected_not_applied",
                                        "next_action": "根据当前已保存草稿和具体错误重新构造修改；"
                                        "失败参数保留在 checkpoint，此处不重复整份未生效方案。",
                                    }
                                    if rejected_plan
                                    else {"arguments": json.loads(call.function.arguments)}
                                ),
                                "observation": observation,
                            }
                        },
                        ensure_ascii=False,
                    ),
                )
            )
        else:
            result.append(message)
    # The short format instruction belongs to system authority, not observations.
    if result and result[0].role is ModelRole.SYSTEM:
        result[0] = result[0].model_copy(
            update={"content": result[0].content + SCHEMA_TURN_INSTRUCTION}
        )
    else:
        result.insert(0, ModelMessage(role=ModelRole.SYSTEM, content=SCHEMA_TURN_INSTRUCTION))
    return result


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _public_summary(buffer: str) -> str | None:
    # Only decode the explicitly public first field, after its closing quote.
    # Escaped quotes, unicode and chunk boundaries are handled by the JSON parser.
    match = re.match(r'\s*\{\s*"public_summary"\s*:\s*', buffer)
    if match is None:
        return None
    try:
        value, _ = json.JSONDecoder().raw_decode(buffer[match.end() :])
    except ValueError:
        return None
    return value if isinstance(value, str) else None


def normalize_public_summary(text: str) -> str:
    """Presentation-only shortening; never rewrite the selected tool or arguments."""
    text = " ".join(text.split())
    if any(token in text for token in ("_", "```", "{", "}")):
        return ""
    if len(text) <= 160:
        return text
    sentence = re.search(r"[。！？!?][”’」』]?", text[:160])
    return text[: sentence.end()] if sentence else text[:159].rstrip() + "…"


def _envelope_issues(schema: dict[str, Any], value: Any) -> tuple[str, ...]:
    """Schema-owned expectations and paths only; never include rejected input values."""
    allowed = [
        choice["properties"]["name"]["enum"][0]
        for choice in schema["properties"]["tool_calls"]["items"]["anyOf"]
    ]

    def leaves(error: Any) -> Any:
        if error.context:
            for child in error.context:
                yield from leaves(child)
        else:
            yield error

    issues = []
    for error in Draft202012Validator(schema).iter_errors(value):
        for leaf in leaves(error):
            path = list(leaf.absolute_path)
            if leaf.validator == "enum" and path[-1:] == ["name"]:
                if leaf.instance in allowed:
                    continue  # Ignore mismatching alternatives for a known tool.
                expected = allowed
            else:
                expected = leaf.validator_value
            detail = {"loc": path, "type": leaf.validator, "expected": expected}
            if leaf.validator == "additionalProperties":
                detail["allowed_fields"] = list(leaf.schema.get("properties", {}))
                if isinstance(leaf.instance, dict):
                    detail["unexpected_fields"] = sorted(
                        set(leaf.instance) - set(detail["allowed_fields"])
                    )
                    detail["repair"] = "删除 unexpected_fields 列出的字段，保留其他方案内容。"
            elif leaf.validator == "maxItems" and path == ["tool_calls"]:
                detail["actual_count"] = len(leaf.instance)
                detail["repair"] = (
                    "写入、修改、试算、评审、完成存在状态依赖，每轮只提交一个。"
                    "先单独提交前置操作，收到成功结果后再请求下一项；本轮全部尚未执行。"
                )
            item = json.dumps(detail, ensure_ascii=False)
            if item not in issues:
                issues.append(item)
    return tuple(issues[:5])


async def generate_schema_tool_turn(
    gateway: ModelGateway,
    request: ModelRequest,
    *,
    cancellation: ModelCancellation,
    on_public_text: Callable[[str], Awaitable[None]] | None = None,
    parallel_tool_names: Collection[str] | None = None,
) -> ToolTurn:
    schema = tool_turn_schema(request, parallel_tool_names=parallel_tool_names)
    messages = schema_dialogue(request.messages)
    # response_format constrains decoding. Do not rely on its annotations being
    # visible semantic input: carry the same discovered descriptions and input
    # schemas in model context, as native function-calling transport would.
    catalog = [
        tool.model_dump(mode="json")
        for tool in request.tools
        if not request.forced_tool_name or tool.name == request.forced_tool_name
    ]
    messages[0] = messages[0].model_copy(
        update={
            "content": messages[0].content
            + "\n## 本轮实际工具定义\n"
            + json.dumps(catalog, ensure_ascii=False, separators=(",", ":"))
        }
    )
    wire = request.model_copy(
        update={
            "messages": messages,
            "tools": (),
            "forced_tool_name": None,
            "structured_output_mode": "json_schema",
            "output_schema_override": flash_wire_schema(schema),
            "thinking_budget_tokens": None,
            "reasoning_timeout_seconds": None,
        }
    )
    buffer = ""
    usage = None
    call_id = None
    finish = None
    public_sent = False

    async def publish(text: str | None) -> None:
        nonlocal public_sent
        if text is None or public_sent:
            return
        public_sent = True
        summary = normalize_public_summary(text)
        if on_public_text and summary:
            await on_public_text(summary)

    def malformed(*issues: str) -> ModelGatewayError:
        return ModelGatewayError(
            ModelFailureCode.MALFORMED_RESPONSE,
            "schema_tool_dialogue",
            retryable=False,
            validation_issues=issues,
            audit_call_id=call_id,
        )

    try:
        async for chunk in gateway.stream_text(wire, cancellation=cancellation):
            call_id = chunk.audit_call_id or call_id
            cancellation.raise_if_cancelled("schema_tool_dialogue")
            usage = chunk.usage or usage
            finish = chunk.finish_reason or finish
            if chunk.tool_call_deltas:
                raise malformed(
                    "$.tool_calls: expected JSON decisions, received native tool deltas"
                )
            buffer += chunk.delta
            if len(buffer) > 131_072:
                raise malformed("$: response exceeds 131072 characters; reduce the decision size")
            if not public_sent:
                await publish(_public_summary(buffer))
        if finish != "stop":
            raise malformed(
                "$: incomplete response; return a complete JSON decision within the output budget"
            )
        try:
            value = json.loads(buffer, object_pairs_hook=_unique_object)
        except json.JSONDecodeError as error:
            raise malformed(
                f"$: invalid JSON at line {error.lineno}, column {error.colno}; "
                "return complete JSON"
            ) from None
        except ValueError:
            raise malformed("$: duplicate JSON property; each field must occur once") from None
        # Never recover calls from ambiguous/truncated JSON or unknown names.
        # Argument errors in an otherwise complete decision become rejected
        # receipts, just like native tool calls, rather than ending the run.
        envelope = deepcopy(schema)
        envelope["$defs"] = {key: {"type": "object"} for key in schema["$defs"]}
        # Length is a display concern. A long/empty string must not discard a
        # complete repair request; types, envelope and all tool contracts still apply.
        envelope["properties"]["public_summary"] = {"type": "string"}
        if issues := _envelope_issues(envelope, value):
            if isinstance(value, dict):
                raise SchemaDecisionError(issues, value, call_id)
            raise malformed(*issues)
        await publish(value["public_summary"])
        calls = tuple(
            ModelToolCall(
                id=f"schema-{uuid4()}",
                function=ModelToolFunction(
                    name=call["name"],
                    arguments=json.dumps(call["arguments"], ensure_ascii=False),
                ),
            )
            for call in value["tool_calls"]
        )
        definitions = {tool.name: tool.parameters for tool in request.tools}
        argument_errors = {}
        for call in calls:
            errors = list(
                Draft202012Validator(definitions[call.function.name]).iter_errors(
                    json.loads(call.function.arguments)
                )
            )
            if errors:
                # Validate the original (including dynamically bound) contract.
                # Report only schema-owned expectations, never arbitrary inputs.
                argument_errors[call.id] = tool_schema_error_details(
                    errors, schema=definitions[call.function.name]
                )
    except ModelGatewayError as error:
        rejected = (
            error.operation == "schema_tool_dialogue"
            and error.code is ModelFailureCode.MALFORMED_RESPONSE
        )
        await record_model_call_annotation(
            gateway,
            call_id,
            "llm_business_guard",
            {
                "schema_validation_result": {"status": "rejected" if rejected else "not_completed"},
                "accepted_or_rejected": "rejected",
                "failure_stage": "schema_tool_dialogue" if rejected else error.operation,
                "failure_code": error.code.value,
                "validation_issues": list(error.validation_issues),
            },
        )
        raise
    await record_model_call_annotation(
        gateway,
        call_id,
        "llm_business_guard",
        {
            "schema_validation_result": {
                "status": "arguments_rejected" if argument_errors else "accepted"
            },
            "accepted_or_rejected": "arguments_rejected" if argument_errors else "schema_accepted",
            "public_summary_normalization": {
                "changed": value["public_summary"]
                != normalize_public_summary(value["public_summary"]),
                "input_characters": len(value["public_summary"]),
                "display_characters": len(normalize_public_summary(value["public_summary"])),
            },
        },
    )
    return ToolTurn(
        ModelMessage(
            role=ModelRole.ASSISTANT,
            content=normalize_public_summary(value["public_summary"]),
            tool_calls=calls,
        ),
        usage,
        call_id,
        argument_errors,
    )
