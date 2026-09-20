"""Native tool turns over the audited streaming gateway; never parse prose as calls."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

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
    ModelUsage,
)

MAX_TOOL_BATCH_SIZE = 4


@dataclass(frozen=True)
class ToolTurn:
    message: ModelMessage
    usage: ModelUsage | None
    audit_call_id: str | None
    # A complete, unambiguous decision may contain rejected arguments. Keep
    # these as non-executable receipts so the next decision can repair them.
    argument_errors: dict[str, list[dict[str, Any]]] = field(default_factory=dict)


async def generate_tool_turn(
    gateway: ModelGateway,
    request: ModelRequest,
    *,
    cancellation: ModelCancellation,
    on_public_text: Callable[[str], Awaitable[None]] | None = None,
) -> ToolTurn:
    """Assemble indexed argument fragments and reject incomplete/ambiguous turns."""
    content: list[str] = []
    fragments: dict[int, dict[str, str]] = {}
    usage = None
    call_id = None
    finish = None
    public_buffer = ""
    public_sent = False
    async for chunk in gateway.stream_text(request, cancellation=cancellation):
        call_id = chunk.audit_call_id or call_id
        if chunk.usage is not None:
            usage = chunk.usage
        if chunk.finish_reason is not None:
            finish = chunk.finish_reason
        content.append(chunk.delta)
        if not public_sent:
            public_buffer += chunk.delta
        # Publish complete public sentences, not partial JSON arguments or reasoning.
        if on_public_text and not public_sent:
            while any(mark in public_buffer for mark in ("。", "！", "？", "\n")):
                end = (
                    min(
                        public_buffer.index(m)
                        for m in ("。", "！", "？", "\n")
                        if m in public_buffer
                    )
                    + 1
                )
                sentence, public_buffer = public_buffer[:end].strip(), public_buffer[end:]
                if sentence:
                    # Only the requested short action summary is public. Long
                    # narration, internal identifiers and subsequent analysis
                    # remain in the private tool dialogue, not process history.
                    if len(sentence) <= 160 and not any(
                        token in sentence for token in ("_", "```", "{", "}")
                    ):
                        await on_public_text(sentence)
                    public_sent = True
                    public_buffer = ""
                    break
        for delta in chunk.tool_call_deltas:
            part = fragments.setdefault(delta.index, {"id": "", "name": "", "arguments": ""})
            if delta.id:
                if part["id"] and part["id"] != delta.id:
                    raise _malformed(call_id)
                part["id"] = delta.id
            part["name"] += delta.name
            part["arguments"] += delta.arguments
            if len(part["arguments"]) > 131_072:
                raise _malformed(call_id)
    # Qwen named tool_choice can end a valid native call with "stop". Its
    # structured deltas remain authoritative; never recover a call from prose.
    forced_stop = finish == "stop" and request.forced_tool_name is not None and bool(fragments)
    if finish not in {"stop", "tool_calls"} or (
        bool(fragments) != (finish == "tool_calls") and not forced_stop
    ):
        raise _malformed(call_id)
    calls = []
    seen: set[str] = set()
    for index in sorted(fragments):
        part = fragments[index]
        try:
            arguments = json.loads(part["arguments"])
        except (ValueError, TypeError):
            raise _malformed(call_id) from None
        if (
            not isinstance(arguments, dict)
            or not part["id"]
            or part["id"] in seen
            or not part["name"]
        ):
            raise _malformed(call_id)
        seen.add(part["id"])
        calls.append(
            ModelToolCall(
                id=part["id"],
                function=ModelToolFunction(
                    name=part["name"],
                    arguments=part["arguments"],
                ),
            )
        )
    text = "".join(content)
    if not calls and not text.strip():
        raise _malformed(call_id)
    if on_public_text and not public_sent and public_buffer.strip():
        sentence = public_buffer.strip()
        if len(sentence) <= 160 and not any(token in sentence for token in ("_", "```", "{", "}")):
            await on_public_text(sentence)
    return ToolTurn(
        ModelMessage(role=ModelRole.ASSISTANT, content=text, tool_calls=tuple(calls)),
        usage,
        call_id,
    )


def _malformed(call_id: str | None) -> ModelGatewayError:
    return ModelGatewayError(
        ModelFailureCode.MALFORMED_RESPONSE, "tool_dialogue", retryable=False, audit_call_id=call_id
    )
