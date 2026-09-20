"""Carry one native LangGraph interrupt in the existing workspace checkpoint.

The official saver owns graph scheduling. PostgreSQL durability remains in the
application's CheckpointRepository; no second database or long-lived memory
store is needed. Only the paused graph's current checkpoint is retained.
"""

from base64 import b64decode, b64encode
from collections import defaultdict
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver


async def dump_interrupt(saver: InMemorySaver, config: Any) -> dict[str, Any]:
    saved = await saver.aget_tuple(config)
    if saved is None:
        raise RuntimeError("native_interrupt_checkpoint_missing")
    payload = {}
    for key, value in (
        ("checkpoint", saved.checkpoint),
        ("metadata", saved.metadata),
        ("pending_writes", saved.pending_writes or []),
    ):
        encoding, data = saver.serde.dumps_typed(value)
        payload[key] = [encoding, b64encode(data).decode("ascii")]
    return payload


async def restore_interrupt(saver: InMemorySaver, config: Any, payload: dict[str, Any]) -> None:
    values = {
        key: saver.serde.loads_typed((encoding, b64decode(data)))
        for key, (encoding, data) in payload.items()
    }
    checkpoint = values["checkpoint"]
    restored = await saver.aput(
        config, checkpoint, values["metadata"], checkpoint["channel_versions"]
    )
    grouped: dict[str, list[tuple[str, Any]]] = defaultdict(list)
    for task_id, channel, value in values["pending_writes"]:
        grouped[task_id].append((channel, value))
    for task_id, writes in grouped.items():
        await saver.aput_writes(restored, writes, task_id)
