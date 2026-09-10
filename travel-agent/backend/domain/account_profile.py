"""Deterministic account-profile defaults shared by authentication and REST."""

from __future__ import annotations

from uuid import UUID


def effective_nickname(user_id: UUID, nickname: str | None) -> str:
    """Return a stored nickname or a stable four-digit default for the account."""

    if nickname is not None and nickname.strip():
        return nickname.strip()
    suffix = int(user_id.hex[-8:], 16) % 10_000
    return f"用户{suffix:04d}"
