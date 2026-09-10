"""Stable pseudonymous ownership helpers for temporary V4 trips."""

from __future__ import annotations

from uuid import UUID, uuid5

ANONYMOUS_V4_OWNER_NAMESPACE = UUID("c768d9d2-08eb-4e67-a160-b8f15e56f5d6")
ANONYMOUS_V4_OWNER_NICKNAME = "__iter_v4_guest__"


def anonymous_v4_owner_id(session_id: str) -> UUID:
    """Derive a non-reversible database owner key from a high-entropy session ID."""

    normalized = session_id.strip()
    if not normalized:
        raise ValueError("anonymous session ID cannot be empty")
    return uuid5(ANONYMOUS_V4_OWNER_NAMESPACE, normalized)
