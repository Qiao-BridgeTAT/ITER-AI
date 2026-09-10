"""Expand V4 semantic operation kinds to fit the frozen contract.

Revision ID: 20260830_0008
Revises: 20260829_0007
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision = "20260830_0008"
down_revision = "20260829_0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("semantic_operations") as batch_op:
        batch_op.alter_column(
            "operation_kind",
            existing_type=sa.String(length=24),
            type_=sa.String(length=64),
            existing_nullable=False,
        )


def downgrade() -> None:
    with op.batch_alter_table("semantic_operations") as batch_op:
        batch_op.alter_column(
            "operation_kind",
            existing_type=sa.String(length=64),
            type_=sa.String(length=24),
            existing_nullable=False,
        )
