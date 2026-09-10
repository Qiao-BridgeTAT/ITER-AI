"""Add an optional user nickname.

Revision ID: 20260825_0005
Revises: 20260822_0004
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260825_0005"
down_revision = "20260822_0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("users", sa.Column("nickname", sa.String(length=20), nullable=True))


def downgrade() -> None:
    op.drop_column("users", "nickname")
