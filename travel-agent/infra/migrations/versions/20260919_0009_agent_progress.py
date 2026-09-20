"""Persist public Agent process history separately from committed result outbox."""

import sqlalchemy as sa
from alembic import op

revision = "20260919_0009"
down_revision = "20260830_0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "agent_progress",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("trip_id", sa.Uuid(), nullable=False),
        sa.Column("turn_id", sa.Uuid(), nullable=False),
        sa.Column("generation_id", sa.Uuid(), nullable=False),
        sa.Column("progress_index", sa.Integer(), nullable=False),
        sa.Column("source", sa.String(16), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("emitted_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["turn_id", "trip_id"], ["agent_turns.id", "agent_turns.trip_id"], ondelete="CASCADE"
        ),
        sa.UniqueConstraint("generation_id", "progress_index", name="uq_agent_progress_index"),
    )
    op.create_index("ix_agent_progress_trip", "agent_progress", ["trip_id", "emitted_at"])


def downgrade() -> None:
    op.drop_table("agent_progress")
