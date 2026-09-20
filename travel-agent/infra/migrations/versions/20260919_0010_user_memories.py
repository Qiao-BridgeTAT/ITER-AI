"""Explicit cross-trip user memory and original feedback provenance."""

import sqlalchemy as sa
from alembic import op

revision = "20260919_0010"
down_revision = "20260919_0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "user_memories",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "user_id", sa.Uuid(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("trip_id", sa.Uuid(), sa.ForeignKey("trips.id", ondelete="CASCADE")),
        sa.Column("source_message_id", sa.Uuid(), sa.ForeignKey("messages.id", ondelete="CASCADE")),
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.UniqueConstraint("user_id", "source_message_id", "kind", name="uq_user_memory_source"),
        sa.CheckConstraint("kind IN ('preference', 'feedback')", name="ck_user_memory_kind"),
    )


def downgrade() -> None:
    op.drop_table("user_memories")
