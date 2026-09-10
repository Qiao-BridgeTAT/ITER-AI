"""Make dated and undated ticket-offer observations independently unique.

Revision ID: 20260822_0004
Revises: 20260822_0003
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260822_0004"
down_revision = "20260822_0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # The previous nullable unique constraint admitted duplicate undated observations.
    # Keep the oldest audit record before installing the corrected indexes.
    op.execute(
        sa.text(
            """
            DELETE FROM ticket_offers
            WHERE id IN (
                SELECT id
                FROM (
                    SELECT
                        id,
                        ROW_NUMBER() OVER (
                            PARTITION BY provider, source_offer_id, fetched_at
                            ORDER BY created_at, id
                        ) AS duplicate_rank
                    FROM ticket_offers
                    WHERE visit_date IS NULL
                ) AS duplicate_observations
                WHERE duplicate_rank > 1
            )
            """
        )
    )
    with op.batch_alter_table("ticket_offers") as batch_op:
        batch_op.drop_constraint("uq_ticket_offers_observation", type_="unique")
    op.create_index(
        "uq_ticket_offers_dated_observation",
        "ticket_offers",
        ["provider", "source_offer_id", "visit_date", "fetched_at"],
        unique=True,
        postgresql_where=sa.text("visit_date IS NOT NULL"),
        sqlite_where=sa.text("visit_date IS NOT NULL"),
    )
    op.create_index(
        "uq_ticket_offers_undated_observation",
        "ticket_offers",
        ["provider", "source_offer_id", "fetched_at"],
        unique=True,
        postgresql_where=sa.text("visit_date IS NULL"),
        sqlite_where=sa.text("visit_date IS NULL"),
    )


def downgrade() -> None:
    op.drop_index("uq_ticket_offers_undated_observation", table_name="ticket_offers")
    op.drop_index("uq_ticket_offers_dated_observation", table_name="ticket_offers")
    with op.batch_alter_table("ticket_offers") as batch_op:
        batch_op.create_unique_constraint(
            "uq_ticket_offers_observation",
            ["provider", "source_offer_id", "visit_date", "fetched_at"],
        )
