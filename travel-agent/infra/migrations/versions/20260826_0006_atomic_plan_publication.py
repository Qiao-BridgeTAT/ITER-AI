"""Add idempotent plan publication keys and confirmed-version immutability.

Revision ID: 20260826_0006
Revises: 20260825_0005
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision = "20260826_0006"
down_revision = "20260825_0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("trip_versions") as batch_op:
        batch_op.add_column(sa.Column("publication_key", sa.String(length=128)))
        batch_op.add_column(sa.Column("generation_id", sa.Uuid()))
        batch_op.create_unique_constraint(
            "uq_trip_versions_publication_key", ["trip_id", "publication_key"]
        )
        batch_op.create_unique_constraint(
            "uq_trip_versions_generation_id", ["trip_id", "generation_id"]
        )
    _create_confirmed_immutability_guard()


def downgrade() -> None:
    _drop_confirmed_immutability_guard()
    with op.batch_alter_table("trip_versions") as batch_op:
        batch_op.drop_constraint("uq_trip_versions_generation_id", type_="unique")
        batch_op.drop_constraint("uq_trip_versions_publication_key", type_="unique")
        batch_op.drop_column("generation_id")
        batch_op.drop_column("publication_key")


def _create_confirmed_immutability_guard() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute(
            """
            CREATE FUNCTION reject_confirmed_trip_version_update()
            RETURNS trigger AS $$
            BEGIN
                IF OLD.status = 'confirmed' THEN
                    RAISE EXCEPTION 'confirmed trip versions are immutable';
                END IF;
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql
            """
        )
        op.execute(
            """
            CREATE TRIGGER trg_trip_versions_confirmed_immutable
            BEFORE UPDATE ON trip_versions
            FOR EACH ROW EXECUTE FUNCTION reject_confirmed_trip_version_update()
            """
        )
    elif bind.dialect.name == "sqlite":
        op.execute(
            """
            CREATE TRIGGER trg_trip_versions_confirmed_immutable
            BEFORE UPDATE ON trip_versions
            WHEN OLD.status = 'confirmed'
            BEGIN
                SELECT RAISE(ABORT, 'confirmed trip versions are immutable');
            END
            """
        )


def _drop_confirmed_immutability_guard() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("DROP TRIGGER IF EXISTS trg_trip_versions_confirmed_immutable ON trip_versions")
        op.execute("DROP FUNCTION IF EXISTS reject_confirmed_trip_version_update()")
    elif bind.dialect.name == "sqlite":
        op.execute("DROP TRIGGER IF EXISTS trg_trip_versions_confirmed_immutable")
