"""Enforce that every version reference stays inside its owning trip.

Revision ID: 20260822_0002
Revises: 20260822_0001
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision = "20260822_0002"
down_revision = "20260822_0001"
branch_labels = None
depends_on = None

NAMING_CONVENTION = {
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
}


def _foreign_key_name(
    table_name: str,
    constrained_columns: Sequence[str],
    unnamed_fallback: str,
) -> str:
    expected = list(constrained_columns)
    for foreign_key in sa.inspect(op.get_bind()).get_foreign_keys(table_name):
        if foreign_key["constrained_columns"] == expected:
            return foreign_key["name"] or unnamed_fallback
    raise RuntimeError(f"Expected foreign key on {table_name}({', '.join(expected)}) was not found")


def _clear_cross_trip_references() -> None:
    op.execute(
        sa.text(
            """
            UPDATE trips
            SET current_plan_version_id = NULL
            WHERE current_plan_version_id IS NOT NULL
              AND NOT EXISTS (
                  SELECT 1
                  FROM trip_versions AS version
                  WHERE version.id = trips.current_plan_version_id
                    AND version.trip_id = trips.id
              )
            """
        )
    )
    op.execute(
        sa.text(
            """
            UPDATE trip_snapshots AS snapshot
            SET plan_version_id = NULL
            WHERE plan_version_id IS NOT NULL
              AND NOT EXISTS (
                  SELECT 1
                  FROM trip_versions AS version
                  WHERE version.id = snapshot.plan_version_id
                    AND version.trip_id = snapshot.trip_id
              )
            """
        )
    )
    op.execute(
        sa.text(
            """
            UPDATE planning_runs AS run
            SET plan_version_id = NULL
            WHERE plan_version_id IS NOT NULL
              AND NOT EXISTS (
                  SELECT 1
                  FROM trip_versions AS version
                  WHERE version.id = run.plan_version_id
                    AND version.trip_id = run.trip_id
              )
            """
        )
    )
    op.execute(
        sa.text(
            """
            UPDATE trips
            SET base_confirmed_version_id = NULL
            WHERE base_confirmed_version_id IS NOT NULL
              AND NOT EXISTS (
                  SELECT 1
                  FROM trip_versions AS version
                  WHERE version.id = trips.base_confirmed_version_id
                    AND version.trip_id = trips.id
              )
            """
        )
    )
    op.execute(
        sa.text(
            """
            UPDATE trip_versions AS child
            SET parent_version_id = NULL
            WHERE parent_version_id IS NOT NULL
              AND NOT EXISTS (
                  SELECT 1
                  FROM trip_versions AS parent
                  WHERE parent.id = child.parent_version_id
                    AND parent.trip_id = child.trip_id
              )
            """
        )
    )


def upgrade() -> None:
    _clear_cross_trip_references()

    parent_foreign_key = _foreign_key_name(
        "trip_versions",
        ["parent_version_id"],
        "fk_trip_versions_parent_version_id_trip_versions",
    )
    with op.batch_alter_table("trip_versions", naming_convention=NAMING_CONVENTION) as batch_op:
        batch_op.drop_constraint(parent_foreign_key, type_="foreignkey")
        batch_op.create_unique_constraint("uq_trip_versions_id_trip_id", ["id", "trip_id"])
        batch_op.create_foreign_key(
            "fk_trip_versions_parent_same_trip",
            "trip_versions",
            ["parent_version_id", "trip_id"],
            ["id", "trip_id"],
            ondelete="NO ACTION",
            deferrable=True,
            initially="DEFERRED",
        )

    with op.batch_alter_table("trips") as batch_op:
        batch_op.drop_constraint("fk_trips_current_plan_version", type_="foreignkey")
        batch_op.drop_constraint("fk_trips_base_confirmed_version", type_="foreignkey")
        batch_op.create_foreign_key(
            "fk_trips_current_plan_version_same_trip",
            "trip_versions",
            ["current_plan_version_id", "id"],
            ["id", "trip_id"],
            ondelete="NO ACTION",
            deferrable=True,
            initially="DEFERRED",
        )
        batch_op.create_foreign_key(
            "fk_trips_base_confirmed_version_same_trip",
            "trip_versions",
            ["base_confirmed_version_id", "id"],
            ["id", "trip_id"],
            ondelete="NO ACTION",
            deferrable=True,
            initially="DEFERRED",
        )

    snapshot_foreign_key = _foreign_key_name(
        "trip_snapshots",
        ["plan_version_id"],
        "fk_trip_snapshots_plan_version_id_trip_versions",
    )
    with op.batch_alter_table("trip_snapshots", naming_convention=NAMING_CONVENTION) as batch_op:
        batch_op.drop_constraint(snapshot_foreign_key, type_="foreignkey")
        batch_op.create_foreign_key(
            "fk_trip_snapshots_plan_version_same_trip",
            "trip_versions",
            ["plan_version_id", "trip_id"],
            ["id", "trip_id"],
            ondelete="NO ACTION",
            deferrable=True,
            initially="DEFERRED",
        )

    planning_run_foreign_key = _foreign_key_name(
        "planning_runs",
        ["plan_version_id"],
        "fk_planning_runs_plan_version_id_trip_versions",
    )
    with op.batch_alter_table("planning_runs", naming_convention=NAMING_CONVENTION) as batch_op:
        batch_op.drop_constraint(planning_run_foreign_key, type_="foreignkey")
        batch_op.create_foreign_key(
            "fk_planning_runs_plan_version_same_trip",
            "trip_versions",
            ["plan_version_id", "trip_id"],
            ["id", "trip_id"],
            ondelete="NO ACTION",
            deferrable=True,
            initially="DEFERRED",
        )


def downgrade() -> None:
    with op.batch_alter_table("planning_runs") as batch_op:
        batch_op.drop_constraint("fk_planning_runs_plan_version_same_trip", type_="foreignkey")
        batch_op.create_foreign_key(
            "fk_planning_runs_plan_version",
            "trip_versions",
            ["plan_version_id"],
            ["id"],
            ondelete="SET NULL",
        )

    with op.batch_alter_table("trip_snapshots") as batch_op:
        batch_op.drop_constraint("fk_trip_snapshots_plan_version_same_trip", type_="foreignkey")
        batch_op.create_foreign_key(
            "fk_trip_snapshots_plan_version",
            "trip_versions",
            ["plan_version_id"],
            ["id"],
            ondelete="SET NULL",
        )

    with op.batch_alter_table("trips") as batch_op:
        batch_op.drop_constraint("fk_trips_base_confirmed_version_same_trip", type_="foreignkey")
        batch_op.drop_constraint("fk_trips_current_plan_version_same_trip", type_="foreignkey")
        batch_op.create_foreign_key(
            "fk_trips_current_plan_version",
            "trip_versions",
            ["current_plan_version_id"],
            ["id"],
            ondelete="SET NULL",
            deferrable=True,
            initially="DEFERRED",
        )
        batch_op.create_foreign_key(
            "fk_trips_base_confirmed_version",
            "trip_versions",
            ["base_confirmed_version_id"],
            ["id"],
            ondelete="SET NULL",
            deferrable=True,
            initially="DEFERRED",
        )

    with op.batch_alter_table("trip_versions") as batch_op:
        batch_op.drop_constraint("fk_trip_versions_parent_same_trip", type_="foreignkey")
        batch_op.drop_constraint("uq_trip_versions_id_trip_id", type_="unique")
        batch_op.create_foreign_key(
            "fk_trip_versions_parent_version",
            "trip_versions",
            ["parent_version_id"],
            ["id"],
            ondelete="NO ACTION",
            deferrable=True,
            initially="DEFERRED",
        )
