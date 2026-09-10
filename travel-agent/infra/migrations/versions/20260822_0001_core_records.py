"""Create users, trips, stable state, versions, messages, and planning runs.

Revision ID: 20260822_0001
Revises: None
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260822_0001"
down_revision = None
branch_labels = None
depends_on = None

TRIP_PHASES = (
    "cold_start",
    "city_selection",
    "city_brief",
    "trip_setup",
    "interest_selection",
    "attraction_selection",
    "dining_selection",
    "lodging_selection",
    "task_reflection",
    "planning",
    "draft_ready",
    "revising",
    "confirmed",
)


def _in_values(values: tuple[str, ...]) -> str:
    return ", ".join(repr(value) for value in values)


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.CheckConstraint("status IN ('active', 'disabled')", name="ck_users_status"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "auth_identities",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("provider", sa.String(length=24), nullable=False),
        sa.Column("lookup_hash", sa.String(length=128), nullable=False),
        sa.Column("encrypted_identifier", sa.Text(), nullable=False),
        sa.Column("verified_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.CheckConstraint("provider IN ('phone')", name="ck_auth_identities_provider"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("provider", "lookup_hash", name="uq_auth_identities_provider_lookup"),
    )
    op.create_index("ix_auth_identities_user_id", "auth_identities", ["user_id"])
    op.create_table(
        "user_preferences",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("preference_key", sa.String(length=96), nullable=False),
        sa.Column("value", sa.JSON(), nullable=False),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("confidence", sa.String(length=16), nullable=False),
        sa.Column("active", sa.Boolean(), nullable=False),
        sa.Column("last_confirmed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.CheckConstraint(
            "source IN ('cold_start', 'user_confirmed_inference')",
            name="ck_user_preferences_source",
        ),
        sa.CheckConstraint(
            "confidence IN ('high', 'medium', 'low')", name="ck_user_preferences_confidence"
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_user_preferences_user_active", "user_preferences", ["user_id", "active"])
    op.create_table(
        "trips",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("owner_user_id", sa.Uuid(), nullable=False),
        sa.Column("phase", sa.String(length=40), nullable=False),
        sa.Column("state_version", sa.Integer(), nullable=False),
        sa.Column("schema_version", sa.String(length=24), nullable=False),
        sa.Column("title", sa.String(length=160), nullable=True),
        sa.Column("current_plan_version_id", sa.Uuid(), nullable=True),
        sa.Column("base_confirmed_version_id", sa.Uuid(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.CheckConstraint("state_version >= 0", name="ck_trips_state_version_nonnegative"),
        sa.CheckConstraint(f"phase IN ({_in_values(TRIP_PHASES)})", name="ck_trips_phase"),
        sa.ForeignKeyConstraint(["owner_user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_trips_owner_updated", "trips", ["owner_user_id", "updated_at"])
    op.create_table(
        "trip_versions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("trip_id", sa.Uuid(), nullable=False),
        sa.Column("parent_version_id", sa.Uuid(), nullable=True),
        sa.Column("version_number", sa.Integer(), nullable=False),
        sa.Column("state_version", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("snapshot", sa.JSON(), nullable=False),
        sa.Column("schema_version", sa.String(length=24), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("confirmed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("version_number >= 1", name="ck_trip_versions_number_positive"),
        sa.CheckConstraint("state_version >= 0", name="ck_trip_versions_state_nonnegative"),
        sa.CheckConstraint("status IN ('draft', 'confirmed')", name="ck_trip_versions_status"),
        sa.ForeignKeyConstraint(
            ["parent_version_id"],
            ["trip_versions.id"],
            ondelete="NO ACTION",
            deferrable=True,
            initially="DEFERRED",
        ),
        sa.ForeignKeyConstraint(["trip_id"], ["trips.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("trip_id", "state_version", name="uq_trip_versions_state_version"),
        sa.UniqueConstraint("trip_id", "version_number", name="uq_trip_versions_number"),
    )
    op.create_index("ix_trip_versions_trip_created", "trip_versions", ["trip_id", "created_at"])
    with op.batch_alter_table("trips") as batch_op:
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
    op.create_table(
        "trip_snapshots",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("trip_id", sa.Uuid(), nullable=False),
        sa.Column("plan_version_id", sa.Uuid(), nullable=True),
        sa.Column("state_version", sa.Integer(), nullable=False),
        sa.Column("schema_version", sa.String(length=24), nullable=False),
        sa.Column("snapshot", sa.JSON(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.CheckConstraint("state_version >= 0", name="ck_trip_snapshots_state_nonnegative"),
        sa.ForeignKeyConstraint(["plan_version_id"], ["trip_versions.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["trip_id"], ["trips.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("trip_id", "state_version", name="uq_trip_snapshots_state_version"),
    )
    op.create_index("ix_trip_snapshots_trip_created", "trip_snapshots", ["trip_id", "created_at"])
    op.create_table(
        "messages",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("trip_id", sa.Uuid(), nullable=False),
        sa.Column("client_message_id", sa.Uuid(), nullable=True),
        sa.Column("role", sa.String(length=16), nullable=False),
        sa.Column("message_type", sa.String(length=64), nullable=False),
        sa.Column("text", sa.Text(), nullable=True),
        sa.Column("attachments", sa.JSON(), nullable=False),
        sa.Column("message_metadata", sa.JSON(), nullable=False),
        sa.Column("request_id", sa.Uuid(), nullable=True),
        sa.Column("generation_id", sa.Uuid(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.CheckConstraint(
            "role IN ('user', 'assistant', 'system', 'tool')", name="ck_messages_role"
        ),
        sa.ForeignKeyConstraint(["trip_id"], ["trips.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("trip_id", "client_message_id", name="uq_messages_client_id"),
    )
    op.create_index("ix_messages_generation", "messages", ["generation_id"])
    op.create_index("ix_messages_trip_created", "messages", ["trip_id", "created_at"])
    op.create_table(
        "planning_runs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("trip_id", sa.Uuid(), nullable=False),
        sa.Column("plan_version_id", sa.Uuid(), nullable=True),
        sa.Column("generation_id", sa.Uuid(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("input_summary", sa.JSON(), nullable=False),
        sa.Column("result_summary", sa.JSON(), nullable=True),
        sa.Column("config_versions", sa.JSON(), nullable=False),
        sa.Column("failure_code", sa.String(length=96), nullable=True),
        sa.Column(
            "started_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')",
            name="ck_planning_runs_status",
        ),
        sa.ForeignKeyConstraint(["plan_version_id"], ["trip_versions.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["trip_id"], ["trips.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("trip_id", "generation_id", name="uq_planning_runs_generation"),
    )
    op.create_index("ix_planning_runs_trip_started", "planning_runs", ["trip_id", "started_at"])


def downgrade() -> None:
    op.drop_index("ix_planning_runs_trip_started", table_name="planning_runs")
    op.drop_table("planning_runs")
    op.drop_index("ix_messages_trip_created", table_name="messages")
    op.drop_index("ix_messages_generation", table_name="messages")
    op.drop_table("messages")
    op.drop_index("ix_trip_snapshots_trip_created", table_name="trip_snapshots")
    op.drop_table("trip_snapshots")
    with op.batch_alter_table("trips") as batch_op:
        batch_op.drop_constraint("fk_trips_base_confirmed_version", type_="foreignkey")
        batch_op.drop_constraint("fk_trips_current_plan_version", type_="foreignkey")
    op.drop_index("ix_trip_versions_trip_created", table_name="trip_versions")
    op.drop_table("trip_versions")
    op.drop_index("ix_trips_owner_updated", table_name="trips")
    op.drop_table("trips")
    op.drop_index("ix_user_preferences_user_active", table_name="user_preferences")
    op.drop_table("user_preferences")
    op.drop_index("ix_auth_identities_user_id", table_name="auth_identities")
    op.drop_table("auth_identities")
    op.drop_table("users")
