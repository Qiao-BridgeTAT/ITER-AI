"""Add V4 turn, dual-state, checkpoint, workspace, and outbox persistence.

Revision ID: 20260829_0007
Revises: 20260826_0006
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision = "20260829_0007"
down_revision = "20260826_0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "agent_turns",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("trip_id", sa.Uuid(), nullable=False),
        sa.Column("request_id", sa.Uuid(), nullable=False),
        sa.Column("idempotency_key", sa.String(length=128), nullable=False),
        sa.Column("request_fingerprint", sa.String(length=64), nullable=False),
        sa.Column("result_fingerprint", sa.String(length=64), nullable=True),
        sa.Column("base_state_version", sa.Integer(), nullable=False),
        sa.Column("committed_state_version", sa.Integer(), nullable=True),
        sa.Column("generation_id", sa.Uuid(), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("generation_mode", sa.String(length=16), nullable=True),
        sa.Column("failure_code", sa.String(length=96), nullable=True),
        sa.Column("decision_audit", sa.JSON(), nullable=False),
        sa.Column(
            "accepted_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("committed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("base_state_version >= 0", name="ck_agent_turns_base_version"),
        sa.CheckConstraint(
            "committed_state_version IS NULL OR committed_state_version = base_state_version + 1",
            name="ck_agent_turns_committed_version",
        ),
        sa.CheckConstraint(
            "status IN ('accepted', 'running', 'committed', 'failed', 'cancelled')",
            name="ck_agent_turns_status",
        ),
        sa.CheckConstraint(
            "generation_mode IS NULL OR generation_mode IN ('qwen', 'fallback')",
            name="ck_agent_turns_generation_mode",
        ),
        sa.ForeignKeyConstraint(["trip_id"], ["trips.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("id", "trip_id", name="uq_agent_turns_id_trip_id"),
        sa.UniqueConstraint("trip_id", "idempotency_key", name="uq_agent_turns_idempotency"),
        sa.UniqueConstraint("trip_id", "request_id", name="uq_agent_turns_request"),
    )
    op.create_index("ix_agent_turns_trip_created", "agent_turns", ["trip_id", "created_at"])
    op.create_index("ix_agent_turns_status_updated", "agent_turns", ["status", "updated_at"])

    with op.batch_alter_table("trip_snapshots") as batch_op:
        batch_op.add_column(
            sa.Column(
                "snapshot_kind",
                sa.String(length=16),
                nullable=False,
                server_default="legacy",
            )
        )
        batch_op.add_column(sa.Column("protocol_version", sa.String(length=24), nullable=True))
        batch_op.add_column(sa.Column("turn_id", sa.Uuid(), nullable=True))
        batch_op.add_column(sa.Column("terminal_status", sa.String(length=24), nullable=True))
        batch_op.add_column(sa.Column("terminal_event", sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column("message_cursor", sa.Uuid(), nullable=True))
        batch_op.add_column(sa.Column("outbox_cursor", sa.String(length=128), nullable=True))
        batch_op.create_check_constraint(
            "ck_trip_snapshots_v4_contract_versions",
            "snapshot_kind != 'v4' OR (schema_version = '4.0.0' AND "
            "protocol_version = 'v4' AND terminal_event IS NOT NULL)",
        )
        batch_op.create_foreign_key(
            "fk_trip_snapshots_turn_same_trip",
            "agent_turns",
            ["turn_id", "trip_id"],
            ["id", "trip_id"],
            ondelete="NO ACTION",
            deferrable=True,
            initially="DEFERRED",
        )

    with op.batch_alter_table("messages") as batch_op:
        batch_op.add_column(sa.Column("turn_id", sa.Uuid(), nullable=True))
        batch_op.add_column(sa.Column("state_version", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("ordinal", sa.Integer(), nullable=False, server_default="0"))
        batch_op.add_column(
            sa.Column("status", sa.String(length=16), nullable=False, server_default="committed")
        )
        batch_op.add_column(sa.Column("protocol_version", sa.String(length=24), nullable=True))
        batch_op.add_column(sa.Column("schema_version", sa.String(length=24), nullable=True))
        batch_op.add_column(sa.Column("generation_mode", sa.String(length=16), nullable=True))
        batch_op.add_column(sa.Column("failure_code", sa.String(length=96), nullable=True))
        batch_op.add_column(sa.Column("content_hash", sa.String(length=64), nullable=True))
        batch_op.create_unique_constraint("uq_messages_id_trip_id", ["id", "trip_id"])
        batch_op.create_unique_constraint(
            "uq_messages_turn_role_ordinal", ["turn_id", "role", "ordinal"]
        )
        batch_op.create_check_constraint(
            "ck_messages_status", "status IN ('accepted', 'committed', 'superseded')"
        )
        batch_op.create_foreign_key(
            "fk_messages_turn_same_trip",
            "agent_turns",
            ["turn_id", "trip_id"],
            ["id", "trip_id"],
            ondelete="CASCADE",
            deferrable=True,
            initially="DEFERRED",
        )

    _create_dual_state_tables()
    _create_turn_audit_tables()
    _create_checkpoint_tables()
    _create_outbox_table()


def _create_dual_state_tables() -> None:
    for table_name, constraint_prefix in (
        ("trip_semantic_state_versions", "semantic_state"),
        ("discovery_runtime_state_versions", "discovery_runtime"),
    ):
        op.create_table(
            table_name,
            sa.Column("id", sa.Uuid(), nullable=False),
            sa.Column("trip_id", sa.Uuid(), nullable=False),
            sa.Column("state_version", sa.Integer(), nullable=False),
            sa.Column("schema_version", sa.String(length=24), nullable=False),
            sa.Column("payload", sa.JSON(), nullable=False),
            sa.Column("content_hash", sa.String(length=64), nullable=False),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                nullable=False,
            ),
            sa.CheckConstraint("state_version >= 0", name=f"ck_{constraint_prefix}_version"),
            sa.CheckConstraint(
                "schema_version = '4.0.0'",
                name=f"ck_{constraint_prefix}_v4_schema_version",
            ),
            sa.ForeignKeyConstraint(
                ["trip_id", "state_version"],
                ["trip_snapshots.trip_id", "trip_snapshots.state_version"],
                name=f"fk_{constraint_prefix}_snapshot_version",
                ondelete="CASCADE",
                deferrable=True,
                initially="DEFERRED",
            ),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("trip_id", "state_version", name=f"uq_{constraint_prefix}_version"),
        )
        op.create_index(
            f"ix_{constraint_prefix}_trip_created", table_name, ["trip_id", "created_at"]
        )


def _create_turn_audit_tables() -> None:
    op.create_table(
        "semantic_operations",
        sa.Column("operation_id", sa.Uuid(), nullable=False),
        sa.Column("trip_id", sa.Uuid(), nullable=False),
        sa.Column("turn_id", sa.Uuid(), nullable=False),
        sa.Column("state_version", sa.Integer(), nullable=False),
        sa.Column("target", sa.String(length=96), nullable=False),
        sa.Column("operation_kind", sa.String(length=24), nullable=False),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("evidence", sa.JSON(), nullable=False),
        sa.Column("supersedes_operation_id", sa.Uuid(), nullable=True),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.CheckConstraint("state_version >= 0", name="ck_semantic_operations_state_version"),
        sa.ForeignKeyConstraint(
            ["turn_id", "trip_id"],
            ["agent_turns.id", "agent_turns.trip_id"],
            name="fk_semantic_operations_turn_same_trip",
            ondelete="CASCADE",
            deferrable=True,
            initially="DEFERRED",
        ),
        sa.PrimaryKeyConstraint("operation_id"),
        sa.UniqueConstraint("trip_id", "operation_id", name="uq_semantic_operations_trip_id"),
    )
    op.create_index(
        "ix_semantic_operations_trip_version",
        "semantic_operations",
        ["trip_id", "state_version"],
    )

    op.create_table(
        "tool_observations",
        sa.Column("observation_id", sa.Uuid(), nullable=False),
        sa.Column("trip_id", sa.Uuid(), nullable=False),
        sa.Column("turn_id", sa.Uuid(), nullable=False),
        sa.Column("tool_name", sa.String(length=96), nullable=False),
        sa.Column("provider", sa.String(length=64), nullable=False),
        sa.Column("request_hash", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column("safe_payload", sa.JSON(), nullable=False),
        sa.Column("source_refs", sa.JSON(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.ForeignKeyConstraint(
            ["turn_id", "trip_id"],
            ["agent_turns.id", "agent_turns.trip_id"],
            name="fk_tool_observations_turn_same_trip",
            ondelete="CASCADE",
            deferrable=True,
            initially="DEFERRED",
        ),
        sa.PrimaryKeyConstraint("observation_id"),
        sa.UniqueConstraint("trip_id", "observation_id", name="uq_tool_observations_trip_id"),
    )
    op.create_index(
        "ix_tool_observations_trip_observed",
        "tool_observations",
        ["trip_id", "observed_at"],
    )
    op.create_index("ix_tool_observations_request_hash", "tool_observations", ["request_hash"])

    op.create_table(
        "pending_interactions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("trip_id", sa.Uuid(), nullable=False),
        sa.Column("turn_id", sa.Uuid(), nullable=False),
        sa.Column("source_message_id", sa.Uuid(), nullable=False),
        sa.Column("section", sa.String(length=64), nullable=False),
        sa.Column("interaction_kind", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("issued_state_version", sa.Integer(), nullable=False),
        sa.Column("closed_state_version", sa.Integer(), nullable=True),
        sa.Column("answer_id", sa.Uuid(), nullable=True),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("closed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "issued_state_version >= 0", name="ck_pending_interactions_issued_version"
        ),
        sa.CheckConstraint(
            "closed_state_version IS NULL OR closed_state_version >= issued_state_version",
            name="ck_pending_interactions_closed_version",
        ),
        sa.CheckConstraint(
            "status IN ('active', 'answered', 'superseded', 'expired')",
            name="ck_pending_interactions_status",
        ),
        sa.ForeignKeyConstraint(
            ["turn_id", "trip_id"],
            ["agent_turns.id", "agent_turns.trip_id"],
            name="fk_pending_interactions_turn_same_trip",
            ondelete="CASCADE",
            deferrable=True,
            initially="DEFERRED",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("id", "trip_id", name="uq_pending_interactions_id_trip_id"),
        sa.UniqueConstraint("trip_id", "answer_id", name="uq_pending_interactions_answer"),
    )
    op.create_index(
        "uq_pending_interactions_one_active",
        "pending_interactions",
        ["trip_id"],
        unique=True,
        sqlite_where=sa.text("status = 'active'"),
        postgresql_where=sa.text("status = 'active'"),
    )
    op.create_index(
        "ix_pending_interactions_trip_created",
        "pending_interactions",
        ["trip_id", "created_at"],
    )


def _create_checkpoint_tables() -> None:
    op.create_table(
        "agent_checkpoints",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("trip_id", sa.Uuid(), nullable=False),
        sa.Column("turn_id", sa.Uuid(), nullable=False),
        sa.Column(
            "checkpoint_namespace", sa.String(length=64), nullable=False, server_default="prepare"
        ),
        sa.Column("checkpoint_id", sa.String(length=128), nullable=False),
        sa.Column("checkpoint_version", sa.String(length=32), nullable=False),
        sa.Column("base_state_version", sa.Integer(), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.CheckConstraint("base_state_version >= 0", name="ck_agent_checkpoints_base_version"),
        sa.CheckConstraint(
            "status IN ('active', 'completed', 'superseded')",
            name="ck_agent_checkpoints_status",
        ),
        sa.ForeignKeyConstraint(
            ["turn_id", "trip_id"],
            ["agent_turns.id", "agent_turns.trip_id"],
            name="fk_agent_checkpoints_turn_same_trip",
            ondelete="CASCADE",
            deferrable=True,
            initially="DEFERRED",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "trip_id",
            "turn_id",
            "checkpoint_namespace",
            "checkpoint_id",
            name="uq_agent_checkpoints_identity",
        ),
    )
    op.create_index(
        "ix_agent_checkpoints_turn_updated", "agent_checkpoints", ["turn_id", "updated_at"]
    )
    op.create_index("ix_agent_checkpoints_expires", "agent_checkpoints", ["expires_at"])

    op.create_table(
        "planner_workspaces",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("trip_id", sa.Uuid(), nullable=False),
        sa.Column("turn_id", sa.Uuid(), nullable=False),
        sa.Column("generation_id", sa.Uuid(), nullable=False),
        sa.Column("confirmed_task_book_id", sa.Uuid(), nullable=False),
        sa.Column("confirmed_task_book_version", sa.Integer(), nullable=False),
        sa.Column("confirmed_task_book_hash", sa.String(length=64), nullable=False),
        sa.Column("base_state_version", sa.Integer(), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("checkpoint_version", sa.String(length=32), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.CheckConstraint("base_state_version >= 0", name="ck_planner_workspaces_base_version"),
        sa.CheckConstraint("revision >= 0", name="ck_planner_workspaces_revision"),
        sa.CheckConstraint(
            "status IN ('working', 'awaiting_user', 'completed', 'stale')",
            name="ck_planner_workspaces_status",
        ),
        sa.ForeignKeyConstraint(
            ["turn_id", "trip_id"],
            ["agent_turns.id", "agent_turns.trip_id"],
            name="fk_planner_workspaces_turn_same_trip",
            ondelete="CASCADE",
            deferrable=True,
            initially="DEFERRED",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "trip_id", "generation_id", "revision", name="uq_planner_workspaces_revision"
        ),
    )
    op.create_index(
        "ix_planner_workspaces_trip_updated",
        "planner_workspaces",
        ["trip_id", "updated_at"],
    )


def _create_outbox_table() -> None:
    op.create_table(
        "outbox_events",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("cursor", sa.String(length=128), nullable=False),
        sa.Column("publication_key", sa.String(length=160), nullable=False),
        sa.Column("trip_id", sa.Uuid(), nullable=False),
        sa.Column("turn_id", sa.Uuid(), nullable=False),
        sa.Column("generation_id", sa.Uuid(), nullable=False),
        sa.Column("message_id", sa.Uuid(), nullable=False),
        sa.Column("committed_state_version", sa.Integer(), nullable=False),
        sa.Column("plan_version_id", sa.Uuid(), nullable=True),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("chunks", sa.JSON(), nullable=False),
        sa.Column("terminal_event", sa.JSON(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("chunking_algorithm_version", sa.String(length=64), nullable=False),
        sa.Column("delivery_status", sa.String(length=16), nullable=False),
        sa.Column("retry_count", sa.Integer(), nullable=False),
        sa.Column("delivered_sequence", sa.Integer(), nullable=False),
        sa.Column(
            "next_attempt_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("leased_by", sa.String(length=96), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_error_code", sa.String(length=96), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("delivered_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("committed_state_version >= 0", name="ck_outbox_state_version"),
        sa.CheckConstraint("retry_count >= 0", name="ck_outbox_retry_count"),
        sa.CheckConstraint("delivered_sequence >= 0", name="ck_outbox_delivered_sequence"),
        sa.CheckConstraint(
            "delivery_status IN ('pending', 'leased', 'delivered', 'dead')",
            name="ck_outbox_delivery_status",
        ),
        sa.ForeignKeyConstraint(
            ["turn_id", "trip_id"],
            ["agent_turns.id", "agent_turns.trip_id"],
            name="fk_outbox_events_turn_same_trip",
            ondelete="CASCADE",
            deferrable=True,
            initially="DEFERRED",
        ),
        sa.ForeignKeyConstraint(
            ["message_id", "trip_id"],
            ["messages.id", "messages.trip_id"],
            name="fk_outbox_events_message_same_trip",
            ondelete="CASCADE",
            deferrable=True,
            initially="DEFERRED",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("cursor", name="uq_outbox_events_cursor"),
        sa.UniqueConstraint("trip_id", "publication_key", name="uq_outbox_events_publication"),
    )
    op.create_index(
        "ix_outbox_dispatch",
        "outbox_events",
        ["delivery_status", "next_attempt_at", "lease_expires_at"],
    )
    op.create_index("ix_outbox_trip_created", "outbox_events", ["trip_id", "created_at"])


def downgrade() -> None:
    op.drop_index("ix_outbox_trip_created", table_name="outbox_events")
    op.drop_index("ix_outbox_dispatch", table_name="outbox_events")
    op.drop_table("outbox_events")

    op.drop_index("ix_planner_workspaces_trip_updated", table_name="planner_workspaces")
    op.drop_table("planner_workspaces")
    op.drop_index("ix_agent_checkpoints_expires", table_name="agent_checkpoints")
    op.drop_index("ix_agent_checkpoints_turn_updated", table_name="agent_checkpoints")
    op.drop_table("agent_checkpoints")

    op.drop_index("ix_pending_interactions_trip_created", table_name="pending_interactions")
    op.drop_index("uq_pending_interactions_one_active", table_name="pending_interactions")
    op.drop_table("pending_interactions")
    op.drop_index("ix_tool_observations_request_hash", table_name="tool_observations")
    op.drop_index("ix_tool_observations_trip_observed", table_name="tool_observations")
    op.drop_table("tool_observations")
    op.drop_index("ix_semantic_operations_trip_version", table_name="semantic_operations")
    op.drop_table("semantic_operations")

    op.drop_index(
        "ix_discovery_runtime_trip_created", table_name="discovery_runtime_state_versions"
    )
    op.drop_table("discovery_runtime_state_versions")
    op.drop_index("ix_semantic_state_trip_created", table_name="trip_semantic_state_versions")
    op.drop_table("trip_semantic_state_versions")

    with op.batch_alter_table("messages") as batch_op:
        batch_op.drop_constraint("fk_messages_turn_same_trip", type_="foreignkey")
        batch_op.drop_constraint("ck_messages_status", type_="check")
        batch_op.drop_constraint("uq_messages_turn_role_ordinal", type_="unique")
        batch_op.drop_constraint("uq_messages_id_trip_id", type_="unique")
        batch_op.drop_column("content_hash")
        batch_op.drop_column("failure_code")
        batch_op.drop_column("generation_mode")
        batch_op.drop_column("schema_version")
        batch_op.drop_column("protocol_version")
        batch_op.drop_column("status")
        batch_op.drop_column("ordinal")
        batch_op.drop_column("state_version")
        batch_op.drop_column("turn_id")

    with op.batch_alter_table("trip_snapshots") as batch_op:
        batch_op.drop_constraint("fk_trip_snapshots_turn_same_trip", type_="foreignkey")
        batch_op.drop_constraint("ck_trip_snapshots_v4_contract_versions", type_="check")
        batch_op.drop_column("outbox_cursor")
        batch_op.drop_column("message_cursor")
        batch_op.drop_column("terminal_event")
        batch_op.drop_column("terminal_status")
        batch_op.drop_column("turn_id")
        batch_op.drop_column("protocol_version")
        batch_op.drop_column("snapshot_kind")

    op.drop_index("ix_agent_turns_status_updated", table_name="agent_turns")
    op.drop_index("ix_agent_turns_trip_created", table_name="agent_turns")
    op.drop_table("agent_turns")
