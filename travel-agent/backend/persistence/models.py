"""Core durable tables introduced by M0-02."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    func,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.persistence.database import Base

ID = Uuid(as_uuid=True)


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


class User(TimestampMixin, Base):
    __tablename__ = "users"
    __table_args__ = (CheckConstraint("status IN ('active', 'disabled')", name="ck_users_status"),)

    id: Mapped[UUID] = mapped_column(ID, primary_key=True, default=uuid4)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="active")
    nickname: Mapped[str | None] = mapped_column(String(20))

    identities: Mapped[list[AuthIdentity]] = relationship(
        back_populates="user", cascade="all, delete-orphan", passive_deletes=True
    )
    preferences: Mapped[list[UserPreference]] = relationship(
        back_populates="user", cascade="all, delete-orphan", passive_deletes=True
    )
    trips: Mapped[list[Trip]] = relationship(
        back_populates="owner", cascade="all, delete-orphan", passive_deletes=True
    )


class AuthIdentity(TimestampMixin, Base):
    __tablename__ = "auth_identities"
    __table_args__ = (
        UniqueConstraint("provider", "lookup_hash", name="uq_auth_identities_provider_lookup"),
        CheckConstraint("provider IN ('phone')", name="ck_auth_identities_provider"),
        Index("ix_auth_identities_user_id", "user_id"),
    )

    id: Mapped[UUID] = mapped_column(ID, primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(
        ID, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    provider: Mapped[str] = mapped_column(String(24), nullable=False)
    lookup_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    encrypted_identifier: Mapped[str] = mapped_column(Text, nullable=False)
    verified_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    user: Mapped[User] = relationship(back_populates="identities")


class UserPreference(TimestampMixin, Base):
    __tablename__ = "user_preferences"
    __table_args__ = (
        CheckConstraint(
            "source IN ('cold_start', 'user_confirmed_inference')",
            name="ck_user_preferences_source",
        ),
        CheckConstraint(
            "confidence IN ('high', 'medium', 'low')", name="ck_user_preferences_confidence"
        ),
        Index("ix_user_preferences_user_active", "user_id", "active"),
    )

    id: Mapped[UUID] = mapped_column(ID, primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(
        ID, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    preference_key: Mapped[str] = mapped_column(String(96), nullable=False)
    value: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    confidence: Mapped[str] = mapped_column(String(16), nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    last_confirmed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    user: Mapped[User] = relationship(back_populates="preferences")


class Trip(TimestampMixin, Base):
    __tablename__ = "trips"
    __table_args__ = (
        ForeignKeyConstraint(
            ["current_plan_version_id", "id"],
            ["trip_versions.id", "trip_versions.trip_id"],
            name="fk_trips_current_plan_version_same_trip",
            ondelete="NO ACTION",
            use_alter=True,
            deferrable=True,
            initially="DEFERRED",
        ),
        ForeignKeyConstraint(
            ["base_confirmed_version_id", "id"],
            ["trip_versions.id", "trip_versions.trip_id"],
            name="fk_trips_base_confirmed_version_same_trip",
            ondelete="NO ACTION",
            use_alter=True,
            deferrable=True,
            initially="DEFERRED",
        ),
        UniqueConstraint("id", "owner_user_id", name="uq_trips_id_owner"),
        CheckConstraint("state_version >= 0", name="ck_trips_state_version_nonnegative"),
        CheckConstraint(
            "phase IN ('cold_start', 'city_selection', 'city_brief', 'trip_setup', "
            "'interest_selection', 'attraction_selection', 'dining_selection', "
            "'lodging_selection', 'task_reflection', 'planning', 'draft_ready', "
            "'revising', 'confirmed')",
            name="ck_trips_phase",
        ),
        Index("ix_trips_owner_updated", "owner_user_id", "updated_at"),
    )

    id: Mapped[UUID] = mapped_column(ID, primary_key=True, default=uuid4)
    owner_user_id: Mapped[UUID] = mapped_column(
        ID, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    phase: Mapped[str] = mapped_column(String(40), nullable=False)
    state_version: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    schema_version: Mapped[str] = mapped_column(String(24), nullable=False)
    title: Mapped[str | None] = mapped_column(String(160))
    current_plan_version_id: Mapped[UUID | None] = mapped_column(ID)
    base_confirmed_version_id: Mapped[UUID | None] = mapped_column(ID)

    owner: Mapped[User] = relationship(back_populates="trips")
    snapshots: Mapped[list[TripSnapshot]] = relationship(
        back_populates="trip", cascade="all, delete-orphan", passive_deletes=True
    )
    versions: Mapped[list[TripVersion]] = relationship(
        back_populates="trip",
        cascade="all, delete-orphan",
        passive_deletes=True,
        foreign_keys="TripVersion.trip_id",
    )
    messages: Mapped[list[Message]] = relationship(
        back_populates="trip", cascade="all, delete-orphan", passive_deletes=True
    )
    planning_runs: Mapped[list[PlanningRun]] = relationship(
        back_populates="trip", cascade="all, delete-orphan", passive_deletes=True
    )


class TripVersion(Base):
    __tablename__ = "trip_versions"
    __table_args__ = (
        ForeignKeyConstraint(
            ["parent_version_id", "trip_id"],
            ["trip_versions.id", "trip_versions.trip_id"],
            name="fk_trip_versions_parent_same_trip",
            ondelete="NO ACTION",
            deferrable=True,
            initially="DEFERRED",
        ),
        UniqueConstraint("id", "trip_id", name="uq_trip_versions_id_trip_id"),
        UniqueConstraint("trip_id", "version_number", name="uq_trip_versions_number"),
        UniqueConstraint("trip_id", "state_version", name="uq_trip_versions_state_version"),
        UniqueConstraint("trip_id", "publication_key", name="uq_trip_versions_publication_key"),
        UniqueConstraint("trip_id", "generation_id", name="uq_trip_versions_generation_id"),
        CheckConstraint("version_number >= 1", name="ck_trip_versions_number_positive"),
        CheckConstraint("state_version >= 0", name="ck_trip_versions_state_nonnegative"),
        CheckConstraint("status IN ('draft', 'confirmed')", name="ck_trip_versions_status"),
        Index("ix_trip_versions_trip_created", "trip_id", "created_at"),
    )

    id: Mapped[UUID] = mapped_column(ID, primary_key=True, default=uuid4)
    trip_id: Mapped[UUID] = mapped_column(
        ID, ForeignKey("trips.id", ondelete="CASCADE"), nullable=False
    )
    parent_version_id: Mapped[UUID | None] = mapped_column(ID)
    publication_key: Mapped[str | None] = mapped_column(String(128))
    generation_id: Mapped[UUID | None] = mapped_column(ID)
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    state_version: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    snapshot: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    schema_version: Mapped[str] = mapped_column(String(24), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    confirmed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    trip: Mapped[Trip] = relationship(back_populates="versions", foreign_keys=[trip_id])


class TripSnapshot(Base):
    __tablename__ = "trip_snapshots"
    __table_args__ = (
        ForeignKeyConstraint(
            ["turn_id", "trip_id"],
            ["agent_turns.id", "agent_turns.trip_id"],
            name="fk_trip_snapshots_turn_same_trip",
            ondelete="NO ACTION",
            deferrable=True,
            initially="DEFERRED",
        ),
        ForeignKeyConstraint(
            ["plan_version_id", "trip_id"],
            ["trip_versions.id", "trip_versions.trip_id"],
            name="fk_trip_snapshots_plan_version_same_trip",
            ondelete="NO ACTION",
            deferrable=True,
            initially="DEFERRED",
        ),
        UniqueConstraint("trip_id", "state_version", name="uq_trip_snapshots_state_version"),
        CheckConstraint("state_version >= 0", name="ck_trip_snapshots_state_nonnegative"),
        CheckConstraint(
            "snapshot_kind != 'v4' OR (schema_version = '4.0.0' AND "
            "protocol_version = 'v4' AND terminal_event IS NOT NULL)",
            name="ck_trip_snapshots_v4_contract_versions",
        ),
        Index("ix_trip_snapshots_trip_created", "trip_id", "created_at"),
    )

    id: Mapped[UUID] = mapped_column(ID, primary_key=True, default=uuid4)
    trip_id: Mapped[UUID] = mapped_column(
        ID, ForeignKey("trips.id", ondelete="CASCADE"), nullable=False
    )
    plan_version_id: Mapped[UUID | None] = mapped_column(ID)
    state_version: Mapped[int] = mapped_column(Integer, nullable=False)
    schema_version: Mapped[str] = mapped_column(String(24), nullable=False)
    snapshot: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    snapshot_kind: Mapped[str] = mapped_column(String(16), nullable=False, default="legacy")
    protocol_version: Mapped[str | None] = mapped_column(String(24))
    turn_id: Mapped[UUID | None] = mapped_column(ID)
    terminal_status: Mapped[str | None] = mapped_column(String(24))
    terminal_event: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    message_cursor: Mapped[UUID | None] = mapped_column(ID)
    outbox_cursor: Mapped[str | None] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    trip: Mapped[Trip] = relationship(back_populates="snapshots")


class Message(Base):
    __tablename__ = "messages"
    __table_args__ = (
        ForeignKeyConstraint(
            ["turn_id", "trip_id"],
            ["agent_turns.id", "agent_turns.trip_id"],
            name="fk_messages_turn_same_trip",
            ondelete="CASCADE",
            deferrable=True,
            initially="DEFERRED",
        ),
        UniqueConstraint("id", "trip_id", name="uq_messages_id_trip_id"),
        UniqueConstraint("trip_id", "client_message_id", name="uq_messages_client_id"),
        UniqueConstraint("turn_id", "role", "ordinal", name="uq_messages_turn_role_ordinal"),
        CheckConstraint("role IN ('user', 'assistant', 'system', 'tool')", name="ck_messages_role"),
        CheckConstraint(
            "status IN ('accepted', 'committed', 'superseded')", name="ck_messages_status"
        ),
        Index("ix_messages_trip_created", "trip_id", "created_at"),
        Index("ix_messages_generation", "generation_id"),
    )

    id: Mapped[UUID] = mapped_column(ID, primary_key=True, default=uuid4)
    trip_id: Mapped[UUID] = mapped_column(
        ID, ForeignKey("trips.id", ondelete="CASCADE"), nullable=False
    )
    client_message_id: Mapped[UUID | None] = mapped_column(ID)
    turn_id: Mapped[UUID | None] = mapped_column(ID)
    state_version: Mapped[int | None] = mapped_column(Integer)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="committed")
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    message_type: Mapped[str] = mapped_column(String(64), nullable=False, default="text")
    text: Mapped[str | None] = mapped_column(Text)
    attachments: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False, default=list)
    message_metadata: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    request_id: Mapped[UUID | None] = mapped_column(ID)
    generation_id: Mapped[UUID | None] = mapped_column(ID)
    protocol_version: Mapped[str | None] = mapped_column(String(24))
    schema_version: Mapped[str | None] = mapped_column(String(24))
    generation_mode: Mapped[str | None] = mapped_column(String(16))
    failure_code: Mapped[str | None] = mapped_column(String(96))
    content_hash: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    trip: Mapped[Trip] = relationship(back_populates="messages")


class PlanningRun(Base):
    __tablename__ = "planning_runs"
    __table_args__ = (
        ForeignKeyConstraint(
            ["plan_version_id", "trip_id"],
            ["trip_versions.id", "trip_versions.trip_id"],
            name="fk_planning_runs_plan_version_same_trip",
            ondelete="NO ACTION",
            deferrable=True,
            initially="DEFERRED",
        ),
        UniqueConstraint("trip_id", "generation_id", name="uq_planning_runs_generation"),
        CheckConstraint(
            "status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')",
            name="ck_planning_runs_status",
        ),
        Index("ix_planning_runs_trip_started", "trip_id", "started_at"),
    )

    id: Mapped[UUID] = mapped_column(ID, primary_key=True, default=uuid4)
    trip_id: Mapped[UUID] = mapped_column(
        ID, ForeignKey("trips.id", ondelete="CASCADE"), nullable=False
    )
    plan_version_id: Mapped[UUID | None] = mapped_column(ID)
    generation_id: Mapped[UUID] = mapped_column(ID, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    input_summary: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    result_summary: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    config_versions: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    failure_code: Mapped[str | None] = mapped_column(String(96))
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    trip: Mapped[Trip] = relationship(back_populates="planning_runs")


class AgentTurn(TimestampMixin, Base):
    """Durable admission and terminal record for one idempotent user turn."""

    __tablename__ = "agent_turns"
    __table_args__ = (
        UniqueConstraint("id", "trip_id", name="uq_agent_turns_id_trip_id"),
        UniqueConstraint("trip_id", "idempotency_key", name="uq_agent_turns_idempotency"),
        UniqueConstraint("trip_id", "request_id", name="uq_agent_turns_request"),
        CheckConstraint("base_state_version >= 0", name="ck_agent_turns_base_version"),
        CheckConstraint(
            "committed_state_version IS NULL OR committed_state_version = base_state_version + 1",
            name="ck_agent_turns_committed_version",
        ),
        CheckConstraint(
            "status IN ('accepted', 'running', 'committed', 'failed', 'cancelled')",
            name="ck_agent_turns_status",
        ),
        CheckConstraint(
            "generation_mode IS NULL OR generation_mode IN ('qwen', 'fallback')",
            name="ck_agent_turns_generation_mode",
        ),
        Index("ix_agent_turns_trip_created", "trip_id", "created_at"),
        Index("ix_agent_turns_status_updated", "status", "updated_at"),
    )

    id: Mapped[UUID] = mapped_column(ID, primary_key=True, default=uuid4)
    trip_id: Mapped[UUID] = mapped_column(
        ID, ForeignKey("trips.id", ondelete="CASCADE"), nullable=False
    )
    request_id: Mapped[UUID] = mapped_column(ID, nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(128), nullable=False)
    request_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    result_fingerprint: Mapped[str | None] = mapped_column(String(64))
    base_state_version: Mapped[int] = mapped_column(Integer, nullable=False)
    committed_state_version: Mapped[int | None] = mapped_column(Integer)
    generation_id: Mapped[UUID | None] = mapped_column(ID)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="accepted")
    generation_mode: Mapped[str | None] = mapped_column(String(16))
    failure_code: Mapped[str | None] = mapped_column(String(96))
    decision_audit: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    accepted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    committed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class TripSemanticStateVersion(Base):
    """Authoritative semantic projection for one committed trip version."""

    __tablename__ = "trip_semantic_state_versions"
    __table_args__ = (
        ForeignKeyConstraint(
            ["trip_id", "state_version"],
            ["trip_snapshots.trip_id", "trip_snapshots.state_version"],
            name="fk_semantic_state_snapshot_version",
            ondelete="CASCADE",
            deferrable=True,
            initially="DEFERRED",
        ),
        UniqueConstraint("trip_id", "state_version", name="uq_semantic_state_version"),
        CheckConstraint("state_version >= 0", name="ck_semantic_state_version"),
        CheckConstraint("schema_version = '4.0.0'", name="ck_semantic_state_v4_schema_version"),
        Index("ix_semantic_state_trip_created", "trip_id", "created_at"),
    )

    id: Mapped[UUID] = mapped_column(ID, primary_key=True, default=uuid4)
    trip_id: Mapped[UUID] = mapped_column(ID, nullable=False)
    state_version: Mapped[int] = mapped_column(Integer, nullable=False)
    schema_version: Mapped[str] = mapped_column(String(24), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class DiscoveryRuntimeStateVersion(Base):
    """Committed discovery progress sharing the semantic state's version."""

    __tablename__ = "discovery_runtime_state_versions"
    __table_args__ = (
        ForeignKeyConstraint(
            ["trip_id", "state_version"],
            ["trip_snapshots.trip_id", "trip_snapshots.state_version"],
            name="fk_discovery_runtime_snapshot_version",
            ondelete="CASCADE",
            deferrable=True,
            initially="DEFERRED",
        ),
        UniqueConstraint("trip_id", "state_version", name="uq_discovery_runtime_version"),
        CheckConstraint("state_version >= 0", name="ck_discovery_runtime_version"),
        CheckConstraint("schema_version = '4.0.0'", name="ck_discovery_runtime_v4_schema_version"),
        Index("ix_discovery_runtime_trip_created", "trip_id", "created_at"),
    )

    id: Mapped[UUID] = mapped_column(ID, primary_key=True, default=uuid4)
    trip_id: Mapped[UUID] = mapped_column(ID, nullable=False)
    state_version: Mapped[int] = mapped_column(Integer, nullable=False)
    schema_version: Mapped[str] = mapped_column(String(24), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class SemanticOperationRecord(Base):
    """Append-only accepted semantic operation and its provenance."""

    __tablename__ = "semantic_operations"
    __table_args__ = (
        ForeignKeyConstraint(
            ["turn_id", "trip_id"],
            ["agent_turns.id", "agent_turns.trip_id"],
            name="fk_semantic_operations_turn_same_trip",
            ondelete="CASCADE",
            deferrable=True,
            initially="DEFERRED",
        ),
        UniqueConstraint("trip_id", "operation_id", name="uq_semantic_operations_trip_id"),
        CheckConstraint("state_version >= 0", name="ck_semantic_operations_state_version"),
        Index("ix_semantic_operations_trip_version", "trip_id", "state_version"),
    )

    operation_id: Mapped[UUID] = mapped_column(ID, primary_key=True)
    trip_id: Mapped[UUID] = mapped_column(ID, nullable=False)
    turn_id: Mapped[UUID] = mapped_column(ID, nullable=False)
    state_version: Mapped[int] = mapped_column(Integer, nullable=False)
    target: Mapped[str] = mapped_column(String(96), nullable=False)
    operation_kind: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="active")
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    evidence: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    supersedes_operation_id: Mapped[UUID | None] = mapped_column(ID)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class ToolObservationRecord(Base):
    """Safe, normalized observation; raw provider responses are deliberately excluded."""

    __tablename__ = "tool_observations"
    __table_args__ = (
        ForeignKeyConstraint(
            ["turn_id", "trip_id"],
            ["agent_turns.id", "agent_turns.trip_id"],
            name="fk_tool_observations_turn_same_trip",
            ondelete="CASCADE",
            deferrable=True,
            initially="DEFERRED",
        ),
        UniqueConstraint("trip_id", "observation_id", name="uq_tool_observations_trip_id"),
        Index("ix_tool_observations_trip_observed", "trip_id", "observed_at"),
        Index("ix_tool_observations_request_hash", "request_hash"),
    )

    observation_id: Mapped[UUID] = mapped_column(ID, primary_key=True)
    trip_id: Mapped[UUID] = mapped_column(ID, nullable=False)
    turn_id: Mapped[UUID] = mapped_column(ID, nullable=False)
    tool_name: Mapped[str] = mapped_column(String(96), nullable=False)
    provider: Mapped[str] = mapped_column(String(64), nullable=False)
    request_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    safe_payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    source_refs: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class PendingInteractionRecord(Base):
    """Server-issued interaction whose answer is version-bound and idempotent."""

    __tablename__ = "pending_interactions"
    __table_args__ = (
        ForeignKeyConstraint(
            ["turn_id", "trip_id"],
            ["agent_turns.id", "agent_turns.trip_id"],
            name="fk_pending_interactions_turn_same_trip",
            ondelete="CASCADE",
            deferrable=True,
            initially="DEFERRED",
        ),
        UniqueConstraint("id", "trip_id", name="uq_pending_interactions_id_trip_id"),
        UniqueConstraint("trip_id", "answer_id", name="uq_pending_interactions_answer"),
        CheckConstraint("issued_state_version >= 0", name="ck_pending_interactions_issued_version"),
        CheckConstraint(
            "closed_state_version IS NULL OR closed_state_version >= issued_state_version",
            name="ck_pending_interactions_closed_version",
        ),
        CheckConstraint(
            "status IN ('active', 'answered', 'superseded', 'expired')",
            name="ck_pending_interactions_status",
        ),
        Index(
            "uq_pending_interactions_one_active",
            "trip_id",
            unique=True,
            sqlite_where=text("status = 'active'"),
            postgresql_where=text("status = 'active'"),
        ),
        Index("ix_pending_interactions_trip_created", "trip_id", "created_at"),
    )

    id: Mapped[UUID] = mapped_column(ID, primary_key=True, default=uuid4)
    trip_id: Mapped[UUID] = mapped_column(ID, nullable=False)
    turn_id: Mapped[UUID] = mapped_column(ID, nullable=False)
    source_message_id: Mapped[UUID] = mapped_column(ID, nullable=False)
    section: Mapped[str] = mapped_column(String(64), nullable=False)
    interaction_kind: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="active")
    issued_state_version: Mapped[int] = mapped_column(Integer, nullable=False)
    closed_state_version: Mapped[int | None] = mapped_column(Integer)
    answer_id: Mapped[UUID | None] = mapped_column(ID)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class AgentCheckpoint(TimestampMixin, Base):
    """In-flight checkpoint keyed by trip and turn, not a domain-state replacement."""

    __tablename__ = "agent_checkpoints"
    __table_args__ = (
        ForeignKeyConstraint(
            ["turn_id", "trip_id"],
            ["agent_turns.id", "agent_turns.trip_id"],
            name="fk_agent_checkpoints_turn_same_trip",
            ondelete="CASCADE",
            deferrable=True,
            initially="DEFERRED",
        ),
        UniqueConstraint(
            "trip_id",
            "turn_id",
            "checkpoint_namespace",
            "checkpoint_id",
            name="uq_agent_checkpoints_identity",
        ),
        CheckConstraint("base_state_version >= 0", name="ck_agent_checkpoints_base_version"),
        CheckConstraint(
            "status IN ('active', 'completed', 'superseded')",
            name="ck_agent_checkpoints_status",
        ),
        Index("ix_agent_checkpoints_turn_updated", "turn_id", "updated_at"),
        Index("ix_agent_checkpoints_expires", "expires_at"),
    )

    id: Mapped[UUID] = mapped_column(ID, primary_key=True, default=uuid4)
    trip_id: Mapped[UUID] = mapped_column(ID, nullable=False)
    turn_id: Mapped[UUID] = mapped_column(ID, nullable=False)
    checkpoint_namespace: Mapped[str] = mapped_column(String(64), nullable=False, default="prepare")
    checkpoint_id: Mapped[str] = mapped_column(String(128), nullable=False)
    checkpoint_version: Mapped[str] = mapped_column(String(32), nullable=False)
    base_state_version: Mapped[int] = mapped_column(Integer, nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="active")
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class PlannerWorkspace(TimestampMixin, Base):
    """Recoverable planning work; deliberately has no formal-plan publication pointer."""

    __tablename__ = "planner_workspaces"
    __table_args__ = (
        ForeignKeyConstraint(
            ["turn_id", "trip_id"],
            ["agent_turns.id", "agent_turns.trip_id"],
            name="fk_planner_workspaces_turn_same_trip",
            ondelete="CASCADE",
            deferrable=True,
            initially="DEFERRED",
        ),
        UniqueConstraint(
            "trip_id", "generation_id", "revision", name="uq_planner_workspaces_revision"
        ),
        CheckConstraint("base_state_version >= 0", name="ck_planner_workspaces_base_version"),
        CheckConstraint("revision >= 0", name="ck_planner_workspaces_revision"),
        CheckConstraint(
            "status IN ('working', 'awaiting_user', 'completed', 'stale')",
            name="ck_planner_workspaces_status",
        ),
        Index("ix_planner_workspaces_trip_updated", "trip_id", "updated_at"),
    )

    id: Mapped[UUID] = mapped_column(ID, primary_key=True, default=uuid4)
    trip_id: Mapped[UUID] = mapped_column(ID, nullable=False)
    turn_id: Mapped[UUID] = mapped_column(ID, nullable=False)
    generation_id: Mapped[UUID] = mapped_column(ID, nullable=False)
    confirmed_task_book_id: Mapped[UUID] = mapped_column(ID, nullable=False)
    confirmed_task_book_version: Mapped[int] = mapped_column(Integer, nullable=False)
    confirmed_task_book_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    base_state_version: Mapped[int] = mapped_column(Integer, nullable=False)
    revision: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    checkpoint_version: Mapped[str] = mapped_column(String(32), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="working")


class OutboxEvent(Base):
    """At-least-once delivery bundle built exclusively from committed messages."""

    __tablename__ = "outbox_events"
    __table_args__ = (
        ForeignKeyConstraint(
            ["turn_id", "trip_id"],
            ["agent_turns.id", "agent_turns.trip_id"],
            name="fk_outbox_events_turn_same_trip",
            ondelete="CASCADE",
            deferrable=True,
            initially="DEFERRED",
        ),
        ForeignKeyConstraint(
            ["message_id", "trip_id"],
            ["messages.id", "messages.trip_id"],
            name="fk_outbox_events_message_same_trip",
            ondelete="CASCADE",
            deferrable=True,
            initially="DEFERRED",
        ),
        UniqueConstraint("cursor", name="uq_outbox_events_cursor"),
        UniqueConstraint("trip_id", "publication_key", name="uq_outbox_events_publication"),
        CheckConstraint("committed_state_version >= 0", name="ck_outbox_state_version"),
        CheckConstraint("retry_count >= 0", name="ck_outbox_retry_count"),
        CheckConstraint("delivered_sequence >= 0", name="ck_outbox_delivered_sequence"),
        CheckConstraint(
            "delivery_status IN ('pending', 'leased', 'delivered', 'dead')",
            name="ck_outbox_delivery_status",
        ),
        Index(
            "ix_outbox_dispatch",
            "delivery_status",
            "next_attempt_at",
            "lease_expires_at",
        ),
        Index("ix_outbox_trip_created", "trip_id", "created_at"),
    )

    id: Mapped[UUID] = mapped_column(ID, primary_key=True, default=uuid4)
    cursor: Mapped[str] = mapped_column(String(128), nullable=False)
    publication_key: Mapped[str] = mapped_column(String(160), nullable=False)
    trip_id: Mapped[UUID] = mapped_column(ID, nullable=False)
    turn_id: Mapped[UUID] = mapped_column(ID, nullable=False)
    generation_id: Mapped[UUID] = mapped_column(ID, nullable=False)
    message_id: Mapped[UUID] = mapped_column(ID, nullable=False)
    committed_state_version: Mapped[int] = mapped_column(Integer, nullable=False)
    plan_version_id: Mapped[UUID | None] = mapped_column(ID)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    chunks: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    terminal_event: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    chunking_algorithm_version: Mapped[str] = mapped_column(String(64), nullable=False)
    delivery_status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    delivered_sequence: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    next_attempt_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    leased_by: Mapped[str | None] = mapped_column(String(96))
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_error_code: Mapped[str | None] = mapped_column(String(96))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    delivered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class CityRegistry(TimestampMixin, Base):
    """Data-driven city catalog; product availability is not encoded in table shape."""

    __tablename__ = "city_registry"
    __table_args__ = (
        UniqueConstraint("country_code", "admin_code", name="uq_city_registry_admin_code"),
        CheckConstraint(
            "support_status IN ('planned', 'enabled', 'disabled')",
            name="ck_city_registry_support_status",
        ),
        CheckConstraint(
            "coverage_level IN ('gold', 'standard', 'provider_only', 'unavailable')",
            name="ck_city_registry_coverage_level",
        ),
        CheckConstraint("coord_system = 'gcj_02'", name="ck_city_registry_gcj02"),
        CheckConstraint(
            "centroid_latitude >= -90 AND centroid_latitude <= 90",
            name="ck_city_registry_latitude",
        ),
        CheckConstraint(
            "centroid_longitude >= -180 AND centroid_longitude <= 180",
            name="ck_city_registry_longitude",
        ),
        Index("ix_city_registry_support", "support_status", "coverage_level"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    country_code: Mapped[str] = mapped_column(String(2), nullable=False, default="CN")
    admin_code: Mapped[str] = mapped_column(String(24), nullable=False)
    display_name: Mapped[str] = mapped_column(String(120), nullable=False)
    aliases: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    timezone: Mapped[str] = mapped_column(String(64), nullable=False)
    centroid_latitude: Mapped[Decimal] = mapped_column(Numeric(9, 6), nullable=False)
    centroid_longitude: Mapped[Decimal] = mapped_column(Numeric(10, 6), nullable=False)
    coord_system: Mapped[str] = mapped_column(String(16), nullable=False, default="gcj_02")
    support_status: Mapped[str] = mapped_column(String(16), nullable=False)
    coverage_level: Mapped[str] = mapped_column(String(24), nullable=False)
    current_content_version: Mapped[str | None] = mapped_column(String(32))
    content_package_key: Mapped[str | None] = mapped_column(String(255))


class Place(TimestampMixin, Base):
    __tablename__ = "places"
    __table_args__ = (
        CheckConstraint(
            "category IN ('attraction', 'restaurant', 'hotel', 'transport', 'activity', 'other')",
            name="ck_places_category",
        ),
        CheckConstraint("coord_system = 'gcj_02'", name="ck_places_gcj02"),
        CheckConstraint("latitude >= -90 AND latitude <= 90", name="ck_places_latitude"),
        CheckConstraint("longitude >= -180 AND longitude <= 180", name="ck_places_longitude"),
        Index("ix_places_city_category", "city_id", "category"),
        Index("ix_places_city_name", "city_id", "name"),
    )

    id: Mapped[UUID] = mapped_column(ID, primary_key=True, default=uuid4)
    city_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("city_registry.id", ondelete="RESTRICT"), nullable=False
    )
    category: Mapped[str] = mapped_column(String(24), nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    address: Mapped[str | None] = mapped_column(String(500))
    latitude: Mapped[Decimal] = mapped_column(Numeric(9, 6), nullable=False)
    longitude: Mapped[Decimal] = mapped_column(Numeric(10, 6), nullable=False)
    coord_system: Mapped[str] = mapped_column(String(16), nullable=False, default="gcj_02")


class PlaceSourceMap(Base):
    __tablename__ = "place_source_map"
    __table_args__ = (
        UniqueConstraint("provider", "source_place_id", name="uq_place_source_identity"),
        CheckConstraint(
            "provider IN ('amap', 'baidu', 'flyai', 'weather', 'city_content', "
            "'official', 'manual')",
            name="ck_place_source_map_provider",
        ),
        CheckConstraint(
            "raw_coord_system IS NULL OR raw_coord_system IN ('gcj_02', 'wgs_84', 'bd_09')",
            name="ck_place_source_map_coord_system",
        ),
        CheckConstraint(
            "(raw_latitude IS NULL AND raw_longitude IS NULL) OR "
            "(raw_latitude IS NOT NULL AND raw_longitude IS NOT NULL)",
            name="ck_place_source_map_coordinate_pair",
        ),
        Index("ix_place_source_map_place", "place_id"),
    )

    id: Mapped[UUID] = mapped_column(ID, primary_key=True, default=uuid4)
    place_id: Mapped[UUID] = mapped_column(
        ID, ForeignKey("places.id", ondelete="CASCADE"), nullable=False
    )
    provider: Mapped[str] = mapped_column(String(24), nullable=False)
    source_place_id: Mapped[str] = mapped_column(String(255), nullable=False)
    raw_name: Mapped[str] = mapped_column(String(255), nullable=False)
    raw_address: Mapped[str | None] = mapped_column(String(500))
    raw_latitude: Mapped[Decimal | None] = mapped_column(Numeric(10, 7))
    raw_longitude: Mapped[Decimal | None] = mapped_column(Numeric(11, 7))
    raw_coord_system: Mapped[str | None] = mapped_column(String(16))
    raw_payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class PlaceFact(Base):
    __tablename__ = "place_facts"
    __table_args__ = (
        CheckConstraint(
            "provider IN ('amap', 'baidu', 'flyai', 'weather', 'city_content', "
            "'official', 'manual')",
            name="ck_place_facts_provider",
        ),
        CheckConstraint(
            "availability IN ('available', 'partial', 'missing')",
            name="ck_place_facts_availability",
        ),
        CheckConstraint(
            "fact_kind IN ('address', 'rating', 'popularity', 'review_count', "
            "'regular_hours', 'price', 'phone', 'website', 'other')",
            name="ck_place_facts_kind",
        ),
        CheckConstraint(
            "evidence_status IN ('confirmed', 'inferred', 'assumed', 'unknown')",
            name="ck_place_facts_evidence_status",
        ),
        CheckConstraint(
            "(availability = 'missing' AND value IS NULL AND missing_reason IS NOT NULL) OR "
            "(availability IN ('available', 'partial') AND value IS NOT NULL)",
            name="ck_place_facts_availability_value",
        ),
        UniqueConstraint(
            "place_id",
            "provider",
            "source_record_id",
            "fact_kind",
            "fetched_at",
            name="uq_place_facts_source_observation",
        ),
        Index("ix_place_facts_place_kind_fetched", "place_id", "fact_kind", "fetched_at"),
        Index("ix_place_facts_expires", "expires_at"),
    )

    id: Mapped[UUID] = mapped_column(ID, primary_key=True, default=uuid4)
    place_id: Mapped[UUID] = mapped_column(
        ID, ForeignKey("places.id", ondelete="CASCADE"), nullable=False
    )
    fact_kind: Mapped[str] = mapped_column(String(48), nullable=False)
    provider: Mapped[str] = mapped_column(String(24), nullable=False)
    source_record_id: Mapped[str] = mapped_column(String(255), nullable=False)
    availability: Mapped[str] = mapped_column(String(16), nullable=False)
    value: Mapped[Any | None] = mapped_column(JSON)
    raw_value: Mapped[Any | None] = mapped_column(JSON)
    evidence_status: Mapped[str] = mapped_column(String(16), nullable=False)
    missing_reason: Mapped[str | None] = mapped_column(String(500))
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    valid_from: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    valid_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class HotelOfferRecord(Base):
    __tablename__ = "hotel_offers"
    __table_args__ = (
        CheckConstraint("check_out > check_in", name="ck_hotel_offers_date_range"),
        CheckConstraint("currency = 'CNY'", name="ck_hotel_offers_currency"),
        CheckConstraint(
            "(price_min_cents IS NULL AND price_max_cents IS NULL) OR "
            "(price_min_cents >= 0 AND price_max_cents >= price_min_cents)",
            name="ck_hotel_offers_price_range",
        ),
        CheckConstraint(
            "provider IN ('amap', 'baidu', 'flyai', 'weather', 'city_content', "
            "'official', 'manual')",
            name="ck_hotel_offers_provider",
        ),
        CheckConstraint(
            "availability IN ('available', 'partial', 'missing')",
            name="ck_hotel_offers_availability",
        ),
        CheckConstraint(
            "(availability = 'available' AND price_min_cents IS NOT NULL) OR "
            "(availability = 'missing' AND price_min_cents IS NULL) OR "
            "availability = 'partial'",
            name="ck_hotel_offers_availability_price",
        ),
        UniqueConstraint(
            "provider",
            "source_offer_id",
            "check_in",
            "check_out",
            "fetched_at",
            name="uq_hotel_offers_observation",
        ),
        Index("ix_hotel_offers_place_dates", "hotel_place_id", "check_in", "check_out"),
        Index("ix_hotel_offers_expires", "expires_at"),
    )

    id: Mapped[UUID] = mapped_column(ID, primary_key=True, default=uuid4)
    hotel_place_id: Mapped[UUID] = mapped_column(
        ID, ForeignKey("places.id", ondelete="CASCADE"), nullable=False
    )
    provider: Mapped[str] = mapped_column(String(24), nullable=False)
    source_offer_id: Mapped[str] = mapped_column(String(255), nullable=False)
    check_in: Mapped[date] = mapped_column(Date, nullable=False)
    check_out: Mapped[date] = mapped_column(Date, nullable=False)
    availability: Mapped[str] = mapped_column(String(16), nullable=False)
    price_min_cents: Mapped[int | None] = mapped_column(Integer)
    price_max_cents: Mapped[int | None] = mapped_column(Integer)
    currency: Mapped[str] = mapped_column(String(3), nullable=False, default="CNY")
    raw_price: Mapped[Any | None] = mapped_column(JSON)
    raw_currency: Mapped[str | None] = mapped_column(String(12))
    rating: Mapped[Decimal | None] = mapped_column(Numeric(3, 2))
    image_urls: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    detail_url: Mapped[str | None] = mapped_column(Text)
    missing_fields: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    missing_reason: Mapped[str | None] = mapped_column(String(500))
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class TicketOfferRecord(Base):
    __tablename__ = "ticket_offers"
    __table_args__ = (
        CheckConstraint("currency = 'CNY'", name="ck_ticket_offers_currency"),
        CheckConstraint(
            "(price_min_cents IS NULL AND price_max_cents IS NULL) OR "
            "(price_min_cents >= 0 AND price_max_cents >= price_min_cents)",
            name="ck_ticket_offers_price_range",
        ),
        CheckConstraint(
            "provider IN ('amap', 'baidu', 'flyai', 'weather', 'city_content', "
            "'official', 'manual')",
            name="ck_ticket_offers_provider",
        ),
        CheckConstraint(
            "availability IN ('available', 'partial', 'missing')",
            name="ck_ticket_offers_availability",
        ),
        CheckConstraint(
            "(availability = 'available' AND price_min_cents IS NOT NULL) OR "
            "(availability = 'missing' AND price_min_cents IS NULL) OR "
            "availability = 'partial'",
            name="ck_ticket_offers_availability_price",
        ),
        Index(
            "uq_ticket_offers_dated_observation",
            "provider",
            "source_offer_id",
            "visit_date",
            "fetched_at",
            unique=True,
            postgresql_where=text("visit_date IS NOT NULL"),
            sqlite_where=text("visit_date IS NOT NULL"),
        ),
        Index(
            "uq_ticket_offers_undated_observation",
            "provider",
            "source_offer_id",
            "fetched_at",
            unique=True,
            postgresql_where=text("visit_date IS NULL"),
            sqlite_where=text("visit_date IS NULL"),
        ),
        Index("ix_ticket_offers_place_date", "place_id", "visit_date"),
        Index("ix_ticket_offers_expires", "expires_at"),
    )

    id: Mapped[UUID] = mapped_column(ID, primary_key=True, default=uuid4)
    place_id: Mapped[UUID] = mapped_column(
        ID, ForeignKey("places.id", ondelete="CASCADE"), nullable=False
    )
    provider: Mapped[str] = mapped_column(String(24), nullable=False)
    source_offer_id: Mapped[str] = mapped_column(String(255), nullable=False)
    visit_date: Mapped[date | None] = mapped_column(Date)
    availability: Mapped[str] = mapped_column(String(16), nullable=False)
    price_min_cents: Mapped[int | None] = mapped_column(Integer)
    price_max_cents: Mapped[int | None] = mapped_column(Integer)
    currency: Mapped[str] = mapped_column(String(3), nullable=False, default="CNY")
    raw_price: Mapped[Any | None] = mapped_column(JSON)
    raw_currency: Mapped[str | None] = mapped_column(String(12))
    detail_url: Mapped[str | None] = mapped_column(Text)
    missing_fields: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    missing_reason: Mapped[str | None] = mapped_column(String(500))
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class Artifact(Base):
    __tablename__ = "artifacts"
    __table_args__ = (
        ForeignKeyConstraint(
            ["plan_version_id", "trip_id"],
            ["trip_versions.id", "trip_versions.trip_id"],
            name="fk_artifacts_plan_version_same_trip",
            ondelete="NO ACTION",
            deferrable=True,
            initially="DEFERRED",
        ),
        CheckConstraint("status IN ('pending', 'ready', 'failed')", name="ck_artifacts_status"),
        CheckConstraint(
            "status != 'ready' OR storage_key IS NOT NULL", name="ck_artifacts_ready_storage"
        ),
        UniqueConstraint(
            "trip_id", "plan_version_id", "artifact_type", name="uq_artifacts_plan_type"
        ),
        Index("ix_artifacts_trip_created", "trip_id", "created_at"),
        Index("ix_artifacts_expires", "expires_at"),
    )

    id: Mapped[UUID] = mapped_column(ID, primary_key=True, default=uuid4)
    trip_id: Mapped[UUID] = mapped_column(
        ID, ForeignKey("trips.id", ondelete="CASCADE"), nullable=False
    )
    plan_version_id: Mapped[UUID] = mapped_column(ID, nullable=False)
    artifact_type: Mapped[str] = mapped_column(String(32), nullable=False, default="long_image")
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    storage_key: Mapped[str | None] = mapped_column(String(1024))
    content_type: Mapped[str | None] = mapped_column(String(128))
    checksum_sha256: Mapped[str | None] = mapped_column(String(64))
    error_code: Mapped[str | None] = mapped_column(String(96))
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


class ProductEvent(Base):
    __tablename__ = "events"
    __table_args__ = (
        ForeignKeyConstraint(
            ["trip_id", "user_id"],
            ["trips.id", "trips.owner_user_id"],
            name="fk_events_trip_owner",
            ondelete="CASCADE",
            deferrable=True,
            initially="DEFERRED",
        ),
        CheckConstraint(
            "actor_type IN ('anonymous', 'user', 'system')", name="ck_events_actor_type"
        ),
        CheckConstraint(
            "(actor_type = 'anonymous' AND anonymous_subject_hash IS NOT NULL "
            "AND user_id IS NULL AND trip_id IS NULL) OR "
            "(actor_type = 'user' AND user_id IS NOT NULL "
            "AND anonymous_subject_hash IS NULL) OR actor_type = 'system'",
            name="ck_events_actor_identity",
        ),
        Index("ix_events_trip_occurred", "trip_id", "occurred_at"),
        Index("ix_events_name_occurred", "event_name", "occurred_at"),
        Index("ix_events_user_occurred", "user_id", "occurred_at"),
    )

    id: Mapped[UUID] = mapped_column(ID, primary_key=True, default=uuid4)
    trip_id: Mapped[UUID | None] = mapped_column(ID, ForeignKey("trips.id", ondelete="CASCADE"))
    user_id: Mapped[UUID | None] = mapped_column(ID, ForeignKey("users.id", ondelete="CASCADE"))
    anonymous_subject_hash: Mapped[str | None] = mapped_column(String(64))
    actor_type: Mapped[str] = mapped_column(String(16), nullable=False)
    event_name: Mapped[str] = mapped_column(String(96), nullable=False)
    event_version: Mapped[str] = mapped_column(String(24), nullable=False)
    request_id: Mapped[UUID | None] = mapped_column(ID)
    generation_id: Mapped[UUID | None] = mapped_column(ID)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    received_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
