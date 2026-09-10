"""Create the data-driven city, place, product, artifact, and event stores.

Revision ID: 20260822_0003
Revises: 20260822_0002
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260822_0003"
down_revision = "20260822_0002"
branch_labels = None
depends_on = None

PROVIDERS = "'amap', 'baidu', 'flyai', 'weather', 'city_content', 'official', 'manual'"


def _timestamps() -> tuple[sa.Column[object], sa.Column[object]]:
    return (
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )


def upgrade() -> None:
    created_at, updated_at = _timestamps()
    op.create_table(
        "city_registry",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("country_code", sa.String(length=2), nullable=False),
        sa.Column("admin_code", sa.String(length=24), nullable=False),
        sa.Column("display_name", sa.String(length=120), nullable=False),
        sa.Column("aliases", sa.JSON(), nullable=False),
        sa.Column("timezone", sa.String(length=64), nullable=False),
        sa.Column("centroid_latitude", sa.Numeric(precision=9, scale=6), nullable=False),
        sa.Column("centroid_longitude", sa.Numeric(precision=10, scale=6), nullable=False),
        sa.Column("coord_system", sa.String(length=16), nullable=False),
        sa.Column("support_status", sa.String(length=16), nullable=False),
        sa.Column("coverage_level", sa.String(length=24), nullable=False),
        sa.Column("current_content_version", sa.String(length=32), nullable=True),
        sa.Column("content_package_key", sa.String(length=255), nullable=True),
        created_at,
        updated_at,
        sa.CheckConstraint(
            "support_status IN ('planned', 'enabled', 'disabled')",
            name="ck_city_registry_support_status",
        ),
        sa.CheckConstraint(
            "coverage_level IN ('gold', 'standard', 'provider_only', 'unavailable')",
            name="ck_city_registry_coverage_level",
        ),
        sa.CheckConstraint("coord_system = 'gcj_02'", name="ck_city_registry_gcj02"),
        sa.CheckConstraint(
            "centroid_latitude >= -90 AND centroid_latitude <= 90",
            name="ck_city_registry_latitude",
        ),
        sa.CheckConstraint(
            "centroid_longitude >= -180 AND centroid_longitude <= 180",
            name="ck_city_registry_longitude",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("country_code", "admin_code", name="uq_city_registry_admin_code"),
    )
    op.create_index(
        "ix_city_registry_support", "city_registry", ["support_status", "coverage_level"]
    )

    created_at, updated_at = _timestamps()
    op.create_table(
        "places",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("city_id", sa.String(length=64), nullable=False),
        sa.Column("category", sa.String(length=24), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("address", sa.String(length=500), nullable=True),
        sa.Column("latitude", sa.Numeric(precision=9, scale=6), nullable=False),
        sa.Column("longitude", sa.Numeric(precision=10, scale=6), nullable=False),
        sa.Column("coord_system", sa.String(length=16), nullable=False),
        created_at,
        updated_at,
        sa.CheckConstraint(
            "category IN ('attraction', 'restaurant', 'hotel', 'transport', 'activity', 'other')",
            name="ck_places_category",
        ),
        sa.CheckConstraint("coord_system = 'gcj_02'", name="ck_places_gcj02"),
        sa.CheckConstraint("latitude >= -90 AND latitude <= 90", name="ck_places_latitude"),
        sa.CheckConstraint("longitude >= -180 AND longitude <= 180", name="ck_places_longitude"),
        sa.ForeignKeyConstraint(["city_id"], ["city_registry.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_places_city_category", "places", ["city_id", "category"])
    op.create_index("ix_places_city_name", "places", ["city_id", "name"])

    op.create_table(
        "place_source_map",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("place_id", sa.Uuid(), nullable=False),
        sa.Column("provider", sa.String(length=24), nullable=False),
        sa.Column("source_place_id", sa.String(length=255), nullable=False),
        sa.Column("raw_name", sa.String(length=255), nullable=False),
        sa.Column("raw_address", sa.String(length=500), nullable=True),
        sa.Column("raw_latitude", sa.Numeric(precision=10, scale=7), nullable=True),
        sa.Column("raw_longitude", sa.Numeric(precision=11, scale=7), nullable=True),
        sa.Column("raw_coord_system", sa.String(length=16), nullable=True),
        sa.Column("raw_payload", sa.JSON(), nullable=False),
        sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.CheckConstraint(f"provider IN ({PROVIDERS})", name="ck_place_source_map_provider"),
        sa.CheckConstraint(
            "raw_coord_system IS NULL OR raw_coord_system IN ('gcj_02', 'wgs_84', 'bd_09')",
            name="ck_place_source_map_coord_system",
        ),
        sa.CheckConstraint(
            "(raw_latitude IS NULL AND raw_longitude IS NULL) OR "
            "(raw_latitude IS NOT NULL AND raw_longitude IS NOT NULL)",
            name="ck_place_source_map_coordinate_pair",
        ),
        sa.ForeignKeyConstraint(["place_id"], ["places.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("provider", "source_place_id", name="uq_place_source_identity"),
    )
    op.create_index("ix_place_source_map_place", "place_source_map", ["place_id"])

    op.create_table(
        "place_facts",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("place_id", sa.Uuid(), nullable=False),
        sa.Column("fact_kind", sa.String(length=48), nullable=False),
        sa.Column("provider", sa.String(length=24), nullable=False),
        sa.Column("source_record_id", sa.String(length=255), nullable=False),
        sa.Column("availability", sa.String(length=16), nullable=False),
        sa.Column("value", sa.JSON(), nullable=True),
        sa.Column("raw_value", sa.JSON(), nullable=True),
        sa.Column("evidence_status", sa.String(length=16), nullable=False),
        sa.Column("missing_reason", sa.String(length=500), nullable=True),
        sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("valid_from", sa.DateTime(timezone=True), nullable=True),
        sa.Column("valid_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.CheckConstraint(f"provider IN ({PROVIDERS})", name="ck_place_facts_provider"),
        sa.CheckConstraint(
            "availability IN ('available', 'partial', 'missing')",
            name="ck_place_facts_availability",
        ),
        sa.CheckConstraint(
            "fact_kind IN ('address', 'rating', 'popularity', 'review_count', "
            "'regular_hours', 'price', 'phone', 'website', 'other')",
            name="ck_place_facts_kind",
        ),
        sa.CheckConstraint(
            "evidence_status IN ('confirmed', 'inferred', 'assumed', 'unknown')",
            name="ck_place_facts_evidence_status",
        ),
        sa.CheckConstraint(
            "(availability = 'missing' AND value IS NULL AND missing_reason IS NOT NULL) OR "
            "(availability IN ('available', 'partial') AND value IS NOT NULL)",
            name="ck_place_facts_availability_value",
        ),
        sa.ForeignKeyConstraint(["place_id"], ["places.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "place_id",
            "provider",
            "source_record_id",
            "fact_kind",
            "fetched_at",
            name="uq_place_facts_source_observation",
        ),
    )
    op.create_index(
        "ix_place_facts_place_kind_fetched",
        "place_facts",
        ["place_id", "fact_kind", "fetched_at"],
    )
    op.create_index("ix_place_facts_expires", "place_facts", ["expires_at"])

    op.create_table(
        "hotel_offers",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("hotel_place_id", sa.Uuid(), nullable=False),
        sa.Column("provider", sa.String(length=24), nullable=False),
        sa.Column("source_offer_id", sa.String(length=255), nullable=False),
        sa.Column("check_in", sa.Date(), nullable=False),
        sa.Column("check_out", sa.Date(), nullable=False),
        sa.Column("availability", sa.String(length=16), nullable=False),
        sa.Column("price_min_cents", sa.Integer(), nullable=True),
        sa.Column("price_max_cents", sa.Integer(), nullable=True),
        sa.Column("currency", sa.String(length=3), nullable=False),
        sa.Column("raw_price", sa.JSON(), nullable=True),
        sa.Column("raw_currency", sa.String(length=12), nullable=True),
        sa.Column("rating", sa.Numeric(precision=3, scale=2), nullable=True),
        sa.Column("image_urls", sa.JSON(), nullable=False),
        sa.Column("detail_url", sa.Text(), nullable=True),
        sa.Column("missing_fields", sa.JSON(), nullable=False),
        sa.Column("missing_reason", sa.String(length=500), nullable=True),
        sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.CheckConstraint("check_out > check_in", name="ck_hotel_offers_date_range"),
        sa.CheckConstraint("currency = 'CNY'", name="ck_hotel_offers_currency"),
        sa.CheckConstraint(
            "(price_min_cents IS NULL AND price_max_cents IS NULL) OR "
            "(price_min_cents >= 0 AND price_max_cents >= price_min_cents)",
            name="ck_hotel_offers_price_range",
        ),
        sa.CheckConstraint(f"provider IN ({PROVIDERS})", name="ck_hotel_offers_provider"),
        sa.CheckConstraint(
            "availability IN ('available', 'partial', 'missing')",
            name="ck_hotel_offers_availability",
        ),
        sa.CheckConstraint(
            "(availability = 'available' AND price_min_cents IS NOT NULL) OR "
            "(availability = 'missing' AND price_min_cents IS NULL) OR "
            "availability = 'partial'",
            name="ck_hotel_offers_availability_price",
        ),
        sa.ForeignKeyConstraint(["hotel_place_id"], ["places.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "provider",
            "source_offer_id",
            "check_in",
            "check_out",
            "fetched_at",
            name="uq_hotel_offers_observation",
        ),
    )
    op.create_index(
        "ix_hotel_offers_place_dates",
        "hotel_offers",
        ["hotel_place_id", "check_in", "check_out"],
    )
    op.create_index("ix_hotel_offers_expires", "hotel_offers", ["expires_at"])

    op.create_table(
        "ticket_offers",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("place_id", sa.Uuid(), nullable=False),
        sa.Column("provider", sa.String(length=24), nullable=False),
        sa.Column("source_offer_id", sa.String(length=255), nullable=False),
        sa.Column("visit_date", sa.Date(), nullable=True),
        sa.Column("availability", sa.String(length=16), nullable=False),
        sa.Column("price_min_cents", sa.Integer(), nullable=True),
        sa.Column("price_max_cents", sa.Integer(), nullable=True),
        sa.Column("currency", sa.String(length=3), nullable=False),
        sa.Column("raw_price", sa.JSON(), nullable=True),
        sa.Column("raw_currency", sa.String(length=12), nullable=True),
        sa.Column("detail_url", sa.Text(), nullable=True),
        sa.Column("missing_fields", sa.JSON(), nullable=False),
        sa.Column("missing_reason", sa.String(length=500), nullable=True),
        sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.CheckConstraint("currency = 'CNY'", name="ck_ticket_offers_currency"),
        sa.CheckConstraint(
            "(price_min_cents IS NULL AND price_max_cents IS NULL) OR "
            "(price_min_cents >= 0 AND price_max_cents >= price_min_cents)",
            name="ck_ticket_offers_price_range",
        ),
        sa.CheckConstraint(f"provider IN ({PROVIDERS})", name="ck_ticket_offers_provider"),
        sa.CheckConstraint(
            "availability IN ('available', 'partial', 'missing')",
            name="ck_ticket_offers_availability",
        ),
        sa.CheckConstraint(
            "(availability = 'available' AND price_min_cents IS NOT NULL) OR "
            "(availability = 'missing' AND price_min_cents IS NULL) OR "
            "availability = 'partial'",
            name="ck_ticket_offers_availability_price",
        ),
        sa.ForeignKeyConstraint(["place_id"], ["places.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "provider",
            "source_offer_id",
            "visit_date",
            "fetched_at",
            name="uq_ticket_offers_observation",
        ),
    )
    op.create_index("ix_ticket_offers_place_date", "ticket_offers", ["place_id", "visit_date"])
    op.create_index("ix_ticket_offers_expires", "ticket_offers", ["expires_at"])

    with op.batch_alter_table("trips") as batch_op:
        batch_op.create_unique_constraint("uq_trips_id_owner", ["id", "owner_user_id"])

    op.create_table(
        "artifacts",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("trip_id", sa.Uuid(), nullable=False),
        sa.Column("plan_version_id", sa.Uuid(), nullable=False),
        sa.Column("artifact_type", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("storage_key", sa.String(length=1024), nullable=True),
        sa.Column("content_type", sa.String(length=128), nullable=True),
        sa.Column("checksum_sha256", sa.String(length=64), nullable=True),
        sa.Column("error_code", sa.String(length=96), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.CheckConstraint("status IN ('pending', 'ready', 'failed')", name="ck_artifacts_status"),
        sa.CheckConstraint(
            "status != 'ready' OR storage_key IS NOT NULL", name="ck_artifacts_ready_storage"
        ),
        sa.ForeignKeyConstraint(["trip_id"], ["trips.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["plan_version_id", "trip_id"],
            ["trip_versions.id", "trip_versions.trip_id"],
            name="fk_artifacts_plan_version_same_trip",
            ondelete="NO ACTION",
            deferrable=True,
            initially="DEFERRED",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "trip_id", "plan_version_id", "artifact_type", name="uq_artifacts_plan_type"
        ),
    )
    op.create_index("ix_artifacts_trip_created", "artifacts", ["trip_id", "created_at"])
    op.create_index("ix_artifacts_expires", "artifacts", ["expires_at"])

    op.create_table(
        "events",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("trip_id", sa.Uuid(), nullable=True),
        sa.Column("user_id", sa.Uuid(), nullable=True),
        sa.Column("anonymous_subject_hash", sa.String(length=64), nullable=True),
        sa.Column("actor_type", sa.String(length=16), nullable=False),
        sa.Column("event_name", sa.String(length=96), nullable=False),
        sa.Column("event_version", sa.String(length=24), nullable=False),
        sa.Column("request_id", sa.Uuid(), nullable=True),
        sa.Column("generation_id", sa.Uuid(), nullable=True),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "received_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.CheckConstraint(
            "actor_type IN ('anonymous', 'user', 'system')", name="ck_events_actor_type"
        ),
        sa.CheckConstraint(
            "(actor_type = 'anonymous' AND anonymous_subject_hash IS NOT NULL "
            "AND user_id IS NULL AND trip_id IS NULL) OR "
            "(actor_type = 'user' AND user_id IS NOT NULL "
            "AND anonymous_subject_hash IS NULL) OR actor_type = 'system'",
            name="ck_events_actor_identity",
        ),
        sa.ForeignKeyConstraint(["trip_id"], ["trips.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["trip_id", "user_id"],
            ["trips.id", "trips.owner_user_id"],
            name="fk_events_trip_owner",
            ondelete="CASCADE",
            deferrable=True,
            initially="DEFERRED",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_events_trip_occurred", "events", ["trip_id", "occurred_at"])
    op.create_index("ix_events_name_occurred", "events", ["event_name", "occurred_at"])
    op.create_index("ix_events_user_occurred", "events", ["user_id", "occurred_at"])


def downgrade() -> None:
    op.drop_index("ix_events_user_occurred", table_name="events")
    op.drop_index("ix_events_name_occurred", table_name="events")
    op.drop_index("ix_events_trip_occurred", table_name="events")
    op.drop_table("events")
    with op.batch_alter_table("trips") as batch_op:
        batch_op.drop_constraint("uq_trips_id_owner", type_="unique")
    op.drop_index("ix_artifacts_expires", table_name="artifacts")
    op.drop_index("ix_artifacts_trip_created", table_name="artifacts")
    op.drop_table("artifacts")
    op.drop_index("ix_ticket_offers_expires", table_name="ticket_offers")
    op.drop_index("ix_ticket_offers_place_date", table_name="ticket_offers")
    op.drop_table("ticket_offers")
    op.drop_index("ix_hotel_offers_expires", table_name="hotel_offers")
    op.drop_index("ix_hotel_offers_place_dates", table_name="hotel_offers")
    op.drop_table("hotel_offers")
    op.drop_index("ix_place_facts_expires", table_name="place_facts")
    op.drop_index("ix_place_facts_place_kind_fetched", table_name="place_facts")
    op.drop_table("place_facts")
    op.drop_index("ix_place_source_map_place", table_name="place_source_map")
    op.drop_table("place_source_map")
    op.drop_index("ix_places_city_name", table_name="places")
    op.drop_index("ix_places_city_category", table_name="places")
    op.drop_table("places")
    op.drop_index("ix_city_registry_support", table_name="city_registry")
    op.drop_table("city_registry")
