"""V3-34 contracts for route-aware spatial anchors and activity clusters."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, time
from enum import StrEnum
from uuid import UUID

from pydantic import AwareDatetime, ConfigDict, Field, model_validator

from backend.contracts.base import ContractModel
from backend.contracts.candidate_recall import RecalledPlace
from backend.contracts.common import NonEmptyText, ShortText
from backend.contracts.enums import AnchorRole, DataAvailability, ProviderCode
from backend.contracts.places import Gcj02Coordinates
from backend.providers.contracts import RouteMode


class ImmutableSpatialModel(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SpatialAnchorStrength(StrEnum):
    STRONG = "strong"
    ADJUSTABLE = "adjustable"


class SpatialIssueCode(StrEnum):
    OUTLIER_STRONG_ANCHOR = "outlier_strong_anchor"


STRONG_ANCHOR_ROLES = {
    AnchorRole.FIXED_EVENT,
    AnchorRole.FIXED_HOTEL,
    AnchorRole.MUST_ATTRACTION,
    AnchorRole.DESTINATION_RESTAURANT,
}
ADJUSTABLE_ANCHOR_ROLES = {
    AnchorRole.WANT_ATTRACTION,
    AnchorRole.CONVENIENT_ATTRACTION,
    AnchorRole.CONVENIENT_RESTAURANT,
}


class FixedTimeWindow(ImmutableSpatialModel):
    event_date: date
    start_time: time
    end_time: time

    @model_validator(mode="after")
    def end_follows_start(self) -> FixedTimeWindow:
        if self.end_time <= self.start_time:
            raise ValueError("fixed time window end must be after start")
        return self


class SpatialNodeInput(ImmutableSpatialModel):
    node_id: UUID
    place_id: UUID
    candidate_id: UUID | None = None
    role: AnchorRole
    available_dates: tuple[date, ...] = Field(min_length=1, max_length=5)
    fixed_time_window: FixedTimeWindow | None = None
    source_reference_ids: tuple[NonEmptyText, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def role_and_dates_are_consistent(self) -> SpatialNodeInput:
        _unique(self.available_dates, "spatial-node available dates")
        _unique(self.source_reference_ids, "spatial-node source references")
        if self.role not in STRONG_ANCHOR_ROLES | ADJUSTABLE_ANCHOR_ROLES:
            raise ValueError("unsupported spatial anchor role")
        if self.role is AnchorRole.FIXED_EVENT:
            if self.fixed_time_window is None:
                raise ValueError("fixed-event spatial node requires a fixed time window")
            if self.fixed_time_window.event_date not in self.available_dates:
                raise ValueError("fixed-event date must be one of the node's available dates")
        elif self.fixed_time_window is not None:
            raise ValueError("only fixed-event nodes may carry a fixed time window")
        return self

    @property
    def strength(self) -> SpatialAnchorStrength:
        return (
            SpatialAnchorStrength.STRONG
            if self.role in STRONG_ANCHOR_ROLES
            else SpatialAnchorStrength.ADJUSTABLE
        )


class SpatialPlanningRequest(ImmutableSpatialModel):
    request_id: UUID
    trip_id: UUID
    input_state_version: int = Field(ge=0, strict=True)
    task_book_id: UUID | None = None
    task_book_revision: int | None = Field(default=None, ge=1, strict=True)
    city_id: NonEmptyText
    start_date: date
    end_date: date
    places: tuple[RecalledPlace, ...] = Field(min_length=1, max_length=16)
    nodes: tuple[SpatialNodeInput, ...] = Field(min_length=1, max_length=16)
    route_modes: tuple[RouteMode, ...] = Field(min_length=1, max_length=4)
    cluster_threshold_minutes: int = Field(default=35, ge=5, le=180, strict=True)
    outlier_threshold_minutes: int = Field(default=75, ge=15, le=360, strict=True)

    @model_validator(mode="after")
    def planning_boundary_and_references_are_valid(self) -> SpatialPlanningRequest:
        if (self.task_book_id is None) != (self.task_book_revision is None):
            raise ValueError("task-book ID and revision must be supplied together")
        day_count = (self.end_date - self.start_date).days + 1
        if not 1 <= day_count <= 5:
            raise ValueError("spatial planning date range must contain 1 to 5 days")
        if self.outlier_threshold_minutes <= self.cluster_threshold_minutes:
            raise ValueError("outlier threshold must be greater than cluster threshold")
        _unique([place.place_id for place in self.places], "spatial place IDs")
        _unique([node.node_id for node in self.nodes], "spatial node IDs")
        _unique([node.place_id for node in self.nodes], "spatial node place IDs")
        _unique(self.route_modes, "spatial route modes")
        if any(place.city_id != self.city_id for place in self.places):
            raise ValueError("every spatial place must belong to the request city")
        places_by_id = {place.place_id: place for place in self.places}
        trip_dates = {
            self.start_date.fromordinal(day)
            for day in range(self.start_date.toordinal(), self.end_date.toordinal() + 1)
        }
        for node in self.nodes:
            place = places_by_id.get(node.place_id)
            if place is None:
                raise ValueError("spatial node references an unknown place")
            if place.city_id != self.city_id:
                raise ValueError("spatial node place cannot cross cities")
            if place.coordinates is None:
                raise ValueError("spatial planning requires GCJ-02 coordinates for every node")
            if not set(node.available_dates) <= trip_dates:
                raise ValueError("spatial node availability must stay inside the trip dates")
            if (
                node.role
                in {
                    AnchorRole.MUST_ATTRACTION,
                    AnchorRole.WANT_ATTRACTION,
                    AnchorRole.CONVENIENT_ATTRACTION,
                }
                and place.category.value != "attraction"
            ):
                raise ValueError("attraction spatial roles require attraction places")
            if (
                node.role
                in {
                    AnchorRole.DESTINATION_RESTAURANT,
                    AnchorRole.CONVENIENT_RESTAURANT,
                }
                and place.category.value != "restaurant"
            ):
                raise ValueError("restaurant spatial roles require restaurant places")
            if node.role is AnchorRole.FIXED_HOTEL and place.category.value != "hotel":
                raise ValueError("fixed-hotel spatial role requires a hotel place")
        return self


class SpatialAnchor(ImmutableSpatialModel):
    node_id: UUID
    place_id: UUID
    candidate_id: UUID | None = None
    role: AnchorRole
    strength: SpatialAnchorStrength
    name: NonEmptyText
    coordinates: Gcj02Coordinates
    available_dates: tuple[date, ...] = Field(min_length=1, max_length=5)
    fixed_time_window: FixedTimeWindow | None = None
    source_reference_ids: tuple[NonEmptyText, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def strength_matches_role(self) -> SpatialAnchor:
        expected = (
            SpatialAnchorStrength.STRONG
            if self.role in STRONG_ANCHOR_ROLES
            else SpatialAnchorStrength.ADJUSTABLE
        )
        if self.strength is not expected:
            raise ValueError("spatial anchor strength must match its role")
        _unique(self.available_dates, "spatial-anchor available dates")
        _unique(self.source_reference_ids, "spatial-anchor source references")
        if self.role is AnchorRole.FIXED_EVENT:
            if self.fixed_time_window is None:
                raise ValueError("fixed-event anchor requires a fixed time window")
        elif self.fixed_time_window is not None:
            raise ValueError("only fixed-event anchors may carry a fixed time window")
        return self


class SpatialRouteOption(ImmutableSpatialModel):
    provider: ProviderCode
    mode: RouteMode
    distance_m: int = Field(ge=0, strict=True)
    duration_seconds: int = Field(ge=0, strict=True)
    walking_distance_m: int | None = Field(default=None, ge=0, strict=True)
    transfer_count: int | None = Field(default=None, ge=0, strict=True)
    source_route_index: int = Field(ge=0, strict=True)
    polyline: tuple[Gcj02Coordinates, ...] = Field(default=(), max_length=20_000)
    fetched_at: AwareDatetime

    @model_validator(mode="after")
    def route_geometry_is_complete_when_present(self) -> SpatialRouteOption:
        if self.polyline and len(self.polyline) < 2:
            raise ValueError("spatial route polyline requires at least two points")
        return self


class SpatialRouteEdge(ImmutableSpatialModel):
    edge_id: UUID
    origin_node_id: UUID
    destination_node_id: UUID
    status: DataAvailability
    routes: tuple[SpatialRouteOption, ...] = ()
    missing_fields: tuple[NonEmptyText, ...] = ()
    missing_reason: ShortText | None = None

    @model_validator(mode="after")
    def availability_matches_routes(self) -> SpatialRouteEdge:
        if self.origin_node_id == self.destination_node_id:
            raise ValueError("spatial route edge requires two different nodes")
        _unique([route.mode for route in self.routes], "spatial edge route modes")
        _unique(self.missing_fields, "spatial edge missing fields")
        if self.status is DataAvailability.AVAILABLE:
            if not self.routes or self.missing_fields or self.missing_reason is not None:
                raise ValueError("available spatial edge requires complete route options")
        elif self.status is DataAvailability.PARTIAL:
            if not self.routes or not self.missing_fields or self.missing_reason is None:
                raise ValueError("partial spatial edge requires routes and missing details")
        elif self.routes or not self.missing_fields or self.missing_reason is None:
            raise ValueError("missing spatial edge requires an explicit unknown cost")
        return self


class ClusterCostSummary(ImmutableSpatialModel):
    pair_count: int = Field(ge=0, strict=True)
    known_pair_count: int = Field(ge=0, strict=True)
    partial_pair_count: int = Field(ge=0, strict=True)
    unknown_pair_count: int = Field(ge=0, strict=True)
    minimum_minutes: int | None = Field(default=None, ge=0, strict=True)
    average_minutes: int | None = Field(default=None, ge=0, strict=True)
    maximum_minutes: int | None = Field(default=None, ge=0, strict=True)

    @model_validator(mode="after")
    def counts_and_duration_are_consistent(self) -> ClusterCostSummary:
        if self.known_pair_count + self.unknown_pair_count != self.pair_count:
            raise ValueError("cluster cost counts must cover every member pair")
        if self.partial_pair_count > self.known_pair_count:
            raise ValueError("partial cluster pairs must also be known pairs")
        values = (self.minimum_minutes, self.average_minutes, self.maximum_minutes)
        if self.known_pair_count == 0:
            if any(value is not None for value in values):
                raise ValueError("cluster without known routes cannot report route minutes")
        elif any(value is None for value in values):
            raise ValueError("cluster with known routes requires complete route statistics")
        elif not self.minimum_minutes <= self.average_minutes <= self.maximum_minutes:  # type: ignore[operator]
            raise ValueError("cluster route statistics must be ordered")
        return self


class SpatialActivityCluster(ImmutableSpatialModel):
    cluster_id: UUID
    label: NonEmptyText
    member_node_ids: tuple[UUID, ...] = Field(min_length=1)
    center: Gcj02Coordinates
    suitable_dates: tuple[date, ...] = Field(min_length=1, max_length=5)
    intra_cluster_cost: ClusterCostSummary

    @model_validator(mode="after")
    def members_and_dates_are_unique(self) -> SpatialActivityCluster:
        _unique(self.member_node_ids, "spatial cluster members")
        _unique(self.suitable_dates, "spatial cluster dates")
        expected_pairs = len(self.member_node_ids) * (len(self.member_node_ids) - 1) // 2
        if self.intra_cluster_cost.pair_count != expected_pairs:
            raise ValueError("cluster cost pair count must match its members")
        return self


class InterClusterCost(ImmutableSpatialModel):
    origin_cluster_id: UUID
    destination_cluster_id: UUID
    status: DataAvailability
    best_edge_id: UUID | None = None
    duration_minutes: int | None = Field(default=None, ge=0, strict=True)
    distance_m: int | None = Field(default=None, ge=0, strict=True)
    missing_reason: ShortText | None = None

    @model_validator(mode="after")
    def availability_matches_cost(self) -> InterClusterCost:
        if self.origin_cluster_id == self.destination_cluster_id:
            raise ValueError("inter-cluster cost requires two different clusters")
        has_cost = (
            self.best_edge_id is not None
            and self.duration_minutes is not None
            and self.distance_m is not None
        )
        if self.status is DataAvailability.AVAILABLE:
            if not has_cost or self.missing_reason is not None:
                raise ValueError("available inter-cluster cost requires one complete best edge")
        elif self.status is DataAvailability.PARTIAL:
            if not has_cost or self.missing_reason is None:
                raise ValueError("partial inter-cluster cost requires cost and missing reason")
        elif has_cost or self.missing_reason is None:
            raise ValueError("missing inter-cluster cost must preserve an unknown cost")
        return self


class SpatialPlanningIssue(ImmutableSpatialModel):
    code: SpatialIssueCode
    anchor_node_id: UUID
    closest_node_id: UUID | None = None
    minimum_known_minutes: int | None = Field(default=None, ge=0, strict=True)
    reason: ShortText
    question: ShortText


class SpatialPlanningResult(ImmutableSpatialModel):
    algorithm_version: str = Field(pattern=r"^[1-9]\d*\.\d+\.\d+$")
    request_id: UUID
    trip_id: UUID
    input_state_version: int = Field(ge=0, strict=True)
    task_book_id: UUID | None = None
    task_book_revision: int | None = Field(default=None, ge=1, strict=True)
    city_id: NonEmptyText
    status: DataAvailability
    anchors: tuple[SpatialAnchor, ...] = Field(min_length=1, max_length=16)
    route_edges: tuple[SpatialRouteEdge, ...]
    clusters: tuple[SpatialActivityCluster, ...] = Field(min_length=1, max_length=16)
    inter_cluster_costs: tuple[InterClusterCost, ...]
    issues: tuple[SpatialPlanningIssue, ...] = ()
    degradation_reasons: tuple[ShortText, ...] = ()
    generated_at: AwareDatetime

    @model_validator(mode="after")
    def result_references_are_a_complete_partition(self) -> SpatialPlanningResult:
        if (self.task_book_id is None) != (self.task_book_revision is None):
            raise ValueError("task-book ID and revision must be supplied together")
        anchor_ids = {anchor.node_id for anchor in self.anchors}
        if len(anchor_ids) != len(self.anchors):
            raise ValueError("spatial result anchor IDs must be unique")
        if len({anchor.place_id for anchor in self.anchors}) != len(self.anchors):
            raise ValueError("spatial result must contain at most one anchor per place")
        expected_edges = len(anchor_ids) * (len(anchor_ids) - 1) // 2
        if len(self.route_edges) != expected_edges:
            raise ValueError("spatial result must preserve every pairwise route edge")
        edge_pairs: set[frozenset[UUID]] = set()
        edge_ids: set[UUID] = set()
        for edge in self.route_edges:
            if {edge.origin_node_id, edge.destination_node_id} - anchor_ids:
                raise ValueError("spatial route edge references an unknown anchor")
            pair = frozenset((edge.origin_node_id, edge.destination_node_id))
            if pair in edge_pairs or edge.edge_id in edge_ids:
                raise ValueError("spatial route edges must be unique")
            edge_pairs.add(pair)
            edge_ids.add(edge.edge_id)
        cluster_ids = {cluster.cluster_id for cluster in self.clusters}
        if len(cluster_ids) != len(self.clusters):
            raise ValueError("spatial cluster IDs must be unique")
        clustered = [node_id for cluster in self.clusters for node_id in cluster.member_node_ids]
        if len(clustered) != len(set(clustered)) or set(clustered) != anchor_ids:
            raise ValueError("spatial clusters must partition every anchor exactly once")
        expected_inter = len(cluster_ids) * (len(cluster_ids) - 1) // 2
        if len(self.inter_cluster_costs) != expected_inter:
            raise ValueError("spatial result must preserve every inter-cluster cost")
        inter_pairs: set[frozenset[UUID]] = set()
        for cost in self.inter_cluster_costs:
            if {cost.origin_cluster_id, cost.destination_cluster_id} - cluster_ids:
                raise ValueError("inter-cluster cost references an unknown cluster")
            pair = frozenset((cost.origin_cluster_id, cost.destination_cluster_id))
            if pair in inter_pairs:
                raise ValueError("inter-cluster costs must be unique")
            inter_pairs.add(pair)
            if cost.best_edge_id is not None and cost.best_edge_id not in edge_ids:
                raise ValueError("inter-cluster cost references an unknown route edge")
        anchors_by_id = {anchor.node_id: anchor for anchor in self.anchors}
        for issue in self.issues:
            anchor = anchors_by_id.get(issue.anchor_node_id)
            if anchor is None or anchor.strength is not SpatialAnchorStrength.STRONG:
                raise ValueError("spatial issue must reference a declared strong anchor")
            if issue.closest_node_id is not None and issue.closest_node_id not in anchor_ids:
                raise ValueError("spatial issue closest node is unknown")
        _unique(self.degradation_reasons, "spatial degradation reasons")
        degraded = bool(self.issues) or any(
            edge.status is not DataAvailability.AVAILABLE for edge in self.route_edges
        )
        if self.status is DataAvailability.AVAILABLE and (degraded or self.degradation_reasons):
            raise ValueError("available spatial result cannot contain degraded data")
        if self.status is DataAvailability.PARTIAL and not (degraded or self.degradation_reasons):
            raise ValueError("partial spatial result requires explicit degradation")
        return self


def _unique(values: Sequence[object], label: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must be unique")


V3_SPATIAL_PLANNING_CONTRACTS: tuple[type[ContractModel], ...] = (
    SpatialPlanningRequest,
    SpatialPlanningResult,
)
