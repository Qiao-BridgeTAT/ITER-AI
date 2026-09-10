"""Resolve decision-local keys and assign authoritative Planner artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast
from uuid import uuid4

from backend.agent.planner.decision_contracts import (
    ModelAskDecision,
    ModelDraftDecision,
    ModelEvidenceDecision,
    ModelPlannerDecision,
    ModelStrategyDecision,
)
from backend.agent.planner.workspace import PlannerGuardError, server_id
from backend.contracts.v4.base import require_unique
from backend.contracts.v4.enums import CandidateEntityKind, CommitmentLevel, InteractionStatus
from backend.contracts.v4.plan_change import (
    AlternativeHotelRecommendation,
    BestHotelRecommendation,
    FormalHotelRecommendationSet,
    validate_hotel_recommendations_against_observation,
)
from backend.contracts.v4.planner_decision import (
    AskUserPayload,
    BuildOrUpdateStrategyPayload,
    MaterializeDraftPayload,
    PlannerCompletionAssessment,
    PlannerDecision,
    PlannerInputRefs,
    PlannerResumeContract,
    RequestEvidencePayload,
)
from backend.contracts.v4.planner_draft import (
    CrossClusterSegment,
    DiscardableObject,
    DraftItem,
    ExpectedRouteCost,
    LodgingBaseline,
    UnassignedIntent,
    WorkingItineraryDay,
    WorkingItineraryDraft,
    planning_projection_digest,
)
from backend.contracts.v4.planner_observations import (
    HotelOfferObservation,
    PlannerCapabilityRequest,
    PlannerInteraction,
    PlannerReadinessIssue,
    PlannerValidationIssue,
    SpatialCluster,
    SpatialRouteEdge,
    SpatialRouteEndpoint,
)
from backend.contracts.v4.planner_refs import (
    CandidateRef,
    FixedCommitmentRef,
    PlannerObjectRef,
    PlannerScope,
)
from backend.contracts.v4.planner_strategy import (
    CandidatePoolEntry,
    CandidatePriority,
    DiningPolicy,
    LodgingPolicy,
    NonBlockingAssumption,
    PendingEvidenceNeed,
    PlanningStrategy,
    StrategyAnchorPolicy,
)
from backend.contracts.v4.planner_workspace import PlannerWorkspaceState


@dataclass(frozen=True)
class PlannerReferenceCatalog:
    workspace: PlannerWorkspaceState

    @property
    def candidates(self) -> dict[str, CandidatePoolEntry]:
        return {
            f"c{index}": entry
            for index, entry in enumerate(self.workspace.candidate_pool.candidates, 1)
        }

    @property
    def fixed(self) -> dict[str, FixedCommitmentRef]:
        return {
            f"f{index}": reference
            for index, reference in enumerate(self.workspace.candidate_pool.fixed_commitments, 1)
        }

    @property
    def clusters(self) -> dict[str, SpatialCluster]:
        observation = self.workspace.spatial_observation
        return (
            {f"g{index}": cluster for index, cluster in enumerate(observation.clusters, 1)}
            if observation
            else {}
        )

    @property
    def routes(self) -> dict[str, SpatialRouteEdge]:
        observation = self.workspace.spatial_observation
        return (
            {f"r{index}": edge for index, edge in enumerate(observation.route_edges, 1)}
            if observation
            else {}
        )

    @property
    def hotels(self) -> dict[str, HotelOfferObservation]:
        observation = self.workspace.hotel_observation
        return (
            {f"h{index}": offer for index, offer in enumerate(observation.offers, 1)}
            if observation
            else {}
        )

    @property
    def current_hotel_key(self) -> str | None:
        """Return the selected baseline's local h key, when it is still current."""

        draft = self.workspace.working_itinerary
        if draft is None or draft.lodging_baseline.mode != "selected_offer":
            return None
        selected = draft.lodging_baseline.selected_offer_ref
        return next(
            (key for key, offer in self.hotels.items() if offer.offer_ref == selected),
            None,
        )

    @property
    def alternative_hotels(self) -> dict[str, HotelOfferObservation]:
        """Expose usable replacement hotels without offering the current baseline."""

        current_key = self.current_hotel_key
        return {
            key: offer
            for key, offer in self.hotels.items()
            if key != current_key and offer.availability_status != "unavailable"
        }

    @property
    def issues(self) -> dict[str, PlannerReadinessIssue]:
        observation = self.workspace.readiness_observation
        return (
            {f"b{index}": issue for index, issue in enumerate(observation.issues, 1)}
            if observation
            else {}
        )

    @property
    def validation_issues(self) -> dict[str, PlannerValidationIssue]:
        observation = self.workspace.validation_observation
        return (
            {f"v{index}": issue for index, issue in enumerate(observation.issues, 1)}
            if observation
            else {}
        )

    @property
    def evidence(self) -> dict[str, str]:
        result = {"pool": self.workspace.candidate_pool.candidate_pool_id}
        if self.workspace.spatial_observation:
            result["spatial"] = self.workspace.spatial_observation.observation_id
        if self.workspace.planning_strategy:
            result["strategy"] = self.workspace.planning_strategy.strategy_id
        if self.workspace.hotel_observation:
            result["hotel"] = self.workspace.hotel_observation.hotel_observation_id
        result.update(
            {
                f"e{index}": fact.fact_reference_id
                for index, fact in enumerate(self.workspace.verified_facts, 1)
            }
        )
        result.update(
            {
                f"obs{index}": observation.observation_id
                for index, observation in enumerate(self.workspace.capability_observations, 1)
            }
        )
        for key, candidate in self.candidates.items():
            for index, reference in enumerate(candidate.source_intent_refs, 1):
                result[f"intent:{key}:{index}"] = reference
            for index, reference in enumerate(candidate.fact_reference_ids, 1):
                result[f"fact:{key}:{index}"] = reference
        result.update(
            {
                f"comparison{index}": item.observation_id
                for index, item in enumerate(self.workspace.route_comparisons, 1)
            }
        )
        return result

    def candidate(self, key: str, *, field: str = "candidate_key") -> CandidateRef:
        entry = self.candidates.get(key)
        if entry is None or entry.selection_permission == "forbidden":
            raise PlannerGuardError(
                f"planner_unknown_or_forbidden_candidate_key:{field}:use_current_c_keys_only"
            )
        return entry.candidate_ref

    def object(self, key: str, *, field: str = "object_key") -> PlannerObjectRef:
        if key in self.fixed:
            return self.fixed[key]
        return self.candidate(key, field=field)

    def cluster_id(self, key: str) -> str:
        cluster = self.clusters.get(key)
        if cluster is None:
            raise PlannerGuardError("planner_unknown_cluster_key")
        return cluster.cluster_id

    def hotel(self, key: str, *, field: str) -> HotelOfferObservation:
        offer = self.hotels.get(key)
        if offer is None:
            allowed = ",".join(self.hotels) or "none_query_hotels_first"
            raise PlannerGuardError(f"planner_unknown_hotel_key:{field}:allowed_keys={allowed}")
        return offer

    def alternative_hotel(self, key: str) -> HotelOfferObservation:
        """Resolve a real replacement or return precise, executable Guard feedback."""

        offer = self.alternative_hotels.get(key)
        allowed = ",".join(self.alternative_hotels)
        suffix = f":allowed_values={allowed}" if allowed else ""
        if key == self.current_hotel_key:
            raise PlannerGuardError(
                f"planner_repair_no_effective_change:path=choice.hotel_offer_key{suffix}"
            )
        known_offer = self.hotels.get(key)
        if known_offer is not None and known_offer.availability_status == "unavailable":
            raise PlannerGuardError("planner_repair_hotel_offer_unavailable")
        if offer is None:
            raise PlannerGuardError(
                f"planner_unknown_hotel_key:path=choice.hotel_offer_key{suffix}"
            )
        return offer

    def evidence_id(self, key: str, *, field: str = "evidence_keys") -> str:
        value = self.evidence.get(key)
        if value is None:
            raise PlannerGuardError(
                f"planner_unknown_evidence_key:{field}:use_current_evidence_keys"
            )
        return value

    def endpoint(self, key: str) -> SpatialRouteEndpoint:
        if key in self.candidates:
            return SpatialRouteEndpoint(
                kind="candidate", reference_id=self.candidate(key).candidate_id
            )
        if key in self.clusters:
            return SpatialRouteEndpoint(kind="cluster", reference_id=self.clusters[key].cluster_id)
        if key in self.fixed:
            return SpatialRouteEndpoint(
                kind="fixed_commitment", reference_id=self.fixed[key].commitment_id
            )
        if key in self.hotels:
            return SpatialRouteEndpoint(
                kind="hotel_offer", reference_id=self.hotels[key].offer_ref.offer_id
            )
        raise PlannerGuardError("planner_unknown_route_endpoint")


def resolve_model_decision(
    proposal: ModelPlannerDecision, workspace: PlannerWorkspaceState
) -> PlannerDecision:
    model = proposal.root
    catalog = PlannerReferenceCatalog(workspace)
    decision_id = str(uuid4())
    strategy = workspace.planning_strategy
    input_refs = PlannerInputRefs(
        strategy_revision=strategy.strategy_revision if strategy else None,
        candidate_pool_revision=workspace.candidate_pool.revision,
        draft_revision=workspace.working_itinerary.draft_revision
        if workspace.working_itinerary
        else None,
    )
    output_scope = workspace.current_scope.model_copy(
        update={"workspace_revision": workspace.workspace_revision + 1}
    )
    payload: (
        BuildOrUpdateStrategyPayload
        | RequestEvidencePayload
        | MaterializeDraftPayload
        | AskUserPayload
    )
    if isinstance(model, ModelStrategyDecision):
        data = model.strategy
        if (model.mode == "initialize") != (strategy is None):
            raise PlannerGuardError("planner_strategy_mode_conflict")
        if model.mode == "replace" and (
            strategy is None or model.base_strategy_revision != strategy.strategy_revision
        ):
            raise PlannerGuardError("planner_stale_strategy_base")
        previous_revisions = [
            decision.payload.proposed_strategy.strategy_revision
            for decision in workspace.decision_trace
            if isinstance(decision.payload, BuildOrUpdateStrategyPayload)
        ]
        lodging = data.lodging_policy
        if (
            lodging.fixed_commitment_key is not None
            and lodging.fixed_commitment_key not in catalog.fixed
        ):
            raise PlannerGuardError("planner_unknown_fixed_commitment_key")
        current_strategy = PlanningStrategy(
            strategy_id=server_id(workspace.generation_id, "strategy"),
            strategy_revision=max(previous_revisions, default=0) + 1,
            scope=output_scope,
            candidate_pool_revision=workspace.candidate_pool.revision,
            core_experience_summary=data.core_experience_summary,
            anchor_policy=StrategyAnchorPolicy(
                immutable_refs=tuple(workspace.candidate_pool.fixed_commitments),
                strong_candidate_refs=tuple(
                    entry.candidate_ref
                    for entry in workspace.candidate_pool.candidates
                    if entry.commitment_level is CommitmentLevel.STRONG
                ),
                soft_candidate_refs=tuple(
                    entry.candidate_ref
                    for entry in workspace.candidate_pool.candidates
                    if entry.commitment_level is CommitmentLevel.SOFT
                ),
                filler_candidate_refs=tuple(
                    entry.candidate_ref
                    for entry in workspace.candidate_pool.candidates
                    if entry.commitment_level is CommitmentLevel.FILLER
                ),
            ),
            candidate_priority=tuple(
                CandidatePriority(
                    candidate_ref=catalog.candidate(
                        item.candidate_key, field="strategy.candidate_priority"
                    ),
                    priority_band=item.priority_band,
                    reason_code=item.reason_code,
                )
                for item in data.candidate_priority
            ),
            daily_capacity_policy=data.daily_capacity_policy,
            spatial_policy=data.spatial_policy,
            lodging_policy=LodgingPolicy(
                **lodging.model_dump(exclude={"fixed_commitment_key"}),
                fixed_commitment_ref=catalog.fixed[lodging.fixed_commitment_key]
                if lodging.fixed_commitment_key
                else None,
            ),
            dining_policy=DiningPolicy(
                **data.dining_policy.model_dump(),
                destination_restaurant_refs=tuple(
                    entry.candidate_ref
                    for entry in workspace.candidate_pool.candidates
                    if entry.entity_kind is CandidateEntityKind.RESTAURANT
                    and entry.commitment_level is CommitmentLevel.STRONG
                ),
            ),
            conflict_policy=data.conflict_policy,
            pending_evidence=tuple(
                PendingEvidenceNeed(
                    evidence_need_id=server_id(decision_id, item.local_key),
                    capability=item.capability,
                    target_refs=tuple(
                        catalog.object(key, field="strategy.pending_evidence.target_keys")
                        for key in item.target_keys
                    ),
                    affected_dates=item.affected_dates,
                    blocking=item.blocking,
                )
                for item in data.pending_evidence
            ),
            non_blocking_assumptions=tuple(
                NonBlockingAssumption(
                    assumption_id=server_id(decision_id, item.local_key),
                    summary=item.summary,
                    source_ref=catalog.evidence.get(item.source_ref, item.source_ref),
                )
                for item in data.non_blocking_assumptions
            ),
            reason_summary=data.reason_summary,
        )
        if strategy is not None and _strategy_revision_projection(
            current_strategy
        ) == _strategy_revision_projection(strategy):
            raise PlannerGuardError(
                "planner_strategy_no_effective_change:request_evidence_or_materialize_or_ask_user"
            )
        payload = BuildOrUpdateStrategyPayload.model_validate(
            {
                "mode": model.mode,
                "proposed_strategy": current_strategy,
                **(
                    {"base_strategy_revision": model.base_strategy_revision}
                    if model.mode == "replace"
                    else {}
                ),
            }
        )
    elif isinstance(model, ModelEvidenceDecision):
        require_unique([item.local_key for item in model.requests], "model capability local keys")
        requests = []
        for item in model.requests:
            if item.purpose == "resolve_validation_issue" and not item.based_on_issue_keys:
                raise PlannerGuardError("planner_issue_resolution_requires_current_b_keys")
            if item.purpose != "resolve_validation_issue" and item.based_on_issue_keys:
                raise PlannerGuardError(
                    "planner_issue_keys_require_resolve_validation_issue_purpose"
                )
            if any(key not in catalog.issues for key in item.based_on_issue_keys):
                raise PlannerGuardError("planner_unknown_readiness_issue_key")
            args = item.arguments.model_dump(mode="json")
            capability = args.pop("capability")
            for field, target in (
                ("candidate_keys", "candidate_refs"),
                ("nearby_candidate_keys", "nearby_candidate_refs"),
            ):
                if field in args:
                    args[target] = [
                        catalog.candidate(key, field=f"requests.arguments.{field}").model_dump(
                            mode="json"
                        )
                        for key in args.pop(field)
                    ]
            for field, target in (
                ("nearby_cluster_keys", "nearby_cluster_refs"),
                ("activity_cluster_keys", "activity_cluster_refs"),
            ):
                if field in args:
                    args[target] = [catalog.cluster_id(key) for key in args.pop(field)]
            if "offer_key" in args:
                offer = catalog.hotel(args.pop("offer_key"), field="requests.arguments.offer_key")
                args["offer_ref"] = offer.offer_ref.model_dump(mode="json")
            if "endpoint_pairs" in args:
                args["endpoint_pairs"] = [
                    {
                        "origin": catalog.endpoint(pair["origin_key"]),
                        "destination": catalog.endpoint(pair["destination_key"]),
                    }
                    for pair in args["endpoint_pairs"]
                ]
                if args.get("comparison"):
                    for field in ("baseline_days", "proposed_days"):
                        for day in args["comparison"][field]:
                            day["ordered_endpoints"] = [
                                catalog.endpoint(key).model_dump(mode="json")
                                for key in day.pop("ordered_endpoint_keys")
                            ]
            requests.append(
                PlannerCapabilityRequest.model_validate(
                    {
                        "request_id": server_id(decision_id, item.local_key),
                        "scope": workspace.current_scope,
                        "capability": capability,
                        "purpose": item.purpose,
                        "blocking": item.blocking,
                        "based_on_issue_ids": [
                            catalog.issues[key].issue_id for key in item.based_on_issue_keys
                        ],
                        "service_dates": args.get("service_dates", ()),
                        "arguments": args,
                    }
                )
            )
        payload = RequestEvidencePayload(
            capability_requests=tuple(requests), resume_goal=model.resume_goal
        )
        if any(item.based_on_issue_ids for item in requests) and workspace.readiness_observation:
            input_refs = input_refs.model_copy(
                update={"readiness_observation_id": workspace.readiness_observation.observation_id}
            )
    elif isinstance(model, ModelDraftDecision):
        if (
            strategy is None
            or workspace.spatial_observation is None
            or workspace.working_itinerary is not None
        ):
            raise PlannerGuardError("planner_initial_draft_prerequisites_missing")
        draft_data = model.draft
        object_keys = [item.object_key for day in draft_data.days for item in day.ordered_items]
        seen_object_keys: set[str] = set()
        duplicate_object_keys: list[str] = []
        for key in object_keys:
            if key in seen_object_keys and key not in duplicate_object_keys:
                duplicate_object_keys.append(key)
            seen_object_keys.add(key)
        if duplicate_object_keys:
            raise PlannerGuardError(
                "planner_draft_duplicate_objects:keys=" + ",".join(duplicate_object_keys)
            )
        unassigned_candidate_keys = [item.candidate_key for item in draft_data.unassigned_intents]
        assigned_and_unassigned = sorted(set(object_keys) & set(unassigned_candidate_keys))
        if assigned_and_unassigned:
            raise PlannerGuardError(
                "planner_candidate_assigned_and_unassigned:keys="
                + ",".join(assigned_and_unassigned)
            )
        allowed_unassigned_keys = [
            key
            for key, entry in catalog.candidates.items()
            if entry.commitment_level in {CommitmentLevel.STRONG, CommitmentLevel.SOFT}
        ]
        for unassigned in draft_data.unassigned_intents:
            entry = catalog.candidates.get(unassigned.candidate_key)
            if entry is None or entry.selection_permission == "forbidden":
                raise PlannerGuardError(
                    "planner_unknown_or_forbidden_candidate_key:"
                    "draft.unassigned_intents.candidate_key:use_current_c_keys_only"
                )
            if entry.commitment_level not in {
                CommitmentLevel.STRONG,
                CommitmentLevel.SOFT,
            }:
                raise PlannerGuardError(
                    "planner_unassigned_candidate_not_protected:"
                    f"candidate_key={unassigned.candidate_key}:"
                    f"commitment={entry.commitment_level.value}:"
                    f"allowed_keys={','.join(allowed_unassigned_keys) or 'none'}"
                )
        item_ids = {key: server_id(decision_id, "item", key) for key in object_keys}
        days = []
        for day_index, day in enumerate(draft_data.days):
            primary_cluster_id = (
                catalog.cluster_id(day.primary_cluster_key) if day.primary_cluster_key else None
            )
            items = []
            for index, draft_item in enumerate(day.ordered_items):
                reference = catalog.object(
                    draft_item.object_key,
                    field=f"draft.days.{day_index}.ordered_items.{index}.object_key",
                )
                entry = catalog.candidates.get(draft_item.object_key)
                items.append(
                    DraftItem(
                        draft_item_id=item_ids[draft_item.object_key],
                        position=index,
                        item_kind=draft_item.item_kind,
                        object_ref=reference,
                        cluster_id=entry.cluster_ids[0] if entry and entry.cluster_ids else None,
                        expected_window=draft_item.expected_window,
                        meal_slot=draft_item.meal_slot,
                        duration_preference=draft_item.duration_preference,
                        commitment_level=cast(
                            Literal["immutable", "strong", "soft", "filler", "neutral"],
                            entry.commitment_level.value if entry else "immutable",
                        ),
                    )
                )
            expected_coverage = {
                proposed.object_key
                for proposed, resolved in zip(day.ordered_items, items, strict=True)
                if resolved.cluster_id is not None and resolved.cluster_id != primary_cluster_id
            }
            submitted_coverage = [
                key for segment in day.cross_cluster_segments for key in segment.covered_object_keys
            ]
            if set(submitted_coverage) != expected_coverage or len(submitted_coverage) != len(
                set(submitted_coverage)
            ):
                # Explain the failed relation using only current catalog keys,
                # never arbitrary rejected free text.
                required_candidates = ",".join(
                    item.object_key
                    for item in day.ordered_items
                    if item.object_key in expected_coverage
                    and item.object_key in catalog.candidates
                )
                raise PlannerGuardError(
                    f"planner_cross_cluster_coverage:day_index={day_index}:"
                    f"must_cover_candidates={required_candidates or 'none'}:"
                    "exactly_once_and_only_non_primary_items"
                )
            segments = []
            for segment_index, segment in enumerate(day.cross_cluster_segments):
                from_cluster_id = catalog.cluster_id(segment.from_cluster_key)
                to_cluster_id = catalog.cluster_id(segment.to_cluster_key)
                if from_cluster_id == to_cluster_id:
                    raise PlannerGuardError(
                        "planner_cross_cluster_requires_distinct_clusters:"
                        f"day_index={day_index}:segment_index={segment_index}:"
                        f"cluster_key={segment.from_cluster_key}"
                    )
                if primary_cluster_id not in {from_cluster_id, to_cluster_id}:
                    raise PlannerGuardError(
                        "planner_cross_cluster_must_connect_primary:"
                        f"day_index={day_index}:segment_index={segment_index}:"
                        f"primary={day.primary_cluster_key or 'none'}:"
                        f"from={segment.from_cluster_key}:to={segment.to_cluster_key}"
                    )
                if any(key not in catalog.routes for key in segment.route_edge_keys):
                    raise PlannerGuardError("planner_unknown_cross_cluster_route_key")
                if any(key not in item_ids for key in segment.covered_object_keys):
                    raise PlannerGuardError("planner_unknown_cross_cluster_item_key")
                routes = [catalog.routes[key] for key in segment.route_edge_keys]
                route_keys_by_pair: dict[tuple[str, str, str, str], list[str]] = {}
                for route_key, edge in zip(segment.route_edge_keys, routes, strict=True):
                    pair = (
                        edge.origin.kind,
                        edge.origin.reference_id,
                        edge.destination.kind,
                        edge.destination.reference_id,
                    )
                    route_keys_by_pair.setdefault(pair, []).append(route_key)
                duplicate_alternatives = next(
                    (keys for keys in route_keys_by_pair.values() if len(keys) > 1), None
                )
                if duplicate_alternatives:
                    raise PlannerGuardError(
                        "planner_cross_cluster_duplicate_route_alternative:"
                        f"day_index={day_index}:segment_index={segment_index}:"
                        "choose_exactly_one_of=" + ",".join(duplicate_alternatives)
                    )
                if any(edge.duration_minutes is None for edge in routes):
                    raise PlannerGuardError("planner_cross_cluster_route_missing")
                segments.append(
                    CrossClusterSegment(
                        from_cluster_id=from_cluster_id,
                        to_cluster_id=to_cluster_id,
                        covered_item_ids=tuple(
                            item_ids[key] for key in segment.covered_object_keys
                        ),
                        reason_code=segment.reason_code,
                        supporting_intent_or_fact_refs=tuple(
                            catalog.evidence_id(
                                key,
                                field=f"draft.days.{day_index}.cross_cluster_segments.{segment_index}.supporting_evidence_keys",
                            )
                            for key in segment.supporting_evidence_keys
                        ),
                        route_edge_ids=tuple(edge.route_edge_id for edge in routes),
                        expected_route_cost=ExpectedRouteCost(
                            duration_minutes=sum(edge.duration_minutes or 0 for edge in routes),
                            distance_meters=sum(edge.distance_meters or 0 for edge in routes)
                            if all(edge.distance_meters is not None for edge in routes)
                            else None,
                        ),
                        comparison_observation_ref=catalog.evidence_id(
                            segment.comparison_observation_key,
                            field=f"draft.days.{day_index}.cross_cluster_segments.{segment_index}.comparison_observation_key",
                        )
                        if segment.comparison_observation_key
                        else None,
                    )
                )
            days.append(
                WorkingItineraryDay(
                    service_date=day.service_date,
                    day_kind=day.day_kind,
                    day_theme=day.day_theme,
                    primary_cluster_id=primary_cluster_id,
                    ordered_items=tuple(items),
                    cross_cluster_segments=tuple(segments),
                    dining_goals=day.dining_goals,
                    transport_preferences=day.transport_preferences,
                )
            )
        baseline = draft_data.lodging_baseline
        if any(item.object_key not in item_ids for item in draft_data.discardable_objects):
            raise PlannerGuardError("planner_unknown_discardable_item_key")
        hotel_recommendations: FormalHotelRecommendationSet | None = None
        if strategy.lodging_policy.mode == "not_applicable":
            if baseline.selected_offer_key is not None:
                raise PlannerGuardError("planner_day_trip_cannot_select_hotel")
            compiled_lodging = LodgingBaseline(mode="not_applicable")
        elif strategy.lodging_policy.mode == "fixed":
            if baseline.selected_offer_key is not None:
                raise PlannerGuardError("planner_fixed_hotel_changed")
            compiled_lodging = LodgingBaseline(
                mode="fixed",
                fixed_commitment_ref=strategy.lodging_policy.fixed_commitment_ref,
            )
        else:
            if baseline.selected_offer_key is None:
                raise PlannerGuardError("planner_draft_hotel_selection_required")
            if (
                baseline.better_value_offer_key is None
                or baseline.alternative_experience_offer_key is None
            ):
                raise PlannerGuardError("planner_draft_hotel_alternatives_required")
            compiled_lodging = LodgingBaseline(
                mode="selected_offer",
                selected_offer_ref=catalog.hotel(
                    baseline.selected_offer_key,
                    field="draft.lodging_baseline.selected_offer_key",
                ).offer_ref,
            )
            hotel_recommendations = compile_hotel_recommendations(
                workspace,
                output_scope=output_scope,
                best_key=baseline.selected_offer_key,
                better_value_key=baseline.better_value_offer_key,
                alternative_experience_key=baseline.alternative_experience_offer_key,
                reason_summary=draft_data.reason_summary,
            )
        draft = WorkingItineraryDraft(
            draft_id=server_id(workspace.generation_id, "draft"),
            draft_revision=1,
            content_digest="0" * 64,
            scope=output_scope,
            based_on_strategy_revision=strategy.strategy_revision,
            candidate_pool_revision=workspace.candidate_pool.revision,
            spatial_observation_id=workspace.spatial_observation.observation_id,
            hotel_observation_id=workspace.hotel_observation.hotel_observation_id
            if workspace.hotel_observation and compiled_lodging.mode != "not_applicable"
            else None,
            lodging_baseline=compiled_lodging,
            days=tuple(days),
            discardable_objects=tuple(
                DiscardableObject(
                    draft_item_id=item_ids[item.object_key],
                    mode=item.mode,
                    trigger_codes=item.trigger_codes,
                    discard_rank=item.discard_rank,
                    authorization_ref=catalog.evidence_id(
                        item.authorization_key, field="draft.discardable_objects.authorization_key"
                    ),
                    reason_summary=item.reason_summary,
                )
                for item in draft_data.discardable_objects
            ),
            unassigned_intents=tuple(
                UnassignedIntent(
                    candidate_ref=catalog.candidate(
                        item.candidate_key, field="draft.unassigned_intents.candidate_key"
                    ),
                    commitment_level=cast(
                        Literal["strong", "soft"],
                        catalog.candidates[item.candidate_key].commitment_level.value,
                    ),
                    reason_code=item.reason_code,
                    supporting_observation_refs=tuple(
                        catalog.evidence_id(key, field="draft.unassigned_intents.observation_keys")
                        for key in item.observation_keys
                    ),
                    requires_user_resolution=item.requires_user_resolution,
                )
                for item in draft_data.unassigned_intents
            ),
            reason_summary=draft_data.reason_summary,
        )
        draft = draft.model_copy(update={"content_digest": planning_projection_digest(draft)})
        payload = MaterializeDraftPayload(
            proposed_working_draft=draft,
            hotel_recommendations=hotel_recommendations,
            declared_affected_dates=tuple(day.service_date for day in draft.days),
        )
    else:
        assert isinstance(model, ModelAskDecision)
        require_unique(model.issue_keys, "model ask-user issues")
        if any(key not in catalog.issues for key in model.issue_keys):
            raise PlannerGuardError("planner_unknown_ask_user_issue_key")
        issues = [catalog.issues[key] for key in model.issue_keys]
        if (
            not issues
            or any(not issue.user_authority_required for issue in issues)
            or len({issue.ask_user_reason for issue in issues}) != 1
        ):
            raise PlannerGuardError("planner_ask_user_not_authorized")
        observation = workspace.readiness_observation
        if observation is None:
            raise PlannerGuardError("planner_readiness_observation_missing")
        assert issues[0].ask_user_reason is not None
        interaction = PlannerInteraction(
            interaction_id=str(uuid4()),
            scope=output_scope,
            reason_code=issues[0].ask_user_reason,
            issue_ids=tuple(issue.issue_id for issue in issues),
            decision_scope="global",
            affected_dates=tuple(sorted({day for issue in issues for day in issue.affected_dates})),
            option_contracts=tuple(option for issue in issues for option in issue.option_contracts),
            allow_free_text=True,
            based_on_workspace_revision=output_scope.workspace_revision,
            resume_token=str(uuid4()),
            status=InteractionStatus.ACTIVE,
        )
        input_refs = input_refs.model_copy(
            update={"readiness_observation_id": observation.observation_id}
        )
        payload = AskUserPayload(
            user_decision_request=interaction,
            blocking_issue_ids=interaction.issue_ids,
            resume_contract=PlannerResumeContract(
                resume_token=interaction.resume_token,
                resume_goal=model.current_goal,
                expected_next_actions=("build_or_update_strategy", "request_evidence"),
            ),
        )
    blockers = (
        payload.blocking_issue_ids
        if isinstance(payload, AskUserPayload)
        else tuple(
            dict.fromkeys(
                (*model.remaining_blockers, "materialization_pending", "validation_pending")
            )
        )
    )
    return PlannerDecision(
        decision_id=decision_id,
        scope=workspace.current_scope,
        action=model.action,
        current_goal=model.current_goal,
        reason_summary=model.reason_summary,
        input_refs=input_refs,
        completion_assessment=PlannerCompletionAssessment(
            ready_to_finalize=False, blocking_issue_ids=blockers
        ),
        payload=payload,
    )


def _strategy_revision_projection(strategy: PlanningStrategy) -> dict[str, Any]:
    projection = strategy.model_dump(
        mode="json", exclude={"strategy_id", "strategy_revision", "scope", "reason_summary"}
    )
    for item in projection["pending_evidence"]:
        item.pop("evidence_need_id")
    for item in projection["non_blocking_assumptions"]:
        item.pop("assumption_id")
    return projection


def compile_hotel_recommendations(
    workspace: PlannerWorkspaceState,
    *,
    output_scope: PlannerScope,
    best_key: str,
    better_value_key: str,
    alternative_experience_key: str,
    reason_summary: str,
) -> FormalHotelRecommendationSet:
    observation = workspace.hotel_observation
    if observation is None or observation.mode != "search":
        raise PlannerGuardError("planner_draft_hotel_observation_missing")
    catalog = PlannerReferenceCatalog(workspace)
    best = catalog.hotel(best_key, field="draft.lodging_baseline.selected_offer_key")
    better_value = catalog.hotel(
        better_value_key,
        field="draft.lodging_baseline.better_value_offer_key",
    )
    alternative = catalog.hotel(
        alternative_experience_key,
        field="draft.lodging_baseline.alternative_experience_offer_key",
    )
    offers = (best, better_value, alternative)
    if any(offer.availability_status not in {"available", "limited"} for offer in offers):
        raise PlannerGuardError("planner_draft_hotel_offer_unavailable")
    if len({offer.offer_ref.offer_id for offer in offers}) != 3:
        raise PlannerGuardError("planner_draft_hotel_roles_must_be_distinct")
    if not _is_value_alternative(best, better_value):
        raise PlannerGuardError("planner_draft_value_hotel_not_differentiated")
    if not _is_experience_alternative(best, alternative):
        raise PlannerGuardError("planner_draft_experience_hotel_not_differentiated")
    recommendations = FormalHotelRecommendationSet(
        recommendation_set_id=server_id(
            workspace.generation_id,
            observation.hotel_observation_id,
            "hotel-recommendations",
        ),
        scope=output_scope,
        hotel_observation_id=observation.hotel_observation_id,
        recommended_hotel=BestHotelRecommendation(
            hotel_offer_ref=best.offer_ref,
            area_reason="位于已核验住宿区位，并与本次主要活动簇相匹配。",
            route_fit=_hotel_route_summary(best),
            quality_and_price_fit=_hotel_quality_price_summary(best),
            main_tradeoff=_hotel_tradeoff(best),
        ),
        alternative_hotels=(
            AlternativeHotelRecommendation(
                role="better_value",
                hotel_offer_ref=better_value.offer_ref,
                difference_from_best=_value_difference(best, better_value),
            ),
            AlternativeHotelRecommendation(
                role="alternative_location_or_experience",
                hotel_offer_ref=alternative.offer_ref,
                difference_from_best=_experience_difference(best, alternative),
            ),
        ),
        reason_summary=reason_summary,
    )
    validate_hotel_recommendations_against_observation(recommendations, observation)
    return recommendations


def _is_value_alternative(
    best: HotelOfferObservation,
    alternative: HotelOfferObservation,
) -> bool:
    if (
        best.total_price is not None
        and alternative.total_price is not None
        and best.total_price.currency == alternative.total_price.currency
        and alternative.total_price.amount_minor < best.total_price.amount_minor
    ):
        return True
    return (
        alternative.quality_feature_refs != best.quality_feature_refs
        or alternative.facility_feature_refs != best.facility_feature_refs
    )


def _is_experience_alternative(
    best: HotelOfferObservation,
    alternative: HotelOfferObservation,
) -> bool:
    return (
        alternative.area_ref != best.area_ref
        or alternative.quality_feature_refs != best.quality_feature_refs
        or alternative.facility_feature_refs != best.facility_feature_refs
    )


def _hotel_route_summary(offer: HotelOfferObservation) -> str:
    durations = [
        item.duration_minutes
        for item in offer.commute_to_clusters
        if item.duration_minutes is not None
    ]
    if not durations:
        return "当前通勤时长仍有缺口，正式发布前必须继续校验。"
    average = round(sum(durations) / len(durations))
    return f"已核验 {len(durations)} 个活动簇，平均通勤约 {average} 分钟。"


def _hotel_quality_price_summary(offer: HotelOfferObservation) -> str:
    if offer.total_price is None:
        price = "价格暂缺"
    else:
        price = (
            f"当前总价约 {offer.total_price.amount_minor / 100:.0f} {offer.total_price.currency}"
        )
    return (
        f"{price}；已核验 {len(offer.quality_feature_refs)} 项品质依据和 "
        f"{len(offer.facility_feature_refs)} 项设施依据。"
    )


def _hotel_tradeoff(offer: HotelOfferObservation) -> str:
    if offer.availability_status == "limited":
        return "库存有限，发布前需刷新当前报价与库存。"
    if offer.price_missing:
        return "当前价格缺失，预算覆盖仍不完整。"
    return "最终体验仍取决于房型和到店实际情况。"


def _value_difference(
    best: HotelOfferObservation,
    alternative: HotelOfferObservation,
) -> str:
    if (
        best.total_price is not None
        and alternative.total_price is not None
        and best.total_price.currency == alternative.total_price.currency
    ):
        difference = best.total_price.amount_minor - alternative.total_price.amount_minor
        if difference > 0:
            return f"同一住宿周期当前总价低约 {difference / 100:.0f} 元。"
    return "价格、品质或设施组合与综合最佳项不同，可作为性价比取舍。"


def _experience_difference(
    best: HotelOfferObservation,
    alternative: HotelOfferObservation,
) -> str:
    if best.area_ref != alternative.area_ref:
        return "位于不同的已核验住宿区位，可换取不同街区与通勤体验。"
    return "品质或设施组合不同，可换取另一种住宿体验。"
