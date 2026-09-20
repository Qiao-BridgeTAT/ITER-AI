"""Dependency-aware V3-42 modification scope planning."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from uuid import NAMESPACE_URL, UUID, uuid5

from backend.agent.semantic_operations import SemanticImpactKind, SemanticTarget
from backend.agent.state_merge import RecomputeDomain, SemanticMergeImpact
from backend.contracts.cost_estimation import CostSubjectKind
from backend.contracts.daily_scheduling import ScheduleActivityKind
from backend.contracts.plan_modification import (
    PendingPlanModification,
    PlanArtifactKind,
    PlanDayDependency,
    PlanDependencyIndex,
)
from backend.contracts.plan_publication import PublishedPlan


class PlanModificationError(ValueError):
    """Raised when a semantic change cannot be safely scoped to the base plan."""


class PlanModificationService:
    def __init__(
        self,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._clock = clock

    def dependency_index(self, plan: PublishedPlan) -> PlanDependencyIndex:
        hotel_ids = {
            line.subject_id
            for day in plan.cost_estimate.days
            for line in day.lines
            if line.subject_kind is CostSubjectKind.HOTEL_NIGHT
        }
        hotel_place_id = next(iter(hotel_ids)) if len(hotel_ids) == 1 else None
        days = tuple(
            PlanDayDependency(
                day_number=index,
                service_date=day.service_date,
                activity_ids=tuple(activity.activity_id for activity in day.activities),
                attraction_place_ids=tuple(
                    activity.place_id
                    for activity in day.activities
                    if activity.kind is ScheduleActivityKind.ATTRACTION
                ),
                restaurant_place_ids=tuple(
                    activity.place_id
                    for activity in day.activities
                    if activity.kind is ScheduleActivityKind.RESTAURANT
                ),
                route_leg_ids=tuple(leg.leg_id for leg in day.transport_legs),
            )
            for index, day in enumerate(plan.schedule.days, start=1)
        )
        return PlanDependencyIndex(
            trip_id=plan.trip_id,
            plan_version_id=plan.plan_version_id,
            city_id=plan.schedule.city_id,
            days=days,
            hotel_place_id=hotel_place_id,
        )

    def plan(
        self,
        base_plan: PublishedPlan,
        impacts: Iterable[SemanticMergeImpact],
        *,
        generation_id: UUID,
        source_message_id: UUID,
        base_state_version: int,
        base_confirmed_version_id: UUID | None,
    ) -> PendingPlanModification | None:
        materialized = tuple(
            impact
            for impact in impacts
            if impact.target
            not in {SemanticTarget.TASK_BOOK_CONFIRMATION, SemanticTarget.PLAN_CONFIRMATION}
        )
        if not materialized:
            return None
        index = self.dependency_index(base_plan)
        known_days = tuple(day.day_number for day in index.days)
        all_attractions = tuple(
            place_id for day in index.days for place_id in day.attraction_place_ids
        )
        global_required = any(impact.global_replan_required for impact in materialized)
        affected_days: set[int] = set(known_days if global_required else ())
        affected_items: set[UUID] = set()
        artifacts: set[PlanArtifactKind] = set()
        domains: set[RecomputeDomain] = set()
        targets: list[SemanticTarget] = []
        scopes = []
        expanded_reasons: list[str] = []
        preserved_hotel = True

        for impact in materialized:
            targets.append(impact.target)
            scopes.append(impact.impact_scope)
            domains.update(impact.recompute_domains)
            artifacts.update(_artifacts_for(impact.target))
            scope = impact.impact_scope
            if global_required or scope.kind is SemanticImpactKind.WHOLE_TRIP:
                affected_days.update(known_days)
            elif scope.kind is SemanticImpactKind.SPECIFIC_DAY:
                assert scope.day_number is not None
                if scope.day_number not in known_days:
                    raise PlanModificationError("the requested day does not exist in the base plan")
                affected_days.add(scope.day_number)
            else:
                assert scope.item_id is not None
                affected_items.add(scope.item_id)
                matched = _days_for_item(index, impact.target, scope.item_id)
                if matched:
                    affected_days.update(matched)
                else:
                    affected_days.update(known_days)
                    expanded_reasons.append(
                        "无法把该项目安全定位到某一天，因此扩大到整趟行程重新核对。"
                    )
            if impact.target is SemanticTarget.LODGING_PREFERENCES:
                preserved_hotel = False
                affected_days.update(known_days)
                artifacts.update(
                    {
                        PlanArtifactKind.TASK_BOOK,
                        PlanArtifactKind.HOTEL,
                        PlanArtifactKind.SCHEDULE_DAY,
                        PlanArtifactKind.ROUTES,
                    }
                )
            if impact.target is SemanticTarget.DINING_PREFERENCES:
                artifacts.add(PlanArtifactKind.MEAL_WINDOW)

        if global_required:
            artifacts.update(set(PlanArtifactKind))
            preserved_hotel = False
        affected = tuple(sorted(affected_days))
        preserved_days = tuple(day for day in known_days if day not in affected_days)
        preserved_attractions = () if global_required else all_attractions
        if SemanticTarget.ATTRACTION_INTENTS in targets:
            preserved_attractions = tuple(
                place_id for place_id in all_attractions if place_id not in affected_items
            )

        changed_summary = _changed_summary(targets, affected, global_required)
        preserved_summary = _preserved_summary(
            preserved_days,
            preserved_hotel,
            bool(preserved_attractions),
            global_required,
        )
        created_at = self._clock()
        if created_at.tzinfo is None or created_at.utcoffset() is None:
            raise PlanModificationError("modification clock must return an aware datetime")
        fingerprint = ":".join(str(impact.operation_id) for impact in materialized)
        return PendingPlanModification(
            modification_id=uuid5(
                NAMESPACE_URL,
                f"iter:v3-42:{base_plan.trip_id}:{base_plan.plan_version_id}:{fingerprint}",
            ),
            trip_id=base_plan.trip_id,
            generation_id=generation_id,
            source_message_id=source_message_id,
            base_state_version=base_state_version,
            base_plan_version_id=base_plan.plan_version_id,
            base_confirmed_version_id=base_confirmed_version_id,
            targets=tuple(targets),
            impact_scopes=tuple(scopes),
            recompute_domains=tuple(sorted(domains, key=str)),
            affected_artifacts=tuple(sorted(artifacts, key=str)),
            affected_day_numbers=affected,
            affected_item_ids=tuple(sorted(affected_items, key=str)),
            preserved_day_numbers=preserved_days,
            preserved_hotel=preserved_hotel,
            preserved_attraction_place_ids=tuple(dict.fromkeys(preserved_attractions)),
            global_replan_required=global_required,
            changed_summary=changed_summary,
            preserved_summary=preserved_summary,
            scope_reason=_scope_reason(targets, global_required),
            expanded_scope_reason="".join(dict.fromkeys(expanded_reasons)) or None,
            dependency_index=index,
            created_at=created_at,
        )


def _days_for_item(
    index: PlanDependencyIndex,
    target: SemanticTarget,
    item_id: UUID,
) -> tuple[int, ...]:
    if target is SemanticTarget.ATTRACTION_INTENTS:
        return tuple(day.day_number for day in index.days if item_id in day.attraction_place_ids)
    if target is SemanticTarget.DINING_PREFERENCES:
        return tuple(day.day_number for day in index.days if item_id in day.restaurant_place_ids)
    if target is SemanticTarget.LODGING_PREFERENCES and item_id == index.hotel_place_id:
        return tuple(day.day_number for day in index.days)
    return ()


def _artifacts_for(target: SemanticTarget) -> set[PlanArtifactKind]:
    mapping = {
        SemanticTarget.DESTINATION: set(PlanArtifactKind),
        SemanticTarget.DATE_RANGE: set(PlanArtifactKind),
        SemanticTarget.ATTRACTION_INTENTS: {
            PlanArtifactKind.CANDIDATES,
            PlanArtifactKind.SCHEDULE_DAY,
            PlanArtifactKind.ROUTES,
            PlanArtifactKind.COST,
            PlanArtifactKind.VALIDATION,
            PlanArtifactKind.MAP,
        },
        SemanticTarget.DINING_PREFERENCES: {
            PlanArtifactKind.CANDIDATES,
            PlanArtifactKind.MEAL_WINDOW,
            PlanArtifactKind.SCHEDULE_DAY,
            PlanArtifactKind.ROUTES,
            PlanArtifactKind.COST,
            PlanArtifactKind.VALIDATION,
            PlanArtifactKind.MAP,
        },
        SemanticTarget.LODGING_PREFERENCES: {
            PlanArtifactKind.PROVIDER_FACTS,
            PlanArtifactKind.LODGING_STRATEGY,
            PlanArtifactKind.HOTEL,
            PlanArtifactKind.TASK_BOOK,
            PlanArtifactKind.SCHEDULE_DAY,
            PlanArtifactKind.ROUTES,
            PlanArtifactKind.COST,
            PlanArtifactKind.VALIDATION,
            PlanArtifactKind.MAP,
        },
        SemanticTarget.TRANSPORT_PREFERENCES: {
            PlanArtifactKind.LODGING_STRATEGY,
            PlanArtifactKind.ROUTES,
            PlanArtifactKind.COST,
            PlanArtifactKind.VALIDATION,
        },
        SemanticTarget.PACE_PREFERENCES: {
            PlanArtifactKind.SCHEDULE_DAY,
            PlanArtifactKind.ROUTES,
            PlanArtifactKind.COST,
            PlanArtifactKind.VALIDATION,
        },
        SemanticTarget.EXPERIENCE_PREFERENCES: {
            PlanArtifactKind.CANDIDATES,
            PlanArtifactKind.SCHEDULE_DAY,
            PlanArtifactKind.VALIDATION,
        },
        SemanticTarget.SPECIAL_CONSTRAINTS: {
            PlanArtifactKind.LODGING_STRATEGY,
            PlanArtifactKind.SCHEDULE_DAY,
            PlanArtifactKind.ROUTES,
            PlanArtifactKind.VALIDATION,
        },
        SemanticTarget.TASK_BOOK_CONFIRMATION: {PlanArtifactKind.TASK_BOOK},
        SemanticTarget.PLAN_CONFIRMATION: {PlanArtifactKind.VALIDATION},
    }
    return mapping[target]


def _changed_summary(
    targets: list[SemanticTarget], affected_days: tuple[int, ...], global_required: bool
) -> str:
    if global_required:
        return "这次修改会重新计算整趟行程的候选、住宿、路线、排程和费用。"
    day_text = "、".join(f"第{day}天" for day in affected_days)
    if targets == [SemanticTarget.PACE_PREFERENCES]:
        return f"这次只重新计算{day_text}的行程负荷、相关路线和费用。"
    if targets == [SemanticTarget.DINING_PREFERENCES]:
        return f"这次只重新计算{day_text}的用餐时段、餐厅及相关路线。"
    if targets == [SemanticTarget.LODGING_PREFERENCES]:
        return "这次会重新选择住宿，并重新计算每天从酒店出发和返回的路线。"
    return f"这次只重新计算{day_text}及其直接依赖的路线、费用和校验结果。"


def _preserved_summary(
    preserved_days: tuple[int, ...],
    preserved_hotel: bool,
    preserves_attractions: bool,
    global_required: bool,
) -> str:
    if global_required:
        return "旧方案会继续保留到新方案准备完成，但目的地或全局条件相关内容不能直接沿用。"
    parts = []
    if preserved_days:
        parts.append("、".join(f"第{day}天" for day in preserved_days))
    if preserved_hotel:
        parts.append("当前酒店")
    if preserves_attractions:
        parts.append("未受影响的景点意愿")
    return "、".join(parts) + "保持不变，旧方案在新方案完成前仍可查看。"


def _scope_reason(targets: list[SemanticTarget], global_required: bool) -> str:
    if global_required:
        return "目的地、日期或整趟旅行偏好会改变多个下游依赖，因此必须全局重算。"
    if SemanticTarget.LODGING_PREFERENCES in targets:
        return "酒店是每天出发和返回的共同边界，因此住宿变化会影响全部酒店路线。"
    return "依赖索引能把修改定位到具体日期或项目，所以只重算直接受影响的部分。"
