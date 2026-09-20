import type {
  MapRouteSegment,
  MapUpdatePayload,
  PlannerPublishedPlan
} from "../generated/v4/contracts";

const MODE_BY_SCHEDULE = {
  walking: "walk",
  cycling: "bicycle",
  transit: "public_transit",
  driving: "taxi"
} as const;

export function buildV4PublishedPlanDayMap(
  plan: PlannerPublishedPlan,
  dayIndex: number
): MapUpdatePayload | null {
  const day = plan.materialized_schedule.days[dayIndex];
  if (!day) return null;
  const placeIds = new Set([
    day.start_place_id,
    ...(day.activities ?? []).map((activity) => activity.place_id),
    day.end_place_id
  ]);
  const legKeys = new Set(
    (day.transport_legs ?? []).map(
      (leg) =>
        `${leg.origin_place_id}:${leg.destination_place_id}:${MODE_BY_SCHEDULE[leg.mode]}`
    )
  );
  const markersById = new Map(
    (plan.map_projection.markers ?? []).map((marker) => [
      marker.place_id,
      marker
    ])
  );
  return {
    selected_day_index: dayIndex,
    // Storage order is not visit order: start -> visits -> finish.
    markers: [...placeIds].flatMap((id) => {
      const marker = markersById.get(id);
      return marker ? [marker] : [];
    }),
    routes: (plan.map_projection.routes ?? []).filter(
      (route) =>
        placeIds.has(route.from_place_id) &&
        placeIds.has(route.to_place_id) &&
        legKeys.has(routeKey(route))
    )
  };
}

export function v4PublishedPlanRouteNotice(
  plan: PlannerPublishedPlan,
  dayIndex: number
): string | null {
  const day = plan.materialized_schedule.days[dayIndex];
  const legs = day?.transport_legs ?? [];
  if (!day) return null;
  const omittedRouteCount = (plan.validation_observation.issues ?? []).filter(
    (issue) =>
      issue.code === "route_unavailable" &&
      (issue.affected_dates ?? []).includes(day.service_date)
  ).length;
  if (legs.length === 0) {
    return omittedRouteCount > 0
      ? `第 ${dayIndex + 1} 天有 ${omittedRouteCount} 段相邻安排缺少 Provider 路线；地点顺序已保留，真实交通耗时待补充。`
      : null;
  }
  const projected = buildV4PublishedPlanDayMap(plan, dayIndex);
  const available = new Set((projected?.routes ?? []).map(routeKey));
  const missingGeometryCount = legs.filter(
    (leg) =>
      leg.availability !== "available" ||
      !available.has(
        `${leg.origin_place_id}:${leg.destination_place_id}:${MODE_BY_SCHEDULE[leg.mode]}`
      )
  ).length;
  const availableCount = legs.length - missingGeometryCount;
  if (omittedRouteCount > 0) {
    const knownRouteNotice =
      availableCount > 0 ? `已有 ${availableCount} 段路线可绘制；` : "";
    const geometryNotice =
      missingGeometryCount > 0
        ? `另有 ${missingGeometryCount} 段仅保留已核验耗时；`
        : "";
    return `第 ${dayIndex + 1} 天${knownRouteNotice}${geometryNotice}${omittedRouteCount} 段相邻安排缺少 Provider 路线，仅用非事实占位预留时间。`;
  }
  if (missingGeometryCount === 0) return null;
  return availableCount > 0
    ? `第 ${dayIndex + 1} 天已有 ${availableCount} 段路线可绘制，另有 ${missingGeometryCount} 段仅保留耗时。`
    : `第 ${dayIndex + 1} 天路线几何暂不可绘制，已保留地点与核验后的交通耗时。`;
}

function routeKey(route: MapRouteSegment): string {
  return `${route.from_place_id}:${route.to_place_id}:${route.mode}`;
}
