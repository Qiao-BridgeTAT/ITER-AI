import type {
  MapRouteSegment,
  MapUpdatePayload,
  PublishedPlan,
  ScheduledTransport
} from "../generated/contracts";

const MODE_BY_SCHEDULE: Record<
  ScheduledTransport["mode"],
  MapRouteSegment["mode"]
> = {
  walking: "walk",
  cycling: "bicycle",
  transit: "public_transit",
  driving: "taxi"
};

export function buildPublishedPlanDayMap(
  plan: PublishedPlan,
  dayIndex: number
): MapUpdatePayload | null {
  const day = plan.schedule.days[dayIndex];
  if (!day) return null;

  const placeIds = new Set([
    day.start_place_id,
    ...day.activities.map((activity) => activity.place_id),
    day.end_place_id
  ]);
  const legKeys = new Set(
    day.transport_legs.map(
      (leg) =>
        `${leg.origin_place_id}:${leg.destination_place_id}:${MODE_BY_SCHEDULE[leg.mode]}`
    )
  );
  return {
    selected_day_index: dayIndex,
    markers: (plan.map_projection.markers ?? []).filter((marker) =>
      placeIds.has(marker.place_id)
    ),
    routes: (plan.map_projection.routes ?? []).filter(
      (route) =>
        placeIds.has(route.from_place_id) &&
        placeIds.has(route.to_place_id) &&
        legKeys.has(`${route.from_place_id}:${route.to_place_id}:${route.mode}`)
    )
  };
}

export function publishedPlanRouteNotice(
  plan: PublishedPlan,
  dayIndex: number
): string | null {
  const day = plan.schedule.days[dayIndex];
  if (!day || day.transport_legs.length === 0) return null;
  const projected = buildPublishedPlanDayMap(plan, dayIndex);
  const availableKeys = new Set(
    (projected?.routes ?? []).map(
      (route) => `${route.from_place_id}:${route.to_place_id}:${route.mode}`
    )
  );
  const missingCount = day.transport_legs.filter(
    (leg) =>
      leg.availability !== "available" ||
      !availableKeys.has(
        `${leg.origin_place_id}:${leg.destination_place_id}:${MODE_BY_SCHEDULE[leg.mode]}`
      )
  ).length;
  if (missingCount === 0) return null;
  const availableCount = day.transport_legs.length - missingCount;
  return availableCount > 0
    ? `第 ${dayIndex + 1} 天已有 ${availableCount} 段路线可绘制，另有 ${missingCount} 段路线资料暂缺。`
    : `第 ${dayIndex + 1} 天路线暂不可绘制，已保留当天地点。`;
}
