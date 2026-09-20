import type { TripListItem } from "../generated/contracts";

export interface DisplayTrip {
  tripId: string;
  title: string;
  phase: TripListItem["phase"];
  updatedLabel: string;
  current: boolean;
}

const MAX_VISIBLE_TRIPS = 4;

function timestamp(value: string): number {
  const parsed = new Date(value).getTime();
  return Number.isNaN(parsed) ? Number.NEGATIVE_INFINITY : parsed;
}

function updatedLabel(updatedAt: string, now = new Date()): string {
  const updatedTime = timestamp(updatedAt);
  if (!Number.isFinite(updatedTime)) return "更新时间未知";

  const updated = new Date(updatedTime);
  const elapsedDays = Math.floor(
    Math.max(0, now.getTime() - updatedTime) / 86_400_000
  );
  if (elapsedDays === 0) return "今天更新";
  if (elapsedDays === 1) return "昨天更新";
  if (elapsedDays <= 30) return `${elapsedDays}天前`;
  return `${updated.getMonth() + 1}月${updated.getDate()}日更新`;
}

export function tripRows(
  history: readonly TripListItem[],
  currentTripId: string,
  limit = MAX_VISIBLE_TRIPS
): DisplayTrip[] {
  const sorted = [...history].sort(
    (left, right) => timestamp(right.updated_at) - timestamp(left.updated_at)
  );
  const current = sorted.find((trip) => trip.trip_id === currentTripId);
  const visible = sorted.slice(0, limit);
  if (
    current !== undefined &&
    !visible.some((trip) => trip.trip_id === currentTripId)
  ) {
    visible.splice(limit - 1, 1, current);
  }
  return visible.map((trip) => ({
    tripId: trip.trip_id,
    title: trip.title,
    phase: trip.phase,
    updatedLabel: updatedLabel(trip.updated_at),
    current: trip.trip_id === currentTripId
  }));
}
