import type { MapUpdatePayload } from "../generated/contracts";

/** Prefer a newer formal map event; use the stable snapshot only until one arrives. */
export function mergeMapUpdates(
  stable: MapUpdatePayload | null | undefined,
  incoming: MapUpdatePayload | null | undefined
): MapUpdatePayload | null {
  if (!stable && !incoming) return null;
  if (!stable) return structuredClone(incoming!);
  if (!incoming) return structuredClone(stable);

  const markers = new Map<
    string,
    NonNullable<MapUpdatePayload["markers"]>[number]
  >();
  for (const marker of incoming.markers ?? []) {
    markers.set(marker.place_id, marker);
  }
  const routes = new Map<
    string,
    NonNullable<MapUpdatePayload["routes"]>[number]
  >();
  for (const route of incoming.routes ?? []) {
    routes.set(routeIdentity(route), route);
  }
  return {
    selected_day_index: incoming.selected_day_index,
    markers: [...markers.values()],
    routes: [...routes.values()]
  };
}

function routeIdentity(
  route: NonNullable<MapUpdatePayload["routes"]>[number]
): string {
  return `${route.from_place_id}:${route.to_place_id}:${route.mode}`;
}
