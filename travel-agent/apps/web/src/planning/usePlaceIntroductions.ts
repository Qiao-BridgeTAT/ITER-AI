import { useEffect, useMemo, useState } from "react";
import type { PlaceIntroductionView } from "../generated/v4/contracts";

export type IntroductionLoader = (
  scopeKind: PlaceIntroductionView["scope_kind"],
  scopeId: string,
  signal?: AbortSignal
) => Promise<PlaceIntroductionView>;

export function usePlaceIntroductions(
  scopeId: string | undefined,
  scopeKind: PlaceIntroductionView["scope_kind"],
  enabled: boolean,
  load?: IntroductionLoader
): ReadonlyMap<string, string> {
  const [loaded, setLoaded] = useState<{
    result: PlaceIntroductionView;
    loader: IntroductionLoader;
  } | null>(null);
  useEffect(() => {
    if (!enabled || !scopeId || !load) return;
    const controller = new AbortController();
    void Promise.resolve()
      .then(() => load(scopeKind, scopeId, controller.signal))
      .then((result) => {
        if (
          !controller.signal.aborted &&
          result.scope_id === scopeId &&
          result.scope_kind === scopeKind
        ) {
          setLoaded({ result, loader: load });
        }
      })
      .catch(() => {
        // Optional prose never blocks cards, restores, images, or final plans.
      });
    return () => controller.abort();
  }, [scopeId, scopeKind, enabled, load]);
  return useMemo(
    () =>
      new Map(
        loaded?.loader === load &&
          loaded?.result.scope_id === scopeId &&
          loaded?.result.scope_kind === scopeKind
          ? (loaded.result.places ?? []).map((place) => [
              place.place_id,
              place.description
            ])
          : []
      ),
    [loaded, load, scopeId, scopeKind]
  );
}
