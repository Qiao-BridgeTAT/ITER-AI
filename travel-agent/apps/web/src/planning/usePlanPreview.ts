import { useEffect, useMemo, useState } from "react";
import { discoveryOptionDescription } from "../conversation/discoveryCopy";
import type {
  ConversationMessageV4,
  PlannerPlanPreview,
  PlannerPlacePreview,
  PlannerWeatherEvidence,
  PlannerPublishedPlan
} from "../generated/v4/contracts";

export function usePlanPreview(
  plan: PlannerPublishedPlan | null,
  messages: readonly ConversationMessageV4[],
  enabled: boolean,
  load: (planVersionId: string) => Promise<PlannerPlanPreview>
): {
  places: readonly PlannerPlacePreview[];
  weather: readonly PlannerWeatherEvidence[];
} {
  const [loaded, setLoaded] = useState<PlannerPlanPreview | null>(null);
  const version = plan?.plan_version_id;
  const tripId = plan?.trip_id;
  useEffect(() => {
    if (!enabled || !version) return;
    let disposed = false;
    void load(version)
      .then((result) => {
        if (
          !disposed &&
          result.plan_version_id === version &&
          result.trip_id === tripId
        )
          setLoaded(result);
      })
      .catch(() => {
        // Media is optional. Keep the known card images and the complete itinerary.
      });
    return () => {
      disposed = true;
    };
  }, [enabled, version, tripId, load]);

  return useMemo(() => {
    const previews = new Map<string, PlannerPlacePreview>();
    for (const message of messages) {
      for (const card of message.attachments ?? []) {
        if (!("kind" in card) || card.kind !== "specific_card") continue;
        for (const option of card.options) {
          const id = option.entity_ref?.canonical_entity_id;
          if (!id) continue;
          const previous = previews.get(id);
          const description =
            card.domain === "attraction" &&
            ["attraction-v2", "attraction-v3"].includes(
              card.generation_metadata.strategy_version ?? "legacy"
            )
              ? discoveryOptionDescription(
                  "attraction_specific",
                  option.description
                )
              : null;
          previews.set(id, {
            place_id: id,
            image_url: option.image_url ?? previous?.image_url ?? null,
            image_source_ref: option.image_url
              ? option.image_source_ref
              : (previous?.image_source_ref ?? null),
            description: description ?? previous?.description ?? null
          });
        }
      }
    }
    if (
      loaded &&
      loaded.plan_version_id === version &&
      loaded.trip_id === tripId
    ) {
      for (const preview of loaded.places ?? []) {
        const previous = previews.get(preview.place_id);
        previews.set(preview.place_id, {
          ...preview,
          image_url: preview.image_url ?? previous?.image_url ?? null,
          image_source_ref:
            preview.image_source_ref ?? previous?.image_source_ref ?? null,
          description:
            discoveryOptionDescription(
              "attraction_specific",
              preview.description
            ) ??
            previous?.description ??
            null
        });
      }
    }
    return {
      places: [...previews.values()],
      weather:
        loaded?.plan_version_id === version && loaded?.trip_id === tripId
          ? (loaded?.weather_evidence ?? [])
          : []
    };
  }, [messages, loaded, version, tripId]);
}
