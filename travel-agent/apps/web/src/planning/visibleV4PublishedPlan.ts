import type {
  ConversationMessageV4,
  PlannerPublishedPlan,
  V4TripStateEnvelope
} from "../generated/v4/contracts";
import { getRegisteredCityName } from "../city/cityContentRepository";

export function visibleV4PlanCityName(
  plan: PlannerPublishedPlan,
  state: V4TripStateEnvelope | null | undefined,
  messages: readonly ConversationMessageV4[]
): string {
  const cityId = plan.materialized_schedule.city_id;
  const registered = getRegisteredCityName(cityId);
  if (registered) return registered;
  for (const message of messages) {
    if (message.trip_id !== plan.trip_id || message.status !== "committed")
      continue;
    for (const book of message.attachments ?? []) {
      if (
        "destination_and_dates" in book &&
        book.task_book_id === plan.based_on_task_book_id &&
        book.version === plan.based_on_task_book_version &&
        book.destination_and_dates.destination_canonical_id === cityId
      )
        return book.destination_and_dates.destination_name;
    }
  }
  const basics = state?.semantic_state.trip_basics;
  return basics?.destination_canonical_id === cityId
    ? (basics.destination_name ?? "本次旅行")
    : "本次旅行";
}

/** Display only: never use historical plans to authorize a planning command. */
export function visibleV4PublishedPlan(
  state: V4TripStateEnvelope | null | undefined,
  messages: readonly ConversationMessageV4[]
): PlannerPublishedPlan | null {
  if (!state) return null;
  if (state.published_plan) return state.published_plan;
  const { trip_id: tripId, state_version: stateVersion } = state.semantic_state;
  const candidates = messages.flatMap((message) =>
    message.role === "assistant" &&
    message.status === "committed" &&
    message.trip_id === tripId &&
    message.state_version <= (stateVersion ?? 0)
      ? (message.attachments ?? []).flatMap((attachment) =>
          "plan_version_id" in attachment &&
          attachment.trip_id === tripId &&
          attachment.based_on_state_version <= message.state_version
            ? [{ plan: attachment, version: message.state_version }]
            : []
        )
      : []
  );
  candidates.sort(
    (a, b) =>
      b.version - a.version ||
      Date.parse(b.plan.published_at) - Date.parse(a.plan.published_at) ||
      b.plan.plan_version_id.localeCompare(a.plan.plan_version_id)
  );
  return candidates[0]?.plan ?? null;
}
