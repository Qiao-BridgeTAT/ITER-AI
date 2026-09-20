import type {
  ConversationMessageV4,
  PlannerPublishedPlan
} from "../generated/v4/contracts";

export function isPublishedPlanMessage(
  message: ConversationMessageV4,
  plan: PlannerPublishedPlan
): boolean {
  return (
    message.trip_id === plan.trip_id &&
    message.role === "assistant" &&
    message.status === "committed" &&
    ((message.attachments ?? []).some(
      (attachment) =>
        "plan_version_id" in attachment &&
        attachment.plan_version_id === plan.plan_version_id
    ) ||
      (message.message_type === "plan" &&
        message.generation_id === plan.generation_id))
  );
}

/** Insert before this message index, including when the plan's turn is paged out. */
export function v4PlanInsertionIndex(
  messages: readonly ConversationMessageV4[],
  plan: PlannerPublishedPlan
): number {
  const sourceIndex = messages.findIndex((message) =>
    isPublishedPlanMessage(message, plan)
  );
  if (sourceIndex !== -1) return sourceIndex + 1;
  const publishedAt = Date.parse(plan.published_at);
  const nextIndex = messages.findIndex(
    (message) =>
      message.trip_id === plan.trip_id &&
      Date.parse(message.created_at) >= publishedAt
  );
  return nextIndex === -1 ? messages.length : nextIndex;
}
