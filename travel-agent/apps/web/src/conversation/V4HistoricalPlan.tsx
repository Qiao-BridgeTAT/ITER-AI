import { useEffect, useMemo, useRef, useState } from "react";
import type {
  ConversationMessageV4,
  V4TripStateEnvelope,
} from "../generated/v4/contracts";
import { buildV4PublishedPlanPresentation } from "../planning/v4PublishedPlanPresentation";
import { visibleV4PlanCityName } from "../planning/visibleV4PublishedPlan";
import { PlanReadyAttachment } from "./PlanReadyAttachment";

export function V4HistoricalPlan({
  message,
  deferred,
  tripState,
  load,
}: {
  message: ConversationMessageV4;
  deferred: boolean;
  tripState: V4TripStateEnvelope | null;
  load: (
    messageId: string,
    signal?: AbortSignal,
  ) => Promise<ConversationMessageV4>;
}) {
  const [complete, setComplete] = useState<ConversationMessageV4 | null>(null);
  const [expanded, setExpanded] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(false);
  const controller = useRef<AbortController | null>(null);
  useEffect(() => () => controller.current?.abort(), []);
  const presentation = useMemo(() => {
    if (!expanded) return null;
    const source = complete ?? message;
    const plan = (source.attachments ?? []).find(
      (item) => "plan_version_id" in item,
    );
    return plan && "plan_version_id" in plan
      ? buildV4PublishedPlanPresentation(
          plan,
          visibleV4PlanCityName(plan, tripState, [source]),
          "历史版本",
        )
      : null;
  }, [complete, expanded, message, tripState]);
  const toggle = async () => {
    if (busy) return;
    if (expanded) {
      setExpanded(false);
      return;
    }
    if (!deferred || complete) {
      setExpanded(true);
      return;
    }
    const request = new AbortController();
    controller.current = request;
    setBusy(true);
    setError(false);
    try {
      const loaded = await load(message.message_id, request.signal);
      if (request.signal.aborted) return;
      if (
        loaded.trip_id !== message.trip_id ||
        loaded.message_id !== message.message_id ||
        loaded.content_hash !== message.content_hash
      )
        throw new Error("History content mismatch");
      setComplete(loaded);
      setExpanded(true);
    } catch {
      if (!request.signal.aborted) setError(true);
    } finally {
      if (!request.signal.aborted) setBusy(false);
    }
  };
  return (
    <div className="conversation-history-plan">
      <button
        type="button"
        aria-expanded={expanded}
        disabled={busy}
        onClick={() => void toggle()}
      >
        {busy
          ? "正在加载此版行程…"
          : expanded
            ? "收起历史行程"
            : "查看此版行程"}
      </button>
      {error ? <p role="alert">此版行程暂未加载成功，可以重试。</p> : null}
      {presentation ? (
        <PlanReadyAttachment presentation={presentation} readOnly />
      ) : null}
    </div>
  );
}
