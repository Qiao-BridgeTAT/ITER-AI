import { useCallback, useRef, useState } from "react";
import type { ConversationHistoryWindow } from "../generated/v4/contracts";
import type { TripRuntimeStore } from "../realtime/tripRuntimeStore";
import {
  BackendRequestError,
  type RestoreTimings,
  type TimedConversationView,
  type TravelApiClient,
} from "./travelApiClient";

export type AppliedRestoreTimings = RestoreTimings & {
  tripId: string;
  applyMs: number;
  appliedAt: number;
};

/** Presentation history only: never changes semantic state or the event cursor. */
export function useConversationHistory(
  api: TravelApiClient,
  tripId: string,
  store: TripRuntimeStore,
  anonymousSession: { current: string | null },
) {
  const activeTrip = useRef(tripId);
  activeTrip.current = tripId;
  const [history, setHistory] = useState<{
    tripId: string;
    window: ConversationHistoryWindow;
  } | null>(null);
  const historyRef = useRef(history);
  const [restoreTimings, setRestoreTimings] =
    useState<AppliedRestoreTimings | null>(null);
  const [loadingTrip, setLoadingTrip] = useState<string | null>(null);
  const loadRequest = useRef<AbortController | null>(null);
  const [historyError, setHistoryError] = useState<{
    tripId: string;
    message: string;
  } | null>(null);

  const restoreView = useCallback(
    (view: TimedConversationView, recovering: boolean) => {
      if (
        activeTrip.current !== tripId ||
        view.snapshot.trip_state.semantic_state.trip_id !== tripId
      )
        return false;
      const applying = performance.now();
      const previous = store.getState().v4;
      const previousMessages =
        previous.tripState?.semantic_state.trip_id === tripId
          ? previous.presentation.messages
          : [];
      const loaded = recovering
        ? store.getState().recoverFromV4Snapshot(view.snapshot)
        : store.getState().loadV4Snapshot(view.snapshot);
      if (!loaded) return false;
      if (previousMessages.length)
        store.getState().addV4HistoryMessages(tripId, previousMessages);
      const prior =
        historyRef.current?.tripId === tripId
          ? historyRef.current.window
          : null;
      const overlaps =
        prior &&
        (view.history.before_state_version == null ||
          prior.through_state_version >= view.history.before_state_version);
      const window: ConversationHistoryWindow = {
        ...view.history,
        before_state_version: overlaps
          ? prior.before_state_version == null ||
            view.history.before_state_version == null
            ? null
            : Math.min(
                prior.before_state_version,
                view.history.before_state_version,
              )
          : view.history.before_state_version,
        deferred_attachment_message_ids: [
          ...new Set([
            ...(prior?.deferred_attachment_message_ids ?? []),
            ...(view.history.deferred_attachment_message_ids ?? []),
          ]),
        ],
      };
      historyRef.current = { tripId, window };
      setHistory(historyRef.current);
      if (view.timings)
        setRestoreTimings({
          ...view.timings,
          tripId,
          applyMs: performance.now() - applying,
          appliedAt: performance.now(),
        });
      return true;
    },
    [store, tripId],
  );

  const loadOlderMessages = useCallback(async () => {
    const current = historyRef.current;
    if (
      current?.tripId !== tripId ||
      current.window.before_state_version == null ||
      loadRequest.current
    )
      return;
    const controller = new AbortController();
    loadRequest.current = controller;
    setLoadingTrip(tripId);
    setHistoryError(null);
    try {
      const page = await api.getV4History(
        tripId,
        current.window.before_state_version,
        current.window.through_state_version,
        anonymousSession.current ?? undefined,
        controller.signal,
      );
      if (activeTrip.current !== tripId || controller.signal.aborted) return;
      if (!store.getState().addV4HistoryMessages(tripId, page.messages ?? [])) {
        throw new BackendRequestError(409, "history_version_mismatch");
      }
      const latest =
        historyRef.current?.tripId === tripId
          ? historyRef.current.window
          : current.window;
      historyRef.current = {
        tripId,
        window: {
          ...latest,
          before_state_version: page.history.before_state_version,
          deferred_attachment_message_ids: [
            ...new Set([
              ...(latest.deferred_attachment_message_ids ?? []),
              ...(page.history.deferred_attachment_message_ids ?? []),
            ]),
          ],
        },
      };
      setHistory(historyRef.current);
    } catch {
      if (activeTrip.current === tripId && !controller.signal.aborted) {
        setHistoryError({
          tripId,
          message: "更早的消息暂未加载成功，当前行程不受影响。",
        });
      }
    } finally {
      if (loadRequest.current === controller) {
        loadRequest.current = null;
        setLoadingTrip(null);
      }
    }
  }, [api, anonymousSession, store, tripId]);

  const cancelHistory = useCallback(() => {
    loadRequest.current?.abort();
    loadRequest.current = null;
    setLoadingTrip(null);
  }, []);
  const getHistoryMessage = useCallback(
    (messageId: string, signal?: AbortSignal) =>
      api.getV4Message(
        tripId,
        messageId,
        anonymousSession.current ?? undefined,
        signal,
      ),
    [api, anonymousSession, tripId],
  );

  return {
    restoreView,
    loadOlderMessages,
    getHistoryMessage,
    cancelHistory,
    historyWindow: history?.tripId === tripId ? history.window : null,
    restoreTimings: restoreTimings?.tripId === tripId ? restoreTimings : null,
    historyLoading: loadingTrip === tripId,
    historyError: historyError?.tripId === tripId ? historyError.message : null,
  };
}
