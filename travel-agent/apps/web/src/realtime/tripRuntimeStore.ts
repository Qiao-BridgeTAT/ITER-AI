import { createStore, type StoreApi } from "zustand/vanilla";

import { validatePublicContract } from "../contracts/validation";
import type {
  ReplayCursorFixture,
  ServerEvent,
  TripState
} from "../generated/contracts";
import type {
  ConversationEventV4,
  ConversationMessageV4,
  ConversationSnapshotV4
} from "../generated/v4/contracts";
import {
  createEventReplayCursor,
  createRuntimePresentation,
  expectGeneration,
  reduceServerEvent,
  type EventReplayAction,
  type EventReplayCursor,
  type RuntimePresentation
} from "./eventReducer";
import {
  aggregateFromV4Snapshot,
  createV4RuntimeAggregate,
  reduceV4Event,
  type V4RuntimeAggregate
} from "./v4EventReducer";

export type ConnectionStatus =
  "idle" | "connected" | "disconnected" | "recovering";

export interface TripRuntimeState {
  tripState: TripState | null;
  cursor: EventReplayCursor;
  presentation: RuntimePresentation;
  connectionStatus: ConnectionStatus;
  recoveryRequired: boolean;
  lastEventAction: EventReplayAction | null;
  v4: V4RuntimeAggregate;
  loadSnapshot: (snapshot: TripState, cursor?: ReplayCursorFixture) => boolean;
  prepareGeneration: (requestId: string) => void;
  receiveEvent: (event: ServerEvent) => EventReplayAction;
  loadV4Snapshot: (snapshot: ConversationSnapshotV4) => boolean;
  recoverFromV4Snapshot: (snapshot: ConversationSnapshotV4) => boolean;
  addV4HistoryMessages: (
    tripId: string,
    messages: ConversationMessageV4[]
  ) => boolean;
  receiveV4Event: (event: ConversationEventV4) => EventReplayAction;
  markDisconnected: () => void;
  recoverFromSnapshot: (
    snapshot: TripState,
    cursor?: ReplayCursorFixture
  ) => boolean;
  seedCursorForReplay: (cursor: ReplayCursorFixture) => void;
  reset: () => void;
}

export type TripRuntimeStore = StoreApi<TripRuntimeState>;

export function createTripRuntimeStore(): TripRuntimeStore {
  const initial = createInitialRuntimeState();
  return createStore<TripRuntimeState>((set, get) => ({
    ...initial,
    loadSnapshot: (snapshot, cursor) => {
      if (!validatePublicContract("trip_state", snapshot).success) {
        set({ connectionStatus: "recovering", recoveryRequired: true });
        return false;
      }
      set({
        tripState: structuredClone(snapshot),
        cursor: cursorFromSnapshot(snapshot, cursor),
        presentation: {
          ...createRuntimePresentation(),
          conversationMessages: snapshot.conversation_messages ?? [],
          mapUpdate: snapshot.map_view ?? null,
          itinerary: snapshot.itinerary ?? null,
          issues: snapshot.issues ?? []
        },
        connectionStatus: "connected",
        recoveryRequired: false,
        lastEventAction: null
      });
      return true;
    },
    prepareGeneration: (requestId) => {
      set((state) => ({
        cursor: expectGeneration(state.cursor, requestId),
        presentation: {
          ...state.presentation,
          streamText: "",
          latestError: null,
          latestErrorRequestId: null,
          generationStatus: null
        }
      }));
    },
    receiveEvent: (event) => {
      const current = get();
      if (
        current.recoveryRequired ||
        current.connectionStatus === "disconnected" ||
        current.connectionStatus === "recovering"
      ) {
        return "ignore";
      }
      const result = reduceServerEvent(
        current.cursor,
        current.tripState,
        current.presentation,
        event
      );
      if (result.action === "refresh_snapshot") {
        set({
          presentation: result.presentation,
          connectionStatus: "recovering",
          recoveryRequired: true,
          lastEventAction: result.action
        });
        return result.action;
      }
      set({
        cursor: result.cursor,
        tripState: result.tripState,
        presentation: result.presentation,
        lastEventAction: result.action
      });
      return result.action;
    },
    loadV4Snapshot: (snapshot) => {
      const aggregate = aggregateFromV4Snapshot(snapshot);
      if (aggregate === null) {
        set({ connectionStatus: "recovering", recoveryRequired: true });
        return false;
      }
      set({
        v4: aggregate,
        connectionStatus: "connected",
        recoveryRequired: false,
        lastEventAction: null
      });
      return true;
    },
    recoverFromV4Snapshot: (snapshot) => {
      const aggregate = aggregateFromV4Snapshot(snapshot, get().v4);
      if (aggregate === null) {
        set({ connectionStatus: "recovering", recoveryRequired: true });
        return false;
      }
      set({
        v4: aggregate,
        connectionStatus: "connected",
        recoveryRequired: false,
        lastEventAction: null
      });
      return true;
    },
    addV4HistoryMessages: (tripId, messages) => {
      const current = get().v4;
      if (
        current.tripState?.semantic_state.trip_id !== tripId ||
        messages.some(
          (message) =>
            message.trip_id !== tripId ||
            message.state_version > current.cursor.localStateVersion
        )
      )
        return false;
      const existing = new Map(
        current.presentation.messages.map((message) => [
          message.message_id,
          message
        ])
      );
      for (const message of messages) {
        const prior = existing.get(message.message_id);
        if (prior && prior.content_hash !== message.content_hash) return false;
        if (!prior) existing.set(message.message_id, message);
      }
      set({
        v4: {
          ...current,
          presentation: {
            ...current.presentation,
            messages: [...existing.values()].sort(
              (a, b) =>
                a.state_version - b.state_version ||
                Number(a.role !== "user") - Number(b.role !== "user") ||
                a.created_at.localeCompare(b.created_at) ||
                a.message_id.localeCompare(b.message_id)
            )
          }
        }
      });
      return true;
    },
    receiveV4Event: (event) => {
      const current = get();
      if (
        current.recoveryRequired ||
        current.connectionStatus === "disconnected" ||
        current.connectionStatus === "recovering"
      ) {
        return "ignore";
      }
      const reduced = reduceV4Event(current.v4, event);
      set({
        v4: {
          tripState: reduced.tripState,
          pendingInteraction: reduced.pendingInteraction,
          plannerWorkspace: reduced.plannerWorkspace,
          referenceLinks: reduced.referenceLinks,
          lastOutboxCursor: reduced.lastOutboxCursor,
          cursor: reduced.cursor,
          presentation: reduced.presentation
        },
        connectionStatus:
          reduced.action === "refresh_snapshot" ? "recovering" : "connected",
        recoveryRequired: reduced.action === "refresh_snapshot",
        lastEventAction: reduced.action
      });
      return reduced.action;
    },
    markDisconnected: () => {
      set({ connectionStatus: "disconnected", recoveryRequired: true });
    },
    recoverFromSnapshot: (snapshot, cursor) => {
      if (!validatePublicContract("trip_state", snapshot).success) {
        set({ connectionStatus: "recovering", recoveryRequired: true });
        return false;
      }
      set((state) => ({
        tripState: structuredClone(snapshot),
        cursor: cursorFromSnapshot(snapshot, cursor),
        presentation: {
          ...state.presentation,
          conversationMessages: snapshot.conversation_messages ?? [],
          mapUpdate: snapshot.map_view ?? null,
          itinerary: snapshot.itinerary ?? state.presentation.itinerary,
          issues: snapshot.issues ?? state.presentation.issues
        },
        connectionStatus: "connected",
        recoveryRequired: false,
        lastEventAction: null
      }));
      return true;
    },
    seedCursorForReplay: (cursor) => {
      set({ cursor: cursorFromFixture(cursor) });
    },
    reset: () => set(createInitialRuntimeState())
  }));
}

function createInitialRuntimeState(): Pick<
  TripRuntimeState,
  | "tripState"
  | "cursor"
  | "presentation"
  | "connectionStatus"
  | "recoveryRequired"
  | "lastEventAction"
  | "v4"
> {
  return {
    tripState: null,
    cursor: createEventReplayCursor(),
    presentation: createRuntimePresentation(),
    connectionStatus: "idle",
    recoveryRequired: false,
    lastEventAction: null,
    v4: createV4RuntimeAggregate()
  };
}

function cursorFromSnapshot(
  snapshot: TripState,
  fixture?: ReplayCursorFixture
): EventReplayCursor {
  if (fixture !== undefined) {
    return cursorFromFixture(fixture);
  }
  return {
    ...createEventReplayCursor(snapshot.state_version),
    currentGenerationId: snapshot.active_generation_id ?? null
  };
}

function cursorFromFixture(fixture: ReplayCursorFixture): EventReplayCursor {
  return {
    localStateVersion: fixture.local_state_version,
    currentGenerationId: fixture.current_generation_id ?? null,
    lastSequence: fixture.last_sequence ?? 0,
    seenEventIds: new Set(fixture.seen_event_ids ?? []),
    expectedGenerationRequestId: null
  };
}
