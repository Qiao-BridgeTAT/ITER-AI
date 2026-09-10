import { applyPatch, type Operation } from "fast-json-patch";

import { validatePublicContract } from "../contracts/validation";
import type {
  ConversationMessage,
  ErrorEventPayload,
  GenerationStatusPayload,
  Itinerary,
  JsonPatchOperation,
  MapUpdatePayload,
  PlanningIssue,
  ServerEvent,
  TripState,
} from "../generated/contracts";

export type EventReplayAction = "apply" | "ignore" | "refresh_snapshot";

export interface EventReplayCursor {
  localStateVersion: number;
  currentGenerationId: string | null;
  lastSequence: number;
  seenEventIds: ReadonlySet<string>;
  expectedGenerationRequestId: string | null;
}

export interface RuntimePresentation {
  streamText: string;
  generationId: string | null;
  generationRequestId: string | null;
  terminalEventId: string | null;
  conversationMessages: ConversationMessage[];
  mapUpdate: MapUpdatePayload | null;
  itinerary: Itinerary | null;
  issues: PlanningIssue[];
  latestError: ErrorEventPayload | null;
  latestErrorRequestId: string | null;
  generationStatus: GenerationStatusPayload | null;
}

export interface EventReduction {
  action: EventReplayAction;
  cursor: EventReplayCursor;
  tripState: TripState | null;
  presentation: RuntimePresentation;
}

export function createEventReplayCursor(
  localStateVersion = 0,
): EventReplayCursor {
  return {
    localStateVersion,
    currentGenerationId: null,
    lastSequence: 0,
    seenEventIds: new Set(),
    expectedGenerationRequestId: null,
  };
}

export function createRuntimePresentation(): RuntimePresentation {
  return {
    streamText: "",
    generationId: null,
    generationRequestId: null,
    terminalEventId: null,
    conversationMessages: [],
    mapUpdate: null,
    itinerary: null,
    issues: [],
    latestError: null,
    latestErrorRequestId: null,
    generationStatus: null,
  };
}

export function expectGeneration(
  cursor: EventReplayCursor,
  requestId: string,
): EventReplayCursor {
  return {
    ...cursor,
    currentGenerationId: null,
    lastSequence: 0,
    expectedGenerationRequestId: requestId,
  };
}

export function reduceServerEvent(
  cursor: EventReplayCursor,
  tripState: TripState | null,
  presentation: RuntimePresentation,
  event: ServerEvent,
): EventReduction {
  if (!validatePublicContract("server_event", event).success) {
    return refresh(cursor, tripState, presentation);
  }
  if (cursor.seenEventIds.has(event.event_id)) {
    return ignore(cursor, tripState, presentation);
  }

  let workingCursor = cursor;
  if (
    event.generation_id !== null &&
    cursor.expectedGenerationRequestId !== null
  ) {
    const isExpectedStart =
      event.request_id === cursor.expectedGenerationRequestId &&
      event.type === "generation_status" &&
      event.payload.status === "started" &&
      event.sequence === 1;
    if (!isExpectedStart) {
      return ignore(cursor, tripState, presentation);
    }
    workingCursor = {
      ...cursor,
      currentGenerationId: event.generation_id,
      lastSequence: 0,
      expectedGenerationRequestId: null,
    };
  }

  if (event.generation_id !== null) {
    if (event.generation_id !== workingCursor.currentGenerationId) {
      return ignore(cursor, tripState, presentation);
    }
    if (event.sequence <= workingCursor.lastSequence) {
      return ignore(cursor, tripState, presentation);
    }
    if (event.sequence !== workingCursor.lastSequence + 1) {
      return refresh(cursor, tripState, presentation);
    }
  }

  let nextStateVersion = workingCursor.localStateVersion;
  if (event.type === "state_patch") {
    if (event.base_state_version !== workingCursor.localStateVersion) {
      return event.state_version <= workingCursor.localStateVersion
        ? ignore(cursor, tripState, presentation)
        : refresh(cursor, tripState, presentation);
    }
    nextStateVersion = event.state_version;
  } else if (
    event.base_state_version !== workingCursor.localStateVersion ||
    event.state_version !== workingCursor.localStateVersion
  ) {
    return event.type === "error" && event.payload.snapshot_required
      ? refresh(cursor, tripState, applyPresentationEvent(presentation, event))
      : refresh(cursor, tripState, presentation);
  }

  const nextState = applyStatePatch(tripState, event);
  if (nextState === PATCH_FAILED) {
    return refresh(cursor, tripState, presentation);
  }

  const closesGeneration =
    event.type === "generation_status" &&
    ["completed", "cancelled", "failed"].includes(event.payload.status);
  const nextCursor: EventReplayCursor = {
    localStateVersion: nextStateVersion,
    currentGenerationId: closesGeneration
      ? null
      : workingCursor.currentGenerationId,
    lastSequence: closesGeneration
      ? 0
      : event.generation_id === null
        ? workingCursor.lastSequence
        : event.sequence,
    seenEventIds: new Set([...workingCursor.seenEventIds, event.event_id]),
    expectedGenerationRequestId: workingCursor.expectedGenerationRequestId,
  };
  const nextPresentation = applyPresentationEvent(presentation, event);

  return {
    action: "apply",
    cursor: nextCursor,
    tripState: nextState,
    presentation:
      event.type === "state_patch" && nextState !== null
        ? {
            ...nextPresentation,
            conversationMessages: nextState.conversation_messages ?? [],
          }
        : nextPresentation,
  };
}

const PATCH_FAILED = Symbol("patch-failed");

function applyStatePatch(
  state: TripState | null,
  event: ServerEvent,
): TripState | null | typeof PATCH_FAILED {
  if (event.type !== "state_patch") {
    return state;
  }
  if (state === null) {
    return PATCH_FAILED;
  }
  try {
    const operations = event.payload.patch.map(toJsonPatchOperation);
    const candidate = applyPatch(
      structuredClone(state),
      operations,
      false,
      true,
      true,
    ).newDocument;
    return validatePublicContract("trip_state", candidate).success
      ? (candidate as TripState)
      : PATCH_FAILED;
  } catch {
    return PATCH_FAILED;
  }
}

function toJsonPatchOperation(operation: JsonPatchOperation): Operation {
  const { from, ...rest } = operation;
  return (from == null ? rest : { ...rest, from }) as Operation;
}

function applyPresentationEvent(
  presentation: RuntimePresentation,
  event: ServerEvent,
): RuntimePresentation {
  switch (event.type) {
    case "stream_token":
      return {
        ...presentation,
        streamText: presentation.streamText + event.payload.token,
      };
    case "gesture_ready":
      return event.payload.conversation_message
        ? {
            ...presentation,
            conversationMessages: mergeConversationMessage(
              presentation.conversationMessages,
              event.payload.conversation_message,
            ),
          }
        : presentation;
    case "map_update":
      return { ...presentation, mapUpdate: event.payload };
    case "itinerary_update":
      return { ...presentation, itinerary: event.payload.itinerary };
    case "issue":
      return {
        ...presentation,
        issues: [...presentation.issues, event.payload.issue],
      };
    case "error":
      return {
        ...presentation,
        latestError: event.payload,
        latestErrorRequestId: event.request_id,
      };
    case "generation_status":
      return {
        ...presentation,
        streamText:
          event.payload.status === "started" ? "" : presentation.streamText,
        generationId: event.generation_id,
        generationRequestId: event.request_id,
        terminalEventId: ["completed", "cancelled", "failed"].includes(
          event.payload.status,
        )
          ? event.event_id
          : null,
        generationStatus: event.payload,
      };
    case "state_patch":
      return presentation;
  }
}

function mergeConversationMessage(
  messages: ConversationMessage[],
  incoming: ConversationMessage,
): ConversationMessage[] {
  const existing = messages.find(
    (message) => message.message_id === incoming.message_id,
  );
  if (!existing) {
    return [...messages, incoming].sort((left, right) =>
      left.created_at.localeCompare(right.created_at),
    );
  }
  if (incoming.state_version < existing.state_version) {
    return messages;
  }
  const attachments = mergeById(
    existing.attachments ?? [],
    incoming.attachments ?? [],
    (item) => item.attachment_id,
  );
  const attachmentAnswers = mergeById(
    existing.attachment_answers ?? [],
    incoming.attachment_answers ?? [],
    (item) => item.attachment_id,
  );
  return messages.map((message) =>
    message.message_id === incoming.message_id
      ? {
          ...existing,
          ...incoming,
          attachments,
          attachment_answers: attachmentAnswers,
        }
      : message,
  );
}

function mergeById<T>(
  current: T[],
  incoming: T[],
  identify: (item: T) => string,
): T[] {
  const byId = new Map(current.map((item) => [identify(item), item]));
  incoming.forEach((item) => byId.set(identify(item), item));
  return [...byId.values()];
}

function ignore(
  cursor: EventReplayCursor,
  tripState: TripState | null,
  presentation: RuntimePresentation,
): EventReduction {
  return { action: "ignore", cursor, tripState, presentation };
}

function refresh(
  cursor: EventReplayCursor,
  tripState: TripState | null,
  presentation: RuntimePresentation,
): EventReduction {
  return {
    action: "refresh_snapshot",
    cursor,
    tripState,
    presentation,
  };
}
