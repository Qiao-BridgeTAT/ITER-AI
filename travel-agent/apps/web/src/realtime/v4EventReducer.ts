import { validateV4Contract } from "../contracts/v4Validation";
import type {
  AgentProgressEntry,
  ConversationEventV4,
  ConversationMessageV4,
  ConversationSnapshotV4,
  V4TripStateEnvelope
} from "../generated/v4/contracts";
import type { EventReplayAction } from "./eventReducer";

export interface V4EventCursor {
  localStateVersion: number;
  currentGenerationId: string | null;
  currentTurnId: string | null;
  currentMessageId: string | null;
  lastSequence: number;
  nextChunkIndex: number;
  commitObserved: boolean;
  seenEventIds: ReadonlySet<string>;
}

export interface V4RuntimePresentation {
  agentProgress: AgentProgressEntry[];
  messages: ConversationMessageV4[];
  streamText: string;
  generationId: string | null;
  agentStatusCode: string | null;
  agentStatusMessage: string | null;
  terminalEvent: ConversationEventV4 | null;
  readyAttachmentIds: string[];
  failureCode: string | null;
}

export interface V4RuntimeAggregate {
  tripState: V4TripStateEnvelope | null;
  pendingInteraction: ConversationSnapshotV4["pending_interaction"];
  plannerWorkspace: ConversationSnapshotV4["planner_workspace"];
  referenceLinks: NonNullable<ConversationSnapshotV4["reference_links"]>;
  lastOutboxCursor: string | null;
  cursor: V4EventCursor;
  presentation: V4RuntimePresentation;
}

export interface V4EventReduction extends V4RuntimeAggregate {
  action: EventReplayAction;
}

export function createV4RuntimeAggregate(): V4RuntimeAggregate {
  return {
    tripState: null,
    pendingInteraction: null,
    plannerWorkspace: null,
    referenceLinks: [],
    lastOutboxCursor: null,
    cursor: {
      localStateVersion: 0,
      currentGenerationId: null,
      currentTurnId: null,
      currentMessageId: null,
      lastSequence: 0,
      nextChunkIndex: 0,
      commitObserved: false,
      seenEventIds: new Set()
    },
    presentation: {
      agentProgress: [],
      messages: [],
      streamText: "",
      generationId: null,
      agentStatusCode: null,
      agentStatusMessage: null,
      terminalEvent: null,
      readyAttachmentIds: [],
      failureCode: null
    }
  };
}

export function aggregateFromV4Snapshot(
  snapshot: ConversationSnapshotV4,
  previous?: V4RuntimeAggregate
): V4RuntimeAggregate | null {
  if (!validateV4Contract("conversation_snapshot", snapshot).success) {
    return null;
  }
  const activeStatus = snapshot.active_generation_status;
  if (
    activeStatus &&
    (activeStatus.generation_id !== snapshot.active_generation_id ||
      activeStatus.trip_id !== snapshot.trip_state.semantic_state.trip_id ||
      activeStatus.conversation_message != null)
  ) {
    return null;
  }
  const version = snapshot.trip_state.semantic_state.state_version ?? 0;
  return {
    tripState: structuredClone(snapshot.trip_state),
    pendingInteraction: snapshot.pending_interaction ?? null,
    plannerWorkspace: snapshot.planner_workspace ?? null,
    referenceLinks: snapshot.reference_links ?? [],
    lastOutboxCursor: snapshot.last_outbox_cursor ?? null,
    cursor: {
      localStateVersion: version,
      currentGenerationId: snapshot.active_generation_id ?? null,
      currentTurnId:
        snapshot.active_generation_id != null
          ? previous?.cursor.currentGenerationId ===
            snapshot.active_generation_id
            ? previous.cursor.currentTurnId
            : null
          : (snapshot.terminal_event?.turn_id ?? null),
      currentMessageId: null,
      lastSequence: 0,
      nextChunkIndex: 0,
      commitObserved: false,
      seenEventIds: previous?.cursor.seenEventIds ?? new Set()
    },
    presentation: {
      agentProgress: structuredClone(snapshot.agent_progress ?? []),
      messages: structuredClone(snapshot.messages ?? []),
      streamText: "",
      generationId: snapshot.active_generation_id ?? null,
      agentStatusCode:
        snapshot.active_generation_id == null
          ? null
          : (snapshot.active_generation_status?.status_code ??
            "recovering_generation"),
      agentStatusMessage:
        snapshot.active_generation_id == null
          ? null
          : (snapshot.active_generation_status?.message ??
            "正在恢复仍在处理的请求。"),
      terminalEvent: snapshot.terminal_event ?? null,
      readyAttachmentIds: [],
      failureCode: null
    }
  };
}

export function reduceV4Event(
  current: V4RuntimeAggregate,
  event: ConversationEventV4
): V4EventReduction {
  if (!validateV4Contract("conversation_event", event).success) {
    return result("refresh_snapshot", current);
  }
  if (current.cursor.seenEventIds.has(event.event_id)) {
    return result("ignore", current);
  }
  if (
    current.tripState !== null &&
    event.trip_id !== current.tripState.semantic_state.trip_id
  ) {
    return result("ignore", current);
  }

  const seen = new Set([...current.cursor.seenEventIds, event.event_id]);
  if (event.event_type === "turn.accepted") {
    return result("apply", {
      ...current,
      cursor: {
        ...current.cursor,
        currentGenerationId: event.generation_id,
        currentTurnId: event.turn_id,
        currentMessageId: null,
        lastSequence: 0,
        nextChunkIndex: 0,
        commitObserved: false,
        seenEventIds: seen
      },
      presentation: {
        ...current.presentation,
        streamText: "",
        generationId: event.generation_id,
        agentStatusCode: "accepted",
        agentStatusMessage: "消息已接收。",
        terminalEvent: null,
        readyAttachmentIds: [],
        failureCode: null
      }
    });
  }
  // Planner resumes keep the job generation ID, but every execution has a new turn.
  // An old worker's late terminal/status must not stop the resumed execution.
  if (
    current.cursor.currentTurnId !== null &&
    event.turn_id !== current.cursor.currentTurnId
  ) {
    return result("ignore", current);
  }
  if (event.event_type === "agent.progress") {
    if (
      event.generation_id !== current.cursor.currentGenerationId ||
      current.presentation.terminalEvent?.turn_id === event.turn_id
    ) {
      return result("ignore", current);
    }
    const entries = current.presentation.agentProgress;
    const duplicate = entries.some(
      (p) =>
        p.event_id === event.event_id ||
        (p.generation_id === event.generation_id &&
          p.progress_index === event.progress.progress_index)
    );
    if (duplicate) return result("ignore", current);
    return result("apply", {
      ...current,
      cursor: { ...current.cursor, seenEventIds: seen },
      presentation: {
        ...current.presentation,
        agentProgress: [...entries, event.progress].sort((a, b) =>
          a.generation_id === b.generation_id
            ? a.progress_index - b.progress_index
            : a.emitted_at.localeCompare(b.emitted_at)
        )
      }
    });
  }
  if (event.event_type === "agent.status") {
    if (
      current.presentation.terminalEvent?.generation_id === event.generation_id
    ) {
      return result("ignore", current);
    }
    if (
      current.cursor.currentGenerationId !== null &&
      event.generation_id !== current.cursor.currentGenerationId
    ) {
      return result("ignore", current);
    }
    return result("apply", {
      ...current,
      cursor: {
        ...current.cursor,
        currentTurnId: event.turn_id,
        seenEventIds: seen
      },
      presentation: {
        ...current.presentation,
        generationId: event.generation_id,
        agentStatusCode: keepDiscoveryProgress(current, event.status_code)
          ? current.presentation.agentStatusCode
          : event.status_code,
        agentStatusMessage: keepDiscoveryProgress(current, event.status_code)
          ? current.presentation.agentStatusMessage
          : event.message,
        messages: event.conversation_message
          ? [
              ...current.presentation.messages.filter(
                (message) =>
                  message.message_id !== event.conversation_message!.message_id
              ),
              event.conversation_message
            ]
          : current.presentation.messages
      }
    });
  }
  if (
    event.event_type === "turn.cancelled" ||
    event.event_type === "turn.failed"
  ) {
    if (
      current.cursor.currentGenerationId !== null &&
      event.generation_id !== current.cursor.currentGenerationId
    ) {
      return result("ignore", current);
    }
    return result("refresh_snapshot", {
      ...current,
      cursor: {
        ...current.cursor,
        currentGenerationId: null,
        currentMessageId: null,
        lastSequence: 0,
        nextChunkIndex: 0,
        commitObserved: false,
        seenEventIds: seen
      },
      presentation: {
        ...current.presentation,
        generationId: null,
        agentStatusCode: event.event_type,
        agentStatusMessage:
          event.event_type === "turn.cancelled"
            ? "上一条请求已取消。"
            : "这轮处理未能完成。",
        terminalEvent: event,
        failureCode: event.failure_code
      }
    });
  }

  if (
    current.cursor.currentGenerationId !== null &&
    event.generation_id !== current.cursor.currentGenerationId
  ) {
    return result("ignore", current);
  }
  if (event.sequence !== current.cursor.lastSequence + 1) {
    return result("refresh_snapshot", current);
  }
  if (event.event_type === "state.committed") {
    if (
      event.sequence !== 1 ||
      event.base_state_version !== current.cursor.localStateVersion ||
      event.committed_state_version !== event.base_state_version + 1
    ) {
      return result("refresh_snapshot", current);
    }
    return result("apply", {
      ...current,
      cursor: {
        ...current.cursor,
        localStateVersion: event.committed_state_version,
        currentGenerationId: event.generation_id,
        currentTurnId: event.turn_id,
        currentMessageId: event.message_id,
        lastSequence: event.sequence,
        nextChunkIndex: 0,
        commitObserved: true,
        seenEventIds: seen
      },
      presentation: {
        ...current.presentation,
        generationId: event.generation_id,
        agentStatusCode: keepDiscoveryProgress(current, "committed")
          ? current.presentation.agentStatusCode
          : "committed",
        agentStatusMessage: keepDiscoveryProgress(current, "committed")
          ? current.presentation.agentStatusMessage
          : "回复已提交，正在发送。"
      }
    });
  }
  if (
    "state_version" in event &&
    event.state_version !== current.cursor.localStateVersion
  ) {
    return result("refresh_snapshot", current);
  }
  if (!current.cursor.commitObserved) {
    return result("refresh_snapshot", current);
  }
  if (
    "message_id" in event &&
    current.cursor.currentMessageId !== null &&
    event.message_id !== current.cursor.currentMessageId
  ) {
    return result("refresh_snapshot", current);
  }
  if (event.event_type === "assistant.started") {
    return result("apply", {
      ...current,
      cursor: {
        ...current.cursor,
        currentGenerationId: event.generation_id,
        currentMessageId: event.message_id,
        lastSequence: event.sequence,
        seenEventIds: seen
      },
      presentation: {
        ...current.presentation,
        streamText: "",
        generationId: event.generation_id,
        agentStatusCode: keepDiscoveryProgress(current, "sending")
          ? current.presentation.agentStatusCode
          : "sending",
        agentStatusMessage: keepDiscoveryProgress(current, "sending")
          ? current.presentation.agentStatusMessage
          : "正在发送回复。"
      }
    });
  }
  if (event.event_type === "assistant.delta") {
    if (event.chunk_index !== current.cursor.nextChunkIndex) {
      return result("refresh_snapshot", current);
    }
    return result("apply", {
      ...current,
      cursor: {
        ...current.cursor,
        lastSequence: event.sequence,
        nextChunkIndex: event.chunk_index + 1,
        seenEventIds: seen
      },
      presentation: {
        ...current.presentation,
        streamText: current.presentation.streamText + event.delta
      }
    });
  }
  if (event.event_type === "attachment.ready") {
    return result("apply", {
      ...current,
      cursor: {
        ...current.cursor,
        lastSequence: event.sequence,
        seenEventIds: seen
      },
      presentation: {
        ...current.presentation,
        readyAttachmentIds: [
          ...new Set([
            ...current.presentation.readyAttachmentIds,
            event.attachment_id
          ])
        ]
      }
    });
  }
  if (event.event_type !== "assistant.completed") {
    return result("refresh_snapshot", current);
  }
  const committedMessage: ConversationMessageV4 = {
    message_id: event.message_id,
    trip_id: event.trip_id,
    turn_id: event.turn_id,
    generation_id: event.generation_id,
    role: "assistant",
    message_type: "text",
    text: current.presentation.streamText,
    state_version: event.state_version,
    generation_mode: event.generation_mode,
    content_hash: event.content_hash,
    attachments: [],
    status: "committed",
    created_at: event.emitted_at
  };
  return result("refresh_snapshot", {
    ...current,
    lastOutboxCursor: event.outbox_cursor,
    cursor: {
      ...current.cursor,
      currentGenerationId: null,
      currentMessageId: null,
      lastSequence: 0,
      nextChunkIndex: 0,
      commitObserved: false,
      seenEventIds: seen
    },
    presentation: {
      ...current.presentation,
      messages: [
        ...current.presentation.messages.filter(
          (message) => message.message_id !== event.message_id
        ),
        committedMessage
      ],
      streamText: "",
      generationId: null,
      agentStatusCode: "completed",
      agentStatusMessage: null,
      terminalEvent: event,
      failureCode: event.generation_mode === "fallback" ? "fallback" : null
    }
  });
}

function result(
  action: EventReplayAction,
  aggregate: V4RuntimeAggregate
): V4EventReduction {
  return { action, ...aggregate };
}

export function isDiscoveryProgress(code: string | null | undefined): boolean {
  return Boolean(
    code?.startsWith("attraction_") || code?.startsWith("dining_")
  );
}

function keepDiscoveryProgress(
  current: V4RuntimeAggregate,
  nextCode: string
): boolean {
  return (
    isDiscoveryProgress(current.presentation.agentStatusCode) &&
    !isDiscoveryProgress(nextCode) &&
    !nextCode.startsWith("planner_")
  );
}
