import { validateV4Contract } from "../contracts/v4Validation";
import { validatePublicContract } from "../contracts/validation";
import type {
  AccountView,
  AnonymousSessionView,
  ClientCommand,
  ColdStartSubmission,
  PreferenceListView,
  PreferenceValue,
  ServerEvent,
  SmsVerifyRequest,
  TripListView,
  TripShell,
  TripSnapshotView,
} from "../generated/contracts";
import type {
  ConversationEventV4,
  ConversationHistoryPage,
  ConversationMessageV4,
  ConversationSnapshotV4,
  ConversationView,
  PlaceIntroductionView,
  PlannerPlanPreview,
  V4ClientCommand,
} from "../generated/v4/contracts";

type FetchLike = typeof fetch;

export interface RestoreTimings {
  requestMs: number;
  parseMs: number;
  validateMs: number;
}
export type TimedConversationView = ConversationView & {
  timings?: RestoreTimings;
};

export class BackendRequestError extends Error {
  constructor(
    readonly status: number,
    readonly code: string,
  ) {
    super(`Backend request failed: ${code}`);
  }
}

export class TravelApiClient {
  private readonly baseUrl: string;
  private readonly fetchImpl: FetchLike;

  constructor(baseUrl = "/api", fetchImpl?: FetchLike) {
    this.baseUrl = baseUrl;
    this.fetchImpl =
      fetchImpl ?? ((input, init) => globalThis.fetch(input, init));
  }

  async createAnonymousSession(
    idempotencyKey: string,
    existingSessionId?: string,
  ): Promise<AnonymousSessionView> {
    return this.request(
      "/anonymous-sessions",
      "POST",
      "anonymous_session_view",
      undefined,
      { anonymousSessionId: existingSessionId, idempotencyKey },
    );
  }

  async createTrip(
    tripId: string,
    idempotencyKey: string,
    anonymousSessionId?: string,
    signal?: AbortSignal,
  ): Promise<TripSnapshotView> {
    return this.request(
      "/trips",
      "POST",
      "trip_snapshot_view",
      {
        ...(anonymousSessionId === undefined
          ? {}
          : { anonymous_session_id: anonymousSessionId }),
        trip_id: tripId,
      },
      { anonymousSessionId, idempotencyKey, signal },
    );
  }

  async getTrip(
    tripId: string,
    anonymousSessionId?: string,
  ): Promise<TripSnapshotView> {
    return this.request(
      `/trips/${tripId}`,
      "GET",
      "trip_snapshot_view",
      undefined,
      { anonymousSessionId },
    );
  }

  async getTripShell(tripId: string, signal?: AbortSignal): Promise<TripShell> {
    return withRestoreTimeout(signal, (requestSignal) =>
      this.request(`/trips/${tripId}/shell`, "GET", "trip_shell", undefined, {
        signal: requestSignal,
      }),
    );
  }

  async getV4ConversationView(
    tripId: string,
    anonymousSessionId?: string,
    signal?: AbortSignal,
  ): Promise<TimedConversationView> {
    return withRestoreTimeout(signal, async (requestSignal) => {
      const started = performance.now();
      const response = await this.requestRaw(
        `/v4/trips/${tripId}/view`,
        "GET",
        undefined,
        { anonymousSessionId, signal: requestSignal },
      );
      const raw = await response.text();
      const received = performance.now();
      const payload: unknown = JSON.parse(raw);
      const parsed = performance.now();
      // Reuse the already-cached snapshot validator, rather than compiling a
      // second huge schema containing all plan/card definitions on the UI thread.
      if (
        !payload ||
        typeof payload !== "object" ||
        !("snapshot" in payload) ||
        !("history" in payload) ||
        !validateV4Contract("conversation_snapshot", payload.snapshot)
          .success ||
        !validateV4Contract("conversation_history_window", payload.history)
          .success
      ) {
        throw new BackendRequestError(502, "invalid_v4_backend_contract");
      }
      return {
        ...(payload as ConversationView),
        timings: {
          requestMs: received - started,
          parseMs: parsed - received,
          validateMs: performance.now() - parsed,
        },
      };
    });
  }

  async getV4History(
    tripId: string,
    beforeVersion: number,
    throughVersion: number,
    anonymousSessionId?: string,
    signal?: AbortSignal,
  ): Promise<ConversationHistoryPage> {
    return withRestoreTimeout(signal, async (requestSignal) => {
      const response = await this.requestRaw(
        `/v4/trips/${tripId}/history?before_state_version=${beforeVersion}&through_state_version=${throughVersion}`,
        "GET",
        undefined,
        { anonymousSessionId, signal: requestSignal },
      );
      const payload: unknown = await response.json();
      if (!validateV4Contract("conversation_history_page", payload).success) {
        throw new BackendRequestError(502, "invalid_v4_backend_contract");
      }
      return payload as ConversationHistoryPage;
    });
  }

  async getV4Message(
    tripId: string,
    messageId: string,
    anonymousSessionId?: string,
    signal?: AbortSignal,
  ): Promise<ConversationMessageV4> {
    return withRestoreTimeout(signal, async (requestSignal) => {
      const response = await this.requestRaw(
        `/v4/trips/${tripId}/messages/${messageId}`,
        "GET",
        undefined,
        { anonymousSessionId, signal: requestSignal },
      );
      const payload: unknown = await response.json();
      if (!validateV4Contract("conversation_message", payload).success) {
        throw new BackendRequestError(502, "invalid_v4_backend_contract");
      }
      return payload as ConversationMessageV4;
    });
  }

  async getV4Trip(
    tripId: string,
    anonymousSessionId?: string,
  ): Promise<ConversationSnapshotV4> {
    const response = await this.requestRaw(
      `/v4/trips/${tripId}`,
      "GET",
      undefined,
      { anonymousSessionId },
    );
    const payload: unknown = await response.json();
    if (!validateV4Contract("conversation_snapshot", payload).success) {
      throw new BackendRequestError(502, "invalid_v4_backend_contract");
    }
    return payload as ConversationSnapshotV4;
  }

  async attachAnonymousTrip(
    anonymousSessionId: string,
    tripId: string,
    idempotencyKey: string,
  ): Promise<TripSnapshotView> {
    return this.request(
      "/auth/attach-trip",
      "POST",
      "trip_snapshot_view",
      {
        anonymous_session_id: anonymousSessionId,
        trip_id: tripId,
      },
      { idempotencyKey },
    );
  }

  async getV4PlanPreview(
    tripId: string,
    planVersionId: string,
    anonymousSessionId?: string,
  ): Promise<PlannerPlanPreview> {
    const response = await this.requestRaw(
      `/v4/trips/${tripId}/plan-preview?plan_version_id=${encodeURIComponent(planVersionId)}`,
      "GET",
      undefined,
      { anonymousSessionId },
    );
    const payload: unknown = await response.json();
    if (!validateV4Contract("planner_plan_preview", payload).success) {
      throw new BackendRequestError(502, "invalid_v4_backend_contract");
    }
    const result = payload as PlannerPlanPreview;
    if (result.trip_id !== tripId || result.plan_version_id !== planVersionId) {
      throw new BackendRequestError(409, "plan_version_conflict");
    }
    return result;
  }

  async getAccount(): Promise<AccountView> {
    return this.request("/account", "GET", "account_view");
  }

  async getV4PlaceIntroductions(
    tripId: string,
    scopeKind: PlaceIntroductionView["scope_kind"],
    scopeId: string,
    anonymousSessionId?: string,
    signal?: AbortSignal,
  ): Promise<PlaceIntroductionView> {
    const response = await this.requestRaw(
      `/v4/trips/${tripId}/place-introductions?scope_kind=${scopeKind}&scope_id=${encodeURIComponent(scopeId)}`,
      "GET",
      undefined,
      { anonymousSessionId, signal },
    );
    const payload: unknown = await response.json();
    if (!validateV4Contract("place_introduction_view", payload).success) {
      throw new BackendRequestError(502, "invalid_v4_backend_contract");
    }
    const result = payload as PlaceIntroductionView;
    if (
      result.trip_id !== tripId ||
      result.scope_kind !== scopeKind ||
      result.scope_id !== scopeId
    ) {
      throw new BackendRequestError(409, "introduction_scope_conflict");
    }
    return result;
  }

  async updateAccount(
    nickname: string,
    idempotencyKey: string,
  ): Promise<AccountView> {
    return this.request(
      "/account",
      "PATCH",
      "account_view",
      { nickname },
      { idempotencyKey },
    );
  }

  async sendSms(phone: string, idempotencyKey: string): Promise<void> {
    await this.requestRaw("/auth/sms", "POST", { phone }, { idempotencyKey });
  }

  async verifySms(
    payload: SmsVerifyRequest,
    idempotencyKey: string,
  ): Promise<AccountView> {
    return this.request("/auth/verify", "POST", "account_view", payload, {
      idempotencyKey,
    });
  }

  async logout(idempotencyKey: string): Promise<void> {
    await this.requestRaw("/auth/logout", "POST", undefined, {
      idempotencyKey,
    });
  }

  async getPreferences(): Promise<PreferenceListView> {
    return this.request("/preferences", "GET", "preference_list_view");
  }

  async patchPreferences(
    preferences: PreferenceValue[],
    idempotencyKey: string,
  ): Promise<PreferenceListView> {
    return this.request(
      "/preferences",
      "PATCH",
      "preference_list_view",
      {
        preferences: preferences.map(({ preference_id, value, active }) => ({
          preference_id,
          value,
          active,
        })),
      },
      { idempotencyKey },
    );
  }

  async saveColdStartPreference(
    preferences: ColdStartSubmission,
    idempotencyKey: string,
    anonymousSessionId?: string,
  ): Promise<PreferenceListView> {
    return this.request(
      "/preferences/cold-start",
      "POST",
      "preference_list_view",
      preferences,
      { idempotencyKey, anonymousSessionId },
    );
  }

  async listTrips(): Promise<TripListView> {
    return this.request("/trips", "GET", "trip_list_view");
  }

  async deleteAnonymousSession(
    sessionId: string,
    idempotencyKey: string,
    keepalive = false,
  ): Promise<void> {
    await this.requestRaw(
      `/anonymous-sessions/${sessionId}`,
      "DELETE",
      undefined,
      {
        anonymousSessionId: sessionId,
        idempotencyKey,
        keepalive,
      },
    );
  }

  websocketUrl(tripId: string): string {
    const base = new URL(
      this.baseUrl || window.location.origin,
      window.location.origin,
    );
    base.protocol = base.protocol === "https:" ? "wss:" : "ws:";
    const prefix = base.pathname.replace(/\/$/, "");
    base.pathname = `${prefix}/trips/${tripId}/stream`;
    base.search = "";
    base.hash = "";
    return base.toString();
  }

  private async request<T>(
    path: string,
    method: string,
    contractName:
      | "account_view"
      | "anonymous_session_view"
      | "preference_list_view"
      | "trip_list_view"
      | "trip_snapshot_view"
      | "trip_shell",
    body?: unknown,
    options: RequestOptions = {},
  ): Promise<T> {
    const response = await this.requestRaw(path, method, body, options);
    const payload: unknown = await response.json();
    if (!validatePublicContract(contractName, payload).success) {
      throw new BackendRequestError(502, "invalid_backend_contract");
    }
    return payload as T;
  }

  private async requestRaw(
    path: string,
    method: string,
    body?: unknown,
    options: RequestOptions = {},
  ): Promise<Response> {
    const headers = new Headers({ Accept: "application/json" });
    if (body !== undefined) headers.set("Content-Type", "application/json");
    if (options.anonymousSessionId)
      headers.set("X-Anonymous-Session-ID", options.anonymousSessionId);
    if (options.idempotencyKey)
      headers.set("Idempotency-Key", options.idempotencyKey);
    const response = await this.fetchImpl(`${this.baseUrl}${path}`, {
      method,
      headers,
      credentials: "include",
      keepalive: options.keepalive,
      signal: options.signal,
      ...(body === undefined ? {} : { body: JSON.stringify(body) }),
    });
    if (!response.ok) {
      let code = `http_${response.status}`;
      try {
        const payload = (await response.json()) as {
          error?: { code?: string };
          code?: string;
        };
        code = payload.error?.code ?? payload.code ?? code;
      } catch {
        // A non-JSON gateway error still has a stable HTTP classification.
      }
      throw new BackendRequestError(response.status, code);
    }
    return response;
  }
}

async function withRestoreTimeout<T>(
  signal: AbortSignal | undefined,
  request: (signal: AbortSignal) => Promise<T>,
): Promise<T> {
  const controller = new AbortController();
  const cancel = () => controller.abort();
  signal?.addEventListener("abort", cancel, { once: true });
  if (signal?.aborted) cancel();
  let timedOut = false;
  const timer = window.setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, 15_000);
  try {
    const result = await request(controller.signal);
    if (controller.signal.aborted)
      throw new DOMException("Cancelled", "AbortError");
    return result;
  } catch (error) {
    if (timedOut && !signal?.aborted)
      throw new BackendRequestError(408, "trip_restore_timeout");
    throw error;
  } finally {
    window.clearTimeout(timer);
    signal?.removeEventListener("abort", cancel);
  }
}

interface RequestOptions {
  anonymousSessionId?: string;
  idempotencyKey?: string;
  keepalive?: boolean;
  signal?: AbortSignal;
}

export interface TripRealtimeCallbacks {
  onOpen: () => void;
  onEvent: (event: ServerEvent) => void;
  onClose: () => void;
  onTransportError: (code: string) => void;
}

export interface TripRealtimeConnection {
  send(command: ClientCommand): boolean;
  close(): void;
}

export type WebSocketFactory = (url: string) => WebSocket;

export interface V4TripRealtimeCallbacks {
  onOpen: () => void;
  onEvent: (event: ConversationEventV4) => void;
  onClose: () => void;
  onTransportError: (code: string, retryable: boolean) => void;
}

export interface V4TripRealtimeConnection {
  send(command: V4ClientCommand): boolean;
  close(): void;
}

export function connectTripRealtime(
  url: string,
  callbacks: TripRealtimeCallbacks,
  createSocket: WebSocketFactory = (target) => new WebSocket(target),
): TripRealtimeConnection {
  const socket = createSocket(url);
  socket.addEventListener("open", callbacks.onOpen);
  socket.addEventListener("close", callbacks.onClose);
  socket.addEventListener("error", () =>
    callbacks.onTransportError("websocket_transport_error"),
  );
  socket.addEventListener("message", (message) => {
    let payload: unknown;
    try {
      payload = JSON.parse(String(message.data));
    } catch {
      callbacks.onTransportError("invalid_websocket_json");
      return;
    }
    if (!validatePublicContract("server_event", payload).success) {
      if (
        typeof payload === "object" &&
        payload !== null &&
        "type" in payload &&
        payload.type === "transport.error"
      ) {
        callbacks.onTransportError("transport_error");
      } else {
        callbacks.onTransportError("invalid_server_event");
      }
      return;
    }
    callbacks.onEvent(payload as ServerEvent);
  });
  return {
    send(command) {
      if (
        socket.readyState !== WebSocket.OPEN ||
        !validatePublicContract("client_command", command).success
      ) {
        return false;
      }
      socket.send(JSON.stringify(command));
      return true;
    },
    close() {
      socket.close(1000, "runtime_disposed");
    },
  };
}

export function connectV4TripRealtime(
  url: string,
  callbacks: V4TripRealtimeCallbacks,
  createSocket: WebSocketFactory = (target) => new WebSocket(target),
): V4TripRealtimeConnection {
  const socket = createSocket(url);
  socket.addEventListener("open", callbacks.onOpen);
  socket.addEventListener("close", callbacks.onClose);
  socket.addEventListener("error", () =>
    callbacks.onTransportError("websocket_transport_error", true),
  );
  socket.addEventListener("message", (message) => {
    let payload: unknown;
    try {
      payload = JSON.parse(String(message.data));
    } catch {
      callbacks.onTransportError("invalid_websocket_json", false);
      return;
    }
    if (validateV4Contract("conversation_event", payload).success) {
      callbacks.onEvent(payload as ConversationEventV4);
      return;
    }
    if (
      typeof payload === "object" &&
      payload !== null &&
      "type" in payload &&
      payload.type === "transport.error"
    ) {
      const transport = payload as {
        code?: unknown;
        retryable?: unknown;
      };
      callbacks.onTransportError(
        typeof transport.code === "string" ? transport.code : "transport_error",
        transport.retryable === true,
      );
      return;
    }
    callbacks.onTransportError("invalid_v4_server_event", false);
  });
  return {
    send(command) {
      if (
        socket.readyState !== WebSocket.OPEN ||
        !validateV4Contract("client_command", command).success
      ) {
        return false;
      }
      socket.send(JSON.stringify(command));
      return true;
    },
    close() {
      socket.close(1000, "runtime_disposed");
    },
  };
}
