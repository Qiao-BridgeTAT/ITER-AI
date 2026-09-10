/* eslint-disable react-refresh/only-export-components */
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { useLocation, useNavigate } from "react-router-dom";
import type { ClientCommand } from "../generated/contracts";
import type {
  ConversationEventV4,
  ConversationHistoryWindow,
  ConversationMessageV4,
  PlaceIntroductionView,
  PlannerPlanPreview,
  V4ClientCommand,
} from "../generated/v4/contracts";
import { useTripRuntimeStore } from "../realtime/TripRuntimeContext";
import { useTripShell } from "../session/TripShellContext";
import { TripShellRepository } from "../session/tripShellRepository";
import {
  pathWithAgentProtocol,
  requestedAgentProtocol,
  type AgentProtocol,
} from "./agentProtocol";
import { ANONYMOUS_SESSION_KEY } from "./storageKeys";
import {
  BackendRequestError,
  connectTripRealtime,
  connectV4TripRealtime,
  TravelApiClient,
} from "./travelApiClient";
import {
  useConversationHistory,
  type AppliedRestoreTimings,
} from "./useConversationHistory";
export type BackendMode = "mock" | "real";
export type BackendCommand = ClientCommand | V4ClientCommand;
type BackendStatus =
  "mock" | "connecting" | "connected" | "recovering" | "failed";
interface TripBackendContextValue {
  mode: BackendMode;
  requestedProtocol: AgentProtocol;
  protocol: AgentProtocol;
  status: BackendStatus;
  errorCode: string | null;
  sendCommand: (command: BackendCommand) => boolean;
  recover: () => Promise<boolean>;
  getPlanPreview: (planVersionId: string) => Promise<PlannerPlanPreview>;
  getPlaceIntroductions: (
    scopeKind: PlaceIntroductionView["scope_kind"],
    scopeId: string,
    signal?: AbortSignal,
  ) => Promise<PlaceIntroductionView>;
  historyWindow: ConversationHistoryWindow | null;
  restoreTimings: AppliedRestoreTimings | null;
  historyLoading: boolean;
  historyError: string | null;
  loadOlderMessages: () => Promise<void>;
  getHistoryMessage: (
    messageId: string,
    signal?: AbortSignal,
  ) => Promise<ConversationMessageV4>;
  retryConnection: () => void;
}
const TripBackendContext = createContext<TripBackendContextValue | null>(null);
interface BackendRealtimeConnection {
  send(command: BackendCommand): boolean;
  close(): void;
}
export function TripBackendProvider({
  children,
  mode,
  client,
}: {
  children: ReactNode;
  mode: BackendMode;
  client?: TravelApiClient;
}) {
  const { shell, created, needsBackendCreation } = useTripShell();
  const location = useLocation();
  const navigate = useNavigate();
  const store = useTripRuntimeStore();
  const api = useMemo(() => client ?? new TravelApiClient(), [client]);
  const requestedProtocol = requestedAgentProtocol(location.search);
  const isConversationRoute = location.pathname.startsWith("/trips/");
  const protocol: AgentProtocol =
    requestedProtocol === "v4" &&
    (shell.owner_type === "user" || isConversationRoute)
      ? "v4"
      : "v2";
  const [status, setStatus] = useState<BackendStatus>("connecting");
  const connection = useRef<BackendRealtimeConnection | null>(null);
  const [errorCode, setErrorCode] = useState<string | null>(null);
  const permanentError = useRef<string | null>(null);
  const anonymousSessionId = useRef<string | null>(null);
  const anonymousSessionRequestKey = useRef(
    `web:anonymous-session:${crypto.randomUUID()}`,
  );
  const replacementTripId = useRef(crypto.randomUUID());
  const recoveryRetryable = useRef(true);
  const activeTripId = useRef(shell.trip_id);
  activeTripId.current = shell.trip_id;
  const recoveryRequest = useRef<AbortController | null>(null);
  const [restartCount, setRestartCount] = useState(0);
  const retryConnection = useCallback(
    () => setRestartCount((count) => count + 1),
    [],
  );
  const {
    restoreView,
    historyWindow,
    restoreTimings,
    historyLoading,
    historyError,
    loadOlderMessages,
    getHistoryMessage,
    cancelHistory,
  } = useConversationHistory(api, shell.trip_id, store, anonymousSessionId);
  const recover = useCallback(async () => {
    if (permanentError.current !== null) return false;
    setStatus("recovering");
    recoveryRequest.current?.abort();
    const controller = new AbortController();
    recoveryRequest.current = controller;
    try {
      let recovered: boolean;
      if (protocol === "v4") {
        const view = await api.getV4ConversationView(
          shell.trip_id,
          anonymousSessionId.current ?? undefined,
          controller.signal,
        );
        if (controller.signal.aborted || activeTripId.current !== shell.trip_id)
          return false;
        recovered = restoreView(view, true);
      } else {
        const snapshot = await api.getTrip(
          shell.trip_id,
          anonymousSessionId.current ?? undefined,
        );
        if (activeTripId.current !== shell.trip_id) return false;
        recovered = store.getState().recoverFromSnapshot(snapshot.state);
      }
      recoveryRetryable.current = true;
      setErrorCode(recovered ? null : "invalid_v4_backend_contract");
      setStatus(
        recovered
          ? connection.current
            ? "connected"
            : "connecting"
          : "failed",
      );
      return recovered;
    } catch (error) {
      if (controller.signal.aborted || activeTripId.current !== shell.trip_id)
        return false;
      recoveryRetryable.current = shouldRetryBackendError(error);
      if (error instanceof BackendRequestError) {
        setErrorCode(error.code);
        if (!recoveryRetryable.current) permanentError.current = error.code;
      }
      setStatus("failed");
      return false;
    }
  }, [api, mode, protocol, shell.trip_id, store, restoreView]);
  useEffect(() => {
    const loadedTripId =
      store.getState().v4.tripState?.semantic_state.trip_id ??
      store.getState().tripState?.trip_id;
    if (loadedTripId && loadedTripId !== shell.trip_id) {
      // A new trip must not briefly display the previous trip's messages/map
      // while its own authoritative snapshot is still loading.
      store.getState().reset();
    }
    permanentError.current = null;
    recoveryRetryable.current = true;
    setErrorCode(null);
    setStatus("connecting");
    let disposed = false;
    const bootstrapController = new AbortController();
    let reconnectTimer: number | null = null;
    let reconnectAttempt = 0;
    // Task-book confirmation commits a Prepare turn and immediately starts a
    // Planner turn on the same socket. Preserve the second turn while the
    // first turn's authoritative snapshot is being reloaded.
    let v4RecoveryPromise: Promise<boolean> | null = null;
    const queuedV4Events: ConversationEventV4[] = [];
    const clearReconnectTimer = () => {
      if (reconnectTimer === null) return;
      window.clearTimeout(reconnectTimer);
      reconnectTimer = null;
    };
    const reconnectDelay = () =>
      Math.min(250 * 2 ** Math.min(reconnectAttempt, 4), 4000);
    function recoverV4AndReplay(): Promise<boolean> {
      if (v4RecoveryPromise !== null) return v4RecoveryPromise;
      const recovery = (async () => {
        let refreshAgain = true;
        while (!disposed && refreshAgain) {
          const recovered = await recover();
          if (!recovered || disposed) return false;
          refreshAgain = false;
          while (queuedV4Events.length > 0) {
            const event = queuedV4Events.shift();
            if (event === undefined) break;
            const action = store.getState().receiveV4Event(event);
            if (action === "refresh_snapshot") {
              refreshAgain = true;
              break;
            }
          }
        }
        return !disposed;
      })();
      v4RecoveryPromise = recovery.finally(() => {
        v4RecoveryPromise = null;
      });
      return v4RecoveryPromise;
    }
    function recoverCurrentProtocol(): Promise<boolean> {
      return protocol === "v4" ? recoverV4AndReplay() : recover();
    }
    function scheduleRecovery() {
      if (
        disposed ||
        permanentError.current !== null ||
        reconnectTimer !== null
      )
        return;
      setStatus("recovering");
      reconnectTimer = window.setTimeout(() => {
        reconnectTimer = null;
        void recoverCurrentProtocol().then((recovered) => {
          if (disposed) return;
          if (recovered) {
            reconnectAttempt = 0;
            openConnection();
            return;
          }
          if (recoveryRetryable.current) {
            reconnectAttempt += 1;
            scheduleRecovery();
          }
        });
      }, reconnectDelay());
    }
    function openConnection() {
      if (
        disposed ||
        permanentError.current !== null ||
        connection.current !== null
      )
        return;
      if (protocol === "v4") {
        const v4Connection = connectV4TripRealtime(
          api.websocketUrl(shell.trip_id),
          {
            onOpen: () => {
              if (disposed) return;
              reconnectAttempt = 0;
              clearReconnectTimer();
              setStatus("connected");
            },
            onEvent: (event) => {
              if (disposed) return;
              if (v4RecoveryPromise !== null) {
                queuedV4Events.push(event);
                return;
              }
              const action = store.getState().receiveV4Event(event);
              if (action === "refresh_snapshot") {
                void recoverV4AndReplay().then((recovered) => {
                  if (!recovered && recoveryRetryable.current)
                    scheduleRecovery();
                });
              }
            },
            onClose: () => {
              connection.current = null;
              if (disposed) return;
              store.getState().markDisconnected();
              if (permanentError.current !== null) return;
              setStatus("recovering");
              void recoverCurrentProtocol().then((recovered) => {
                if (disposed) return;
                if (recovered) {
                  reconnectAttempt = 0;
                  openConnection();
                } else if (recoveryRetryable.current) {
                  reconnectAttempt += 1;
                  scheduleRecovery();
                }
              });
            },
            onTransportError: (code, retryable) => {
              if (disposed) return;
              recoveryRetryable.current = retryable;
              setErrorCode(code);
              if (!retryable) {
                permanentError.current = code;
                clearReconnectTimer();
                store.getState().markDisconnected();
              }
              setStatus(retryable ? "recovering" : "failed");
              connection.current?.close();
            },
          },
        );
        connection.current = {
          send: (command) =>
            command.protocol_version === "v4" &&
            v4Connection.send(command as V4ClientCommand),
          close: () => v4Connection.close(),
        };
        return;
      }
      const v2Connection = connectTripRealtime(
        api.websocketUrl(shell.trip_id),
        {
          onOpen: () => {
            if (disposed) return;
            reconnectAttempt = 0;
            clearReconnectTimer();
            setStatus("connected");
          },
          onEvent: (event) => {
            if (disposed) return;
            const action = store.getState().receiveEvent(event);
            if (action === "refresh_snapshot") {
              void recover().then((recovered) => {
                if (!recovered && recoveryRetryable.current) scheduleRecovery();
              });
            }
          },
          onClose: () => {
            connection.current = null;
            if (disposed) return;
            store.getState().markDisconnected();
            setStatus("recovering");
            void recover().then((recovered) => {
              if (disposed) return;
              if (recovered) {
                reconnectAttempt = 0;
                openConnection();
              } else if (recoveryRetryable.current) {
                reconnectAttempt += 1;
                scheduleRecovery();
              }
            });
          },
          onTransportError: () => {
            if (disposed) return;
            setStatus("recovering");
            connection.current?.close();
          },
        },
      );
      connection.current = {
        send: (command) =>
          command.protocol_version !== "v4"
            ? v2Connection.send(command as ClientCommand)
            : false,
        close: () => v2Connection.close(),
      };
    }
    async function bootstrap() {
      try {
        let sessionId = window.sessionStorage.getItem(ANONYMOUS_SESSION_KEY);
        let createdAnonymousSession = false;
        let changedAnonymousSession = false;
        if (shell.owner_type === "anonymous") {
          const previousSessionId = sessionId;
          const session = await api.createAnonymousSession(
            anonymousSessionRequestKey.current,
            previousSessionId ?? undefined,
          );
          sessionId = session.session_id;
          createdAnonymousSession = previousSessionId === null;
          changedAnonymousSession =
            previousSessionId !== null && previousSessionId !== sessionId;
          if (changedAnonymousSession && isConversationRoute) {
            throw new BackendRequestError(401, "v4_anonymous_session_required");
          }
          window.sessionStorage.setItem(ANONYMOUS_SESSION_KEY, sessionId);
        }
        anonymousSessionId.current = sessionId;
        if (protocol === "v4") {
          if (needsBackendCreation || createdAnonymousSession) {
            await api.createTrip(
              shell.trip_id,
              `web:create-trip:${shell.trip_id}`,
              shell.owner_type === "anonymous"
                ? (sessionId ?? undefined)
                : undefined,
            );
          }
          let snapshot;
          try {
            snapshot = await api.getV4ConversationView(
              shell.trip_id,
              shell.owner_type === "anonymous"
                ? (sessionId ?? undefined)
                : undefined,
              bootstrapController.signal,
            );
          } catch (error) {
            if (
              shell.owner_type !== "anonymous" ||
              !(error instanceof BackendRequestError) ||
              error.status !== 404
            ) {
              throw error;
            }
            if (
              !created &&
              !createdAnonymousSession &&
              !changedAnonymousSession
            ) {
              await api.createTrip(
                shell.trip_id,
                `web:create-trip:${shell.trip_id}`,
                sessionId ?? undefined,
              );
              try {
                snapshot = await api.getV4ConversationView(
                  shell.trip_id,
                  sessionId ?? undefined,
                  bootstrapController.signal,
                );
              } catch (retryError) {
                if (
                  !(retryError instanceof BackendRequestError) ||
                  retryError.status !== 404
                ) {
                  throw retryError;
                }
              }
            }
            if (snapshot === undefined) {
              const nextTripId = replacementTripId.current;
              replacementTripId.current = crypto.randomUUID();
              const replacement = await api.createTrip(
                nextTripId,
                `web:create-trip:${nextTripId}`,
                sessionId ?? undefined,
              );
              const activated = new TripShellRepository({
                sessionStorage: window.sessionStorage,
                persistentStorage: window.localStorage,
              }).activateAnonymousState(sessionId ?? "", replacement.state);
              if (!activated) {
                throw new BackendRequestError(
                  502,
                  "invalid_anonymous_replacement_trip",
                );
              }
              if (!disposed) {
                navigate(
                  pathWithAgentProtocol(
                    `/trips/${nextTripId}`,
                    requestedProtocol,
                  ),
                  { replace: true },
                );
              }
              return;
            }
          }
          if (disposed) return;
          if (!restoreView(snapshot, false))
            throw new BackendRequestError(502, "invalid_v4_backend_contract");
          new TripShellRepository({
            sessionStorage: window.sessionStorage,
            persistentStorage: window.localStorage,
          }).acknowledgeBackendCreation({
            trip_id: shell.trip_id,
            owner_type: shell.owner_type,
            owner_id: shell.owner_id,
          });
          openConnection();
          return;
        }
        let snapshot;
        try {
          snapshot = await api.getTrip(shell.trip_id, sessionId ?? undefined);
        } catch (error) {
          if (shell.owner_type !== "anonymous" && !created) throw error;
          snapshot = await api.createTrip(
            shell.trip_id,
            `web:create-trip:${shell.trip_id}`,
            sessionId ?? undefined,
          );
        }
        if (disposed || !store.getState().loadSnapshot(snapshot.state)) return;
        openConnection();
      } catch (error) {
        if (disposed || bootstrapController.signal.aborted) return;
        if (
          !disposed &&
          reconnectAttempt < 2 &&
          shouldRetryBackendError(error)
        ) {
          reconnectAttempt += 1;
          setStatus("recovering");
          if (reconnectTimer === null) {
            reconnectTimer = window.setTimeout(() => {
              reconnectTimer = null;
              void bootstrap();
            }, reconnectDelay());
          }
        } else if (!disposed) {
          const code =
            error instanceof BackendRequestError
              ? error.code
              : "connection_failed";
          permanentError.current = code;
          setErrorCode(code);
          setStatus("failed");
        }
      }
    }
    void bootstrap();
    // Active typing/planning keeps the existing authenticated trip alive. Idle or
    // hidden tabs do not indefinitely renew a guest session, and GET cannot create one.
    let lastActivity = Date.now();
    let heartbeatRunning = false;
    const onActivity = () => {
      lastActivity = Date.now();
    };
    window.addEventListener("pointerdown", onActivity);
    window.addEventListener("keydown", onActivity);
    const heartbeat = window.setInterval(() => {
      if (
        disposed ||
        protocol !== "v4" ||
        shell.owner_type !== "anonymous" ||
        document.visibilityState !== "visible" ||
        permanentError.current !== null ||
        connection.current === null ||
        heartbeatRunning
      )
        return;
      const running = store.getState().v4.cursor.currentGenerationId !== null;
      if (!running && Date.now() - lastActivity > 5 * 60000) return;
      heartbeatRunning = true;
      void api
        .getV4ConversationView(
          shell.trip_id,
          anonymousSessionId.current ?? undefined,
          bootstrapController.signal,
        )
        .catch((error: unknown) => {
          if (disposed || shouldRetryBackendError(error)) return;
          const code =
            error instanceof BackendRequestError
              ? error.code
              : "connection_failed";
          permanentError.current = code;
          setErrorCode(code);
          setStatus("failed");
          connection.current?.close();
        })
        .finally(() => {
          heartbeatRunning = false;
        });
    }, 60000);
    return () => {
      disposed = true;
      bootstrapController.abort();
      recoveryRequest.current?.abort();
      cancelHistory();
      window.clearInterval(heartbeat);
      window.removeEventListener("pointerdown", onActivity);
      window.removeEventListener("keydown", onActivity);
      clearReconnectTimer();
      queuedV4Events.length = 0;
      connection.current?.close();
      connection.current = null;
    };
  }, [
    api,
    created,
    isConversationRoute,
    needsBackendCreation,
    mode,
    navigate,
    protocol,
    recover,
    restoreView,
    cancelHistory,
    restartCount,
    requestedProtocol,
    shell.owner_type,
    shell.owner_id,
    shell.trip_id,
    store,
  ]);
  const sendCommand = useCallback(
    (command: BackendCommand) => {
      if (status !== "connected") return false;
      const sent = connection.current?.send(command) ?? false;
      if (!sent) return false;
      if (
        protocol === "v2" &&
        (command.type === "user_message" ||
          command.type === "attachment_answer" ||
          command.type === "task_book_confirm")
      ) {
        store.getState().prepareGeneration(command.request_id);
      }
      return true;
    },
    [mode, protocol, status, store],
  );
  const getPlanPreview = useCallback(
    (planVersionId: string) =>
      api.getV4PlanPreview(
        shell.trip_id,
        planVersionId,
        anonymousSessionId.current ?? undefined,
      ),
    [api, shell.trip_id],
  );
  const getPlaceIntroductions = useCallback(
    (
      scopeKind: PlaceIntroductionView["scope_kind"],
      scopeId: string,
      signal?: AbortSignal,
    ) =>
      api.getV4PlaceIntroductions(
        shell.trip_id,
        scopeKind,
        scopeId,
        anonymousSessionId.current ?? undefined,
        signal,
      ),
    [api, shell.trip_id],
  );
  const value = useMemo(
    // Optional copy is fetched independently of trip restore and message delivery.
    () => ({
      mode,
      requestedProtocol,
      protocol,
      status,
      errorCode,
      sendCommand,
      recover,
      getPlanPreview,
      getPlaceIntroductions,
      historyWindow,
      restoreTimings,
      historyLoading,
      historyError,
      loadOlderMessages,
      getHistoryMessage,
      retryConnection,
    }),
    [
      mode,
      protocol,
      recover,
      requestedProtocol,
      sendCommand,
      status,
      errorCode,
      getPlanPreview,
      getPlaceIntroductions,
      historyWindow,
      restoreTimings,
      historyLoading,
      historyError,
      loadOlderMessages,
      getHistoryMessage,
      retryConnection,
    ],
  );
  return (
    <TripBackendContext.Provider value={value}>
      {children}
    </TripBackendContext.Provider>
  );
}
function shouldRetryBackendError(error: unknown): boolean {
  if (!(error instanceof BackendRequestError)) return true;
  if (
    [
      "invalid_v4_backend_contract",
      "invalid_backend_contract",
      "v4_snapshot_incompatible",
      "trip_restore_timeout",
    ].includes(error.code)
  )
    return false;
  return (
    error.status >= 500 ||
    error.status === 408 ||
    error.status === 409 ||
    error.status === 425 ||
    error.status === 429
  );
}
export function useTripBackend(): TripBackendContextValue {
  const value = useContext(TripBackendContext);
  if (value === null)
    throw new Error("useTripBackend must be used inside TripBackendProvider");
  return value;
}
