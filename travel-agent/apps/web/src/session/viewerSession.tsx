/* eslint-disable react-refresh/only-export-components */
import {
  createContext,
  type ReactNode,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { RealAccountRepository } from "../account/realAccountRepository";
import { ANONYMOUS_SESSION_KEY } from "../backend/storageKeys";
import { BackendRequestError } from "../backend/travelApiClient";
import type {
  AccountView,
  ColdStartSubmission,
  PreferenceValue,
  TripListItem,
  TripShell,
  TripState,
} from "../generated/contracts";
import {
  ANONYMOUS_SHELL_STORAGE_KEY,
  TripShellRepository,
} from "./tripShellRepository";
export type ViewerSession =
  | {
      kind: "anonymous";
    }
  | {
      kind: "user";
      account: AccountView;
      hasPersonalDefaults: boolean;
      personalDefaults?: ColdStartSubmission;
      history: TripListItem[];
      historyStatus?: "ready" | "loading" | "error";
      preferences?: PreferenceValue[];
    };
export type TripResumeResult =
  | {
      ok: true;
      protocol?: "v2" | "v4";
    }
  | {
      ok: false;
      reason:
        | "not_found"
        | "forbidden"
        | "invalid"
        | "unavailable"
        | "timeout"
        | "cancelled"
        | "session_expired";
    };
export type NewTripResult =
  | {
      ok: true;
      tripId: string;
      needsColdStart: boolean;
    }
  | {
      ok: false;
      reason: "session_expired" | "invalid" | "unavailable";
    };
const ANONYMOUS_VIEWER: ViewerSession = { kind: "anonymous" };
interface SignInInput {
  phone: string;
  code: string;
  shell: TripShell;
  state?: TripState;
  title: string;
  hasV4Work?: boolean;
}
interface ViewerSessionActions {
  sendVerificationCode: (phone: string) => Promise<boolean>;
  signInAndAttach: (input: SignInInput) => Promise<boolean>;
  signOut: () => Promise<boolean>;
  updateNickname: (nickname: string) => Promise<boolean>;
  updatePersonalDefaults: (
    preferences: ColdStartSubmission,
  ) => Promise<boolean>;
  resumeTrip: (
    tripId: string,
    signal?: AbortSignal,
  ) => Promise<TripResumeResult>;
  startNewTrip: () => Promise<NewTripResult>;
  refreshTripHistory: () => Promise<boolean>;
  updatePreferences: (preferences: PreferenceValue[]) => boolean;
}
interface ViewerSessionContextValue {
  viewer: ViewerSession;
  actions: ViewerSessionActions;
}
const ViewerSessionContext = createContext<ViewerSessionContextValue | null>(
  null,
);
interface ViewerSessionProviderProps {
  children: ReactNode;
  initialViewer?: ViewerSession;
  mode?: "mock" | "real";
}
export function ViewerSessionProvider({
  children,
  initialViewer,
  mode = "real",
}: ViewerSessionProviderProps) {
  const [viewer, setViewer] = useState<ViewerSession>(
    () => initialViewer ?? ANONYMOUS_VIEWER,
  );
  const [restorationComplete, setRestorationComplete] = useState(
    () => initialViewer !== undefined,
  );
  const realRepository = useMemo(() => new RealAccountRepository(), []);
  const latestViewer = useRef(viewer);
  latestViewer.current = viewer;
  const newTripAttempt = useRef<{
    owner: string;
    tripId: string;
  } | null>(null);
  const newTripRequest = useRef<Promise<NewTripResult> | null>(null);
  useEffect(() => {
    if (initialViewer !== undefined) return;
    let active = true;
    void realRepository
      .restore()
      .then((restored) => {
        if (active) {
          setViewer(toViewer(restored));
          setRestorationComplete(true);
        }
      })
      .catch(() => {
        if (active) {
          setViewer(ANONYMOUS_VIEWER);
          setRestorationComplete(true);
        }
      });
    return () => {
      active = false;
    };
  }, [initialViewer, mode, realRepository]);
  const sendVerificationCode = useCallback(
    async (phone: string) => {
      try {
        await realRepository.sendCode(phone);
        return true;
      } catch {
        return false;
      }
    },
    [mode, realRepository],
  );
  const signInAndAttach = useCallback(
    async (input: SignInInput) => {
      try {
        const anonymousSessionId = window.sessionStorage.getItem(
          ANONYMOUS_SESSION_KEY,
        );
        const result = await realRepository.verifyAndAttach(
          input.phone,
          input.code,
          anonymousSessionId,
          input.shell.trip_id,
          input.hasV4Work === true ||
            (input.state !== undefined &&
              hasRecoverableTripWork(input.shell, input.state)),
        );
        if (result.attachedTrip !== null) {
          const activated = new TripShellRepository({
            sessionStorage: window.sessionStorage,
            persistentStorage: window.localStorage,
          }).activateUserShell(
            result.viewer.account.user_id,
            shellFromState(result.attachedTrip),
          );
          if (!activated) return false;
          window.sessionStorage.removeItem(ANONYMOUS_SESSION_KEY);
          window.sessionStorage.removeItem(ANONYMOUS_SHELL_STORAGE_KEY);
          window.sessionStorage.removeItem(
            `travel-agent.cold-start-completed:${input.shell.trip_id}`,
          );
        }
        setViewer(toViewer(result.viewer));
        return true;
      } catch {
        return false;
      }
    },
    [mode, realRepository],
  );
  const signOut = useCallback(async () => {
    try {
      await realRepository.logout();
      setViewer(ANONYMOUS_VIEWER);
      return true;
    } catch {
      return false;
    }
  }, [mode, realRepository]);
  const updateNickname = useCallback(
    async (nickname: string) => {
      if (viewer.kind !== "user") return false;
      try {
        const account = await realRepository.updateNickname(nickname);
        setViewer((current) =>
          current.kind === "user" ? { ...current, account } : current,
        );
        return true;
      } catch {
        return false;
      }
    },
    [mode, realRepository, viewer],
  );
  const updatePersonalDefaults = useCallback(
    async (preferences: ColdStartSubmission) => {
      try {
        if (viewer.kind === "anonymous") {
          const sessionId = window.sessionStorage.getItem(
            ANONYMOUS_SESSION_KEY,
          );
          await realRepository.saveAnonymousColdStart(
            preferences,
            sessionId ?? undefined,
          );
          return true;
        }
        const next = await realRepository.saveColdStart(preferences);
        setViewer(toViewer(next));
        return true;
      } catch {
        return false;
      }
    },
    [mode, realRepository, viewer.kind],
  );
  const resumeTrip = useCallback(
    async (tripId: string, signal?: AbortSignal): Promise<TripResumeResult> => {
      if (signal?.aborted) return { ok: false, reason: "cancelled" };
      if (viewer.kind !== "user") {
        return { ok: false, reason: "forbidden" };
      }
      try {
        const state = await realRepository.loadTripShell(tripId, signal);
        if (signal?.aborted) return { ok: false, reason: "cancelled" };
        if (
          state.trip_id !== tripId ||
          state.owner_type !== "user" ||
          state.owner_id !== viewer.account.user_id
        ) {
          return { ok: false, reason: "invalid" };
        }
        const activated = new TripShellRepository({
          sessionStorage: window.sessionStorage,
          persistentStorage: window.localStorage,
        }).activateUserShell(viewer.account.user_id, shellFromState(state));
        if (!activated) return { ok: false, reason: "invalid" };
        setViewer({ ...viewer });
        // Empty V4 trips still begin with a legacy-compatible shell. Never
        // downgrade their requested protocol merely because no V4 turn exists yet.
        return {
          ok: true,
          protocol: state.protocol_version === "v4" ? "v4" : undefined,
        };
      } catch (error) {
        if (signal?.aborted) return { ok: false, reason: "cancelled" };
        if (error instanceof BackendRequestError) {
          if (error.status === 401)
            return { ok: false, reason: "session_expired" };
          if (error.code === "trip_restore_timeout")
            return { ok: false, reason: "timeout" };
          if (error.status === 404) return { ok: false, reason: "not_found" };
          if (error.status === 403) return { ok: false, reason: "forbidden" };
          if (
            error.code === "invalid_backend_contract" ||
            error.status === 409
          ) {
            return { ok: false, reason: "invalid" };
          }
        }
        return { ok: false, reason: "unavailable" };
      }
    },
    [mode, realRepository, viewer],
  );
  const startNewTrip = useCallback(async (): Promise<NewTripResult> => {
    if (newTripRequest.current) return newTripRequest.current;
    const sessionId =
      viewer.kind === "anonymous"
        ? window.sessionStorage.getItem(ANONYMOUS_SESSION_KEY)
        : null;
    const owner = viewer.kind === "user" ? viewer.account.user_id : sessionId;
    if (owner === null) {
      return { ok: false, reason: "session_expired" };
    }
    // Retrying an uncertain POST must reuse its ID, including after the menu
    // closes, so a delayed successful response cannot create duplicate trips.
    const ownerKey = `${viewer.kind}:${owner ?? "local"}`;
    if (newTripAttempt.current?.owner !== ownerKey) {
      newTripAttempt.current = { owner: ownerKey, tripId: crypto.randomUUID() };
    }
    const attempt = newTripAttempt.current;
    const request = (async (): Promise<NewTripResult> => {
      try {
        const shells = new TripShellRepository({
          sessionStorage: window.sessionStorage,
          persistentStorage: window.localStorage,
        });
        const state = await realRepository.createTrip(
          attempt.tripId,
          sessionId ?? undefined,
        );
        const current = latestViewer.current;
        const currentOwner =
          current.kind === "user"
            ? current.account.user_id
            : window.sessionStorage.getItem(ANONYMOUS_SESSION_KEY);
        if (`${current.kind}:${currentOwner}` !== ownerKey) {
          return { ok: false, reason: "session_expired" };
        }
        if (
          state.trip_id !== attempt.tripId ||
          state.owner_type !== viewer.kind ||
          state.owner_id !== owner
        ) {
          return { ok: false, reason: "invalid" };
        }
        const activated =
          viewer.kind === "user"
            ? shells.activateUserShell(
                viewer.account.user_id,
                shellFromState(state),
              )
            : shells.activateAnonymousState(sessionId ?? "", state);
        if (!activated) return { ok: false, reason: "invalid" };
        newTripAttempt.current = null;
        return {
          ok: true,
          tripId: state.trip_id,
          needsColdStart: state.phase === "cold_start",
        };
      } catch (error) {
        if (error instanceof BackendRequestError) {
          if (error.status === 401 || error.status === 403) {
            return { ok: false, reason: "session_expired" };
          }
          if (error.code === "invalid_backend_contract") {
            return { ok: false, reason: "invalid" };
          }
        }
        return { ok: false, reason: "unavailable" };
      }
    })();
    newTripRequest.current = request;
    try {
      return await request;
    } finally {
      if (newTripRequest.current === request) newTripRequest.current = null;
    }
  }, [mode, realRepository, viewer]);
  const refreshTripHistory = useCallback(async () => {
    if (viewer.kind !== "user") return false;
    setViewer((current) =>
      current.kind === "user"
        ? { ...current, historyStatus: "loading" }
        : current,
    );
    try {
      const history = await realRepository.listTrips();
      setViewer((current) =>
        current.kind === "user"
          ? { ...current, history, historyStatus: "ready" }
          : current,
      );
      return true;
    } catch {
      setViewer((current) =>
        current.kind === "user"
          ? { ...current, historyStatus: "error" }
          : current,
      );
      return false;
    }
  }, [realRepository, viewer.kind]);
  const updatePreferences = useCallback(
    (preferences: PreferenceValue[]) => {
      if (viewer.kind !== "user") return false;
      void realRepository
        .updatePreferences(preferences)
        .then((next) => setViewer(toViewer(next)));
      return true;
    },
    [mode, realRepository, viewer],
  );
  const value = useMemo<ViewerSessionContextValue>(
    () => ({
      viewer,
      actions: {
        sendVerificationCode,
        signInAndAttach,
        signOut,
        updateNickname,
        updatePersonalDefaults,
        resumeTrip,
        startNewTrip,
        refreshTripHistory,
        updatePreferences,
      },
    }),
    [
      resumeTrip,
      startNewTrip,
      refreshTripHistory,
      sendVerificationCode,
      signInAndAttach,
      signOut,
      updateNickname,
      updatePersonalDefaults,
      updatePreferences,
      viewer,
    ],
  );
  return (
    <ViewerSessionContext.Provider value={value}>
      {restorationComplete ? children : null}
    </ViewerSessionContext.Provider>
  );
}
function toViewer(
  snapshot:
    import("../account/realAccountRepository").RealViewerSnapshot | null,
): ViewerSession {
  if (snapshot === null) return ANONYMOUS_VIEWER;
  return {
    kind: "user",
    account: snapshot.account,
    hasPersonalDefaults: snapshot.personalDefaults !== undefined,
    ...(snapshot.personalDefaults
      ? { personalDefaults: snapshot.personalDefaults }
      : {}),
    history: snapshot.history,
    historyStatus: snapshot.historyStatus,
    preferences: snapshot.preferences,
  };
}
export function useViewerSession(): ViewerSession {
  const value = useContext(ViewerSessionContext);
  if (value === null)
    throw new Error("useViewerSession must be used inside its provider");
  return value.viewer;
}
export function useViewerSessionActions(): ViewerSessionActions {
  const value = useContext(ViewerSessionContext);
  if (value === null)
    throw new Error("useViewerSessionActions must be used inside its provider");
  return value.actions;
}
function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
function shellFromState(state: TripState | TripShell): TripShell {
  return {
    trip_id: state.trip_id,
    owner_type: state.owner_type,
    owner_id: state.owner_id,
    city: state.city,
    phase: state.phase,
    state_version: state.state_version,
    protocol_version:
      "protocol_version" in state ? state.protocol_version : "v2",
  };
}
function hasRecoverableTripWork(shell: TripShell, state: TripState): boolean {
  if (shell.trip_id !== state.trip_id || shell.owner_type !== "anonymous") {
    return false;
  }
  return (
    state.cold_start_completed_at !== null ||
    state.city !== null ||
    (state.conversation_messages?.length ?? 0) > 0 ||
    state.task_book !== null ||
    state.itinerary !== null ||
    (state.provider_display?.places?.length ?? 0) > 0 ||
    (state.provider_display?.routes?.length ?? 0) > 0
  );
}
