import type { TripShell, TripState } from "../generated/contracts";
import { validatePublicContract } from "../contracts/validation";
import type { ViewerSession } from "./viewerSession";

export const ANONYMOUS_SHELL_STORAGE_KEY =
  "travel-agent.anonymous-trip-shell.v1";
export const USER_SHELL_STORAGE_PREFIX = "travel-agent.user-trip-shell.v1";
export const DEFAULT_ANONYMOUS_SHELL_TTL_MS = 4 * 60 * 60 * 1000;

interface StoredTripShell {
  record_version: "1";
  expires_at: string | null;
  shell: TripShell;
  backend_creation_pending?: boolean;
}

export interface TripShellResolution {
  shell: TripShell;
  created: boolean;
  expiredPrevious: boolean;
  needsBackendCreation: boolean;
}

export interface TripShellRepositoryOptions {
  sessionStorage: Storage;
  persistentStorage: Storage;
  now?: () => Date;
  createId?: () => string;
  anonymousTtlMs?: number;
}

export class TripShellRepository {
  private readonly sessionStorage: Storage;
  private readonly persistentStorage: Storage;
  private readonly now: () => Date;
  private readonly createId: () => string;
  private readonly anonymousTtlMs: number;

  constructor(options: TripShellRepositoryOptions) {
    this.sessionStorage = options.sessionStorage;
    this.persistentStorage = options.persistentStorage;
    this.now = options.now ?? (() => new Date());
    this.createId = options.createId ?? (() => crypto.randomUUID());
    this.anonymousTtlMs =
      options.anonymousTtlMs ?? DEFAULT_ANONYMOUS_SHELL_TTL_MS;
  }

  resolve(
    viewer: ViewerSession,
    requestedTripId?: string
  ): TripShellResolution {
    const storage =
      viewer.kind === "anonymous"
        ? this.sessionStorage
        : this.persistentStorage;
    const key = storageKey(viewer);
    const previous = this.read(storage, key, viewer);
    if (
      previous.record !== null &&
      (requestedTripId === undefined ||
        previous.record.shell.trip_id === requestedTripId)
    ) {
      return {
        shell: previous.record.shell,
        created: false,
        expiredPrevious: false,
        needsBackendCreation: previous.record.backend_creation_pending === true
      };
    }

    if (previous.record !== null && requestedTripId !== undefined) {
      return {
        shell: this.createShell(viewer, requestedTripId),
        created: false,
        expiredPrevious: false,
        needsBackendCreation: false
      };
    }

    const shell = this.createShell(viewer, requestedTripId);
    const expiresAt =
      viewer.kind === "anonymous"
        ? new Date(this.now().getTime() + this.anonymousTtlMs).toISOString()
        : null;
    const record: StoredTripShell = {
      record_version: "1",
      expires_at: expiresAt,
      shell,
      // Only locally initiated trips may be created. An arbitrary signed-in
      // history/deep link must never become a new empty trip after a 404.
      backend_creation_pending:
        requestedTripId === undefined || viewer.kind === "anonymous"
    };
    storage.setItem(key, JSON.stringify(record));
    return {
      shell,
      created: true,
      expiredPrevious: previous.expired,
      needsBackendCreation: record.backend_creation_pending === true
    };
  }

  acknowledgeBackendCreation(
    shell: Pick<TripShell, "trip_id" | "owner_type" | "owner_id">
  ): void {
    const storage =
      shell.owner_type === "anonymous"
        ? this.sessionStorage
        : this.persistentStorage;
    const key =
      shell.owner_type === "anonymous"
        ? ANONYMOUS_SHELL_STORAGE_KEY
        : userShellStorageKey(shell.owner_id);
    const raw = storage.getItem(key);
    if (raw === null) return;
    try {
      const record: unknown = JSON.parse(raw);
      if (!isRecord(record) || !isRecord(record.shell)) return;
      if (
        record.shell.trip_id !== shell.trip_id ||
        record.shell.owner_type !== shell.owner_type ||
        record.shell.owner_id !== shell.owner_id
      )
        return;
      storage.setItem(
        key,
        JSON.stringify({
          ...record,
          backend_creation_pending: false
        })
      );
    } catch {
      // Invalid local storage is handled by resolve; never alter another shell.
    }
  }

  startNew(viewer: ViewerSession): TripShell {
    const shell = this.createShell(viewer);
    const storage =
      viewer.kind === "anonymous"
        ? this.sessionStorage
        : this.persistentStorage;
    const record: StoredTripShell = {
      record_version: "1",
      expires_at:
        viewer.kind === "anonymous"
          ? new Date(this.now().getTime() + this.anonymousTtlMs).toISOString()
          : null,
      shell,
      backend_creation_pending: true
    };
    storage.setItem(storageKey(viewer), JSON.stringify(record));
    return shell;
  }

  clear(viewer: ViewerSession): void {
    const storage =
      viewer.kind === "anonymous"
        ? this.sessionStorage
        : this.persistentStorage;
    storage.removeItem(storageKey(viewer));
  }

  activateUserShell(userId: string, shell: TripShell): boolean {
    if (
      shell.owner_type !== "user" ||
      shell.owner_id !== userId ||
      !validatePublicContract("trip_shell", shell).success
    ) {
      return false;
    }
    const record: StoredTripShell = {
      record_version: "1",
      expires_at: null,
      shell: structuredClone(shell)
    };
    this.persistentStorage.setItem(
      userShellStorageKey(userId),
      JSON.stringify(record)
    );
    return true;
  }

  activateAnonymousState(sessionId: string, state: TripState): boolean {
    if (
      state.owner_type !== "anonymous" ||
      state.owner_id !== sessionId ||
      !validatePublicContract("trip_state", state).success
    ) {
      return false;
    }
    const shell: TripShell = {
      trip_id: state.trip_id,
      owner_type: "anonymous",
      owner_id: sessionId,
      city: state.city,
      phase: state.phase,
      state_version: state.state_version
    };
    if (!validatePublicContract("trip_shell", shell).success) return false;
    const record: StoredTripShell = {
      record_version: "1",
      expires_at: new Date(
        this.now().getTime() + this.anonymousTtlMs
      ).toISOString(),
      shell
    };
    this.sessionStorage.setItem(
      ANONYMOUS_SHELL_STORAGE_KEY,
      JSON.stringify(record)
    );
    return true;
  }

  private read(
    storage: Storage,
    key: string,
    viewer: ViewerSession
  ): { record: StoredTripShell | null; expired: boolean } {
    const raw = storage.getItem(key);
    if (raw === null) {
      return { record: null, expired: false };
    }
    try {
      const parsed: unknown = JSON.parse(raw);
      if (!isStoredTripShell(parsed, viewer)) {
        storage.removeItem(key);
        return { record: null, expired: false };
      }
      if (
        parsed.expires_at !== null &&
        new Date(parsed.expires_at).getTime() <= this.now().getTime()
      ) {
        storage.removeItem(key);
        return { record: null, expired: true };
      }
      return { record: parsed, expired: false };
    } catch {
      storage.removeItem(key);
      return { record: null, expired: false };
    }
  }

  private createShell(
    viewer: ViewerSession,
    requestedTripId?: string
  ): TripShell {
    const shell: TripShell = {
      trip_id: requestedTripId ?? this.createId(),
      owner_type: viewer.kind === "anonymous" ? "anonymous" : "user",
      owner_id:
        viewer.kind === "anonymous"
          ? `anonymous-${this.createId()}`
          : viewer.account.user_id,
      city: null,
      phase:
        viewer.kind === "user" && viewer.hasPersonalDefaults
          ? "city_selection"
          : "cold_start",
      state_version: 0
    };
    if (!validatePublicContract("trip_shell", shell).success) {
      throw new Error("Unable to create a valid trip shell");
    }
    return shell;
  }
}

function storageKey(viewer: ViewerSession): string {
  return viewer.kind === "anonymous"
    ? ANONYMOUS_SHELL_STORAGE_KEY
    : userShellStorageKey(viewer.account.user_id);
}

export function userShellStorageKey(userId: string): string {
  return `${USER_SHELL_STORAGE_PREFIX}:${userId}`;
}

function isStoredTripShell(
  value: unknown,
  viewer: ViewerSession
): value is StoredTripShell {
  if (!isRecord(value) || value.record_version !== "1") {
    return false;
  }
  if (value.expires_at !== null && typeof value.expires_at !== "string") {
    return false;
  }
  if (
    value.backend_creation_pending !== undefined &&
    typeof value.backend_creation_pending !== "boolean"
  )
    return false;
  if (!validatePublicContract("trip_shell", value.shell).success) {
    return false;
  }
  const shell = value.shell as TripShell;
  return viewer.kind === "anonymous"
    ? shell.owner_type === "anonymous"
    : shell.owner_type === "user" && shell.owner_id === viewer.account.user_id;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
