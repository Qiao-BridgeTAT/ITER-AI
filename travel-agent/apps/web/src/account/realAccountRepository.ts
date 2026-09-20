import {
  BackendRequestError,
  TravelApiClient
} from "../backend/travelApiClient";
import { validatePublicContract } from "../contracts/validation";
import type {
  AccountView,
  ColdStartSubmission,
  PreferenceValue,
  TripListItem,
  TripShell,
  TripState
} from "../generated/contracts";

export interface RealViewerSnapshot {
  account: AccountView;
  history: TripListItem[];
  historyStatus: "ready" | "error";
  preferences: PreferenceValue[];
  personalDefaults?: ColdStartSubmission;
}

export interface RealSignInResult {
  viewer: RealViewerSnapshot;
  attachedTrip: TripShell | TripState | null;
}

export class RealAccountRepository {
  constructor(private readonly api = new TravelApiClient()) {}

  async restore(): Promise<RealViewerSnapshot | null> {
    try {
      return await this.loadViewer();
    } catch (error) {
      if (error instanceof BackendRequestError && error.status === 401) {
        return null;
      }
      throw error;
    }
  }

  async sendCode(phone: string): Promise<void> {
    await this.api.sendSms(phone, `web:sms:${crypto.randomUUID()}`);
  }

  async verify(phone: string, code: string): Promise<RealViewerSnapshot> {
    await this.api.verifySms(
      { phone, code },
      `web:verify:${crypto.randomUUID()}`
    );
    return this.loadViewer();
  }

  async verifyAndAttach(
    phone: string,
    code: string,
    anonymousSessionId: string | null,
    tripId: string,
    migrateAnonymousTrip: boolean
  ): Promise<RealSignInResult> {
    await this.api.verifySms(
      { phone, code },
      `web:verify:${crypto.randomUUID()}`
    );
    let attachedTrip: TripShell | TripState | null = null;
    if (anonymousSessionId && migrateAnonymousTrip) {
      await this.api.attachAnonymousTrip(
        anonymousSessionId,
        tripId,
        `web:attach-trip:${tripId}`
      );
      attachedTrip = await this.api.getTripShell(tripId);
    }
    const viewer = await this.loadViewer();
    if (attachedTrip !== null) return { viewer, attachedTrip };
    const mostRecentTrip = viewer.history[0];
    return {
      viewer,
      attachedTrip: mostRecentTrip
        ? await this.api.getTripShell(mostRecentTrip.trip_id)
        : null
    };
  }

  async logout(): Promise<void> {
    await this.api.logout(`web:logout:${crypto.randomUUID()}`);
  }

  async updateNickname(nickname: string): Promise<AccountView> {
    return this.api.updateAccount(
      nickname,
      `web:update-account:${crypto.randomUUID()}`
    );
  }

  async listTrips(): Promise<TripListItem[]> {
    return (await this.api.listTrips()).trips ?? [];
  }

  async loadTrip(tripId: string): Promise<TripState> {
    return (await this.api.getTrip(tripId)).state;
  }

  async loadTripShell(
    tripId: string,
    signal?: AbortSignal
  ): Promise<TripShell> {
    return this.api.getTripShell(tripId, signal);
  }

  async createTrip(
    tripId: string,
    anonymousSessionId?: string
  ): Promise<TripState> {
    return (
      await this.api.createTrip(
        tripId,
        `web:create-trip:${tripId}`,
        anonymousSessionId,
        AbortSignal.timeout(15_000)
      )
    ).state;
  }

  async saveColdStart(
    preferences: ColdStartSubmission
  ): Promise<RealViewerSnapshot> {
    await this.api.saveColdStartPreference(
      preferences,
      `web:cold-start-preference:${crypto.randomUUID()}`
    );
    return this.loadViewer();
  }

  async saveAnonymousColdStart(
    preferences: ColdStartSubmission,
    anonymousSessionId?: string
  ): Promise<void> {
    await this.api.saveColdStartPreference(
      preferences,
      `web:anonymous-cold-start:${crypto.randomUUID()}`,
      anonymousSessionId
    );
  }

  async updatePreferences(
    preferences: PreferenceValue[]
  ): Promise<RealViewerSnapshot> {
    await this.api.patchPreferences(
      preferences,
      `web:preferences:${crypto.randomUUID()}`
    );
    return this.loadViewer();
  }

  private async loadViewer(): Promise<RealViewerSnapshot> {
    const [account, preferenceList] = await Promise.all([
      this.api.getAccount(),
      this.api.getPreferences()
    ]);
    let history: TripListItem[] = [];
    let historyStatus: RealViewerSnapshot["historyStatus"] = "ready";
    try {
      history = await this.listTrips();
    } catch {
      historyStatus = "error";
    }
    const preferences = preferenceList.preferences ?? [];
    const personalDefaults = parseColdStartPreference(preferences);
    return {
      account,
      history,
      historyStatus,
      preferences,
      ...(personalDefaults ? { personalDefaults } : {})
    };
  }
}

export function parseColdStartPreference(
  preferences: readonly PreferenceValue[]
): ColdStartSubmission | undefined {
  const value = preferences.find(
    (preference) => preference.key === "cold_start" && preference.active
  )?.value;
  if (!value) return undefined;
  try {
    const parsed: unknown = JSON.parse(value);
    return validatePublicContract("cold_start_submission", parsed).success
      ? (parsed as ColdStartSubmission)
      : undefined;
  } catch {
    return undefined;
  }
}
