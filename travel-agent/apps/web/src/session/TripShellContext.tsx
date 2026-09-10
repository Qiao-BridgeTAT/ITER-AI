/* eslint-disable react-refresh/only-export-components */
import { createContext, type ReactNode, useContext, useMemo } from "react";
import { matchPath, useLocation } from "react-router-dom";

import type { TripShell } from "../generated/contracts";
import { TripShellRepository } from "./tripShellRepository";
import { useViewerSession } from "./viewerSession";

interface TripShellContextValue {
  shell: TripShell;
  created: boolean;
  expiredPrevious: boolean;
  needsBackendCreation: boolean;
}

const TripShellContext = createContext<TripShellContextValue | null>(null);

export function TripShellProvider({ children }: { children: ReactNode }) {
  const viewer = useViewerSession();
  const location = useLocation();
  const value = useMemo(() => {
    const repository = new TripShellRepository({
      sessionStorage: window.sessionStorage,
      persistentStorage: window.localStorage,
    });
    const route = matchPath("/trips/:tripId/*", location.pathname);
    const routeTripId = route?.params.tripId;
    const requestedTripId =
      routeTripId && UUID_PATTERN.test(routeTripId) ? routeTripId : undefined;
    return repository.resolve(viewer, requestedTripId);
  }, [location.pathname, viewer]);

  return (
    <TripShellContext.Provider value={value}>
      {children}
    </TripShellContext.Provider>
  );
}

const UUID_PATTERN =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

export function useTripShell(): TripShellContextValue {
  const value = useContext(TripShellContext);
  if (value === null) {
    throw new Error("useTripShell must be used inside TripShellProvider");
  }
  return value;
}
