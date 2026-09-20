/* eslint-disable react-refresh/only-export-components */
import { createContext, type ReactNode, useContext, useRef } from "react";
import { useStore } from "zustand";

import {
  createTripRuntimeStore,
  type TripRuntimeState,
  type TripRuntimeStore
} from "./tripRuntimeStore";

const TripRuntimeContext = createContext<TripRuntimeStore | null>(null);

interface TripRuntimeProviderProps {
  children: ReactNode;
  store?: TripRuntimeStore;
}

export function TripRuntimeProvider({
  children,
  store
}: TripRuntimeProviderProps) {
  const storeRef = useRef<TripRuntimeStore | undefined>(undefined);
  if (storeRef.current === undefined) {
    storeRef.current = store ?? createTripRuntimeStore();
  }
  return (
    <TripRuntimeContext.Provider value={storeRef.current}>
      {children}
    </TripRuntimeContext.Provider>
  );
}

export function useTripRuntime<T>(selector: (state: TripRuntimeState) => T): T {
  const store = useContext(TripRuntimeContext);
  if (store === null) {
    throw new Error("useTripRuntime must be used inside TripRuntimeProvider");
  }
  return useStore(store, selector);
}

export function useTripRuntimeStore(): TripRuntimeStore {
  const store = useContext(TripRuntimeContext);
  if (store === null) {
    throw new Error(
      "useTripRuntimeStore must be used inside TripRuntimeProvider"
    );
  }
  return store;
}
