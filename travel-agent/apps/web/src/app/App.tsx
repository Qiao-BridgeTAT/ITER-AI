import type { ReactNode } from "react";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import { TripBackendProvider } from "../backend/TripBackendContext";
import { CoCreationPage } from "../pages/CoCreationPage";
import { LandingPage } from "../pages/LandingPage";
import { NotFoundPage } from "../pages/NotFoundPage";
import { TripRuntimeProvider } from "../realtime/TripRuntimeContext";
import { TripShellProvider } from "../session/TripShellContext";
import { ViewerSessionProvider } from "../session/viewerSession";
import { PageBootScreen } from "./PageBootScreen";

export function AppProviders({ children }: { children: ReactNode }) {
  return (
    <ViewerSessionProvider>
      <TripShellProvider>
        <TripRuntimeProvider>
          <TripBackendProvider mode="real">{children}</TripBackendProvider>
        </TripRuntimeProvider>
      </TripShellProvider>
    </ViewerSessionProvider>
  );
}

export function App() {
  return (
    <BrowserRouter>
      <PageBootScreen />
      <div className="app-content-root">
        <AppProviders>
          <Routes>
            <Route path="/" element={<LandingPage />} />
            <Route path="/trips/:tripId" element={<CoCreationPage />} />
            <Route path="*" element={<NotFoundPage />} />
          </Routes>
        </AppProviders>
      </div>
    </BrowserRouter>
  );
}
