import { useCallback, useState } from "react";
import { useLocation } from "react-router-dom";

import { AppBootScreen, type AppBootScreenProps } from "./AppBootScreen";

/** Bootstrap once, then recheck the remounted background on every home entry. */
export function PageBootScreen(props: Omit<AppBootScreenProps, "onComplete">) {
  const location = useLocation();
  const [initialLoading, setInitialLoading] = useState(true);
  const completeInitialLoad = useCallback(() => setInitialLoading(false), []);

  if (!initialLoading && location.pathname !== "/") return null;

  return (
    <AppBootScreen
      key={location.key}
      {...props}
      onComplete={completeInitialLoad}
    />
  );
}
