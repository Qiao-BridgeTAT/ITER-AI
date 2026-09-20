import { useLayoutEffect, useState } from "react";
import type { AppliedRestoreTimings } from "../backend/useConversationHistory";

/** Local performance diagnostics: timings only, never message or account data. */
export function TripRestoreTiming({
  timings
}: {
  timings: AppliedRestoreTimings | null;
}) {
  const [measured, setMeasured] = useState<string | undefined>();
  useLayoutEffect(() => {
    if (!timings) {
      setMeasured(undefined);
      return;
    }
    const frame = requestAnimationFrame(() => {
      const renderMs = performance.now() - timings.appliedAt;
      setMeasured(
        JSON.stringify({
          request_ms: Math.round(timings.requestMs),
          parse_ms: Math.round(timings.parseMs),
          validate_ms: Math.round(timings.validateMs),
          apply_ms: Math.round(timings.applyMs),
          render_frame_ms: Math.round(renderMs),
          total_ms: Math.round(
            timings.requestMs +
              timings.parseMs +
              timings.validateMs +
              timings.applyMs +
              renderMs
          )
        })
      );
    });
    return () => cancelAnimationFrame(frame);
  }, [timings]);
  return <span hidden data-trip-restore-timing={measured} />;
}
