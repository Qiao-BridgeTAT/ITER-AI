import { useLayoutEffect, useState } from "react";
import { BootLoadingVisual } from "./BootLoadingVisual";
import { preloadInitialExperience } from "./preloadInitialExperience";
type BootPhase = "loading" | "exiting" | "hidden";
export interface AppBootScreenProps {
  load?: (signal: AbortSignal, attempt: number) => Promise<void>;
  minimumDuration?: number;
  /** Show recovery controls after this delay; never claim resources are ready. */
  maximumDuration?: number;
  exitDuration?: number;
  onComplete?: () => void;
}
function delay(duration: number, signal: AbortSignal) {
  if (signal.aborted || duration === 0) return Promise.resolve();
  return new Promise<void>((resolve) => {
    const finish = () => {
      window.clearTimeout(timer);
      signal.removeEventListener("abort", finish);
      resolve();
    };
    const timer = window.setTimeout(finish, duration);
    signal.addEventListener("abort", finish, { once: true });
  });
}
export function AppBootScreen({
  load = preloadInitialExperience,
  minimumDuration = 0,
  maximumDuration = 12000,
  exitDuration = 380,
  onComplete,
}: AppBootScreenProps) {
  const [phase, setPhase] = useState<BootPhase>("loading");
  const [attempt, setAttempt] = useState(0);
  const [skip, setSkip] = useState(false);
  const [notice, setNotice] = useState<"slow" | "error" | null>(null);
  const isPreview = null;
  const effectiveMinimumDuration = minimumDuration;
  const effectiveMaximumDuration = maximumDuration;
  useLayoutEffect(() => {
    const controller = new AbortController();
    const root = document.documentElement;
    const contentRoot =
      document.querySelector<HTMLElement>(".app-content-root");
    const prefersReducedMotion =
      window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;
    setPhase("loading");
    setNotice(null);
    root.classList.add("app-is-booting");
    root.setAttribute("aria-busy", "true");
    if (contentRoot) {
      contentRoot.setAttribute("inert", "");
      contentRoot.setAttribute("aria-hidden", "true");
    }
    let safetyTimer = 0;
    const releaseContent = () => {
      root.classList.remove("app-is-booting");
      root.removeAttribute("aria-busy");
      contentRoot?.removeAttribute("inert");
      contentRoot?.removeAttribute("aria-hidden");
    };
    const finishLoading = async () => {
      try {
        if (!skip) {
          await Promise.all([
            load(controller.signal, attempt),
            delay(
              prefersReducedMotion ? 0 : effectiveMinimumDuration,
              controller.signal,
            ),
          ]);
        }
      } catch {
        if (!controller.signal.aborted) {
          setNotice("error");
          controller.abort();
        }
        return;
      }
      if (controller.signal.aborted) return;
      window.clearTimeout(safetyTimer);
      setPhase("exiting");
      releaseContent();
      window.dispatchEvent(new CustomEvent("iter:boot-complete"));
      await delay(prefersReducedMotion ? 0 : exitDuration, controller.signal);
      if (!controller.signal.aborted) {
        setPhase("hidden");
        onComplete?.();
      }
    };
    safetyTimer = window.setTimeout(() => {
      setNotice("slow");
    }, effectiveMaximumDuration);
    void finishLoading().finally(() => window.clearTimeout(safetyTimer));
    return () => {
      controller.abort();
      window.clearTimeout(safetyTimer);
      releaseContent();
    };
  }, [
    attempt,
    skip,
    effectiveMaximumDuration,
    effectiveMinimumDuration,
    exitDuration,
    load,
    onComplete,
  ]);
  if (phase === "hidden") return null;
  return (
    <div
      className={`app-boot-screen app-boot-screen-${phase}`}
      role="status"
      aria-live="polite"
      aria-label="稍等一下，在为你准备中。"
    >
      <BootLoadingVisual />
      {notice && phase === "loading" ? (
        <div className="app-boot-feedback">
          <p role={notice === "error" ? "alert" : undefined}>
            {notice === "error"
              ? "页面资源暂时未能加载，可以重试或先进入页面。"
              : "加载比平时慢一些，正在继续准备。"}
          </p>
          <div className="app-boot-actions">
            <button
              type="button"
              onClick={() => setAttempt((value) => value + 1)}
            >
              重新加载
            </button>
            <button type="button" onClick={() => setSkip(true)}>
              先进入页面
            </button>
          </div>
        </div>
      ) : null}
    </div>
  );
}
