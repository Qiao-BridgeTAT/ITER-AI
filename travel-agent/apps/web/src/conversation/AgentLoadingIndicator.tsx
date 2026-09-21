import { CaretRight } from "@phosphor-icons/react";
import { useEffect, useId, useRef, useState } from "react";

import type { AgentProgressEntry } from "../generated/v4/contracts";
import { AgentProgressContent } from "./AgentProgressHistory";
import animationData from "./assets/iter-street-connections.json";
import { LoadingSplitText } from "./LoadingSplitText";

const PLANNING_COPY = "ITER AI正在为您努力规划中";
const STAGE_DURATION_MS = 5000;
const PLANNING_DURATION_MS = 3000;

interface LoadingAnimation {
  play(): void;
  pause(): void;
  goToAndStop(frame: number, isFrame: boolean): void;
  destroy(): void;
}

interface LottiePlayer {
  loadAnimation(options: {
    container: HTMLElement;
    renderer: "svg";
    loop: boolean;
    autoplay: boolean;
    animationData: unknown;
  }): LoadingAnimation;
}

let playerPromise: Promise<LottiePlayer> | undefined;

function loadPlayer(): Promise<LottiePlayer> {
  playerPromise ??= new Promise<LottiePlayer>((resolve, reject) => {
    const script = document.createElement("script");
    script.src = "/vendor/lottie.min.js";
    script.async = true;
    script.onload = () => {
      const player = (window as Window & { lottie?: LottiePlayer }).lottie;
      if (player) resolve(player);
      else reject(new Error("loading_animation_unavailable"));
    };
    script.onerror = () => {
      script.remove();
      playerPromise = undefined;
      reject(new Error("loading_animation_unavailable"));
    };
    document.head.append(script);
  });
  return playerPromise;
}

export function AgentLoadingIndicator({
  message,
  initialMessage,
  progressEntries
}: {
  message?: string;
  initialMessage?: string;
  progressEntries?: AgentProgressEntry[];
}) {
  const progressId = useId();
  const [progressOpen, setProgressOpen] = useState(false);
  const [visibleMessage, setVisibleMessage] = useState(
    message ?? initialMessage ?? "正在为你整理…"
  );
  useEffect(() => {
    if (message) setVisibleMessage(message);
  }, [message]);
  const [rotation, setRotation] = useState({
    stage: visibleMessage,
    alternate: false
  });
  const alternate = rotation.stage === visibleMessage && rotation.alternate;
  useEffect(() => {
    let timer: ReturnType<typeof setTimeout>;
    let showingPlanning = false;
    const schedule = () => {
      timer = setTimeout(
        () => {
          showingPlanning = !showingPlanning;
          setRotation({ stage: visibleMessage, alternate: showingPlanning });
          schedule();
        },
        showingPlanning ? PLANNING_DURATION_MS : STAGE_DURATION_MS
      );
    };
    const reset = () => {
      clearTimeout(timer);
      showingPlanning = false;
      setRotation({ stage: visibleMessage, alternate: false });
      if (!document.hidden) schedule();
    };
    reset();
    document.addEventListener("visibilitychange", reset);
    return () => {
      clearTimeout(timer);
      document.removeEventListener("visibilitychange", reset);
    };
  }, [visibleMessage]);
  const container = useRef<HTMLSpanElement>(null);
  useEffect(() => {
    let disposed = false;
    let animation: LoadingAnimation | undefined;
    const reduced = window.matchMedia?.("(prefers-reduced-motion: reduce)");
    const visibility = () => {
      if (reduced?.matches) animation?.goToAndStop(220, true);
      else if (document.hidden) animation?.pause();
      else animation?.play();
    };
    void loadPlayer()
      .then((player) => {
        if (disposed || !container.current) return;
        animation = player.loadAnimation({
          container: container.current,
          renderer: "svg",
          loop: true,
          autoplay: false,
          // Lottie mutates its input; mounts must not share animation objects.
          animationData: structuredClone(animationData)
        });
        visibility();
      })
      .catch(() => {
        // The readable status remains available if the decorative player fails.
      });
    reduced?.addEventListener("change", visibility);
    document.addEventListener("visibilitychange", visibility);
    return () => {
      disposed = true;
      animation?.destroy();
      reduced?.removeEventListener("change", visibility);
      document.removeEventListener("visibilitychange", visibility);
    };
  }, []);
  const loadingLabel = (
    <span className="agent-loading-label" aria-hidden="true">
      <span className="agent-loading-copy-measure">{visibleMessage}</span>
      <span className="agent-loading-copy-measure">{PLANNING_COPY}</span>
      <LoadingSplitText
        text={alternate ? PLANNING_COPY : visibleMessage}
        durationMs={alternate ? PLANNING_DURATION_MS : STAGE_DURATION_MS}
      />
    </span>
  );
  return (
    <article className="conversation-message-row conversation-message-row-agent">
      <div className="conversation-message-stack">
        <div className="agent-loading-row">
          <div
            className="message-bubble message-bubble-agent message-bubble-loading"
            role="status"
            aria-label="Agent 正在生成回复"
            aria-live="polite"
            aria-atomic="true"
          >
            <span
              ref={container}
              className="agent-loading-animation"
              aria-hidden="true"
            />
            {progressEntries !== undefined ? (
              <button
                type="button"
                className="agent-loading-text-trigger"
                aria-label={
                  progressOpen ? "点击收起规划思考过程" : "点击展开规划思考过程"
                }
                aria-expanded={progressOpen}
                aria-controls={progressId}
                onClick={() => setProgressOpen((open) => !open)}
              >
                {loadingLabel}
              </button>
            ) : (
              loadingLabel
            )}
            <span className="visually-hidden">{visibleMessage}</span>
          </div>
          {progressEntries !== undefined ? (
            <button
              type="button"
              className="agent-progress-toggle"
              aria-label={progressOpen ? "收起规划过程" : "展开规划过程"}
              aria-expanded={progressOpen}
              aria-controls={progressId}
              onClick={() => setProgressOpen((open) => !open)}
            >
              <CaretRight
                className="agent-progress-chevron"
                size={14}
                aria-hidden="true"
              />
            </button>
          ) : null}
        </div>
        {progressOpen && progressEntries !== undefined ? (
          <AgentProgressContent
            entries={progressEntries}
            id={progressId}
            active
          />
        ) : null}
      </div>
    </article>
  );
}
