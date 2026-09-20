import { useEffect, useRef, useState } from "react";

import type { TripListItem } from "../generated/contracts";
import type {
  NewTripResult,
  TripResumeResult,
  ViewerSession
} from "../session/viewerSession";
import { tripRows } from "./tripHistoryView";

type TripPhase = TripListItem["phase"];

interface TripHistoryPopoverProps {
  viewer: ViewerSession;
  currentTripId: string;
  onSelectTrip: (
    tripId: string,
    signal?: AbortSignal
  ) => Promise<TripResumeResult>;
  onStartNewTrip: () => Promise<NewTripResult>;
  onRetryHistory: () => Promise<boolean>;
  onSignIn: () => void;
}

function phaseLabel(phase: TripPhase): string {
  if (phase === "confirmed") return "已确认";
  if (phase === "cold_start" || phase === "city_selection") return "待完善";
  return "规划中";
}

function resumeFailureMessage(
  reason: Exclude<TripResumeResult, { ok: true }>["reason"]
): string {
  if (reason === "not_found") return "这段行程已经不存在，请重新加载列表。";
  if (reason === "forbidden") return "你没有权限打开这段行程。";
  if (reason === "invalid") return "这段行程的数据暂时无法恢复。";
  if (reason === "timeout") return "打开行程超时，当前行程已保留。请重试。";
  if (reason === "session_expired") return "登录已失效，请重新登录后打开行程。";
  if (reason === "cancelled") return "";
  return "暂时无法打开这段行程，请检查网络后重试。";
}

export function TripHistoryPopover({
  viewer,
  currentTripId,
  onSelectTrip,
  onStartNewTrip,
  onRetryHistory,
  onSignIn
}: TripHistoryPopoverProps) {
  const sectionRef = useRef<HTMLElement>(null);
  const [busyTripId, setBusyTripId] = useState<string | null>(null);
  const [visibleCount, setVisibleCount] = useState(4);
  const [retrying, setRetrying] = useState(false);
  const [creating, setCreating] = useState(false);
  const creatingRef = useRef(false);
  const switchRequest = useRef<AbortController | null>(null);
  const [error, setError] = useState<string | null>(null);
  const rows =
    viewer.kind === "user"
      ? tripRows(viewer.history, currentTripId, visibleCount)
      : [];
  const currentListed = rows.some((trip) => trip.current);
  const historyStatus =
    viewer.kind === "user" ? (viewer.historyStatus ?? "ready") : "ready";

  useEffect(() => {
    const target = sectionRef.current?.querySelector<HTMLElement>(
      "button:not(:disabled), [tabindex='0']"
    );
    (target ?? sectionRef.current)?.focus();
    return () => switchRequest.current?.abort();
  }, []);

  const selectTrip = async (tripId: string) => {
    if (switchRequest.current !== null || creatingRef.current) return;
    const controller = new AbortController();
    switchRequest.current = controller;
    setBusyTripId(tripId);
    setError(null);
    try {
      const result = await onSelectTrip(tripId, controller.signal);
      if (!controller.signal.aborted && !result.ok) {
        setError(resumeFailureMessage(result.reason));
      }
    } catch {
      if (!controller.signal.aborted)
        setError(resumeFailureMessage("unavailable"));
    } finally {
      if (switchRequest.current === controller) {
        switchRequest.current = null;
        setBusyTripId(null);
      }
    }
  };

  const startNewTrip = async () => {
    if (creatingRef.current || busyTripId !== null) return;
    creatingRef.current = true;
    setCreating(true);
    setError(null);
    try {
      const result = await onStartNewTrip();
      if (!result.ok) {
        setError(
          result.reason === "session_expired"
            ? "登录或临时会话已失效，请重新登录后再新建旅行。"
            : result.reason === "invalid"
              ? "新行程返回的数据异常，当前行程已保留，请重试。"
              : "暂时无法新建旅行，当前行程已保留，请重试。"
        );
      }
    } catch {
      setError("暂时无法新建旅行，当前行程已保留，请重试。");
    } finally {
      creatingRef.current = false;
      setCreating(false);
    }
  };

  const retryHistory = async () => {
    if (retrying) return;
    setRetrying(true);
    setError(null);
    const loaded = await onRetryHistory();
    if (!loaded) setError("行程列表暂时没有加载成功，请稍后重试。");
    setRetrying(false);
  };

  return (
    <section
      ref={sectionRef}
      id="trip-history-popover"
      className="trip-history-popover"
      role="dialog"
      aria-modal="false"
      aria-labelledby="trip-history-popover-title"
      aria-busy={
        creating ||
        busyTripId !== null ||
        retrying ||
        historyStatus === "loading"
      }
      tabIndex={-1}
    >
      <header className="trip-history-popover-header">
        <h2 id="trip-history-popover-title">我的行程</h2>
      </header>

      {viewer.kind === "anonymous" ? (
        <div className="trip-history-popover-empty">
          <p>登录后可保存并继续之前的行程。</p>
          <button type="button" disabled={creating} onClick={onSignIn}>
            登录并保存旅程
          </button>
        </div>
      ) : (
        <>
          {historyStatus === "error" ? (
            <div className="trip-history-popover-alert" role="alert">
              <span>行程列表暂时没有加载成功。</span>
              <button
                type="button"
                disabled={retrying}
                onClick={() => void retryHistory()}
              >
                {retrying ? "正在重试…" : "重新加载"}
              </button>
            </div>
          ) : null}
          {rows.length > 0 ? (
            <ul className="trip-history-popover-list">
              {rows.map((trip) => (
                <li key={trip.tripId}>
                  <button
                    className={trip.current ? "is-current" : undefined}
                    type="button"
                    data-trip-id={trip.tripId}
                    aria-current={trip.current ? "page" : undefined}
                    disabled={creating || busyTripId !== null}
                    onClick={() => void selectTrip(trip.tripId)}
                  >
                    <span className="trip-history-popover-copy">
                      <strong title={trip.title}>{trip.title}</strong>
                      <span>
                        {phaseLabel(trip.phase)} · {trip.updatedLabel}
                      </span>
                    </span>
                    {trip.current || busyTripId === trip.tripId ? (
                      <span className="trip-history-popover-current">
                        {busyTripId === trip.tripId ? "正在恢复…" : "继续规划"}
                      </span>
                    ) : (
                      <svg
                        className="trip-history-popover-chevron"
                        viewBox="0 0 16 16"
                        fill="none"
                        stroke="currentColor"
                        aria-hidden="true"
                      >
                        <path
                          d="m6 3.5 4.5 4.5L6 12.5"
                          strokeWidth="1.5"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        />
                      </svg>
                    )}
                  </button>
                </li>
              ))}
            </ul>
          ) : historyStatus === "loading" ? (
            <div className="trip-history-popover-empty">
              <p>正在读取已保存的行程…</p>
            </div>
          ) : historyStatus === "error" ? null : (
            <div className="trip-history-popover-empty">
              <p>账号中还没有已保存的行程。</p>
              <button
                type="button"
                disabled={retrying}
                onClick={() => void retryHistory()}
              >
                {retrying ? "正在加载…" : "重新加载"}
              </button>
            </div>
          )}
          {viewer.history.length > rows.length ? (
            <button
              type="button"
              className="trip-history-popover-more"
              disabled={creating || busyTripId !== null}
              onClick={() => setVisibleCount((count) => count + 4)}
            >
              查看更多行程
            </button>
          ) : null}
          {!currentListed && rows.length > 0 ? (
            <p className="trip-history-popover-note">
              当前行程尚未出现在账号记录中。
            </p>
          ) : null}
        </>
      )}

      <button
        className="trip-history-popover-new"
        type="button"
        disabled={creating || busyTripId !== null}
        onClick={() => void startNewTrip()}
      >
        <svg
          width="18"
          height="18"
          viewBox="0 0 20 20"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.7"
          strokeLinecap="round"
          aria-hidden="true"
        >
          <path d="M10 4v12M4 10h12" />
        </svg>
        <span>{creating ? "正在创建…" : "新建旅行"}</span>
      </button>

      {busyTripId !== null ? (
        <div className="trip-history-popover-alert" role="status">
          <span>正在打开行程…</span>
          <button
            type="button"
            onClick={() => {
              switchRequest.current?.abort();
              switchRequest.current = null;
              setBusyTripId(null);
            }}
          >
            取消
          </button>
        </div>
      ) : null}

      {error ? (
        <p className="trip-history-popover-error" role="alert">
          {error}
        </p>
      ) : null}
    </section>
  );
}
