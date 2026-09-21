import { useCallback, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { AccountSessionOverlay } from "../account/AccountSessionOverlay";
import { LoginOverlay } from "../account/LoginOverlay";
import { useTripBackend } from "../backend/TripBackendContext";
import { pathWithAgentProtocol } from "../backend/agentProtocol";
import { ColdStartModal } from "../cold-start/ColdStartModal";
import type { ColdStartSubmission } from "../generated/contracts";
import {
  CURRENT_PROTOCOL_VERSION,
  CURRENT_SCHEMA_VERSION,
} from "../generated/protocol";
import { useTripRuntime } from "../realtime/TripRuntimeContext";
import { useTripShell } from "../session/TripShellContext";
import {
  useViewerSession,
  useViewerSessionActions,
} from "../session/viewerSession";
import { AboutIterModal } from "./AboutIterModal";
import { MobileHomeNotice } from "./MobileHomeNotice";
import { isMobilePhone } from "./mobileDevice";
const capabilities = ["理解偏好", "权衡取舍", "编排行程", "随时调整"] as const;
export function LandingPage() {
  const { shell } = useTripShell();
  const navigate = useNavigate();
  const location = useLocation();
  const backend = useTripBackend();
  const viewer = useViewerSession();
  const viewerActions = useViewerSessionActions();
  const runtimeStateVersion = useTripRuntime(
    (state) => state.tripState?.state_version,
  );
  const tripState = useTripRuntime((state) => state.tripState);
  const v4TripState = useTripRuntime((state) => state.v4.tripState);
  const homeMainRef = useRef<HTMLElement>(null);
  const needsMobileNotice = isMobilePhone(window.navigator) || null;
  const [homeOverlay, setHomeOverlay] = useState<"about" | "account" | "mobile" | null>(
    () => (needsMobileNotice ? "mobile" : null),
  );
  const closeHomeOverlay = useCallback(() => setHomeOverlay(null), []);
  const newTripNeedsColdStart =
    location.state?.startNewTrip === true && shell.phase === "cold_start";
  const [coldStartOpen, setColdStartOpen] = useState(newTripNeedsColdStart);
  const [coldStartSaving, setColdStartSaving] = useState(false);
  const [coldStartError, setColdStartError] = useState<string | null>(null);
  useEffect(() => {
    setColdStartOpen(newTripNeedsColdStart);
  }, [location.key, newTripNeedsColdStart]);
  useEffect(() => {
    // Every router entry is a new visit; dismissal is never persisted.
    setHomeOverlay(needsMobileNotice ? "mobile" : null);
    const restoreHome = (event: PageTransitionEvent) => {
      if (!event.persisted) return;
      setHomeOverlay(needsMobileNotice ? "mobile" : null);
      setColdStartOpen(false);
    };
    window.addEventListener("pageshow", restoreHome);
    return () => window.removeEventListener("pageshow", restoreHome);
  }, [location.key, needsMobileNotice]);
  useEffect(() => {
    const root = document.documentElement;
    let safetyTimer = 0;
    let active = true;
    const beginEntrance = () => {
      root.classList.add("anim");
      const animationElements = document.querySelectorAll(
        ".iter-home [data-entrance]",
      );
      const animations = Array.from(animationElements).flatMap((element) =>
        typeof element.getAnimations === "function"
          ? element.getAnimations()
          : [],
      );
      void Promise.allSettled(
        animations.map((animation) => animation.finished),
      ).then(() => {
        if (active) root.classList.remove("anim");
      });
      safetyTimer = window.setTimeout(
        () => root.classList.remove("anim"),
        6000,
      );
    };
    if (root.classList.contains("app-is-booting")) {
      window.addEventListener("iter:boot-complete", beginEntrance, {
        once: true,
      });
    } else {
      beginEntrance();
    }
    return () => {
      active = false;
      window.clearTimeout(safetyTimer);
      window.removeEventListener("iter:boot-complete", beginEntrance);
      root.classList.remove("anim");
    };
  }, []);
  const tripHref = pathWithAgentProtocol(
    `/trips/${shell.trip_id}`,
    backend.requestedProtocol,
  );
  const homeHref = pathWithAgentProtocol("/", backend.requestedProtocol);
  const completionKey = `travel-agent.cold-start-completed:${shell.trip_id}`;
  const coldStartAlreadyCompleted =
    shell.phase !== "cold_start" ||
    window.sessionStorage.getItem(completionKey) === "1";
  const startPlanning = () => {
    if (coldStartAlreadyCompleted) {
      navigate(tripHref);
      return;
    }
    setColdStartError(null);
    setColdStartOpen(true);
  };
  const completeColdStart = useCallback(
    (submission: ColdStartSubmission) => {
      const finish = () => {
        window.sessionStorage.setItem(completionKey, "1");
        setColdStartOpen(false);
        navigate(tripHref);
      };
      setColdStartError(null);
      if (backend.requestedProtocol === "v4") {
        setColdStartSaving(true);
        void viewerActions.updatePersonalDefaults(submission).then((saved) => {
          setColdStartSaving(false);
          if (!saved) {
            setColdStartError("长期偏好暂时没有保存，请检查连接后重试。");
            return;
          }
          finish();
        });
        return;
      }
      const requestId = crypto.randomUUID();
      const sent = backend.sendCommand({
        type: "cold_start_submit",
        protocol_version: CURRENT_PROTOCOL_VERSION,
        schema_version: CURRENT_SCHEMA_VERSION,
        request_id: requestId,
        idempotency_key: `cold-start:${shell.trip_id}:${requestId}`,
        expected_state_version: runtimeStateVersion ?? shell.state_version,
        payload: submission,
      });
      if (!sent) {
        setColdStartError("连接暂时不可用，长期偏好尚未保存，请稍后重试。");
        return;
      }
      finish();
    },
    [
      backend,
      completionKey,
      navigate,
      runtimeStateVersion,
      shell.state_version,
      shell.trip_id,
      tripHref,
      viewerActions,
    ],
  );
  return (
    <div className="iter-home">
      <a className="skip-link" href="#main-content">
        跳到主要内容
      </a>
      <div className="iter-stage">
        {import.meta.env.VITE_HOME_BACKGROUND_URL ? (
          <video
            className="iter-stage-video"
            autoPlay={
              !(
                window.matchMedia?.("(prefers-reduced-motion: reduce)")
                  .matches ?? false
              )
            }
            muted
            loop
            playsInline
            preload="auto"
            aria-hidden="true"
            src={import.meta.env.VITE_HOME_BACKGROUND_URL}
          />
        ) : null}

        <header className="iter-home-header">
          <Link
            className="iter-home-brand"
            to={homeHref}
            aria-label="ITER AI 首页"
            data-entrance
          >
            <img
              className="iter-mark"
              src="/brand/iter-mark-white-64.png"
              alt=""
            />
            <strong>ITER AI</strong>
          </Link>
          <nav className="iter-home-nav" aria-label="首页导航">
            <button
              type="button"
              data-entrance
              aria-haspopup="dialog"
              onClick={() => setHomeOverlay("about")}
            >
              关于 ITER AI
            </button>
            <button
              type="button"
              data-entrance
              aria-haspopup="dialog"
              onClick={() => setHomeOverlay("account")}
            >
              {viewer.kind === "user" ? "我的账号" : "登录"}
            </button>
          </nav>
        </header>
        <div className="iter-frame">
          <main
            ref={homeMainRef}
            className="iter-home-hero"
            id="main-content"
            tabIndex={-1}
          >
            <h1 id="iter-landing-title">
              <span className="iter-line">
                <span className="iter-line-inner" data-entrance>
                  会理解，也会取舍。
                </span>
              </span>
              <span className="iter-line">
                <span className="iter-line-inner" data-entrance>
                  把想法，变成一段旅程。
                </span>
              </span>
            </h1>
            <p className="iter-home-sub" data-entrance>
              从一个念头开始。偏好、地点和路线，
              <br />
              会在对话里逐步变清楚。
            </p>
            <button
              className="iter-home-button iter-home-button-hero"
              type="button"
              data-entrance
              onClick={startPlanning}
            >
              <span>开始规划</span>
            </button>
          </main>

          <div className="iter-capabilities" aria-label="规划能力">
            {capabilities.map((capability, index) => (
              <div
                className={`iter-capability iter-capability-${index + 1}`}
                key={capability}
                data-entrance
              >
                <span>{capability}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
      {homeOverlay === "mobile" ? (
        <MobileHomeNotice
          onContinue={closeHomeOverlay}
          returnFocusRef={homeMainRef}
        />
      ) : null}
      {homeOverlay === "about" ? (
        <AboutIterModal onClose={closeHomeOverlay} />
      ) : null}
      {homeOverlay === "account"
        ? createPortal(
            viewer.kind === "anonymous" ? (
              <LoginOverlay
                onClose={closeHomeOverlay}
                onSendCode={viewerActions.sendVerificationCode}
                onSubmit={async (phone, code) => {
                  const completed = await viewerActions.signInAndAttach({
                    phone,
                    code,
                    shell,
                    state:
                      tripState?.trip_id === shell.trip_id
                        ? tripState
                        : undefined,
                    title: "新的旅行",
                    hasV4Work:
                      backend.requestedProtocol === "v4" &&
                      v4TripState?.semantic_state.trip_id === shell.trip_id,
                  });
                  if (completed) closeHomeOverlay();
                  return completed;
                }}
              />
            ) : (
              <AccountSessionOverlay
                userId={viewer.account.user_id}
                maskedPhone={viewer.account.masked_phone}
                nickname={viewer.account.nickname}
                onClose={closeHomeOverlay}
                onUpdateNickname={viewerActions.updateNickname}
                onLogout={viewerActions.signOut}
              />
            ),
            document.body,
          )
        : null}
      {coldStartOpen && !homeOverlay ? (
        <ColdStartModal
          onClose={() => {
            if (!coldStartSaving) setColdStartOpen(false);
          }}
          onComplete={completeColdStart}
          submitting={coldStartSaving}
          submissionError={coldStartError}
        />
      ) : null}
    </div>
  );
}
