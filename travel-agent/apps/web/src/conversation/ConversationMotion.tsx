import {
  createContext,
  useContext,
  useLayoutEffect,
  useRef,
  useState,
  type HTMLAttributes,
  type ReactNode
} from "react";
import { useReducedMotion } from "motion/react";
import { AgentMarkdown } from "./AgentMarkdown";
import "./conversation-motion.css";

type Entry = { animate: boolean; entered: boolean; visible: string };
type Registry = { ready: boolean; entries: Map<string, Entry> };
const MotionContext = createContext<Registry | null>(null);

/** A trip-scoped presentation cache; never changes saved messages or Agent state. */
export function ConversationMotionProvider({
  enabled = true,
  children
}: {
  enabled?: boolean;
  children: ReactNode;
}) {
  const registry = useRef<Registry>({ ready: false, entries: new Map() });
  useLayoutEffect(() => {
    registry.current.ready = enabled;
  }, [enabled]);
  return (
    <MotionContext.Provider value={registry.current}>
      {children}
    </MotionContext.Provider>
  );
}

function useEntry(key: string) {
  const registry = useContext(MotionContext);
  const fallback = useRef<Entry>({
    animate: false,
    entered: false,
    visible: ""
  });
  if (!registry) return fallback.current;
  let entry = registry.entries.get(key);
  if (!entry) {
    entry = {
      animate: registry.ready && document.visibilityState !== "hidden",
      entered: false,
      visible: ""
    };
    registry.entries.set(key, entry);
  }
  return entry;
}

export function ConversationEntrance({
  motionKey,
  as: Tag = "div",
  className = "",
  children,
  ...props
}: HTMLAttributes<HTMLElement> & {
  motionKey: string;
  as?: "div" | "article";
}) {
  const entry = useEntry(`entrance:${motionKey}`);
  const reduced = useReducedMotion();
  const [enter] = useState(() => entry.animate && !entry.entered);
  useLayoutEffect(() => {
    entry.entered = true;
  }, [entry]);
  return (
    <Tag
      {...props}
      className={`${className}${enter && !reduced ? " conversation-enter" : ""}`}
    >
      {children}
    </Tag>
  );
}

/** Smooth received chunks only; reuse the visible prefix across stream/commit mounts. */
export function ProgressiveAgentText({
  text,
  replyKey
}: {
  text: string;
  replyKey: string;
}) {
  const entry = useEntry(`text:${replyKey}`);
  const reduced = useReducedMotion();
  const lastTick = useRef(performance.now());
  const [visible, setVisible] = useState(() =>
    !entry.animate || reduced || !text.startsWith(entry.visible)
      ? text
      : entry.visible
  );
  useLayoutEffect(() => {
    let frame = 0;
    let cancelled = false;
    const show = (value: string) => {
      entry.visible = value;
      setVisible(value);
    };
    const flush = () => {
      cancelAnimationFrame(frame);
      show(text);
    };
    if (
      !entry.animate ||
      reduced ||
      !text.startsWith(entry.visible) ||
      document.hidden
    ) {
      flush();
      return;
    }
    // Intl segments preserve emoji/combining sequences; no UTF-16 half-characters.
    const segments = [
      ...new Intl.Segmenter(undefined, { granularity: "grapheme" }).segment(
        text
      )
    ].map((segment) => segment.index + segment.segment.length);
    let count = segments.filter((end) => end <= entry.visible.length).length;
    let last = lastTick.current;
    const rate = Math.max(70, (segments.length - count) / 1.6);
    const step = () => {
      if (cancelled) return;
      const now = performance.now();
      // Normal short replies read naturally; a large delivered block catches up in ~2s.
      const amount = Math.floor(((now - last) * rate) / 1000);
      if (amount > 0) {
        count = Math.min(segments.length, count + amount);
        show(text.slice(0, segments[count - 1] ?? 0));
        last = now;
        lastTick.current = now;
      }
      if (count < segments.length) frame = requestAnimationFrame(step);
    };
    frame = requestAnimationFrame(step);
    const onVisibility = () => {
      if (document.hidden) flush();
    };
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      cancelled = true;
      cancelAnimationFrame(frame);
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, [text, entry, reduced]);
  const displayed =
    reduced || !entry.animate || !text.startsWith(visible) ? text : visible;
  return (
    <div className="progressive-agent-text" aria-busy={displayed !== text}>
      <AgentMarkdown text={displayed} />
    </div>
  );
}
