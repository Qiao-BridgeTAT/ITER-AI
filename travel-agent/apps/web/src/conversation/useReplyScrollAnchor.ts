import {
  useCallback,
  useLayoutEffect,
  useRef,
  useState,
  type RefObject
} from "react";

const NEAR_BOTTOM = 72;

/** Follow the conversation's tail only while the reader stays there. No artificial space. */
export function useReplyScrollAnchor(
  canvasRef: RefObject<HTMLDivElement>,
  tripId: string,
  enabled: boolean
) {
  const [showLatest, setShowLatest] = useState(false);
  const controls = useRef({ latest: () => {}, pause: () => {} });
  const scrollToLatest = useCallback(() => controls.current.latest(), []);
  const pauseFollowing = useCallback(() => controls.current.pause(), []);
  useLayoutEffect(() => {
    const canvas = canvasRef.current;
    if (!enabled || !canvas) return;
    const reduced = window.matchMedia?.("(prefers-reduced-motion: reduce)") ?? {
      matches: false
    };
    let following = true;
    let disposed = false;
    let frame = 0;
    let moveFrame = 0;
    let target = 0;
    let programmedTop = -1;
    const observed = new Set<Element>();
    const cards = new Set<Element>(
      canvas.querySelectorAll("[data-conversation-attachment]")
    );
    const bottom = () => Math.max(0, canvas.scrollHeight - canvas.clientHeight);
    const updateButton = () =>
      setShowLatest(bottom() - canvas.scrollTop > NEAR_BOTTOM);
    const stop = () => {
      following = false;
      cancelAnimationFrame(moveFrame);
      moveFrame = 0;
    };
    const move = () => {
      moveFrame = 0;
      if (disposed) return;
      const delta = target - canvas.scrollTop;
      const next =
        reduced.matches || Math.abs(delta) < 2
          ? target
          : canvas.scrollTop + delta * 0.3;
      programmedTop = next;
      canvas.scrollTop = next;
      updateButton();
      if (next !== target) moveFrame = requestAnimationFrame(move);
    };
    const aim = (top: number, immediate = false) => {
      target = Math.max(0, Math.min(bottom(), top));
      if (immediate || reduced.matches) {
        cancelAnimationFrame(moveFrame);
        moveFrame = 0;
        programmedTop = target;
        canvas.scrollTop = target;
        updateButton();
      } else if (!moveFrame) moveFrame = requestAnimationFrame(move);
    };
    const reconcile = () => {
      frame = 0;
      if (disposed) return;
      const allCards = [
        ...canvas.querySelectorAll<HTMLElement>(
          "[data-conversation-attachment]"
        )
      ];
      const fresh = allCards.find(
        (card) =>
          !cards.has(card) && card.offsetHeight > canvas.clientHeight * 0.5
      );
      allCards
        .filter((card) => card.offsetHeight > canvas.clientHeight * 0.5)
        .forEach((card) => cards.add(card));
      if (following && fresh) {
        const top =
          fresh.getBoundingClientRect().top -
          canvas.getBoundingClientRect().top +
          canvas.scrollTop;
        // Keep context above a large card visible instead of following its far-away end.
        aim(Math.max(canvas.scrollTop, top - canvas.clientHeight * 0.45));
        following = false;
      } else if (following) aim(bottom());
      updateButton();
      const children = new Set<Element>([canvas, ...canvas.children]);
      for (const child of observed) {
        if (!children.has(child)) {
          resize?.unobserve(child);
          observed.delete(child);
        }
      }
      for (const child of children) {
        if (!observed.has(child)) {
          resize?.observe(child);
          observed.add(child);
        }
      }
    };
    const schedule = () => {
      if (!frame && !disposed) frame = requestAnimationFrame(reconcile);
    };
    const onScroll = () => {
      if (Math.abs(canvas.scrollTop - programmedTop) > 2) {
        stop();
        following = bottom() - canvas.scrollTop <= NEAR_BOTTOM;
      }
      updateButton();
    };
    const onWheel = (event: WheelEvent) => {
      // Scrolling a card or the planning-process list must not change outer follow mode.
      let element = event.target instanceof Element ? event.target : null;
      while (element && element !== canvas) {
        if (
          element.scrollHeight > element.clientHeight &&
          /auto|scroll/.test(getComputedStyle(element).overflowY)
        )
          return;
        element = element.parentElement;
      }
      if (event.deltaY < 0) stop();
    };
    const onExpand = (event: Event) => {
      if (
        event.target instanceof Element &&
        event.target.closest("summary, .agent-progress-toggle")
      )
        stop();
    };
    const onKey = (event: KeyboardEvent) => {
      if (["ArrowUp", "PageUp", "Home"].includes(event.key)) stop();
    };
    controls.current = {
      latest: () => {
        following = true;
        aim(bottom());
      },
      pause: stop
    };
    const resize =
      typeof ResizeObserver === "undefined"
        ? null
        : new ResizeObserver(schedule);
    const mutation = new MutationObserver(schedule);
    mutation.observe(canvas, {
      childList: true,
      subtree: true,
      characterData: true
    });
    canvas.addEventListener("scroll", onScroll, { passive: true });
    canvas.addEventListener("wheel", onWheel, { passive: true });
    canvas.addEventListener("touchstart", stop, { passive: true });
    canvas.addEventListener("keydown", onKey);
    canvas.addEventListener("click", onExpand);
    aim(bottom(), true);
    reconcile();
    return () => {
      disposed = true;
      cancelAnimationFrame(frame);
      cancelAnimationFrame(moveFrame);
      mutation.disconnect();
      resize?.disconnect();
      controls.current = { latest: () => {}, pause: () => {} };
      canvas.removeEventListener("scroll", onScroll);
      canvas.removeEventListener("wheel", onWheel);
      canvas.removeEventListener("touchstart", stop);
      canvas.removeEventListener("keydown", onKey);
      canvas.removeEventListener("click", onExpand);
    };
  }, [canvasRef, tripId, enabled]);
  return { showLatest, scrollToLatest, pauseFollowing };
}
