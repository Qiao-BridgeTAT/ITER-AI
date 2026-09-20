import { gsap } from "gsap";
import { SplitText } from "gsap/SplitText";
import { useLayoutEffect, useRef } from "react";

gsap.registerPlugin(SplitText);

const SHINE_SECONDS = 1.2;
const SHINE_PAUSE_SECONDS = 0.6;

// Adapted from the supplied React Bits SplitText: loading copy animates on
// change, rather than waiting for a one-time scroll trigger.
export function LoadingSplitText({
  text,
  durationMs
}: {
  text: string;
  durationMs: number;
}) {
  const ref = useRef<HTMLSpanElement>(null);
  useLayoutEffect(() => {
    const element = ref.current;
    if (!element) return;
    const reduced = window.matchMedia?.("(prefers-reduced-motion: reduce)");
    let split: SplitText | undefined;
    let entrance: gsap.core.Tween | undefined;
    let exit: gsap.core.Tween | undefined;
    let shine: gsap.core.Tween | undefined;
    let shinyText: HTMLSpanElement | undefined;
    const exitSeconds = element.textContent && !reduced?.matches ? 0.14 : 0;
    const clear = () => {
      exit?.kill();
      entrance?.kill();
      shine?.kill();
      split?.revert();
      split = undefined;
      shinyText?.classList.remove("is-shining");
      shinyText?.style.removeProperty("background-position");
      gsap.set(element, { clearProps: "opacity,transform" });
    };
    const enter = () => {
      element.textContent = text;
      gsap.set(element, { clearProps: "opacity,transform" });
      if (reduced?.matches) return;
      shinyText = document.createElement("span");
      shinyText.className = "agent-loading-shiny-text";
      shinyText.textContent = text;
      element.replaceChildren(shinyText);
      split = new SplitText(shinyText, {
        type: "chars",
        aria: "none",
        smartWrap: true,
        charsClass: "agent-loading-char"
      });
      const chars = split.chars;
      const entranceTime = 0.24 + Math.max(0, chars.length - 1) * 0.012;
      entrance = gsap.fromTo(
        chars,
        { opacity: 0, y: 5 },
        {
          opacity: 1,
          y: 0,
          duration: 0.24,
          stagger: 0.012,
          ease: "power2.out",
          onComplete: () => {
            // Restore whole text before applying the supplied ShinyText gradient:
            // the light is one continuous band, not a separate glow per letter.
            split?.revert();
            split = undefined;
            if (!shinyText) return;
            const available =
              durationMs / 1000 - exitSeconds - entranceTime - 0.1;
            const passes = Math.floor(
              (available + SHINE_PAUSE_SECONDS) /
                (SHINE_SECONDS + SHINE_PAUSE_SECONDS)
            );
            if (passes < 1) return;
            shinyText.classList.add("is-shining");
            shine = gsap.fromTo(
              shinyText,
              { backgroundPosition: "150% 50%" },
              {
                backgroundPosition: "-50% 50%",
                duration: SHINE_SECONDS,
                repeat: passes - 1,
                repeatDelay: SHINE_PAUSE_SECONDS,
                ease: "none"
              }
            );
          }
        }
      );
    };
    if (element.textContent && !reduced?.matches) {
      exit = gsap.to(element, {
        opacity: 0,
        y: -3,
        duration: 0.14,
        onComplete: enter
      });
    } else enter();
    const motionChanged = () => {
      clear();
      element.textContent = text;
    };
    reduced?.addEventListener("change", motionChanged);
    return () => {
      clear();
      reduced?.removeEventListener("change", motionChanged);
    };
  }, [text, durationMs]);
  return <span ref={ref} className="agent-loading-copy" />;
}
