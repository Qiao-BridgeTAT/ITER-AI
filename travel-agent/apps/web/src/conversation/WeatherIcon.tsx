import { useEffect, useState } from "react";

import { weatherIconMap } from "./weatherIconMap";
import type { WeatherCondition } from "./weatherTypes";

type WeatherIconProps = {
  condition: WeatherCondition;
  label?: string;
  forceStatic?: boolean;
  simulateAnimatedFailure?: boolean;
};

function reducedMotionIsPreferred(): boolean {
  return (
    typeof window !== "undefined" &&
    typeof window.matchMedia === "function" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches
  );
}

export function WeatherIcon({
  condition,
  label,
  forceStatic = false,
  simulateAnimatedFailure = false
}: WeatherIconProps) {
  const definition = weatherIconMap[condition];
  const [prefersReducedMotion, setPrefersReducedMotion] = useState(
    reducedMotionIsPreferred
  );
  const [animatedFailed, setAnimatedFailed] = useState(false);
  const [staticFailed, setStaticFailed] = useState(false);

  useEffect(() => {
    if (typeof window.matchMedia !== "function") return;

    const mediaQuery = window.matchMedia("(prefers-reduced-motion: reduce)");
    const handleChange = (event: MediaQueryListEvent) =>
      setPrefersReducedMotion(event.matches);
    mediaQuery.addEventListener?.("change", handleChange);
    return () => mediaQuery.removeEventListener?.("change", handleChange);
  }, []);

  const shouldUseStatic = forceStatic || prefersReducedMotion || animatedFailed;
  const source = shouldUseStatic
    ? definition.static
    : simulateAnimatedFailure
      ? "/__weather-icon-animation-unavailable__.svg"
      : definition.animated;

  if (staticFailed) {
    return (
      <span className="weather-icon-fallback" aria-label="天气图标暂无">
        —
      </span>
    );
  }

  return (
    <img
      className="weather-icon"
      src={source}
      alt={`${label ?? definition.label}天气图标`}
      data-icon-mode={shouldUseStatic ? "static" : "animated"}
      onError={() => {
        if (shouldUseStatic) {
          setStaticFailed(true);
        } else {
          setAnimatedFailed(true);
        }
      }}
    />
  );
}
