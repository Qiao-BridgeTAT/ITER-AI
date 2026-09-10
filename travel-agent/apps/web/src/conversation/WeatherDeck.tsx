import { useState, type CSSProperties } from "react";

import { WeatherCard } from "./WeatherCard";
import type { WeatherDayData } from "./weatherTypes";

import "./weather.css";

type WeatherDeckProps = {
  days: readonly WeatherDayData[];
  ariaLabel?: string;
  forceReducedMotion?: boolean;
  simulateIconFailureForIds?: readonly string[];
};

type WeatherCardPosition = CSSProperties & {
  "--weather-card-width": string;
  "--weather-card-x": string;
};

const COMPACT_CARD_WIDTH = 106;
const EXPANDED_CARD_WIDTH = 218;
const CARD_GAP = 10;
const DECK_WIDTH = 610;

function getCardPosition(
  index: number,
  count: number,
  expandedIndex: number,
): WeatherCardPosition {
  const hasExpandedCard = expandedIndex >= 0;
  const expandedDelta = hasExpandedCard
    ? EXPANDED_CARD_WIDTH - COMPACT_CARD_WIDTH
    : 0;
  const contentWidth =
    count * COMPACT_CARD_WIDTH +
    Math.max(0, count - 1) * CARD_GAP +
    expandedDelta;
  const start = (DECK_WIDTH - contentWidth) / 2;
  const precedingExpansion =
    hasExpandedCard && index > expandedIndex ? expandedDelta : 0;

  return {
    "--weather-card-width": `${
      index === expandedIndex ? EXPANDED_CARD_WIDTH : COMPACT_CARD_WIDTH
    }px`,
    "--weather-card-x": `${
      start + index * (COMPACT_CARD_WIDTH + CARD_GAP) + precedingExpansion
    }px`,
  };
}

export function WeatherDeck({
  days,
  ariaLabel = "旅行期间天气",
  forceReducedMotion = false,
  simulateIconFailureForIds = [],
}: WeatherDeckProps) {
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const expandedIndex = days.findIndex((weather) => weather.id === expandedId);

  if (days.length === 0) return null;

  return (
    <div
      className={`weather-deck${forceReducedMotion ? " is-reduced-motion" : ""}`}
      role="group"
      aria-label={ariaLabel}
      onMouseLeave={() => setExpandedId(null)}
    >
      {days.map((weather, index) => (
        <WeatherCard
          key={weather.id}
          weather={weather}
          expanded={expandedId === weather.id}
          forceReducedMotion={forceReducedMotion}
          simulateIconFailure={simulateIconFailureForIds.includes(weather.id)}
          style={getCardPosition(index, days.length, expandedIndex)}
          onExpandedChange={(expanded) =>
            setExpandedId(expanded ? weather.id : null)
          }
        />
      ))}
    </div>
  );
}
