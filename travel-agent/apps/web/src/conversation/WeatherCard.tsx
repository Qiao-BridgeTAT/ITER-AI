import type { CSSProperties } from "react";

import { WeatherIcon } from "./WeatherIcon";
import { weatherIconMap } from "./weatherIconMap";
import type { WeatherDayData } from "./weatherTypes";

type WeatherCardProps = {
  weather: WeatherDayData;
  expanded: boolean;
  forceReducedMotion?: boolean;
  simulateIconFailure?: boolean;
  style?: CSSProperties;
  onExpandedChange: (expanded: boolean) => void;
};

function temperatureLabel(value: number | null | undefined): string {
  return value == null ? "暂无" : `${value}°`;
}

export function WeatherCard({
  weather,
  expanded,
  forceReducedMotion = false,
  simulateIconFailure = false,
  style,
  onExpandedChange,
}: WeatherCardProps) {
  const conditionLabel =
    weather.conditionLabel ?? weatherIconMap[weather.condition].label;
  const detailCount = [
    weather.precipitationProbability,
    weather.humidityPercent,
    weather.wind,
    weather.travelNote,
  ].filter((value) => value != null && value !== "").length;

  return (
    <button
      className={`weather-card${expanded ? " is-expanded" : ""}${
        forceReducedMotion ? " is-reduced-motion" : ""
      }`}
      data-condition={weather.condition}
      style={style}
      type="button"
      aria-expanded={expanded}
      aria-label={`${weather.dateLabel}${weather.weekday ? ` ${weather.weekday}` : ""}，${conditionLabel}，${temperatureLabel(weather.temperatureC)}`}
      onMouseEnter={() => onExpandedChange(true)}
      onFocus={() => onExpandedChange(true)}
      onClick={() => onExpandedChange(true)}
    >
      <span className="weather-card-date">
        <strong>{weather.dateLabel}</strong>
        {weather.weekday ? <small>{weather.weekday}</small> : null}
      </span>

      <span className="weather-card-illustration" aria-hidden="true">
        <WeatherIcon
          condition={weather.condition}
          label={conditionLabel}
          forceStatic={forceReducedMotion}
          simulateAnimatedFailure={simulateIconFailure}
        />
      </span>

      <span className="weather-card-temperature">
        <strong>{temperatureLabel(weather.temperatureC)}</strong>
        {weather.lowTemperatureC != null ? (
          <small>低 {weather.lowTemperatureC}°</small>
        ) : null}
      </span>

      <span className="weather-card-details" aria-hidden={!expanded}>
        <strong>{conditionLabel}</strong>
        {weather.precipitationProbability != null ? (
          <small>降水 {weather.precipitationProbability}%</small>
        ) : null}
        {weather.humidityPercent != null ? (
          <small>湿度 {weather.humidityPercent}%</small>
        ) : null}
        {weather.wind ? <small>{weather.wind}</small> : null}
        {weather.travelNote ? <em>{weather.travelNote}</em> : null}
        {detailCount === 0 ? <small>暂无详细天气</small> : null}
      </span>
    </button>
  );
}
