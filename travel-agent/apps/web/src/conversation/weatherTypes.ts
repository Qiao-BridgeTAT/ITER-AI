export const WEATHER_CONDITIONS = [
  "clear",
  "partly-cloudy",
  "cloudy",
  "overcast",
  "light-rain",
  "heavy-rain",
  "thunderstorm",
  "snow",
  "fog-haze",
  "wind",
  "unknown",
] as const;

export type WeatherCondition = (typeof WEATHER_CONDITIONS)[number];

export type WeatherDayData = {
  id: string;
  dateLabel: string;
  weekday?: string;
  condition: WeatherCondition;
  conditionLabel?: string;
  temperatureC?: number | null;
  lowTemperatureC?: number | null;
  precipitationProbability?: number | null;
  humidityPercent?: number | null;
  wind?: string | null;
  travelNote?: string | null;
};
