import clearDayAnimated from "@meteocons/svg/fill/clear-day.svg";
import cloudyAnimated from "@meteocons/svg/fill/cloudy.svg";
import drizzleAnimated from "@meteocons/svg/fill/drizzle.svg";
import mistAnimated from "@meteocons/svg/fill/mist.svg";
import notAvailableAnimated from "@meteocons/svg/fill/not-available.svg";
import overcastAnimated from "@meteocons/svg/fill/overcast.svg";
import overcastRainAnimated from "@meteocons/svg/fill/overcast-rain.svg";
import overcastSnowAnimated from "@meteocons/svg/fill/overcast-snow.svg";
import partlyCloudyDayAnimated from "@meteocons/svg/fill/partly-cloudy-day.svg";
import thunderstormAnimated from "@meteocons/svg/fill/thunderstorms-day-rain.svg";
import windAnimated from "@meteocons/svg/fill/wind.svg";
import clearDayStatic from "@meteocons/svg-static/fill/clear-day.svg";
import cloudyStatic from "@meteocons/svg-static/fill/cloudy.svg";
import drizzleStatic from "@meteocons/svg-static/fill/drizzle.svg";
import mistStatic from "@meteocons/svg-static/fill/mist.svg";
import notAvailableStatic from "@meteocons/svg-static/fill/not-available.svg";
import overcastStatic from "@meteocons/svg-static/fill/overcast.svg";
import overcastRainStatic from "@meteocons/svg-static/fill/overcast-rain.svg";
import overcastSnowStatic from "@meteocons/svg-static/fill/overcast-snow.svg";
import partlyCloudyDayStatic from "@meteocons/svg-static/fill/partly-cloudy-day.svg";
import thunderstormStatic from "@meteocons/svg-static/fill/thunderstorms-day-rain.svg";
import windStatic from "@meteocons/svg-static/fill/wind.svg";

import type { WeatherCondition } from "./weatherTypes";

export type WeatherIconDefinition = {
  animated: string;
  static: string;
  label: string;
};

export const weatherIconMap: Record<WeatherCondition, WeatherIconDefinition> = {
  clear: {
    animated: clearDayAnimated,
    static: clearDayStatic,
    label: "晴"
  },
  "partly-cloudy": {
    animated: partlyCloudyDayAnimated,
    static: partlyCloudyDayStatic,
    label: "晴间多云"
  },
  cloudy: {
    animated: cloudyAnimated,
    static: cloudyStatic,
    label: "多云"
  },
  overcast: {
    animated: overcastAnimated,
    static: overcastStatic,
    label: "阴"
  },
  "light-rain": {
    animated: drizzleAnimated,
    static: drizzleStatic,
    label: "小雨"
  },
  "heavy-rain": {
    animated: overcastRainAnimated,
    static: overcastRainStatic,
    label: "中到大雨"
  },
  thunderstorm: {
    animated: thunderstormAnimated,
    static: thunderstormStatic,
    label: "雷雨"
  },
  snow: {
    animated: overcastSnowAnimated,
    static: overcastSnowStatic,
    label: "雪"
  },
  "fog-haze": {
    animated: mistAnimated,
    static: mistStatic,
    label: "雾或霾"
  },
  wind: {
    animated: windAnimated,
    static: windStatic,
    label: "大风"
  },
  unknown: {
    animated: notAvailableAnimated,
    static: notAvailableStatic,
    label: "暂无天气"
  }
};
