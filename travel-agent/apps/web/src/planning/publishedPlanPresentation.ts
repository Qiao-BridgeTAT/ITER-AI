import type {
  PlanDay,
  PlanOverviewSummary,
  PlanReadyPresentation,
  PlanStop,
  PlanTransportKind,
} from "../conversation/PlanReadyAttachment";
import type {
  WeatherCondition,
  WeatherDayData,
} from "../conversation/weatherTypes";
import type {
  DailyWeatherCoverage,
  PublishedPlan,
  RecalledCandidate,
  SchedulePlaceFact,
  ScheduledActivity,
  ScheduledTransport,
  TripState,
} from "../generated/contracts";
import { buildPublishedPlanBudget } from "./publishedPlanBudget";

const WEEKDAYS = ["周日", "周一", "周二", "周三", "周四", "周五", "周六"];
const ORDINALS = ["第一天", "第二天", "第三天", "第四天", "第五天"];

export type FormalPlanStatus = {
  label: string;
  tone: "stable" | "working" | "confirmed";
};

export function buildPublishedPlanPresentation(
  state: TripState,
  cityName: string,
  status?: FormalPlanStatus,
): PlanReadyPresentation | null {
  const plan = state.published_plan;
  if (!plan) return null;

  const placeById = new Map(
    (plan.places ?? []).map((place) => [place.place_id, place]),
  );
  const candidateByPlaceId = new Map(
    (plan.selected_candidates ?? []).map((candidate) => [
      candidate.place.place_id,
      candidate,
    ]),
  );
  const days = plan.schedule.days.map((day, index) => {
    const stops = day.activities.map((activity) =>
      activityToStop(
        activity,
        day.transport_legs.find(
          (leg) => leg.origin_place_id === activity.place_id,
        ),
        placeById.get(activity.place_id),
        candidateByPlaceId.get(activity.place_id),
      ),
    );
    return {
      id: `day-${index + 1}` as const,
      date: formatMonthDay(day.service_date),
      weekday: weekday(day.service_date),
      ordinal: ORDINALS[index] ?? `第${index + 1}天`,
      theme: dayTheme(day.activities, index),
      startTime: day.start_time,
      endTime: day.end_time,
      walking: formatDistance(day.walking_m),
      finalNote: endNote(day.end_place_id, placeById),
      stops,
    } satisfies PlanDay;
  });
  const overview = buildOverview(plan);
  const nightCount = Math.max(days.length - 1, 0);

  return {
    title: `${cityName} · ${days.length}天${nightCount > 0 ? `${nightCount}晚` : ""}`,
    description: planDescription(plan),
    dateLabel: `${formatMonthDay(plan.schedule.start_date)} - ${formatMonthDay(plan.schedule.end_date)}`,
    paceLabel: paceLabel(state.resolved_preferences?.pace_level),
    days,
    weatherDays: (plan.weather ?? []).map(weatherToPresentation),
    overview,
    budgetEstimate: buildPublishedPlanBudget(plan.cost_estimate),
    statusLabel: status?.label,
    statusTone: status?.tone,
  };
}

function activityToStop(
  activity: ScheduledActivity,
  transport: ScheduledTransport | undefined,
  place: SchedulePlaceFact | undefined,
  candidate: RecalledCandidate | undefined,
): PlanStop {
  const openingWindow = place?.opening_windows?.find(
    (window) => window.service_date === activity.service_date,
  );
  const description =
    activity.timing_notice ??
    candidate?.reasons[0] ??
    activity.missing_reason ??
    "已按确认任务书纳入本日安排。";
  return {
    placeId: activity.place_id,
    time: activity.start_time,
    timeIsFixed: activity.kind === "fixed_event",
    category: activityCategory(activity),
    title: activity.title,
    description,
    image: candidate?.place.image_url ?? undefined,
    imageAlt: candidate?.place.image_url ? `${activity.title}的地点图片` : "",
    plannedStay: `计划停留 ${formatDuration(activity.duration_minutes)}`,
    openingHours: openingWindow
      ? `营业 ${openingWindow.start_time}–${openingWindow.end_time}`
      : undefined,
    includeInOverview: activity.kind === "attraction",
    transportAfter: transport ? [transportOption(transport)] : undefined,
  };
}

function transportOption(transport: ScheduledTransport) {
  const mode: Record<ScheduledTransport["mode"], PlanTransportKind> = {
    walking: "walk",
    cycling: "cycling",
    transit: "transit",
    driving: "car",
  };
  const label: Record<ScheduledTransport["mode"], string> = {
    walking: "步行",
    cycling: "骑行",
    transit: "公共交通",
    driving: "打车",
  };
  return {
    id: mode[transport.mode],
    label: label[transport.mode],
    duration:
      transport.availability === "missing"
        ? "耗时待补充"
        : `${transport.duration_minutes}分钟`,
    distance: formatDistance(transport.distance_m),
  };
}

function buildOverview(plan: PublishedPlan): PlanOverviewSummary {
  const dayTitles = plan.schedule.days
    .map(
      (day) =>
        day.activities.find((activity) => activity.kind === "attraction")
          ?.title,
    )
    .filter((title): title is string => Boolean(title));
  const modes = Array.from(
    new Set(
      plan.schedule.days.flatMap((day) =>
        day.transport_legs.map((leg) => leg.mode),
      ),
    ),
  );
  const selectedHotel = plan.hotel_selection?.candidates?.find(
    (hotel) =>
      hotel.hotel_place_id === plan.hotel_selection?.selected_hotel_place_id,
  );
  return {
    route: {
      title:
        dayTitles.length > 0
          ? dayTitles.slice(0, 3).join(" · ")
          : "逐日路线已生成",
      description: `${plan.schedule.days.length} 天行程已按真实地点、时间与交通资料编排。`,
    },
    lodging: selectedHotel
      ? {
          title: selectedHotel.name,
          description: selectedHotel.fit_reason,
        }
      : {
          title: "本方案暂无住宿信息",
          description: "住宿未进入当前已发布结果，不展示推测内容。",
        },
    transport: {
      title:
        modes.length > 0
          ? modes.map(transportModeLabel).join(" · ")
          : "交通资料待补充",
      description:
        plan.schedule.status === "available"
          ? "各段交通来自当前已发布方案。"
          : "部分交通资料暂缺，页面保留已验证的可用部分。",
    },
  };
}

function weatherToPresentation(weather: DailyWeatherCoverage): WeatherDayData {
  const label =
    weather.condition_day ?? weather.condition_night ?? "天气资料暂缺";
  return {
    id: weather.service_date,
    dateLabel: formatMonthDay(weather.service_date),
    weekday: weekday(weather.service_date),
    condition: weatherCondition(label),
    conditionLabel: label,
    temperatureC: weather.high_celsius ?? weather.low_celsius ?? null,
    lowTemperatureC: weather.low_celsius ?? null,
    travelNote:
      weather.availability === "missing"
        ? (weather.missing_reason ?? "天气资料暂未返回")
        : undefined,
  };
}

function weatherCondition(label: string): WeatherCondition {
  if (/雷/.test(label)) return "thunderstorm";
  if (/暴雨|大雨/.test(label)) return "heavy-rain";
  if (/雨/.test(label)) return "light-rain";
  if (/雪/.test(label)) return "snow";
  if (/雾|霾/.test(label)) return "fog-haze";
  if (/风/.test(label)) return "wind";
  if (/阴/.test(label)) return "overcast";
  if (/多云/.test(label)) return "partly-cloudy";
  if (/云/.test(label)) return "cloudy";
  if (/晴/.test(label)) return "clear";
  return "unknown";
}

function activityCategory(activity: ScheduledActivity) {
  if (activity.kind === "restaurant") return "餐饮";
  if (activity.kind === "fixed_event") return "固定安排";
  return "景点";
}

function dayTheme(activities: ScheduledActivity[], index: number) {
  const titles = activities.slice(0, 2).map((activity) => activity.title);
  return titles.length > 0 ? titles.join(" · ") : `第${index + 1}天安排`;
}

function endNote(placeId: string, places: Map<string, SchedulePlaceFact>) {
  const place = places.get(placeId);
  return place ? `结束于 ${place.name}` : "当天行程结束";
}

function planDescription(plan: PublishedPlan) {
  if (plan.validation.status === "review") {
    return "方案已通过可执行性校验，仍有少量提醒可在行前复查。";
  }
  if (plan.availability === "partial") {
    return "方案已生成；部分资料暂缺，未验证内容不会被伪装成确定信息。";
  }
  return "方案已根据确认任务书完成编排和可执行性校验。";
}

function paceLabel(level: number | undefined) {
  const labels = [
    "节奏紧凑",
    "节奏偏充实",
    "节奏适中",
    "节奏偏从容",
    "节奏从容",
  ];
  return labels[(level ?? 3) - 1] ?? "节奏适中";
}

function transportModeLabel(mode: ScheduledTransport["mode"]) {
  return {
    walking: "步行",
    cycling: "骑行",
    transit: "公共交通",
    driving: "打车",
  }[mode];
}

function formatDuration(minutes: number) {
  if (minutes < 60) return `${minutes}分钟`;
  const hours = Math.floor(minutes / 60);
  const remainder = minutes % 60;
  return remainder === 0 ? `${hours}小时` : `${hours}小时${remainder}分钟`;
}

function formatDistance(distanceM: number) {
  return distanceM < 1000
    ? `约${distanceM}米`
    : `约${(distanceM / 1000).toFixed(distanceM >= 10000 ? 0 : 1)}公里`;
}

function formatMonthDay(date: string) {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(date);
  return match ? `${Number(match[2])}月${Number(match[3])}日` : date;
}

function weekday(date: string) {
  const parsed = new Date(`${date}T00:00:00Z`);
  return Number.isNaN(parsed.getTime()) ? "" : WEEKDAYS[parsed.getUTCDay()];
}
