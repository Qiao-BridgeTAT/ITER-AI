import type {
  TaskBookEntityIntent,
  TaskBookV4,
  TripSemanticState
} from "../generated/v4/contracts";

export type CompactTaskBookSummary = {
  destinationAndDates: string;
  attractionPreference: string;
  diningPreference: string;
  lodgingPreference: string;
};

const HOTEL_QUALITY_LABELS: Record<string, string> = {
  economy: "经济实用",
  comfort: "舒适中档",
  upscale: "高档品质",
  luxury: "豪华享受"
};

const CLASSIC_NICHE_LABELS: Record<number, string> = {
  1: "经典优先",
  2: "经典为主",
  3: "经典与兴趣平衡",
  4: "兴趣为主",
  5: "跟着兴趣走"
};

export function buildCompactTaskBookSummary(
  taskBook: TaskBookV4,
  semanticState: TripSemanticState | null = null
): CompactTaskBookSummary {
  const attractionPreference = firstPreference(
    taskBook.attraction_direction.preferences,
    coldStartAttractionPreference(semanticState)
  );
  const diningPreference = firstPreference(
    taskBook.dining_direction.preferences,
    coldStartDiningPreference(semanticState)
  );

  return {
    destinationAndDates: destinationAndDates(taskBook, semanticState),
    attractionPreference: joinPreferenceAndIntents(
      attractionPreference,
      "必去",
      taskBook.attraction_direction.must_visit ?? [],
      "想去",
      [
        ...(taskBook.attraction_direction.wanted ?? []),
        ...(taskBook.attraction_direction.if_convenient ?? [])
      ]
    ),
    diningPreference: joinPreferenceAndIntents(
      diningPreference,
      "必吃",
      taskBook.dining_direction.destination_restaurants ?? [],
      "想吃",
      taskBook.dining_direction.if_convenient_restaurants ?? []
    ),
    lodgingPreference: lodgingPreference(taskBook)
  };
}

function destinationAndDates(
  taskBook: TaskBookV4,
  semanticState: TripSemanticState | null
): string {
  const destination = taskBook.destination_and_dates.destination_name;
  const datesWereProvided =
    (semanticState?.trip_basics?.date_source_operation_refs?.length ?? 0) > 0;
  if (!datesWereProvided) return destination;
  const dateRange = formatDateRange(
    taskBook.destination_and_dates.start_date,
    taskBook.destination_and_dates.end_date
  );
  return dateRange ? `${destination} · ${dateRange}` : destination;
}

function formatDateRange(startDate: string, endDate: string): string | null {
  const start = parseDateParts(startDate);
  const end = parseDateParts(endDate);
  if (!start || !end) return null;
  if (start.year === end.year && start.month === end.month) {
    return `${start.year}年${start.month}月${start.day}日—${end.day}日`;
  }
  if (start.year === end.year) {
    return `${start.year}年${start.month}月${start.day}日—${end.month}月${end.day}日`;
  }
  return `${start.year}年${start.month}月${start.day}日—${end.year}年${end.month}月${end.day}日`;
}

function parseDateParts(value: string) {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/u.exec(value);
  if (!match) return null;
  return {
    year: Number(match[1]),
    month: Number(match[2]),
    day: Number(match[3])
  };
}

function firstPreference(
  preferences: Array<{ value: string }> | undefined,
  fallback: string
): string {
  const values = uniqueStrings((preferences ?? []).map((item) => item.value));
  return values.length > 0 ? values.join("、") : fallback;
}

function coldStartAttractionPreference(
  semanticState: TripSemanticState | null
): string {
  const level =
    semanticState?.cold_start_profile_snapshot?.preferences.classic_niche_level;
  return level ? CLASSIC_NICHE_LABELS[level] : "灵活安排";
}

function coldStartDiningPreference(
  semanticState: TripSemanticState | null
): string {
  const priorities =
    semanticState?.cold_start_profile_snapshot?.preferences.priority_goals ??
    [];
  return priorities.includes("satisfying_food") ? "吃得满意优先" : "灵活安排";
}

function joinPreferenceAndIntents(
  preference: string,
  strongLabel: string,
  strongIntents: TaskBookEntityIntent[],
  secondaryLabel: string,
  secondaryIntents: TaskBookEntityIntent[]
): string {
  const strong = uniqueIntents(strongIntents);
  const strongIds = new Set(strong.map(intentKey));
  const secondary = uniqueIntents(secondaryIntents).filter(
    (item) => !strongIds.has(intentKey(item))
  );
  const visibleStrong = strong.slice(0, 3);
  const visibleSecondary =
    strong.length > 3 ? [] : secondary.slice(0, Math.max(0, 3 - strong.length));
  const parts = preference === "灵活安排" ? [] : [preference];

  if (visibleStrong.length > 0) {
    parts.push(
      `${strongLabel} ${visibleStrong.map((item) => item.display_name).join("、")}${
        strong.length > 3 ? "等" : ""
      }`
    );
  }
  if (visibleSecondary.length > 0) {
    parts.push(
      `${secondaryLabel} ${visibleSecondary
        .map((item) => item.display_name)
        .join("、")}${secondary.length > visibleSecondary.length ? "等" : ""}`
    );
  }
  return parts.length > 0 ? parts.join(" · ") : preference;
}

function lodgingPreference(taskBook: TaskBookV4): string {
  if (taskBook.lodging_direction.not_applicable) return "无需住宿";
  const existingBooking =
    taskBook.lodging_direction.existing_booking?.user_description.trim();
  const areas = uniqueStrings(
    (taskBook.lodging_direction.area_preferences ?? []).map(
      (item) => item.value
    )
  );
  const quality = taskBook.lodging_direction.hotel_quality_tier;
  const parts = [
    ...(existingBooking ? [existingBooking] : []),
    ...(areas.length > 0 ? [areas.join("、")] : []),
    ...((taskBook.lodging_direction.hotel_quality_tiers ?? []).length > 0
      ? [
          (taskBook.lodging_direction.hotel_quality_tiers ?? [])
            .map((tier) => HOTEL_QUALITY_LABELS[tier] ?? tier)
            .join("、")
        ]
      : quality
        ? [HOTEL_QUALITY_LABELS[quality] ?? quality]
        : []),
    ...(taskBook.lodging_direction.property_type_preferences ?? []).map(
      (item) => item.value
    )
  ];
  return parts.length > 0 ? parts.join(" · ") : "灵活安排";
}

function uniqueIntents(items: TaskBookEntityIntent[]): TaskBookEntityIntent[] {
  const seen = new Set<string>();
  return items.filter((item) => {
    const key = intentKey(item);
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function intentKey(item: TaskBookEntityIntent): string {
  return item.canonical_entity_id || item.display_name;
}

function uniqueStrings(items: string[]): string[] {
  return [...new Set(items.map((item) => item.trim()).filter(Boolean))];
}
