import type {
  DraftScheduledActivity,
  DraftScheduledPause,
  DraftScheduledTransport,
  HotelOfferObservation,
  PlannerPlacePreview,
  PlannerPublishedPlan,
  PlannerWeatherEvidence
} from "../generated/v4/contracts";
import type {
  PlanDay,
  PlanHotelOption,
  PlanReadyPresentation,
  PlanStop,
  PlanTransportKind,
  PlanTransportOption
} from "../conversation/PlanReadyAttachment";
import type { PlanBudgetEstimate } from "../conversation/PlanBudgetSummary";
import type {
  WeatherCondition,
  WeatherDayData
} from "../conversation/weatherTypes";

type SelectedHotelRecommendation = {
  hotel_offer_ref: { offer_id: string };
  area_reason: string;
  route_fit: string;
  selection_reason: string;
  main_tradeoff: string;
};

type PublishedPlanWithSelectedHotel = PlannerPublishedPlan & {
  selected_hotel?: SelectedHotelRecommendation | null;
};

type HotelOfferWithReferencePrice = HotelOfferObservation & {
  reference_price?: {
    currency: string;
    minimum_minor: number;
    maximum_minor: number;
  } | null;
};

const WEEKDAYS = ["周日", "周一", "周二", "周三", "周四", "周五", "周六"];
const ORDINALS = ["第一天", "第二天", "第三天", "第四天", "第五天"];

export function buildV4PublishedPlanPresentation(
  plan: PlannerPublishedPlan,
  cityName: string,
  statusLabel = "",
  previews: readonly PlannerPlacePreview[] = [],
  weatherPreview: readonly PlannerWeatherEvidence[] = []
): PlanReadyPresentation {
  const lodgingMode = plan.working_itinerary.lodging_baseline.mode;
  const selectedOffer = (plan.hotel_observation?.offers ?? []).find(
    (offer) =>
      offer.offer_ref.offer_id ===
      plan.working_itinerary.lodging_baseline.selected_offer_ref?.offer_id
  );
  const hotelLocation = plan.hotel_location_evidence?.[0];
  const hotelName =
    selectedOffer?.property_name ??
    hotelLocation?.display_name ??
    "已确认的住宿";
  const days = plan.materialized_schedule.days.map((day, index) => {
    const activities = (day.activities ?? []).map((activity, activityIndex) =>
      activityStop(
        plan,
        activity,
        (day.transport_legs ?? []).find(
          (leg) =>
            leg.origin_place_id === activity.place_id &&
            leg.departure_time >= activity.end_time &&
            !day.activities
              ?.slice(activityIndex + 1)
              .some(
                (next) =>
                  next.place_id === activity.place_id &&
                  next.start_time < leg.departure_time
              )
        ),
        previews.find((item) => item.place_id === activity.place_id)
      )
    );
    const pauses = (day.pauses ?? []).map((pause) => {
      const before = day.activities?.find(
        (activity) => activity.end_time === pause.start_time
      );
      const after = day.activities?.find(
        (activity) => activity.start_time === pause.end_time
      );
      const onsite =
        before &&
        after &&
        before.node_id === after.node_id &&
        plan.working_itinerary.days[index]?.ordered_items?.some(
          (item) =>
            item.onsite_lunch &&
            "canonical_entity_id" in item.object_ref &&
            item.object_ref.canonical_entity_id === before.place_id
        );
      return pauseStop(pause, onsite ? before.title : undefined);
    });
    const stops = [...activities, ...pauses].sort(
      (left, right) =>
        left.time.localeCompare(right.time) ||
        left.title.localeCompare(right.title)
    );
    if (
      ["selected_offer", "fixed"].includes(lodgingMode) &&
      day.start_place_id === day.end_place_id
    ) {
      const departure = day.transport_legs?.find(
        (leg) => leg.origin_place_id === day.start_place_id
      );
      const arrival = day.transport_legs?.find(
        (leg) => leg.destination_place_id === day.end_place_id
      );
      if (departure)
        stops.unshift({
          placeId: day.start_place_id,
          time: trimSeconds(departure.departure_time),
          category: "从酒店出发",
          title: hotelName,
          description: "",
          image:
            previews.find((item) => item.place_id === day.start_place_id)
              ?.image_url ?? undefined,
          imageAlt: `${hotelName}实景`,
          plannedStay: "",
          transportAfter: transportOptions(plan, departure)
        });
      if (arrival)
        stops.push({
          placeId: day.end_place_id,
          time: trimSeconds(arrival.arrival_time),
          category: "返回酒店",
          title: hotelName,
          description: "",
          image:
            previews.find((item) => item.place_id === day.end_place_id)
              ?.image_url ?? undefined,
          imageAlt: `${hotelName}实景`,
          plannedStay: "",
          rating: hotelLocation?.rating ?? undefined,
          ratingSource: "高德"
        });
    }
    return {
      id: `day-${index + 1}` as const,
      date: formatMonthDay(day.service_date),
      weekday: weekday(day.service_date),
      ordinal: ORDINALS[index] ?? `第${index + 1}天`,
      theme:
        plan.working_itinerary.days[index]?.day_theme ||
        activities
          .slice(0, 2)
          .map((stop) => stop.title)
          .join(" · ") ||
        `第${index + 1}天安排`,
      startTime: trimSeconds(day.start_time),
      endTime: trimSeconds(day.end_time),
      walking: formatDistance(day.walking_m),
      finalNote:
        ["selected_offer", "fixed"].includes(lodgingMode) &&
        day.start_place_id === day.end_place_id
          ? ""
          : "当天行程结束",
      stops
    } satisfies PlanDay;
  });
  const hotelOptions = buildHotelOptions(plan);
  const transportModes = Array.from(
    new Set(
      plan.materialized_schedule.days.flatMap((day) =>
        (day.transport_legs ?? []).map((leg) => transportLabel(leg.mode))
      )
    )
  );
  const attractionNames = new Map<string | undefined, string>();
  for (const stop of days.flatMap((day) => day.stops)) {
    if (stop.includeInOverview && !attractionNames.has(stop.placeId)) {
      attractionNames.set(
        stop.placeId,
        plan.place_evidence?.find(
          (place) => place.canonical_entity_id === stop.placeId
        )?.display_name ?? stop.title
      );
    }
  }
  const firstAttractions = [...attractionNames.values()];
  const bestHotel = hotelOptions?.find((hotel) => hotel.role === "best");
  const constraintNotices = Array.from(
    new Set(
      [
        ...(plan.planning_notes ?? []),
        ...(plan.validation_observation.issues ?? [])
          .filter(
            (issue) =>
              issue.severity !== "warning" ||
              (issue.violated_constraint_refs ?? []).some((reference) =>
                /^(hard|dietary|facility):/u.test(reference)
              )
          )
          .map((issue) => issue.message_summary)
      ].map((note) =>
        note
          .replace(/[（(](?:c|h)\d+[)）]/gu, "")
          .replace(/程序提示/gu, "当前安排")
      )
    )
  );

  return {
    title: `${cityName} · ${days.length}天${days.length > 1 ? `${days.length - 1}晚` : ""}`,
    description: "",
    dateLabel: `${formatMonthDay(plan.materialized_schedule.start_date)} - ${formatMonthDay(plan.materialized_schedule.end_date)}`,
    paceLabel: "",
    days,
    weatherDays: [
      ...new Map(
        [...(plan.weather_evidence ?? []), ...weatherPreview]
          .filter((weather) =>
            plan.materialized_schedule.days.some(
              (day) => day.service_date === weather.service_date
            )
          )
          .map((weather) => [weather.service_date, weather])
      ).values()
    ]
      .sort((a, b) => a.service_date.localeCompare(b.service_date))
      .map(weatherPresentation),
    overview: {
      route: {
        title: [...new Set(firstAttractions)].join(" · ") || "查看每日安排",
        description: ""
      },
      lodging: bestHotel
        ? { title: bestHotel.title, description: hotelLocation?.address ?? "" }
        : plan.working_itinerary.lodging_baseline.mode === "fixed"
          ? {
              title: hotelName,
              description: hotelLocation?.address ?? ""
            }
          : plan.working_itinerary.lodging_baseline.mode === "not_applicable"
            ? {
                title: "本次无需住宿",
                description: ""
              }
            : {
                title: "住宿待补充",
                description: ""
              },
      transport: {
        title:
          plan.travel_style_summary ??
          (transportModes.length
            ? `以${transportModes[0]}为主，游览与休息兼顾。`
            : "游览与休息兼顾。"),
        description: ""
      }
    },
    budgetEstimate: budgetPresentation(plan),
    hotelOptions,
    notices: constraintNotices,
    statusLabel,
    statusTone: "stable"
  };
}

function activityStop(
  plan: PlannerPublishedPlan,
  activity: DraftScheduledActivity,
  transport: DraftScheduledTransport | undefined,
  preview?: PlannerPlacePreview
): PlanStop {
  return {
    placeId: activity.place_id,
    time:
      activity.kind === "fixed_event" && activity.availability !== "available"
        ? "时间待确认"
        : trimSeconds(activity.start_time),
    timeIsFixed: activity.kind === "fixed_event",
    category:
      activity.kind === "restaurant"
        ? "餐饮"
        : activity.kind === "fixed_event"
          ? "固定安排"
          : "景点",
    title: activity.title,
    description: preview?.description ?? "",
    diningDetails:
      activity.kind === "restaurant"
        ? (preview?.dining_details ?? undefined)
        : undefined,
    image: preview?.image_url ?? undefined,
    imageAlt: `${activity.title}实景`,
    plannedStay: `计划停留 ${formatDuration(activity.duration_minutes)}`,
    includeInOverview: activity.kind === "attraction",
    openingHours: openingHoursForActivity(plan, activity),
    rating:
      preview?.dining_details?.rating ??
      plan.place_evidence?.find(
        (place) => place.canonical_entity_id === activity.place_id
      )?.rating ??
      undefined,
    ratingSource: preview?.dining_details?.source_name ?? "高德",
    transportAfter: transport ? transportOptions(plan, transport) : undefined
  };
}

function openingHoursForActivity(
  plan: PlannerPublishedPlan,
  activity: DraftScheduledActivity
): string | undefined {
  const evidence = (plan.hours_evidence ?? [])
    .filter((item) => item.canonical_entity_id === activity.place_id)
    .sort((a, b) => Date.parse(b.observed_at) - Date.parse(a.observed_at));
  const day = evidence
    .flatMap((item) => item.days)
    .find((item) => item.service_date === activity.service_date);
  if (!day || day.status === "unknown") return undefined;
  if (day.status === "closed") return "当日闭馆";
  if (day.status === "conflict") return "营业时间待确认";
  return (day.intervals ?? [])
    .map((interval) => {
      const hours = `${trimSeconds(interval.opens_at)}–${trimSeconds(interval.closes_at)}`;
      return interval.last_entry_at
        ? `${hours}（${trimSeconds(interval.last_entry_at)}停止入场）`
        : hours;
    })
    .join(" / ");
}

function pauseStop(pause: DraftScheduledPause, onsiteVenue?: string): PlanStop {
  return {
    time: trimSeconds(pause.start_time),
    category: onsiteVenue
      ? "餐饮"
      : pause.kind === "meal"
        ? "用餐留白"
        : "休息",
    title: onsiteVenue
      ? `${onsiteVenue} · 园内午餐`
      : pause.kind === "meal"
        ? "预留用餐时间"
        : "预留休息时间",
    description: onsiteVenue ? "具体餐厅待确认" : "",
    imageAlt: "",
    plannedStay: `计划停留 ${formatDuration(pause.duration_minutes)}`
  };
}

function transportPresentation(transport: DraftScheduledTransport) {
  const modes: Record<DraftScheduledTransport["mode"], PlanTransportKind> = {
    walking: "walk",
    cycling: "cycling",
    transit: "transit",
    driving: "car"
  };
  return {
    id: modes[transport.mode],
    label: transportLabel(transport.mode),
    duration:
      transport.availability === "missing"
        ? "耗时待补充"
        : `${transport.duration_minutes}分钟`,
    distance:
      transport.availability === "missing"
        ? ""
        : routeDistance(transport.distance_m)
  };
}

function routeDistance(meters: number): string {
  return meters < 1000
    ? `${meters} m`
    : `${Number((meters / 1000).toFixed(1))} km`;
}

function transportOptions(
  plan: PlannerPublishedPlan,
  leg: DraftScheduledTransport
): PlanTransportOption[] {
  const edges = [...(plan.route_evidence ?? [])].reverse();
  const selectedEdge = edges.find((edge) =>
    edge.fact_reference_ids?.some((id) =>
      leg.source_reference_ids?.includes(id)
    )
  );
  const fallback = transportPresentation(leg);
  const alternatives = [
    { mode: "taxi", id: "car", label: "打车" },
    { mode: "public_transit", id: "transit", label: "公共交通" },
    { mode: "walking", id: "walk", label: "步行" }
  ] as const;
  return alternatives.map(({ mode, id, label }) => {
    const edge = selectedEdge
      ? edges.find(
          (item) =>
            item.origin.kind === selectedEdge.origin.kind &&
            item.origin.reference_id === selectedEdge.origin.reference_id &&
            item.destination.kind === selectedEdge.destination.kind &&
            item.destination.reference_id ===
              selectedEdge.destination.reference_id &&
            item.transport_mode === mode
        )
      : undefined;
    const selected =
      (selectedEdge?.transport_mode === "driving"
        ? "taxi"
        : selectedEdge?.transport_mode) === mode ||
      (!selectedEdge && fallback.id === id);
    return {
      id,
      mode,
      label,
      legId: leg.leg_id,
      selected,
      unavailable:
        !edge || edge.status === "missing" || edge.duration_minutes == null,
      duration:
        edge?.duration_minutes != null
          ? `${edge.duration_minutes}分钟`
          : selected
            ? fallback.duration
            : "",
      distance:
        edge?.distance_meters != null
          ? routeDistance(edge.distance_meters)
          : "",
      price: edge?.fare
        ? `约${money(edge.fare.maximum_fen, "CNY")}`
        : undefined,
      transfers:
        edge?.transfer_count != null
          ? edge.transfer_count === 0
            ? "无需换乘"
            : `换乘${edge.transfer_count}次`
          : "换乘信息待确认"
    };
  });
}

function buildHotelOptions(
  plan: PlannerPublishedPlan
): PlanHotelOption[] | undefined {
  const observation = plan.hotel_observation;
  if (!observation) return undefined;
  const nights = Math.max(1, plan.materialized_schedule.days.length - 1);
  const findOffer = (offerId: string) =>
    (observation.offers ?? []).find(
      (offer) => offer.offer_ref.offer_id === offerId
    );
  const selected = (plan as PublishedPlanWithSelectedHotel).selected_hotel;
  if (selected) {
    const offer = findOffer(selected.hotel_offer_ref.offer_id);
    return [
      {
        role: "best",
        label: "已选酒店",
        title: hotelTitle(offer, nights),
        description: joinDescription(
          selected.area_reason,
          selected.route_fit,
          selected.selection_reason,
          `主要取舍：${selected.main_tradeoff}`,
          hotelPriceNotice(offer)
        )
      }
    ];
  }
  const recommendations = plan.hotel_recommendations;
  if (!recommendations) return undefined;
  const best = recommendations.recommended_hotel;
  const bestOffer = findOffer(best.hotel_offer_ref.offer_id);
  return [
    {
      role: "best",
      label: "已选酒店",
      title: hotelTitle(bestOffer, nights),
      description: joinDescription(
        best.area_reason,
        best.route_fit,
        best.quality_and_price_fit,
        `主要取舍：${best.main_tradeoff}`,
        hotelPriceNotice(bestOffer)
      )
    }
  ];
}

function hotelTitle(
  offer: HotelOfferObservation | undefined,
  nights: number
): string {
  if (!offer) return "当前酒店选项";
  if (offer.total_price) {
    return `${offer.property_name} · 参考房价 ${money(Math.round(offer.total_price.amount_minor / nights), offer.total_price.currency)} / 晚`;
  }
  const referencePrice = (offer as HotelOfferWithReferencePrice)
    .reference_price;
  if (referencePrice) {
    return `${offer.property_name} · 参考房价 ${moneyRange(referencePrice)} / 晚`;
  }
  return `${offer.property_name} · 房价待实时确认`;
}

function hotelPriceNotice(offer: HotelOfferObservation | undefined): string {
  if (!offer) return "酒店房价与房态待实时确认。";
  if (offer.total_price) return "金额为当前可核验的整段住宿总价。";
  if ((offer as HotelOfferWithReferencePrice).reference_price) {
    return "列表参考房价仅用于规划，不代表实时房态或整段住宿可订总价。";
  }
  return "酒店房价与房态待实时确认。";
}

function joinDescription(...parts: Array<string | undefined>): string {
  return parts.filter((part) => part?.trim()).join(" ");
}

function moneyRange(
  value: NonNullable<HotelOfferWithReferencePrice["reference_price"]>
): string {
  const minimum = money(value.minimum_minor, value.currency);
  if (value.minimum_minor === value.maximum_minor) return minimum;
  return `${minimum}–${money(value.maximum_minor, value.currency)}`;
}

function budgetPresentation(plan: PlannerPublishedPlan): PlanBudgetEstimate {
  const activities = new Map(
    plan.materialized_schedule.days.flatMap((day) =>
      (day.activities ?? []).map(
        (activity) => [activity.activity_id, activity] as const
      )
    )
  );
  const places = new Map(
    (plan.place_evidence ?? []).map((place) => [
      place.canonical_entity_id,
      place
    ])
  );
  const hotel = plan.hotel_observation?.offers?.find(
    (offer) =>
      offer.offer_ref.offer_id ===
      plan.working_itinerary.lodging_baseline.selected_offer_ref?.offer_id
  );
  return {
    detail_items: (plan.cost_draft.days ?? []).flatMap((day) =>
      (day.lines ?? []).map((line) => {
        const activity = activities.get(line.subject_id);
        const ticket = activity
          ? plan.ticket_evidence?.find(
              (item) =>
                item.canonical_entity_id === activity.place_id &&
                item.service_date === line.service_date
            )
          : undefined;
        const place = activity ? places.get(activity.place_id) : undefined;
        const leg = plan.materialized_schedule.days
          .flatMap((item) => item.transport_legs ?? [])
          .find((item) => item.leg_id === line.subject_id);
        const placeName = (id: string) =>
          places.get(id)?.display_name ?? hotel?.property_name ?? "住宿";
        return {
          id: line.price_fact_id,
          service_date: line.service_date,
          category: line.category,
          label:
            (activity &&
            line.category === "dining" &&
            activity.kind === "attraction"
              ? `${activity.title} · 园内午餐`
              : activity?.title) ??
            (line.category === "lodging"
              ? (hotel?.property_name ?? "住宿")
              : leg
                ? `${placeName(leg.origin_place_id)} → ${placeName(leg.destination_place_id)}`
                : "用餐"),
          amount_per_person: line.amount_per_person,
          missing_label:
            ticket?.admission_status === "paid"
              ? "非免费 · 价格待查"
              : undefined,
          reference_consumption:
            line.category === "attraction_tickets" && !line.amount_per_person
              ? place?.average_cost
              : undefined
        };
      })
    ),
    items: plan.cost_draft.categories.map((category) => ({
      category: category.category,
      availability:
        category.status === "available"
          ? "available"
          : category.status === "partial"
            ? "partial"
            : "missing",
      amount_per_person: category.amount_per_person,
      missing_reason:
        category.note ??
        (category.status === "not_applicable" ? "本次未计" : "价格资料暂缺")
    })),
    total_per_person: plan.cost_draft.known_total_per_person,
    lodging_share_divisor: plan.cost_draft.lodging_share_divisor,
    fetched_at_note: plan.cost_draft.pricing_note
  };
}

function weatherPresentation(weather: PlannerWeatherEvidence): WeatherDayData {
  const label =
    weather.condition_day ?? weather.condition_night ?? "天气资料暂缺";
  return {
    id: weather.service_date,
    dateLabel: formatMonthDay(weather.service_date),
    weekday: weekday(weather.service_date),
    condition: weatherCondition(label),
    conditionLabel:
      weather.forecast_kind === "outlook" ? `长期趋势 · ${label}` : label,
    temperatureC: weather.high_celsius,
    lowTemperatureC: weather.low_celsius,
    travelNote:
      weather.forecast_kind === "outlook"
        ? "远期趋势，临近出发再确认"
        : (weather.source_name ?? undefined)
  };
}

function weatherCondition(label: string): WeatherCondition {
  if (/雷/u.test(label)) return "thunderstorm";
  if (/暴雨|大雨/u.test(label)) return "heavy-rain";
  if (/雨/u.test(label)) return "light-rain";
  if (/雪/u.test(label)) return "snow";
  if (/雾|霾/u.test(label)) return "fog-haze";
  if (/风/u.test(label)) return "wind";
  if (/阴/u.test(label)) return "overcast";
  if (/多云/u.test(label)) return "partly-cloudy";
  if (/云/u.test(label)) return "cloudy";
  if (/晴/u.test(label)) return "clear";
  return "unknown";
}

function transportLabel(mode: DraftScheduledTransport["mode"]): string {
  return {
    walking: "步行",
    cycling: "骑行",
    transit: "公共交通",
    driving: "打车"
  }[mode];
}

function trimSeconds(value: string): string {
  return value.replace(/:00$/, "");
}

function formatDuration(minutes: number): string {
  if (minutes < 60) return `${minutes}分钟`;
  const hours = Math.floor(minutes / 60);
  const remainder = minutes % 60;
  return remainder === 0 ? `${hours}小时` : `${hours}小时${remainder}分钟`;
}

function formatDistance(distanceM: number): string {
  return distanceM < 1000
    ? `约${distanceM}米`
    : `约${(distanceM / 1000).toFixed(distanceM >= 10_000 ? 0 : 1)}公里`;
}

function formatMonthDay(value: string): string {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/u.exec(value);
  return match ? `${Number(match[2])}月${Number(match[3])}日` : value;
}

function weekday(value: string): string {
  const parsed = new Date(`${value}T00:00:00Z`);
  return Number.isNaN(parsed.getTime()) ? "" : WEEKDAYS[parsed.getUTCDay()];
}

function money(amountMinor: number, currency: string): string {
  const amount = new Intl.NumberFormat("zh-CN", {
    minimumFractionDigits: amountMinor % 100 === 0 ? 0 : 2,
    maximumFractionDigits: 2
  }).format(amountMinor / 100);
  return currency === "CNY" ? `¥${amount}` : `${currency} ${amount}`;
}
