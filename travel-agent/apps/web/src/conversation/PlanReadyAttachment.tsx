import {
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  type CSSProperties
} from "react";
import { createPortal } from "react-dom";
import {
  Bicycle,
  Buildings,
  CalendarBlank,
  Car,
  CaretRight,
  Clock,
  MapTrifold,
  PersonSimpleWalk,
  Storefront,
  Star,
  Train,
  Tree,
  type Icon
} from "@phosphor-icons/react";

import DriftWall from "./DriftWall";
import { PlacePhoto } from "./PlacePhoto";
import { PlanBudgetSummary } from "./PlanBudgetSummary";
import { WeatherDeck } from "./WeatherDeck";
import type { PlanBudgetEstimate } from "./PlanBudgetSummary";
import type { WeatherDayData } from "./weatherTypes";
import { formatItineraryDayTimes } from "../planning/itineraryTime";
import type { DiningDisplayFacts } from "../generated/v4/contracts";
import { DiningFacts } from "./DiningFacts";

type PlanSectionId = "overview" | `day-${number}`;
export type PlanTransportKind = "car" | "walk" | "transit" | "cycling";

export type PlanTransportOption = {
  id: PlanTransportKind;
  label: string;
  duration: string;
  distance: string;
  price?: string;
  transfers?: string;
  selected?: boolean;
  unavailable?: boolean;
  legId?: string;
  mode?: "taxi" | "public_transit" | "walking";
};
export type PlanTransportSelectionHandler = (
  legId: string,
  mode: "taxi" | "public_transit" | "walking"
) => void;

export type PlanStop = {
  placeId?: string;
  time: string;
  timeIsFixed?: boolean;
  category: string;
  title: string;
  description: string;
  image?: string;
  imageAlt: string;
  plannedStay: string;
  openingHours?: string;
  setting?: "indoor" | "outdoor";
  rating?: number;
  ratingSource?: string;
  diningDetails?: DiningDisplayFacts;
  includeInOverview?: boolean;
  transportAfter?: PlanTransportOption[];
};

export type PlanDay = {
  id: Exclude<PlanSectionId, "overview">;
  date: string;
  weekday: string;
  ordinal: string;
  theme: string;
  startTime: string;
  endTime: string;
  walking: string;
  finalNote: string;
  stops: PlanStop[];
  startStop?: PlanStop;
  endStop?: PlanStop;
};

export type PlanOverviewSummary = {
  route: { title: string; description: string };
  lodging: { title: string; description: string };
  transport: { title: string; description: string };
};

export type PlanHotelOption = {
  role: "best" | "better_value" | "alternative_location_or_experience";
  label: string;
  title: string;
  description: string;
};

export type PlanReadyPresentation = {
  title: string;
  description: string;
  dateLabel: string;
  paceLabel: string;
  days: PlanDay[];
  weatherDays?: readonly WeatherDayData[];
  overview: PlanOverviewSummary;
  budgetEstimate?: PlanBudgetEstimate | null;
  hotelOptions?: PlanHotelOption[];
  notices?: readonly string[];
  statusLabel?: string;
  statusTone?: "stable" | "working" | "confirmed";
};

const IMAGE_PARAMS = "auto=format&fit=crop&w=720&q=86";

const PLAN_DAYS: PlanDay[] = [
  {
    id: "day-1",
    date: "9月18日",
    weekday: "周五",
    ordinal: "第一天",
    theme: "抵达海边，先把节奏慢下来",
    startTime: "14:00",
    endTime: "20:30",
    walking: "约5公里",
    finalNote: "返回酒店休息",
    stops: [
      {
        time: "14:00",
        category: "住宿",
        title: "海景酒店入住",
        description: "先放下行李、熟悉周边，把抵达日留给恢复体力。",
        image: `https://images.unsplash.com/photo-1566073771259-6a8506099945?${IMAGE_PARAMS}`,
        imageAlt: "面向海湾的安静酒店客房",
        plannedStay: "计划停留 1小时",
        setting: "indoor",
        transportAfter: [
          {
            id: "walk",
            label: "步行",
            duration: "12分钟",
            distance: "0.8公里"
          },
          {
            id: "car",
            label: "打车",
            duration: "5分钟",
            distance: "1.1公里",
            price: "约¥12"
          }
        ]
      },
      {
        time: "16:20",
        category: "散步",
        title: "湖岸长堤",
        includeInOverview: true,
        description: "沿着开阔水岸慢慢走，在日落前留一段自由停留。",
        image: `https://images.unsplash.com/photo-1470770841072-f978cf4d019e?${IMAGE_PARAMS}`,
        imageAlt: "群山与湖岸相接的开阔景色",
        plannedStay: "计划停留 2小时",
        setting: "outdoor",
        transportAfter: [
          {
            id: "walk",
            label: "步行",
            duration: "16分钟",
            distance: "1.2公里"
          },
          {
            id: "car",
            label: "打车",
            duration: "7分钟",
            distance: "2.4公里",
            price: "约¥14"
          }
        ]
      },
      {
        time: "18:50",
        category: "晚餐",
        title: "码头小馆晚餐",
        description: "用一顿清淡的本地菜结束第一天，不再跨区移动。",
        image: `https://images.unsplash.com/photo-1515003197210-e0cd71810b5f?${IMAGE_PARAMS}`,
        imageAlt: "靠近水岸的本地餐桌",
        plannedStay: "计划停留 1.5小时",
        setting: "indoor",
        transportAfter: [
          {
            id: "walk",
            label: "步行",
            duration: "11分钟",
            distance: "0.7公里"
          },
          {
            id: "car",
            label: "打车",
            duration: "5分钟",
            distance: "1.5公里",
            price: "约¥12"
          }
        ]
      }
    ]
  },
  {
    id: "day-2",
    date: "9月19日",
    weekday: "周六",
    ordinal: "第二天",
    theme: "走进山路，给下午留白",
    startTime: "08:30",
    endTime: "19:30",
    walking: "约8公里",
    finalNote: "返回酒店休息",
    stops: [
      {
        time: "08:30",
        category: "酒店早餐",
        title: "海景酒店早餐",
        description: "面朝大海享用在地食材制作的早餐，开启轻松的一天。",
        image: `https://images.unsplash.com/photo-1414235077428-338989a2e8c0?${IMAGE_PARAMS}`,
        imageAlt: "窗边海景早餐桌",
        plannedStay: "计划停留 45分钟",
        setting: "indoor",
        transportAfter: [
          {
            id: "car",
            label: "打车",
            duration: "08分钟",
            distance: "12.6公里",
            price: "约¥38"
          },
          {
            id: "transit",
            label: "公交",
            duration: "34分钟",
            distance: "13.1公里",
            price: "约¥4"
          }
        ]
      },
      {
        time: "10:00",
        category: "林间旧径",
        title: "山间步道漫步",
        includeInOverview: true,
        description: "沿着林间小径漫步，呼吸清新的空气，聆听风吹树叶的声音。",
        image: `https://images.unsplash.com/photo-1441974231531-c6227db76b6e?${IMAGE_PARAMS}`,
        imageAlt: "阳光穿过树冠照在林间小径上",
        plannedStay: "计划停留 2小时",
        setting: "outdoor",
        transportAfter: [
          {
            id: "walk",
            label: "步行",
            duration: "12分钟",
            distance: "0.9公里"
          },
          {
            id: "car",
            label: "打车",
            duration: "5分钟",
            distance: "1.3公里",
            price: "约¥13"
          }
        ]
      },
      {
        time: "12:30",
        category: "海岸晚餐",
        title: "海边小馆晚餐",
        description: "品尝新鲜海鲜与在地风味菜肴，享受海风与日落。",
        image: `https://images.unsplash.com/photo-1515003197210-e0cd71810b5f?${IMAGE_PARAMS}`,
        imageAlt: "夕阳下靠近海岸的晚餐",
        plannedStay: "计划停留 1.5小时",
        setting: "indoor",
        transportAfter: [
          {
            id: "car",
            label: "打车",
            duration: "45分钟",
            distance: "18.3公里"
          },
          {
            id: "transit",
            label: "公交",
            duration: "62分钟",
            distance: "19.1公里",
            price: "约¥6"
          }
        ]
      }
    ]
  },
  {
    id: "day-3",
    date: "9月20日",
    weekday: "周日",
    ordinal: "第三天",
    theme: "旧城慢游，把吃饭也放进路线",
    startTime: "09:20",
    endTime: "20:40",
    walking: "约7公里",
    finalNote: "乘地铁返回酒店",
    stops: [
      {
        time: "09:20",
        category: "街区",
        title: "老城晨市",
        includeInOverview: true,
        description: "从本地人的早市开始，慢慢走进旧城的日常。",
        image: `https://images.unsplash.com/photo-1569058242253-92a9c755a0ec?${IMAGE_PARAMS}`,
        imageAlt: "清晨的老城街巷与市场",
        plannedStay: "计划停留 1.5小时",
        setting: "outdoor",
        transportAfter: [
          { id: "walk", label: "步行", duration: "9分钟", distance: "0.6公里" },
          {
            id: "transit",
            label: "公交",
            duration: "14分钟",
            distance: "2.5公里",
            price: "约¥2"
          }
        ]
      },
      {
        time: "11:30",
        category: "展览",
        title: "街角博物馆",
        includeInOverview: true,
        description: "用一场小型展览理解城市，不把上午塞得太满。",
        image: `https://images.unsplash.com/photo-1518005020951-eccb494ad742?${IMAGE_PARAMS}`,
        imageAlt: "光线柔和的城市博物馆空间",
        plannedStay: "计划停留 2小时",
        setting: "indoor",
        transportAfter: [
          { id: "walk", label: "步行", duration: "8分钟", distance: "0.5公里" },
          {
            id: "car",
            label: "打车",
            duration: "4分钟",
            distance: "1.1公里",
            price: "约¥12"
          }
        ]
      },
      {
        time: "14:20",
        category: "漫游",
        title: "旧城街巷",
        includeInOverview: true,
        description: "只保留几家真正感兴趣的小店，中途安排咖啡休息。",
        image: `https://images.unsplash.com/photo-1477959858617-67f85cf4f1df?${IMAGE_PARAMS}`,
        imageAlt: "适合步行漫游的旧城街区",
        plannedStay: "计划停留 3小时",
        setting: "outdoor",
        transportAfter: [
          {
            id: "transit",
            label: "地铁",
            duration: "20分钟",
            distance: "6.4公里",
            price: "约¥3"
          },
          {
            id: "car",
            label: "打车",
            duration: "18分钟",
            distance: "7.8公里",
            price: "约¥26"
          }
        ]
      }
    ]
  },
  {
    id: "day-4",
    date: "9月21日",
    weekday: "周一",
    ordinal: "第四天",
    theme: "轻松收尾，把返程握在手里",
    startTime: "09:00",
    endTime: "14:30",
    walking: "约3公里",
    finalNote: "抵达车站，方案结束",
    stops: [
      {
        time: "09:00",
        category: "早餐",
        title: "海边早餐店",
        description: "在住处附近吃早餐，最后一天不再安排跨区移动。",
        image: `https://images.unsplash.com/photo-1551024506-0bccd828d307?${IMAGE_PARAMS}`,
        imageAlt: "明亮安静的早餐桌",
        plannedStay: "计划停留 1小时",
        setting: "indoor",
        transportAfter: [
          {
            id: "walk",
            label: "步行",
            duration: "10分钟",
            distance: "0.7公里"
          },
          {
            id: "car",
            label: "打车",
            duration: "5分钟",
            distance: "1.4公里",
            price: "约¥12"
          }
        ]
      },
      {
        time: "10:20",
        category: "书店",
        title: "海边书店",
        includeInOverview: true,
        description: "挑一本书或坐一会儿，给旅程留下最后一段空白。",
        image: `https://images.unsplash.com/photo-1500534314209-a25ddb2bd429?${IMAGE_PARAMS}`,
        imageAlt: "安静的海边阅读空间",
        plannedStay: "计划停留 1.5小时",
        setting: "indoor",
        transportAfter: [
          {
            id: "walk",
            label: "步行",
            duration: "14分钟",
            distance: "1.0公里"
          },
          {
            id: "car",
            label: "打车",
            duration: "6分钟",
            distance: "1.8公里",
            price: "约¥13"
          }
        ]
      },
      {
        time: "12:10",
        category: "住宿",
        title: "海景酒店取行李",
        description: "退房、取行李，再检查一次证件和返程信息。",
        image: `https://images.unsplash.com/photo-1551882547-ff40c63fe5fa?${IMAGE_PARAMS}`,
        imageAlt: "安静整洁的酒店前台",
        plannedStay: "计划停留 40分钟",
        setting: "indoor",
        transportAfter: [
          {
            id: "car",
            label: "打车",
            duration: "35分钟",
            distance: "16.8公里",
            price: "约¥42"
          },
          {
            id: "transit",
            label: "地铁",
            duration: "48分钟",
            distance: "17.2公里",
            price: "约¥5"
          }
        ]
      }
    ]
  }
];

const TRANSPORT_ICONS: Record<PlanTransportKind, Icon> = {
  car: Car,
  walk: PersonSimpleWalk,
  transit: Train,
  cycling: Bicycle
};

function transportDetails(option: PlanTransportOption): string {
  if (option.unavailable) return "路线数据暂未取得";
  return (
    option.id === "transit"
      ? [option.duration, option.transfers]
      : option.id === "walk"
        ? [option.duration, option.distance]
        : [option.duration, option.distance, option.price]
  )
    .filter(Boolean)
    .join(" · ");
}

function TransportSelector({
  options,
  onSelect,
  disabled = false,
  readOnly = false
}: {
  options: PlanTransportOption[];
  onSelect?: PlanTransportSelectionHandler;
  disabled?: boolean;
  readOnly?: boolean;
}) {
  const [selectedId, setSelectedId] = useState(options[0].id);
  const [open, setOpen] = useState(false);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);
  const [position, setPosition] = useState({ top: 0, left: 0 });
  const selected =
    options.find((option) => option.selected) ??
    options.find((option) => option.id === selectedId) ??
    options[0];
  const SelectedIcon = TRANSPORT_ICONS[selected.id];
  useEffect(() => {
    if (!open) return;
    const closeOutside = (event: PointerEvent) => {
      if (
        event.target instanceof Node &&
        !menuRef.current?.contains(event.target) &&
        !triggerRef.current?.contains(event.target)
      )
        setOpen(false);
    };
    const close = () => setOpen(false);
    const keydown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setOpen(false);
        triggerRef.current?.focus();
      }
      if (["ArrowDown", "ArrowUp", "Home", "End"].includes(event.key)) {
        event.preventDefault();
        const buttons = [
          ...(menuRef.current?.querySelectorAll<HTMLButtonElement>(
            "button:not(:disabled)"
          ) ?? [])
        ];
        const index = buttons.findIndex(
          (button) => button === document.activeElement
        );
        const next =
          event.key === "Home"
            ? 0
            : event.key === "End"
              ? buttons.length - 1
              : (index +
                  (event.key === "ArrowDown" ? 1 : -1) +
                  buttons.length) %
                buttons.length;
        buttons[next]?.focus();
      }
      if (event.key === "Tab") close();
    };
    document.addEventListener("pointerdown", closeOutside);
    document.addEventListener("keydown", keydown);
    window.addEventListener("resize", close);
    window.addEventListener("scroll", close, true);
    menuRef.current
      ?.querySelector<HTMLButtonElement>("button[aria-checked='true']")
      ?.focus();
    return () => {
      document.removeEventListener("pointerdown", closeOutside);
      document.removeEventListener("keydown", keydown);
      window.removeEventListener("resize", close);
      window.removeEventListener("scroll", close, true);
    };
  }, [open]);

  return (
    <div className="inline-plan-transport">
      <button
        className="inline-plan-transport-trigger"
        type="button"
        ref={triggerRef}
        disabled={disabled}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-label={`交通方式：${selected.label} · ${transportDetails(selected)}`}
        onClick={() => {
          const rect = triggerRef.current?.getBoundingClientRect();
          if (rect)
            setPosition({
              left: Math.max(8, Math.min(rect.left, window.innerWidth - 336)),
              top:
                rect.bottom + 8 + 164 > window.innerHeight
                  ? Math.max(8, rect.top - 172)
                  : rect.bottom + 8
            });
          setOpen((current) => !current);
        }}
      >
        <SelectedIcon size={15} weight="bold" />
        <span>{selected.label}</span>
        <span aria-hidden="true">·</span>
        <span>{transportDetails(selected)}</span>
        <CaretRight className={open ? "is-open" : ""} size={13} weight="bold" />
      </button>

      {open
        ? createPortal(
            <div
              className="inline-plan-transport-menu"
              ref={menuRef}
              style={{
                position: "fixed",
                top: position.top,
                left: position.left
              }}
              role="menu"
              aria-label={readOnly ? "上一版交通方案（只读）" : "切换交通方式"}
            >
              {options.map((option) => {
                const OptionIcon = TRANSPORT_ICONS[option.id];
                return (
                  <button
                    key={option.id}
                    type="button"
                    role="menuitemradio"
                    aria-checked={option.id === selected.id}
                    disabled={option.unavailable || disabled || readOnly}
                    onClick={() => {
                      if (option.legId && option.mode && onSelect)
                        onSelect(option.legId, option.mode);
                      else setSelectedId(option.id);
                      setOpen(false);
                      triggerRef.current?.focus();
                    }}
                  >
                    <OptionIcon size={15} weight="bold" />
                    <strong>{option.label}</strong>
                    <span>{transportDetails(option)}</span>
                  </button>
                );
              })}
            </div>,
            document.body
          )
        : null}
    </div>
  );
}

function PlaceCard({
  stop,
  highlighted,
  onHighlight
}: {
  stop: PlanStop;
  highlighted: boolean;
  onHighlight?: (placeId: string) => void;
}) {
  const SettingIcon = stop.setting === "indoor" ? Buildings : Tree;
  const placeId = stop.placeId;
  const className = `inline-plan-place${
    highlighted ? " is-highlighted" : ""
  }${placeId ? "" : " is-static"}`;
  const content = (
    <>
      <PlacePhoto
        key={stop.image ?? "missing"}
        src={stop.image}
        alt={stop.imageAlt || `${stop.title}实景`}
      />
      <span className="inline-plan-place-copy">
        <small>{stop.category}</small>
        <strong>{stop.title}</strong>
        <DiningFacts facts={stop.diningDetails} />
        {stop.description ? (
          <span className="place-introduction-line" title={stop.description}>
            {stop.description}
          </span>
        ) : null}
      </span>
      <span className="inline-plan-place-facts" aria-label="地点信息">
        {stop.rating != null && stop.diningDetails?.rating == null ? (
          <span
            className="inline-plan-rating"
            title={`${stop.ratingSource ?? "高德"}评分，满分5分`}
            aria-label={`${stop.ratingSource ?? "高德"}评分 ${stop.rating.toFixed(1)} 分`}
          >
            <Star size={13} weight="fill" />
            {stop.rating.toFixed(1)}
          </span>
        ) : null}
        {stop.plannedStay ? (
          <span>
            <Clock size={13} />
            {stop.plannedStay}
          </span>
        ) : null}
        {stop.openingHours ? (
          <span>
            <Storefront size={13} />
            {stop.openingHours}
          </span>
        ) : null}
        {stop.setting ? (
          <span>
            <SettingIcon size={13} />
            {stop.setting === "indoor" ? "室内" : "室外"}
          </span>
        ) : null}
      </span>
      {placeId ? (
        <CaretRight
          className="inline-plan-place-arrow"
          size={17}
          weight="bold"
        />
      ) : null}
    </>
  );

  if (!placeId) {
    return <div className={className}>{content}</div>;
  }

  return (
    <button
      className={className}
      type="button"
      aria-label={`查看 ${stop.title}`}
      aria-pressed={highlighted}
      onClick={() => onHighlight?.(placeId)}
    >
      {content}
    </button>
  );
}

function DayPanel({
  idPrefix,
  day,
  highlightedPlaceId,
  onHighlightPlace,
  onSelectTransport,
  transportBusy,
  readOnly
}: {
  idPrefix: string;
  day: PlanDay;
  highlightedPlaceId?: string | null;
  onHighlightPlace?: (placeId: string) => void;
  onSelectTransport?: PlanTransportSelectionHandler;
  transportBusy?: boolean;
  readOnly?: boolean;
}) {
  const times = formatItineraryDayTimes(day);
  return (
    <div
      className="inline-plan-day"
      role="tabpanel"
      id={`${idPrefix}-panel-${day.id}`}
      aria-labelledby={`${idPrefix}-tab-${day.id}`}
    >
      <div className="inline-plan-day-heading">
        <span>{day.ordinal}</span>
        <h3>{day.theme}</h3>
      </div>
      <div className="inline-plan-day-meta" aria-label="当天概览">
        <Clock size={14} />
        <span>约 {times.startTime} 开始</span>
        <span aria-hidden="true">·</span>
        <Clock size={14} />
        <span>约 {times.endTime} 结束</span>
        <span aria-hidden="true">·</span>
        <PersonSimpleWalk size={14} />
        <span>{day.walking}</span>
      </div>

      <div className="inline-plan-timeline">
        {day.stops.map((stop, index) => (
          <div
            className="inline-plan-stop-group"
            key={`${day.id}-${stop.time}-${index}`}
          >
            <div className="inline-plan-stop-row">
              <time
                title={stop.timeIsFixed ? "已确认的固定时间" : "预计到达时间"}
              >
                {times.stopTimes[index]}
              </time>
              <span className="inline-plan-node" aria-hidden="true" />
              <PlaceCard
                stop={stop}
                highlighted={Boolean(
                  stop.placeId && stop.placeId === highlightedPlaceId
                )}
                onHighlight={onHighlightPlace}
              />
            </div>
            {stop.transportAfter?.length ? (
              <div className="inline-plan-transit-row">
                <span />
                <span className="inline-plan-rail" aria-hidden="true" />
                <TransportSelector
                  options={stop.transportAfter}
                  onSelect={onSelectTransport}
                  disabled={transportBusy}
                  readOnly={readOnly}
                />
              </div>
            ) : index < day.stops.length - 1 ? (
              <div className="inline-plan-same-place-gap" aria-hidden="true" />
            ) : null}
            {index === day.stops.length - 1 && day.finalNote ? (
              <div className="inline-plan-end-row">
                <time title="预计时间，按15分钟取整">{times.endTime}</time>
                <span className="inline-plan-node" aria-hidden="true" />
                <strong>{day.finalNote}</strong>
              </div>
            ) : null}
          </div>
        ))}
      </div>
    </div>
  );
}

function OverviewPanel({
  idPrefix,
  onSelectDay,
  days,
  overview,
  weatherDays,
  budgetEstimate
}: {
  idPrefix: string;
  onSelectDay: (dayId: Exclude<PlanSectionId, "overview">) => void;
  days: PlanDay[];
  overview: PlanOverviewSummary;
  weatherDays: readonly WeatherDayData[];
  budgetEstimate?: PlanBudgetEstimate | null;
}) {
  const overviewAttractions = days.flatMap((day) =>
    day.stops
      .filter((stop) => stop.includeInOverview && stop.image)
      .map((stop) => ({
        id: day.id,
        image: stop.image!,
        title: stop.title
      }))
  );
  return (
    <div
      className="inline-plan-overview-panel"
      role="tabpanel"
      id={`${idPrefix}-panel-overview`}
      aria-labelledby={`${idPrefix}-tab-overview`}
    >
      {overviewAttractions.length > 0 ? (
        <section
          className="inline-plan-attraction-overview"
          aria-labelledby={`${idPrefix}-attraction-title`}
        >
          <header>
            <div>
              <h3 id={`${idPrefix}-attraction-title`}>沿途景点</h3>
              <p>悬停慢慢看，点击可直接进入景点所在日期。</p>
            </div>
            <span>{overviewAttractions.length}处</span>
          </header>
          <div className="inline-plan-attraction-wall">
            <DriftWall
              items={overviewAttractions}
              ariaLabel="全部景点概览"
              onItemSelect={(item) =>
                onSelectDay(item.id as Exclude<PlanSectionId, "overview">)
              }
            />
          </div>
        </section>
      ) : null}

      <section
        className="weather-overview-section"
        aria-labelledby={`${idPrefix}-weather-title`}
      >
        <header>
          <div>
            <h3 id={`${idPrefix}-weather-title`}>旅行天气</h3>
            <p>按旅行日期查询；远期天气仅供趋势参考。</p>
          </div>
          <span>
            {weatherDays.length > 0 ? `${weatherDays.length}天` : "暂未取得"}
          </span>
        </header>
        {weatherDays.length > 0 ? (
          <WeatherDeck days={weatherDays} />
        ) : (
          <p className="weather-empty-state">
            当前日期超出可查询范围，或天气服务暂未返回数据。临近出发时刷新查看。
          </p>
        )}
      </section>

      <div className="inline-plan-overview">
        <div>
          <small>主要景点</small>
          <strong>{overview.route.title}</strong>
        </div>
        <div>
          <small>住宿</small>
          <strong>{overview.lodging.title}</strong>
          {overview.lodging.description ? (
            <p>{overview.lodging.description}</p>
          ) : null}
        </div>
        <div>
          <small>出行方式与旅行节奏</small>
          <strong>{overview.transport.title}</strong>
        </div>
      </div>

      <PlanBudgetSummary estimate={budgetEstimate} />
    </div>
  );
}

export function PlanReadyAttachment({
  presentation,
  weatherDays = [],
  budgetEstimate,
  selectedDayIndex,
  highlightedPlaceId,
  onSelectDay,
  onHighlightPlace,
  onSelectTransport,
  transportBusy,
  readOnly,
  onOpenMap
}: {
  presentation?: PlanReadyPresentation;
  weatherDays?: readonly WeatherDayData[];
  budgetEstimate?: PlanBudgetEstimate | null;
  selectedDayIndex?: number;
  highlightedPlaceId?: string | null;
  onSelectDay?: (dayIndex: number) => void;
  onHighlightPlace?: (placeId: string) => void;
  onSelectTransport?: PlanTransportSelectionHandler;
  transportBusy?: boolean;
  readOnly?: boolean;
  onOpenMap?: () => void;
}) {
  const idPrefix = useId();
  const days = presentation?.days ?? PLAN_DAYS;
  const resolvedWeatherDays = presentation?.weatherDays ?? weatherDays;
  const overview = presentation?.overview ?? {
    route: {
      title: "海岸 · 山路 · 旧城 · 留白",
      description:
        "每天只安排一个主要区域，减少跨区往返，也为天气和临时变化留出余地。"
    },
    lodging: {
      title: "海边固定住处",
      description: "四天不换酒店，主要地点均以住处为起点和终点。"
    },
    transport: {
      title: "步行为主，远段打车",
      description: "每一段交通都可以在日程内打开并切换备选方式。"
    }
  };
  const tabs = [
    { id: "overview" as const, label: "总览" },
    ...days.map((day) => ({ id: day.id, label: `${day.date} ${day.weekday}` }))
  ];
  const [activeSection, setActiveSection] = useState<PlanSectionId>("overview");
  const activeDay = useMemo(
    () => days.find((day) => day.id === activeSection),
    [activeSection, days]
  );
  const selectedDayId =
    selectedDayIndex == null ? undefined : days[selectedDayIndex]?.id;

  useEffect(() => {
    if (
      activeSection !== "overview" &&
      !days.some((day) => day.id === activeSection)
    ) {
      setActiveSection("overview");
    }
  }, [activeSection, days]);

  useEffect(() => {
    if (selectedDayId) setActiveSection(selectedDayId);
  }, [selectedDayId]);

  useEffect(() => {
    if (!highlightedPlaceId) return;
    // Shared hotels occur on every day. Prefer the day already being viewed.
    if (
      days
        .find((candidate) => candidate.id === (selectedDayId ?? activeSection))
        ?.stops.some((stop) => stop.placeId === highlightedPlaceId)
    )
      return;
    const day = days.find((candidate) =>
      candidate.stops.some((stop) => stop.placeId === highlightedPlaceId)
    );
    if (day) setActiveSection(day.id);
  }, [activeSection, days, highlightedPlaceId, selectedDayId]);

  const tabStyle = {
    "--plan-day-count": Math.max(days.length, 1)
  } as CSSProperties;

  return (
    <section className="plan-ready-attachment" aria-label="旅行方案">
      <header className="inline-plan-header">
        <div className="inline-plan-intro">
          <div className="inline-plan-title-row">
            <h2>{presentation?.title ?? "沿海慢行 · 4天3晚"}</h2>
            {presentation?.statusLabel ? (
              <span
                className="inline-plan-status"
                data-tone={presentation.statusTone ?? "stable"}
              >
                {presentation.statusLabel}
              </span>
            ) : null}
          </div>
          {presentation == null || presentation.description ? (
            <p>
              {presentation?.description ??
                "放慢脚步，走进山海小镇，感受自然与在地生活。"}
            </p>
          ) : null}
          <div className="inline-plan-summary">
            <span>
              <CalendarBlank size={15} />
              {presentation?.dateLabel ?? "9月18日 - 9月21日"}
            </span>
            {presentation == null || presentation.paceLabel ? (
              <span>
                <Clock size={15} />
                {presentation?.paceLabel ?? "节奏偏松弛"}
              </span>
            ) : null}
          </div>
        </div>
        {onOpenMap ? (
          <button
            className="inline-plan-open-map"
            type="button"
            onClick={onOpenMap}
          >
            <MapTrifold size={18} />
            行程地图
          </button>
        ) : null}
      </header>

      <div
        className="inline-plan-tabs"
        style={tabStyle}
        role="tablist"
        aria-label="选择行程日期"
      >
        {tabs.map((tab) => {
          const selected = activeSection === tab.id;
          return (
            <button
              key={tab.id}
              className={selected ? "is-active" : ""}
              type="button"
              role="tab"
              id={`${idPrefix}-tab-${tab.id}`}
              aria-selected={selected}
              aria-controls={`${idPrefix}-panel-${tab.id}`}
              onClick={() => {
                setActiveSection(tab.id);
                if (tab.id !== "overview") {
                  onSelectDay?.(
                    days.findIndex((candidate) => candidate.id === tab.id)
                  );
                }
              }}
            >
              {tab.label}
            </button>
          );
        })}
      </div>

      <div className="inline-plan-body">
        {activeDay ? (
          <DayPanel
            idPrefix={idPrefix}
            day={activeDay}
            highlightedPlaceId={highlightedPlaceId}
            onHighlightPlace={onHighlightPlace}
            onSelectTransport={onSelectTransport}
            transportBusy={transportBusy}
            readOnly={readOnly}
          />
        ) : (
          <OverviewPanel
            idPrefix={idPrefix}
            onSelectDay={(dayId) => {
              setActiveSection(dayId);
              onSelectDay?.(
                days.findIndex((candidate) => candidate.id === dayId)
              );
            }}
            days={days}
            overview={overview}
            weatherDays={resolvedWeatherDays}
            budgetEstimate={
              presentation ? presentation.budgetEstimate : budgetEstimate
            }
          />
        )}
      </div>
    </section>
  );
}
