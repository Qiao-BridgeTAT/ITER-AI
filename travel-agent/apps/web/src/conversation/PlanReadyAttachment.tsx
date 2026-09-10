import {
  Bicycle,
  Buildings,
  CalendarBlank,
  Car,
  CaretRight,
  Clock,
  MapTrifold,
  PersonSimpleWalk,
  Star,
  Storefront,
  Train,
  Tree,
  type Icon,
} from "@phosphor-icons/react";
import {
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
} from "react";
import { createPortal } from "react-dom";
import type { DiningDisplayFacts } from "../generated/v4/contracts";
import { formatItineraryDayTimes } from "../planning/itineraryTime";
import { DiningFacts } from "./DiningFacts";
import DriftWall from "./DriftWall";
import { PlacePhoto } from "./PlacePhoto";
import type { PlanBudgetEstimate } from "./PlanBudgetSummary";
import { PlanBudgetSummary } from "./PlanBudgetSummary";
import { WeatherDeck } from "./WeatherDeck";
import type { WeatherDayData } from "./weatherTypes";
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
  mode: "taxi" | "public_transit" | "walking",
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
};
export type PlanOverviewSummary = {
  route: {
    title: string;
    description: string;
  };
  lodging: {
    title: string;
    description: string;
  };
  transport: {
    title: string;
    description: string;
  };
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
const EMPTY_DAYS: PlanDay[] = [];
const EMPTY_OVERVIEW: PlanOverviewSummary = {
  route: { title: "", description: "" },
  lodging: { title: "", description: "" },
  transport: { title: "", description: "" },
};
const TRANSPORT_ICONS: Record<PlanTransportKind, Icon> = {
  car: Car,
  walk: PersonSimpleWalk,
  transit: Train,
  cycling: Bicycle,
};
function transportDetails(option: PlanTransportOption): string {
  if (option.unavailable) return "暂无路线";
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
  readOnly = false,
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
            "button:not(:disabled)",
          ) ?? []),
        ];
        const index = buttons.findIndex(
          (button) => button === document.activeElement,
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
                  : rect.bottom + 8,
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
                left: position.left,
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
            document.body,
          )
        : null}
    </div>
  );
}
function PlaceCard({
  stop,
  highlighted,
  onHighlight,
}: {
  stop: PlanStop;
  highlighted: boolean;
  onHighlight?: (placeId: string) => void;
}) {
  const SettingIcon = stop.setting === "indoor" ? Buildings : Tree;
  const placeId = stop.placeId;
  const className = `inline-plan-place${highlighted ? " is-highlighted" : ""}${placeId ? "" : " is-static"}`;
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
  readOnly,
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
                  stop.placeId && stop.placeId === highlightedPlaceId,
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
  budgetEstimate,
}: {
  idPrefix: string;
  onSelectDay: (dayId: Exclude<PlanSectionId, "overview">) => void;
  days: PlanDay[];
  overview: PlanOverviewSummary;
  weatherDays: readonly WeatherDayData[];
  budgetEstimate?: PlanBudgetEstimate | null;
  notices?: readonly string[];
}) {
  const overviewAttractions = days.flatMap((day) =>
    day.stops
      .filter((stop) => stop.includeInOverview && stop.image)
      .map((stop) => ({
        id: day.id,
        image: stop.image!,
        title: stop.title,
      })),
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

      {weatherDays.length > 0 ? (
        <section
          className="weather-overview-section"
          aria-labelledby={`${idPrefix}-weather-title`}
        >
          <header>
            <div>
              <h3 id={`${idPrefix}-weather-title`}>旅行天气</h3>
              <p>按旅行日期查询；远期天气仅供趋势参考。</p>
            </div>
            <span>{weatherDays.length}天</span>
          </header>
          <WeatherDeck days={weatherDays} />
        </section>
      ) : null}

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
  onOpenMap,
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
  const days = presentation?.days ?? EMPTY_DAYS;
  const resolvedWeatherDays = presentation?.weatherDays ?? weatherDays;
  const overview = presentation?.overview ?? EMPTY_OVERVIEW;
  const tabs = [
    { id: "overview" as const, label: "总览" },
    ...days.map((day) => ({ id: day.id, label: `${day.date} ${day.weekday}` })),
  ];
  const [activeSection, setActiveSection] = useState<PlanSectionId>("overview");
  const activeDay = useMemo(
    () => days.find((day) => day.id === activeSection),
    [activeSection, days],
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
      candidate.stops.some((stop) => stop.placeId === highlightedPlaceId),
    );
    if (day) setActiveSection(day.id);
  }, [activeSection, days, highlightedPlaceId, selectedDayId]);
  const tabStyle = {
    "--plan-day-count": Math.max(days.length, 1),
  } as CSSProperties;
  if (!presentation) return null;
  return (
    <section className="plan-ready-attachment" aria-label="旅行方案">
      <header className="inline-plan-header">
        <div className="inline-plan-intro">
          <div className="inline-plan-title-row">
            <h2>{presentation?.title ?? undefined}</h2>
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
            <p>{presentation?.description ?? undefined}</p>
          ) : null}
          <div className="inline-plan-summary">
            <span>
              <CalendarBlank size={15} />
              {presentation?.dateLabel ?? undefined}
            </span>
            {presentation == null || presentation.paceLabel ? (
              <span>
                <Clock size={15} />
                {presentation?.paceLabel ?? undefined}
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
                    days.findIndex((candidate) => candidate.id === tab.id),
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
                days.findIndex((candidate) => candidate.id === dayId),
              );
            }}
            days={days}
            overview={overview}
            weatherDays={resolvedWeatherDays}
            notices={presentation?.notices}
            budgetEstimate={
              presentation ? presentation.budgetEstimate : budgetEstimate
            }
          />
        )}
      </div>
    </section>
  );
}
