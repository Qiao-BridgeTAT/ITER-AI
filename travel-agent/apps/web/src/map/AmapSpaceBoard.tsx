import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties
} from "react";
import { createPortal } from "react-dom";
import { ArrowLeft, ArrowsOut, MapTrifold, X } from "@phosphor-icons/react";

import type { MapUpdatePayload } from "../generated/contracts";
import type { PlanDay, PlanStop } from "../conversation/PlanReadyAttachment";
import { PlacePhoto } from "../conversation/PlacePhoto";
import { formatItineraryDayTimes } from "../planning/itineraryTime";
import {
  ItineraryMapCanvas,
  type AmapApi,
  type AmapConfig
} from "./ItineraryMapCanvas";
import "./itinerary-map.css";
import { DiscoveryMapCanvas } from "./DiscoveryMapCanvas";
import type { DiscoveryMapPreview } from "./discoveryMap";

interface AmapSpaceBoardProps {
  update: MapUpdatePayload | null;
  discovery?: DiscoveryMapPreview;
  onExitDiscovery?: () => void;
  highlightedPlaceId: string | null;
  onHighlightPlace: (placeId: string) => void;
  routeNotice?: string | null;
  config?: AmapConfig | null;
  loadApi?: (config: AmapConfig) => Promise<AmapApi>;
  days?: readonly PlanDay[];
  onSelectDay?: (dayIndex: number) => void;
  expanded?: boolean;
  onExpandedChange?: (expanded: boolean) => void;
}

const DAY_COLORS = ["#176de5", "#257969", "#926039", "#7962a8", "#2f7789"];

export function AmapSpaceBoard({
  update,
  discovery,
  onExitDiscovery,
  highlightedPlaceId,
  onHighlightPlace,
  routeNotice = null,
  config,
  loadApi,
  days = [],
  onSelectDay,
  expanded: controlledExpanded,
  onExpandedChange
}: AmapSpaceBoardProps) {
  const [localExpanded, setLocalExpanded] = useState(false);
  const expanded = controlledExpanded ?? localExpanded;
  const dialogRef = useRef<HTMLDialogElement>(null);
  const toggleRef = useRef<HTMLButtonElement>(null);
  const selectedRowRef = useRef<HTMLButtonElement>(null);
  const dayIndex = update?.selected_day_index ?? 0;
  const day = days[dayIndex];
  const times = day ? formatItineraryDayTimes(day) : null;
  const color = DAY_COLORS[dayIndex % DAY_COLORS.length];
  const setExpanded = (next: boolean) => {
    setLocalExpanded(next);
    onExpandedChange?.(next);
  };
  const rows = useMemo<PlanStop[]>(() => {
    if (day) {
      const stops = day.stops.filter((stop) => stop.placeId);
      if (day.startStop?.placeId && day.startStop.placeId !== stops[0]?.placeId)
        stops.unshift(day.startStop);
      if (day.endStop?.placeId && day.endStop.placeId !== stops.at(-1)?.placeId)
        stops.push(day.endStop);
      return stops;
    }
    return (update?.markers ?? []).map((marker) => ({
      placeId: marker.place_id,
      title: marker.label,
      category: entryKindLabel(marker.kind),
      time: "",
      plannedStay: "",
      imageAlt: `${marker.label}实景`,
      description: ""
    }));
  }, [day, update]);
  // Closing the portal mounts a new sidebar button; focus its latest DOM node.
  const focusToggle = useCallback(() => toggleRef.current?.focus(), []);

  useEffect(() => {
    if (!expanded) return;
    const dialog = dialogRef.current;
    const previous = document.activeElement;
    const overflow = document.body.style.overflow;
    dialog?.showModal();
    document.body.style.overflow = "hidden";
    return () => {
      dialog?.close();
      document.body.style.overflow = overflow;
      if (
        previous instanceof HTMLElement &&
        previous !== document.body &&
        previous.isConnected
      )
        previous.focus();
      else focusToggle();
    };
  }, [expanded, focusToggle]);
  useEffect(() => {
    const button = selectedRowRef.current;
    const list = button?.closest("ol");
    if (!highlightedPlaceId || !button || !list) return;
    const rowRect = button.getBoundingClientRect();
    const listRect = list.getBoundingClientRect();
    if (rowRect.bottom > listRect.bottom)
      list.scrollTop += rowRect.bottom - listRect.bottom;
    else if (rowRect.top < listRect.top)
      list.scrollTop -= listRect.top - rowRect.top;
  }, [highlightedPlaceId]);

  const body = (
    <div
      className={`itinerary-map${expanded ? " is-expanded" : ""}${discovery ? " is-discovery" : ""}`}
      style={{ "--map-day-color": color } as CSSProperties}
    >
      <header className="itinerary-map-header">
        <div>
          <MapTrifold size={20} />
          <h2 id="route-map-title">行程地图</h2>
        </div>
        <div>
          {discovery && onExitDiscovery ? (
            <button
              className="map-icon-button"
              type="button"
              aria-label="返回行程地图"
              title="返回行程地图"
              onClick={onExitDiscovery}
            >
              <ArrowLeft size={18} />
            </button>
          ) : null}
          <button
            className="map-icon-button"
            ref={toggleRef}
            type="button"
            aria-label={expanded ? "关闭行程地图" : "展开行程地图"}
            onClick={() => setExpanded(!expanded)}
          >
            {expanded ? <X size={20} /> : <ArrowsOut size={18} />}
          </button>
        </div>
      </header>
      {days.length > 0 ? (
        <nav className="itinerary-map-days" aria-label="地图日期">
          {days.map((item, index) => (
            <button
              key={item.id}
              type="button"
              aria-pressed={index === dayIndex}
              onClick={() => onSelectDay?.(index)}
            >
              <span>{item.date}</span>
              <small>{item.weekday}</small>
            </button>
          ))}
        </nav>
      ) : null}
      <div className="itinerary-map-layout">
        {discovery ? (
          <DiscoveryMapCanvas
            preview={discovery}
            onHighlightPlace={onHighlightPlace}
            config={config}
            loadApi={loadApi}
          />
        ) : (
          <ItineraryMapCanvas
            update={update}
            highlightedPlaceId={highlightedPlaceId}
            onHighlightPlace={onHighlightPlace}
            config={config}
            loadApi={loadApi}
            color={color}
          />
        )}
        {expanded && !discovery ? (
          <div className="itinerary-map-itinerary">
            <div className="itinerary-map-route-heading">
              <strong>{day ? `${day.ordinal}路线` : "当天路线"}</strong>
              {day ? (
                <span>
                  约 {times?.startTime} - {times?.endTime}
                </span>
              ) : null}
            </div>
            <ol className="itinerary-map-stops" aria-label="当前日期地图地点">
              {rows.map((stop, index) => {
                const selected = stop.placeId === highlightedPlaceId;
                const mapped = update?.markers?.some(
                  (marker) => marker.place_id === stop.placeId
                );
                return (
                  <li key={`${stop.placeId}-${index}`}>
                    <button
                      type="button"
                      aria-pressed={selected}
                      ref={
                        selected &&
                        index ===
                          rows.findIndex(
                            (row) => row.placeId === highlightedPlaceId
                          )
                          ? selectedRowRef
                          : undefined
                      }
                      disabled={!mapped}
                      onClick={() =>
                        stop.placeId && onHighlightPlace(stop.placeId)
                      }
                    >
                      <span className="map-stop-number">{index + 1}</span>
                      {expanded ? (
                        <PlacePhoto
                          key={stop.image ?? "missing"}
                          src={stop.image}
                          alt={stop.imageAlt || `${stop.title}实景`}
                        />
                      ) : null}
                      <span className="map-stop-copy">
                        <span className="map-stop-meta">
                          {stop.time ? (
                            <time
                              title={
                                stop.timeIsFixed
                                  ? "已确认的固定时间"
                                  : "预计到达时间"
                              }
                            >
                              {stop === day?.startStop
                                ? times?.startTime
                                : stop === day?.endStop
                                  ? times?.endTime
                                  : (times?.stopTimes[
                                      day?.stops.indexOf(stop) ?? -1
                                    ] ?? stop.time)}
                            </time>
                          ) : null}
                          {stop.category}
                        </span>
                        <strong>{stop.title}</strong>
                        {expanded && stop.plannedStay ? (
                          <span className="map-stop-stay">
                            {stop.plannedStay}
                          </span>
                        ) : null}
                        {!mapped ? (
                          <span className="map-stop-stay">位置待补充</span>
                        ) : null}
                      </span>
                    </button>
                  </li>
                );
              })}
            </ol>
            {routeNotice ? (
              <details className="itinerary-map-notice">
                <summary>部分路段暂无轨迹</summary>
                <p>{routeNotice}</p>
              </details>
            ) : null}
          </div>
        ) : null}
      </div>
    </div>
  );
  return expanded
    ? createPortal(
        <dialog
          className="itinerary-map-dialog"
          ref={dialogRef}
          aria-labelledby="route-map-title"
          onCancel={(event) => {
            event.preventDefault();
            setExpanded(false);
          }}
          onClick={(event) => {
            if (event.target === event.currentTarget) setExpanded(false);
          }}
        >
          {body}
        </dialog>,
        document.body
      )
    : body;
}

function entryKindLabel(kind: string): string {
  return (
    (
      {
        attraction: "景点",
        restaurant: "餐厅",
        hotel: "住宿",
        transport: "交通"
      } as Record<string, string>
    )[kind] ?? "行程地点"
  );
}
