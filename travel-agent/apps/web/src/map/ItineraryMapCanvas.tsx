import type { DiscoveryApi } from "./discoveryMap";
import { readAmapConfig, loadAmapApi } from "./amapRuntime";
import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import {
  ArrowClockwise,
  CornersOut,
  MapTrifold,
  Minus,
  Plus
} from "@phosphor-icons/react";
import type { MapUpdatePayload } from "../generated/contracts";
import { MapPlaceIcon } from "./MapPlaceIcon";
import { MAP_PLACE_LEGEND, mapPlaceCategory } from "./mapPlaceKinds";

export interface AmapConfig {
  key: string;
  securityCode?: string;
  serviceHost?: string;
}
declare global {
  interface Window {
    _AMapSecurityConfig?: { securityJsCode?: string; serviceHost?: string };
  }
}
interface AmapMarker {
  on: (event: string, handler: () => void) => void;
}
export interface AmapMap {
  add: (overlays: unknown[]) => void;
  remove?: (overlays: unknown[]) => void;
  destroy: () => void;
  setFitView: (
    overlays?: unknown[],
    immediately?: boolean,
    avoid?: number[],
    maxZoom?: number
  ) => void;
  setCenter?: (
    center: number[],
    immediately?: boolean,
    duration?: number
  ) => void;
  getZoom?: () => number;
  setZoomAndCenter?: (
    zoom: number,
    center: number[],
    immediately?: boolean,
    duration?: number
  ) => void;
  setBounds?: (bounds: unknown) => void;
  getFitZoomAndCenterByBounds?: (
    bounds: unknown,
    avoid?: number[],
    maxZoom?: number
  ) => [number, unknown];
  getFitZoomAndCenterByOverlays?: (
    overlays: unknown[],
    avoid?: number[],
    maxZoom?: number
  ) => [number, unknown];
  addControl?: (control: unknown) => void;
  resize?: () => void;
  zoomIn?: () => void;
  zoomOut?: () => void;
}
export interface AmapApi extends DiscoveryApi {
  Map: new (
    container: HTMLElement,
    options: Record<string, unknown>
  ) => AmapMap;
  Marker: new (options: Record<string, unknown>) => AmapMarker;
  Bounds?: new (southWest: number[], northEast: number[]) => unknown;
  Polygon?: new (options: Record<string, unknown>) => unknown;
  Scale?: new () => unknown;
  Polyline: new (options: Record<string, unknown>) => unknown;
}
const DEFAULT_AMAP_CONFIG = readAmapConfig();
// AMap uses top, bottom, left, right (not CSS order). Reserve the toolbar edge.
const MAP_FIT_AVOID = [72, 64, 40, 96];

export function ItineraryMapCanvas({
  update,
  highlightedPlaceId,
  onHighlightPlace,
  config = DEFAULT_AMAP_CONFIG,
  loadApi = loadAmapApi,
  color
}: {
  update: MapUpdatePayload | null;
  highlightedPlaceId: string | null;
  onHighlightPlace: (placeId: string) => void;
  loadApi?: (config: AmapConfig) => Promise<AmapApi>;
  config?: AmapConfig | null;
  color: string;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<AmapMap | null>(null);
  const overlaysRef = useRef<unknown[]>([]);
  const buttonsRef = useRef<Map<string, HTMLButtonElement>>(new Map());
  const onHighlightRef = useRef(onHighlightPlace);
  onHighlightRef.current = onHighlightPlace;
  const [mapResource, setMapResource] = useState<{
    api: AmapApi;
    map: AmapMap;
  } | null>(null);
  const [loadState, setLoadState] = useState("idle");
  const [retry, setRetry] = useState(0);
  const [pinTargets, setPinTargets] = useState<
    { id: string; kind: string; button: HTMLButtonElement }[]
  >([]);
  const hasMarkers = Boolean(update?.markers?.length);
  const selectedMarker = update?.markers?.find(
    (marker) => marker.place_id === highlightedPlaceId
  );

  useEffect(() => {
    if (!config || !containerRef.current || !hasMarkers) return;
    let disposed = false;
    let map: AmapMap | null = null;
    setLoadState("loading");
    void loadApi(config)
      .then((AMap) => {
        if (disposed || !containerRef.current) return;
        map = new AMap.Map(containerRef.current, {
          mapStyle: "amap://styles/normal",
          showLabel: true,
          features: ["bg", "road", "building", "point"],
          zoom: 12,
          viewMode: "2D",
          resizeEnable: true,
          keyboardEnable: false
        });
        mapRef.current = map;
        setMapResource({ api: AMap, map });
        setLoadState("ready");
      })
      .catch(() => {
        if (!disposed) setLoadState("failed");
      });
    const buttons = buttonsRef.current;
    return () => {
      disposed = true;
      map?.destroy();
      mapRef.current = null;
      overlaysRef.current = [];
      buttons.clear();
      setMapResource(null);
      setPinTargets([]);
    };
  }, [config, loadApi, retry, hasMarkers]);

  useEffect(() => {
    if (!mapResource || mapResource.map !== mapRef.current || !update) return;
    const { map, api } = mapResource;
    map.remove?.(overlaysRef.current);
    buttonsRef.current.clear();
    const targets: typeof pinTargets = [];
    const markers = (update.markers ?? []).map((marker) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "amap-trip-marker itinerary-map-pin";
      const category = mapPlaceCategory(marker.kind);
      // Names remain available to assistive technology, not as map labels.
      button.setAttribute("aria-label", `${marker.label}，${category.label}`);
      button.addEventListener("click", (event) => {
        event.stopPropagation();
        onHighlightRef.current(marker.place_id);
      });
      button.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          event.stopPropagation();
          onHighlightRef.current(marker.place_id);
        }
      });
      buttonsRef.current.set(marker.place_id, button);
      targets.push({ id: marker.place_id, kind: marker.kind, button });
      const overlay = new api.Marker({
        position: [marker.coordinates.longitude, marker.coordinates.latitude],
        title: category.label,
        content: button,
        anchor: "center",
        zIndex: 120
      });
      return overlay;
    });
    const routes = (update.routes ?? [])
      .filter((route) => route.polyline.length > 1)
      .map(
        (route) =>
          new api.Polyline({
            path: route.polyline.map((point) => [
              point.longitude,
              point.latitude
            ]),
            strokeColor: color,
            strokeOpacity: 0.82,
            strokeWeight: 4,
            isOutline: true,
            borderWeight: 2,
            outlineColor: "#ffffff",
            showDir: true,
            lineJoin: "round",
            zIndex: 50
          })
      );
    const overlays = [...routes, ...markers];
    map.add(overlays);
    overlaysRef.current = overlays;
    setPinTargets(targets);
    map.setFitView(overlays, true, MAP_FIT_AVOID, 16);
  }, [mapResource, update, color]);

  useEffect(() => {
    buttonsRef.current.forEach((button, id) => {
      const active = id === highlightedPlaceId;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-pressed", String(active));
    });
    if (selectedMarker) {
      const map = mapRef.current;
      const center = [
        selectedMarker.coordinates.longitude,
        selectedMarker.coordinates.latitude
      ];
      const immediate =
        window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? true;
      if (map?.setZoomAndCenter) {
        map.setZoomAndCenter(
          Math.max(map.getZoom?.() ?? 12, 14),
          center,
          immediate,
          200
        );
      } else map?.setCenter?.(center, immediate, 200);
    }
  }, [highlightedPlaceId, selectedMarker, mapResource, update, color]);

  return (
    <div
      className="itinerary-map-viewport"
      data-map-state={config ? loadState : "unconfigured"}
    >
      {pinTargets.map(({ id, kind, button }) =>
        createPortal(<MapPlaceIcon kind={kind} />, button, id)
      )}
      <div
        ref={containerRef}
        className="itinerary-map-canvas"
        role="group"
        aria-label={`第 ${(update?.selected_day_index ?? 0) + 1} 天行程地图，共 ${update?.markers?.length ?? 0} 个地点`}
        data-route-count={update?.routes?.length ?? 0}
      />
      {hasMarkers && config && loadState === "ready" ? (
        <ul className="itinerary-map-legend" aria-label="地图图例">
          {MAP_PLACE_LEGEND.map(({ kind, label }) => (
            <li key={kind}>
              <MapPlaceIcon kind={kind} />
              <span>{label}</span>
            </li>
          ))}
        </ul>
      ) : null}
      {!hasMarkers || !config || loadState !== "ready" ? (
        <div
          className="itinerary-map-message"
          role={loadState === "failed" ? "alert" : "status"}
        >
          <MapTrifold size={30} weight="light" />
          <strong>
            {!hasMarkers
              ? "这一天暂时没有可绘制路线"
              : !config
                ? "高德底图尚未配置"
                : loadState === "failed"
                  ? "地图暂时未能加载"
                  : "正在加载行程地图"}
          </strong>
          {loadState === "failed" ? (
            <button
              type="button"
              onClick={() => setRetry((value) => value + 1)}
            >
              <ArrowClockwise size={16} />
              重新加载
            </button>
          ) : (
            <span>
              {hasMarkers
                ? "地点与行程仍可正常查看"
                : "地点信息补齐后会显示在这里"}
            </span>
          )}
        </div>
      ) : (
        <div className="itinerary-map-controls" aria-label="地图视野控制">
          <button
            type="button"
            title="查看完整路线"
            aria-label="查看完整路线"
            onClick={() =>
              mapRef.current?.setFitView(
                overlaysRef.current,
                true,
                MAP_FIT_AVOID,
                16
              )
            }
          >
            <CornersOut size={19} />
          </button>
          <button
            type="button"
            aria-label="放大地图"
            onClick={() => mapRef.current?.zoomIn?.()}
          >
            <Plus size={19} />
          </button>
          <button
            type="button"
            aria-label="缩小地图"
            onClick={() => mapRef.current?.zoomOut?.()}
          >
            <Minus size={19} />
          </button>
        </div>
      )}
    </div>
  );
}
