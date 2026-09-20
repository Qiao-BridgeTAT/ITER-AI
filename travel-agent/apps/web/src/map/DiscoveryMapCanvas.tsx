import { loadAmapApi, readAmapConfig } from "./amapRuntime";
import { useEffect, useRef, useState } from "react";
import { ArrowClockwise, CornersOut, Minus, Plus } from "@phosphor-icons/react";
import {
  type AmapApi,
  type AmapMap,
  type AmapConfig
} from "./ItineraryMapCanvas";
import {
  loadDistrict,
  loadPlacePoint,
  mapPoint,
  type DiscoveryMapPreview,
  type MapPoint
} from "./discoveryMap";

const CONFIG = readAmapConfig();
const PADDING = [40, 48, 32, 64];
export function DiscoveryMapCanvas({
  preview,
  onHighlightPlace,
  config = CONFIG,
  loadApi = loadAmapApi
}: {
  preview: DiscoveryMapPreview;
  onHighlightPlace: (id: string) => void;
  config?: AmapConfig | null;
  loadApi?: (config: AmapConfig) => Promise<AmapApi>;
}) {
  const container = useRef<HTMLDivElement>(null);
  const activeMap = useRef<AmapMap | null>(null);
  const [resource, setResource] = useState<{
    api: AmapApi;
    map: AmapMap;
  } | null>(null);
  const [status, setStatus] = useState("loading");
  const [retry, setRetry] = useState(0);
  const [sizeVersion, setSizeVersion] = useState(0);
  const [boundary, setBoundary] = useState<{
    code: string;
    polygons: unknown[];
    center?: MapPoint;
  } | null>(null);
  const [locations, setLocations] = useState<Record<string, MapPoint>>({});
  const [cityError, setCityError] = useState<string | null>(null);
  const [failedPlaces, setFailedPlaces] = useState<Set<string>>(new Set());
  const [overviewFor, setOverviewFor] = useState<{
    id: string | null;
    request?: number;
  } | null>(null);
  const lastView = useRef("");
  const markers = useRef(new Map<string, HTMLButtonElement>());
  const highlight = useRef(onHighlightPlace);
  highlight.current = onHighlightPlace;
  const cityCode = preview.city?.code;
  const focused = preview.places.find(
    (place) => place.id === preview.focusedId
  );
  const focusedPoint =
    focused?.point ?? (focused?.amapId ? locations[focused.amapId] : undefined);
  const showFocus = Boolean(
    focusedPoint &&
    !(
      overviewFor &&
      overviewFor.id === preview.focusedId &&
      overviewFor.request === preview.focusRequest
    )
  );

  useEffect(() => {
    if (!config || !container.current) return;
    let disposed = false;
    let map: AmapMap | undefined;
    let observer: ResizeObserver | undefined;
    setStatus("loading");
    void loadApi(config)
      .then((api) => {
        if (disposed || !container.current) return;
        map = new api.Map(container.current, {
          center: [104, 35],
          zoom: 3,
          viewMode: "2D",
          resizeEnable: true,
          keyboardEnable: false,
          mapStyle: "amap://styles/normal"
        });
        activeMap.current = map;
        if (api.Scale) map.addControl?.(new api.Scale());
        setResource({ api, map });
        setStatus("ready");
        if (typeof ResizeObserver !== "undefined") {
          observer = new ResizeObserver(() => {
            if (disposed) return;
            map?.resize?.();
            setSizeVersion((value) => value + 1);
          });
          observer.observe(container.current);
        }
      })
      .catch(() => {
        if (!disposed) setStatus("failed");
      });
    return () => {
      disposed = true;
      observer?.disconnect();
      // Later effect cleanups must not remove overlays from a destroyed SDK map.
      if (activeMap.current === map) activeMap.current = null;
      map?.destroy();
      setResource(null);
      lastView.current = "";
    };
  }, [config, loadApi, retry]);

  useEffect(() => {
    if (!resource || !cityCode) {
      setBoundary(null);
      return;
    }
    let disposed = false;
    let polygons: unknown[] = [];
    setCityError(null);
    void loadDistrict(resource.api, cityCode)
      .then((result) => {
        if (disposed) return;
        polygons = resource.api.Polygon
          ? (result.boundaries ?? []).map(
              (path) =>
                new resource.api.Polygon!({
                  path,
                  strokeColor: "#7ba8db",
                  strokeWeight: 1.5,
                  fillColor: "#a7cafa",
                  fillOpacity: 0.08,
                  zIndex: 10,
                  bubble: true
                })
            )
          : [];
        resource.map.add(polygons);
        setBoundary({
          code: cityCode,
          polygons,
          center: mapPoint(result.center)
        });
      })
      .catch(() => {
        if (!disposed) setCityError(cityCode);
      });
    return () => {
      disposed = true;
      if (activeMap.current === resource.map) resource.map.remove?.(polygons);
    };
  }, [resource, cityCode, retry]);

  // New cards carry Provider coordinates. Resolve only older cards by exact POI ID.
  useEffect(() => {
    if (!resource) return;
    let disposed = false;
    const missing = preview.places.filter(
      (place) => !place.point && place.amapId
    );
    let next = 0;
    async function worker() {
      while (!disposed && next < missing.length) {
        const id = missing[next++].amapId!;
        try {
          const point = await loadPlacePoint(resource!.api, id);
          if (!disposed)
            setLocations((current) => ({ ...current, [id]: point }));
        } catch {
          if (!disposed) setFailedPlaces((current) => new Set(current).add(id));
        }
      }
    }
    void Promise.all(
      Array.from({ length: Math.min(3, missing.length) }, worker)
    );
    return () => {
      disposed = true;
    };
  }, [resource, preview.places, retry]);

  useEffect(() => {
    if (!resource) return;
    const markerButtons = markers.current;
    markerButtons.clear();
    const overlays = preview.places.flatMap((place) => {
      const point =
        place.point ?? (place.amapId ? locations[place.amapId] : undefined);
      if (!point) return [];
      const button = document.createElement("button");
      button.type = "button";
      button.className = "discovery-map-marker";
      button.setAttribute("aria-label", place.label);
      const label = document.createElement("span");
      label.textContent = place.label;
      button.append(label);
      button.onclick = () => highlight.current(place.id);
      markerButtons.set(place.id, button);
      return [
        new resource.api.Marker({
          position: point,
          content: button,
          anchor: "center",
          zIndex: 120
        })
      ];
    });
    resource.map.add(overlays);
    return () => {
      if (activeMap.current === resource.map) resource.map.remove?.(overlays);
      markerButtons.clear();
    };
  }, [resource, preview.places, locations]); // Focus only changes marker styling and viewport.

  useEffect(() => {
    markers.current.forEach((button, id) => {
      button.classList.toggle("is-active", id === preview.focusedId);
      button.setAttribute("aria-pressed", String(id === preview.focusedId));
    });
    if (!resource) return;
    const { map, api } = resource;
    const matchingBoundary = boundary?.code === cityCode ? boundary : null;
    if (cityCode && !matchingBoundary && !showFocus && cityError !== cityCode)
      return;
    const key = `${cityCode}:${showFocus ? preview.focusedId + ":" + focusedPoint : matchingBoundary ? "city" : "country"}:${sizeVersion}:${preview.focusRequest}`;
    if (lastView.current === key) return;
    const immediate =
      lastView.current === "" ||
      (window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ??
        false);
    lastView.current = key;
    const move = (target: [number, unknown] | undefined, duration: number) => {
      if (!target) return false;
      const center = Array.isArray(target[1])
        ? (target[1] as number[])
        : mapPoint(target[1] as Parameters<typeof mapPoint>[0]);
      if (!center || !map.setZoomAndCenter) return false;
      map.setZoomAndCenter(target[0], center, immediate, duration);
      return true;
    };
    if (showFocus && focusedPoint) {
      // A point has no known area: show its neighbourhood, never imply a boundary.
      map.setZoomAndCenter?.(15, focusedPoint, immediate, 450);
    } else if (matchingBoundary?.polygons.length) {
      if (
        !move(
          map.getFitZoomAndCenterByOverlays?.(
            matchingBoundary.polygons,
            PADDING,
            13
          ),
          1400
        )
      )
        map.setFitView(matchingBoundary.polygons, immediate, PADDING, 13);
    } else if (matchingBoundary?.center) {
      map.setZoomAndCenter?.(10, matchingBoundary.center, immediate, 1400);
    } else if (api.Bounds) {
      const bounds = new api.Bounds([73, 3], [135.1, 53.6]);
      if (!move(map.getFitZoomAndCenterByBounds?.(bounds, PADDING, 5), 1200))
        map.setBounds?.(bounds);
    }
  }, [
    resource,
    boundary,
    cityError,
    cityCode,
    preview.focusedId,
    preview.focusRequest,
    focusedPoint,
    showFocus,
    sizeVersion,
    locations
  ]);

  const waitingForInitialCity = Boolean(
    cityCode &&
    boundary?.code !== cityCode &&
    !focusedPoint &&
    cityError !== cityCode &&
    lastView.current === ""
  );
  const locationMissing =
    focused &&
    !focusedPoint &&
    (!focused.amapId || failedPlaces.has(focused.amapId));
  return (
    <div
      className="itinerary-map-viewport"
      data-map-state={!config ? "unconfigured" : status}
      data-discovery-map-mode={
        showFocus ? "place" : cityCode ? "city" : "country"
      }
    >
      <div
        ref={container}
        className="itinerary-map-canvas"
        role="group"
        aria-label={
          showFocus
            ? `${focused?.label}的位置`
            : preview.city
              ? `${preview.city.name}行政区地图`
              : "中国全域地图"
        }
      />
      {config && status === "ready" && !waitingForInitialCity ? (
        <>
          <div className="discovery-map-caption" role="status">
            {showFocus ? focused?.label : (preview.city?.name ?? "中国")}
            {locationMissing
              ? " · 该地点位置暂不可用"
              : cityError === cityCode && cityCode
                ? " · 城市边界暂不可用"
                : ""}
          </div>
          <div className="itinerary-map-controls" aria-label="地图视野控制">
            <button
              type="button"
              aria-label={cityCode ? "查看全市" : "查看全国"}
              onClick={() => {
                lastView.current = "";
                setOverviewFor({
                  id: preview.focusedId,
                  request: preview.focusRequest
                });
                setSizeVersion((value) => value + 1);
              }}
            >
              <CornersOut size={19} />
            </button>
            <button
              type="button"
              aria-label="放大地图"
              onClick={() => resource?.map.zoomIn?.()}
            >
              <Plus size={19} />
            </button>
            <button
              type="button"
              aria-label="缩小地图"
              onClick={() => resource?.map.zoomOut?.()}
            >
              <Minus size={19} />
            </button>
          </div>
        </>
      ) : (
        <div className="itinerary-map-message" role="status">
          <strong>
            {!config
              ? "高德底图尚未配置"
              : status === "failed"
                ? "地图暂时未能加载"
                : "正在加载地图"}
          </strong>
          {config && status === "failed" ? (
            <button
              type="button"
              onClick={() => setRetry((value) => value + 1)}
            >
              <ArrowClockwise size={16} />
              重新加载
            </button>
          ) : null}
        </div>
      )}
    </div>
  );
}
