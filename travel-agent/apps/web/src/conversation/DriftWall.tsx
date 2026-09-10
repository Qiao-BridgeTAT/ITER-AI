import {
  type CSSProperties,
  type PointerEvent as ReactPointerEvent,
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import "./DriftWall.css";
import {
  getFittedPlaneScale,
  getLoopPlaneHeight,
  getLoopTrackLayout,
  getPlaneHorizontalOffset,
  wrapLoopOffset,
} from "./driftWallLayout";
import { shuffledWallColumns } from "./driftWallShuffle";
import { PlaceImage } from "./PlaceImage";

export type DriftWallItem = {
  id: string;
  image: string;
  title: string;
};

type DriftWallProps = {
  items: DriftWallItem[];
  columns?: number;
  tileWidth?: number;
  tileHeight?: number;
  gap?: number;
  radius?: number;
  tilt?: number;
  turn?: number;
  roll?: number;
  perspective?: number;
  depth?: number;
  speed?: number;
  direction?: "up" | "down";
  variance?: number;
  parallax?: number;
  pauseOnHover?: boolean;
  lift?: number;
  fade?: number;
  dim?: number;
  grayscale?: boolean;
  overlayColor?: string;
  className?: string;
  style?: CSSProperties;
  ariaLabel?: string;
  onItemSelect?: (item: DriftWallItem) => void;
};

type DriftWallVariables = CSSProperties &
  Record<
    | "--dw-tile-w"
    | "--dw-tile-h"
    | "--dw-gap"
    | "--dw-radius"
    | "--dw-perspective"
    | "--dw-lift"
    | "--dw-dim"
    | "--dw-gray"
    | "--dw-overlay"
    | "--dw-edge",
    string | number
  >;

const prefersReducedMotion = () =>
  typeof window !== "undefined" &&
  typeof window.matchMedia === "function" &&
  window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const columnFactor = (index: number, variance: number) => {
  const pseudo = ((index * 0.6180339887 + 0.35) % 1) * 2 - 1;
  return 1 + variance * pseudo;
};

export default function DriftWall({
  items,
  columns = 4,
  tileWidth = 176,
  tileHeight = 116,
  gap = 14,
  radius = 10,
  tilt = 6,
  turn = -8,
  roll = 0,
  perspective = 1500,
  depth = 48,
  speed = 18,
  direction = "up",
  variance = 0.28,
  parallax = 0.24,
  pauseOnHover = false,
  lift = 22,
  fade = 0.3,
  dim = 0.74,
  grayscale = false,
  overlayColor = "#17324d",
  className = "",
  style,
  ariaLabel = "行程景点概览",
  onItemSelect,
}: DriftWallProps) {
  const safeColumns = Math.max(1, Math.floor(columns));
  const containerRef = useRef<HTMLDivElement>(null);
  const planeRef = useRef<HTMLDivElement>(null);
  const trackRefs = useRef<Array<HTMLDivElement | null>>([]);
  const rafRef = useRef<number | null>(null);
  const offsetsRef = useRef<number[]>([]);
  const velocitiesRef = useRef<number[]>([]);
  const hoveredColRef = useRef(-1);
  const wallHoveredRef = useRef(false);
  const pointerRef = useRef({ x: 0, y: 0 });
  const pointerDampedRef = useRef({ x: 0, y: 0 });
  const lastTsRef = useRef<number | null>(null);
  const [containerHeight, setContainerHeight] = useState(320);
  const [containerWidth, setContainerWidth] = useState(760);
  const [activeId, setActiveId] = useState<string | null>(null);
  const activeIdRef = useRef<string | null>(null);
  const [reduced, setReduced] = useState(prefersReducedMotion);
  const [shuffleSeed] = useState(() => Math.floor(Math.random() * 0x100000000));

  useEffect(() => {
    setReduced(prefersReducedMotion());
    if (typeof window.matchMedia !== "function") return;

    const mediaQuery = window.matchMedia("(prefers-reduced-motion: reduce)");
    const onChange = (event: MediaQueryListEvent) => setReduced(event.matches);
    mediaQuery.addEventListener?.("change", onChange);
    return () => mediaQuery.removeEventListener?.("change", onChange);
  }, []);

  // Content, not parent-array identity, controls the shuffle. Hover, resize and
  // streamed status updates must not reshuffle the wall while it is being read.
  const itemsKey = JSON.stringify(items);
  const columnItems = useMemo(
    () =>
      shuffledWallColumns(
        JSON.parse(itemsKey) as DriftWallItem[],
        safeColumns,
        shuffleSeed,
      ),
    [itemsKey, safeColumns, shuffleSeed],
  );

  const planeScale = useMemo(
    () => getFittedPlaneScale(containerWidth, safeColumns, tileWidth, gap),
    [containerWidth, gap, safeColumns, tileWidth],
  );
  const planeHeight = getLoopPlaneHeight(
    containerHeight,
    planeScale,
    tileHeight,
    gap,
  );
  const planeWidth = safeColumns * (tileWidth + gap);
  const horizontalOffset = useMemo(
    () =>
      getPlaneHorizontalOffset({
        width: planeWidth,
        height: planeHeight,
        scale: planeScale,
        tilt,
        turn,
        roll,
        depth,
        perspective,
      }),
    [depth, perspective, planeHeight, planeScale, planeWidth, roll, tilt, turn],
  );
  const columnMeta = useMemo(
    () =>
      columnItems.map((column) =>
        getLoopTrackLayout(planeHeight, column.length, tileHeight, gap),
      ),
    [columnItems, gap, planeHeight, tileHeight],
  );

  useLayoutEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const bounds = container.getBoundingClientRect();
    setContainerHeight(bounds.height || 320);
    setContainerWidth(bounds.width || 760);
    if (typeof ResizeObserver === "undefined") return;

    const observer = new ResizeObserver(([entry]) => {
      setContainerHeight(entry.contentRect.height || 320);
      setContainerWidth(entry.contentRect.width || 760);
    });
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  const baseVelocities = useMemo(() => {
    const directionSign = direction === "up" ? 1 : -1;
    return columnItems.map((_, columnIndex) => {
      const alternateSign = columnIndex % 2 === 0 ? 1 : -1;
      return (
        speed *
        columnFactor(columnIndex, variance) *
        directionSign *
        alternateSign
      );
    });
  }, [columnItems, direction, speed, variance]);

  useLayoutEffect(() => {
    offsetsRef.current = columnMeta.map((meta, columnIndex) =>
      wrapLoopOffset(
        offsetsRef.current[columnIndex] ??
          meta.copyHeight * ((columnIndex * 0.37) % 1),
        meta.copyHeight,
      ),
    );
    velocitiesRef.current = columnMeta.map(
      (_, columnIndex) => velocitiesRef.current[columnIndex] ?? 0,
    );
    // Position every column before paint, including resize, reduced motion,
    // and environments without animation frames. Resizing keeps its phase.
    columnMeta.forEach((_, columnIndex) => {
      const track = trackRefs.current[columnIndex];
      if (track) {
        const offset = reduced ? 0 : offsetsRef.current[columnIndex];
        track.style.transform = `translate3d(0, ${-offset}px, 0)`;
      }
    });
  }, [columnMeta, reduced]);

  const applyPlaneTransform = useCallback(
    (pointerX: number, pointerY: number) => {
      const plane = planeRef.current;
      if (!plane) return;
      plane.style.transform =
        `translate(-50%, -50%) translateX(${horizontalOffset}px) scale(${planeScale}) ` +
        `rotateX(${tilt + pointerY}deg) rotateY(${turn + pointerX}deg) ` +
        `rotateZ(${roll}deg) translateZ(${-depth}px)`;
    },
    [depth, horizontalOffset, planeScale, roll, tilt, turn],
  );

  useLayoutEffect(() => {
    applyPlaneTransform(0, 0);
  }, [applyPlaneTransform]);

  useEffect(() => {
    if (reduced || typeof window.requestAnimationFrame !== "function") {
      applyPlaneTransform(0, 0);
      return;
    }

    const animate = (timestamp: number) => {
      if (lastTsRef.current === null) lastTsRef.current = timestamp;
      const delta = Math.min(
        0.05,
        Math.max(0, timestamp - lastTsRef.current) / 1000,
      );
      lastTsRef.current = timestamp;

      const maxTilt = parallax * 8;
      const targetX = pointerRef.current.x * maxTilt;
      const targetY = -pointerRef.current.y * maxTilt;
      const damp = 1 - Math.exp(-delta / 0.12);
      pointerDampedRef.current.x +=
        (targetX - pointerDampedRef.current.x) * damp;
      pointerDampedRef.current.y +=
        (targetY - pointerDampedRef.current.y) * damp;
      applyPlaneTransform(
        pointerDampedRef.current.x,
        pointerDampedRef.current.y,
      );

      columnMeta.forEach((meta, columnIndex) => {
        const track = trackRefs.current[columnIndex];
        if (!track) return;

        const paused = wallHoveredRef.current && pauseOnHover;
        const moving = paused || hoveredColRef.current === columnIndex ? 0 : 1;
        const targetVelocity = baseVelocities[columnIndex] * moving;
        const ease =
          1 - Math.exp(-delta / (targetVelocity === 0 ? 0.16 : 0.28));
        velocitiesRef.current[columnIndex] +=
          (targetVelocity - velocitiesRef.current[columnIndex]) * ease;
        const next = wrapLoopOffset(
          (offsetsRef.current[columnIndex] ?? 0) +
            velocitiesRef.current[columnIndex] * delta,
          meta.copyHeight,
        );
        offsetsRef.current[columnIndex] = next;
        track.style.transform = `translate3d(0, ${-next}px, 0)`;
      });

      rafRef.current = window.requestAnimationFrame(animate);
    };

    rafRef.current = window.requestAnimationFrame(animate);
    return () => {
      if (rafRef.current !== null) {
        window.cancelAnimationFrame(rafRef.current);
      }
      rafRef.current = null;
      lastTsRef.current = null;
    };
  }, [
    applyPlaneTransform,
    baseVelocities,
    columnMeta,
    parallax,
    pauseOnHover,
    reduced,
  ]);

  const activate = useCallback((id: string, columnIndex: number) => {
    if (activeIdRef.current === id) return;
    activeIdRef.current = id;
    hoveredColRef.current = columnIndex;
    velocitiesRef.current[columnIndex] = 0;
    setActiveId(id);
  }, []);

  const release = useCallback(() => {
    if (activeIdRef.current === null) return;
    activeIdRef.current = null;
    hoveredColRef.current = -1;
    setActiveId(null);
  }, []);

  const handlePointerMove = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
      const rect = containerRef.current?.getBoundingClientRect();
      if (rect && parallax > 0 && !reduced) {
        pointerRef.current = {
          x: (event.clientX - rect.left) / rect.width - 0.5,
          y: (event.clientY - rect.top) / rect.height - 0.5,
        };
      }

      const target = event.target;
      const tile =
        target instanceof Element
          ? target.closest<HTMLElement>("[data-tile-id]")
          : null;
      if (!tile) return;
      activate(tile.dataset.tileId ?? "", Number(tile.dataset.col));
    },
    [activate, parallax, reduced],
  );

  const cssVariables = useMemo<DriftWallVariables>(
    () => ({
      "--dw-tile-w": `${tileWidth}px`,
      "--dw-tile-h": `${tileHeight}px`,
      "--dw-gap": `${gap}px`,
      "--dw-radius": `${radius}px`,
      "--dw-perspective": `${perspective}px`,
      "--dw-lift": `${lift}px`,
      "--dw-dim": dim,
      "--dw-gray": grayscale ? 1 : 0,
      "--dw-overlay": overlayColor,
      "--dw-edge": `${Math.max(0, (1 - fade) * 100)}%`,
      ...style,
    }),
    [
      dim,
      fade,
      gap,
      grayscale,
      lift,
      overlayColor,
      perspective,
      radius,
      style,
      tileHeight,
      tileWidth,
    ],
  );

  const rootClassName = [
    "drift-wall",
    reduced ? "drift-wall--reduced" : "",
    className,
  ]
    .filter(Boolean)
    .join(" ");

  if (items.length === 0) return null;

  return (
    <div
      ref={containerRef}
      className={rootClassName}
      style={cssVariables}
      onPointerMove={handlePointerMove}
      onPointerEnter={() => {
        wallHoveredRef.current = true;
      }}
      onPointerLeave={() => {
        wallHoveredRef.current = false;
        pointerRef.current = { x: 0, y: 0 };
        release();
      }}
      role="group"
      aria-label={ariaLabel}
    >
      <div
        ref={planeRef}
        className="drift-wall__plane"
        style={{ width: planeWidth, height: planeHeight }}
      >
        {columnItems.map((column, columnIndex) => {
          const meta = columnMeta[columnIndex];
          return (
            <div className="drift-wall__col" key={`column-${columnIndex}`}>
              <div
                className="drift-wall__track"
                ref={(element) => {
                  trackRefs.current[columnIndex] = element;
                }}
              >
                {Array.from({ length: meta.copies }).flatMap((_, copyIndex) =>
                  column.map((item, itemIndex) => {
                    const tileId = `${columnIndex}-${copyIndex}-${item.id}-${itemIndex}`;
                    const isPrimaryCopy = copyIndex === 0 && columnIndex === 0;
                    return (
                      <button
                        className={`drift-wall__tile${activeId === tileId ? " is-active" : ""}`}
                        type="button"
                        key={tileId}
                        data-tile-id={tileId}
                        data-col={columnIndex}
                        aria-label={`查看 ${item.title}`}
                        aria-hidden={!isPrimaryCopy}
                        tabIndex={isPrimaryCopy ? 0 : -1}
                        onFocus={() => activate(tileId, columnIndex)}
                        onBlur={release}
                        onPointerDown={(event) => {
                          if (event.button === 0) onItemSelect?.(item);
                        }}
                        onClick={() => onItemSelect?.(item)}
                      >
                        <span className="drift-wall__inner">
                          <PlaceImage
                            src={item.image}
                            alt=""
                            draggable={false}
                          />
                          <span
                            className="drift-wall__overlay"
                            aria-hidden="true"
                          />
                          <span className="drift-wall__caption">
                            <strong>{item.title}</strong>
                          </span>
                        </span>
                      </button>
                    );
                  }),
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
