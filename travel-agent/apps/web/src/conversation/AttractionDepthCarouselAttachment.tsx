import {
  CSSProperties,
  KeyboardEvent,
  PointerEvent,
  WheelEvent,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState
} from "react";
import { gsap } from "gsap";

import type {
  AttractionAccordionItem,
  RecommendationIntent,
  RecommendationIntentOption
} from "./AttractionAccordionAttachment";
import { AttractionFeedbackPanel } from "./AttractionFeedbackPanel";
import { RecommendationImagePlaceholder } from "./RecommendationImagePlaceholder";
import { PlaceImage } from "./PlaceImage";

type AttractionDepthCarouselAttachmentProps = {
  label: string;
  items: readonly AttractionAccordionItem[];
  values: Record<string, RecommendationIntent>;
  itemNoun?: string;
  intentOptions?: readonly RecommendationIntentOption[];
  disabled?: boolean;
  confirmDisabled?: boolean;
  onChange: (itemId: string, intent: RecommendationIntent) => void;
  onConfirm: () => void;
  onFocusItem?: (id: string) => void;
};

type CarouselConfig = {
  count: number;
  cardWidth: number;
  cardHeight: number;
  depth: number;
  spread: number;
  tilt: number;
  visibleCards: number;
  falloff: number;
  blur: number;
  duration: number;
  loop: boolean;
};

type DragState = {
  x: number;
  startPosition: number;
  lastX: number;
  lastTime: number;
  velocity: number;
  moved: boolean;
  pointerId: number;
};

const REDUCED_MOTION_QUERY = "(prefers-reduced-motion: reduce)";

function clamp(value: number, minimum: number, maximum: number) {
  return Math.min(Math.max(value, minimum), maximum);
}

export function AttractionDepthCarouselAttachment({
  label,
  items,
  values,
  itemNoun = "景点",
  intentOptions,
  disabled = false,
  confirmDisabled = false,
  onChange,
  onConfirm,
  onFocusItem
}: AttractionDepthCarouselAttachmentProps) {
  const data = useMemo(() => [...items], [items]);
  const [activeIndex, setActiveIndex] = useState(0);
  const activeItemId = data[activeIndex]?.id;
  useEffect(() => {
    if (activeItemId) onFocusItem?.(activeItemId);
  }, [activeItemId, onFocusItem]);
  const rootRef = useRef<HTMLDivElement>(null);
  const cardRefs = useRef<Array<HTMLButtonElement | null>>([]);
  const imageRefs = useRef<Array<HTMLImageElement | null>>([]);
  const tintRefs = useRef<Array<HTMLSpanElement | null>>([]);
  const positionRef = useRef(0);
  const focusRef = useRef(0);
  const scaleRef = useRef(1);
  const tweenRef = useRef<gsap.core.Tween | null>(null);
  const dragRef = useRef<DragState | null>(null);
  const suppressClickRef = useRef(false);
  const wheelTimerRef = useRef<number | null>(null);
  const reducedMotionRef = useRef(false);
  const configRef = useRef<CarouselConfig>({
    count: data.length,
    cardWidth: 300,
    cardHeight: 320,
    depth: 220,
    spread: 98,
    tilt: 22,
    visibleCards: 5,
    falloff: 0.2,
    blur: 6,
    duration: 700,
    loop: true
  });
  const activeItem = data[activeIndex] ?? data[0];

  configRef.current.count = data.length;
  configRef.current.duration = activeItem?.compact ? 260 : 700;

  const layout = useCallback((position: number) => {
    const config = configRef.current;
    if (!config.count) {
      return;
    }

    cardRefs.current.forEach((card, index) => {
      if (!card) {
        return;
      }

      let distance = index - position;
      if (config.loop && config.count > 1) {
        distance = ((distance % config.count) + config.count) % config.count;
        if (distance > config.count / 2) {
          distance -= config.count;
        }
      }

      const depthPosition = Math.max(0, distance);
      const absoluteDistance = Math.abs(distance);
      const visible = absoluteDistance <= config.visibleCards + 0.5;
      const translateZ = -config.depth * distance;
      const translateX = config.spread * distance;
      const translateY = depthPosition * 2.5;
      const rotateY = config.tilt * clamp(distance, 0, 1);
      const depthScale = Math.max(0.68, 1 - depthPosition * 0.075);
      const cardScale = scaleRef.current * depthScale;
      let opacity = distance < 0 ? Math.max(0, 1 + distance) : 1;
      opacity *= Math.max(0.48, 1 - depthPosition * 0.075);
      if (!visible) {
        opacity = 0;
      }

      const brightness = Math.max(0.18, 1 - depthPosition * config.falloff);
      const blur = Math.min(
        config.blur,
        (depthPosition / Math.max(1, config.visibleCards)) * config.blur
      );

      card.style.transform = `translate(-50%, -50%) translateY(${translateY.toFixed(2)}px) scale(${cardScale.toFixed(4)}) translateX(${translateX.toFixed(2)}px) translateZ(${translateZ.toFixed(2)}px) rotateY(${rotateY.toFixed(3)}deg)`;
      card.style.opacity = opacity.toFixed(3);
      card.style.zIndex = String(Math.round(2000 - distance * 20));
      card.style.pointerEvents = visible && opacity > 0.05 ? "auto" : "none";

      const image = imageRefs.current[index];
      if (image) {
        image.style.filter = `brightness(${brightness.toFixed(3)}) blur(${blur.toFixed(2)}px)`;
        image.style.transform = `scale(${(1.02 + blur * 0.004).toFixed(4)})`;
      }

      const tint = tintRefs.current[index];
      if (tint) {
        tint.style.opacity = clamp(
          depthPosition * config.falloff * 1.25,
          0,
          0.86
        ).toFixed(3);
      }
    });
  }, []);

  const tweenTo = useCallback(
    (target: number, animate: boolean) => {
      tweenRef.current?.kill();
      const proxy = { position: positionRef.current };
      const duration =
        animate && !reducedMotionRef.current
          ? configRef.current.duration / 1000
          : 0;
      tweenRef.current = gsap.to(proxy, {
        position: target,
        duration,
        ease: "power3.out",
        overwrite: "auto",
        onUpdate: () => {
          positionRef.current = proxy.position;
          layout(proxy.position);
        },
        onComplete: () => {
          const count = configRef.current.count;
          if (count > 0) {
            positionRef.current =
              ((positionRef.current % count) + count) % count;
          }
          layout(positionRef.current);
        }
      });
    },
    [layout]
  );

  const setFocus = useCallback(
    (rawIndex: number, animate = true) => {
      const config = configRef.current;
      if (!config.count) {
        return;
      }
      const index = config.loop
        ? ((rawIndex % config.count) + config.count) % config.count
        : clamp(rawIndex, 0, config.count - 1);
      let delta = index - positionRef.current;
      if (config.loop && config.count > 1) {
        delta = ((delta % config.count) + config.count) % config.count;
        if (delta > config.count / 2) {
          delta -= config.count;
        }
      }
      tweenTo(positionRef.current + delta, animate);
      focusRef.current = index;
      setActiveIndex(index);
    },
    [tweenTo]
  );

  const navigateBy = useCallback(
    (step: number) => setFocus(focusRef.current + step, true),
    [setFocus]
  );

  const handleAttachmentKeyDown = (event: KeyboardEvent<HTMLElement>) => {
    if (disabled || (event.key !== "ArrowLeft" && event.key !== "ArrowRight")) {
      return;
    }
    const target = event.target as HTMLElement;
    if (target.closest("input, textarea, select, [contenteditable='true']")) {
      return;
    }
    event.preventDefault();
    event.stopPropagation();
    navigateBy(event.key === "ArrowRight" ? 1 : -1);
  };

  useEffect(() => {
    reducedMotionRef.current =
      window.matchMedia?.(REDUCED_MOTION_QUERY).matches ?? false;
    const root = rootRef.current;
    if (!root) {
      return;
    }

    const updateScale = (width: number, height: number) => {
      const config = configRef.current;
      const neededWidth = config.cardWidth + Math.abs(config.spread) * 2 + 120;
      const widthScale = width / neededWidth;
      const heightScale = (height - 40) / config.cardHeight;
      scaleRef.current = clamp(Math.min(widthScale, heightScale), 0.55, 1);
      layout(positionRef.current);
    };

    const initialRect = root.getBoundingClientRect();
    updateScale(initialRect.width || 840, initialRect.height || 360);
    if (typeof ResizeObserver === "undefined") {
      return;
    }
    const observer = new ResizeObserver((entries) => {
      updateScale(
        entries[0]?.contentRect.width ?? 840,
        entries[0]?.contentRect.height ?? 360
      );
    });
    observer.observe(root);
    return () => observer.disconnect();
  }, [layout]);

  useEffect(() => {
    layout(positionRef.current);
  }, [data.length, layout]);

  useEffect(
    () => () => {
      tweenRef.current?.kill();
      if (wheelTimerRef.current !== null) {
        window.clearTimeout(wheelTimerRef.current);
      }
    },
    []
  );

  const handleWheel = (event: WheelEvent<HTMLDivElement>) => {
    if (
      disabled ||
      data.length < 2 ||
      (!event.shiftKey && Math.abs(event.deltaX) <= Math.abs(event.deltaY))
    ) {
      return;
    }
    event.preventDefault();
    const delta = event.deltaX || event.deltaY;
    const step = clamp(delta / 250, -0.7, 0.7);
    positionRef.current += step;
    layout(positionRef.current);
    if (wheelTimerRef.current !== null) {
      window.clearTimeout(wheelTimerRef.current);
    }
    wheelTimerRef.current = window.setTimeout(() => {
      setFocus(Math.round(positionRef.current), true);
    }, 120);
  };

  const handlePointerDown = (event: PointerEvent<HTMLDivElement>) => {
    if (disabled || data.length < 2) {
      return;
    }
    tweenRef.current?.kill();
    dragRef.current = {
      x: event.clientX,
      startPosition: positionRef.current,
      lastX: event.clientX,
      lastTime: performance.now(),
      velocity: 0,
      moved: false,
      pointerId: event.pointerId
    };
  };

  const handlePointerMove = (event: PointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current;
    if (!drag) {
      return;
    }
    const stepPixels = Math.max(
      configRef.current.cardWidth * 0.55 * scaleRef.current,
      40
    );
    const deltaX = event.clientX - drag.x;
    if (!drag.moved && Math.abs(deltaX) > 4) {
      drag.moved = true;
      event.currentTarget.setPointerCapture(drag.pointerId);
    }
    if (!drag.moved) {
      return;
    }
    const now = performance.now();
    const deltaTime = Math.max(now - drag.lastTime, 1);
    drag.velocity = (event.clientX - drag.lastX) / deltaTime;
    drag.lastX = event.clientX;
    drag.lastTime = now;
    positionRef.current = drag.startPosition - deltaX / stepPixels;
    layout(positionRef.current);
  };

  const handlePointerEnd = () => {
    const drag = dragRef.current;
    if (!drag) {
      return;
    }
    dragRef.current = null;
    if (!drag.moved) {
      return;
    }
    suppressClickRef.current = true;
    window.setTimeout(() => {
      suppressClickRef.current = false;
    }, 0);
    const stepPixels = Math.max(
      configRef.current.cardWidth * 0.55 * scaleRef.current,
      40
    );
    const projected = positionRef.current - (drag.velocity * 160) / stepPixels;
    setFocus(Math.round(projected), true);
  };

  if (!activeItem) {
    return null;
  }

  return (
    <section
      className="attraction-depth-attachment"
      aria-label={label}
      style={{ "--depth-card-count": data.length } as CSSProperties}
      onKeyDownCapture={handleAttachmentKeyDown}
    >
      <div className="attraction-accordion-heading">
        <span className={activeItem.compact ? "visually-hidden" : undefined}>
          {label}
        </span>
        <span
          className={activeItem.compact ? "visually-hidden" : undefined}
          aria-live="polite"
        >
          {activeIndex + 1} / {data.length}
        </span>
      </div>

      <div
        ref={rootRef}
        className="attraction-depth-carousel"
        role="group"
        aria-roledescription="carousel"
        aria-label={`${label}，${data.length} 个候选`}
        tabIndex={0}
        onWheel={handleWheel}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerEnd}
        onPointerCancel={handlePointerEnd}
      >
        <div className="attraction-depth-stage">
          {data.map((item, index) => (
            <button
              key={item.id}
              ref={(node) => {
                cardRefs.current[index] = node;
              }}
              type="button"
              className={`attraction-depth-card${
                activeIndex === index ? " is-active" : ""
              }`}
              aria-label={`查看 ${item.name}`}
              aria-current={activeIndex === index ? "true" : undefined}
              disabled={disabled}
              onClick={() => {
                if (!suppressClickRef.current) {
                  setFocus(index, true);
                }
              }}
              onMouseEnter={() => {
                if (
                  item.compact &&
                  !disabled &&
                  !dragRef.current &&
                  !tweenRef.current?.isActive()
                ) {
                  setFocus(index, true);
                }
              }}
              onFocus={() => !disabled && setFocus(index, true)}
            >
              <span className="attraction-depth-card-clip">
                <RecommendationImagePlaceholder />
                {item.image ? (
                  <PlaceImage
                    ref={(node) => {
                      imageRefs.current[index] = node;
                    }}
                    src={item.image}
                    alt={item.alt ?? ""}
                    loading="eager"
                    draggable={false}
                  />
                ) : null}
                <span
                  ref={(node) => {
                    tintRefs.current[index] = node;
                  }}
                  className="attraction-depth-tint"
                  aria-hidden="true"
                />
                <span className="attraction-depth-card-label">
                  <strong>{item.name}</strong>
                  <small data-badge-tone={item.badgeTone}>{item.reason}</small>
                </span>
              </span>
            </button>
          ))}
        </div>

        <button
          type="button"
          className="attraction-depth-arrow attraction-depth-arrow-previous"
          aria-label={`上一个${itemNoun}`}
          disabled={disabled}
          onClick={() => navigateBy(-1)}
        >
          <span aria-hidden="true">‹</span>
        </button>
        <button
          type="button"
          className="attraction-depth-arrow attraction-depth-arrow-next"
          aria-label={`下一个${itemNoun}`}
          disabled={disabled}
          onClick={() => navigateBy(1)}
        >
          <span aria-hidden="true">›</span>
        </button>

        <div
          className="attraction-depth-indicators"
          role="tablist"
          aria-label={`${itemNoun}位置`}
        >
          {data.map((item, index) => (
            <button
              key={item.id}
              type="button"
              role="tab"
              className={activeIndex === index ? "is-active" : undefined}
              aria-label={`转到 ${item.name}`}
              aria-selected={activeIndex === index}
              disabled={disabled}
              onClick={() => setFocus(index, true)}
            />
          ))}
        </div>
      </div>

      <AttractionFeedbackPanel
        item={activeItem}
        value={values[activeItem.id]}
        intentOptions={intentOptions}
        disabled={disabled}
        confirmDisabled={confirmDisabled}
        onChange={(intent) => onChange(activeItem.id, intent)}
        onConfirm={onConfirm}
        autoAdvance={itemNoun === "景点"}
        onNext={() => navigateBy(1)}
        nextItemName={data[(activeIndex + 1) % data.length]?.name}
      />
    </section>
  );
}
