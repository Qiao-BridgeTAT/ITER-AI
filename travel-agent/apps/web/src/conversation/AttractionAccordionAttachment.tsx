import {
  CSSProperties,
  KeyboardEvent,
  PointerEvent,
  useEffect,
  useLayoutEffect,
  useRef,
  useState
} from "react";
import { gsap } from "gsap";

import { AttractionFeedbackPanel } from "./AttractionFeedbackPanel";
import { RecommendationImagePlaceholder } from "./RecommendationImagePlaceholder";
import { PlaceImage } from "./PlaceImage";

export type AttractionIntent = "must" | "want" | "if_convenient" | "avoid";

export type RecommendationIntent =
  AttractionIntent | "interested" | "dedicated_trip" | "consider" | "favorite";

export type RecommendationIntentOption = {
  value: RecommendationIntent;
  label: string;
};

export type AttractionAccordionItem = {
  id: string;
  name: string;
  image?: string;
  alt?: string;
  reason: string;
  badgeTone?: "city" | "personalized";
  experience: string;
  duration?: string;
  tradeoff?: string;
  sourceLabel?: string;
  updatedAt?: string;
  compact?: boolean;
};

type AttractionAccordionAttachmentProps = {
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

const REDUCED_MOTION_QUERY = "(prefers-reduced-motion: reduce)";

export function AttractionAccordionAttachment({
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
}: AttractionAccordionAttachmentProps) {
  const defaultIndex = Math.floor(items.length / 2);
  const [activeIndex, setActiveIndex] = useState(defaultIndex);
  const activeItemId = items[activeIndex]?.id;
  useEffect(() => {
    if (activeItemId) onFocusItem?.(activeItemId);
  }, [activeItemId, onFocusItem]);
  const rootRef = useRef<HTMLDivElement>(null);
  const panelRefs = useRef<Array<HTMLButtonElement | null>>([]);
  const imageRefs = useRef<Array<HTMLImageElement | null>>([]);
  const dimRefs = useRef<Array<HTMLSpanElement | null>>([]);
  const tiltControllers = useRef<
    Array<{
      rotationX: (value: number) => void;
      rotationY: (value: number) => void;
    }>
  >([]);
  const pointerFrame = useRef<number | null>(null);
  const latestPointer = useRef<{
    panel: HTMLButtonElement;
    index: number;
    clientX: number;
    clientY: number;
  } | null>(null);
  const firstAnimation = useRef(true);
  const activeItem = items[activeIndex] ?? items[0];

  useLayoutEffect(() => {
    const prefersReducedMotion =
      window.matchMedia?.(REDUCED_MOTION_QUERY).matches;
    const duration =
      firstAnimation.current || prefersReducedMotion
        ? 0
        : activeItem?.compact
          ? 0.24
          : 0.46;
    const panels = panelRefs.current.filter(
      (panel): panel is HTMLButtonElement => panel !== null
    );
    const images = imageRefs.current.filter(
      (image): image is HTMLImageElement => image !== null
    );
    const dims = dimRefs.current.filter(
      (dim): dim is HTMLSpanElement => dim !== null
    );

    gsap.killTweensOf([...panels, ...images, ...dims]);

    const timeline = gsap.timeline({
      defaults: {
        duration,
        ease: "power4.out",
        overwrite: "auto"
      }
    });

    panels.forEach((panel, index) => {
      timeline.to(panel, { flexGrow: index === activeIndex ? 5.2 : 1 }, 0);
    });
    imageRefs.current.forEach((image, index) => {
      if (!image) {
        return;
      }
      timeline.to(
        image,
        {
          xPercent: index === activeIndex ? 0 : index < activeIndex ? -4 : 4,
          scale: index === activeIndex ? 1.035 : 1.085
        },
        0
      );
    });
    dims.forEach((dim, index) => {
      timeline.to(dim, { opacity: index === activeIndex ? 0 : 0.5 }, 0);
    });

    firstAnimation.current = false;
    return () => {
      timeline.kill();
    };
  }, [activeIndex, activeItem?.compact]);

  useEffect(() => {
    const panels = panelRefs.current.filter(
      (panel): panel is HTMLButtonElement => panel !== null
    );
    tiltControllers.current = panelRefs.current.map((panel) => {
      if (!panel) {
        return { rotationX: () => undefined, rotationY: () => undefined };
      }
      return {
        rotationX: gsap.quickTo(panel, "rotationX", {
          duration: 0.24,
          ease: "power3.out"
        }),
        rotationY: gsap.quickTo(panel, "rotationY", {
          duration: 0.24,
          ease: "power3.out"
        })
      };
    });

    return () => {
      if (pointerFrame.current !== null) {
        window.cancelAnimationFrame(pointerFrame.current);
      }
      gsap.killTweensOf(panels);
    };
  }, []);

  const navigateBy = (step: number, focusGalleryPanel = false) => {
    const nextIndex = (activeIndex + step + items.length) % items.length;
    setActiveIndex(nextIndex);
    if (focusGalleryPanel) {
      panelRefs.current[nextIndex]?.focus({ preventScroll: true });
    }
  };

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
    navigateBy(
      event.key === "ArrowRight" ? 1 : -1,
      Boolean(target.closest(".attraction-accordion-gallery"))
    );
  };

  const handlePointerMove = (
    event: PointerEvent<HTMLButtonElement>,
    index: number
  ) => {
    if (
      index !== activeIndex ||
      window.matchMedia?.(REDUCED_MOTION_QUERY).matches
    ) {
      return;
    }
    latestPointer.current = {
      panel: event.currentTarget,
      index,
      clientX: event.clientX,
      clientY: event.clientY
    };
    if (pointerFrame.current !== null) {
      return;
    }

    pointerFrame.current = window.requestAnimationFrame(() => {
      pointerFrame.current = null;
      const pointer = latestPointer.current;
      if (!pointer) {
        return;
      }
      const rect = pointer.panel.getBoundingClientRect();
      const x = (pointer.clientX - rect.left) / rect.width - 0.5;
      const y = (pointer.clientY - rect.top) / rect.height - 0.5;
      tiltControllers.current[pointer.index]?.rotationY(x * 2.4);
      tiltControllers.current[pointer.index]?.rotationX(y * -2);
    });
  };

  const resetTilt = (index: number) => {
    const panel = panelRefs.current[index];
    if (!panel) {
      return;
    }
    latestPointer.current = null;
    tiltControllers.current[index]?.rotationX(0);
    tiltControllers.current[index]?.rotationY(0);
  };

  if (!activeItem) {
    return null;
  }

  return (
    <section
      ref={rootRef}
      className="attraction-accordion-attachment"
      aria-label={label}
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
          {activeIndex + 1} / {items.length}
        </span>
        {activeItem.compact ? (
          <div className="attraction-compact-navigation">
            <button
              type="button"
              aria-label={`上一个${itemNoun}`}
              disabled={disabled || items.length < 2}
              onClick={() => navigateBy(-1)}
            >
              <span aria-hidden="true">‹</span>
            </button>
            <button
              type="button"
              aria-label={`下一个${itemNoun}`}
              disabled={disabled || items.length < 2}
              onClick={() => navigateBy(1)}
            >
              <span aria-hidden="true">›</span>
            </button>
          </div>
        ) : null}
      </div>

      <div
        className="attraction-accordion-gallery"
        aria-label={`${itemNoun}图片浏览`}
      >
        {items.map((item, index) => (
          <button
            key={item.id}
            ref={(node) => {
              panelRefs.current[index] = node;
            }}
            className={`attraction-accordion-panel${
              index === activeIndex ? " is-active" : ""
            }`}
            style={{ "--panel-order": index } as CSSProperties}
            type="button"
            aria-label={`查看 ${item.name}`}
            aria-current={index === activeIndex ? "true" : undefined}
            disabled={disabled}
            onClick={() => setActiveIndex(index)}
            onFocus={() => !disabled && setActiveIndex(index)}
            onMouseEnter={() => !disabled && setActiveIndex(index)}
            onPointerMove={(event) => handlePointerMove(event, index)}
            onPointerLeave={() => resetTilt(index)}
          >
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
                dimRefs.current[index] = node;
              }}
              className="attraction-accordion-dim"
              aria-hidden="true"
            />
            <span className="attraction-accordion-shade" aria-hidden="true" />
            <span className="attraction-accordion-panel-label">
              <strong>{item.name}</strong>
              <small data-badge-tone={item.badgeTone}>{item.reason}</small>
            </span>
          </button>
        ))}
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
        nextItemName={items[(activeIndex + 1) % items.length]?.name}
      />
    </section>
  );
}
