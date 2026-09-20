import { useCallback, useEffect, useRef, useState } from "react";

import type {
  AttractionAccordionItem,
  RecommendationIntent,
  RecommendationIntentOption
} from "./AttractionAccordionAttachment";
import { ATTRACTION_INTENT_OPTIONS } from "./recommendationIntents";

type AttractionFeedbackPanelProps = {
  item: AttractionAccordionItem;
  value?: RecommendationIntent;
  intentOptions?: readonly RecommendationIntentOption[];
  disabled?: boolean;
  confirmDisabled?: boolean;
  onChange: (intent: RecommendationIntent) => void;
  onConfirm: () => void;
  onNext?: () => void;
  nextItemName?: string;
  autoAdvance?: boolean;
};

export function AttractionFeedbackPanel({
  item,
  value,
  intentOptions = ATTRACTION_INTENT_OPTIONS,
  disabled = false,
  confirmDisabled = false,
  onChange,
  onConfirm,
  onNext,
  nextItemName,
  autoAdvance = true
}: AttractionFeedbackPanelProps) {
  const advanceTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const cancelAdvance = useCallback(() => {
    if (advanceTimer.current !== null) {
      clearTimeout(advanceTimer.current);
      advanceTimer.current = null;
    }
  }, []);

  // Manual navigation, disabled cards and unmounts must not retain a pending jump.
  useEffect(
    () => cancelAdvance,
    [item.id, disabled, autoAdvance, cancelAdvance]
  );

  const intentRowRef = useRef<HTMLDivElement>(null);
  const [ratedItemIds, setRatedItemIds] = useState<Set<string>>(
    () => new Set()
  );
  const showQuickNext = Boolean(
    !autoAdvance && onNext && (item.compact ? value : ratedItemIds.has(item.id))
  );

  const handleIntentChange = (intent: RecommendationIntent) => {
    if (disabled) return;
    if (!autoAdvance && !item.compact) {
      setRatedItemIds((current) => {
        const next = new Set(current);
        next.add(item.id);
        return next;
      });
    }
    onChange(intent);
    cancelAdvance();
    if (autoAdvance && onNext) {
      advanceTimer.current = setTimeout(() => {
        advanceTimer.current = null;
        onNext();
      }, 300);
    }
  };

  const handleNext = () => {
    onNext?.();
    intentRowRef.current
      ?.querySelector<HTMLButtonElement>(
        ".attraction-intent-button.is-selected"
      )
      ?.focus({ preventScroll: true });
  };

  return (
    <>
      <div className="attraction-accordion-detail">
        <div className="attraction-accordion-copy">
          {!item.compact ? <strong>{item.name}</strong> : null}
          {item.experience ? <p>{item.experience}</p> : null}
          {!item.compact &&
            (item.duration || item.tradeoff ? (
              <span>
                {[item.duration, item.tradeoff].filter(Boolean).join(" · ")}
              </span>
            ) : (
              <span>更多信息暂未提供</span>
            ))}
        </div>
        {!item.compact ? (
          <details
            className={`attraction-source-disclosure${
              item.sourceLabel ? "" : " is-unavailable"
            }`}
          >
            <summary>来源</summary>
            <p>
              {item.sourceLabel
                ? [item.sourceLabel, item.updatedAt].filter(Boolean).join(" · ")
                : "来源暂时不可用"}
            </p>
          </details>
        ) : null}
      </div>

      <div
        ref={intentRowRef}
        className={`attraction-intent-row${autoAdvance ? " is-auto-advance" : ""}`}
        role="group"
        aria-label={`${item.name}的意愿`}
      >
        <div className="attraction-intent-options">
          {intentOptions.map((option) => {
            const selected = value === option.value;
            return (
              <span
                key={option.value}
                className={`attraction-intent-option${
                  selected && showQuickNext ? " has-quick-next" : ""
                }`}
              >
                <button
                  type="button"
                  className={`attraction-intent-button${autoAdvance ? " v4-preference-choice" : ""}${
                    selected ? " is-selected" : ""
                  }`}
                  aria-pressed={selected}
                  disabled={disabled}
                  onClick={() => handleIntentChange(option.value)}
                >
                  {option.label}
                </button>
                {selected && showQuickNext ? (
                  <button
                    type="button"
                    className="attraction-quick-next-button"
                    aria-label={
                      nextItemName ? `下一张：${nextItemName}` : "下一张"
                    }
                    title="下一张"
                    disabled={disabled}
                    onClick={handleNext}
                  >
                    <svg viewBox="0 0 24 24" aria-hidden="true">
                      <path d="M5 12h13M13 6l6 6-6 6" />
                    </svg>
                  </button>
                ) : null}
              </span>
            );
          })}
        </div>
        <button
          type="button"
          className={`attraction-confirm-button${autoAdvance ? " v4-card-primary-action" : ""}`}
          disabled={disabled || confirmDisabled}
          onClick={() => {
            cancelAdvance();
            onConfirm();
          }}
        >
          {autoAdvance ? "确认选择" : "选好了"}
        </button>
      </div>
    </>
  );
}
