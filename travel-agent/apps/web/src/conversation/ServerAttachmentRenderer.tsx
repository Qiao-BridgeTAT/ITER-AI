import { useEffect, useMemo, useState } from "react";

import type {
  AttachmentAnswerPayload,
  AttachmentAnswerRecord,
  ConversationAttachment as ServerConversationAttachment,
} from "../generated/contracts";
import { CityThemeFlow } from "../interests/CityThemeFlow";
import {
  AttractionAccordionAttachment,
  type AttractionAccordionItem,
  type RecommendationIntent,
} from "./AttractionAccordionAttachment";
import { AttractionDepthCarouselAttachment } from "./AttractionDepthCarouselAttachment";
import {
  CompactChoiceAttachment,
  CompactMultiChoiceAttachment,
} from "./CompactChoiceAttachment";
import { CompletedAttachmentSummary } from "./CompletedAttachmentSummary";
import { DetailedChoiceAttachment } from "./DetailedChoiceAttachment";
import { PreferenceSliderAttachment } from "./PreferenceSliderAttachment";
import { recommendationGalleryMode } from "./recommendationGalleryMode";
import {
  ATTRACTION_INTENT_OPTIONS,
  RESTAURANT_INTENT_OPTIONS,
} from "./recommendationIntents";
import { TextMultiChoiceAttachment } from "./TextMultiChoiceAttachment";
import { WeatherDeck } from "./WeatherDeck";
import type { WeatherCondition, WeatherDayData } from "./weatherTypes";

export type AttachmentAnswer = AttachmentAnswerPayload["answer"];

type ServerAttachmentRendererProps = {
  attachment: ServerConversationAttachment | unknown;
  confirmedAnswer?: AttachmentAnswerRecord;
  disabled?: boolean;
  conflictMessage?: string;
  onSubmit: (answer: AttachmentAnswer) => void;
  onReload?: () => void;
  onOpenTaskBook?: (taskBookId: string) => void;
};

export function ServerAttachmentRenderer({
  attachment: candidate,
  confirmedAnswer,
  disabled = false,
  conflictMessage,
  onSubmit,
  onReload,
  onOpenTaskBook,
}: ServerAttachmentRendererProps) {
  const attachment = isKnownAttachment(candidate) ? candidate : null;
  const [editing, setEditing] = useState(false);
  const [selectedId, setSelectedId] = useState<string>();
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [sliderValue, setSliderValue] = useState(0);
  const [sliderFlexible, setSliderFlexible] = useState(false);
  const [recommendationValues, setRecommendationValues] = useState<
    Record<string, RecommendationIntent>
  >({});

  useEffect(() => {
    if (!attachment) return;
    const answer = confirmedAnswer?.answer;
    if (answer?.answer_type === "single_choice")
      setSelectedId(answer.option_id);
    if (answer?.answer_type === "multi_choice")
      setSelectedIds(answer.option_ids);
    if (answer?.answer_type === "slider") {
      setSliderFlexible(answer.value === null);
      setSliderValue(
        answer.value ??
          ("minimum_value" in attachment
            ? (attachment.minimum_value + attachment.maximum_value) / 2
            : 0),
      );
    }
    if (answer?.answer_type === "recommendation_feedback") {
      setRecommendationValues(
        Object.fromEntries(
          answer.feedback.map((item) => [item.recommendation_id, item.intent]),
        ),
      );
    }
    if (confirmedAnswer) setEditing(false);
  }, [attachment, confirmedAnswer]);

  const summary = useMemo(
    () =>
      attachment && confirmedAnswer
        ? answerSummary(attachment, confirmedAnswer)
        : null,
    [attachment, confirmedAnswer],
  );

  if (!attachment) {
    return (
      <div className="recommendation-empty-attachment" role="status">
        这项互动暂时无法显示，你仍可以继续对话。
      </div>
    );
  }
  if (attachment.availability === "missing") {
    return (
      <div className="recommendation-empty-attachment" role="status">
        {attachment.missing_reason ?? "这项内容暂时不可用。"}
      </div>
    );
  }
  if (attachment.status === "superseded") {
    return (
      <div className="recommendation-empty-attachment" role="status">
        这项互动已被后续内容替代。
      </div>
    );
  }
  if (confirmedAnswer && !editing && summary) {
    return (
      <CompletedAttachmentSummary
        label={attachment.prompt}
        value={summary}
        disabled={disabled || attachment.editable === false}
        onEdit={() => setEditing(true)}
      />
    );
  }

  const feedback = conflictMessage ? (
    <div className="attachment-submit-conflict" role="alert">
      <span>{conflictMessage}</span>
      {onReload ? (
        <button type="button" onClick={onReload}>
          刷新后保留选择
        </button>
      ) : null}
    </div>
  ) : null;

  if (attachment.kind === "compact_choice") {
    return (
      <div>
        <CompactChoiceAttachment
          label={attachment.prompt}
          name={`server-choice-${attachment.attachment_id}`}
          options={attachment.options.map(toChoiceOption)}
          selectedId={selectedId}
          disabled={disabled}
          onSelect={(option) => {
            if (disabled) return;
            setSelectedId(option.id);
            onSubmit({ answer_type: "single_choice", option_id: option.id });
          }}
        />
        {feedback}
      </div>
    );
  }
  if (attachment.kind === "detailed_choice") {
    return (
      <div>
        <DetailedChoiceAttachment
          label={attachment.prompt}
          name={`server-detailed-${attachment.attachment_id}`}
          options={attachment.options.map((option) => ({
            ...toChoiceOption(option),
            meta: option.description ?? "按这次旅行采用",
          }))}
          selectedId={selectedId}
          disabled={disabled}
          onSelect={(option) => {
            if (disabled) return;
            setSelectedId(option.id);
            onSubmit({ answer_type: "single_choice", option_id: option.id });
          }}
        />
        {feedback}
      </div>
    );
  }
  if (
    attachment.kind === "compact_multi" ||
    attachment.kind === "text_multi_choice"
  ) {
    if (
      attachment.kind === "text_multi_choice" &&
      attachment.interaction_domain === "city_theme"
    ) {
      const answer = confirmedAnswer?.answer;
      const initialSelection =
        answer?.answer_type === "multi_choice"
          ? {
              mode: answer.option_ids.includes(
                attachment.exclusive_option_id ?? "",
              )
                ? ("open_to_any" as const)
                : ("selected" as const),
              selected_theme_ids: answer.option_ids.filter(
                (id) => id !== attachment.exclusive_option_id,
              ),
              ...(answer.free_text ? { free_text: answer.free_text } : {}),
            }
          : undefined;
      return (
        <div>
          <CityThemeFlow
            cityName={attachment.context_label ?? "这座城市"}
            themes={attachment.options
              .filter(
                (option) => option.option_id !== attachment.exclusive_option_id,
              )
              .map((option) => ({
                theme_id: option.option_id,
                label: option.label,
                summary: option.description ?? option.label,
                source_ids: option.semantic_value?.source_fact_ids ?? [],
              }))}
            initialSelection={initialSelection}
            disabled={disabled}
            onSubmit={(selection) => {
              onSubmit({
                answer_type: "multi_choice",
                option_ids:
                  selection.mode === "open_to_any"
                    ? [
                        attachment.exclusive_option_id ??
                          "city-theme:open-to-any",
                      ]
                    : (selection.selected_theme_ids ?? []),
                ...(selection.free_text
                  ? { free_text: selection.free_text }
                  : {}),
              });
              return true;
            }}
          />
          {feedback}
        </div>
      );
    }
    const toggle = (id: string) => {
      if (disabled) return;
      setSelectedIds((current) => toggleSelection(current, id, attachment));
    };
    const confirm = () => {
      if (disabled) return;
      onSubmit({ answer_type: "multi_choice", option_ids: selectedIds });
    };
    return (
      <div>
        {attachment.kind === "compact_multi" ? (
          <CompactMultiChoiceAttachment
            label={attachment.prompt}
            options={attachment.options.map(toChoiceOption)}
            selectedIds={selectedIds}
            disabled={disabled}
            onToggle={(option) => toggle(option.id)}
            onConfirm={confirm}
          />
        ) : (
          <TextMultiChoiceAttachment
            label={attachment.prompt}
            options={attachment.options.map(toChoiceOption)}
            selectedIds={selectedIds}
            exclusiveOptionId={attachment.exclusive_option_id ?? undefined}
            disabled={disabled}
            onToggle={(option) => toggle(option.id)}
            onConfirm={confirm}
          />
        )}
        {feedback}
      </div>
    );
  }
  if (attachment.kind === "preference_slider") {
    const effectiveValue =
      sliderValue || (attachment.minimum_value + attachment.maximum_value) / 2;
    return (
      <div>
        <PreferenceSliderAttachment
          label={attachment.prompt}
          startLabel={attachment.minimum_label}
          endLabel={attachment.maximum_label}
          value={effectiveValue}
          valueText={String(effectiveValue)}
          flexibleSelected={sliderFlexible}
          disabled={disabled}
          onChange={(value) => {
            if (disabled) return;
            setSliderFlexible(false);
            setSliderValue(value);
          }}
          onFlexible={() => {
            if (disabled) return;
            setSliderFlexible(true);
            onSubmit({ answer_type: "slider", value: null });
          }}
          onConfirm={() => {
            if (disabled) return;
            onSubmit({ answer_type: "slider", value: effectiveValue });
          }}
        />
        {feedback}
      </div>
    );
  }
  if (attachment.kind === "recommendation_set") {
    const isRestaurant = attachment.recommendation_domain === "restaurant";
    const items: AttractionAccordionItem[] = (attachment.items ?? []).map(
      (item) => ({
        id: item.recommendation_id,
        name: item.title,
        image: item.image_url ?? undefined,
        alt: item.title,
        reason: isRestaurant
          ? (item.match_reason ?? item.summary)
          : (item.city_significance ?? item.summary),
        experience: item.experience_summary ?? item.summary,
        duration: item.time_cost ?? undefined,
        tradeoff: isRestaurant
          ? (item.main_cost ?? undefined)
          : (item.physical_cost ?? undefined),
        sourceLabel:
          item.source_fact_ids && item.source_fact_ids.length > 0
            ? `${item.source_fact_ids.length} 个可追溯来源`
            : undefined,
        updatedAt:
          item.source_fact_ids && item.source_fact_ids.length > 0
            ? attachment.created_at
            : undefined,
      }),
    );
    const props = {
      label: attachment.prompt,
      items,
      values: recommendationValues,
      intentOptions: isRestaurant
        ? RESTAURANT_INTENT_OPTIONS
        : ATTRACTION_INTENT_OPTIONS,
      itemNoun: isRestaurant ? "餐厅" : "景点",
      disabled,
      onChange: (itemId: string, intent: RecommendationIntent) => {
        if (disabled) return;
        setRecommendationValues((current) => ({
          ...current,
          [itemId]: intent,
        }));
      },
      onConfirm: () => {
        if (disabled) return;
        onSubmit({
          answer_type: "recommendation_feedback",
          feedback: Object.entries(recommendationValues).map(
            ([recommendation_id, intent]) => ({
              recommendation_id,
              intent: intent as "must" | "want" | "if_convenient" | "avoid",
            }),
          ),
        });
      },
    };
    return (
      <div>
        {recommendationGalleryMode(items.length) === "depth" ? (
          <AttractionDepthCarouselAttachment {...props} />
        ) : (
          <AttractionAccordionAttachment {...props} />
        )}
        {feedback}
      </div>
    );
  }
  if (attachment.kind === "weather") {
    return (
      <WeatherDeck
        days={(attachment.days ?? []).map(toWeatherDay)}
        ariaLabel={attachment.prompt}
      />
    );
  }
  return (
    <button
      type="button"
      className="task-book-reference-link server-task-book-reference"
      disabled={disabled || attachment.task_book_id === null}
      onClick={() =>
        attachment.task_book_id && onOpenTaskBook?.(attachment.task_book_id)
      }
    >
      <span>{attachment.label}</span>
      <small>查看</small>
    </button>
  );
}

function isKnownAttachment(
  value: unknown,
): value is ServerConversationAttachment {
  if (!value || typeof value !== "object") return false;
  return [
    "compact_choice",
    "compact_multi",
    "detailed_choice",
    "preference_slider",
    "text_multi_choice",
    "recommendation_set",
    "task_book_reference",
    "weather",
  ].includes(String((value as { kind?: unknown }).kind));
}

function toChoiceOption(option: {
  option_id: string;
  label: string;
  description?: string | null;
}) {
  return {
    id: option.option_id,
    label: option.label,
    description: option.description ?? undefined,
  };
}

function toggleSelection(
  current: string[],
  id: string,
  attachment: Extract<
    ServerConversationAttachment,
    { kind: "compact_multi" | "text_multi_choice" }
  >,
) {
  if (
    attachment.kind === "text_multi_choice" &&
    attachment.exclusive_option_id === id
  ) {
    return current.length === 1 && current[0] === id ? [] : [id];
  }
  const withoutExclusive =
    attachment.kind === "text_multi_choice" && attachment.exclusive_option_id
      ? current.filter((value) => value !== attachment.exclusive_option_id)
      : current;
  if (withoutExclusive.includes(id))
    return withoutExclusive.filter((value) => value !== id);
  return withoutExclusive.length < attachment.maximum_selections
    ? [...withoutExclusive, id]
    : withoutExclusive;
}

function answerSummary(
  attachment: ServerConversationAttachment,
  record: AttachmentAnswerRecord,
) {
  const answer = record.answer;
  if (answer.answer_type === "single_choice" && "options" in attachment) {
    return (
      attachment.options.find((item) => item.option_id === answer.option_id)
        ?.label ?? "已选择"
    );
  }
  if (answer.answer_type === "multi_choice" && "options" in attachment) {
    const labels = attachment.options
      .filter((item) => answer.option_ids.includes(item.option_id))
      .map((item) => item.label)
      .join("、");
    return [labels, answer.free_text].filter(Boolean).join("；") || "已确认";
  }
  if (answer.answer_type === "slider")
    return answer.value === null ? "都可以" : String(answer.value);
  if (answer.answer_type === "recommendation_feedback")
    return "已记录这组候选的取舍";
  return "已确认";
}

function toWeatherDay(day: {
  date: string;
  daytime_condition: string;
  nighttime_condition?: string | null;
  minimum_celsius: number;
  maximum_celsius: number;
}): WeatherDayData {
  return {
    id: day.date,
    dateLabel: day.date.slice(5),
    condition: weatherCondition(day.daytime_condition),
    conditionLabel: day.daytime_condition,
    temperatureC: day.maximum_celsius,
    lowTemperatureC: day.minimum_celsius,
    travelNote: day.nighttime_condition
      ? `夜间：${day.nighttime_condition}`
      : null,
  };
}

function weatherCondition(label: string): WeatherCondition {
  if (/雷/.test(label)) return "thunderstorm";
  if (/雪/.test(label)) return "snow";
  if (/大雨|暴雨/.test(label)) return "heavy-rain";
  if (/雨/.test(label)) return "light-rain";
  if (/雾|霾/.test(label)) return "fog-haze";
  if (/风/.test(label)) return "wind";
  if (/多云/.test(label)) return "partly-cloudy";
  if (/阴/.test(label)) return "overcast";
  if (/晴/.test(label)) return "clear";
  return "unknown";
}
