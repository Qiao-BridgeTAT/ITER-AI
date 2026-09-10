import { useMemo, useState } from "react";

import type {
  CardControlAction,
  CardOption,
  PendingInteraction,
  PlannerPublishedPlan,
  TaskBookV4,
  TripSemanticState,
  V4Attachment,
  V4CardAnswerPayload,
  V4CardSelection,
} from "../generated/v4/contracts";
import {
  usePlaceIntroductions,
  type IntroductionLoader,
} from "../planning/usePlaceIntroductions";
import { V4TaskBookPreview } from "../task-book/V4TaskBookPreview";
import {
  AttractionAccordionAttachment,
  type AttractionAccordionItem,
  type RecommendationIntent,
} from "./AttractionAccordionAttachment";
import { AttractionDepthCarouselAttachment } from "./AttractionDepthCarouselAttachment";
import { DiningFacts } from "./DiningFacts";
import { discoveryOptionDescription } from "./discoveryCopy";
import { recommendationGalleryMode } from "./recommendationGalleryMode";

export type V4CardAnswerDraft = Omit<V4CardAnswerPayload, "answer_id">;

type CardAttachment = Exclude<V4Attachment, TaskBookV4 | PlannerPublishedPlan>;
type DiscoveryAttachment = CardAttachment | TaskBookV4;
type CardDisposition = V4CardSelection["disposition"];
type CardSection = NonNullable<CardAttachment["section"]>;

type V4DiscoveryAttachmentProps = {
  attachment: DiscoveryAttachment;
  authoritativeTaskBook?: TaskBookV4 | null;
  semanticState?: TripSemanticState | null;
  pendingInteraction: PendingInteraction | null;
  disabled?: boolean;
  conflictMessage?: string;
  onSubmitCard: (answer: V4CardAnswerDraft) => boolean;
  onConfirmTaskBook: (taskBook: TaskBookV4) => boolean;
  onModifyTaskBook?: () => void;
  onReload?: () => void;
  loadIntroductions?: IntroductionLoader;
};

const DISPOSITION_LABELS: Record<CardDisposition, string> = {
  selected: "喜欢",
  excluded: "排除",
  must: "必去",
  want: "想去",
  destination: "必吃",
  if_convenient: "顺路去",
  avoid: "不去",
};

const CARD_SECTION_LABELS: Record<CardSection, string> = {
  attraction_preference: "景点方向",
  attraction_specific: "具体景点",
  dining_preference: "饮食方向",
  dining_specific: "具体餐厅",
  lodging_area_preference: "住宿区位",
  lodging_class_preference: "住宿档次与类型",
};

export function V4DiscoveryAttachment({
  attachment,
  authoritativeTaskBook,
  semanticState,
  pendingInteraction,
  disabled = false,
  conflictMessage,
  onSubmitCard,
  onConfirmTaskBook,
  onModifyTaskBook,
  onReload,
  loadIntroductions,
}: V4DiscoveryAttachmentProps) {
  if (isTaskBook(attachment)) {
    const taskBook =
      authoritativeTaskBook?.task_book_id === attachment.task_book_id &&
      authoritativeTaskBook.version === attachment.version
        ? authoritativeTaskBook
        : attachment;
    const isActive =
      pendingInteraction?.kind === "confirmation" &&
      pendingInteraction.target_ids.includes(taskBook.task_book_id) &&
      pendingInteraction.based_on_state_version ===
        taskBook.based_on_state_version;
    return (
      <V4TaskBookPreview
        taskBook={taskBook}
        semanticState={semanticState}
        disabled={disabled || !isActive}
        conflictMessage={conflictMessage}
        onConfirm={() => onConfirmTaskBook(taskBook)}
        onModify={onModifyTaskBook}
        onReload={onReload}
      />
    );
  }

  const isActive =
    pendingInteraction?.status !== "superseded" &&
    pendingInteraction?.interaction_id === attachment.interaction_id;
  return (
    <DiscoveryCard
      key={attachment.attachment_id}
      card={attachment}
      semanticState={semanticState}
      disabled={disabled || !isActive}
      inactive={!isActive}
      conflictMessage={conflictMessage}
      onSubmit={onSubmitCard}
      onReload={onReload}
      loadIntroductions={loadIntroductions}
    />
  );
}

function DiscoveryCard({
  card,
  semanticState,
  disabled,
  inactive,
  conflictMessage,
  onSubmit,
  onReload,
  loadIntroductions,
}: {
  card: CardAttachment;
  semanticState?: TripSemanticState | null;
  disabled: boolean;
  inactive: boolean;
  conflictMessage?: string;
  onSubmit: (answer: V4CardAnswerDraft) => boolean;
  onReload?: () => void;
  loadIntroductions?: IntroductionLoader;
}) {
  const introductions = usePlaceIntroductions(
    card.attachment_id,
    "card",
    card.section === "dining_specific",
    loadIntroductions,
  );
  const [selections, setSelections] = useState<Record<string, CardDisposition>>(
    () => initialSelections(card, semanticState),
  );
  const [textControl, setTextControl] = useState<CardControlAction | null>(
    null,
  );
  const [freeText, setFreeText] = useState("");
  const selected = useMemo<V4CardSelection[]>(
    () =>
      Object.entries(selections).map(([option_id, disposition]) => ({
        option_id,
        disposition,
      })),
    [selections],
  );
  const isSingleChoice = card.section === "lodging_class_preference";
  const isAttractionGallery =
    card.kind === "specific_card" && card.domain === "attraction";
  const visibleOptions = useMemo(
    () =>
      card.section === "dining_specific"
        ? [...card.options].sort(
            (a, b) =>
              Number(b.composition_role === "representative_extra") -
              Number(a.composition_role === "representative_extra"),
          )
        : card.options,
    [card.options, card.section],
  );
  const galleryItems: AttractionAccordionItem[] = isAttractionGallery
    ? card.options.map((option) => ({
        id: option.option_id,
        name: option.label,
        image: option.image_url ?? undefined,
        alt: `${option.label}实景`,
        badgeTone:
          option.composition_role === "representative_extra"
            ? "city"
            : "personalized",
        reason:
          option.composition_role === "representative_extra"
            ? "城市代表"
            : "个性化推荐",
        experience:
          card.generation_metadata.strategy_version === "attraction-v2"
            ? (option.description ?? "")
            : "",
        compact: true,
      }))
    : [];
  const galleryValues: Record<string, RecommendationIntent> = {};
  for (const [id, value] of Object.entries(selections)) {
    if (
      value === "must" ||
      value === "want" ||
      value === "if_convenient" ||
      value === "avoid"
    ) {
      galleryValues[id] = value;
    }
  }
  const Gallery =
    recommendationGalleryMode(galleryItems.length) === "depth"
      ? AttractionDepthCarouselAttachment
      : AttractionAccordionAttachment;

  const setDisposition = (optionId: string, disposition: CardDisposition) => {
    if (disabled) return;
    setSelections((current) => {
      if (isSingleChoice) {
        return current[optionId] === disposition
          ? {}
          : { [optionId]: disposition };
      }
      if (current[optionId] === disposition) {
        const next = { ...current };
        delete next[optionId];
        return next;
      }
      return { ...current, [optionId]: disposition };
    });
  };

  const submitSelections = () => {
    if (disabled || selected.length === 0) return;
    onSubmit({
      interaction_id: card.interaction_id,
      selections: selected,
    });
  };

  const submitControl = (control: CardControlAction) => {
    if (disabled) return;
    if (control.kind === "free_text" || control.kind === "existing_booking") {
      setTextControl(control);
      return;
    }
    onSubmit({
      interaction_id: card.interaction_id,
      selections: [],
      control_action_id: control.control_id,
    });
  };

  const submitFreeText = () => {
    const value = freeText.trim();
    if (disabled || textControl === null || !value) return;
    if (
      onSubmit({
        interaction_id: card.interaction_id,
        selections: [],
        control_action_id: textControl.control_id,
        optional_user_text: value,
      })
    ) {
      setTextControl(null);
      setFreeText("");
    }
  };

  return (
    <section
      className={`v4-discovery-card${inactive ? " is-inactive" : ""}${isAttractionGallery ? " v4-attraction-gallery" : ""}`}
      data-v4-attachment-kind={card.kind}
      data-v4-attachment-id={card.attachment_id}
      data-v4-interaction-id={card.interaction_id}
      data-v4-card-section={card.section}
      data-v4-card-mode={card.generation_metadata.generation_mode}
      data-v4-gallery-mode={
        isAttractionGallery
          ? recommendationGalleryMode(galleryItems.length)
          : undefined
      }
    >
      <header className="v4-discovery-card-header">
        <div>
          <span className="v4-discovery-card-eyebrow">
            {sectionLabel(card.section)}
          </span>
          <h3>{card.prompt}</h3>
        </div>
        {card.domain !== "attraction" ? (
          <span className="v4-discovery-card-count">
            {card.options.length} 项
          </span>
        ) : null}
      </header>

      {isAttractionGallery ? (
        <Gallery
          label="景点推荐"
          items={galleryItems}
          values={galleryValues}
          intentOptions={[
            { value: "must", label: "必去" },
            { value: "want", label: "想去" },
            { value: "if_convenient", label: "顺路去" },
            { value: "avoid", label: "不去" },
          ]}
          disabled={disabled}
          confirmDisabled={selected.length === 0}
          onChange={(id, intent) => {
            if (
              intent === "must" ||
              intent === "want" ||
              intent === "if_convenient" ||
              intent === "avoid"
            ) {
              setDisposition(id, intent);
            }
          }}
          onConfirm={submitSelections}
        />
      ) : (
        <div className="v4-discovery-options">
          {visibleOptions.map((option) => (
            <DiscoveryOption
              key={option.option_id}
              card={card}
              option={option}
              introduction={introductions.get(
                option.entity_ref?.canonical_entity_id ?? "",
              )}
              value={selections[option.option_id]}
              disabled={disabled}
              onChange={(disposition) =>
                setDisposition(option.option_id, disposition)
              }
            />
          ))}
        </div>
      )}

      {!inactive && !isAttractionGallery ? (
        <div className="v4-discovery-actions">
          <button
            type="button"
            className="v4-card-primary-action"
            disabled={disabled || selected.length === 0}
            onClick={submitSelections}
          >
            确认选择
          </button>
          <div className="v4-card-control-actions" aria-label="其他处理方式">
            {(card.control_actions ?? []).map((control) => (
              <button
                key={control.control_id}
                type="button"
                disabled={disabled}
                onClick={() => submitControl(control)}
              >
                {control.label}
              </button>
            ))}
          </div>
        </div>
      ) : inactive ? (
        <p className="v4-discovery-card-complete" role="status">
          这张卡已经完成，或已被后续互动替代。
        </p>
      ) : null}

      {textControl !== null && !inactive ? (
        <div className="v4-card-free-text">
          <label htmlFor={`v4-card-text-${card.attachment_id}`}>
            {textControl.kind === "existing_booking"
              ? "告诉我已经预订的住宿和位置"
              : "补充你的真实要求"}
          </label>
          <div>
            <input
              id={`v4-card-text-${card.attachment_id}`}
              value={freeText}
              disabled={disabled}
              autoFocus
              onChange={(event) => setFreeText(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter") submitFreeText();
              }}
            />
            <button
              type="button"
              disabled={disabled || !freeText.trim()}
              onClick={submitFreeText}
            >
              发送
            </button>
          </div>
        </div>
      ) : null}

      {conflictMessage ? (
        <div className="attachment-submit-conflict" role="alert">
          <span>{conflictMessage}</span>
          {onReload ? (
            <button type="button" onClick={onReload}>
              刷新最新状态
            </button>
          ) : null}
        </div>
      ) : null}
    </section>
  );
}

function DiscoveryOption({
  card,
  option,
  value,
  disabled,
  onChange,
  introduction,
}: {
  card: CardAttachment;
  option: CardOption;
  value?: CardDisposition;
  disabled: boolean;
  onChange: (value: CardDisposition) => void;
  introduction?: string;
}) {
  const dispositions = optionDispositions(card, option);
  const description = discoveryOptionDescription(
    card.section,
    introduction ?? option.description,
  );
  return (
    <article
      className={`v4-discovery-option${value ? " is-selected" : ""}`}
      data-option-id={option.option_id}
      data-composition-role={option.composition_role ?? undefined}
      data-entity-kind={option.entity_ref?.entity_kind ?? undefined}
    >
      <div className="v4-discovery-option-copy">
        <div className="v4-discovery-option-title">
          <strong>{option.label}</strong>
          {option.composition_role === "representative_extra" ? (
            <span data-badge-tone="city">城市代表</span>
          ) : (card.domain === "attraction" || card.domain === "dining") &&
            option.composition_role === "personalized_top" ? (
            <span data-badge-tone="personalized">个性化推荐</span>
          ) : null}
        </div>
        {card.section === "dining_specific" ? (
          <DiningFacts facts={option.dining_details} />
        ) : null}
        {description ? (
          <p
            className={
              card.section === "dining_specific"
                ? "place-introduction-line"
                : undefined
            }
            title={card.section === "dining_specific" ? description : undefined}
          >
            {description}
          </p>
        ) : null}
      </div>
      <div className="v4-discovery-dispositions">
        {dispositions.map((disposition) => (
          <button
            key={disposition}
            type="button"
            aria-pressed={value === disposition}
            disabled={disabled || option.selection_state === "unavailable"}
            onClick={() => onChange(disposition)}
          >
            {specificDispositionLabel(card, disposition)}
          </button>
        ))}
      </div>
    </article>
  );
}

function optionDispositions(
  card: CardAttachment,
  option: CardOption,
): CardDisposition[] {
  if (card.section === "lodging_class_preference") return ["selected"];
  if (card.kind === "specific_card") {
    return option.semantic_value.kind === "entity_disposition"
      ? option.semantic_value.allowed_dispositions
      : [];
  }
  return ["selected", "excluded"];
}

function specificDispositionLabel(
  card: CardAttachment,
  disposition: CardDisposition,
): string {
  if (card.section === "lodging_class_preference") return "选择";
  if (card.domain === "dining" && disposition === "if_convenient") {
    return "可吃";
  }
  if (card.domain === "dining" && disposition === "avoid") return "不吃";
  if (card.domain !== "attraction" && disposition === "avoid") return "排除";
  return DISPOSITION_LABELS[disposition];
}

function initialSelections(
  card: CardAttachment,
  semanticState?: TripSemanticState | null,
): Record<string, CardDisposition> {
  const selections: Record<string, CardDisposition> = {};
  if (card.kind === "specific_card") {
    if (card.domain !== "attraction" || !semanticState) return selections;
    const intents = [
      ...(semanticState.attractions?.concrete_intents ?? []),
      ...(semanticState.attractions?.exclusions ?? []),
    ];
    for (const option of card.options) {
      const intent = intents.find(
        (item) =>
          item.canonical_entity_id === option.entity_ref?.canonical_entity_id,
      );
      if (
        intent &&
        (intent.disposition === "must" ||
          intent.disposition === "want" ||
          intent.disposition === "if_convenient" ||
          intent.disposition === "avoid")
      ) {
        selections[option.option_id] = intent.disposition;
      }
    }
    return selections;
  }
  for (const option of card.options) {
    const semantic = option.semantic_value;
    const savedDirection =
      card.domain === "attraction" && semantic.kind === "direction"
        ? semanticState?.attractions?.preference_directions?.find(
            (direction) => direction.direction_id === semantic.direction_id,
          )
        : undefined;
    if (savedDirection) {
      selections[option.option_id] = savedDirection.selected
        ? "selected"
        : "excluded";
    } else if (option.selection_state === "selected") {
      selections[option.option_id] = "selected";
    } else if (option.selection_state === "excluded") {
      selections[option.option_id] = "excluded";
    }
  }
  return selections;
}

function sectionLabel(section: CardAttachment["section"]): string {
  return section ? CARD_SECTION_LABELS[section] : "旅行偏好";
}

function isTaskBook(attachment: DiscoveryAttachment): attachment is TaskBookV4 {
  return "task_book_id" in attachment;
}
