import type { RecommendationIntentOption } from "./AttractionAccordionAttachment";

export const ATTRACTION_INTENT_OPTIONS: readonly RecommendationIntentOption[] =
  [
    { value: "must", label: "必去" },
    { value: "want", label: "想去" },
    { value: "if_convenient", label: "顺路可去" },
    { value: "avoid", label: "不想去" },
  ];

export const RESTAURANT_INTENT_OPTIONS: readonly RecommendationIntentOption[] =
  [
    { value: "must", label: "想专程去" },
    { value: "if_convenient", label: "顺路可以" },
    { value: "avoid", label: "不感兴趣" },
  ];
