import type { ConversationAttachment } from "./attachments";
import type {
  AttractionAccordionItem,
  RecommendationIntent,
  RecommendationIntentOption,
} from "./AttractionAccordionAttachment";
import type { CompactChoiceOption } from "./CompactChoiceAttachment";
import type { DetailedChoiceOption } from "./DetailedChoiceAttachment";
import type { TextChoiceOption } from "./TextMultiChoiceAttachment";
import type { WeatherDayData } from "./weatherTypes";

type MessageStatus = "sending" | "sent" | "failed";

export type ConversationMessage = {
  id: string;
  role: "user" | "agent";
  content: string;
  status: MessageStatus;
  attachments?: ConversationAttachment[];
  choiceLabel?: string;
  choiceSummaryLabel?: string;
  choiceOptions?: CompactChoiceOption[];
  detailedChoiceLabel?: string;
  detailedChoiceSummaryLabel?: string;
  detailedChoiceOptions?: DetailedChoiceOption[];
  textMultiChoiceLabel?: string;
  textMultiChoiceSummaryLabel?: string;
  textMultiChoiceOptions?: TextChoiceOption[];
  textMultiChoiceExclusiveId?: string;
  selectedChoiceId?: string;
  choiceConfirmed?: boolean;
  multiChoiceLabel?: string;
  multiChoiceSummaryLabel?: string;
  multiChoiceOptions?: CompactChoiceOption[];
  selectedChoiceIds?: string[];
  multiChoiceConfirmed?: boolean;
  sliderLabel?: string;
  sliderStartLabel?: string;
  sliderEndLabel?: string;
  sliderValue?: number;
  sliderFlexible?: boolean;
  sliderConfirmed?: boolean;
  attractionLabel?: string;
  attractionSummaryLabel?: string;
  attractionItemNoun?: string;
  attractionItems?: AttractionAccordionItem[];
  attractionValues?: Record<string, RecommendationIntent>;
  attractionIntentOptions?: readonly RecommendationIntentOption[];
  attractionDefaultIntent?: RecommendationIntent;
  attractionConfirmed?: boolean;
  recommendationEmpty?: boolean;
  planReady?: boolean;
  planWeatherDays?: WeatherDayData[];
  weatherDemo?: boolean;
};
