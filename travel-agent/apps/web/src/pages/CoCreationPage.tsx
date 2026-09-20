import { TripReferenceLinks } from "../conversation/TripReferenceLinks";
import { useReplyScrollAnchor } from "../conversation/useReplyScrollAnchor";
import {
  tripLength,
  type TripSetupFields,
} from "../trip-setup/dateRangeSelection";
import { destinationCities } from "../trip-setup/citySearch";
import { DestinationDateCard } from "../trip-setup/DestinationDateCard";
import { AgentMarkdown } from "../conversation/AgentMarkdown";
import {
  ConversationMotionProvider,
  ConversationEntrance,
  ProgressiveAgentText,
} from "../conversation/ConversationMotion";
import { planCompletionCopy } from "../conversation/planCompletionCopy";
import { TripRestoreStatus } from "../conversation/TripRestoreStatus";
import { TripRestoreTiming } from "../conversation/TripRestoreTiming";
import { V4HistoricalPlan } from "../conversation/V4HistoricalPlan";
import {
  isPublishedPlanMessage,
  v4PlanInsertionIndex,
} from "../conversation/v4PlanMessagePosition";
import {
  ChangeEvent,
  FormEvent,
  Fragment,
  KeyboardEvent,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  Link,
  useNavigate,
  useParams,
  useSearchParams,
} from "react-router-dom";

import { useTripBackend } from "../backend/TripBackendContext";
import { pathWithAgentProtocol } from "../backend/agentProtocol";
import {
  ConversationAttachment,
  FileAttachment,
  LocationAttachment,
  formatFileSize,
} from "../conversation/attachments";
import { AgentLoadingIndicator } from "../conversation/AgentLoadingIndicator";
import { LoginOverlay } from "../account/LoginOverlay";
import { AccountSessionOverlay } from "../account/AccountSessionOverlay";
import { TripHistoryPopover } from "../account/TripHistoryPopover";
import { PlanReadyAttachment } from "../conversation/PlanReadyAttachment";
import {
  ServerAttachmentRenderer,
  type AttachmentAnswer,
} from "../conversation/ServerAttachmentRenderer";
import type { WeatherDayData } from "../conversation/weatherTypes";
import {
  CompactChoiceAttachment,
  CompactMultiChoiceAttachment,
  CompactChoiceOption,
} from "../conversation/CompactChoiceAttachment";
import {
  DetailedChoiceAttachment,
  DetailedChoiceOption,
} from "../conversation/DetailedChoiceAttachment";
import {
  TextChoiceOption,
  TextMultiChoiceAttachment,
} from "../conversation/TextMultiChoiceAttachment";
import { CompletedAttachmentSummary } from "../conversation/CompletedAttachmentSummary";
import { PreferenceSliderAttachment } from "../conversation/PreferenceSliderAttachment";
import { ReferencePreviewDialog } from "../conversation/ReferencePreviewDialog";
import {
  AttractionAccordionAttachment,
  AttractionAccordionItem,
  RecommendationIntent,
  RecommendationIntentOption,
} from "../conversation/AttractionAccordionAttachment";
import { AttractionDepthCarouselAttachment } from "../conversation/AttractionDepthCarouselAttachment";
import { ATTRACTION_INTENT_OPTIONS } from "../conversation/recommendationIntents";
import { RecommendationEmptyAttachment } from "../conversation/RecommendationEmptyAttachment";
import {
  V4DiscoveryAttachment,
  type V4CardAnswerDraft,
} from "../conversation/V4DiscoveryAttachment";
import { V4CardRecovery } from "../conversation/V4CardRecovery";
import { MemoryManager } from "../account/MemoryManager";
import { AgentProgressHistory } from "../conversation/AgentProgressHistory";
import { positionAgentProgress } from "../conversation/agentProgressPresentation";
import { V4PlannerPanel } from "../conversation/V4PlannerPanel";
import { recommendationGalleryMode } from "../conversation/recommendationGalleryMode";
import { Dock, DockItemData } from "../components/Dock";
import { ColdStartModal } from "../cold-start/ColdStartModal";
import { HelpModal } from "./HelpModal";
import { MOCK_SAVED_PERSONAL_DEFAULTS } from "../cold-start/coldStartProfiles";
import {
  getRegisteredCityName,
  getSupportedCity,
} from "../city/cityContentRepository";
import {
  discoveryCity,
  discoveryPlace,
  type DiscoveryMapPreview,
} from "../map/discoveryMap";
import { AmapSpaceBoard } from "../map/AmapSpaceBoard";
import { mergeMapUpdates } from "../map/mergeMapUpdates";
import { buildPublishedPlanPresentation } from "../planning/publishedPlanPresentation";
import {
  buildPublishedPlanDayMap,
  publishedPlanRouteNotice,
} from "../planning/publishedPlanMap";
import {
  buildV4PublishedPlanDayMap,
  v4PublishedPlanRouteNotice,
} from "../planning/v4PublishedPlanMap";
import { buildV4PublishedPlanPresentation } from "../planning/v4PublishedPlanPresentation";
import {
  visibleV4PublishedPlan,
  visibleV4PlanCityName,
} from "../planning/visibleV4PublishedPlan";
import { usePlanPreview } from "../planning/usePlanPreview";
import { usePlaceIntroductions } from "../planning/usePlaceIntroductions";
import type {
  CancelGenerationCommand,
  AttachmentAnswerCommand,
  ColdStartSubmission,
  TaskBookConfirmCommand,
  UserMessageCommand,
} from "../generated/contracts";
import type {
  SpecificCandidateCard,
  TaskBookV4,
  V4CancelGenerationCommand,
  V4CardAnswerCommand,
  V4RetryInteractionCommand,
  V4TaskBookConfirmationCommand,
  V4UserMessageCommand,
  V4TripSetupCommand,
} from "../generated/v4/contracts";
import type {
  V4PlannerResumeCommand,
  V4PlannerAnswerCommand,
  V4PlanTransportSelectionCommand,
} from "../generated/v4/contracts";
import {
  CURRENT_PROTOCOL_VERSION,
  CURRENT_SCHEMA_VERSION,
} from "../generated/protocol";
import { useTripRuntime } from "../realtime/TripRuntimeContext";
import { isDiscoveryProgress } from "../realtime/v4EventReducer";
import { useTripShell } from "../session/TripShellContext";
import {
  useViewerSession,
  useViewerSessionActions,
} from "../session/viewerSession";
import {
  FloatingTaskBook,
  TravelTaskBookContent,
} from "../task-book/FloatingTaskBook";
import { ServerTaskBookPreview } from "../task-book/ServerTaskBookPreview";
import { V4TaskBookPreview } from "../task-book/V4TaskBookPreview";
import {
  calendarDayDifference,
  destinationToday,
  parseCalendarDate,
} from "../trip-setup/dateMath";
type GenerationState = "idle" | "generating";
type MessageStatus = "sending" | "sent" | "failed";

type ConversationMessage = {
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

function TransientRemoteMessageRow({
  message,
  onSelectReference,
}: {
  message: ConversationMessage;
  onSelectReference: (file: FileAttachment) => void;
}) {
  return (
    <ConversationEntrance
      as="article"
      motionKey={`row:${message.id}`}
      className={`conversation-message-row conversation-message-row-${message.role}`}
      data-local-message={message.id}
      data-message-role={message.role}
      data-message-status={message.status}
    >
      <div className="conversation-message-stack">
        <div
          className={`message-bubble message-bubble-${message.role}`}
          role={message.status === "failed" ? "alert" : undefined}
        >
          {message.content ? (
            <p style={{ whiteSpace: "pre-wrap", overflowWrap: "anywhere" }}>
              {message.content}
            </p>
          ) : null}
          {message.attachments?.length ? (
            <div className="message-attachments" aria-label="消息附件">
              {message.attachments.map((attachment) =>
                attachment.kind === "file" ? (
                  <button
                    key={attachment.id}
                    type="button"
                    onClick={() => onSelectReference(attachment)}
                  >
                    <img src="/icons/attachment-figma.svg" alt="" />
                    <span>
                      <strong>{attachment.name}</strong>
                      <small>{formatFileSize(attachment.size)}</small>
                    </span>
                  </button>
                ) : (
                  <div
                    key={attachment.id}
                    className="message-location-attachment"
                  >
                    <span aria-hidden="true">⌖</span>
                    <span>
                      <strong>{attachment.name}</strong>
                      <small>{attachment.detail}</small>
                    </span>
                  </div>
                ),
              )}
            </div>
          ) : null}
          {message.status === "sending" ? (
            <span className="message-status">发送中…</span>
          ) : null}
        </div>
      </div>
    </ConversationEntrance>
  );
}

const INITIAL_MESSAGES: ConversationMessage[] = [
  {
    id: "welcome",
    role: "agent",
    content: "先说说你想去哪里，或者想拥有一段怎样的旅程。",
    status: "sent",
  },
];

type CompactChoiceCount = 2 | 3 | 4;

function formatShortDate(value: string): string {
  const date = parseCalendarDate(value);
  return date ? `${date.month} 月 ${date.day} 日` : value;
}

function tripSummaryTitle(
  cityName?: string | null,
  startDate?: string | null,
  endDate?: string | null,
): string | null {
  if (
    !cityName?.trim() ||
    !startDate ||
    !endDate ||
    !parseCalendarDate(startDate) ||
    !parseCalendarDate(endDate)
  ) {
    return null;
  }
  const length = tripLength({ start: startDate, end: endDate });
  return length ? `${cityName.trim()}${length.days}天${length.nights}晚` : null;
}
const AGENT_ERROR_REPLY = "抱歉，好像出了点问题，请稍后再试。";

function describeSliderValue(value: number): string {
  if (value <= 20) {
    return "很松弛";
  }
  if (value <= 40) {
    return "偏松弛";
  }
  if (value <= 60) {
    return "松紧平衡";
  }
  if (value <= 80) {
    return "偏充实";
  }
  return "安排更满";
}

function sliderSubmitText(value: number): string {
  const description = describeSliderValue(value);
  if (description === "松紧平衡") {
    return "我希望每天松紧平衡。";
  }
  if (description === "安排更满") {
    return "我希望每天安排得更满。";
  }
  return `我希望每天安排得${description}一点。`;
}

type MenuIconName = "trips" | "preferences" | "help" | "account";

function MenuIcon({ name }: { name: MenuIconName }) {
  if (name === "trips") {
    return (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
        <path
          d="M7 4.5h8l3 3V19.5H7zM15 4.5v3h3M10 11h5M10 15h5"
          strokeWidth="1.6"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
    );
  }

  if (name === "preferences") {
    return (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
        <path
          d="M5 7h8M17 7h2M5 12h2M11 12h8M5 17h7M16 17h3"
          strokeWidth="1.6"
          strokeLinecap="round"
        />
        <circle cx="15" cy="7" r="2" strokeWidth="1.6" />
        <circle cx="9" cy="12" r="2" strokeWidth="1.6" />
        <circle cx="14" cy="17" r="2" strokeWidth="1.6" />
      </svg>
    );
  }

  if (name === "help") {
    return (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
        <circle cx="12" cy="12" r="8" strokeWidth="1.6" />
        <path
          d="M9.8 9.5a2.35 2.35 0 1 1 3.1 2.23c-.62.27-.9.66-.9 1.27v.35M12 17h.01"
          strokeWidth="1.6"
          strokeLinecap="round"
        />
      </svg>
    );
  }

  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
      <circle cx="12" cy="8.5" r="3" strokeWidth="1.6" />
      <path
        d="M6.5 19c.6-3.15 2.43-4.75 5.5-4.75s4.9 1.6 5.5 4.75"
        strokeWidth="1.6"
        strokeLinecap="round"
      />
    </svg>
  );
}

export function CoCreationPage() {
  const { tripId } = useParams();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const { shell } = useTripShell();
  const backend = useTripBackend();
  const viewer = useViewerSession();
  const viewerActions = useViewerSessionActions();
  const demoMode: string = "normal";
  const choiceCount = 3;
  const tripState = useTripRuntime((state) => state.tripState);
  const runtimePresentation = useTripRuntime((state) => state.presentation);
  const v4Runtime = useTripRuntime((state) => state.v4);
  const isV4 = backend.protocol === "v4";
  const isRestoringTrip =
    backend.mode === "real" &&
    (isV4
      ? v4Runtime.tripState?.semantic_state.trip_id
      : tripState?.trip_id) !== shell.trip_id;
  const v4TaskBook = isV4
    ? (v4Runtime.tripState?.discovery_runtime_state.task_book_candidate
        ?.value ?? null)
    : null;
  const v4TaskBookIsActive = Boolean(
    v4TaskBook &&
    v4Runtime.pendingInteraction?.kind === "confirmation" &&
    v4Runtime.pendingInteraction.target_ids.includes(v4TaskBook.task_book_id) &&
    v4Runtime.pendingInteraction.based_on_state_version ===
      v4TaskBook.based_on_state_version,
  );
  const homeHref = pathWithAgentProtocol("/", backend.requestedProtocol);
  const serverConversationMessages = runtimePresentation.conversationMessages;
  const v4ConversationMessages = v4Runtime.presentation.messages;
  const currentGenerationId = useTripRuntime(
    (state) => state.cursor.currentGenerationId,
  );
  const runtimePersonalDefaults = useTripRuntime(
    (state) => state.tripState?.personal_defaults ?? null,
  );
  const [messages, setMessages] = useState<ConversationMessage[]>(
    () => INITIAL_MESSAGES,
  );
  const [draft, setDraft] = useState("");
  const [generationState, setGenerationState] = useState<GenerationState>(() =>
    demoMode === "loading" ? "generating" : "idle",
  );
  const [loadingStartMessage, setLoadingStartMessage] = useState<string>();
  const [submittedTripTitle, setSubmittedTripTitle] = useState<{
    tripId: string;
    stateVersion: number;
    title: string | null;
  } | null>(null);
  useEffect(() => {
    if (generationState === "idle") setLoadingStartMessage(undefined);
  }, [generationState]);
  const [isSending, setIsSending] = useState(false);
  const [v4StopRequestTripId, setV4StopRequestTripId] = useState<string | null>(
    null,
  );
  const [menuOpen, setMenuOpen] = useState(false);
  const [helpOpen, setHelpOpen] = useState(false);
  const [tripsOpen, setTripsOpen] = useState(false);
  const [longTermPreferencesOpen, setLongTermPreferencesOpen] = useState(false);
  const [accountOpen, setAccountOpen] = useState(false);
  const [longTermPreferences, setLongTermPreferences] = useState<
    ColdStartSubmission | undefined
  >(
    () =>
      (viewer.kind === "user" ? viewer.personalDefaults : undefined) ??
      runtimePersonalDefaults ??
      undefined,
  );
  const [locationOpen, setLocationOpen] = useState(false);
  const [locationQuery, setLocationQuery] = useState("");
  const [locationError, setLocationError] = useState<string | null>(null);
  const [isLocating, setIsLocating] = useState(false);
  const [pendingAttachments, setPendingAttachments] = useState<
    ConversationAttachment[]
  >([]);
  const [referenceFiles, setReferenceFiles] = useState<FileAttachment[]>([]);
  const [selectedReference, setSelectedReference] =
    useState<FileAttachment | null>(null);
  const [pendingAttachmentRequests, setPendingAttachmentRequests] = useState<
    Record<string, string>
  >({});
  const [attachmentConflicts, setAttachmentConflicts] = useState<
    Record<string, string>
  >({});
  const [openServerTaskBook, setOpenServerTaskBook] = useState<{
    taskBookId: string;
    label: string;
  } | null>(null);
  const [taskBookConfirmRequestId, setTaskBookConfirmRequestId] = useState<
    string | null
  >(null);
  const [taskBookConfirmError, setTaskBookConfirmError] = useState<
    string | null
  >(null);
  const [highlightedMapPlaceId, setHighlightedMapPlaceId] = useState<
    string | null
  >(null);
  const [selectedPlanDayIndex, setSelectedPlanDayIndex] = useState(0);
  const messageCounter = useRef(0);
  const v4ClientSequence = useRef(0);
  const timers = useRef<Set<number>>(new Set());
  const generationTimer = useRef<number | null>(null);
  const didSimulateSendFailure = useRef(false);
  const didSimulateGenerationFailure = useRef(false);
  const handledRemoteTerminalEvent = useRef<string | null>(null);
  const handledV4AttachmentTerminalEvent = useRef<string | null>(null);
  const renderedTripId = useRef(shell.trip_id);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const menuButtonRef = useRef<HTMLButtonElement>(null);
  const menuDockRef = useRef<HTMLElement>(null);
  const locationButtonRef = useRef<HTMLButtonElement>(null);
  const locationPopoverRef = useRef<HTMLDivElement>(null);
  const conversationCanvasRef = useRef<HTMLDivElement>(null);
  const objectUrls = useRef<string[]>([]);
  const remoteConversation =
    demoMode === "normal" &&
    (backend.mode === "real" ||
      (isV4
        ? v4ConversationMessages.length > 0 ||
          v4Runtime.presentation.generationId !== null ||
          Boolean(v4Runtime.presentation.streamText)
        : serverConversationMessages.length > 0 ||
          runtimePresentation.generationId !== null ||
          Boolean(runtimePresentation.streamText)));
  const serverMessageIds = new Set(
    (isV4 ? v4ConversationMessages : serverConversationMessages).map(
      (message) => message.message_id,
    ),
  );
  const localConversationMessages = messages.filter(
    (message) => !serverMessageIds.has(message.id),
  );
  const transientRemoteMessages = localConversationMessages.filter(
    (message) => message.id !== "welcome",
  );
  const activeGenerationId = isV4
    ? v4Runtime.cursor.currentGenerationId
    : currentGenerationId;
  const activeStreamText = isV4
    ? v4Runtime.presentation.streamText
    : runtimePresentation.streamText;
  const activePresentationGenerationId = isV4
    ? v4Runtime.presentation.generationId
    : runtimePresentation.generationId;
  const hasAssistantMessageForGeneration = (
    isV4 ? v4ConversationMessages : serverConversationMessages
  ).some(
    (message) =>
      message.role === "assistant" &&
      message.generation_id === activePresentationGenerationId,
  );
  const v4PublishedPlan = isV4
    ? (v4Runtime.tripState?.published_plan ?? null)
    : null;
  const v4VisiblePlan = useMemo(
    () =>
      isV4
        ? visibleV4PublishedPlan(v4Runtime.tripState, v4ConversationMessages)
        : null,
    [isV4, v4Runtime.tripState, v4ConversationMessages],
  );
  const historicalV4Plan = Boolean(v4VisiblePlan && !v4PublishedPlan);
  const planPreviews = usePlanPreview(
    v4VisiblePlan,
    v4ConversationMessages,
    remoteConversation && isV4 && backend.status === "connected",
    backend.getPlanPreview,
  );
  const planIntroductions = usePlaceIntroductions(
    v4VisiblePlan?.plan_version_id,
    "plan",
    remoteConversation && isV4 && backend.status === "connected",
    backend.getPlaceIntroductions,
  );
  const planPlacePreviews = useMemo(() => {
    const places = new Map(
      planPreviews.places.map((place) => [place.place_id, place]),
    );
    for (const [place_id, description] of planIntroductions) {
      places.set(place_id, { ...places.get(place_id), place_id, description });
    }
    return [...places.values()];
  }, [planPreviews.places, planIntroductions]);
  const [expandedPlanMap, setExpandedPlanMap] = useState(false);
  const [mapFocusedDayIndex, setMapFocusedDayIndex] = useState<
    number | undefined
  >();
  const v4PlannerStatus = v4Runtime.plannerWorkspace?.status;
  const referenceLinks = isV4 ? (v4Runtime.referenceLinks ?? []) : [];
  const showV4PlannerPanel =
    remoteConversation &&
    isV4 &&
    (!v4PublishedPlan ||
      generationState === "generating" ||
      (v4PlannerStatus != null &&
        ["planning", "awaiting_user", "failed", "cancelled", "stale"].includes(
          v4PlannerStatus,
        )));
  const formalMapUpdate = useMemo(
    () =>
      v4VisiblePlan
        ? buildV4PublishedPlanDayMap(v4VisiblePlan, selectedPlanDayIndex)
        : tripState?.published_plan
          ? buildPublishedPlanDayMap(
              tripState.published_plan,
              selectedPlanDayIndex,
            )
          : mergeMapUpdates(tripState?.map_view, runtimePresentation.mapUpdate),
    [
      v4VisiblePlan,
      selectedPlanDayIndex,
      tripState?.published_plan,
      tripState?.map_view,
      runtimePresentation.mapUpdate,
    ],
  );
  const [discoveryFocus, setDiscoveryFocus] = useState<{
    attachmentId: string;
    optionId: string;
    request: number;
  } | null>(null);
  const onDiscoveryFocus = useCallback(
    (attachmentId: string, optionId: string) => {
      setDiscoveryFocus((current) => ({
        attachmentId,
        optionId,
        request: (current?.request ?? 0) + 1,
      }));
    },
    [],
  );
  useEffect(() => {
    setDiscoveryFocus(null);
  }, [shell.trip_id, v4Runtime.pendingInteraction?.interaction_id]);
  const discoveryCard = useMemo(() => {
    const cards = v4ConversationMessages
      .flatMap((message) => message.attachments ?? [])
      .filter(
        (attachment): attachment is SpecificCandidateCard =>
          "kind" in attachment &&
          attachment.kind === "specific_card" &&
          (attachment.section === "attraction_specific" ||
            attachment.section === "dining_specific"),
      );
    return (
      cards.find(
        (attachment) =>
          attachment.attachment_id === discoveryFocus?.attachmentId,
      ) ??
      cards.find(
        (attachment) =>
          attachment.interaction_id ===
          v4Runtime.pendingInteraction?.interaction_id,
      )
    );
  }, [
    v4ConversationMessages,
    v4Runtime.pendingInteraction?.interaction_id,
    discoveryFocus?.attachmentId,
  ]);
  const discoveryPlaces = useMemo(
    () =>
      discoveryCard && "options" in discoveryCard
        ? discoveryCard.options.map(discoveryPlace)
        : [],
    [discoveryCard],
  );
  const tripBasics = v4Runtime.tripState?.semantic_state.trip_basics;
  const conversationTitle =
    (isV4 && !isRestoringTrip
      ? (tripSummaryTitle(
          tripBasics?.destination_name,
          tripBasics?.start_date,
          tripBasics?.end_date,
        ) ??
        (submittedTripTitle?.tripId === shell.trip_id &&
        submittedTripTitle.stateVersion === v4Runtime.cursor.localStateVersion
          ? submittedTripTitle.title
          : null))
      : null) ?? "新的旅行";
  const discoveryPreview = useMemo<DiscoveryMapPreview>(() => {
    const card =
      discoveryCard && "options" in discoveryCard ? discoveryCard : null;
    const places = discoveryPlaces;
    const defaultIndex = places.length > 7 ? 0 : Math.floor(places.length / 2);
    return {
      city: discoveryCity(
        tripBasics?.destination_canonical_id,
        tripBasics?.destination_name,
      ),
      places,
      focusedId:
        card && discoveryFocus?.attachmentId === card.attachment_id
          ? discoveryFocus.optionId
          : card?.section === "dining_specific"
            ? null
            : (places[defaultIndex]?.id ?? null),
      focusRequest: discoveryFocus?.request,
    };
  }, [
    discoveryCard,
    discoveryPlaces,
    discoveryFocus,
    tripBasics?.destination_canonical_id,
    tripBasics?.destination_name,
  ]);
  const showDiscoveryMap =
    !formalMapUpdate ||
    Boolean(
      discoveryCard?.section === "dining_specific" &&
      discoveryFocus?.attachmentId === discoveryCard.attachment_id,
    );
  const mapRouteNotice = formalMapUpdate
    ? v4VisiblePlan
      ? v4PublishedPlanRouteNotice(v4VisiblePlan, selectedPlanDayIndex)
      : tripState?.published_plan
        ? publishedPlanRouteNotice(
            tripState.published_plan,
            selectedPlanDayIndex,
          )
        : routeAvailabilityNotice(
            formalMapUpdate.routes?.length ?? 0,
            tripState?.provider_display?.routes ?? [],
          )
    : null;
  const taskBookCityName = tripState?.task_book?.city
    ? getSupportedCity(tripState.task_book.city).name
    : tripState?.task_book?.city_id
      ? getRegisteredCityName(tripState.task_book.city_id)
      : null;
  const formalPlanCityName = tripState?.published_plan
    ? (getRegisteredCityName(tripState.published_plan.schedule.city_id) ??
      taskBookCityName ??
      "本次旅行")
    : null;
  const formalPlanStatus = useMemo(
    () =>
      tripState?.published_plan
        ? generationState === "generating"
          ? {
              label: "正在更新，当前展示上一版",
              tone: "working" as const,
            }
          : tripState.phase === "confirmed"
            ? { label: "已确认", tone: "confirmed" as const }
            : tripState.phase === "revising"
              ? {
                  label: "修改中，当前展示已发布版本",
                  tone: "working" as const,
                }
              : { label: "行程草案", tone: "stable" as const }
        : undefined,
    [tripState, generationState],
  );
  const formalPlanPresentation = useMemo(
    () =>
      v4VisiblePlan
        ? buildV4PublishedPlanPresentation(
            v4VisiblePlan,
            visibleV4PlanCityName(
              v4VisiblePlan,
              v4Runtime.tripState,
              v4ConversationMessages,
            ),
            generationState === "generating"
              ? "正在更新，当前展示上一版"
              : historicalV4Plan
                ? "上一版 · 待更新"
                : "",
            planPlacePreviews,
            planPreviews.weather,
          )
        : tripState && formalPlanCityName
          ? buildPublishedPlanPresentation(
              tripState,
              formalPlanCityName,
              formalPlanStatus,
            )
          : null,
    [
      v4VisiblePlan,
      v4Runtime.tripState,
      v4ConversationMessages,
      generationState,
      historicalV4Plan,
      planPlacePreviews,
      planPreviews.weather,
      tripState,
      formalPlanCityName,
      formalPlanStatus,
    ],
  );

  const displayedPlanVersion = useRef<string | undefined>();
  useEffect(() => {
    const version =
      v4VisiblePlan?.plan_version_id ??
      tripState?.published_plan?.plan_version_id;
    if (version === displayedPlanVersion.current) return;
    displayedPlanVersion.current = version;
    setSelectedPlanDayIndex(0);
    setMapFocusedDayIndex(undefined);
    setHighlightedMapPlaceId(null);
  }, [
    tripState?.published_plan?.plan_version_id,
    v4VisiblePlan?.plan_version_id,
  ]);

  const schedule = (callback: () => void, delay: number) => {
    const timer = window.setTimeout(() => {
      timers.current.delete(timer);
      callback();
    }, delay);
    timers.current.add(timer);
    return timer;
  };

  useEffect(() => {
    const scheduledTimers = timers.current;
    const localObjectUrls = objectUrls.current;
    return () => {
      scheduledTimers.forEach((timer) => window.clearTimeout(timer));
      scheduledTimers.clear();
      localObjectUrls.forEach((url) => URL.revokeObjectURL(url));
    };
  }, []);

  useEffect(() => {
    if (renderedTripId.current === shell.trip_id) return;
    renderedTripId.current = shell.trip_id;
    timers.current.forEach((timer) => window.clearTimeout(timer));
    timers.current.clear();
    if (generationTimer.current !== null) {
      window.clearTimeout(generationTimer.current);
      generationTimer.current = null;
    }
    objectUrls.current.forEach((url) => URL.revokeObjectURL(url));
    objectUrls.current = [];
    handledRemoteTerminalEvent.current = null;
    handledV4AttachmentTerminalEvent.current = null;
    v4ClientSequence.current = 0;
    setMessages(INITIAL_MESSAGES);
    setDraft("");
    setSubmittedTripTitle(null);
    setGenerationState("idle");
    setIsSending(false);
    setLocationOpen(false);
    setLocationQuery("");
    setLocationError(null);
    setIsLocating(false);
    setPendingAttachments([]);
    setReferenceFiles([]);
    setSelectedReference(null);
    setPendingAttachmentRequests({});
    setAttachmentConflicts({});
    setOpenServerTaskBook(null);
    setTaskBookConfirmRequestId(null);
    setTaskBookConfirmError(null);
    setHighlightedMapPlaceId(null);
  }, [choiceCount, demoMode, shell.trip_id]);

  useEffect(() => {
    if (runtimePersonalDefaults !== null) {
      setLongTermPreferences(runtimePersonalDefaults);
    }
  }, [runtimePersonalDefaults]);

  useEffect(() => {
    if (viewer.kind === "user" && viewer.personalDefaults) {
      setLongTermPreferences(viewer.personalDefaults);
    }
  }, [viewer]);

  useEffect(() => {
    if (
      isV4 ||
      taskBookConfirmRequestId === null ||
      runtimePresentation.latestErrorRequestId !== taskBookConfirmRequestId
    ) {
      return;
    }
    setTaskBookConfirmError(
      runtimePresentation.latestError?.snapshot_required
        ? "任务书已经变化，请刷新最新内容后再确认。"
        : (runtimePresentation.latestError?.message ??
            "任务书确认没有完成，请稍后重试。"),
    );
    setTaskBookConfirmRequestId(null);
  }, [
    runtimePresentation.latestError,
    runtimePresentation.latestErrorRequestId,
    isV4,
    taskBookConfirmRequestId,
  ]);

  useEffect(() => {
    if (tripState?.task_book?.status === "confirmed") {
      setTaskBookConfirmRequestId(null);
      setTaskBookConfirmError(null);
    }
  }, [tripState?.task_book?.status]);

  useEffect(() => {
    if (
      isV4 ||
      !remoteConversation ||
      runtimePresentation.generationStatus === null
    ) {
      return;
    }
    const status = runtimePresentation.generationStatus.status;
    if (status === "started" || status === "running") {
      setGenerationState("generating");
      setIsSending(false);
      return;
    }
    const terminalEventId = runtimePresentation.terminalEventId;
    if (
      terminalEventId === null ||
      handledRemoteTerminalEvent.current === terminalEventId
    ) {
      return;
    }
    handledRemoteTerminalEvent.current = terminalEventId;
    setGenerationState("idle");
    setIsSending(false);
    const replyAlreadyPublished = serverConversationMessages.some(
      (message) =>
        message.role === "assistant" &&
        message.generation_id === runtimePresentation.generationId,
    );
    if (
      status === "completed" &&
      runtimePresentation.streamText.trim() &&
      !replyAlreadyPublished
    ) {
      setMessages((current) => [
        ...current,
        {
          id: `remote-agent-${terminalEventId}`,
          role: "agent",
          content: runtimePresentation.streamText.trim(),
          status: "sent",
        },
      ]);
    } else if (status === "failed") {
      const serverFailure = runtimePresentation.latestError?.message?.trim();
      setMessages((current) => [
        ...current,
        {
          id: `remote-agent-error-${terminalEventId}`,
          role: "agent",
          content: serverFailure || AGENT_ERROR_REPLY,
          status: "failed",
        },
      ]);
    }
  }, [
    isV4,
    remoteConversation,
    runtimePresentation,
    serverConversationMessages,
  ]);

  useEffect(() => {
    if (!isV4 || !remoteConversation) return;
    if (backend.status === "failed") {
      setGenerationState("idle");
      setIsSending(false);
      setPendingAttachmentRequests({});
      return;
    }
    if (
      v4Runtime.cursor.currentGenerationId !== null ||
      v4Runtime.presentation.generationId !== null
    ) {
      setGenerationState("generating");
      setIsSending(false);
      return;
    }
    setGenerationState("idle");
    setIsSending(false);
    const terminal = v4Runtime.presentation.terminalEvent;
    if (
      terminal?.event_type !== "turn.failed" ||
      handledRemoteTerminalEvent.current === terminal.event_id
    ) {
      return;
    }
    handledRemoteTerminalEvent.current = terminal.event_id;
    setMessages((current) => [
      ...current,
      {
        id: `v4-agent-error-${terminal.event_id}`,
        role: "agent",
        content: AGENT_ERROR_REPLY,
        status: "failed",
      },
    ]);
  }, [backend.status, isV4, remoteConversation, v4Runtime]);

  useEffect(() => {
    if (
      !isV4 ||
      !remoteConversation ||
      v4Runtime.cursor.currentGenerationId !== null ||
      v4Runtime.presentation.generationId !== null
    ) {
      return;
    }
    const terminal = v4Runtime.presentation.terminalEvent;
    if (terminal === null) return;
    if (handledV4AttachmentTerminalEvent.current === terminal.event_id) return;
    handledV4AttachmentTerminalEvent.current = terminal.event_id;
    const pendingIds = Object.keys(pendingAttachmentRequests);
    if (pendingIds.length === 0) return;
    if (terminal.event_type === "turn.failed") {
      setAttachmentConflicts((current) => ({
        ...current,
        ...Object.fromEntries(
          pendingIds.map((attachmentId) => [
            attachmentId,
            "这次处理没有完成，你的选择仍保留；刷新后可以再次提交。",
          ]),
        ),
      }));
    } else if (terminal.event_type === "assistant.completed") {
      setAttachmentConflicts((current) =>
        Object.fromEntries(
          Object.entries(current).filter(([id]) => !pendingIds.includes(id)),
        ),
      );
    }
    setPendingAttachmentRequests({});
  }, [
    isV4,
    pendingAttachmentRequests,
    remoteConversation,
    v4Runtime.cursor.currentGenerationId,
    v4Runtime.presentation.generationId,
    v4Runtime.presentation.terminalEvent,
  ]);

  useEffect(() => {
    if (isV4) return;
    const requestId = runtimePresentation.latestErrorRequestId;
    if (!requestId || !runtimePresentation.latestError) return;
    const attachmentId = Object.entries(pendingAttachmentRequests).find(
      ([, pendingRequestId]) => pendingRequestId === requestId,
    )?.[0];
    if (!attachmentId) return;
    setAttachmentConflicts((current) => ({
      ...current,
      [attachmentId]: runtimePresentation.latestError?.snapshot_required
        ? "这项内容已变化，你的选择仍保留；刷新后可以再次确认。"
        : "这次确认没有成功，你的选择仍保留。",
    }));
    setPendingAttachmentRequests((current) => {
      const next = { ...current };
      delete next[attachmentId];
      return next;
    });
  }, [
    pendingAttachmentRequests,
    runtimePresentation.latestError,
    runtimePresentation.latestErrorRequestId,
    isV4,
  ]);

  useEffect(() => {
    if (isV4) return;
    const confirmedIds = new Set(
      serverConversationMessages.flatMap((message) =>
        (message.attachment_answers ?? []).map(
          (answer) => answer.attachment_id,
        ),
      ),
    );
    if (confirmedIds.size === 0) return;
    setPendingAttachmentRequests((current) =>
      Object.fromEntries(
        Object.entries(current).filter(([id]) => !confirmedIds.has(id)),
      ),
    );
    setAttachmentConflicts((current) =>
      Object.fromEntries(
        Object.entries(current).filter(([id]) => !confirmedIds.has(id)),
      ),
    );
  }, [isV4, serverConversationMessages]);

  useEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) {
      return;
    }

    textarea.style.height = "0px";
    textarea.style.height = `${Math.min(Math.max(textarea.scrollHeight, 52), 124)}px`;
  }, [draft]);

  const { showLatest, scrollToLatest, pauseFollowing } = useReplyScrollAnchor(
    conversationCanvasRef,
    shell.trip_id,
    !isRestoringTrip,
  );

  useEffect(() => {
    if (!menuOpen) {
      return;
    }

    const closeMenu = (event: MouseEvent | globalThis.KeyboardEvent) => {
      if (event instanceof globalThis.KeyboardEvent && event.key === "Escape") {
        if (tripsOpen) {
          setTripsOpen(false);
          menuDockRef.current
            ?.querySelector<HTMLButtonElement>('button[aria-label="我的行程"]')
            ?.focus();
          return;
        }
        setMenuOpen(false);
        menuButtonRef.current?.focus();
        return;
      }

      if (
        event instanceof MouseEvent &&
        !menuDockRef.current?.contains(event.target as Node) &&
        !menuButtonRef.current?.contains(event.target as Node)
      ) {
        setTripsOpen(false);
        setMenuOpen(false);
      }
    };

    document.addEventListener("mousedown", closeMenu);
    document.addEventListener("keydown", closeMenu);
    return () => {
      document.removeEventListener("mousedown", closeMenu);
      document.removeEventListener("keydown", closeMenu);
    };
  }, [menuOpen, tripsOpen]);

  useEffect(() => {
    if (!locationOpen) {
      return;
    }

    const closeLocationPicker = (
      event: MouseEvent | globalThis.KeyboardEvent,
    ) => {
      if (event instanceof globalThis.KeyboardEvent && event.key === "Escape") {
        setLocationOpen(false);
        locationButtonRef.current?.focus();
        return;
      }

      if (
        event instanceof MouseEvent &&
        !locationPopoverRef.current?.contains(event.target as Node) &&
        !locationButtonRef.current?.contains(event.target as Node)
      ) {
        setLocationOpen(false);
      }
    };

    document.addEventListener("mousedown", closeLocationPicker);
    document.addEventListener("keydown", closeLocationPicker);
    return () => {
      document.removeEventListener("mousedown", closeLocationPicker);
      document.removeEventListener("keydown", closeLocationPicker);
    };
  }, [locationOpen]);

  const nextMessageId = (prefix: string) => {
    messageCounter.current += 1;
    return `${prefix}-${messageCounter.current}`;
  };

  const finishGeneration = () => {
    generationTimer.current = null;
    setGenerationState("idle");

    if (
      demoMode === "generation-error" &&
      !didSimulateGenerationFailure.current
    ) {
      didSimulateGenerationFailure.current = true;
      setMessages((current) => [
        ...current,
        {
          id: nextMessageId("agent-error"),
          role: "agent",
          content: AGENT_ERROR_REPLY,
          status: "failed",
        },
      ]);
      return;
    }

    setMessages((current) => [
      ...current,
      {
        id: nextMessageId("agent"),
        role: "agent",
        content: "请连接服务后继续规划。",
        status: "sent",
      },
    ]);
  };

  const startGeneration = () => {
    setGenerationState("generating");
    generationTimer.current = schedule(finishGeneration, 1200);
  };

  const stopGeneration = () => {
    if (remoteConversation && isV4) {
      // The stop button appears immediately, before turn.accepted necessarily
      // arrives. Remember the click and use the server's generation ID once known.
      setV4StopRequestTripId(shell.trip_id);
      return;
    }
    if (remoteConversation) {
      if (
        activeGenerationId === null ||
        (isV4 ? v4Runtime.tripState === null : tripState === null)
      ) {
        return;
      }
      const requestId = crypto.randomUUID();
      const command: CancelGenerationCommand | V4CancelGenerationCommand = isV4
        ? {
            type: "cancel_generation",
            protocol_version: "v4",
            schema_version: "4.0.0",
            request_id: requestId,
            idempotency_key: `cancel-generation:${shell.trip_id}:${activeGenerationId}`,
            expected_state_version: v4Runtime.cursor.localStateVersion,
            client_sequence: ++v4ClientSequence.current,
            payload: { generation_id: activeGenerationId },
          }
        : {
            type: "cancel_generation",
            protocol_version: CURRENT_PROTOCOL_VERSION,
            schema_version: CURRENT_SCHEMA_VERSION,
            request_id: requestId,
            idempotency_key: `cancel-generation:${shell.trip_id}:${activeGenerationId}`,
            expected_state_version: tripState!.state_version,
            payload: { generation_id: activeGenerationId },
          };
      if (!backend.sendCommand(command)) {
        setMessages((current) => [
          ...current,
          {
            id: nextMessageId("agent-error"),
            role: "agent",
            content: "连接暂时不可用，没能停止这次回复，请稍后再试。",
            status: "failed",
          },
        ]);
      }
      return;
    }
    if (generationTimer.current !== null) {
      window.clearTimeout(generationTimer.current);
      timers.current.delete(generationTimer.current);
      generationTimer.current = null;
    }
    setGenerationState("idle");
    window.requestAnimationFrame(() => textareaRef.current?.focus());
  };

  useEffect(() => {
    if (v4StopRequestTripId === null) return;
    if (
      v4StopRequestTripId !== shell.trip_id ||
      !remoteConversation ||
      !isV4 ||
      generationState === "idle" ||
      backend.status !== "connected"
    ) {
      setV4StopRequestTripId(null);
      return;
    }
    if (activeGenerationId === null || v4Runtime.tripState === null) return;
    setV4StopRequestTripId(null);
    const command: V4CancelGenerationCommand = {
      type: "cancel_generation",
      protocol_version: "v4",
      schema_version: "4.0.0",
      request_id: crypto.randomUUID(),
      idempotency_key: `cancel-generation:${shell.trip_id}:${activeGenerationId}`,
      expected_state_version: v4Runtime.cursor.localStateVersion,
      client_sequence: ++v4ClientSequence.current,
      payload: { generation_id: activeGenerationId },
    };
    if (!backend.sendCommand(command)) {
      setMessages((current) => [
        ...current,
        {
          id: `agent-error-${crypto.randomUUID()}`,
          role: "agent",
          content: "连接暂时不可用，没能停止这次回复，请稍后再试。",
          status: "failed",
        },
      ]);
    }
  }, [
    v4StopRequestTripId,
    shell.trip_id,
    remoteConversation,
    isV4,
    generationState,
    backend,
    activeGenerationId,
    v4Runtime.tripState,
    v4Runtime.cursor.localStateVersion,
  ]);

  const completeSend = (messageId: string) => {
    if (demoMode === "send-error" && !didSimulateSendFailure.current) {
      didSimulateSendFailure.current = true;
      setMessages((current) => [
        ...current.map((message) =>
          message.id === messageId
            ? { ...message, status: "sent" as const }
            : message,
        ),
        {
          id: nextMessageId("agent-error"),
          role: "agent",
          content: AGENT_ERROR_REPLY,
          status: "failed",
        },
      ]);
      setIsSending(false);
      return;
    }

    setMessages((current) =>
      current.map((message) =>
        message.id === messageId ? { ...message, status: "sent" } : message,
      ),
    );
    setIsSending(false);
    startGeneration();
  };

  const sendContent = (
    content: string,
    attachments: ConversationAttachment[] = [],
  ): boolean => {
    const messageId = remoteConversation
      ? crypto.randomUUID()
      : nextMessageId("user");
    setMessages((current) => [
      ...current,
      {
        id: messageId,
        role: "user",
        content,
        status: "sending",
        attachments,
      },
    ]);
    const sentFiles = attachments.filter(
      (attachment): attachment is FileAttachment => attachment.kind === "file",
    );
    const restoreUnsentMessage = () => {
      setMessages((current) => [
        ...current.filter((message) => message.id !== messageId),
        {
          id: nextMessageId("agent-error"),
          role: "agent",
          content:
            "连接正在恢复，刚才的内容已保留在输入框，请在连接完成后再次发送。",
          status: "failed",
        },
      ]);
      setDraft(content);
      setPendingAttachments((current) => [
        ...attachments,
        ...current.filter(
          (currentAttachment) =>
            !attachments.some(
              (attachment) => attachment.id === currentAttachment.id,
            ),
        ),
      ]);
      setReferenceFiles((current) =>
        current.filter(
          (file) => !sentFiles.some((sentFile) => sentFile.id === file.id),
        ),
      );
      setIsSending(false);
    };
    if (sentFiles.length > 0) {
      setReferenceFiles((current) => [
        ...current,
        ...sentFiles.filter(
          (file) => !current.some((currentFile) => currentFile.id === file.id),
        ),
      ]);
    }
    setIsSending(true);
    if (remoteConversation) {
      if (isV4 ? v4Runtime.tripState === null : tripState === null) {
        restoreUnsentMessage();
        return false;
      }
      const requestId = crypto.randomUUID();
      const attachmentSummary = attachments
        .map((attachment) =>
          attachment.kind === "file"
            ? `附件：${attachment.name}`
            : `地点：${attachment.name}${attachment.detail ? `（${attachment.detail}）` : ""}`,
        )
        .join("；");
      const commandText = [content.trim(), attachmentSummary]
        .filter(Boolean)
        .join("\n");
      const command: UserMessageCommand | V4UserMessageCommand = isV4
        ? {
            type: "user_message",
            protocol_version: "v4",
            schema_version: "4.0.0",
            request_id: requestId,
            idempotency_key: `user-message:${shell.trip_id}:${requestId}`,
            expected_state_version: v4Runtime.cursor.localStateVersion,
            client_sequence: ++v4ClientSequence.current,
            payload: {
              message_id: messageId,
              text: commandText,
            },
          }
        : {
            type: "user_message",
            protocol_version: CURRENT_PROTOCOL_VERSION,
            schema_version: CURRENT_SCHEMA_VERSION,
            request_id: requestId,
            idempotency_key: `user-message:${shell.trip_id}:${requestId}`,
            expected_state_version: tripState!.state_version,
            payload: {
              message_id: messageId,
              text: commandText,
            },
          };
      const sent = backend.sendCommand(command);
      if (sent) {
        setMessages((current) =>
          current.map((message) =>
            message.id === messageId ? { ...message, status: "sent" } : message,
          ),
        );
        setGenerationState("generating");
      } else {
        restoreUnsentMessage();
      }
      setIsSending(false);
      return sent;
    }
    schedule(() => completeSend(messageId), 260);
    return true;
  };

  const submitServerAttachment = (
    attachmentId: string,
    sourceMessageId: string,
    answer: AttachmentAnswer,
  ) => {
    if (isV4 || !remoteConversation || tripState === null) return;
    const requestId = crypto.randomUUID();
    const command: AttachmentAnswerCommand = {
      type: "attachment_answer",
      protocol_version: CURRENT_PROTOCOL_VERSION,
      schema_version: CURRENT_SCHEMA_VERSION,
      request_id: requestId,
      idempotency_key: `attachment-answer:${shell.trip_id}:${requestId}`,
      expected_state_version: tripState.state_version,
      payload: {
        attachment_id: attachmentId,
        source_message_id: sourceMessageId,
        answer,
      },
    };
    if (backend.sendCommand(command)) {
      setPendingAttachmentRequests((current) => ({
        ...current,
        [attachmentId]: requestId,
      }));
      setAttachmentConflicts((current) => {
        const next = { ...current };
        delete next[attachmentId];
        return next;
      });
    } else {
      setAttachmentConflicts((current) => ({
        ...current,
        [attachmentId]: "连接暂时不可用，你的选择仍保留。",
      }));
    }
  };

  const submitTripSetup = (fields: TripSetupFields): boolean => {
    if (
      !isV4 ||
      !remoteConversation ||
      v4Runtime.tripState === null ||
      generationState !== "idle"
    )
      return false;
    const requestId = crypto.randomUUID();
    const command: V4TripSetupCommand = {
      type: "trip_setup",
      protocol_version: "v4",
      schema_version: "4.0.0",
      request_id: requestId,
      idempotency_key: `trip-setup:${shell.trip_id}:${requestId}`,
      expected_state_version: v4Runtime.cursor.localStateVersion,
      client_sequence: ++v4ClientSequence.current,
      payload: { ...fields, message_id: crypto.randomUUID() },
    };
    if (!backend.sendCommand(command)) return false;
    const cityName =
      getRegisteredCityName(fields.city_id) ??
      destinationCities
        .find((city) => city.cityId === fields.city_id)
        ?.name.replace(/市$/, "");
    setSubmittedTripTitle({
      tripId: shell.trip_id,
      stateVersion: v4Runtime.cursor.localStateVersion,
      title: tripSummaryTitle(cityName, fields.start_date, fields.end_date),
    });
    setLoadingStartMessage("正在整理当地的特色玩法…");
    setGenerationState("generating");
    return true;
  };

  const submitV4CardAnswer = (
    attachmentId: string,
    answer: V4CardAnswerDraft,
  ): boolean => {
    if (!isV4 || !remoteConversation || v4Runtime.tripState === null) {
      return false;
    }
    const requestId = crypto.randomUUID();
    const command: V4CardAnswerCommand = {
      type: "card_answer",
      protocol_version: "v4",
      schema_version: "4.0.0",
      request_id: requestId,
      idempotency_key: `card-answer:${shell.trip_id}:${requestId}`,
      expected_state_version: v4Runtime.cursor.localStateVersion,
      client_sequence: ++v4ClientSequence.current,
      payload: {
        ...answer,
        answer_id: crypto.randomUUID(),
      },
    };
    if (!backend.sendCommand(command)) {
      setAttachmentConflicts((current) => ({
        ...current,
        [attachmentId]: "连接暂时不可用，你的选择仍保留。",
      }));
      return false;
    }
    setPendingAttachmentRequests((current) => ({
      ...current,
      [attachmentId]: requestId,
    }));
    setAttachmentConflicts((current) => {
      const next = { ...current };
      delete next[attachmentId];
      return next;
    });
    setLoadingStartMessage(
      v4Runtime.pendingInteraction?.section === "attraction_preference"
        ? "正在查找符合偏好的景点…"
        : v4Runtime.pendingInteraction?.section === "attraction_specific"
          ? "正在整理当地的特色风味…"
          : v4Runtime.pendingInteraction?.section === "dining_preference"
            ? "正在查找符合偏好的餐厅…"
            : undefined,
    );
    setGenerationState("generating");
    return true;
  };

  const retryV4Interaction = (interactionId: string): void => {
    if (
      !isV4 ||
      !remoteConversation ||
      generationState === "generating" ||
      v4Runtime.pendingInteraction?.interaction_id !== interactionId ||
      !v4Runtime.pendingInteraction.recovery ||
      pendingAttachmentRequests[interactionId]
    ) {
      return;
    }
    const requestId = crypto.randomUUID();
    const command: V4RetryInteractionCommand = {
      type: "retry_interaction",
      protocol_version: "v4",
      schema_version: "4.0.0",
      request_id: requestId,
      idempotency_key: `retry-interaction:${shell.trip_id}:${requestId}`,
      expected_state_version: v4Runtime.cursor.localStateVersion,
      client_sequence: ++v4ClientSequence.current,
      payload: { interaction_id: interactionId },
    };
    if (!backend.sendCommand(command)) {
      setAttachmentConflicts((current) => ({
        ...current,
        [interactionId]: "连接暂时不可用，进度和重试入口仍保留。",
      }));
      return;
    }
    setPendingAttachmentRequests((current) => ({
      ...current,
      [interactionId]: requestId,
    }));
    setAttachmentConflicts((current) => {
      const next = { ...current };
      delete next[interactionId];
      return next;
    });
    setGenerationState("generating");
  };

  const confirmV4TaskBook = (
    attachmentId: string,
    taskBook: TaskBookV4,
  ): boolean => {
    if (!isV4 || !remoteConversation || v4Runtime.tripState === null) {
      return false;
    }
    const requestId = crypto.randomUUID();
    const command: V4TaskBookConfirmationCommand = {
      type: "task_book_confirmation",
      protocol_version: "v4",
      schema_version: "4.0.0",
      request_id: requestId,
      idempotency_key: `task-book-confirmation:${shell.trip_id}:${requestId}`,
      expected_state_version: v4Runtime.cursor.localStateVersion,
      client_sequence: ++v4ClientSequence.current,
      payload: {
        task_book_id: taskBook.task_book_id,
        task_book_version: taskBook.version,
      },
    };
    if (!backend.sendCommand(command)) {
      setAttachmentConflicts((current) => ({
        ...current,
        [attachmentId]: "连接暂时不可用，任务书尚未确认。",
      }));
      return false;
    }
    setPendingAttachmentRequests((current) => ({
      ...current,
      [attachmentId]: requestId,
    }));
    setAttachmentConflicts((current) => {
      const next = { ...current };
      delete next[attachmentId];
      return next;
    });
    setGenerationState("generating");
    return true;
  };

  const submitPlannerControl = (optionId?: string, userText?: string): void => {
    const workspace = v4Runtime.plannerWorkspace;
    if (
      !isV4 ||
      !remoteConversation ||
      !workspace ||
      generationState === "generating"
    )
      return;
    const requestId = crypto.randomUUID();
    const common = {
      protocol_version: "v4" as const,
      schema_version: "4.0.0" as const,
      request_id: requestId,
      idempotency_key: `planner-control:${requestId}`,
      expected_state_version: v4Runtime.cursor.localStateVersion,
      client_sequence: ++v4ClientSequence.current,
    };
    const payload = {
      generation_id: workspace.generation_id,
      expected_workspace_revision: workspace.workspace_revision,
    };
    let command: V4PlannerResumeCommand | V4PlannerAnswerCommand;
    if (optionId && workspace.active_interaction) {
      command = {
        ...common,
        type: "planner_answer",
        payload: {
          ...payload,
          option_id: optionId,
          answer_id: crypto.randomUUID(),
          interaction_id: workspace.active_interaction.interaction_id,
          resume_token: workspace.active_interaction.resume_token,
          optional_user_text: userText,
        },
      };
    } else {
      command = { ...common, type: "planner_resume", payload };
    }
    if (!backend.sendCommand(command)) {
      setAttachmentConflicts((current) => ({
        ...current,
        [workspace.generation_id]: "连接暂时不可用，已保存的规划工作仍保留。",
      }));
      return;
    }
    setPendingAttachmentRequests((current) => ({
      ...current,
      [workspace.generation_id]: requestId,
    }));
    setGenerationState("generating");
  };

  const selectPlanTransport = (
    legId: string,
    mode: "taxi" | "public_transit" | "walking",
  ) => {
    const workspace = v4Runtime.plannerWorkspace;
    if (!v4PublishedPlan || !workspace || generationState === "generating")
      return;
    const requestId = crypto.randomUUID();
    const command: V4PlanTransportSelectionCommand = {
      type: "plan_transport_selection",
      protocol_version: "v4",
      schema_version: "4.0.0",
      request_id: requestId,
      idempotency_key: `plan-transport:${requestId}`,
      expected_state_version: v4Runtime.cursor.localStateVersion,
      client_sequence: ++v4ClientSequence.current,
      payload: {
        generation_id: workspace.generation_id,
        expected_workspace_revision: workspace.workspace_revision,
        plan_version_id: v4PublishedPlan.plan_version_id,
        leg_id: legId,
        transport_mode: mode,
      },
    };
    if (backend.sendCommand(command)) setGenerationState("generating");
    else
      setAttachmentConflicts((current) => ({
        ...current,
        [workspace.generation_id]: "连接暂时不可用，交通方式尚未更改。",
      }));
  };

  const submitDraft = () => {
    const content = draft.trim();
    if (
      (!content && pendingAttachments.length === 0) ||
      isSending ||
      (remoteConversation && backend.status !== "connected")
    ) {
      return;
    }

    if (generationState === "generating" && !remoteConversation) {
      stopGeneration();
    }

    setDraft("");
    const attachments = pendingAttachments;
    setPendingAttachments([]);
    sendContent(content, attachments);
  };

  const handleChoiceSelect = (
    messageId: string,
    option: CompactChoiceOption,
  ) => {
    if (isSending) {
      return;
    }

    if (generationState === "generating") {
      stopGeneration();
    }

    setMessages((current) =>
      current.map((message) =>
        message.id === messageId
          ? {
              ...message,
              selectedChoiceId: option.id,
              choiceConfirmed: true,
            }
          : message,
      ),
    );
    sendContent(option.submitText ?? option.label);
  };

  const handleMultiChoiceToggle = (
    messageId: string,
    option: CompactChoiceOption,
  ) => {
    if (isSending) {
      return;
    }

    setMessages((current) =>
      current.map((message) => {
        if (message.id !== messageId || message.multiChoiceConfirmed) {
          return message;
        }
        const selectedIds = message.selectedChoiceIds ?? [];
        const exclusiveId = message.textMultiChoiceExclusiveId;
        if (option.id === exclusiveId) {
          return {
            ...message,
            selectedChoiceIds: selectedIds.includes(option.id)
              ? []
              : [option.id],
          };
        }
        const selectableIds = exclusiveId
          ? selectedIds.filter((id) => id !== exclusiveId)
          : selectedIds;
        return {
          ...message,
          selectedChoiceIds: selectableIds.includes(option.id)
            ? selectableIds.filter((id) => id !== option.id)
            : [...selectableIds, option.id],
        };
      }),
    );
  };

  const handleMultiChoiceConfirm = (messageId: string) => {
    if (isSending) {
      return;
    }

    const message = messages.find((candidate) => candidate.id === messageId);
    const selectedIds = message?.selectedChoiceIds ?? [];
    const selectedLabels = (
      message?.textMultiChoiceOptions ??
      message?.multiChoiceOptions ??
      []
    )
      .filter((option) => selectedIds.includes(option.id))
      .map((option) => option.label);
    if (selectedLabels.length === 0) {
      return;
    }

    if (generationState === "generating") {
      stopGeneration();
    }

    setMessages((current) =>
      current.map((candidate) =>
        candidate.id === messageId
          ? { ...candidate, multiChoiceConfirmed: true }
          : candidate,
      ),
    );
    sendContent(selectedLabels.join("、"));
  };

  const handleSliderChange = (messageId: string, value: number) => {
    if (isSending) {
      return;
    }

    setMessages((current) =>
      current.map((message) =>
        message.id === messageId && !message.sliderConfirmed
          ? { ...message, sliderValue: value, sliderFlexible: false }
          : message,
      ),
    );
  };

  const handleSliderConfirm = (messageId: string) => {
    if (isSending) {
      return;
    }

    const message = messages.find((candidate) => candidate.id === messageId);
    if (message?.sliderValue === undefined || message.sliderConfirmed) {
      return;
    }

    if (generationState === "generating") {
      stopGeneration();
    }

    setMessages((current) =>
      current.map((candidate) =>
        candidate.id === messageId
          ? {
              ...candidate,
              sliderFlexible: false,
              sliderConfirmed: true,
            }
          : candidate,
      ),
    );
    sendContent(sliderSubmitText(message.sliderValue));
  };

  const handleSliderFlexible = (messageId: string) => {
    if (isSending) {
      return;
    }

    if (generationState === "generating") {
      stopGeneration();
    }

    setMessages((current) =>
      current.map((candidate) =>
        candidate.id === messageId
          ? {
              ...candidate,
              sliderFlexible: true,
              sliderConfirmed: true,
            }
          : candidate,
      ),
    );
    sendContent("都可以，按整体行程灵活安排。");
  };

  const handleAttractionIntent = (
    messageId: string,
    itemId: string,
    intent: RecommendationIntent,
  ) => {
    if (isSending) {
      return;
    }

    setMessages((current) =>
      current.map((message) =>
        message.id === messageId && !message.attractionConfirmed
          ? {
              ...message,
              attractionValues: {
                ...message.attractionValues,
                [itemId]: intent,
              },
            }
          : message,
      ),
    );
  };

  const handleAttractionConfirm = (messageId: string) => {
    if (isSending) {
      return;
    }

    const message = messages.find((candidate) => candidate.id === messageId);
    if (!message?.attractionItems?.length || message.attractionConfirmed) {
      return;
    }

    if (generationState === "generating") {
      stopGeneration();
    }

    const intentOptions =
      message.attractionIntentOptions ?? ATTRACTION_INTENT_OPTIONS;
    const defaultIntent = message.attractionDefaultIntent ?? "if_convenient";
    const counts = new Map<RecommendationIntent, number>();
    message.attractionItems.forEach((item) => {
      const intent = message.attractionValues?.[item.id] ?? defaultIntent;
      counts.set(intent, (counts.get(intent) ?? 0) + 1);
    });
    const summary = intentOptions
      .map((option) => {
        const count = counts.get(option.value) ?? 0;
        return count ? `${count} 个${option.label}` : "";
      })
      .filter(Boolean)
      .join("，");

    setMessages((current) =>
      current.map((candidate) =>
        candidate.id === messageId
          ? { ...candidate, attractionConfirmed: true }
          : candidate,
      ),
    );
    sendContent(summary);
  };

  const handleAttachmentEdit = (
    messageId: string,
    kind: "choice" | "multi-choice" | "slider",
  ) => {
    if (isSending) {
      return;
    }

    setMessages((current) =>
      current.map((message) => {
        if (message.id !== messageId) {
          return message;
        }
        if (kind === "choice") {
          return { ...message, choiceConfirmed: false };
        }
        if (kind === "multi-choice") {
          return { ...message, multiChoiceConfirmed: false };
        }
        return { ...message, sliderConfirmed: false };
      }),
    );
  };

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    submitDraft();
  };

  const handleComposerKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      submitDraft();
    }
  };

  const addPendingAttachment = (attachment: ConversationAttachment) => {
    setPendingAttachments((current) => [...current, attachment]);
    setLocationError(null);
    setLocationOpen(false);
    window.requestAnimationFrame(() => textareaRef.current?.focus());
  };

  const removePendingAttachment = (attachmentId: string) => {
    setPendingAttachments((current) =>
      current.filter((attachment) => attachment.id !== attachmentId),
    );
  };

  const handleFileSelection = async (event: ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files ?? []);
    if (files.length === 0) {
      return;
    }

    const attachments = await Promise.all(
      files.map(async (file): Promise<FileAttachment> => {
        const url =
          typeof URL.createObjectURL === "function"
            ? URL.createObjectURL(file)
            : "";
        if (url) {
          objectUrls.current.push(url);
        }
        const isText = file.type === "text/plain" || /\.txt$/i.test(file.name);
        return {
          id: nextMessageId("file"),
          kind: "file",
          name: file.name,
          mimeType: file.type,
          size: file.size,
          url,
          textPreview: isText ? await file.text() : undefined,
        };
      }),
    );

    setPendingAttachments((current) => [...current, ...attachments]);
    event.target.value = "";
    window.requestAnimationFrame(() => textareaRef.current?.focus());
  };

  const addNamedLocation = () => {
    const name = locationQuery.trim();
    if (!name) {
      return;
    }

    addPendingAttachment({
      id: nextMessageId("location"),
      kind: "location",
      name,
      detail: "查找地点",
    });
    setLocationQuery("");
  };

  const addCurrentLocation = () => {
    if (!navigator.geolocation) {
      setLocationError("当前浏览器无法获取位置，请改用地点名称。 ");
      return;
    }

    setIsLocating(true);
    setLocationError(null);
    navigator.geolocation.getCurrentPosition(
      (position) => {
        const coordinates = {
          latitude: position.coords.latitude,
          longitude: position.coords.longitude,
        };
        addPendingAttachment({
          id: nextMessageId("location"),
          kind: "location",
          name: "我的位置",
          detail: `${coordinates.latitude.toFixed(5)}, ${coordinates.longitude.toFixed(5)}`,
          coordinates,
        });
        setIsLocating(false);
      },
      () => {
        setLocationError("没有获取到当前位置，请检查定位权限或改用地点名称。");
        setIsLocating(false);
      },
      { enableHighAccuracy: true, timeout: 10000, maximumAge: 60000 },
    );
  };

  const sharedLocations = messages.flatMap(
    (message) =>
      message.attachments?.filter(
        (attachment): attachment is LocationAttachment =>
          attachment.kind === "location",
      ) ?? [],
  );

  const closeMenu = () => {
    setTripsOpen(false);
    setMenuOpen(false);
  };
  const toggleTrips = () => {
    const next = !tripsOpen;
    setTripsOpen(next);
    if (next && viewer.kind === "user") {
      void viewerActions.refreshTripHistory();
    }
  };
  const selectTrip = async (selectedTripId: string, signal?: AbortSignal) => {
    if (selectedTripId === shell.trip_id) {
      closeMenu();
      return { ok: true } as const;
    }
    const result = await viewerActions.resumeTrip(selectedTripId, signal);
    if (signal?.aborted) return { ok: false, reason: "cancelled" } as const;
    if (result.ok) {
      closeMenu();
      navigate(
        pathWithAgentProtocol(
          `/trips/${selectedTripId}`,
          result.protocol ?? backend.requestedProtocol,
        ),
      );
    }
    return result;
  };
  const startNewTrip = async () => {
    const result = await viewerActions.startNewTrip();
    if (result.ok) {
      closeMenu();
      navigate(
        pathWithAgentProtocol(
          result.needsColdStart ? "/" : `/trips/${result.tripId}`,
          backend.requestedProtocol,
        ),
        result.needsColdStart ? { state: { startNewTrip: true } } : undefined,
      );
    }
    return result;
  };
  const openAccount = () => {
    closeMenu();
    setAccountOpen(true);
  };
  const openLongTermPreferences = () => {
    closeMenu();
    setLongTermPreferencesOpen(true);
  };
  const menuItems: DockItemData[] = [
    {
      icon: <MenuIcon name="trips" />,
      label: "我的行程",
      onClick: toggleTrips,
      active: tripsOpen,
      suppressLabel: tripsOpen,
    },
    {
      icon: <MenuIcon name="preferences" />,
      label: "长期偏好",
      onClick: openLongTermPreferences,
    },
    {
      icon: <MenuIcon name="help" />,
      label: "帮助",
      onClick: () => {
        closeMenu();
        menuButtonRef.current?.focus();
        setHelpOpen(true);
      },
    },
    {
      icon: <MenuIcon name="account" />,
      label: viewer.kind === "user" ? "我的账号" : "登录 / 注册",
      onClick: openAccount,
    },
  ];

  const handleTaskBookConfirm = () => {
    setMessages((current) => {
      if (current.some((message) => message.id === "task-book-confirmed")) {
        return current;
      }
      return [
        ...current,
        {
          id: "task-book-confirmed",
          role: "agent",
          content: "好，这版旅行任务书已经确认。",
          status: "sent",
        },
      ];
    });
  };

  const confirmServerTaskBook = () => {
    if (
      isV4 ||
      !remoteConversation ||
      tripState?.task_book?.status !== "pending"
    ) {
      return;
    }
    const requestId = crypto.randomUUID();
    const command: TaskBookConfirmCommand = {
      type: "task_book_confirm",
      protocol_version: CURRENT_PROTOCOL_VERSION,
      schema_version: CURRENT_SCHEMA_VERSION,
      request_id: requestId,
      idempotency_key: `task-book-confirm:${shell.trip_id}:${requestId}`,
      expected_state_version: tripState.state_version,
      payload: { trip_id: tripState.trip_id },
    };
    setTaskBookConfirmError(null);
    if (backend.sendCommand(command)) {
      setTaskBookConfirmRequestId(requestId);
    } else {
      setTaskBookConfirmError("连接暂时不可用，任务书没有确认。");
    }
  };

  const formalPlanRow = formalPlanPresentation ? (
    <ConversationEntrance
      as="article"
      motionKey={`plan:${v4VisiblePlan?.plan_version_id ?? tripState?.published_plan?.plan_version_id}`}
      data-conversation-attachment="true"
      key="formal-published-plan"
      className="conversation-message-row conversation-message-row-agent"
      data-formal-published-plan="true"
      data-conversation-reply={`reply:${v4VisiblePlan?.generation_id ?? tripState?.published_plan?.plan_version_id}`}
    >
      <div className="conversation-message-stack">
        <PlanReadyAttachment
          key={
            v4VisiblePlan?.plan_version_id ??
            tripState?.published_plan?.plan_version_id
          }
          presentation={formalPlanPresentation}
          selectedDayIndex={mapFocusedDayIndex}
          onOpenMap={() => setExpandedPlanMap(true)}
          onSelectTransport={selectPlanTransport}
          transportBusy={generationState === "generating"}
          readOnly={historicalV4Plan}
          highlightedPlaceId={highlightedMapPlaceId}
          onSelectDay={(dayIndex) => {
            setSelectedPlanDayIndex(dayIndex);
            setMapFocusedDayIndex(undefined);
            setHighlightedMapPlaceId(null);
          }}
          onHighlightPlace={setHighlightedMapPlaceId}
        />
      </div>
    </ConversationEntrance>
  ) : null;
  const planInsertionIndex = v4VisiblePlan
    ? v4PlanInsertionIndex(v4ConversationMessages, v4VisiblePlan)
    : -1;
  const v4Timeline = [...v4ConversationMessages] as (
    (typeof v4ConversationMessages)[number] | null
  )[];
  if (formalPlanRow && planInsertionIndex >= 0)
    v4Timeline.splice(planInsertionIndex, 0, null);
  const progressPlacement = positionAgentProgress(
    v4Runtime.presentation.agentProgress,
    v4ConversationMessages,
    generationState === "generating" ? activeGenerationId : null,
  );
  const plannerProgressActive =
    isV4 &&
    (progressPlacement.active.length > 0 ||
      v4Runtime.presentation.agentStatusCode?.startsWith("planner_") ||
      (activeGenerationId != null &&
        v4Runtime.plannerWorkspace?.generation_id === activeGenerationId));

  return (
    <div
      className="conversation-workspace"
      data-trip-id={tripId}
      data-backend-status={backend.status}
      data-agent-protocol={backend.protocol}
      data-requested-agent-protocol={backend.requestedProtocol}
      data-agent-status={
        isV4 ? (v4Runtime.presentation.agentStatusCode ?? "idle") : undefined
      }
      data-state-version={
        isV4 ? v4Runtime.cursor.localStateVersion : tripState?.state_version
      }
      data-active-generation-id={activeGenerationId ?? undefined}
      data-plan-ready={
        messages.some((message) => message.planReady) ||
        Boolean(tripState?.published_plan) ||
        Boolean(v4VisiblePlan) ||
        undefined
      }
    >
      <a className="skip-link" href="#conversation-main">
        跳到主要内容
      </a>
      <header className="conversation-header">
        <Link
          className="conversation-header-brand"
          to={homeHref}
          aria-label="返回首页"
        >
          <img src="/brand/iter-mark-black-64.png" alt="" />
        </Link>
        <h1>{conversationTitle}</h1>
        <button
          ref={menuButtonRef}
          className="conversation-menu"
          type="button"
          aria-label={menuOpen ? "关闭菜单" : "打开菜单"}
          aria-expanded={menuOpen}
          aria-haspopup="menu"
          onClick={() => {
            if (menuOpen) setTripsOpen(false);
            setMenuOpen((current) => !current);
          }}
        >
          <img src="/icons/menu-figma.svg" alt="" />
        </button>
        {menuOpen ? (
          <nav
            ref={menuDockRef}
            className="conversation-menu-dock"
            aria-label="旅行菜单"
          >
            <Dock
              items={menuItems}
              ariaLabel="旅行菜单快捷操作"
              baseItemSize={44}
            />
            {tripsOpen ? (
              <TripHistoryPopover
                viewer={viewer}
                currentTripId={shell.trip_id}
                onSelectTrip={selectTrip}
                onStartNewTrip={startNewTrip}
                onRetryHistory={viewerActions.refreshTripHistory}
                onSignIn={openAccount}
              />
            ) : null}
          </nav>
        ) : null}
      </header>

      <main className="conversation-main" id="conversation-main" tabIndex={-1}>
        <TripRestoreTiming timings={backend.restoreTimings} />
        {isRestoringTrip ? (
          <TripRestoreStatus
            failed={backend.status === "failed"}
            errorCode={backend.errorCode}
            onRetry={backend.retryConnection}
            onCancel={() => navigate(homeHref)}
          />
        ) : (
          <>
            <section className="conversation-column" aria-label="旅行对话">
              <ConversationMotionProvider
                key={shell.trip_id}
                enabled={!backend.historyLoading}
              >
                <div
                  ref={conversationCanvasRef}
                  className="conversation-canvas"
                  aria-live="polite"
                >
                  {isV4 &&
                  remoteConversation &&
                  v4Runtime.tripState !== null &&
                  !v4Runtime.tripState.semantic_state.trip_basics
                    ?.destination_canonical_id &&
                  !v4Runtime.tripState?.semantic_state.trip_basics
                    ?.start_date &&
                  !v4Runtime.pendingInteraction &&
                  backend.historyWindow?.before_state_version == null &&
                  v4ConversationMessages.every(
                    (message) => message.role === "system",
                  ) &&
                  !localConversationMessages.some(
                    (message) => message.role === "user",
                  ) &&
                  activeGenerationId === null &&
                  generationState === "idle" &&
                  !longTermPreferencesOpen ? (
                    <DestinationDateCard
                      key={shell.trip_id}
                      connected={backend.status === "connected"}
                      onSubmit={submitTripSetup}
                    />
                  ) : null}
                  {isV4 &&
                  backend.historyWindow?.before_state_version != null ? (
                    <div className="conversation-history-controls">
                      <button
                        type="button"
                        disabled={backend.historyLoading}
                        onClick={async () => {
                          pauseFollowing();
                          const canvas = conversationCanvasRef.current;
                          const height = canvas?.scrollHeight ?? 0;
                          const top = canvas?.scrollTop ?? 0;
                          await backend.loadOlderMessages();
                          requestAnimationFrame(() => {
                            if (
                              canvas &&
                              canvas === conversationCanvasRef.current
                            )
                              canvas.scrollTop =
                                top + canvas.scrollHeight - height;
                          });
                        }}
                      >
                        {backend.historyLoading
                          ? "正在加载更早消息…"
                          : "查看更早消息"}
                      </button>
                      {backend.historyError ? (
                        <p role="alert">{backend.historyError}</p>
                      ) : null}
                    </div>
                  ) : null}
                  {localConversationMessages
                    .filter(() => !remoteConversation)
                    .map((message) => {
                      const selectedChoice = [
                        ...(message.choiceOptions ?? []),
                        ...(message.detailedChoiceOptions ?? []),
                      ].find(
                        (option) => option.id === message.selectedChoiceId,
                      );
                      const selectedConstraintLabels = (
                        message.textMultiChoiceOptions ??
                        message.multiChoiceOptions ??
                        []
                      )
                        .filter((option) =>
                          (message.selectedChoiceIds ?? []).includes(option.id),
                        )
                        .map((option) => option.label);
                      const textMultiChoice = message.textMultiChoiceOptions
                        ?.length ? (
                        message.multiChoiceConfirmed ? (
                          <CompletedAttachmentSummary
                            label={
                              message.textMultiChoiceSummaryLabel ?? "已选择"
                            }
                            value={selectedConstraintLabels.join("、")}
                            disabled={isSending}
                            onEdit={() =>
                              handleAttachmentEdit(message.id, "multi-choice")
                            }
                          />
                        ) : (
                          <TextMultiChoiceAttachment
                            label={
                              message.textMultiChoiceLabel ?? "选择多个选项"
                            }
                            options={message.textMultiChoiceOptions}
                            selectedIds={message.selectedChoiceIds ?? []}
                            exclusiveOptionId={
                              message.textMultiChoiceExclusiveId
                            }
                            disabled={isSending}
                            onToggle={(option) =>
                              handleMultiChoiceToggle(message.id, option)
                            }
                            onConfirm={() =>
                              handleMultiChoiceConfirm(message.id)
                            }
                          />
                        )
                      ) : null;
                      const bubble = (
                        <div
                          className={`message-bubble message-bubble-${message.role}${
                            message.textMultiChoiceOptions?.length
                              ? " message-bubble-text-choice"
                              : ""
                          }`}
                          role={
                            message.status === "failed" ? "alert" : undefined
                          }
                        >
                          {message.content ? (
                            message.role === "user" ? (
                              <p>{message.content}</p>
                            ) : (
                              <AgentMarkdown text={message.content} />
                            )
                          ) : null}
                          {textMultiChoice}
                          {message.attachments?.length ? (
                            <div
                              className="message-attachments"
                              aria-label="消息附件"
                            >
                              {message.attachments.map((attachment) =>
                                attachment.kind === "file" ? (
                                  <button
                                    key={attachment.id}
                                    type="button"
                                    onClick={() =>
                                      setSelectedReference(attachment)
                                    }
                                  >
                                    <img
                                      src="/icons/attachment-figma.svg"
                                      alt=""
                                    />
                                    <span>
                                      <strong>{attachment.name}</strong>
                                      <small>
                                        {formatFileSize(attachment.size)}
                                      </small>
                                    </span>
                                  </button>
                                ) : (
                                  <div
                                    key={attachment.id}
                                    className="message-location-attachment"
                                  >
                                    <span aria-hidden="true">⌖</span>
                                    <span>
                                      <strong>{attachment.name}</strong>
                                      <small>{attachment.detail}</small>
                                    </span>
                                  </div>
                                ),
                              )}
                            </div>
                          ) : null}
                          {message.status === "sending" ? (
                            <span className="message-status">发送中…</span>
                          ) : null}
                        </div>
                      );

                      return (
                        <article
                          key={message.id}
                          data-conversation-reply={
                            message.role === "agent"
                              ? `local:${message.id}`
                              : undefined
                          }
                          className={`conversation-message-row conversation-message-row-${message.role}`}
                        >
                          {message.role === "agent" &&
                          (message.choiceOptions?.length ||
                            message.detailedChoiceOptions?.length ||
                            message.multiChoiceOptions?.length ||
                            message.sliderValue !== undefined ||
                            message.attractionItems?.length ||
                            message.recommendationEmpty ||
                            message.planReady ||
                            message.weatherDemo) ? (
                            <div className="conversation-message-stack">
                              {bubble}
                              {message.choiceOptions?.length ? (
                                message.choiceConfirmed && selectedChoice ? (
                                  <CompletedAttachmentSummary
                                    label={
                                      message.choiceSummaryLabel ?? "已选择"
                                    }
                                    value={selectedChoice.label}
                                    disabled={isSending}
                                    onEdit={() =>
                                      handleAttachmentEdit(message.id, "choice")
                                    }
                                  />
                                ) : (
                                  <CompactChoiceAttachment
                                    label={
                                      message.choiceLabel ?? "选择一个选项"
                                    }
                                    name={`choice-${message.id}`}
                                    options={message.choiceOptions}
                                    selectedId={message.selectedChoiceId}
                                    disabled={isSending}
                                    onSelect={(option) =>
                                      handleChoiceSelect(message.id, option)
                                    }
                                  />
                                )
                              ) : null}
                              {message.detailedChoiceOptions?.length ? (
                                message.choiceConfirmed && selectedChoice ? (
                                  <CompletedAttachmentSummary
                                    label={
                                      message.detailedChoiceSummaryLabel ??
                                      "已选择"
                                    }
                                    value={selectedChoice.label}
                                    disabled={isSending}
                                    onEdit={() =>
                                      handleAttachmentEdit(message.id, "choice")
                                    }
                                  />
                                ) : (
                                  <DetailedChoiceAttachment
                                    label={
                                      message.detailedChoiceLabel ??
                                      "选择一个详细选项"
                                    }
                                    name={`detailed-choice-${message.id}`}
                                    options={message.detailedChoiceOptions}
                                    selectedId={message.selectedChoiceId}
                                    disabled={isSending}
                                    onSelect={(option) =>
                                      handleChoiceSelect(message.id, option)
                                    }
                                  />
                                )
                              ) : null}
                              {message.multiChoiceOptions?.length ? (
                                message.multiChoiceConfirmed ? (
                                  <CompletedAttachmentSummary
                                    label={
                                      message.multiChoiceSummaryLabel ??
                                      "已选择"
                                    }
                                    value={selectedConstraintLabels.join("、")}
                                    disabled={isSending}
                                    onEdit={() =>
                                      handleAttachmentEdit(
                                        message.id,
                                        "multi-choice",
                                      )
                                    }
                                  />
                                ) : (
                                  <CompactMultiChoiceAttachment
                                    label={
                                      message.multiChoiceLabel ?? "选择多个选项"
                                    }
                                    options={message.multiChoiceOptions}
                                    selectedIds={
                                      message.selectedChoiceIds ?? []
                                    }
                                    disabled={isSending}
                                    onToggle={(option) =>
                                      handleMultiChoiceToggle(
                                        message.id,
                                        option,
                                      )
                                    }
                                    onConfirm={() =>
                                      handleMultiChoiceConfirm(message.id)
                                    }
                                  />
                                )
                              ) : null}
                              {message.sliderValue !== undefined ? (
                                message.sliderConfirmed ? (
                                  <CompletedAttachmentSummary
                                    label={message.sliderLabel ?? "偏好程度"}
                                    value={
                                      message.sliderFlexible
                                        ? "灵活安排"
                                        : describeSliderValue(
                                            message.sliderValue,
                                          )
                                    }
                                    disabled={isSending}
                                    onEdit={() =>
                                      handleAttachmentEdit(message.id, "slider")
                                    }
                                  />
                                ) : (
                                  <PreferenceSliderAttachment
                                    label={message.sliderLabel ?? "偏好程度"}
                                    startLabel={
                                      message.sliderStartLabel ?? "更少"
                                    }
                                    endLabel={message.sliderEndLabel ?? "更多"}
                                    value={message.sliderValue}
                                    valueText={describeSliderValue(
                                      message.sliderValue,
                                    )}
                                    flexibleSelected={message.sliderFlexible}
                                    disabled={isSending}
                                    onChange={(value) =>
                                      handleSliderChange(message.id, value)
                                    }
                                    onFlexible={() =>
                                      handleSliderFlexible(message.id)
                                    }
                                    onConfirm={() =>
                                      handleSliderConfirm(message.id)
                                    }
                                  />
                                )
                              ) : null}
                              {message.attractionItems?.length ? (
                                message.attractionConfirmed ? (
                                  <CompletedAttachmentSummary
                                    label={
                                      message.attractionSummaryLabel ??
                                      message.attractionLabel ??
                                      "推荐建议"
                                    }
                                    value="已记录这组候选的取舍"
                                    disabled={isSending}
                                    onEdit={() =>
                                      setMessages((current) =>
                                        current.map((candidate) =>
                                          candidate.id === message.id
                                            ? {
                                                ...candidate,
                                                attractionConfirmed: false,
                                              }
                                            : candidate,
                                        ),
                                      )
                                    }
                                  />
                                ) : recommendationGalleryMode(
                                    message.attractionItems.length,
                                  ) === "depth" ? (
                                  <AttractionDepthCarouselAttachment
                                    label={
                                      message.attractionLabel ?? "景点建议"
                                    }
                                    items={message.attractionItems}
                                    values={message.attractionValues ?? {}}
                                    itemNoun={message.attractionItemNoun}
                                    intentOptions={
                                      message.attractionIntentOptions
                                    }
                                    disabled={isSending}
                                    onChange={(itemId, intent) =>
                                      handleAttractionIntent(
                                        message.id,
                                        itemId,
                                        intent,
                                      )
                                    }
                                    onConfirm={() =>
                                      handleAttractionConfirm(message.id)
                                    }
                                  />
                                ) : (
                                  <AttractionAccordionAttachment
                                    label={
                                      message.attractionLabel ?? "景点建议"
                                    }
                                    items={message.attractionItems}
                                    values={message.attractionValues ?? {}}
                                    itemNoun={message.attractionItemNoun}
                                    intentOptions={
                                      message.attractionIntentOptions
                                    }
                                    disabled={isSending}
                                    onChange={(itemId, intent) =>
                                      handleAttractionIntent(
                                        message.id,
                                        itemId,
                                        intent,
                                      )
                                    }
                                    onConfirm={() =>
                                      handleAttractionConfirm(message.id)
                                    }
                                  />
                                )
                              ) : null}
                              {message.recommendationEmpty ? (
                                <RecommendationEmptyAttachment
                                  label={
                                    message.attractionLabel ?? "暂无合适候选"
                                  }
                                />
                              ) : null}
                              {message.planReady ? (
                                <PlanReadyAttachment
                                  weatherDays={message.planWeatherDays}
                                />
                              ) : null}
                            </div>
                          ) : (
                            bubble
                          )}
                        </article>
                      );
                    })}

                  {remoteConversation && !isV4
                    ? serverConversationMessages.map((message) => (
                        <ConversationEntrance
                          as="article"
                          motionKey={
                            message.role === "user"
                              ? `row:${message.message_id}`
                              : `row:reply:${message.generation_id}`
                          }
                          key={`server-${message.message_id}`}
                          data-conversation-reply={
                            message.role !== "user"
                              ? `reply:${message.generation_id}`
                              : undefined
                          }
                          className={`conversation-message-row conversation-message-row-${
                            message.role === "user" ? "user" : "agent"
                          }`}
                        >
                          <div className="conversation-message-stack">
                            {message.text ? (
                              <div
                                className={`message-bubble message-bubble-${
                                  message.role === "user" ? "user" : "agent"
                                }`}
                              >
                                {message.role === "user" ? (
                                  <p>{message.text}</p>
                                ) : (
                                  <ProgressiveAgentText
                                    text={message.text}
                                    replyKey={`reply:${message.generation_id}`}
                                  />
                                )}
                              </div>
                            ) : null}
                            {(message.attachments ?? []).map((attachment) => (
                              <ServerAttachmentRenderer
                                key={attachment.attachment_id}
                                attachment={attachment}
                                confirmedAnswer={(
                                  message.attachment_answers ?? []
                                ).find(
                                  (answer) =>
                                    answer.attachment_id ===
                                    attachment.attachment_id,
                                )}
                                disabled={Boolean(
                                  pendingAttachmentRequests[
                                    attachment.attachment_id
                                  ],
                                )}
                                conflictMessage={
                                  attachmentConflicts[attachment.attachment_id]
                                }
                                onSubmit={(answer) =>
                                  submitServerAttachment(
                                    attachment.attachment_id,
                                    message.message_id,
                                    answer,
                                  )
                                }
                                onReload={() => void backend.recover()}
                                onOpenTaskBook={(taskBookId) =>
                                  setOpenServerTaskBook({
                                    taskBookId,
                                    label:
                                      attachment.kind === "task_book_reference"
                                        ? attachment.label
                                        : "旅行任务书",
                                  })
                                }
                              />
                            ))}
                          </div>
                        </ConversationEntrance>
                      ))
                    : null}

                  {remoteConversation && isV4
                    ? v4Timeline.map((message) =>
                        message === null ? (
                          formalPlanRow
                        ) : (
                          <Fragment key={`v4-server-${message.message_id}`}>
                            <AgentProgressHistory
                              entries={
                                progressPlacement.beforeMessage.get(
                                  message.message_id,
                                ) ?? []
                              }
                            />
                            <ConversationEntrance
                              as="article"
                              motionKey={
                                message.role === "assistant"
                                  ? `row:reply:${message.generation_id}`
                                  : `row:${message.message_id}`
                              }
                              className={`conversation-message-row conversation-message-row-${
                                message.role === "user" ? "user" : "agent"
                              }`}
                              data-conversation-reply={
                                message.role === "system"
                                  ? `status:${message.message_id}`
                                  : message.role === "assistant"
                                    ? `reply:${message.generation_id}`
                                    : undefined
                              }
                              data-v4-message={message.message_id}
                              data-message-role={message.role}
                              data-generation-mode={message.generation_mode}
                              data-message-status={message.status}
                            >
                              <div className="conversation-message-stack">
                                {message.text ? (
                                  <div
                                    className={`message-bubble message-bubble-${
                                      message.role === "user" ? "user" : "agent"
                                    }`}
                                  >
                                    {message.role === "user" ? (
                                      <p
                                        style={{
                                          whiteSpace: "pre-wrap",
                                          overflowWrap: "anywhere",
                                        }}
                                      >
                                        {message.text}
                                      </p>
                                    ) : (
                                      <ProgressiveAgentText
                                        replyKey={
                                          message.role === "assistant"
                                            ? `reply:${message.generation_id}`
                                            : `status:${message.message_id}`
                                        }
                                        text={
                                          message.message_type === "plan"
                                            ? planCompletionCopy(message.text)
                                            : message.text
                                        }
                                      />
                                    )}
                                  </div>
                                ) : null}
                                {message.message_type === "plan" &&
                                !(
                                  v4VisiblePlan &&
                                  isPublishedPlanMessage(message, v4VisiblePlan)
                                ) ? (
                                  <V4HistoricalPlan
                                    message={message}
                                    deferred={
                                      backend.historyWindow?.deferred_attachment_message_ids?.includes(
                                        message.message_id,
                                      ) ?? false
                                    }
                                    tripState={v4Runtime.tripState}
                                    load={backend.getHistoryMessage}
                                  />
                                ) : null}
                                {(message.attachments ?? []).map(
                                  (attachment) => {
                                    if ("plan_version_id" in attachment)
                                      return null;
                                    const attachmentKey =
                                      "attachment_id" in attachment
                                        ? attachment.attachment_id
                                        : attachment.task_book_id;
                                    return (
                                      <ConversationEntrance
                                        key={attachmentKey}
                                        motionKey={`attachment:${attachmentKey}`}
                                        className="conversation-attachment-entry"
                                        data-conversation-attachment="true"
                                      >
                                        <V4DiscoveryAttachment
                                          attachment={attachment}
                                          onFocusOption={onDiscoveryFocus}
                                          loadIntroductions={
                                            backend.getPlaceIntroductions
                                          }
                                          authoritativeTaskBook={v4TaskBook}
                                          semanticState={
                                            v4Runtime.tripState
                                              ?.semantic_state ?? null
                                          }
                                          pendingInteraction={
                                            v4Runtime.pendingInteraction ?? null
                                          }
                                          disabled={Boolean(
                                            pendingAttachmentRequests[
                                              attachmentKey
                                            ],
                                          )}
                                          conflictMessage={
                                            attachmentConflicts[attachmentKey]
                                          }
                                          onSubmitCard={(answer) =>
                                            submitV4CardAnswer(
                                              attachmentKey,
                                              answer,
                                            )
                                          }
                                          onConfirmTaskBook={(taskBook) =>
                                            confirmV4TaskBook(
                                              attachmentKey,
                                              taskBook,
                                            )
                                          }
                                          onModifyTaskBook={() =>
                                            textareaRef.current?.focus()
                                          }
                                          onReload={() =>
                                            void backend.recover()
                                          }
                                        />
                                      </ConversationEntrance>
                                    );
                                  },
                                )}
                              </div>
                            </ConversationEntrance>
                          </Fragment>
                        ),
                      )
                    : null}

                  {remoteConversation
                    ? transientRemoteMessages.map((message) => (
                        <TransientRemoteMessageRow
                          key={`local-${message.id}`}
                          message={message}
                          onSelectReference={setSelectedReference}
                        />
                      ))
                    : null}

                  {remoteConversation &&
                  isV4 &&
                  (!v4PublishedPlan || generationState === "generating") ? (
                    <V4CardRecovery
                      pending={v4Runtime.pendingInteraction ?? null}
                      disabled={
                        generationState === "generating" ||
                        Boolean(
                          pendingAttachmentRequests[
                            v4Runtime.pendingInteraction?.interaction_id ?? ""
                          ],
                        )
                      }
                      conflictMessage={
                        attachmentConflicts[
                          v4Runtime.pendingInteraction?.interaction_id ?? ""
                        ]
                      }
                      onRetry={retryV4Interaction}
                      onReload={() => void backend.recover()}
                    />
                  ) : null}

                  <AgentProgressHistory entries={progressPlacement.unmatched} />
                  {showV4PlannerPanel ? (
                    <V4PlannerPanel
                      workspace={v4Runtime.plannerWorkspace ?? null}
                      generating={generationState === "generating"}
                      progressMessage={
                        v4Runtime.presentation.agentStatusCode?.startsWith(
                          "planner_",
                        )
                          ? v4Runtime.presentation.agentStatusMessage
                          : null
                      }
                      disabled={
                        generationState === "generating" ||
                        Boolean(
                          pendingAttachmentRequests[
                            v4Runtime.plannerWorkspace?.generation_id ?? ""
                          ],
                        )
                      }
                      conflictMessage={
                        attachmentConflicts[
                          v4Runtime.plannerWorkspace?.generation_id ?? ""
                        ]
                      }
                      onResume={() => submitPlannerControl()}
                      onAnswer={submitPlannerControl}
                      onReload={() => void backend.recover()}
                    />
                  ) : null}

                  {!isV4 ? formalPlanRow : null}

                  {remoteConversation &&
                  activeStreamText.trim() &&
                  !hasAssistantMessageForGeneration ? (
                    <ConversationEntrance
                      as="article"
                      motionKey={`row:reply:${activePresentationGenerationId}`}
                      className="conversation-message-row conversation-message-row-agent"
                      data-streaming-reply="true"
                      data-conversation-reply={`reply:${activePresentationGenerationId}`}
                      data-generation-id={
                        activePresentationGenerationId ?? undefined
                      }
                    >
                      <div className="message-bubble message-bubble-agent">
                        <ProgressiveAgentText
                          text={activeStreamText}
                          replyKey={`reply:${activePresentationGenerationId}`}
                        />
                      </div>
                    </ConversationEntrance>
                  ) : null}

                  {generationState === "generating" ? (
                    <AgentLoadingIndicator
                      key={activeGenerationId ?? "pending-generation"}
                      initialMessage={loadingStartMessage}
                      progressEntries={
                        plannerProgressActive
                          ? progressPlacement.active
                          : undefined
                      }
                      message={
                        plannerProgressActive
                          ? (v4Runtime.presentation.agentStatusMessage ??
                            "正在为你规划行程…")
                          : isV4 &&
                              isDiscoveryProgress(
                                v4Runtime.presentation.agentStatusCode,
                              )
                            ? (v4Runtime.presentation.agentStatusMessage ??
                              undefined)
                            : undefined
                      }
                    />
                  ) : null}

                  <div className="conversation-divider" aria-hidden="true" />
                </div>
              </ConversationMotionProvider>
              {showLatest ? (
                <div className="conversation-latest-control">
                  <button type="button" onClick={scrollToLatest}>
                    <span aria-hidden="true">↓</span>回到最新
                  </button>
                </div>
              ) : null}

              <form
                className={`conversation-composer${
                  pendingAttachments.length > 0
                    ? " conversation-composer-with-attachments"
                    : ""
                }`}
                onSubmit={handleSubmit}
              >
                <input
                  ref={fileInputRef}
                  className="composer-file-input"
                  type="file"
                  multiple
                  accept=".pdf,.doc,.docx,.txt,.png,.jpg,.jpeg,.heic,.webp,application/pdf,image/*,text/plain,application/msword,application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                  onChange={handleFileSelection}
                />
                {pendingAttachments.length > 0 ? (
                  <div className="composer-attachments" aria-label="待发送附件">
                    {pendingAttachments.map((attachment) => (
                      <span
                        key={attachment.id}
                        className="composer-attachment-chip"
                      >
                        <span>{attachment.name}</span>
                        <button
                          type="button"
                          aria-label={`移除 ${attachment.name}`}
                          onClick={() => removePendingAttachment(attachment.id)}
                        >
                          ×
                        </button>
                      </span>
                    ))}
                  </div>
                ) : null}
                <label
                  className="visually-hidden"
                  htmlFor="conversation-message"
                >
                  描述你的旅行想法
                </label>
                <textarea
                  ref={textareaRef}
                  id="conversation-message"
                  rows={2}
                  value={draft}
                  placeholder="说说你想去哪里，或想拥有怎样的旅程…"
                  onChange={(event) => setDraft(event.target.value)}
                  onKeyDown={handleComposerKeyDown}
                />
                {remoteConversation && backend.status !== "connected" ? (
                  <p className="composer-connection-status" role="status">
                    {backend.status === "failed"
                      ? backend.errorCode?.includes("session")
                        ? "临时会话已失效，无法继续提交。输入仍保留，请返回首页重新进入。"
                        : "连接未能恢复，输入仍保留。请刷新后重试。"
                      : backend.status === "connecting"
                        ? "正在连接，输入内容会保留。"
                        : "正在恢复连接，输入内容会保留。"}
                  </p>
                ) : null}
                {locationOpen ? (
                  <div
                    ref={locationPopoverRef}
                    className="location-picker"
                    role="dialog"
                    aria-label="添加地点"
                  >
                    <button
                      type="button"
                      className="location-current"
                      aria-label={
                        isLocating ? "正在获取我的位置" : "发送我的位置"
                      }
                      onClick={addCurrentLocation}
                      disabled={isLocating}
                    >
                      <span aria-hidden="true">⌖</span>
                      <span>
                        <strong>
                          {isLocating ? "正在定位…" : "发送我的位置"}
                        </strong>
                        <small>使用当前设备位置</small>
                      </span>
                    </button>
                    <div className="location-search">
                      <label htmlFor="location-query">查找地点</label>
                      <div>
                        <input
                          id="location-query"
                          type="search"
                          value={locationQuery}
                          placeholder="输入地点名称"
                          onChange={(event) =>
                            setLocationQuery(event.target.value)
                          }
                          onKeyDown={(event) => {
                            if (event.key === "Enter") {
                              event.preventDefault();
                              addNamedLocation();
                            }
                          }}
                        />
                        <button
                          type="button"
                          onClick={addNamedLocation}
                          disabled={!locationQuery.trim()}
                        >
                          添加
                        </button>
                      </div>
                    </div>
                    {locationError ? (
                      <p className="location-error" role="alert">
                        {locationError}
                      </p>
                    ) : null}
                  </div>
                ) : null}
                <div className="composer-toolbar">
                  <button
                    type="button"
                    className="composer-tool composer-attachment"
                    onClick={() => fileInputRef.current?.click()}
                  >
                    <img src="/icons/attachment-figma.svg" alt="" />
                    附件
                  </button>
                  <button
                    ref={locationButtonRef}
                    type="button"
                    className="composer-tool composer-place"
                    aria-expanded={locationOpen}
                    aria-haspopup="dialog"
                    onClick={() => {
                      setLocationError(null);
                      setLocationOpen((current) => !current);
                    }}
                  >
                    地点
                  </button>
                  {generationState === "generating" ? (
                    <button
                      type="button"
                      className="composer-send composer-stop"
                      aria-label="停止生成"
                      onClick={stopGeneration}
                    >
                      <span aria-hidden="true" />
                    </button>
                  ) : (
                    <button
                      type="submit"
                      className="composer-send"
                      aria-label="发送消息"
                      disabled={
                        (!draft.trim() && pendingAttachments.length === 0) ||
                        isSending ||
                        (remoteConversation && backend.status !== "connected")
                      }
                    >
                      <img src="/icons/send-figma.svg" alt="" />
                    </button>
                  )}
                </div>
              </form>
            </section>

            <aside className="conversation-sidebar" aria-label="地图与本次资料">
              <section
                className="route-map-section"
                aria-labelledby="route-map-title"
              >
                {!formalMapUpdate && !isV4 ? (
                  <h2 id="route-map-title">行程地图</h2>
                ) : null}
                {formalMapUpdate || isV4 ? (
                  <AmapSpaceBoard
                    update={showDiscoveryMap ? null : formalMapUpdate}
                    discovery={showDiscoveryMap ? discoveryPreview : undefined}
                    onExitDiscovery={
                      showDiscoveryMap && formalMapUpdate
                        ? () => setDiscoveryFocus(null)
                        : undefined
                    }
                    highlightedPlaceId={highlightedMapPlaceId}
                    onHighlightPlace={
                      showDiscoveryMap
                        ? (id) => {
                            if (
                              discoveryCard &&
                              "attachment_id" in discoveryCard
                            )
                              onDiscoveryFocus(discoveryCard.attachment_id, id);
                          }
                        : setHighlightedMapPlaceId
                    }
                    routeNotice={showDiscoveryMap ? null : mapRouteNotice}
                    days={
                      showDiscoveryMap
                        ? undefined
                        : formalPlanPresentation?.days
                    }
                    onSelectDay={(dayIndex) => {
                      setSelectedPlanDayIndex(dayIndex);
                      setMapFocusedDayIndex(dayIndex);
                      setHighlightedMapPlaceId(null);
                    }}
                    expanded={expandedPlanMap}
                    onExpandedChange={setExpandedPlanMap}
                  />
                ) : sharedLocations.length > 0 ? (
                  <div className="route-map-locations">
                    {sharedLocations.map((location) => (
                      <div key={location.id}>
                        <span aria-hidden="true">⌖</span>
                        <span>
                          <strong>{location.name}</strong>
                          <small>{location.detail}</small>
                        </span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="route-map-empty">
                    <p>地点会随着对话出现在这里</p>
                  </div>
                )}
              </section>

              <section
                className="reference-files-section"
                aria-labelledby="reference-files-title"
              >
                <header>
                  <h2 id="reference-files-title">本次资料</h2>
                  {referenceFiles.length > 0 ||
                  referenceLinks.length > 0 ||
                  demoMode === "task-book" ||
                  v4TaskBook ||
                  (!isV4 && tripState?.task_book) ? (
                    <span>
                      {referenceFiles.length +
                        referenceLinks.length +
                        (demoMode === "task-book" ||
                        v4TaskBook ||
                        (!isV4 && tripState?.task_book)
                          ? 1
                          : 0)}
                    </span>
                  ) : null}
                </header>

                {demoMode !== "task-book" && v4TaskBook ? (
                  <V4TaskBookPreview
                    taskBook={v4TaskBook}
                    semanticState={v4Runtime.tripState?.semantic_state ?? null}
                    disabled={
                      Boolean(
                        pendingAttachmentRequests[v4TaskBook.task_book_id],
                      ) || !v4TaskBookIsActive
                    }
                    conflictMessage={
                      attachmentConflicts[v4TaskBook.task_book_id]
                    }
                    onModify={() => textareaRef.current?.focus()}
                    onConfirm={() =>
                      confirmV4TaskBook(v4TaskBook.task_book_id, v4TaskBook)
                    }
                    onReload={() => void backend.recover()}
                  />
                ) : null}
                {demoMode !== "task-book" && !isV4 && tripState?.task_book ? (
                  <button
                    className="task-book-reference-link"
                    type="button"
                    aria-label={`查看${taskBookCityName ?? "当前"}旅行任务书`}
                    onClick={() =>
                      setOpenServerTaskBook({
                        taskBookId: `task-book:${tripState.trip_id}:${tripState.state_version}`,
                        label: `${taskBookCityName ?? "当前"}旅行任务书`,
                      })
                    }
                  >
                    <span>旅行任务书</span>
                    <small>
                      {tripState.task_book.status === "confirmed"
                        ? "已确认"
                        : "待确认"}
                    </small>
                  </button>
                ) : null}
                <TripReferenceLinks links={referenceLinks} />
                {referenceFiles.length > 0 ? (
                  <ul className="reference-file-list">
                    {referenceFiles.map((file) => (
                      <li key={file.id}>
                        <button
                          type="button"
                          aria-label={`预览 ${file.name}`}
                          onClick={() => setSelectedReference(file)}
                        >
                          <img src="/icons/attachment-figma.svg" alt="" />
                          <span>
                            <strong>{file.name}</strong>
                            <small>{formatFileSize(file.size)}</small>
                          </span>
                        </button>
                      </li>
                    ))}
                  </ul>
                ) : demoMode !== "task-book" &&
                  referenceLinks.length === 0 &&
                  !v4TaskBook &&
                  !tripState?.task_book ? (
                  <div className="reference-files-empty">
                    <p>文件和检索到的参考链接会显示在这里</p>
                  </div>
                ) : null}
              </section>
            </aside>
          </>
        )}
      </main>

      {selectedReference ? (
        <ReferencePreviewDialog
          attachment={selectedReference}
          onClose={() => setSelectedReference(null)}
        />
      ) : null}
      {openServerTaskBook ? (
        <ServerTaskBookPreview
          taskBookId={openServerTaskBook.taskBookId}
          label={openServerTaskBook.label}
          taskBook={tripState?.task_book ?? null}
          placeNames={
            new Map(
              (tripState?.candidate_places ?? []).map((place) => [
                place.place_id,
                place.name,
              ]),
            )
          }
          onClose={() => setOpenServerTaskBook(null)}
          onModify={() => textareaRef.current?.focus()}
          onConfirm={confirmServerTaskBook}
          confirmError={taskBookConfirmError}
        />
      ) : null}
      {helpOpen ? <HelpModal onClose={() => setHelpOpen(false)} /> : null}
      {longTermPreferencesOpen ? (
        <ColdStartModal
          mode={longTermPreferences ? "editing" : "onboarding"}
          memoryManager={
            viewer.kind !== "anonymous" && backend.mode !== "mock" ? (
              <MemoryManager key={viewer.account.user_id} />
            ) : undefined
          }
          initialSubmission={longTermPreferences}
          onClose={() => setLongTermPreferencesOpen(false)}
          onComplete={(submission) => {
            void viewerActions
              .updatePersonalDefaults(submission)
              .then((saved) => {
                if (!saved) return;
                setLongTermPreferences(submission);
                setLongTermPreferencesOpen(false);
              });
          }}
        />
      ) : null}
      {accountOpen &&
      (tripState || v4Runtime.tripState) &&
      viewer.kind === "anonymous" ? (
        <LoginOverlay
          onClose={() => setAccountOpen(false)}
          onSendCode={viewerActions.sendVerificationCode}
          onSubmit={async (phone, code) => {
            const cityName = tripState?.city
              ? getSupportedCity(tripState.city).name
              : (v4Runtime.tripState?.semantic_state.trip_basics
                  ?.destination_name ?? "未命名");
            const completed = await viewerActions.signInAndAttach({
              phone,
              code,
              shell,
              state: tripState ?? undefined,
              title: `${cityName}${tripState?.day_count ? ` ${tripState.day_count} 日` : ""}行程`,
              hasV4Work:
                backend.requestedProtocol === "v4" &&
                v4Runtime.tripState !== null,
            });
            if (completed) setAccountOpen(false);
            return completed;
          }}
        />
      ) : null}
      {accountOpen && viewer.kind === "user" ? (
        <AccountSessionOverlay
          userId={viewer.account.user_id}
          maskedPhone={viewer.account.masked_phone}
          nickname={viewer.account.nickname}
          onClose={() => setAccountOpen(false)}
          onUpdateNickname={viewerActions.updateNickname}
          onLogout={viewerActions.signOut}
        />
      ) : null}
    </div>
  );
}

function routeAvailabilityNotice(
  visibleRouteCount: number,
  routes: readonly { availability: string }[],
): string | null {
  const unavailableCount = routes.filter(
    (route) => route.availability === "missing",
  ).length;
  const partialCount = routes.filter(
    (route) => route.availability === "partial",
  ).length;
  if (visibleRouteCount === 0) {
    return "路线暂不可绘制，已保留本次行程地点，不会用直线或估算路线补齐。";
  }
  if (unavailableCount > 0 || partialCount > 0) {
    return "部分路线资料暂缺；地图只显示已经取得可靠几何数据的路线。";
  }
  return null;
}
