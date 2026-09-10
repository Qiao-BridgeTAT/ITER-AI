import Ajv2020, { type ValidateFunction } from "ajv/dist/2020.js";

import {
  PUBLIC_SCHEMAS,
  type PublicContractName,
} from "../generated/schemaRegistry";

export type { PublicContractName } from "../generated/schemaRegistry";

export interface PublicContractValidationContext {
  today?: string;
}

export type PublicContractValidationResult =
  | { success: true; errors: readonly [] }
  | { success: false; errors: readonly string[] };

interface DateRangeRule {
  startField: string;
  endField: string;
  minimumDays: number;
  maximumDays: number;
  minimumStartOffsetDays: number;
  maximumStartYears: number;
}

interface TimeRangeRule {
  startField: string;
  endField: string;
}

interface EventsWithinDateRangeRule {
  dateSelectionField: string;
  startField: string;
  endField: string;
  eventsField: string;
  eventDateField: string;
}

interface UniqueByRule {
  arrayField: string;
  itemField: string;
}

interface DayPlanRule {
  entriesField: string;
  legsField: string;
}

interface CostTotalRule {
  itemsField: string;
  amountField: string;
  totalField: string;
}

interface ItineraryRule {
  startField: string;
  endField: string;
  daysField: string;
  dayDateField: string;
}

interface ItineraryResultRule {
  itineraryField: string;
  costField: string;
  daysField: string;
  dailyCostField: string;
  totalField: string;
}

interface LodgingPlanRule {
  startField: string;
  endField: string;
  nightCountField: string;
  anchorsField: string;
  clustersField: string;
  strategiesField: string;
  favoritesField: string;
  selectedHotelField: string;
}

interface TripStateRule {
  cityField: string;
  cityIdField: string;
  dateRangeField: string;
  dayCountField: string;
  nightCountField: string;
  phaseField: string;
  diningPreferencesField: string;
  strategiesField: string;
  selectedHotelField: string;
  taskBookField: string;
  itineraryField: string;
  costField: string;
  publishedPlanField: string;
  pendingPlanModificationField: string;
  cityRequiredPhases: string[];
  profileCompletePhases: string[];
  dateRequiredPhases: string[];
  taskAndPlanPhases: string[];
  planPhases: string[];
  visiblePlanPhases: string[];
  derivedPlanPhases: string[];
  coldStartPhase: string;
  tripSetupPhase: string;
  attractionSelectionPhase: string;
  lodgingSelectionPhase: string;
  taskReflectionPhase: string;
}

interface StateVersionIncrementRule {
  baseField: string;
  stateField: string;
  increment: number;
}

interface CityContentRule {
  createdField: string;
  updatedField: string;
  sourcesField: string;
  sourceIdField: string;
  assetsField: string;
  assetIdField: string;
  briefField: string;
  themesField: string;
  themeIdField: string;
  attractionsField: string;
  placeIdField: string;
}

interface ExceptionalReplayRule {
  scenariosField: string;
  scenarioIdField: string;
  kindField: string;
  eventsField: string;
  actionsField: string;
  commandFields: string[];
  requiredKinds: string[];
}

interface SelectionBoundsRule {
  itemsField: string;
  itemIdField: string;
  minimumField: string;
  maximumField: string;
  exclusiveField?: string;
}

interface SliderBoundsRule {
  minimumField: string;
  maximumField: string;
  stepField: string;
}

interface WeatherDaysRule {
  daysField: string;
  minimumField: string;
  maximumField: string;
}

interface ConversationMessageRule {
  messageIdField: string;
  textField: string;
  attachmentsField: string;
  attachmentAnswersField: string;
  attachmentIdField: string;
  sourceMessageIdField: string;
  stateVersionField: string;
  generationIdField: string;
  factsField: string;
  factIdField: string;
  attachmentFactIdsField: string;
  textFactIdsField: string;
}

interface RecommendationSetRule {
  domainField: string;
  itemsField: string;
  placeIdField: string;
  itemFactIdsField: string;
  attachmentFactIdsField: string;
}

interface SemanticChoiceRule {
  domainField: string;
  optionsField: string;
  exclusiveField: string;
  attachmentFactIdsField: string;
}

interface ProviderDisplayRule {
  sourcesField: string;
  placesField: string;
  routesField: string;
  factsField: string;
}

const ajv = new Ajv2020({
  allErrors: true,
  discriminator: true,
  passContext: true,
  strict: true,
});

ajv.addFormat("date", {
  type: "string",
  validate: (value: string) => parseIsoDate(value) !== null,
});
ajv.addFormat("time", {
  type: "string",
  validate: (value: string) => parseTime(value) !== null,
});
ajv.addFormat("uuid", {
  type: "string",
  validate: (value: string) =>
    /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(
      value,
    ),
});
ajv.addFormat("uri", {
  type: "string",
  validate: (value: string) => {
    try {
      const url = new URL(value);
      return url.protocol === "http:" || url.protocol === "https:";
    } catch {
      return false;
    }
  },
});
ajv.addFormat("date-time", {
  type: "string",
  validate: (value: string) =>
    /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$/.test(
      value,
    ) && !Number.isNaN(Date.parse(value)),
});

ajv.addKeyword({
  keyword: "x-travel-date-range",
  schemaType: "object",
  type: "object",
  errors: false,
  validate: validateDateRange,
});
ajv.addKeyword({
  keyword: "x-travel-time-range",
  schemaType: "object",
  type: "object",
  errors: false,
  validate: validateTimeRange,
});
ajv.addKeyword({
  keyword: "x-travel-events-within-date-range",
  schemaType: "object",
  type: "object",
  errors: false,
  validate: validateEventsWithinDateRange,
});
ajv.addKeyword({
  keyword: "x-travel-unique-by",
  schemaType: "object",
  type: "object",
  errors: false,
  validate: validateUniqueBy,
});
ajv.addKeyword({
  keyword: "x-travel-day-plan",
  schemaType: "object",
  type: "object",
  errors: false,
  validate: validateDayPlan,
});
ajv.addKeyword({
  keyword: "x-travel-cost-total",
  schemaType: "object",
  type: "object",
  errors: false,
  validate: validateCostTotal,
});
ajv.addKeyword({
  keyword: "x-travel-itinerary",
  schemaType: "object",
  type: "object",
  errors: false,
  validate: validateItinerary,
});
ajv.addKeyword({
  keyword: "x-travel-itinerary-result",
  schemaType: "object",
  type: "object",
  errors: false,
  validate: validateItineraryResult,
});
ajv.addKeyword({
  keyword: "x-travel-lodging-plan",
  schemaType: "object",
  type: "object",
  errors: false,
  validate: validateLodgingPlan,
});
ajv.addKeyword({
  keyword: "x-travel-trip-state",
  schemaType: "object",
  type: "object",
  errors: false,
  validate: validateTripState,
});
ajv.addKeyword({
  keyword: "x-travel-state-version-increment",
  schemaType: "object",
  type: "object",
  errors: false,
  validate: validateStateVersionIncrement,
});
ajv.addKeyword({
  keyword: "x-travel-city-content",
  schemaType: "object",
  type: "object",
  errors: false,
  validate: validateCityContent,
});
ajv.addKeyword({
  keyword: "x-travel-exceptional-replay",
  schemaType: "object",
  type: "object",
  errors: false,
  validate: validateExceptionalReplaySuite,
});
ajv.addKeyword({
  keyword: "x-travel-selection-bounds",
  schemaType: "object",
  type: "object",
  errors: false,
  validate: validateSelectionBounds,
});
ajv.addKeyword({
  keyword: "x-travel-slider-bounds",
  schemaType: "object",
  type: "object",
  errors: false,
  validate: validateSliderBounds,
});
ajv.addKeyword({
  keyword: "x-travel-weather-days",
  schemaType: "object",
  type: "object",
  errors: false,
  validate: validateWeatherDays,
});
ajv.addKeyword({
  keyword: "x-travel-conversation-message",
  schemaType: "object",
  type: "object",
  errors: false,
  validate: validateConversationMessage,
});
ajv.addKeyword({
  keyword: "x-travel-recommendation-set",
  schemaType: "object",
  type: "object",
  errors: false,
  validate: validateRecommendationSet,
});
ajv.addKeyword({
  keyword: "x-travel-semantic-choice",
  schemaType: "object",
  type: "object",
  errors: false,
  validate: validateSemanticChoice,
});
ajv.addKeyword({
  keyword: "x-travel-provider-display",
  schemaType: "object",
  type: "object",
  errors: false,
  validate: validateProviderDisplay,
});
ajv.addKeyword({
  keyword: "x-travel-map-update",
  schemaType: "boolean",
  type: "object",
  errors: false,
  validate: validateMapUpdate,
});
ajv.addKeyword({
  keyword: "x-travel-cost-estimation-request",
  schemaType: "boolean",
  type: "object",
  errors: false,
  validate: validateCostEstimationRequest,
});
ajv.addKeyword({
  keyword: "x-travel-trip-cost-estimate",
  schemaType: "boolean",
  type: "object",
  errors: false,
  validate: validateTripCostEstimate,
});
ajv.addKeyword({
  keyword: "x-travel-itinerary-validation-request",
  schemaType: "boolean",
  type: "object",
  errors: false,
  validate: validateItineraryValidationRequest,
});
ajv.addKeyword({
  keyword: "x-travel-itinerary-validation-result",
  schemaType: "boolean",
  type: "object",
  errors: false,
  validate: validateItineraryValidationResult,
});
ajv.addKeyword({
  keyword: "x-travel-itinerary-repair-request",
  schemaType: "boolean",
  type: "object",
  errors: false,
  validate: validateItineraryRepairRequest,
});
ajv.addKeyword({
  keyword: "x-travel-itinerary-repair-result",
  schemaType: "boolean",
  type: "object",
  errors: false,
  validate: validateItineraryRepairResult,
});
ajv.addKeyword({
  keyword: "x-travel-plan-publication-request",
  schemaType: "boolean",
  type: "object",
  errors: false,
  validate: validatePlanPublicationRequest,
});
ajv.addKeyword({
  keyword: "x-travel-published-plan",
  schemaType: "boolean",
  type: "object",
  errors: false,
  validate: validatePublishedPlan,
});
ajv.addKeyword({
  keyword: "x-travel-pending-plan-modification",
  schemaType: "boolean",
  type: "object",
  errors: false,
  validate: validatePendingPlanModification,
});

const validators = Object.fromEntries(
  Object.entries(PUBLIC_SCHEMAS).map(([name, schema]) => [
    name,
    ajv.compile(schema),
  ]),
) as Record<PublicContractName, ValidateFunction>;

export function validatePublicContract(
  name: PublicContractName,
  payload: unknown,
  context: PublicContractValidationContext = {},
): PublicContractValidationResult {
  const validator = validators[name];
  const success = validator.call(
    { today: context.today ?? destinationToday() },
    payload,
  ) as boolean;
  if (success) {
    return { success: true, errors: [] };
  }
  return {
    success: false,
    errors: (validator.errors ?? []).map(
      (error) =>
        `${error.instancePath || "/"} ${error.message ?? "is invalid"}`,
    ),
  };
}

function validateDateRange(
  this: PublicContractValidationContext,
  rule: DateRangeRule,
  payload: unknown,
): boolean {
  if (!isRecord(payload)) {
    return false;
  }
  const start = parseIsoDate(payload[rule.startField]);
  const end = parseIsoDate(payload[rule.endField]);
  const today = parseIsoDate(this.today ?? destinationToday());
  if (start === null || end === null || today === null) {
    return false;
  }

  const days = end.epochDay - start.epochDay + 1;
  const earliestStart = today.epochDay + rule.minimumStartOffsetDays;
  const latestStart = addYearsClamped(today, rule.maximumStartYears).epochDay;
  return (
    start.epochDay >= earliestStart &&
    start.epochDay <= latestStart &&
    end.epochDay >= start.epochDay &&
    days >= rule.minimumDays &&
    days <= rule.maximumDays
  );
}

function validateTimeRange(
  this: PublicContractValidationContext,
  rule: TimeRangeRule,
  payload: unknown,
): boolean {
  void this;
  if (!isRecord(payload)) {
    return false;
  }
  const startValue = payload[rule.startField];
  const endValue = payload[rule.endField];
  if (startValue == null && endValue == null) {
    return true;
  }
  const start = parseTime(startValue);
  const end = parseTime(endValue);
  return start !== null && end !== null && end > start;
}

function validateEventsWithinDateRange(
  this: PublicContractValidationContext,
  rule: EventsWithinDateRangeRule,
  payload: unknown,
): boolean {
  void this;
  if (!isRecord(payload)) {
    return false;
  }
  const selection = payload[rule.dateSelectionField];
  const rawEvents = payload[rule.eventsField];
  const events = rawEvents === undefined ? [] : rawEvents;
  if (!isRecord(selection) || !Array.isArray(events)) {
    return false;
  }
  const start = parseIsoDate(selection[rule.startField]);
  const end = parseIsoDate(selection[rule.endField]);
  if (start === null || end === null) {
    return false;
  }
  return events.every((event) => {
    if (!isRecord(event)) {
      return false;
    }
    const eventDate = parseIsoDate(event[rule.eventDateField]);
    return (
      eventDate !== null &&
      eventDate.epochDay >= start.epochDay &&
      eventDate.epochDay <= end.epochDay
    );
  });
}

function validateUniqueBy(
  this: PublicContractValidationContext,
  rule: UniqueByRule,
  payload: unknown,
): boolean {
  void this;
  if (!isRecord(payload)) {
    return false;
  }
  const items = payload[rule.arrayField];
  if (!Array.isArray(items)) {
    return false;
  }
  const values: unknown[] = [];
  for (const item of items) {
    if (!isRecord(item) || !(rule.itemField in item)) {
      return false;
    }
    values.push(item[rule.itemField]);
  }
  return new Set(values).size === values.length;
}

function validateSelectionBounds(
  this: PublicContractValidationContext,
  rule: SelectionBoundsRule,
  payload: unknown,
): boolean {
  void this;
  if (!isRecord(payload)) {
    return false;
  }
  const items = payload[rule.itemsField];
  const minimum = payload[rule.minimumField];
  const maximum = payload[rule.maximumField];
  if (
    !Array.isArray(items) ||
    typeof minimum !== "number" ||
    !Number.isInteger(minimum) ||
    typeof maximum !== "number" ||
    !Number.isInteger(maximum) ||
    minimum < 0 ||
    maximum < minimum ||
    maximum > items.length
  ) {
    return false;
  }
  const ids = items.map((item) =>
    isRecord(item) ? item[rule.itemIdField] : undefined,
  );
  if (
    !ids.every((value) => typeof value === "string") ||
    new Set(ids).size !== ids.length
  ) {
    return false;
  }
  if (rule.exclusiveField === undefined) {
    return true;
  }
  const exclusive = payload[rule.exclusiveField];
  return (
    exclusive == null ||
    (typeof exclusive === "string" && ids.includes(exclusive))
  );
}

function validateSliderBounds(
  this: PublicContractValidationContext,
  rule: SliderBoundsRule,
  payload: unknown,
): boolean {
  void this;
  if (!isRecord(payload)) {
    return false;
  }
  const minimum = payload[rule.minimumField];
  const maximum = payload[rule.maximumField];
  const step = payload[rule.stepField];
  return (
    typeof minimum === "number" &&
    Number.isFinite(minimum) &&
    typeof maximum === "number" &&
    Number.isFinite(maximum) &&
    typeof step === "number" &&
    Number.isFinite(step) &&
    maximum > minimum &&
    step > 0 &&
    step <= maximum - minimum
  );
}

function validateWeatherDays(
  this: PublicContractValidationContext,
  rule: WeatherDaysRule,
  payload: unknown,
): boolean {
  void this;
  if (!isRecord(payload)) {
    return false;
  }
  const days = payload[rule.daysField];
  if (!Array.isArray(days)) {
    return false;
  }
  return days.every((day) => {
    if (!isRecord(day)) {
      return false;
    }
    const minimum = day[rule.minimumField];
    const maximum = day[rule.maximumField];
    return (
      typeof minimum === "number" &&
      Number.isFinite(minimum) &&
      typeof maximum === "number" &&
      Number.isFinite(maximum) &&
      maximum >= minimum
    );
  });
}

function validateConversationMessage(
  this: PublicContractValidationContext,
  rule: ConversationMessageRule,
  payload: unknown,
): boolean {
  void this;
  if (!isRecord(payload)) {
    return false;
  }
  const messageId = payload[rule.messageIdField];
  const stateVersion = payload[rule.stateVersionField];
  const generationId = payload[rule.generationIdField];
  const attachments = payload[rule.attachmentsField];
  const attachmentAnswers = payload[rule.attachmentAnswersField] ?? [];
  const facts = payload[rule.factsField];
  const textFactIds = payload[rule.textFactIdsField];
  const text = payload[rule.textField];
  if (
    typeof messageId !== "string" ||
    typeof stateVersion !== "number" ||
    !Number.isInteger(stateVersion) ||
    !Array.isArray(attachments) ||
    !Array.isArray(attachmentAnswers) ||
    !Array.isArray(facts) ||
    !Array.isArray(textFactIds) ||
    (text == null && attachments.length === 0)
  ) {
    return false;
  }

  const attachmentIds: unknown[] = [];
  const referencedFactIds: unknown[] = [...textFactIds];
  for (const wrappedAttachment of attachments) {
    const attachment = unwrapRootModel(wrappedAttachment);
    if (
      attachment === null ||
      attachment[rule.sourceMessageIdField] !== messageId ||
      attachment[rule.stateVersionField] !== stateVersion ||
      attachment[rule.generationIdField] !== generationId
    ) {
      return false;
    }
    attachmentIds.push(attachment[rule.attachmentIdField]);
    const attachmentFactIds = attachment[rule.attachmentFactIdsField];
    if (!Array.isArray(attachmentFactIds)) {
      return false;
    }
    referencedFactIds.push(...attachmentFactIds);
  }

  const declaredFactIds = facts.map((fact) =>
    isRecord(fact) ? fact[rule.factIdField] : undefined,
  );
  if (
    attachmentIds.some((value) => typeof value !== "string") ||
    new Set(attachmentIds).size !== attachmentIds.length ||
    declaredFactIds.some((value) => typeof value !== "string") ||
    new Set(declaredFactIds).size !== declaredFactIds.length ||
    referencedFactIds.some((value) => typeof value !== "string")
  ) {
    return false;
  }
  const answeredAttachmentIds = attachmentAnswers.map((answer) =>
    isRecord(answer) ? answer[rule.attachmentIdField] : undefined,
  );
  if (
    answeredAttachmentIds.some((value) => typeof value !== "string") ||
    new Set(answeredAttachmentIds).size !== answeredAttachmentIds.length ||
    answeredAttachmentIds.some((value) => !attachmentIds.includes(value)) ||
    attachmentAnswers.some(
      (answer) =>
        !isRecord(answer) || answer[rule.stateVersionField] !== stateVersion,
    )
  ) {
    return false;
  }
  const declared = new Set(declaredFactIds);
  const referenced = new Set(referencedFactIds);
  return (
    declared.size === referenced.size &&
    [...referenced].every((factId) => declared.has(factId))
  );
}

function validateRecommendationSet(
  this: PublicContractValidationContext,
  rule: RecommendationSetRule,
  payload: unknown,
): boolean {
  void this;
  if (!isRecord(payload)) {
    return false;
  }
  const domain = payload[rule.domainField];
  if (domain !== "attraction" && domain !== "restaurant") {
    return true;
  }
  const items = payload[rule.itemsField];
  const attachmentFactIds = payload[rule.attachmentFactIdsField];
  if (!Array.isArray(items) || !Array.isArray(attachmentFactIds)) {
    return false;
  }
  const declaredFacts = new Set(attachmentFactIds);
  const placeIds = new Set<unknown>();
  for (const item of items) {
    if (!isRecord(item) || typeof item[rule.placeIdField] !== "string") {
      return false;
    }
    const placeId = item[rule.placeIdField];
    if (placeIds.has(placeId)) return false;
    placeIds.add(placeId);
    const itemFactIds = item[rule.itemFactIdsField];
    if (
      !Array.isArray(itemFactIds) ||
      itemFactIds.length === 0 ||
      new Set(itemFactIds).size !== itemFactIds.length ||
      itemFactIds.some((factId) => !declaredFacts.has(factId))
    ) {
      return false;
    }
    if (
      domain === "restaurant" &&
      (typeof item.match_reason !== "string" ||
        item.match_reason.trim().length === 0 ||
        typeof item.main_cost !== "string" ||
        item.main_cost.trim().length === 0)
    ) {
      return false;
    }
  }
  return true;
}

function validateSemanticChoice(
  this: PublicContractValidationContext,
  rule: SemanticChoiceRule,
  payload: unknown,
): boolean {
  void this;
  if (
    !isRecord(payload) ||
    payload[rule.domainField] === undefined ||
    payload[rule.domainField] === "generic"
  ) {
    return true;
  }
  const domain = payload[rule.domainField];
  if (domain !== "dining_direction" && domain !== "city_theme") return false;
  const options = payload[rule.optionsField];
  const exclusive = payload[rule.exclusiveField];
  const attachmentFacts = payload[rule.attachmentFactIdsField];
  if (
    !Array.isArray(options) ||
    !Array.isArray(attachmentFacts) ||
    typeof exclusive !== "string"
  ) {
    return false;
  }
  const declaredFacts = new Set(attachmentFacts);
  let openOptionId: string | null = null;
  for (const option of options) {
    if (!isRecord(option) || !isRecord(option.semantic_value)) return false;
    const semantic = option.semantic_value;
    if (semantic.domain !== domain) return false;
    if (semantic.kind === "open_to_any") {
      if (openOptionId !== null || typeof option.option_id !== "string")
        return false;
      openOptionId = option.option_id;
      continue;
    }
    const sourceFactIds = semantic.source_fact_ids;
    if (
      typeof semantic.value !== "string" ||
      semantic.value.trim().length === 0 ||
      !Array.isArray(sourceFactIds) ||
      sourceFactIds.length === 0 ||
      new Set(sourceFactIds).size !== sourceFactIds.length ||
      sourceFactIds.some((factId) => !declaredFacts.has(factId))
    ) {
      return false;
    }
    if (domain === "city_theme" && semantic.kind !== "theme") return false;
    if (domain === "city_theme" && semantic.place_id != null) return false;
    if (
      domain === "dining_direction" &&
      semantic.kind === "specific_restaurant" &&
      typeof semantic.place_id !== "string"
    ) {
      return false;
    }
  }
  return openOptionId === exclusive;
}

function validateProviderDisplay(
  this: PublicContractValidationContext,
  rule: ProviderDisplayRule,
  payload: unknown,
): boolean {
  void this;
  if (!isRecord(payload)) return false;
  const sources = payload[rule.sourcesField];
  const places = payload[rule.placesField];
  const routes = payload[rule.routesField];
  const facts = payload[rule.factsField];
  if (
    !Array.isArray(sources) ||
    !Array.isArray(places) ||
    !Array.isArray(routes) ||
    !Array.isArray(facts)
  ) {
    return false;
  }
  const sourceIds = uniqueFieldSet(sources, "fact_id");
  const placeIds = uniqueFieldSet(places, "place_id");
  const routeIds = uniqueFieldSet(routes, "route_id");
  const factIds = uniqueFieldSet(facts, "fact_id");
  if (!sourceIds || !placeIds || !routeIds || !factIds) return false;

  const referencesKnownSources = (item: unknown): boolean =>
    isRecord(item) &&
    Array.isArray(item.source_fact_ids) &&
    item.source_fact_ids.every((id) => sourceIds.has(id));
  if (![...places, ...routes, ...facts].every(referencesKnownSources)) {
    return false;
  }
  if (
    routes.some(
      (route) =>
        !isRecord(route) ||
        !placeIds.has(route.from_place_id) ||
        !placeIds.has(route.to_place_id),
    ) ||
    facts.some(
      (fact) =>
        !isRecord(fact) ||
        (fact.place_id != null && !placeIds.has(fact.place_id)),
    )
  ) {
    return false;
  }

  return (
    places.every((place) => validateProviderAvailability(place, "place")) &&
    routes.every((route) => validateProviderAvailability(route, "route")) &&
    facts.every((fact) => validateProviderAvailability(fact, "fact"))
  );
}

export function validateMapUpdate(
  this: PublicContractValidationContext,
  enabled: boolean,
  payload: unknown,
): boolean {
  void this;
  if (!enabled) return true;
  if (!isRecord(payload)) return false;
  const markers = payload.markers;
  const routes = payload.routes;
  if (!Array.isArray(markers) || !Array.isArray(routes)) return false;

  const markerIds = markers.map((marker) =>
    isRecord(marker) ? marker.place_id : undefined,
  );
  if (
    markerIds.some((placeId) => typeof placeId !== "string") ||
    new Set(markerIds).size !== markerIds.length
  ) {
    return false;
  }
  const visiblePlaceIds = new Set(markerIds);
  const routeKeys = routes.map((route) =>
    isRecord(route)
      ? `${String(route.from_place_id)}:${String(route.to_place_id)}:${String(route.mode)}`
      : undefined,
  );
  return (
    routeKeys.every((key) => typeof key === "string") &&
    new Set(routeKeys).size === routeKeys.length &&
    routes.every(
      (route) =>
        isRecord(route) &&
        visiblePlaceIds.has(route.from_place_id) &&
        visiblePlaceIds.has(route.to_place_id),
    )
  );
}

function validateProviderAvailability(
  value: unknown,
  kind: "place" | "route" | "fact",
): boolean {
  if (!isRecord(value) || !Array.isArray(value.source_fact_ids)) return false;
  const availability = value.availability;
  const hasSources = value.source_fact_ids.length > 0;
  const hasReason = typeof value.missing_reason === "string";
  if (availability === "available") {
    return hasSources && !hasReason;
  }
  if (availability === "partial") {
    return hasReason && (kind === "place" || hasSources);
  }
  if (availability !== "missing" || !hasReason) return false;
  if (kind === "route") {
    return (
      value.distance_m == null &&
      value.duration_minutes == null &&
      value.walking_m == null &&
      value.fare == null &&
      Array.isArray(value.polyline) &&
      value.polyline.length === 0
    );
  }
  if (kind === "fact") {
    const hasValue =
      value.display_text != null ||
      value.amount != null ||
      value.minimum_celsius != null ||
      value.maximum_celsius != null;
    const hasAnyTrace =
      hasSources || value.provider != null || value.fetched_at != null;
    const hasCompleteTrace =
      hasSources && value.provider != null && value.fetched_at != null;
    return !hasValue && (!hasAnyTrace || hasCompleteTrace);
  }
  return true;
}

function unwrapRootModel(value: unknown): Record<string, unknown> | null {
  if (!isRecord(value)) {
    return null;
  }
  return isRecord(value.root) ? value.root : value;
}

function validateDayPlan(
  this: PublicContractValidationContext,
  rule: DayPlanRule,
  payload: unknown,
): boolean {
  void this;
  if (!isRecord(payload)) {
    return false;
  }
  const entries = payload[rule.entriesField];
  const legs = payload[rule.legsField];
  if (!Array.isArray(entries) || !Array.isArray(legs) || entries.length === 0) {
    return false;
  }
  const entryIds = entries.map((entry) =>
    isRecord(entry) ? entry.entry_id : undefined,
  );
  const legIds = legs.map((leg) => (isRecord(leg) ? leg.leg_id : undefined));
  if (
    entryIds.some((value) => typeof value !== "string") ||
    legIds.some((value) => typeof value !== "string") ||
    new Set(entryIds).size !== entryIds.length ||
    new Set(legIds).size !== legIds.length ||
    legs.length !== Math.max(0, entries.length - 1)
  ) {
    return false;
  }
  for (let index = 0; index < entries.length; index += 1) {
    const entry = entries[index];
    if (!isRecord(entry)) {
      return false;
    }
    const start = parseTime(entry.start_time);
    const end = parseTime(entry.end_time);
    if (start === null || end === null || end <= start) {
      return false;
    }
    if (index > 0) {
      const previous = entries[index - 1];
      if (!isRecord(previous)) {
        return false;
      }
      const previousEnd = parseTime(previous.end_time);
      if (previousEnd === null || start < previousEnd) {
        return false;
      }
      const leg = legs[index - 1];
      if (
        !isRecord(leg) ||
        leg.from_entry_id !== previous.entry_id ||
        leg.to_entry_id !== entry.entry_id
      ) {
        return false;
      }
    }
  }
  return true;
}

function validateCostTotal(
  this: PublicContractValidationContext,
  rule: CostTotalRule,
  payload: unknown,
): boolean {
  void this;
  if (!isRecord(payload)) {
    return false;
  }
  const items = payload[rule.itemsField];
  const total = parseAmountRange(payload[rule.totalField]);
  if (!Array.isArray(items) || total === null) {
    return false;
  }
  let minimum = 0;
  let maximum = 0;
  for (const item of items) {
    if (!isRecord(item)) {
      return false;
    }
    const amountValue = item[rule.amountField];
    if (amountValue == null) {
      continue;
    }
    const amount = parseAmountRange(amountValue);
    if (amount === null) {
      return false;
    }
    minimum += amount.minimum;
    maximum += amount.maximum;
  }
  return minimum === total.minimum && maximum === total.maximum;
}

function validateItinerary(
  this: PublicContractValidationContext,
  rule: ItineraryRule,
  payload: unknown,
): boolean {
  void this;
  if (!isRecord(payload)) {
    return false;
  }
  const start = parseIsoDate(payload[rule.startField]);
  const end = parseIsoDate(payload[rule.endField]);
  const days = payload[rule.daysField];
  if (start === null || end === null || !Array.isArray(days)) {
    return false;
  }
  const dayCount = end.epochDay - start.epochDay + 1;
  if (dayCount < 1 || dayCount > 5 || days.length !== dayCount) {
    return false;
  }
  return days.every((day, index) => {
    if (!isRecord(day)) {
      return false;
    }
    const dayDate = parseIsoDate(day[rule.dayDateField]);
    return dayDate !== null && dayDate.epochDay === start.epochDay + index;
  });
}

function validateItineraryResult(
  this: PublicContractValidationContext,
  rule: ItineraryResultRule,
  payload: unknown,
): boolean {
  void this;
  if (!isRecord(payload)) {
    return false;
  }
  const itinerary = payload[rule.itineraryField];
  const cost = payload[rule.costField];
  if (!isRecord(itinerary) || !isRecord(cost)) {
    return false;
  }
  const days = itinerary[rule.daysField];
  const total = parseAmountRange(cost[rule.totalField]);
  if (!Array.isArray(days) || total === null) {
    return false;
  }
  const daily = sumDailyCosts(days, rule.dailyCostField);
  return (
    daily !== null &&
    daily.minimum === total.minimum &&
    daily.maximum === total.maximum
  );
}

function validateLodgingPlan(
  this: PublicContractValidationContext,
  rule: LodgingPlanRule,
  payload: unknown,
): boolean {
  void this;
  if (!isRecord(payload)) {
    return false;
  }
  const start = parseIsoDate(payload[rule.startField]);
  const end = parseIsoDate(payload[rule.endField]);
  const nightCount = payload[rule.nightCountField];
  const anchors = payload[rule.anchorsField];
  const clusters = payload[rule.clustersField];
  const strategies = payload[rule.strategiesField];
  const favorites = payload[rule.favoritesField];
  const selected = payload[rule.selectedHotelField];
  if (
    start === null ||
    end === null ||
    typeof nightCount !== "number" ||
    !Number.isInteger(nightCount) ||
    !Array.isArray(anchors) ||
    !Array.isArray(clusters) ||
    !Array.isArray(strategies) ||
    !Array.isArray(favorites)
  ) {
    return false;
  }
  const dayCount = end.epochDay - start.epochDay + 1;
  if (dayCount < 1 || dayCount > 5 || nightCount !== dayCount - 1) {
    return false;
  }
  const anchorIds = new Set(
    anchors.map((anchor) => (isRecord(anchor) ? anchor.anchor_id : undefined)),
  );
  if (anchorIds.has(undefined) || anchorIds.size !== anchors.length) {
    return false;
  }
  const referencesKnownAnchors = (value: unknown): boolean =>
    Array.isArray(value) && value.every((anchorId) => anchorIds.has(anchorId));
  if (
    clusters.some(
      (cluster) =>
        !isRecord(cluster) || !referencesKnownAnchors(cluster.anchor_ids),
    ) ||
    strategies.some(
      (strategy) =>
        !isRecord(strategy) || !referencesKnownAnchors(strategy.anchor_ids),
    )
  ) {
    return false;
  }
  if (nightCount === 0) {
    return strategies.length === 0 && selected == null;
  }
  if (strategies.length < 2 || strategies.length > 3 || !isRecord(selected)) {
    return false;
  }
  const selectedCheckIn = parseIsoDate(selected.check_in);
  const selectedCheckOut = parseIsoDate(selected.check_out);
  if (
    selectedCheckIn?.epochDay !== start.epochDay ||
    selectedCheckOut?.epochDay !== end.epochDay ||
    selected.night_count !== nightCount
  ) {
    return false;
  }
  const strategiesById = new Map<unknown, Record<string, unknown>>();
  const candidateIds = new Set<unknown>();
  for (const strategy of strategies) {
    if (!isRecord(strategy) || strategiesById.has(strategy.strategy_id)) {
      return false;
    }
    strategiesById.set(strategy.strategy_id, strategy);
    if (!Array.isArray(strategy.representative_hotels)) {
      return false;
    }
    for (const hotel of strategy.representative_hotels) {
      if (!isRecord(hotel)) {
        return false;
      }
      candidateIds.add(hotel.place_id);
    }
  }
  if (!favorites.every((placeId) => candidateIds.has(placeId))) {
    return false;
  }
  if (selected.strategy_id == null) {
    return selected.source === "prebooked";
  }
  const selectedStrategy = strategiesById.get(selected.strategy_id);
  if (
    !selectedStrategy ||
    !Array.isArray(selectedStrategy.representative_hotels)
  ) {
    return false;
  }
  const belongsToStrategy = selectedStrategy.representative_hotels.some(
    (hotel) => isRecord(hotel) && hotel.place_id === selected.place_id,
  );
  if (!belongsToStrategy) {
    return false;
  }
  if (
    selected.source === "user_favorites" &&
    !favorites.includes(selected.place_id)
  ) {
    return false;
  }
  return (
    selected.source !== "system_candidates" ||
    candidateIds.has(selected.place_id)
  );
}

function validatePlanPublicationRequest(
  this: PublicContractValidationContext,
  _enabled: boolean,
  payload: unknown,
): boolean {
  void this;
  if (!isRecord(payload)) return false;
  const validationRequest = payload.validation_request;
  const repair = payload.repair_result;
  if (!isRecord(validationRequest) || !isRecord(repair)) return false;
  const finalValidation = repair.final_validation;
  const assumptions = payload.assumptions;
  return (
    isRecord(finalValidation) &&
    Array.isArray(assumptions) &&
    uniqueRecordField(assumptions, "assumption_id") &&
    validationRequest.trip_id === payload.trip_id &&
    repair.trip_id === payload.trip_id &&
    validationRequest.input_state_version === payload.expected_state_version &&
    repair.input_state_version === payload.expected_state_version &&
    repair.generation_id === payload.generation_id &&
    finalValidation.request_id === validationRequest.request_id &&
    repair.strict_ready === true &&
    repair.status !== "failed" &&
    repair.status !== "cancelled"
  );
}

function validatePublishedPlan(
  this: PublicContractValidationContext,
  _enabled: boolean,
  payload: unknown,
): boolean {
  void this;
  if (!isRecord(payload)) return false;
  const schedule = payload.schedule;
  const cost = payload.cost_estimate;
  const validation = payload.validation;
  const issues = payload.issues;
  const assumptions = payload.assumptions;
  const map = payload.map_projection;
  if (
    !isRecord(schedule) ||
    !isRecord(cost) ||
    !isRecord(validation) ||
    !Array.isArray(issues) ||
    !Array.isArray(assumptions) ||
    !isRecord(map) ||
    !Array.isArray(schedule.days) ||
    !Array.isArray(map.markers) ||
    !Array.isArray(map.routes)
  ) {
    return false;
  }
  if (
    schedule.trip_id !== payload.trip_id ||
    cost.trip_id !== payload.trip_id ||
    validation.trip_id !== payload.trip_id ||
    schedule.input_state_version !== payload.input_state_version ||
    cost.input_state_version !== payload.input_state_version ||
    validation.input_state_version !== payload.input_state_version ||
    schedule.task_book_id !== cost.task_book_id ||
    schedule.task_book_id !== validation.task_book_id ||
    schedule.task_book_revision !== cost.task_book_revision ||
    schedule.task_book_revision !== validation.task_book_revision ||
    schedule.start_date !== cost.start_date ||
    schedule.end_date !== cost.end_date ||
    validation.status === "blocked" ||
    JSON.stringify(issues) !== JSON.stringify(validation.issues) ||
    !uniqueRecordField(assumptions, "assumption_id")
  ) {
    return false;
  }
  const formalVersion = payload.result_contract_version;
  const formalFieldsPresent =
    payload.availability != null ||
    (Array.isArray(payload.places) && payload.places.length > 0) ||
    (Array.isArray(payload.selected_candidates) &&
      payload.selected_candidates.length > 0) ||
    payload.hotel_selection != null ||
    (Array.isArray(payload.weather) && payload.weather.length > 0);
  if (formalVersion == null) {
    if (formalFieldsPresent) return false;
  } else {
    const places = payload.places;
    const selectedCandidates = payload.selected_candidates;
    const hotel = payload.hotel_selection;
    const weather = payload.weather;
    if (
      formalVersion !== "1.0.0" ||
      !new Set(["available", "partial"]).has(String(payload.availability)) ||
      !Array.isArray(places) ||
      places.length === 0 ||
      !Array.isArray(selectedCandidates) ||
      !isRecord(hotel) ||
      !Array.isArray(weather) ||
      weather.length === 0
    ) {
      return false;
    }
    const placeIds = places.map((item) =>
      isRecord(item) ? item.place_id : undefined,
    );
    if (
      placeIds.some((value) => typeof value !== "string") ||
      new Set(placeIds).size !== placeIds.length ||
      places.some(
        (item) => !isRecord(item) || item.city_id !== schedule.city_id,
      )
    ) {
      return false;
    }
    const activityPlaceIds = new Set<unknown>();
    const serviceDates: unknown[] = [];
    for (const day of schedule.days) {
      if (!isRecord(day) || !Array.isArray(day.activities)) return false;
      serviceDates.push(day.service_date);
      for (const activity of day.activities) {
        if (!isRecord(activity)) return false;
        activityPlaceIds.add(activity.place_id);
        if (!placeIds.includes(activity.place_id)) return false;
      }
    }
    const selectedPlaceIds = selectedCandidates.map((item) =>
      isRecord(item) && isRecord(item.place) ? item.place.place_id : undefined,
    );
    if (
      selectedPlaceIds.some((value) => typeof value !== "string") ||
      new Set(selectedPlaceIds).size !== selectedPlaceIds.length ||
      selectedCandidates.some(
        (item) =>
          !isRecord(item) ||
          !isRecord(item.place) ||
          item.place.city_id !== schedule.city_id ||
          !activityPlaceIds.has(item.place.place_id),
      )
    ) {
      return false;
    }
    if (
      hotel.trip_id !== payload.trip_id ||
      hotel.city_id !== schedule.city_id ||
      hotel.input_state_version !== payload.input_state_version ||
      hotel.check_in !== schedule.start_date ||
      hotel.check_out !== schedule.end_date ||
      (Number(hotel.night_count) > 0 && hotel.decision_status !== "final") ||
      (Number(hotel.night_count) > 0 &&
        schedule.days.some(
          (day) =>
            !isRecord(day) ||
            day.start_place_id !== hotel.selected_hotel_place_id ||
            day.end_place_id !== hotel.selected_hotel_place_id,
        )) ||
      JSON.stringify(
        weather.map((item) => (isRecord(item) ? item.service_date : undefined)),
      ) !== JSON.stringify(serviceDates)
    ) {
      return false;
    }
    const categories = isRecord(cost) ? cost.categories : undefined;
    const scheduleHasPartialItems = schedule.days.some(
      (day) =>
        isRecord(day) &&
        [
          ...(Array.isArray(day.activities) ? day.activities : []),
          ...(Array.isArray(day.transport_legs) ? day.transport_legs : []),
        ].some((item) => isRecord(item) && item.availability !== "available"),
    );
    const degraded =
      schedule.status === "partial" ||
      validation.status === "review" ||
      scheduleHasPartialItems ||
      places.some(
        (item) => isRecord(item) && item.opening_availability !== "available",
      ) ||
      selectedCandidates.some(
        (item) => isRecord(item) && item.availability !== "available",
      ) ||
      weather.some(
        (item) => isRecord(item) && item.availability !== "available",
      ) ||
      (Array.isArray(categories) &&
        categories.some(
          (item) =>
            isRecord(item) &&
            new Set(["partial", "missing"]).has(String(item.status)),
        )) ||
      (Number(hotel.night_count) > 0 && hotel.status !== "available");
    if (payload.availability !== (degraded ? "partial" : "available")) {
      return false;
    }
  }
  const knownPlaceIds = new Set<unknown>();
  for (const day of schedule.days) {
    if (!isRecord(day) || !Array.isArray(day.activities)) return false;
    knownPlaceIds.add(day.start_place_id);
    knownPlaceIds.add(day.end_place_id);
    for (const activity of day.activities) {
      if (!isRecord(activity)) return false;
      knownPlaceIds.add(activity.place_id);
    }
  }
  return (
    Number.isInteger(map.selected_day_index) &&
    Number(map.selected_day_index) < schedule.days.length &&
    map.markers.every(
      (marker) => isRecord(marker) && knownPlaceIds.has(marker.place_id),
    ) &&
    map.routes.every(
      (route) =>
        isRecord(route) &&
        knownPlaceIds.has(route.from_place_id) &&
        knownPlaceIds.has(route.to_place_id),
    )
  );
}

function validatePendingPlanModification(
  this: PublicContractValidationContext,
  _enabled: boolean,
  payload: unknown,
): boolean {
  void this;
  if (!isRecord(payload)) return false;
  const index = payload.dependency_index;
  const targets = payload.targets;
  const scopes = payload.impact_scopes;
  const affected = payload.affected_day_numbers;
  const preserved = payload.preserved_day_numbers;
  if (
    !isRecord(index) ||
    !Array.isArray(index.days) ||
    !Array.isArray(targets) ||
    !Array.isArray(scopes) ||
    !Array.isArray(affected) ||
    !Array.isArray(preserved) ||
    targets.length !== scopes.length ||
    index.trip_id !== payload.trip_id ||
    index.plan_version_id !== payload.base_plan_version_id
  ) {
    return false;
  }
  const knownDays = index.days.map((item) =>
    isRecord(item) ? item.day_number : undefined,
  );
  const expectedDays = index.days.map((_, indexValue) => indexValue + 1);
  if (JSON.stringify(knownDays) !== JSON.stringify(expectedDays)) return false;
  const known = new Set(knownDays);
  if (
    affected.some((day) => !known.has(day)) ||
    preserved.some((day) => !known.has(day)) ||
    affected.some((day) => preserved.includes(day))
  ) {
    return false;
  }
  return (
    payload.global_replan_required !== true ||
    (affected.length === known.size && preserved.length === 0)
  );
}

function uniqueRecordField(values: unknown[], field: string): boolean {
  const items = values.map((value) =>
    isRecord(value) ? value[field] : undefined,
  );
  return (
    items.every((value) => typeof value === "string") &&
    new Set(items).size === items.length
  );
}

function validateTripState(
  this: PublicContractValidationContext,
  rule: TripStateRule,
  payload: unknown,
): boolean {
  void this;
  if (!isRecord(payload)) {
    return false;
  }
  const phase = payload[rule.phaseField];
  const city = payload[rule.cityField];
  const cityId = payload[rule.cityIdField];
  const dateRange = payload[rule.dateRangeField];
  const taskBook = payload[rule.taskBookField];
  const itinerary = payload[rule.itineraryField];
  const cost = payload[rule.costField];
  const publishedPlan = payload[rule.publishedPlanField];
  const pendingModification = payload[rule.pendingPlanModificationField];
  const selected = payload[rule.selectedHotelField];
  const strategies = payload[rule.strategiesField];
  const phasesAfterCity = new Set(rule.cityRequiredPhases);
  const profilePhases = new Set(rule.profileCompletePhases);
  const dateRequiredPhases = new Set(rule.dateRequiredPhases);
  const taskAndPlanPhases = new Set(rule.taskAndPlanPhases);
  const planPhases = new Set(rule.planPhases);
  const visiblePlanPhases = new Set(rule.visiblePlanPhases);
  if (
    (phasesAfterCity.has(String(phase)) && city == null && cityId == null) ||
    (phase === rule.coldStartPhase && (city != null || cityId != null)) ||
    (payload.cold_start_completed_at != null &&
      payload.personal_defaults == null)
  ) {
    return false;
  }
  if (
    profilePhases.has(String(phase)) &&
    (payload.city_brief == null ||
      payload.profile_decision == null ||
      payload.resolved_preferences == null)
  ) {
    return false;
  }
  if (dateRequiredPhases.has(String(phase)) && dateRange == null) {
    return false;
  }
  if (
    isRecord(payload.profile_decision) &&
    payload.profile_decision.mode === "use_defaults" &&
    payload.personal_defaults == null
  ) {
    return false;
  }
  let nightCount: number | null = null;
  if (dateRange == null) {
    if (
      payload[rule.dayCountField] != null ||
      payload[rule.nightCountField] != null
    ) {
      return false;
    }
  } else {
    if (!isRecord(dateRange)) {
      return false;
    }
    const start = parseIsoDate(dateRange.start_date);
    const end = parseIsoDate(dateRange.end_date);
    if (start === null || end === null) {
      return false;
    }
    const dayCount = end.epochDay - start.epochDay + 1;
    nightCount = dayCount - 1;
    if (
      dayCount < 1 ||
      dayCount > 5 ||
      payload[rule.dayCountField] !== dayCount ||
      payload[rule.nightCountField] !== nightCount
    ) {
      return false;
    }
  }
  if (
    phase === rule.attractionSelectionPhase &&
    payload.city_theme_selection == null
  ) {
    return false;
  }
  if (
    phase === rule.lodgingSelectionPhase &&
    payload[rule.diningPreferencesField] == null
  ) {
    return false;
  }
  if (phase === rule.taskReflectionPhase && taskBook == null) {
    return false;
  }
  if (
    taskAndPlanPhases.has(String(phase)) &&
    nightCount !== null &&
    nightCount > 0 &&
    selected == null
  ) {
    return false;
  }
  if (
    planPhases.has(String(phase)) &&
    (!isRecord(taskBook) || taskBook.status !== "confirmed")
  ) {
    return false;
  }
  if (
    visiblePlanPhases.has(String(phase)) &&
    (payload.current_plan_version_id == null ||
      (publishedPlan == null && (itinerary == null || cost == null)))
  ) {
    return false;
  }
  if ((itinerary == null) !== (cost == null)) {
    return false;
  }
  if (publishedPlan != null && (itinerary != null || cost != null)) {
    return false;
  }
  if (publishedPlan != null) {
    if (!isRecord(publishedPlan) || !planPhases.has(String(phase))) {
      return false;
    }
    if (
      (cityId != null &&
        isRecord(publishedPlan.schedule) &&
        publishedPlan.schedule.city_id !== cityId) ||
      (isRecord(dateRange) &&
        isRecord(publishedPlan.schedule) &&
        (publishedPlan.schedule.start_date !== dateRange.start_date ||
          publishedPlan.schedule.end_date !== dateRange.end_date))
    ) {
      return false;
    }
    if (
      publishedPlan.trip_id !== payload.trip_id ||
      publishedPlan.plan_version_id !== payload.current_plan_version_id ||
      JSON.stringify(publishedPlan.map_projection) !==
        JSON.stringify(payload.map_view)
    ) {
      return false;
    }
    if (
      phase === "draft_ready" &&
      (typeof payload.state_version !== "number" ||
        publishedPlan.input_state_version !== payload.state_version - 1 ||
        (publishedPlan.base_confirmed_version_id ?? null) !==
          (payload.base_confirmed_version_id ?? null))
    ) {
      return false;
    }
  }
  if (pendingModification != null) {
    if (
      !isRecord(pendingModification) ||
      !new Set(["revising", "planning"]).has(String(phase)) ||
      publishedPlan == null ||
      pendingModification.trip_id !== payload.trip_id ||
      pendingModification.base_plan_version_id !==
        payload.current_plan_version_id ||
      typeof pendingModification.base_state_version !== "number" ||
      typeof payload.state_version !== "number" ||
      pendingModification.base_state_version >= payload.state_version
    ) {
      return false;
    }
  }
  if (
    payload.base_confirmed_version_id != null &&
    !new Set(rule.derivedPlanPhases).has(String(phase))
  ) {
    return false;
  }
  if (!validateUniqueStateCollections(payload)) {
    return false;
  }
  const stateStart = isRecord(dateRange) ? dateRange.start_date : undefined;
  const stateEnd = isRecord(dateRange) ? dateRange.end_date : undefined;
  if (
    isRecord(taskBook) &&
    ((((taskBook.city ?? null) !== city ||
      (taskBook.city_id ?? null) !== null) &&
      ((taskBook.city_id ?? null) !== cityId ||
        (taskBook.city ?? null) !== null)) ||
      taskBook.start_date !== stateStart ||
      taskBook.end_date !== stateEnd)
  ) {
    return false;
  }
  if (
    isRecord(itinerary) &&
    (itinerary.city !== city ||
      itinerary.start_date !== stateStart ||
      itinerary.end_date !== stateEnd)
  ) {
    return false;
  }
  const selectedHotelId = isRecord(selected) ? selected.place_id : null;
  if (
    isRecord(taskBook) &&
    (taskBook.selected_hotel_id ?? null) !== selectedHotelId
  ) {
    return false;
  }
  if (isRecord(selected)) {
    if (
      selected.check_in !== stateStart ||
      selected.check_out !== stateEnd ||
      selected.night_count !== nightCount
    ) {
      return false;
    }
    if (!Array.isArray(strategies)) {
      return false;
    }
    if (selected.strategy_id != null) {
      const strategy = strategies.find(
        (item) => isRecord(item) && item.strategy_id === selected.strategy_id,
      );
      if (
        !isRecord(strategy) ||
        !Array.isArray(strategy.representative_hotels) ||
        !strategy.representative_hotels.some(
          (hotel) => isRecord(hotel) && hotel.place_id === selected.place_id,
        )
      ) {
        return false;
      }
    } else if (selected.source !== "prebooked") {
      return false;
    }
  }
  if (!stateStrategiesReferenceKnownAnchors(payload, strategies)) {
    return false;
  }
  if (isRecord(itinerary) && isRecord(cost)) {
    const days = itinerary.days;
    const total = parseAmountRange(cost.total_per_person);
    if (!Array.isArray(days) || total === null) {
      return false;
    }
    const daily = sumDailyCosts(days, "daily_cost_per_person");
    if (
      daily === null ||
      daily.minimum !== total.minimum ||
      daily.maximum !== total.maximum
    ) {
      return false;
    }
  }
  return true;
}

function validateStateVersionIncrement(
  this: PublicContractValidationContext,
  rule: StateVersionIncrementRule,
  payload: unknown,
): boolean {
  void this;
  if (!isRecord(payload)) {
    return false;
  }
  const base = payload[rule.baseField];
  const state = payload[rule.stateField];
  return (
    typeof base === "number" &&
    Number.isInteger(base) &&
    typeof state === "number" &&
    Number.isInteger(state) &&
    state === base + rule.increment
  );
}

function validateCityContent(
  this: PublicContractValidationContext,
  rule: CityContentRule,
  payload: unknown,
): boolean {
  void this;
  if (!isRecord(payload)) {
    return false;
  }
  const created = payload[rule.createdField];
  const updated = payload[rule.updatedField];
  const sources = payload[rule.sourcesField];
  const assets = payload[rule.assetsField];
  const brief = payload[rule.briefField];
  const themes = payload[rule.themesField];
  const attractions = payload[rule.attractionsField];
  if (
    typeof created !== "string" ||
    typeof updated !== "string" ||
    Number.isNaN(Date.parse(created)) ||
    Number.isNaN(Date.parse(updated)) ||
    Date.parse(updated) < Date.parse(created) ||
    !Array.isArray(sources) ||
    !Array.isArray(assets) ||
    !isRecord(brief) ||
    !Array.isArray(themes) ||
    !Array.isArray(attractions)
  ) {
    return false;
  }
  const sourceIds = uniqueFieldSet(sources, rule.sourceIdField);
  const assetIds = uniqueFieldSet(assets, rule.assetIdField);
  const themeIds = uniqueFieldSet(themes, rule.themeIdField);
  const placeIds = uniqueFieldSet(attractions, rule.placeIdField);
  if (!sourceIds || !assetIds || !themeIds || !placeIds) {
    return false;
  }
  const referencesKnown = (values: unknown, known: Set<unknown>): boolean =>
    Array.isArray(values) && values.every((value) => known.has(value));
  const sourceBearing: unknown[] = [
    ...(Array.isArray(brief.history_nodes) ? brief.history_nodes : []),
    ...(Array.isArray(brief.experience_directions)
      ? brief.experience_directions
      : []),
    ...(Array.isArray(brief.spatial_relationships)
      ? brief.spatial_relationships
      : []),
    ...themes,
    ...attractions,
  ];
  if (
    sourceBearing.some(
      (item) => !isRecord(item) || !referencesKnown(item.source_ids, sourceIds),
    )
  ) {
    return false;
  }
  const directions = brief.experience_directions;
  if (!Array.isArray(directions)) {
    return false;
  }
  for (const direction of directions) {
    if (!isRecord(direction) || !Array.isArray(direction.typical_places)) {
      return false;
    }
    for (const place of direction.typical_places) {
      if (
        !isRecord(place) ||
        !placeIds.has(place.place_id) ||
        !referencesKnown(place.source_ids, sourceIds)
      ) {
        return false;
      }
    }
  }
  const relationships = brief.spatial_relationships;
  if (
    !Array.isArray(relationships) ||
    relationships.some(
      (item) =>
        !isRecord(item) || !referencesKnown(item.related_place_ids, placeIds),
    )
  ) {
    return false;
  }
  return attractions.every(
    (item) =>
      isRecord(item) &&
      assetIds.has(item.asset_id) &&
      referencesKnown(item.theme_ids, themeIds),
  );
}

function validateExceptionalReplaySuite(
  this: PublicContractValidationContext,
  rule: ExceptionalReplayRule,
  payload: unknown,
): boolean {
  void this;
  if (!isRecord(payload)) {
    return false;
  }
  const scenariosValue = payload[rule.scenariosField];
  if (!Array.isArray(scenariosValue)) {
    return false;
  }
  const scenarios: unknown[] = scenariosValue;
  if (scenarios.length !== rule.requiredKinds.length) {
    return false;
  }
  const scenarioIds = uniqueFieldSet(scenarios, rule.scenarioIdField);
  const kinds = uniqueFieldSet(scenarios, rule.kindField);
  if (scenarioIds === null || kinds === null) {
    return false;
  }
  if (!rule.requiredKinds.every((kind) => kinds.has(kind))) {
    return false;
  }
  return scenarios.every((scenario) => {
    if (!isRecord(scenario)) {
      return false;
    }
    const events = scenario[rule.eventsField];
    const actions = scenario[rule.actionsField];
    if (
      !Array.isArray(events) ||
      !Array.isArray(actions) ||
      events.length !== actions.length
    ) {
      return false;
    }
    const commandPresence = rule.commandFields.map(
      (field) => scenario[field] !== undefined && scenario[field] !== null,
    );
    return commandPresence.every((present) => present === commandPresence[0]);
  });
}

function uniqueFieldSet(values: unknown[], field: string): Set<unknown> | null {
  const result = new Set<unknown>();
  for (const value of values) {
    if (!isRecord(value) || !(field in value) || result.has(value[field])) {
      return null;
    }
    result.add(value[field]);
  }
  return result;
}

function validateUniqueStateCollections(
  payload: Record<string, unknown>,
): boolean {
  const definitions: Array<[unknown, string]> = [
    [payload.attraction_feedback, "place_id"],
    [payload.restaurant_feedback, "place_id"],
    [payload.hotel_favorites, ""],
    [payload.anchors, "anchor_id"],
    [payload.candidate_places, "place_id"],
    [payload.lodging_strategies, "strategy_id"],
    [payload.issues, "issue_id"],
    [payload.assumptions, "assumption_id"],
    [payload.messages, "message_id"],
    [payload.conversation_messages, "message_id"],
  ];
  const collectionsAreUnique = definitions.every(([value, field]) => {
    const items = value === undefined ? [] : value;
    if (!Array.isArray(items)) {
      return false;
    }
    const keys = field
      ? items.map((item) => (isRecord(item) ? item[field] : undefined))
      : items;
    return !keys.includes(undefined) && new Set(keys).size === keys.length;
  });
  if (!collectionsAreUnique) return false;
  const messages = payload.conversation_messages ?? [];
  if (!Array.isArray(messages) || typeof payload.state_version !== "number")
    return false;
  const attachmentIds: unknown[] = [];
  for (const message of messages) {
    if (
      !isRecord(message) ||
      typeof message.state_version !== "number" ||
      message.state_version > payload.state_version ||
      !Array.isArray(message.attachments)
    ) {
      return false;
    }
    for (const attachment of message.attachments) {
      const unwrapped = unwrapRootModel(attachment);
      attachmentIds.push(unwrapped?.attachment_id);
    }
  }
  return (
    !attachmentIds.includes(undefined) &&
    new Set(attachmentIds).size === attachmentIds.length
  );
}

function stateStrategiesReferenceKnownAnchors(
  payload: Record<string, unknown>,
  strategies: unknown,
): boolean {
  const anchors = payload.anchors === undefined ? [] : payload.anchors;
  const lodgingStrategies = strategies === undefined ? [] : strategies;
  if (!Array.isArray(anchors) || !Array.isArray(lodgingStrategies)) {
    return false;
  }
  const anchorIds = new Set(
    anchors.map((anchor) => (isRecord(anchor) ? anchor.anchor_id : undefined)),
  );
  if (anchorIds.has(undefined)) {
    return false;
  }
  return lodgingStrategies.every(
    (strategy) =>
      isRecord(strategy) &&
      Array.isArray(strategy.anchor_ids) &&
      strategy.anchor_ids.every((anchorId) => anchorIds.has(anchorId)),
  );
}

function validateCostEstimationRequest(
  this: PublicContractValidationContext,
  enabled: boolean,
  payload: unknown,
): boolean {
  void this;
  if (!enabled || !isRecord(payload)) return !enabled;
  const scheduling = payload.scheduling_request;
  const schedule = payload.schedule_result;
  const facts = payload.price_facts;
  const rates = payload.exchange_rates;
  if (
    !isRecord(scheduling) ||
    !isRecord(schedule) ||
    !Array.isArray(facts) ||
    !Array.isArray(rates) ||
    scheduling.trip_id !== payload.trip_id ||
    schedule.trip_id !== payload.trip_id ||
    scheduling.input_state_version !== payload.input_state_version ||
    schedule.input_state_version !== payload.input_state_version ||
    schedule.request_id !== scheduling.request_id
  ) {
    return false;
  }
  const expected = new Set<string>();
  const days = schedule.days;
  if (!Array.isArray(days)) return false;
  for (const day of days) {
    if (!isRecord(day) || typeof day.service_date !== "string") return false;
    if (!Array.isArray(day.activities) || !Array.isArray(day.transport_legs)) {
      return false;
    }
    for (const activity of day.activities) {
      if (!isRecord(activity)) return false;
      const category =
        activity.kind === "attraction"
          ? "attraction_tickets"
          : activity.kind === "restaurant"
            ? "dining"
            : null;
      if (category !== null) {
        expected.add(
          costSubjectKey(
            "activity",
            activity.activity_id,
            day.service_date,
            category,
          ),
        );
      }
    }
    for (const leg of day.transport_legs) {
      if (!isRecord(leg)) return false;
      expected.add(
        costSubjectKey(
          "transport_leg",
          leg.leg_id,
          day.service_date,
          "local_transport",
        ),
      );
    }
  }
  const hotel = scheduling.hotel_result;
  if (!isRecord(hotel)) return false;
  if (typeof hotel.selected_hotel_place_id === "string") {
    const start = parseIsoDate(schedule.start_date);
    const end = parseIsoDate(schedule.end_date);
    if (start === null || end === null) return false;
    for (let day = start.epochDay; day < end.epochDay; day += 1) {
      expected.add(
        costSubjectKey(
          "hotel_night",
          hotel.selected_hotel_place_id,
          isoDateFromEpochDay(day),
          "lodging",
        ),
      );
    }
  }
  const actual = new Set<string>();
  const currencies = new Set<string>();
  const businessTime = Date.parse(String(payload.business_time));
  const maximumAgeHours = payload.maximum_fact_age_hours;
  if (
    Number.isNaN(businessTime) ||
    typeof maximumAgeHours !== "number" ||
    !Number.isInteger(maximumAgeHours)
  ) {
    return false;
  }
  for (const fact of facts) {
    if (
      !isRecord(fact) ||
      !validatePriceFact(fact) ||
      !isCurrentFact(fact.fetched_at, businessTime, maximumAgeHours)
    ) {
      return false;
    }
    actual.add(
      costSubjectKey(
        fact.subject_kind,
        fact.subject_id,
        fact.service_date,
        fact.category,
      ),
    );
    const original = fact.original_amount;
    if (isRecord(original) && original.currency !== "CNY") {
      if (typeof original.currency !== "string") return false;
      currencies.add(original.currency);
    }
  }
  if (
    actual.size !== facts.length ||
    actual.size !== expected.size ||
    [...expected].some((key) => !actual.has(key))
  ) {
    return false;
  }
  const rateCurrencies = rates.map((rate) =>
    isRecord(rate) ? rate.source_currency : undefined,
  );
  return (
    rates.every(
      (rate) =>
        isRecord(rate) &&
        isCurrentFact(rate.fetched_at, businessTime, maximumAgeHours),
    ) &&
    rateCurrencies.every((value) => typeof value === "string") &&
    new Set(rateCurrencies).size === rateCurrencies.length &&
    currencies.size === rateCurrencies.length &&
    [...currencies].every((currency) => rateCurrencies.includes(currency))
  );
}

function validateTripCostEstimate(
  this: PublicContractValidationContext,
  enabled: boolean,
  payload: unknown,
): boolean {
  void this;
  if (!enabled || !isRecord(payload)) return !enabled;
  const days = payload.days;
  const categories = payload.categories;
  const start = parseIsoDate(payload.start_date);
  const end = parseIsoDate(payload.end_date);
  if (
    !Array.isArray(days) ||
    !Array.isArray(categories) ||
    start === null ||
    end === null ||
    days.length !== end.epochDay - start.epochDay + 1 ||
    !validateCostCategories(categories)
  ) {
    return false;
  }
  const dailyRanges: AmountRange[] = [];
  const allLines: Record<string, unknown>[] = [];
  for (const [index, day] of days.entries()) {
    if (
      !isRecord(day) ||
      day.service_date !== isoDateFromEpochDay(start.epochDay + index) ||
      !Array.isArray(day.categories) ||
      !validateCostCategories(day.categories) ||
      !Array.isArray(day.lines) ||
      !day.lines.every(
        (line) =>
          isRecord(line) &&
          line.service_date === day.service_date &&
          validateCostLine(line),
      )
    ) {
      return false;
    }
    const dayLines = day.lines.filter(isRecord);
    if (
      dayLines.length !== day.lines.length ||
      !validateSummaryDerivation(day.categories, dayLines)
    ) {
      return false;
    }
    allLines.push(...dayLines);
    const expectedSubtotal = sumOptionalRanges(
      day.categories.map((item) =>
        isRecord(item) ? item.amount_per_person : undefined,
      ),
    );
    if (!sameOptionalRange(day.known_subtotal_per_person, expectedSubtotal)) {
      return false;
    }
    if (expectedSubtotal !== null) dailyRanges.push(expectedSubtotal);
  }
  const expectedFromDays = sumRanges(dailyRanges);
  const expectedFromCategories = sumOptionalRanges(
    categories.map((item) =>
      isRecord(item) ? item.amount_per_person : undefined,
    ),
  );
  const excluded = payload.excluded_costs;
  return (
    validateSummaryDerivation(categories, allLines) &&
    allLines.every((line) => {
      const expectedDivisor =
        line.basis === "per_vehicle"
          ? payload.party_size
          : line.basis === "per_room_night"
            ? 2
            : 1;
      return line.share_divisor === expectedDivisor;
    }) &&
    sameOptionalRange(payload.known_total_per_person, expectedFromDays) &&
    sameOptionalRange(payload.known_total_per_person, expectedFromCategories) &&
    Array.isArray(excluded) &&
    new Set(excluded).size === 3 &&
    ["airfare", "rail", "intercity_transport"].every((item) =>
      excluded.includes(item),
    )
  );
}

function validateItineraryValidationRequest(
  this: PublicContractValidationContext,
  enabled: boolean,
  payload: unknown,
): boolean {
  void this;
  if (!enabled || !isRecord(payload)) return !enabled;
  const schedule = payload.schedule_draft;
  const cost = payload.cost_draft;
  if (!isRecord(schedule) || !isRecord(cost)) {
    return false;
  }
  const weather = payload.weather;
  const days = schedule.days;
  if (!Array.isArray(weather) || !Array.isArray(days)) return false;
  return weather.every(
    (item) => isRecord(item) && validateWeatherCoverage(item),
  );
}

export function validateItineraryValidationResult(
  this: PublicContractValidationContext,
  enabled: boolean,
  payload: unknown,
): boolean {
  void this;
  if (!enabled || !isRecord(payload) || !Array.isArray(payload.issues)) {
    return !enabled;
  }
  const ids = new Set<unknown>();
  let hasHard = false;
  for (const issue of payload.issues) {
    if (!isRecord(issue) || ids.has(issue.issue_id)) return false;
    ids.add(issue.issue_id);
    hasHard ||= issue.severity === "hard_conflict";
    const hasRepair = issue.repair_action !== "none";
    if (issue.repairable !== hasRepair) return false;
    if (issue.target_kind === "day" && typeof issue.service_date !== "string") {
      return false;
    }
    if (
      issue.target_kind === "activity" &&
      typeof issue.activity_id !== "string" &&
      typeof issue.unscheduled_node_id !== "string"
    ) {
      return false;
    }
    if (
      (issue.target_kind === "transport_leg" &&
        typeof issue.transport_leg_id !== "string") ||
      (issue.target_kind === "hotel" &&
        typeof issue.hotel_place_id !== "string") ||
      (issue.target_kind === "cost_category" &&
        typeof issue.cost_category !== "string")
    ) {
      return false;
    }
  }
  const expected = hasHard
    ? "blocked"
    : payload.issues.length > 0
      ? "review"
      : "valid";
  return payload.status === expected;
}

function validateItineraryRepairRequest(
  this: PublicContractValidationContext,
  enabled: boolean,
  payload: unknown,
): boolean {
  void this;
  if (!enabled || !isRecord(payload)) return !enabled;
  const request = payload.validation_request;
  const result = payload.validation_result;
  const proposals = payload.proposals;
  if (!isRecord(request) || !isRecord(result) || !Array.isArray(proposals)) {
    return false;
  }
  if (
    request.trip_id !== payload.trip_id ||
    result.trip_id !== payload.trip_id ||
    request.input_state_version !== payload.input_state_version ||
    result.input_state_version !== payload.input_state_version ||
    result.request_id !== request.request_id ||
    payload.max_rounds !== 2 ||
    !Array.isArray(result.issues)
  ) {
    return false;
  }
  const issueIds = new Set(
    result.issues.filter(isRecord).map((issue) => issue.issue_id),
  );
  const proposalIds = new Set<unknown>();
  const roundIssueKeys = new Set<string>();
  for (const proposal of proposals) {
    if (!isRecord(proposal) || proposalIds.has(proposal.proposal_id)) {
      return false;
    }
    proposalIds.add(proposal.proposal_id);
    if (!issueIds.has(proposal.issue_id)) return false;
    const key = `${String(proposal.round_number)}:${String(proposal.issue_id)}`;
    if (roundIssueKeys.has(key)) return false;
    roundIssueKeys.add(key);
  }
  return true;
}

function validateItineraryRepairResult(
  this: PublicContractValidationContext,
  enabled: boolean,
  payload: unknown,
): boolean {
  void this;
  if (!enabled || !isRecord(payload)) return !enabled;
  const rounds = payload.rounds;
  const validation = payload.final_validation;
  const remaining = payload.remaining_issue_ids;
  if (
    !Array.isArray(rounds) ||
    !isRecord(validation) ||
    !Array.isArray(validation.issues) ||
    !Array.isArray(remaining)
  ) {
    return false;
  }
  if (
    rounds.length > 2 ||
    rounds.some(
      (round, index) => !isRecord(round) || round.round_number !== index + 1,
    )
  ) {
    return false;
  }
  const remainingSet = new Set(remaining);
  const issueIds = new Set(
    validation.issues.filter(isRecord).map((issue) => issue.issue_id),
  );
  if (
    remainingSet.size !== remaining.length ||
    remainingSet.size !== issueIds.size ||
    [...remainingSet].some((issueId) => !issueIds.has(issueId))
  ) {
    return false;
  }
  const strictReady = validation.status !== "blocked";
  if (payload.strict_ready !== strictReady) return false;
  if (payload.status === "not_needed") {
    return rounds.length === 0 && validation.status === "valid";
  }
  if (payload.status === "repaired") {
    return rounds.length > 0 && strictReady;
  }
  if (payload.status === "partial") return validation.status === "review";
  if (payload.status === "failed") return validation.status === "blocked";
  return (
    payload.status === "cancelled" &&
    rounds.length > 0 &&
    isRecord(rounds[rounds.length - 1]) &&
    rounds[rounds.length - 1].status === "cancelled"
  );
}

function validatePriceFact(fact: Record<string, unknown>): boolean {
  const missing = fact.availability === "missing";
  const partial = fact.availability === "partial";
  if (
    !Array.isArray(fact.source_reference_ids) ||
    fact.source_reference_ids.length === 0
  ) {
    return false;
  }
  const expectedBasis =
    fact.category === "dining" || fact.category === "attraction_tickets"
      ? "per_person"
      : fact.category === "lodging"
        ? "per_room_night"
        : null;
  if (expectedBasis !== null && fact.basis !== expectedBasis) return false;
  if (missing)
    return (
      fact.original_amount == null && typeof fact.missing_reason === "string"
    );
  if (
    !isRecord(fact.original_amount) ||
    typeof fact.original_amount.currency !== "string" ||
    typeof fact.original_amount.minimum_minor !== "number" ||
    typeof fact.original_amount.maximum_minor !== "number" ||
    fact.original_amount.minimum_minor < 0 ||
    fact.original_amount.maximum_minor < fact.original_amount.minimum_minor
  ) {
    return false;
  }
  return partial
    ? typeof fact.missing_reason === "string"
    : fact.availability === "available" && fact.missing_reason == null;
}

function validateWeatherCoverage(item: Record<string, unknown>): boolean {
  const sources = item.source_reference_ids;
  if (!Array.isArray(sources)) return false;
  if (item.availability === "available") {
    return (
      item.night_condition_available === true &&
      sources.length > 0 &&
      item.missing_reason == null &&
      typeof item.fetched_at === "string"
    );
  }
  if (item.availability === "partial") {
    return (
      sources.length > 0 &&
      typeof item.missing_reason === "string" &&
      typeof item.fetched_at === "string"
    );
  }
  return (
    item.availability === "missing" &&
    item.night_condition_available === false &&
    sources.length === 0 &&
    typeof item.missing_reason === "string" &&
    item.fetched_at == null
  );
}

function validateCostCategories(categories: unknown[]): boolean {
  const expected = new Set([
    "lodging",
    "dining",
    "attraction_tickets",
    "local_transport",
  ]);
  const actual = categories.map((item) =>
    isRecord(item) ? item.category : undefined,
  );
  return (
    actual.length === expected.size &&
    new Set(actual).size === expected.size &&
    actual.every((item) => expected.has(item as string)) &&
    categories.every((item) => isRecord(item) && validateCategorySummary(item))
  );
}

function validateSummaryDerivation(
  categories: unknown[],
  lines: Record<string, unknown>[],
): boolean {
  return categories.every((rawSummary) => {
    if (!isRecord(rawSummary)) return false;
    const applicable = lines.filter(
      (line) => line.category === rawSummary.category,
    );
    const priced = applicable.filter(
      (line) => parseAmountRange(line.amount_per_person) !== null,
    );
    const missing = applicable.filter(
      (line) =>
        line.amount_per_person == null || line.availability === "partial",
    );
    const expectedStatus =
      applicable.length === 0
        ? "not_applicable"
        : priced.length === 0
          ? "missing"
          : missing.length > 0
            ? "partial"
            : "available";
    const expectedAmount = sumOptionalRanges(
      priced.map((line) => line.amount_per_person),
    );
    const expectedSources = new Set(
      applicable.flatMap((line) =>
        Array.isArray(line.source_reference_ids)
          ? line.source_reference_ids
          : [],
      ),
    );
    const actualSources = rawSummary.source_reference_ids;
    return (
      rawSummary.status === expectedStatus &&
      sameOptionalRange(rawSummary.amount_per_person, expectedAmount) &&
      rawSummary.priced_item_count === priced.length &&
      rawSummary.missing_item_count === missing.length &&
      Array.isArray(actualSources) &&
      actualSources.length === expectedSources.size &&
      actualSources.every((source) => expectedSources.has(source))
    );
  });
}

function validateCategorySummary(item: Record<string, unknown>): boolean {
  const amount = item.amount_per_person;
  const priced = item.priced_item_count;
  const missing = item.missing_item_count;
  const note = item.note;
  if (
    typeof priced !== "number" ||
    typeof missing !== "number" ||
    !Number.isInteger(priced) ||
    !Number.isInteger(missing)
  ) {
    return false;
  }
  if (item.status === "not_applicable") {
    return (
      amount == null &&
      priced === 0 &&
      missing === 0 &&
      typeof note === "string"
    );
  }
  if (item.status === "missing") {
    return (
      amount == null && priced === 0 && missing > 0 && typeof note === "string"
    );
  }
  if (parseAmountRange(amount) === null || priced < 1) return false;
  if (item.status === "available") return missing === 0 && note == null;
  return item.status === "partial" && missing > 0 && typeof note === "string";
}

function validateCostLine(line: Record<string, unknown>): boolean {
  if (
    !Array.isArray(line.source_reference_ids) ||
    line.source_reference_ids.length === 0
  ) {
    return false;
  }
  if (line.availability === "missing") {
    return (
      line.amount_per_person == null &&
      line.original_amount == null &&
      line.exchange_rate_id == null &&
      line.exchange_rate_fetched_at == null &&
      typeof line.missing_reason === "string"
    );
  }
  if (
    parseAmountRange(line.amount_per_person) === null ||
    !isRecord(line.original_amount)
  ) {
    return false;
  }
  const currency = line.original_amount.currency;
  if (
    (currency === "CNY" &&
      (line.exchange_rate_id != null ||
        line.exchange_rate_fetched_at != null)) ||
    (currency !== "CNY" &&
      (typeof line.exchange_rate_id !== "string" ||
        typeof line.exchange_rate_fetched_at !== "string"))
  ) {
    return false;
  }
  if (line.availability === "available") return line.missing_reason == null;
  return (
    line.availability === "partial" && typeof line.missing_reason === "string"
  );
}

function isCurrentFact(
  fetchedAt: unknown,
  businessTime: number,
  maximumAgeHours: number,
): boolean {
  if (typeof fetchedAt !== "string") return false;
  const fetched = Date.parse(fetchedAt);
  return (
    !Number.isNaN(fetched) &&
    fetched <= businessTime &&
    businessTime - fetched <= maximumAgeHours * 3_600_000
  );
}

function costSubjectKey(
  kind: unknown,
  id: unknown,
  serviceDate: unknown,
  category: unknown,
): string {
  return `${String(kind)}:${String(id)}:${String(serviceDate)}:${String(category)}`;
}

function isoDateFromEpochDay(epochDay: number): string {
  return new Date(epochDay * 86_400_000).toISOString().slice(0, 10);
}

function sumOptionalRanges(values: unknown[]): AmountRange | null {
  const parsed = values
    .filter((value) => value != null)
    .map((value) => parseAmountRange(value));
  if (parsed.some((value) => value === null)) return null;
  return sumRanges(parsed as AmountRange[]);
}

function sumRanges(values: AmountRange[]): AmountRange | null {
  if (values.length === 0) return null;
  return {
    minimum: values.reduce((sum, item) => sum + item.minimum, 0),
    maximum: values.reduce((sum, item) => sum + item.maximum, 0),
  };
}

function sameOptionalRange(
  value: unknown,
  expected: AmountRange | null,
): boolean {
  if (expected === null) return value == null;
  const actual = parseAmountRange(value);
  return (
    actual !== null &&
    actual.minimum === expected.minimum &&
    actual.maximum === expected.maximum
  );
}

interface AmountRange {
  minimum: number;
  maximum: number;
}

function parseAmountRange(value: unknown): AmountRange | null {
  if (!isRecord(value)) {
    return null;
  }
  const minimum = value.minimum_fen;
  const maximum = value.maximum_fen;
  if (
    typeof minimum !== "number" ||
    typeof maximum !== "number" ||
    !Number.isInteger(minimum) ||
    !Number.isInteger(maximum) ||
    minimum < 0 ||
    maximum < minimum
  ) {
    return null;
  }
  return { minimum, maximum };
}

function sumDailyCosts(days: unknown[], field: string): AmountRange | null {
  let minimum = 0;
  let maximum = 0;
  for (const day of days) {
    if (!isRecord(day)) {
      return null;
    }
    const amount = parseAmountRange(day[field]);
    if (amount === null) {
      return null;
    }
    minimum += amount.minimum;
    maximum += amount.maximum;
  }
  return { minimum, maximum };
}

interface ParsedDate {
  year: number;
  month: number;
  day: number;
  epochDay: number;
}

function parseIsoDate(value: unknown): ParsedDate | null {
  if (typeof value !== "string") {
    return null;
  }
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(value);
  if (!match) {
    return null;
  }
  const year = Number(match[1]);
  const month = Number(match[2]);
  const day = Number(match[3]);
  if (year < 1 || month < 1 || month > 12 || day < 1) {
    return null;
  }
  const timestamp = Date.UTC(year, month - 1, day);
  const parsed = new Date(timestamp);
  if (
    parsed.getUTCFullYear() !== year ||
    parsed.getUTCMonth() !== month - 1 ||
    parsed.getUTCDate() !== day
  ) {
    return null;
  }
  return { year, month, day, epochDay: timestamp / 86_400_000 };
}

function addYearsClamped(value: ParsedDate, years: number): ParsedDate {
  const year = value.year + years;
  const lastDay = new Date(Date.UTC(year, value.month, 0)).getUTCDate();
  const day = Math.min(value.day, lastDay);
  const result = parseIsoDate(
    `${year.toString().padStart(4, "0")}-${value.month
      .toString()
      .padStart(2, "0")}-${day.toString().padStart(2, "0")}`,
  );
  if (result === null) {
    throw new Error("Unable to calculate the contract date boundary");
  }
  return result;
}

function parseTime(value: unknown): number | null {
  if (typeof value !== "string") {
    return null;
  }
  const match = /^(\d{2}):(\d{2})(?::(\d{2})(?:\.(\d{1,6}))?)?$/.exec(value);
  if (!match) {
    return null;
  }
  const hour = Number(match[1]);
  const minute = Number(match[2]);
  const second = Number(match[3] ?? "0");
  const fraction = Number(`0.${match[4] ?? "0"}`);
  if (hour > 23 || minute > 59 || second > 59) {
    return null;
  }
  return hour * 3600 + minute * 60 + second + fraction;
}

function destinationToday(): string {
  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone: "Asia/Shanghai",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).formatToParts(new Date());
  const values = Object.fromEntries(
    parts.map((part) => [part.type, part.value]),
  );
  return `${values.year}-${values.month}-${values.day}`;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
