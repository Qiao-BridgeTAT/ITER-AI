import Ajv2020, { type ValidateFunction } from "ajv/dist/2020.js";
import {
  validateItineraryValidationResult,
  validateMapUpdate,
} from "./validation";

import {
  V4_PUBLIC_SCHEMAS,
  type V4PublicContractName,
} from "../generated/v4/schemaRegistry";

export type { V4PublicContractName } from "../generated/v4/schemaRegistry";

export type V4ContractValidationResult =
  | { success: true; errors: readonly [] }
  | { success: false; errors: readonly string[] };

const ajv = new Ajv2020({
  allErrors: true,
  discriminator: true,
  strict: true,
});
// Planner snapshots embed the existing V3 validator result contract. Keep its
// actual predicate; do not disable strict mode or treat the keyword as a no-op.
ajv.addKeyword({
  keyword: "x-travel-itinerary-validation-result",
  schemaType: "boolean",
  type: "object",
  errors: false,
  validate: validateItineraryValidationResult,
});
// The V4 published-plan snapshot embeds the formal map projection. Reuse the
// same referential-integrity predicate as the existing public event contract.
ajv.addKeyword({
  keyword: "x-travel-map-update",
  schemaType: "boolean",
  type: "object",
  errors: false,
  validate: validateMapUpdate,
});

ajv.addFormat("date", /^\d{4}-\d{2}-\d{2}$/);
ajv.addFormat("time", /^\d{2}:\d{2}:\d{2}(?:\.\d+)?$/);
ajv.addFormat(
  "date-time",
  /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$/,
);
ajv.addFormat(
  "uuid",
  /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i,
);
ajv.addFormat("uri", {
  type: "string",
  validate: (value: string) => {
    try {
      const parsed = new URL(value);
      return parsed.protocol === "http:" || parsed.protocol === "https:";
    } catch {
      return false;
    }
  },
});

const validators = new Map<V4PublicContractName, ValidateFunction>();

export function validateV4Contract(
  contract: V4PublicContractName,
  payload: unknown,
): V4ContractValidationResult {
  let validator = validators.get(contract);
  if (validator === undefined) {
    validator = ajv.compile(V4_PUBLIC_SCHEMAS[contract]);
    validators.set(contract, validator);
  }
  if (validator(payload)) {
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
