import type {
  ColdStartSubmission,
  ResolvedTripPreferences
} from "../generated/contracts";

export const MOCK_SAVED_PERSONAL_DEFAULTS: ColdStartSubmission = {
  day_start: "around_09",
  day_return: "around_21",
  pace_level: 3,
  classic_niche_level: 3,
  walking_tolerance: "around_10",
  bike_tolerance: "never",
  transit_taxi_level: 3,
  priority_goals: ["smooth_routes"]
};

export const NEUTRAL_RESOLVED_PREFERENCES: ResolvedTripPreferences = {
  day_start: "around_09",
  day_return: "around_21",
  pace_level: 3,
  classic_niche_level: 3,
  walking_tolerance: "around_10",
  bike_tolerance: "never",
  transit_taxi_level: 3,
  priority_goals: []
};
