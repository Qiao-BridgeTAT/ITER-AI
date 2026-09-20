import type { V4TripSetupPayload } from "../generated/v4/contracts";
import { calendarDayDifference, parseCalendarDate } from "./dateMath";

export interface DateRangeSelection {
  start: string | null;
  end: string | null;
}

export function selectRangeDate(
  range: DateRangeSelection,
  date: string
): DateRangeSelection {
  if (!parseCalendarDate(date)) throw new Error("Invalid selected date");
  if (!range.start || range.end || date < range.start) {
    return { start: date, end: null };
  }
  return { start: range.start, end: date };
}

export function tripLength(range: DateRangeSelection) {
  if (!range.start || !range.end) return null;
  const nights = calendarDayDifference(range.start, range.end);
  if (nights < 0) return null;
  return { days: nights + 1, nights };
}

export type TripSetupFields = Omit<V4TripSetupPayload, "message_id">;

export function tripSetupFields(
  cityId: string,
  range: DateRangeSelection
): TripSetupFields {
  const length = tripLength(range);
  if (!/^cn-[0-9]{6}$/.test(cityId) || !length || length.days > 5) {
    throw new Error("Incomplete or unsupported trip setup");
  }
  return { city_id: cityId, start_date: range.start!, end_date: range.end! };
}
