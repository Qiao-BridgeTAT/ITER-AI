const DAY_IN_MILLISECONDS = 86_400_000;

export interface CalendarDate {
  year: number;
  month: number;
  day: number;
}

export function destinationToday(now = new Date()): string {
  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone: "Asia/Shanghai",
    year: "numeric",
    month: "2-digit",
    day: "2-digit"
  }).formatToParts(now);
  const values = Object.fromEntries(
    parts.map((part) => [part.type, part.value])
  );
  return `${values.year}-${values.month}-${values.day}`;
}

export function parseCalendarDate(value: string): CalendarDate | null {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(value);
  if (match === null) {
    return null;
  }
  const year = Number(match[1]);
  const month = Number(match[2]);
  const day = Number(match[3]);
  const candidate = new Date(Date.UTC(year, month - 1, day));
  if (
    candidate.getUTCFullYear() !== year ||
    candidate.getUTCMonth() !== month - 1 ||
    candidate.getUTCDate() !== day
  ) {
    return null;
  }
  return { year, month, day };
}

export function formatCalendarDate(value: CalendarDate): string {
  return `${String(value.year).padStart(4, "0")}-${String(value.month).padStart(
    2,
    "0"
  )}-${String(value.day).padStart(2, "0")}`;
}

export function createCalendarDate(
  year: number,
  month: number,
  day: number
): string | null {
  const value = formatCalendarDate({ year, month, day });
  return parseCalendarDate(value) === null ? null : value;
}

export function addCalendarDays(value: string, days: number): string {
  const date = requireCalendarDate(value);
  const result = new Date(
    toEpochMilliseconds(date) + days * DAY_IN_MILLISECONDS
  );
  return formatCalendarDate({
    year: result.getUTCFullYear(),
    month: result.getUTCMonth() + 1,
    day: result.getUTCDate()
  });
}

export function calendarDayDifference(start: string, end: string): number {
  return Math.round(
    (toEpochMilliseconds(requireCalendarDate(end)) -
      toEpochMilliseconds(requireCalendarDate(start))) /
      DAY_IN_MILLISECONDS
  );
}

export function calendarDayOfWeek(value: string): number {
  return new Date(toEpochMilliseconds(requireCalendarDate(value))).getUTCDay();
}

export function compareCalendarDates(left: string, right: string): number {
  return (
    toEpochMilliseconds(requireCalendarDate(left)) -
    toEpochMilliseconds(requireCalendarDate(right))
  );
}

export function formatChineseDate(value: string): string {
  const date = requireCalendarDate(value);
  return `${date.year} 年 ${date.month} 月 ${date.day} 日`;
}

function requireCalendarDate(value: string): CalendarDate {
  const parsed = parseCalendarDate(value);
  if (parsed === null) {
    throw new Error(`Invalid calendar date: ${value}`);
  }
  return parsed;
}

function toEpochMilliseconds(value: CalendarDate): number {
  return Date.UTC(value.year, value.month - 1, value.day);
}
