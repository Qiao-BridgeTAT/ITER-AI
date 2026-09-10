/** Presentation only: never use rounded times for ordering or feasibility. */
export function formatItineraryTime(
  value: string,
  { exact = false }: { exact?: boolean } = {},
): string {
  const minutes = clockMinutes(value);
  if (minutes === null) return value;
  if (exact) {
    const [hour, minute, second] = value.split(":");
    const clock = `${hour.padStart(2, "0")}:${minute}`;
    return Number(second) ? `${clock}:${second}` : clock;
  }
  return quarterClock(Math.round(minutes / 15) * 15);
}

type TimelineClock = { time: string; timeIsFixed?: boolean };

/** Preserve fixed anchors and raw ordering, even when adjacent labels coincide. */
export function formatItineraryDayTimes(day: {
  startTime: string;
  endTime: string;
  stops: readonly TimelineClock[];
}): { startTime: string; endTime: string; stopTimes: string[] } {
  const entries: TimelineClock[] = [
    { time: day.startTime },
    ...day.stops,
    { time: day.endTime },
  ];
  const nextFixed: number[] = [];
  let upper = 24 * 60;
  for (let index = entries.length - 1; index >= 0; index--) {
    nextFixed[index] = upper;
    const value = clockMinutes(entries[index].time);
    if (entries[index].timeIsFixed && value !== null) upper = value;
  }
  let lower = 0;
  const labels = entries.map((entry, index) => {
    const value = clockMinutes(entry.time);
    if (value === null) return entry.time;
    if (entry.timeIsFixed) {
      lower = value;
      return formatItineraryTime(entry.time, { exact: true });
    }
    const earliest = Math.ceil(lower / 15) * 15;
    const latest = Math.floor(nextFixed[index] / 15) * 15;
    // No quarter-hour exists between close reservations: accuracy takes priority.
    if (earliest > latest)
      return formatItineraryTime(entry.time, { exact: true });
    return quarterClock(
      Math.min(latest, Math.max(earliest, Math.round(value / 15) * 15)),
    );
  });
  return {
    startTime: labels[0],
    endTime: labels[labels.length - 1],
    stopTimes: labels.slice(1, -1),
  };
}

function clockMinutes(value: string): number | null {
  const match = /^(\d{1,2}):(\d{2})(?::(\d{2}))?$/u.exec(value);
  if (!match) return null;
  const hour = Number(match[1]);
  const minute = Number(match[2]);
  const second = Number(match[3] ?? 0);
  if (
    hour > 24 ||
    minute > 59 ||
    second > 59 ||
    (hour === 24 && (minute !== 0 || second !== 0))
  )
    return null;
  return hour * 60 + minute + second / 60;
}

function quarterClock(rounded: number): string {
  // Keep the day's end at 24:00 rather than displaying a backwards 00:00.
  return `${String(Math.floor(rounded / 60)).padStart(2, "0")}:${String(rounded % 60).padStart(2, "0")}`;
}
