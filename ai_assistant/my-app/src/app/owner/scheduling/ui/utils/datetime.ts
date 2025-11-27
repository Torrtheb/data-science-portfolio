// src/app/owner/scheduling/ui/utils/datetime.ts
export function to24h(h12: number, ampm: "AM" | "PM") {
  let h = h12 % 12;
  if (ampm === "PM") h += 12;
  return h;
}
export function pad2(n: number) {
  return n.toString().padStart(2, "0");
}
/** Combines local date (yyyy-mm-dd) + time (HH:mm or HH:mm:ss) into UTC ISO string */
export function localDateTimeToISO(dateStr: string, timeStr: string) {
  const t = timeStr.length === 5 ? `${timeStr}:00` : timeStr;
  const d = new Date(`${dateStr}T${t}`);
  if (Number.isNaN(d.getTime())) throw new Error("Invalid date or time");
  return d.toISOString();
}
/** Always return a Date instance (eliminates string|Date union errors) */
export function ensureDate(x: string | Date): Date {
  return x instanceof Date ? x : new Date(x);
}

/**
 * Format a Date as "YYYY-MM-DDTHH:MM" in a specific IANA timezone,
 * suitable for use in <input type="datetime-local">.
 */
export function toInputLocalInTZ(d: Date, tz: string): string {
  const fmt = new Intl.DateTimeFormat("en-CA", {
    timeZone: tz,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  });
  const parts = fmt.formatToParts(d);
  const get = (t: string) => parts.find((p) => p.type === t)?.value || "";
  const Y = get("year");
  const M = get("month");
  const D = get("day");
  const H = get("hour");
  const m = get("minute");
  return `${Y}-${M}-${D}T${H}:${m}`;
}

/**
 * Convert a naive local datetime string (YYYY-MM-DDTHH:MM) that is intended to
 * be in the given IANA timezone into a UTC ISO string.
 *
 * This uses an offset-diff approach and retries once to handle DST edges.
 */
export function fromLocalInTZToUTC(dtLocal: string, tz: string): string {
  const [date, time] = dtLocal.split("T");
  if (!date || !time) throw new Error("Invalid datetime-local value");
  const [y, mo, d] = date.split("-").map((n) => parseInt(n, 10));
  const [hh, mm] = time.split(":").map((n) => parseInt(n, 10));
  if ([y, mo, d, hh, mm].some((n) => Number.isNaN(n))) throw new Error("Invalid datetime-local value");

  const desiredUTC = Date.UTC(y, mo - 1, d, hh, mm);
  const fmt = new Intl.DateTimeFormat("en-CA", {
    timeZone: tz,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  });
  const partsAt = (ms: number) => {
    const parts = fmt.formatToParts(new Date(ms));
    const get = (t: string) => parseInt(parts.find((p) => p.type === t)?.value || "0", 10);
    return { Y: get("year"), M: get("month"), D: get("day"), H: get("hour"), m: get("minute") };
  };

  // initial guess: interpret provided wall time as if UTC
  let guess = Date.UTC(y, mo - 1, d, hh, mm);
  let shown = partsAt(guess);
  let shownUTC = Date.UTC(shown.Y, shown.M - 1, shown.D, shown.H, shown.m);
  let delta = desiredUTC - shownUTC;
  guess += delta;

  // retry once for DST boundaries
  shown = partsAt(guess);
  shownUTC = Date.UTC(shown.Y, shown.M - 1, shown.D, shown.H, shown.m);
  delta = desiredUTC - shownUTC;
  guess += delta;

  return new Date(guess).toISOString();
}
