// src/app/scheduling/ui/UserSchedulingClient.tsx
"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import useSWR, { mutate } from "swr";
import { parseISO, format, startOfWeek as dfStartOfWeek } from "date-fns";
import { Calendar, dateFnsLocalizer, View } from "react-big-calendar";
import moment from "moment-timezone";
import "react-big-calendar/lib/css/react-big-calendar.css";
import { enUS } from "date-fns/locale";
import {
  fetchPublicSlots,
  listMyAppointments,
  myBookAppointment,
  myBookRecurringAppointments,
  myCancelAppointment,
  type PublicSlot,
  type UserAppt,
  ApiError,
  getPublicHolidays,
  type OwnerHoliday,
} from "@/lib/api";
import {
  listPublicServiceOptions,
  listPublicSlotsPriced,
  type ServiceOption,
} from "@/lib/api";

// ==== Variable-duration helpers ====
function chainableStartsForDuration(slots: PublicSlot[], durationMinutes: number): PublicSlot[] {
  if (!slots?.length || !durationMinutes) return [];
  const arr = [...slots]
    .map((s) => ({ s: new Date(s.start).getTime(), e: new Date(s.end).getTime() }))
    .sort((a, b) => a.s - b.s);

  const startIdx = new Map<number, number>();
  arr.forEach((x, i) => startIdx.set(x.s, i));

  const out: PublicSlot[] = [];
  const seen = new Set<string>();
  const targetMs = durationMinutes * 60000;

  for (let i = 0; i < arr.length; i++) {
    let need = targetMs;
    let t = arr[i].s;
    let j = i;
    let ok = true;

    while (need > 0) {
      const cur = arr[j];
      if (!cur || cur.s !== t) {
        ok = false;
        break;
      }
      const chunk = cur.e - cur.s;
      need -= chunk;
      t = cur.e;
      if (need > 0) {
        const nextIdx = startIdx.get(t);
        if (nextIdx == null) {
          ok = false;
          break;
        }
        j = nextIdx;
      }
    }

    if (ok) {
      const startISO = new Date(arr[i].s).toISOString();
      const endISO = new Date(arr[i].s + targetMs).toISOString();
      const key = `${startISO}__${endISO}`;
      if (!seen.has(key)) {
        seen.add(key);
        out.push({ start: startISO, end: endISO });
      }
    }
  }

  return out;
}

// (Optional) simple price table by duration (minutes). Edit to your liking.
const _PRICE_TABLE: Record<number, { price_cents: number; currency: string }> = {
  30: { price_cents: 6000, currency: "CAD" },
  60: { price_cents: 10000, currency: "CAD" },
};

const locales = { "en-US": enUS };
// ✅ Proper startOfWeek for date-fns localizer (prevents weird week starts)
const localizer = dateFnsLocalizer({
  format,
  parse: (str: string) => new Date(str),
  startOfWeek: (date: Date) => dfStartOfWeek(date, { weekStartsOn: 0, locale: enUS }),
  getDay: (d: Date) => d.getDay(),
  locales,
});

import PriceList from "@/components/PriceList";

export default function UserSchedulingClient({ initialTimezone }: { initialTimezone: string }) {
  // Main calendar state
  const [view, setView] = useState<View>("week");
  const [date, setDate] = useState(new Date());
  // Slot side panel (book)
  const [slotPanelOpen, setSlotPanelOpen] = useState(false);
  const [slotPanelError, setSlotPanelError] = useState<string | null>(null);
  const [slotDayISO, setSlotDayISO] = useState(format(new Date(), "yyyy-MM-dd"));
  const [slotPanelView, setSlotPanelView] = useState<"list" | "calendar">("list");
  const [slotPanelLessonFor, setSlotPanelLessonFor] = useState("");
  const [repeatWeekly, setRepeatWeekly] = useState(false);
  const [repeatCount, setRepeatCount] = useState<number>(4);
  const [actingId, setActingId] = useState<string | null>(null); // disable buttons while booking/moving
  const closeSlotPanel = useCallback(() => {
    setSlotPanelOpen(false);
    setSlotPanelError(null);
    setSlotPanelLessonFor("");
  }, []);
  const [cancelDialog, setCancelDialog] = useState<{ id: string; start: Date; end: Date } | null>(null);
  const [cancelReason, setCancelReason] = useState("");
  const [cancelError, setCancelError] = useState<string | null>(null);
  const [cancelSubmitting, setCancelSubmitting] = useState(false);
  const closeCancelDialog = useCallback(
    (force = false) => {
      if (cancelSubmitting && !force) return;
      setCancelDialog(null);
      setCancelReason("");
      setCancelError(null);
    },
    [cancelSubmitting]
  );
  const tz = initialTimezone;
  function toDisplayDate(utcIso: string, tzName?: string) {
    if (!tzName) return new Date(utcIso);
    const m = moment.utc(utcIso).tz(tzName);
    return new Date(m.format("YYYY-MM-DDTHH:mm:ss"));
  }
  type HolidayEvent = { id: string; title: string; start: Date; end: Date; resource: { type: "holiday" } };
  const [holidayEvents, setHolidayEvents] = useState<HolidayEvent[]>([]);
  // Initialize from localStorage synchronously to avoid first-render flicker
  const [showHolidays, setShowHolidays] = useState<boolean>(() => {
    try {
      if (typeof window !== "undefined") {
        const v = window.localStorage.getItem("client.showHolidays");
        if (v !== null) return v === "1";
      }
    } catch {}
    return true;
  });

  // Close panel with ESC for nicer UX
  useEffect(() => {
    if (!slotPanelOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") closeSlotPanel();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [slotPanelOpen, closeSlotPanel]);

  useEffect(() => {
    if (!cancelDialog) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") closeCancelDialog();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [cancelDialog, closeCancelDialog]);

  // --- Availability (loaded only when panel is open) ---
  const { data: slots, isLoading: slotsLoading } = useSWR<PublicSlot[]>(
    slotPanelOpen ? ["/public/slots", slotDayISO, tz] : null,
    () => fetchPublicSlots(slotDayISO, tz),
    { refreshInterval: 10000, revalidateOnFocus: true }
  );

  // Pricing + duration + priced slots
  const [serviceOptions, setServiceOptions] = useState<ServiceOption[]>([]);
  const servicePriceByDuration = useMemo(() => {
    const map: Record<number, { price_cents: number; currency: string }> = {};
    for (const o of serviceOptions) map[o.duration_minutes] = { price_cents: o.price_cents, currency: o.currency };
    return map;
  }, [serviceOptions]);
  const [selectedDuration, setSelectedDuration] = useState<number | null>(null);
  const [pricedSlots, setPricedSlots] = useState<
    Array<{ start: string; end: string; duration_minutes: number; price_cents: number; currency: string }>
  >([]);
  const [_pricedLoading, setPricedLoading] = useState(false);
  const [_pricedError, setPricedError] = useState<string | null>(null);
  // All priced slots for the selected day (across durations), used to filter available durations
  const [pricedDayAll, setPricedDayAll] = useState<
    Array<{ start: string; end: string; duration_minutes: number; price_cents: number; currency: string }>
  >([]);

  // Base slot size (minutes) from the first slot of the day (fallback 30)
  const baseStepMinutes = useMemo(() => {
    const first = slots?.[0];
    if (!first) return 30;
    const d = (parseISO(first.end).getTime() - parseISO(first.start).getTime()) / 60000;
    return Math.max(5, Math.round(d));
  }, [slots]);

  // Max number of contiguous base slots we have in this day (to cap choices)
  const maxChainSlots = useMemo(() => {
    if (!slots?.length) return 1;
    const times = [...slots]
      .map((s) => ({ s: parseISO(s.start).getTime(), e: parseISO(s.end).getTime() }))
      .sort((a, b) => a.s - b.s);
    let maxRun = 1,
      run = 1;
    for (let i = 1; i < times.length; i++) {
      if (times[i].s === times[i - 1].e) run += 1;
      else {
        maxRun = Math.max(maxRun, run);
        run = 1;
      }
    }
    maxRun = Math.max(maxRun, run);
    return Math.max(1, maxRun);
  }, [slots]);

  // Allowed durations derived from service options, filtered by real availability for the chosen day
  const allowedDurations = useMemo(() => {
    // Base from active ServiceOptions (capped to 60 min in UI)
    const base = (serviceOptions || [])
      .map((o) => o.duration_minutes)
      .filter((d) => d <= 60);

    // If we have priced-day data, restrict to durations that actually have at least one slot
    if (base.length && pricedDayAll.length) {
      const present = new Set(pricedDayAll.map((p) => p.duration_minutes));
      const filtered = base.filter((d) => present.has(d));
      if (filtered.length) return Array.from(new Set(filtered)).sort((a, b) => a - b);
    }

    // Fallback: if priced-day not available yet, use base options; if no options, infer from chainable slots
    if (base.length) return Array.from(new Set(base)).sort((a, b) => a - b);
    return Array.from({ length: maxChainSlots }, (_, i) => (i + 1) * baseStepMinutes);
  }, [serviceOptions, pricedDayAll, baseStepMinutes, maxChainSlots]);

  // Pick a default duration when the day changes/loads
  useEffect(() => {
    if (!allowedDurations.length) return;
    if (!selectedDuration || !allowedDurations.includes(selectedDuration)) {
      setSelectedDuration(allowedDurations[0]);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [slotDayISO, allowedDurations, baseStepMinutes]);

  // The slots we’ll actually display (respecting selectedDuration)
  const displaySlots = useMemo(() => {
    if (!selectedDuration) return slots ?? [];
    // Prefer server-priced slots for the selected duration (accurate + respects availability)
    if (pricedSlots && pricedSlots.length) {
      return pricedSlots.map((p) => ({ start: p.start, end: p.end }));
    }
    // Fallback: chain contiguous public slots to reach the duration
    if (!slots) return [];
    if (selectedDuration === baseStepMinutes) return slots;
    return chainableStartsForDuration(slots, selectedDuration);
  }, [slots, selectedDuration, baseStepMinutes, pricedSlots]);

  // --- Appointments (tab-aware) ---
  const { data: appts, isLoading: apptsLoading } = useSWR<UserAppt[]>(
    ["/me/appts", tz],
    () => listMyAppointments("upcoming", tz),
    { refreshInterval: 8000, revalidateOnFocus: true }
  );

  const events = useMemo(() => {
    return (appts ?? []).map((a) => {
      const start = parseISO(a.start_utc);
      const end = parseISO(a.end_utc);
      const displayStart = toDisplayDate(a.start_utc, tz);
      const displayEnd = toDisplayDate(a.end_utc, tz);
      return {
        id: a.id,
        title: a.status === "canceled" ? "Canceled" : "Appointment",
        start,
        end,
        displayStart,
        displayEnd,
        resource: { type: "appointment", status: a.status, payment_status: a.payment_status },
      };
    });
  }, [appts, tz]);

  const slotEvents = useMemo(() => {
    return (displaySlots ?? []).map((s, i) => ({
      id: `slot-${i}-${s.start}`,
      title: "Available",
      start: parseISO(s.start),
      end: parseISO(s.end),
      resource: { type: "slot" as const },
    }));
  }, [displaySlots]);

  // ---- Holidays overlay for visible range ----
  function startOfDay(d: Date) { const x = new Date(d); x.setHours(0,0,0,0); return x; }
  function endOfDay(d: Date) { const x = new Date(d); x.setHours(23,59,59,999); return x; }
  function startOfMonth(d: Date) { return new Date(d.getFullYear(), d.getMonth(), 1); }
  function endOfMonth(d: Date) { return new Date(d.getFullYear(), d.getMonth()+1, 0, 23,59,59,999); }
  function getWeekRange(anchor: Date) {
    const day = anchor.getDay();
    const start = new Date(anchor); start.setDate(anchor.getDate() - day); start.setHours(0,0,0,0);
    const end   = new Date(start); end.setDate(start.getDate() + 6); end.setHours(23,59,59,999);
    return { start, end };
  }
  function getVisibleRange(view: View, currentDate: Date) {
    if (view === "day") return { start: startOfDay(currentDate), end: endOfDay(currentDate) };
    if (view === "week") return getWeekRange(currentDate);
    if (view === "month") return { start: startOfMonth(currentDate), end: endOfMonth(currentDate) };
    return getWeekRange(currentDate);
  }

  // Persist showHolidays in localStorage
  useEffect(() => {
    try { localStorage.setItem("client.showHolidays", showHolidays ? "1" : "0"); } catch {}
  }, [showHolidays]);

  useEffect(() => {
    const { start, end } = getVisibleRange(view, date);
    const yyyyMmDd = (d: Date) => {
      const y = d.getFullYear();
      const m = String(d.getMonth() + 1).padStart(2, "0");
      const dd = String(d.getDate()).padStart(2, "0");
      return `${y}-${m}-${dd}`;
    };
    let ignore = false;
    (async () => {
      try {
        if (showHolidays) {
          const hols: OwnerHoliday[] = await getPublicHolidays(yyyyMmDd(start), yyyyMmDd(end), tz);
          const evts = hols.map(h => ({
            id: `holiday-${h.date}`,
            title: `Holiday — ${h.name}`,
            start: parseISO(h.start_utc),
            end: parseISO(h.end_utc),
            resource: { type: "holiday" as const },
          }));
          if (!ignore) setHolidayEvents(evts);
        } else {
          if (!ignore) setHolidayEvents([]);
        }
      } catch (_e) {
        if (!ignore) setHolidayEvents([]);
      }
    })();
    return () => { ignore = true; };
  }, [view, date, tz, showHolidays]);

  // Load active service options once and pick a default duration
  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const opts = await listPublicServiceOptions();
        if (!alive) return;
        setServiceOptions(opts);
        if (!selectedDuration && opts.length) {
          setSelectedDuration(opts[0].duration_minutes);
        }
      } catch {
        // ignore
      }
    })();
    return () => {
      alive = false;
    };
  }, []);

  // Load all priced slots for the selected day (to determine which durations are truly available on that day)
  useEffect(() => {
    if (!slotPanelOpen || !slotDayISO) {
      setPricedDayAll([]);
      return;
    }
    let alive = true;
    (async () => {
      try {
        setPricedDayAll(await listPublicSlotsPriced(slotDayISO));
      } catch {
        if (alive) setPricedDayAll([]);
      }
    })();
    return () => {
      alive = false;
    };
  }, [slotPanelOpen, slotDayISO]);

  // Priced slots (when panel open + duration selected)
  useEffect(() => {
    if (!slotPanelOpen) return;
    if (!selectedDuration) return;
    let alive = true;
    (async () => {
      try {
        setPricedLoading(true);
        setPricedError(null);
        const rows = await listPublicSlotsPriced(slotDayISO, selectedDuration);
        if (!alive) return;
        setPricedSlots(rows);
      } catch (e: any) {
        if (!alive) return;
        setPricedError(e?.message || "Failed to load slots");
        setPricedSlots([]);
      } finally {
        if (alive) setPricedLoading(false);
      }
    })();
    return () => {
      alive = false;
    };
  }, [slotPanelOpen, slotDayISO, selectedDuration]);

  // --- Actions (with friendly messages) ---
  async function cancelAppt(id: string, reason: string, affectedDayISO?: string) {
    try {
      setActingId(id);
      await myCancelAppointment(id, reason);
      mutate(["/me/appts", tz]);
      mutate(["/public/slots", slotDayISO, tz]);
      if (affectedDayISO && affectedDayISO !== slotDayISO) {
        mutate(["/public/slots", affectedDayISO, tz]);
      }
    } catch (e: any) {
      const msg =
        e instanceof ApiError
          ? e.status === 403
            ? "Cancellations are locked less than 24 hours before your appointment. Please contact the owner."
            : e.status === 400
            ? e.message || "Please provide a cancellation reason."
            : e.message
          : "Sorry, we couldn’t cancel the appointment.";
      throw new Error(msg);
    } finally {
      setActingId(null);
    }
  }

  async function book(slot: PublicSlot) {
    try {
      const start = parseISO(slot.start);
      const end = parseISO(slot.end);
      const durMin = Math.round((end.getTime() - start.getTime()) / 60000);

      const recurNote = repeatWeekly ? ` weekly for ${repeatCount} occurrence(s)` : "";
      const yes = window.confirm(`Book ${format(start, "PPpp")} (${durMin} min)${recurNote}?`);
      if (!yes) return;

      setActingId(`${slot.start}-${slot.end}`);
      if (repeatWeekly) {
        const payload: any = {
          start_local: slot.start,
          duration_minutes: durMin,
          repeat_every_weeks: 1,
          lesson_person_name: slotPanelLessonFor.trim() || undefined,
        };
        payload.occurrences = Math.max(1, repeatCount | 0);
        const res = await myBookRecurringAppointments(payload);
        if (res?.conflicts?.length) {
          const lines = res.conflicts.slice(0, 5).map(c => `- ${c.start_local}: ${c.reason}`);
          alert(`Booked ${res.count} occurrence(s). ${res.conflicts.length} could not be booked.\n\nConflicts:\n${lines.join("\n")}${res.conflicts.length > 5 ? "\n…" : ""}`);
        }
      } else {
        await myBookAppointment({
          start_local: slot.start,
          duration_minutes: durMin,
          lesson_person_name: slotPanelLessonFor.trim() || undefined,
        });
      }

      mutate(["/me/appts", tz]);
      mutate(["/public/slots", slotDayISO, tz]);
      closeSlotPanel();
    } catch (e: unknown) {
      const msg =
        e instanceof ApiError
          ? e.status === 409
            ? "That slot was just taken. Please choose another time."
            : e.message
          : "Sorry, we couldn’t complete the booking.";
      setSlotPanelError(msg);
    } finally {
      setActingId(null);
    }
  }

  function openCancelDialogFor(apptId: string, start: Date, end: Date) {
    setCancelDialog({ id: apptId, start, end });
    setCancelReason("");
    setCancelError(null);
  }

  function onSelectEvent(evt: unknown) {
    const e = evt as { id?: string | number; start?: Date; end?: Date; displayStart?: Date; displayEnd?: Date; resource?: { type?: string; status?: string } };
    if (e?.resource?.type !== "appointment") return;
    if (e?.resource?.status === "canceled") return;
    const s = (e.displayStart as Date) || (e.start as Date);
    const en = (e.displayEnd as Date) || (e.end as Date);
    openCancelDialogFor(String(e.id), s, en);
  }

  return (
    <div className="grid grid-cols-1 gap-6">
      {/* Book button only (tabs removed) */}
      <div className="flex flex-wrap items-center justify-end gap-3">
        <button
          className="px-3 py-1 rounded bg-blue-600 text-white hover:bg-blue-700"
          onClick={() => {
            setSlotPanelOpen(true);
            setSlotPanelError(null);
            setSlotDayISO(format(new Date(), "yyyy-MM-dd"));
            setSlotPanelView("list");
            setSlotPanelLessonFor("");
          }}
        >
          Book new appointment
        </button>
      </div>

      {/* Calendar (appointments only) */}
      <section className="border rounded p-3">
        <div className="flex justify-between items-center mb-2">
          <h3 className="font-semibold">My Appointments</h3>
          <div className="flex items-center gap-4">
            <label className="flex items-center gap-2 text-sm select-none">
              <input
                type="checkbox"
                className="h-4 w-4"
                checked={showHolidays}
                onChange={(e) => setShowHolidays(e.target.checked)}
              />
              Show holidays
            </label>
            {apptsLoading && <span className="text-sm text-gray-500">Loading…</span>}
          </div>
        </div>
        <Calendar
          localizer={localizer}
          events={events}
          // Render using timezone-adjusted times when available
          startAccessor={(e: any) => e.displayStart ?? e.start}
          endAccessor={(e: any) => e.displayEnd ?? e.end}
          eventPropGetter={(event) => {
            const base: React.CSSProperties = { borderRadius: 8, opacity: 0.95 };
            const t = (event as unknown as { resource?: { type?: string; payment_status?: string } })?.resource?.type;
            if (t === "appointment") {
              const ps = (event as unknown as { resource?: { payment_status?: string } })?.resource?.payment_status;
              if (ps === "paid") {
                return { style: { ...base, backgroundColor: "#16a34a", color: "white" } };
              }
              return { style: { ...base, backgroundColor: "#1d4ed8", color: "white" } };
            }
            if (t === "slot") return { style: { ...base, backgroundColor: "#bfdbfe", color: "#1e3a8a" } };
            return { style: base };
          }}
          backgroundEvents={holidayEvents}
          scrollToTime={new Date(1970, 0, 1, 8, 0, 0)}
          style={{ height: 560 }}
          date={date}
          view={view}
          onView={(v) => setView(v)}
          onNavigate={(d) => setDate(d)}
          views={["day", "week", "month", "agenda"]}
          onSelectEvent={onSelectEvent}
          components={{
            event: ({ event }) => {
              const e = event as unknown as { id?: string | number; start?: Date; end?: Date; resource?: { type?: string; status?: string } };
              const type = e?.resource?.type;
              if (type === "holiday") {
                // Show holiday title; no actions
                return <span>{e?.title || "Holiday"}</span>;
              }
              if (type === "appointment") {
                const isCanceled = e?.resource?.status === "canceled";
                return (
                  <div className="flex items-center justify-between">
                    <span>{isCanceled ? "Canceled" : "Appointment"}</span>
                    {!isCanceled && (
                      <button
                        className="text-red-600 underline disabled:opacity-50"
                        disabled={actingId === String(e.id)}
                        onClick={(ev) => {
                          ev.stopPropagation();
                          openCancelDialogFor(String(e.id), e.start as Date, e.end as Date);
                        }}
                      >
                        Cancel
                      </button>
                    )}
                  </div>
                );
              }
              // Default
              return <span>{e?.title || "Event"}</span>;
            },
          }}
        />
      </section>

      {/* Slot side panel */}
      {slotPanelOpen && (
        <div className="fixed inset-0 z-50">
          <div
            className="absolute inset-0 bg-black/30"
            onClick={closeSlotPanel}
          />
          <aside className="absolute right-0 top-0 h-full w-full max-w-md bg-white shadow-xl flex flex-col">
            {/* Sticky header */}
            <div className="sticky top-0 z-10 border-b bg-white/95 backdrop-blur px-4 py-3 flex items-center justify-between">
              <div>
                <h3 className="text-base font-semibold">Book an appointment</h3>
                <p className="text-xs text-gray-500">
                  Choose a day, duration, then select a time below.
                </p>
              </div>
              <button className="text-sm underline" onClick={closeSlotPanel}>
                Close
              </button>
            </div>

            {/* Content */}
            <div className="p-4 overflow-y-auto grow">
              {/* Error banner */}
              {slotPanelError && (
                <div className="mb-3 rounded border border-red-300 bg-red-50 px-3 py-2 text-sm text-red-800">
                  {slotPanelError}
                </div>
              )}

              {/* Controls */}
              <div className="grid grid-cols-1 gap-3">
                {/* Pricing card first to avoid being cut off; inline in sidebar */}
                <div className="overflow-hidden">
                  <PriceList />
                </div>

                <div className="flex items-center gap-2">
                  <label className="text-sm text-gray-600">Day</label>
                  <input
                    type="date"
                    value={slotDayISO}
                    onChange={(e) => {
                      setSlotDayISO(e.target.value);
                      setSlotPanelError(null);
                    }}
                    className="border rounded px-2 py-1"
                  />

                  <div className="ml-auto flex items-center gap-2">
                    <button
                      className={`px-2 py-1 rounded text-sm ${
                        slotPanelView === "list" ? "bg-gray-900 text-white" : "border"
                      }`}
                      onClick={() => setSlotPanelView("list")}
                    >
                      List
                    </button>
                    <button
                      className={`px-2 py-1 rounded text-sm ${
                        slotPanelView === "calendar" ? "bg-gray-900 text-white" : "border"
                      }`}
                      onClick={() => setSlotPanelView("calendar")}
                    >
                      Calendar
                    </button>
                  </div>
                </div>

                {/* Duration selector + quick price heads-up */}
                {!!allowedDurations.length && (
                  <div className="flex items-center gap-2">
                    <span className="text-sm text-gray-600">Duration</span>
                  <div className="flex gap-1 flex-wrap">
                      {allowedDurations.map((min) => {
                        const price = servicePriceByDuration[min];
                        const title = price
                          ? `${min} min — $${(price.price_cents / 100).toFixed(2)}`
                          : `${min} min`;
                        return (
                          <button
                            key={min}
                            className={`px-2 py-1 rounded text-sm border ${
                              selectedDuration === min ? "bg-black text-white" : ""
                            }`}
                            onClick={() => setSelectedDuration(min)}
                            title={title}
                          >
                            {min}m
                          </button>
                        );
                      })}
                  </div>
                    {selectedDuration && servicePriceByDuration[selectedDuration] && (
                      <span className="ml-2 text-xs text-gray-500">
                        ${ (servicePriceByDuration[selectedDuration].price_cents / 100).toFixed(2) }
                      </span>
                    )}
                  </div>
                )}

                {/* Recurring options (weekly with occurrences only) */}
                <div className="flex items-center gap-2">
                  <label className="text-sm text-gray-600">Repeat</label>
                  <label className="flex items-center gap-2 text-sm select-none">
                    <input
                      type="checkbox"
                      className="h-4 w-4"
                      checked={repeatWeekly}
                      onChange={(e) => setRepeatWeekly(e.target.checked)}
                    />
                    Weekly
                  </label>
                  {repeatWeekly && (
                    <>
                      <label className="text-sm text-gray-600 ml-2">Occurrences</label>
                      <input
                        type="number"
                        min={1}
                        max={15}
                        value={repeatCount}
                        onChange={(e) => setRepeatCount(Math.max(1, Math.min(15, Number(e.target.value) || 1)))}
                        className="w-20 border rounded px-2 py-1 text-sm"
                      />
                    </>
                  )}
                </div>
              </div>

              {/* Lesson for (person) */}
              <div className="mt-3">
                <label className="block text-sm text-gray-700 mb-1">Lesson for (Person name)</label>
                <input
                  className="w-full border rounded px-2 py-2"
                  placeholder="e.g., Fluffy Junior"
                  value={slotPanelLessonFor}
                  onChange={(e) => setSlotPanelLessonFor(e.target.value)}
                />
              </div>

              {/* Client note to owner removed */}

              {/* Availability: List or Calendar */}
              <div className="mt-4">
                {slotPanelView === "list" ? (
                  <div className="border rounded p-3">
                    <div className="flex items-center justify-between mb-2">
                      <h4 className="font-medium">Available Slots</h4>
                      {slotsLoading && <div className="text-sm text-gray-500">Loading…</div>}
                    </div>
                    {!slotsLoading && (!displaySlots || displaySlots.length === 0) && (
                      <div className="text-gray-500">
                        No availability for this day{selectedDuration ? ` (${selectedDuration} min)` : ""}.
                      </div>
                    )}
                    <ul className="space-y-2">
                      {(displaySlots ?? []).map((s, i) => {
                        const start = parseISO(s.start);
                        const end = parseISO(s.end);
                        const durMin = Math.round((end.getTime() - start.getTime()) / 60000);
                        // Match price from server-priced slots for this exact slot
                        const priced = pricedSlots.find((p) => p.start === s.start && p.end === s.end);
                        const priceLabel = priced ? `$${(priced.price_cents / 100).toFixed(2)}` : null;

                        const busy = actingId === `${s.start}-${s.end}`;

                        return (
                          <li key={i} className="flex justify-between items-center border rounded px-3 py-2">
                            <div>
                              <div className="font-medium">{format(start, "PPpp")}</div>
                              <div className="text-sm text-gray-500">
                                Ends {format(end, "p")} • {durMin} min
                                {priceLabel ? ` · ${priceLabel}` : ""}
                              </div>
                            </div>
                            <button
                              onClick={() => book(s)}
                              disabled={busy}
                              className="bg-blue-600 text-white rounded px-3 py-1 hover:bg-blue-700 disabled:opacity-50"
                            >
                              {busy ? "Booking…" : "Book"}
                            </button>
                          </li>
                        );
                      })}
                    </ul>
                  </div>
                ) : (
                  <div className="border rounded p-3 -mx-3 sm:mx-0 overflow-x-auto">
                    <div className="min-w-[560px]">
                    <div className="flex items-center justify-between mb-2">
                      <h4 className="font-medium">Availability</h4>
                      {slotsLoading && <div className="text-sm text-gray-500">Loading…</div>}
                    </div>
                    {!slotsLoading && (!slots || slots.length === 0) && (
                      <div className="text-gray-500">No availability for this day.</div>
                    )}
                    <Calendar
                      localizer={localizer}
                      events={slotEvents}
                      startAccessor="start"
                      endAccessor="end"
                      scrollToTime={new Date(1970, 0, 1, 8, 0, 0)}
                      style={{ height: 420 }}
                      date={parseISO(`${slotDayISO}T00:00:00`)}
                      view="day"
                      toolbar={false}
                      onSelectEvent={(evt: any) => {
                        if (evt?.resource?.type !== "slot") return;
                        book({ start: evt.start.toISOString(), end: evt.end.toISOString() });
                      }}
                      eventPropGetter={(event) => {
                        if (event?.resource?.type === "slot") {
                          return { className: "bg-green-200 border-green-400 text-green-900" };
                        }
                        return {};
                      }}
                      components={{
                        event: ({ event }) =>
                          event?.resource?.type === "slot" ? <span>Available</span> : <span />,
                      }}
                    />
                    <p className="mt-2 text-xs text-gray-500">
                      Tip: Click an “Available” block to book that time.
                    </p>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </aside>
        </div>
      )}

      {cancelDialog && (
        <div className="fixed inset-0 z-50">
          <div
            className="absolute inset-0 bg-black/30"
            onClick={closeCancelDialog}
          />
          <div className="absolute inset-0 flex items-center justify-center px-4">
            <div className="w-full max-w-md overflow-hidden rounded-lg bg-white shadow-xl">
              <div className="flex items-center justify-between border-b px-4 py-3">
                <h3 className="text-base font-semibold">Cancel appointment</h3>
                <button
                  type="button"
                  className="text-sm underline"
                  onClick={closeCancelDialog}
                  disabled={cancelSubmitting}
                >
                  Close
                </button>
              </div>
              <div className="space-y-4 px-4 py-4">
                <p className="text-sm text-gray-600">
                  {`Scheduled for ${format(cancelDialog.start, "PPpp")} – ${format(cancelDialog.end, "p")}`}
                </p>
                <div>
                  <label className="mb-1 block text-sm font-medium text-gray-700">
                    Cancellation reason *
                  </label>
                  <textarea
                    className="w-full rounded-md border px-3 py-2 text-sm"
                    placeholder="Let the owner know why you need to cancel."
                    value={cancelReason}
                    onChange={(e) => setCancelReason(e.target.value)}
                    disabled={cancelSubmitting}
                    rows={4}
                  />
                </div>
                {cancelError && (
                  <div className="rounded border border-red-300 bg-red-50 px-3 py-2 text-sm text-red-700">
                    {cancelError}
                  </div>
                )}
                <div className="flex justify-end gap-2">
                  <button
                    type="button"
                    className="rounded-md border px-3 py-2 text-sm"
                    onClick={closeCancelDialog}
                    disabled={cancelSubmitting}
                  >
                    Keep appointment
                  </button>
                  <button
                    type="button"
                    className="rounded-md bg-red-600 px-3 py-2 text-sm font-medium text-white hover:bg-red-700 disabled:opacity-50"
                    onClick={async () => {
                      const reason = cancelReason.trim();
                      if (!reason) {
                        setCancelError("Please enter a cancellation reason.");
                        return;
                      }
                      setCancelError(null);
                      try {
                        setCancelSubmitting(true);
                        const dayISO = format(cancelDialog.start, "yyyy-MM-dd");
                        await cancelAppt(cancelDialog.id, reason, dayISO);
                        closeCancelDialog(true);
                      } catch (err: any) {
                        setCancelError(err?.message || "Unable to cancel the appointment.");
                      } finally {
                        setCancelSubmitting(false);
                      }
                    }}
                    disabled={cancelSubmitting}
                  >
                    {cancelSubmitting ? "Cancelling…" : "Cancel appointment"}
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
