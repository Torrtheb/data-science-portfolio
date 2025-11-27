// src/app/owner/scheduling/ui/panels/HomePanel.tsx
"use client";

import React, { useCallback, useEffect, useMemo, useState } from "react";
import { Calendar, dateFnsLocalizer, View } from "react-big-calendar";
import "react-big-calendar/lib/css/react-big-calendar.css"; // make sure CSS is imported somewhere (here is fine)

import { format, parse, startOfWeek, getDay } from "date-fns";
import { enUS } from "date-fns/locale";
import moment from "moment-timezone";

import {
  listAppointments,
  listOpenings,
  listTimeOff,
  addOpening,
  deleteTimeOff,
  addTimeOff,
  addAvailability,
  deleteAvailability,
  updateAppointment,
  cancelAppointment,
  listAvailability,
  type AvailabilityRule,
  getOwnerSettings,
  // getOwnerHolidays,   // switch to public to avoid auth edge cases
  getPublicHolidays,
  ownerGetClientDetail,
  ownerGetGroupDetails,
  adminUpdateGroupTime,
  adminGroupCancel,
  adminGroupRemoveAttendees,
  adminGroupAddAttendees,
} from "@/lib/api";

import { emitSchedChanged, useOnSchedChanged } from "../hooks/schedBus";
import ClientPicker from "@/components/ClientPicker";
import EditOpeningModal from "../modals/EditOpeningModal";
import EditTimeOffModal from "../modals/EditTimeOffModal";

// If you use to24h/pad2 inside EditAppointmentModal:
import { to24h, pad2, toInputLocalInTZ, fromLocalInTZToUTC } from "../utils/datetime";

// (re)create localizer here (or import from a shared place if you made one)
const locales = { "en-US": enUS };
const localizer = dateFnsLocalizer({ format, parse, startOfWeek, getDay, locales });
// add near the top (under existing imports)
function toInputLocal(dt: Date) {
  const pad = (n: number) => String(n).padStart(2, "0");
  const y = dt.getFullYear();
  const m = pad(dt.getMonth() + 1);
  const d = pad(dt.getDate());
  const h = pad(dt.getHours());
  const min = pad(dt.getMinutes());
  return `${y}-${m}-${d}T${h}:${min}`;
}

function toDisplayDate(utcIso: string, tz?: string, localIso?: string | null) {
  if (!tz) return new Date(utcIso);
  if (localIso) {
    const m = moment.parseZone(localIso);
    return new Date(m.format("YYYY-MM-DDTHH:mm:ss"));
  }
  const m = moment.utc(utcIso).tz(tz);
  return new Date(m.format("YYYY-MM-DDTHH:mm:ss"));
}


// minimal prop used here
type EditModalProps = { event: any; onClose: () => void; onUpdated: () => void };

type RbcEvent = {
  id: string;
  title: string;
  start: Date;
  end: Date;
  resource?: {
    type: "appointment" | "group" | "opening" | "time_off" | "availability";
    status?: "booked" | "completed" | "canceled" | null;
    client_name?: string | null;
    client_email?: string | null;
    note?: string | null;
    owner_note?: string | null;
    client_note?: string | null;
    paid?: boolean | null;
    late?: boolean | null;
    no_show?: boolean | null;
    amount_paid_cents?: number | null;
    labels?: string[] | null;
    group_id?: string | null;
  };
};

type ApptResource = {
  type: "appointment";
  status?: "booked" | "completed" | "canceled" | null;
  client_email?: string | null;
  client_name?: string | null;
  owner_note?: string | null;
  client_note?: string | null;
  paid?: boolean | null;
  late?: boolean | null;
  no_show?: boolean | null;
  amount_paid_cents?: number | null;
  labels?: string[] | null;
  price_override_cents?: number | null;
};

// -------------------- Home / Calendar (simplified to fix errors) --------------------

export default function HomePanel({ tz }: { tz?: string }) {  // ---- Types ----
  type RbcEvent = {
    id: string
    title: string
    start: Date
    end: Date
    displayStart?: Date
    displayEnd?: Date
    resource?: {
      type: "appointment" | "group" | "opening" | "time_off" | "availability" | "holiday"
      status?: "booked" | "completed" | "canceled" | null
      client_name?: string | null
      client_email?: string | null
      note?: string | null
      // --- admin metadata (appointments only; optional so it's safe on others) ---
      owner_note?: string | null
      client_note?: string | null
      paid?: boolean | null
      late?: boolean | null
      no_show?: boolean | null
      amount_paid_cents?: number | null
      price_override_cents?: number | null
      // optional payment summary (analytics/UI only)
      payment_status?: string | null
      labels?: string[] | null
      // appointment-specific optional fields (safe to include on union)
      client_account_id?: number | null
      lesson_person_id?: number | null
      lesson_name?: string | null
      lesson_email?: string | null
      group_id?: string | null
    }
  }
  type ApptResource = {
    type: "appointment"
    status?: "booked" | "completed" | "canceled" | null
    client_email?: string | null
    client_name?: string | null
    // --- admin metadata (appointments only; optional so it's safe on others) ---
    owner_note?: string | null
    client_note?: string | null
    paid?: boolean | null
    late?: boolean | null
    no_show?: boolean | null
    amount_paid_cents?: number | null
    labels?: string[] | null
    price_override_cents?: number | null;
    startUtc?: Date
    endUtc?: Date
    startLocalIso?: string | null
    endLocalIso?: string | null
    timezone?: string | null
    client_account_id?: number | null
    lesson_person_id?: number | null
    lesson_name?: string | null
    lesson_email?: string | null
  }

  // ---- State ----
  const [view, setView] = useState<View>("week")
  const [date, setDate] = useState<Date>(new Date())
  const [range, setRange] = useState<{ start: Date; end: Date } | null>(null)

  const [events, setEvents] = useState<RbcEvent[]>([])
  const [bgEvents, setBgEvents] = useState<RbcEvent[]>([])

  const [loading, setLoading] = useState<boolean>(true)
  const [err, setErr] = useState<string>("")
  const [showHolidays, setShowHolidays] = useState<boolean>(true)

  // Persist showHolidays in localStorage
  useEffect(() => {
    try {
      const v = localStorage.getItem("owner.showHolidays");
      if (v !== null) setShowHolidays(v === "1");
    } catch {}
     
  }, []);
  useEffect(() => {
    try { localStorage.setItem("owner.showHolidays", showHolidays ? "1" : "0"); } catch {}
  }, [showHolidays]);

  const [selectedEvent, setSelectedEvent] = useState<RbcEvent | null>(null)
  const [editMode, setEditMode] = useState<null | "appointment" | "opening" | "time_off">(null)
  const [edgePadMin, setEdgePadMin] = useState<number>(5);

  // Keep range in sync with the current view/date (fires on first mount too)
  useEffect(() => {
    const next = getVisibleRange(view, date);
    // Avoid infinite loops by only updating when actually different
    if (!range || next.start.getTime() !== range.start.getTime() || next.end.getTime() !== range.end.getTime()) {
      setRange(next);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [view, date]);

  // Prefer day view on small screens for better readability
  useEffect(() => {
    try {
      if (typeof window !== 'undefined' && window.innerWidth < 640) {
        setView('day');
      }
    } catch {}
  }, []);

  // ---- Helpers ----
  function isAppointmentEvent(e: RbcEvent): e is RbcEvent & { resource: ApptResource } {
    return e.resource?.type === "appointment"
  }
  function addDays(d: Date, n: number) { const x = new Date(d); x.setDate(x.getDate() + n); return x }
  function combineDateAndTime(baseDate: Date, hhmm: string) {
    const [h, m] = hhmm.split(":").map(Number)
    const d = new Date(baseDate)
    d.setHours(h, m ?? 0, 0, 0)
    return d
  }
  function startOfDay(d: Date) {
    const x = new Date(d); x.setHours(0,0,0,0); return x;
  }
  function endOfDay(d: Date) {
    const x = new Date(d); x.setHours(23,59,59,999); return x;
  }
  function startOfMonth(d: Date) {
    const x = new Date(d.getFullYear(), d.getMonth(), 1); x.setHours(0,0,0,0); return x;
  }
  function endOfMonth(d: Date) {
    const x = new Date(d.getFullYear(), d.getMonth()+1, 0); x.setHours(23,59,59,999); return x;
  }
  /** rbc uses Sun=0..Sat=6 for getDay; we want a week block similar to its default */
  function getWeekRange(anchor: Date) {
    const day = anchor.getDay(); // 0..6
    const start = new Date(anchor); start.setDate(anchor.getDate() - day); start.setHours(0,0,0,0);
    const end   = new Date(start); end.setDate(start.getDate() + 6); end.setHours(23,59,59,999);
    return { start, end };
  }
  function getVisibleRange(view: View, currentDate: Date) {
    if (view === "day")   return { start: startOfDay(currentDate), ...{ end: endOfDay(currentDate) } };
    if (view === "week")  return getWeekRange(currentDate);
    if (view === "month") return { start: startOfMonth(currentDate), end: endOfMonth(currentDate) };
    // agenda or anything else: use week as a reasonable default
    return getWeekRange(currentDate);
  }

  function onRangeChange(r: Date[] | { start: Date; end: Date }) {
    if (Array.isArray(r)) {
      const start = new Date(Math.min(...r.map(d => d.getTime())))
      const end = new Date(Math.max(...r.map(d => d.getTime())))
      setRange({ start, end })
    } else {
      setRange({ start: r.start, end: r.end })
    }
  }

  // ---- Load events for current range ----
  const load = async (windowStart?: Date, windowEnd?: Date, useHolidays?: boolean) => {
    setErr(""); setLoading(true)
    try {
      const evts: RbcEvent[] = []
      const timeOffWindows: Array<{ start: Date; end: Date }> = []
      // fetch settings first
      // UI smoothing: do not add extra edge padding around appointments on the owner calendar.
      // We still fetch settings for future use, but visually we set PAD=0 so openings/availability
      // butt directly against appointments without a 5-minute gap.
      await getOwnerSettings().catch(() => ({ appt_edge_buffer_min: 5 }));
      const PAD = 0;
      setEdgePadMin(PAD);


      // Always use canonical list endpoints so ids match CRUD endpoints
      const [appts, opens, timeoffs] = await Promise.all([
        listAppointments(tz),
        listOpenings(tz),
        listTimeOff(tz),
      ])

      // Build event list + collect busy windows first
      const busyWindows: Array<{ start: Date; end: Date }> = [];

      // Group aggregation: map group_id -> active attendees (exclude canceled)
      const groupMap = new Map<string, any[]>()
      for (const a of appts as any[]) {
        if ((a as any).status === 'canceled') continue
        const gid = (a as any).group_id as (string | null | undefined)
        if (gid) {
          const arr = groupMap.get(gid) || []
          arr.push(a)
          groupMap.set(gid, arr)
        }
      }

      // appointments → events + busy (non-group only; group seats are represented by a single aggregated event below)
      appts.forEach((a: any) => {
        if (a.status === "canceled") return;
        if ((a as any).group_id) return; // hide individual seats; we'll render one aggregated event per group
        const s = new Date(a.start_utc);
        const e = new Date(a.end_utc);
        const displayStart = toDisplayDate(a.start_utc, tz, a.start_local);
        const displayEnd = toDisplayDate(a.end_utc, tz, a.end_local);
        const lessonName = a.person?.name || null;
        const lessonEmail = a.person?.email || null;
        const title = a.client?.name ? `Appt: ${a.client.name}` : "Appointment"
        evts.push({
          id: a.id,
          title,
          start: s,
          end: e,
          displayStart,
          displayEnd,
          resource: {
            type: "appointment",
            status: a.status ?? null,
            client_name: a.client?.name ?? null,
            client_email: a.client?.email ?? null,
            client_account_id: a.client_account_id ?? null,
            lesson_person_id: a.person?.id ?? null,
            lesson_name: lessonName,
            lesson_email: lessonEmail,
            owner_note: a.owner_note ?? null,
            client_note: a.client_note ?? null,
            paid: a.paid ?? null,
            late: a.late ?? null,
            no_show: a.no_show ?? null,
            amount_paid_cents: a.amount_paid_cents ?? null,
            payment_status: (a as any).payment_status ?? null,
            labels: a.labels ?? null,
            price_override_cents: a.price_override_cents ?? null,
            startUtc: s,
            endUtc: e,
            startLocalIso: a.start_local ?? null,
            endLocalIso: a.end_local ?? null,
            timezone: a.timezone ?? tz ?? null,
          },
        });
        busyWindows.push({ start: s, end: e });
      });

      // Render one aggregated event per group
      for (const [gid, seats] of groupMap) {
        if (!seats.length) continue
        const first = seats[0]
        const s = new Date(first.start_utc)
        const e = new Date(first.end_utc)
        const displayStart = toDisplayDate(first.start_utc, tz, first.start_local)
        const displayEnd = toDisplayDate(first.end_utc, tz, first.end_local)
        const cnt = seats.length
        evts.push({
          id: `group-${gid}`,
          title: `👥 Group: ${cnt} attendee${cnt === 1 ? '' : 's'}`,
          start: s,
          end: e,
          displayStart,
          displayEnd,
          resource: {
            type: "group",
            status: first.status ?? null,
            group_id: gid,
            timezone: first.timezone ?? tz ?? null,
          } as any,
        })
        busyWindows.push({ start: s, end: e })
      }

      // time off → events + busy
      timeoffs.forEach((t: any) => {
        const s = new Date(t.start_utc);
        const e = new Date(t.end_utc);
        const displayStart = toDisplayDate(t.start_utc, tz, t.start_local);
        const displayEnd = toDisplayDate(t.end_utc, tz, t.end_local);
        timeOffWindows.push({ start: s, end: e });
        evts.push({
          id: t.id,
          title: t.note ? `Time Off — ${t.note}` : "Time Off",
          start: s,
          end: e,
          displayStart,
          displayEnd,
          resource: {
            type: "time_off",
            note: t.note ?? null,
            timezone: t.timezone ?? tz ?? null,
          },
        });
        busyWindows.push({ start: s, end: e });
      });

      // Holidays overlay for current visible range (background only)
      const includeHolidays = (useHolidays ?? showHolidays);
      if (includeHolidays && windowStart && windowEnd) {
        const yyyyMmDd = (d: Date) => {
          const y = d.getFullYear();
          const m = String(d.getMonth() + 1).padStart(2, "0");
          const dd = String(d.getDate()).padStart(2, "0");
          return `${y}-${m}-${dd}`;
        };
        try {
          const hols = await getPublicHolidays(yyyyMmDd(windowStart), yyyyMmDd(windowEnd));
          const bgs: RbcEvent[] = hols.map((h: any) => ({
            id: `holiday-${h.date}`,
            title: `Holiday — ${h.name}`,
            start: new Date(h.start_utc),
            end: new Date(h.end_utc),
            resource: { type: "holiday", note: h.name } as any,
          }));
          setBgEvents(bgs);
        } catch (e) {
          // Non-fatal if holidays fail
          console.warn("Failed to load holidays", e);
          setBgEvents([]);
        }
      } else {
        // Explicitly clear background holidays when disabled
        setBgEvents([]);
      }

      // edge pad (owner setting, fallback 5) — PAD is already computed above
      const paddedBusy = busyWindows.map(b => ({
        start: new Date(b.start.getTime() - PAD * 60000),
        end:   new Date(b.end.getTime()   + PAD * 60000),
      }));

      // helper stays here so openings + availability can use it
      function subtractIntervals(
        baseStart: Date,
        baseEnd: Date,
        cuts: Array<{ start: Date; end: Date }>
      ): Array<{ start: Date; end: Date }> {
        const norm = cuts
          .map(c => ({
            start: new Date(Math.max(baseStart.getTime(), c.start.getTime())),
            end:   new Date(Math.min(baseEnd.getTime(),   c.end.getTime())),
          }))
          .filter(c => c.end > c.start)
          .sort((a, b) => a.start.getTime() - b.start.getTime());

        const segments: Array<{ start: Date; end: Date }> = [];
        let cursor = new Date(baseStart);

        for (const c of norm) {
          if (c.start > cursor) segments.push({ start: new Date(cursor), end: new Date(c.start) });
          if (c.end > cursor) cursor = new Date(c.end);
          if (cursor >= baseEnd) break;
        }
        if (cursor < baseEnd) segments.push({ start: new Date(cursor), end: new Date(baseEnd) });
        return segments;
      }

      // openings → subtract from padded busy, then events
      const openingSegments: Array<{ start: Date; end: Date }> = [];
      opens.forEach((o: any) => {
        const oStart = new Date(o.start_utc);
        const oEnd   = new Date(o.end_utc);
        const remaining = subtractIntervals(oStart, oEnd, paddedBusy);
        for (const seg of remaining) {
          const displayStart = toDisplayDate(seg.start.toISOString(), tz, null);
          const displayEnd = toDisplayDate(seg.end.toISOString(), tz, null);
          // record segment as a cut so weekly availability does not render over it
          openingSegments.push({ start: seg.start, end: seg.end });
          evts.push({
            id: `open-${o.id}-${seg.start.toISOString()}`,
            title: o.note ? `Opening — ${o.note}` : "Opening",
            start: seg.start,
            end: seg.end,
            displayStart,
            displayEnd,
            resource: {
              type: "opening",
              note: o.note ?? null,
              timezone: o.timezone ?? tz ?? null,
            },
          });
        }
      });





      // Expand weekly availability across the visible range,
      // then subtract busy windows so availability never overlaps appointments/time-off.
      if (windowStart && windowEnd) {
        const rules = await listAvailability();

        // Iterate days in the visible window
        for (let d = new Date(windowStart); d <= windowEnd; d = addDays(d, 1)) {
          const js = d.getDay();               // Sun=0..Sat=6
          const weekdayMon0 = (js + 6) % 7;    // Mon=0..Sun=6

          for (const r of rules) {
            if (r.weekday !== weekdayMon0) continue;

            const baseStart = combineDateAndTime(d, r.start_local.slice(0, 5));
            const baseEnd   = combineDateAndTime(d, r.end_local.slice(0, 5));
            if (baseEnd <= baseStart) continue;

            // Collect only busy cuts that intersect this availability block
            const cutsBusy = paddedBusy.filter(b => b.end > baseStart && b.start < baseEnd);
            // Also subtract any opening segments so openings visually replace weekly availability
            const cutsOpen = openingSegments.filter(b => b.end > baseStart && b.start < baseEnd);
            const cuts = [...cutsBusy, ...cutsOpen];


            const remaining = subtractIntervals(baseStart, baseEnd, cuts);

            // Emit each remaining segment as its own availability event (still editable)
            for (const seg of remaining) {
              const displayStart = toDisplayDate(seg.start.toISOString(), tz, null);
              const displayEnd = toDisplayDate(seg.end.toISOString(), tz, null);
              evts.push({
                id: `avail-${r.id}-${seg.start.toISOString()}`,
                title: "Available",
                start: seg.start,
                end: seg.end,
                displayStart,
                displayEnd,
                resource: {
                  type: "availability",
                  note: `${r.slot_minutes}m slots${r.buffer_minutes ? `, ${r.buffer_minutes}m buffer` : ""}`,
                },
              });
            }
          }
        }
      }

      // Split into foreground and background:
      // Split by type, but we’ll render ALL of them as normal events
      const fgEvents: any[] = [];       // appointments, openings, time_off
      const availEvents: any[] = [];    // availability (already split by busy windows)


      for (const e of evts) {
        if (e?.resource?.type === "availability") availEvents.push(e);
        else fgEvents.push(e);
      }
      setEvents([...fgEvents, ...availEvents] as any);






    } catch (e: any) {
      setErr(e.message || String(e))
    } finally {
      setLoading(false)
    }
  }



  // Initial + whenever range or timezone changes
  useEffect(() => {
    if (!range) return;
    load(range.start, range.end);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [range, tz]);

  useOnSchedChanged(async () => {
    if (range) await load(range.start, range.end);
    else await load();
  });


  // ---- Styling per type ----
const eventPropGetter = (event: RbcEvent) => {
  const base: React.CSSProperties = { borderRadius: 8, opacity: 0.95 }
  switch (event.resource?.type) {
    case "group": {
      // Same color as appointments; ensure on top
      return { style: { ...base, backgroundColor: "#1d4ed8", color: "white", zIndex: 3 } }
    }
    case "appointment": {
      // Always blue for appointments; ensure on top of availability
      return { style: { ...base, backgroundColor: "#1d4ed8", color: "white", zIndex: 3 } }
    }
    case "opening":
      return { style: { ...base, backgroundColor: "#16a34a", color: "white", zIndex: 2 } }
    case "time_off":
      return { style: { ...base, backgroundColor: "#ef4444", color: "white", zIndex: 2 } }
    case "availability":
      return {
        //className: "rbc-availability",           // <-- add a class for CSS tweaks
        style: {
          ...base,
          backgroundColor: "#bfdbfe",            // darker than “today”
          color: "#1e3a8a",
          // use inset outline so width doesn't shrink
          boxShadow: "inset 0 0 0 1px #60a5fa",  // replaces border
          zIndex: 1,
        },
      }
    case "holiday":
      return {
        style: {
          ...base,
          backgroundColor: "#e9d5ff", // light purple
          color: "#6b21a8",          // purple-800 text
        },
      }
    default:
      return { style: base }
  }
}


  // ---- Interactions ----
  const onSelectEvent = (evt: RbcEvent) => {
    setSelectedEvent(evt)
    const t = evt.resource?.type
    if (t === "group") setEditMode("group" as any)
    else if (t === "appointment" || t === "opening" || t === "time_off") setEditMode(t);
    else if (t === "availability") setEditMode("availability_occurrence" as any);
    else setEditMode(null); // availability = read-only
  }


  // Quick-create a one-off opening by selecting a time range
  const onSelectSlot = async ({ start, end }: { start: Date; end: Date }) => {
    // ignore reverse/zero selections
    if (!start || !end || end <= start) return
    const startLocalStr = toInputLocal(start);
    const endLocalStr = toInputLocal(end);
    const startIsoUtc = tz ? fromLocalInTZToUTC(startLocalStr, tz) : start.toISOString();
    const endIsoUtc = tz ? fromLocalInTZToUTC(endLocalStr, tz) : end.toISOString();
    const startUtcDate = new Date(startIsoUtc);
    const endUtcDate = new Date(endIsoUtc);
    const fmt = (d: Date) =>
      tz
        ? d.toLocaleString([], { dateStyle: "medium", timeStyle: "short", timeZone: tz })
        : d.toLocaleString();
    const yes = confirm(`Create a one-off opening from\n${fmt(startUtcDate)} to ${fmt(endUtcDate)}?`)
    if (!yes) return
    try {
      await addOpening({
        start: startIsoUtc,
        end: endIsoUtc,
        slot_minutes: 30,
        buffer_minutes: 0,
        note: "Quick add",
      })
      // reload current range
      emitSchedChanged();            
      if (range) await load(range.start, range.end)
      else await load()
    } catch (e: any) {
      const msg = e?.message || String(e)
      if (/overlap/i.test(msg) && /time[\s-]?off/i.test(msg)) {
        const offs = await listTimeOff(tz)
        const overlapping = offs.filter(o =>
          new Date(o.end_utc) > startUtcDate && new Date(o.start_utc) < endUtcDate
        )
        const ok = confirm(
          `This opening overlaps ${overlapping.length} time-off block(s). ` +
          `Remove those time-off blocks and add the opening anyway?`
        )
        if (!ok) return

        for (const o of overlapping) await deleteTimeOff(o.id)

        await addOpening({
          start: startIsoUtc,
          end: endIsoUtc,
          slot_minutes: 30,
        buffer_minutes: 0,
          note: "Quick add",
        })

        emitSchedChanged()
        if (range) await load(range.start, range.end); else await load()
      } else {
        alert("Failed to create opening: " + msg)
      }
    }
  };

  const closeModal = () => { setSelectedEvent(null); setEditMode(null) }

  // ---- Group edit modal ----
  function EditGroupModal({ event, onClose, onUpdated, tz: tzProp }: { event: any; onClose: () => void; onUpdated: () => void; tz?: string }) {
    const groupId: string | null = event?.resource?.group_id || null
    const initLocal = tzProp ? toInputLocalInTZ(new Date(event.start), tzProp) : toInputLocal(new Date(event.start))
    const initDate = initLocal.slice(0,10)
    const initTime = initLocal.slice(11,16)
    const [dateStr, setDateStr] = useState(initDate)
    const [timeStr, setTimeStr] = useState(initTime)
    const [duration, setDuration] = useState<number>(Math.max(15, Math.round((new Date(event.end).getTime() - new Date(event.start).getTime())/60000)))
    const [saving, setSaving] = useState(false)
    const [msg, setMsg] = useState("")
    const [attendees, setAttendees] = useState<Array<{ appointment_id: string; person_id?: number | null; name: string; status: string; payment_status: string }>>([])
    const [loading, setLoading] = useState(false)
    const [clientPick, setClientPick] = useState<{ id: string; account_id: number; name?: string | null; email: string } | null>(null)
    const [availablePeople, setAvailablePeople] = useState<Array<{ id: number; full_name: string; email?: string | null }>>([])
    const [selectedToAdd, setSelectedToAdd] = useState<number[]>([])

    useEffect(() => {
      (async () => {
        if (!groupId) return;
        setLoading(true)
        try {
          const g = await ownerGetGroupDetails(groupId)
          setAttendees(g.attendees as any)
        } catch {}
        finally { setLoading(false) }
      })()
    }, [groupId])

    useEffect(() => {
      (async () => {
        if (!clientPick) { setAvailablePeople([]); setSelectedToAdd([]); return }
        try {
          const detail = await ownerGetClientDetail(clientPick.account_id)
          setAvailablePeople(detail.people || [])
          setSelectedToAdd([])
        } catch { setAvailablePeople([]); setSelectedToAdd([]) }
      })()
    }, [clientPick])

    async function save() {
      if (!groupId) return;
      setMsg(""); setSaving(true)
      try {
        const start_local = `${dateStr}T${timeStr}:00`
        await adminUpdateGroupTime(groupId, { start_local, duration_minutes: duration, confirm_if_conflicts: true })
        emitSchedChanged();
        await onUpdated();
        onClose();
      } catch (e:any) {
        setMsg(e?.message || "Failed to update group time")
      } finally {
        setSaving(false)
      }
    }

    async function cancelGroup() {
      if (!groupId) return;
      try {
        await adminGroupCancel(groupId)
        emitSchedChanged();
        await onUpdated();
        onClose();
      } catch (e:any) {
        setMsg(e?.message || "Failed to cancel group")
      }
    }

    async function removeAttendee(pid?: number | null, apptId?: string) {
      if (!groupId) return;
      try {
        if (pid && pid > 0) {
          await adminGroupRemoveAttendees(groupId, [pid])
        } else if (apptId) {
          await adminGroupRemoveAttendees(groupId, [], [apptId])
        } else {
          return;
        }
        const g = await ownerGetGroupDetails(groupId)
        setAttendees(g.attendees as any)
        emitSchedChanged();
        await onUpdated();
      } catch (e:any) {
        setMsg(e?.message || "Failed to remove attendee")
      }
    }

    return (
      <div className="fixed inset-0 z-50 overflow-y-auto">
        <div className="min-h-full flex items-center justify-center p-4 bg-black/40">
          <div className="bg-white rounded-xl shadow-xl p-6 min-w-[360px] max-w-[90vw] w-full max-h-[90vh] overflow-y-auto space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold">Edit Group Lesson</h3>
              <button className="text-sm" onClick={onClose}>✕</button>
            </div>
            {groupId && (
              <div className="text-xs text-gray-600">Group ID: {groupId.slice(0,8)}…</div>
            )}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
              <label className="text-sm">
                <div className="text-xs text-gray-600">Date</div>
                <input type="date" className="border rounded px-2 py-1 w-full" value={dateStr} onChange={(e)=>setDateStr(e.target.value)} />
              </label>
              <label className="text-sm">
                <div className="text-xs text-gray-600">Time</div>
                <input type="time" className="border rounded px-2 py-1 w-full" value={timeStr} onChange={(e)=>setTimeStr(e.target.value)} />
              </label>
              <label className="text-sm">
                <div className="text-xs text-gray-600">Duration (min)</div>
                <input type="number" min={15} step={15} className="border rounded px-2 py-1 w-full" value={duration} onChange={(e)=>setDuration(Math.max(15, Number(e.target.value)||60))} />
              </label>
            </div>
            <div className="flex items-center gap-2">
              <button type="button" className="rounded bg-black text-white px-3 py-1 text-sm disabled:opacity-50" disabled={saving} onClick={save}>{saving ? 'Saving…' : 'Save changes'}</button>
              <button type="button" className="rounded border px-3 py-1 text-sm" onClick={cancelGroup}>Cancel group</button>
              {msg && <span className="text-sm text-red-600">{msg}</span>}
            </div>
            <div className="space-y-2">
              <div className="text-sm font-medium">Attendees</div>
              {loading ? (<div className="text-sm text-gray-600">Loading…</div>) : (
                attendees.length ? attendees.map((a) => (
                  <div key={a.appointment_id} className="flex items-center justify-between rounded border bg-white px-2 py-1 text-sm">
                    <div className="flex items-center gap-2">
                      <span className="font-medium">{a.name}</span>
                      <span className="text-xs text-gray-600">{a.status}</span>
                      <span className="text-xs">{a.payment_status}</span>
                    </div>
                    <button className="text-red-600 underline text-xs" onClick={()=>removeAttendee((a as any).person_id, a.appointment_id)}>Remove</button>
                  </div>
                )) : (<div className="text-sm text-gray-600">No attendees</div>)
              )}
              <div className="pt-2 space-y-2">
                <div className="text-sm font-medium">Add attendees</div>
                <ClientPicker value={clientPick} onChange={setClientPick} placeholder="Search clients by name or email…" showEmailOnlyInInput />
                {clientPick && (
                  <div className="ml-2 space-y-1">
                    {availablePeople.length === 0 ? (
                      <div className="text-xs text-gray-600">No people on this account.</div>
                    ) : availablePeople.map(p => (
                      <label key={p.id} className="flex items-center gap-2 text-xs">
                        <input type="checkbox" checked={selectedToAdd.includes(p.id)} onChange={() => setSelectedToAdd(prev => prev.includes(p.id) ? prev.filter(id => id !== p.id) : [...prev, p.id])} />
                        <span>{p.full_name || p.email || `Person ${p.id}`}</span>
                      </label>
                    ))}
                    <div>
                      <button type="button" className="mt-1 rounded border px-2 py-1 text-xs" disabled={!selectedToAdd.length || !groupId} onClick={async()=>{
                        if (!groupId || !selectedToAdd.length) return;
                        try {
                          await adminGroupAddAttendees(groupId, selectedToAdd)
                          const g = await ownerGetGroupDetails(groupId)
                          setAttendees(g.attendees as any)
                          setClientPick(null); setAvailablePeople([]); setSelectedToAdd([])
                          emitSchedChanged();
                          await onUpdated();
                        } catch (e:any) {
                          setMsg(e?.message || 'Failed to add attendees')
                        }
                      }}>Add selected</button>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    )
  }

  function EditAvailabilityModal({ event, onClose, onUpdated }: EditModalProps) {
  // event.id was created as: `avail-${r.id}-${startIso}`
  const ruleIdMatch = /^avail-(.+?)-/.exec(event.id);
  const ruleId = ruleIdMatch?.[1];

  // --- Weekly rule fields (HH:mm only) ---
  const [start, setStart]   = useState(
    tz ? toInputLocalInTZ(new Date(event.start), tz).slice(11,16) : toInputLocal(new Date(event.start)).slice(11,16)
  ); // "HH:mm" in owner-local
  const [end, setEnd]       = useState(
    tz ? toInputLocalInTZ(new Date(event.end), tz).slice(11,16) : toInputLocal(new Date(event.end)).slice(11,16)
  );
  const [slot, setSlot]     = useState<number>(30);
  const [buffer, setBuffer] = useState<number>(0);

  // --- Single-date exception fields (datetime-local is local time) ---
  const [slotStart, setSlotStart] = useState(
    tz ? toInputLocalInTZ(new Date(event.start), tz) : toInputLocal(new Date(event.start))
  );

  const [slotDuration, setSlotDuration]         = useState<number>(30); // minutes to block/replace
  const [newSlotMinutes, setNewSlotMinutes]     = useState<number>(30); // for replacement opening
  const [newBufferMinutes, setNewBufferMinutes] = useState<number>(0);

  const [msg, setMsg] = useState("");

  // --- Helpers ---
  const hhmmValid = (v: string) => /^\d{2}:\d{2}$/.test(v);
  const toLocalDate = (dtLocal: string) => {
    // dtLocal like "2025-03-11T14:00" (no Z)
    if (tz) {
      const iso = fromLocalInTZToUTC(dtLocal, tz);
      return new Date(iso);
    }
    const d = new Date(dtLocal); // falls back to browser local
    if (isNaN(d.getTime())) throw new Error("Invalid date/time");
    return d;
  };

  // --- Save rule (delete + re-add) ---
  const save = async () => {
    try {
      setMsg("");
      if (!ruleId) throw new Error("Missing rule id");
      if (!hhmmValid(start) || !hhmmValid(end)) throw new Error("Use HH:mm for start/end.");
      if (end <= start) throw new Error("End must be after start.");
      if (slot <= 0) throw new Error("Slot must be positive.");
      if (buffer < 0) throw new Error("Buffer cannot be negative.");

      // Weekday: our rules use Mon=0..Sun=6; JS getDay() is Sun=0..Sat=6
      const weekdayMon0 = ((new Date(event.start).getDay() + 6) % 7);

      // Replace the rule by id
      await deleteAvailability(ruleId, tz);
      await addAvailability({
        weekday: weekdayMon0,
        start_local: start,
        end_local: end,
        slot_minutes: slot,
        buffer_minutes: buffer,
      }, tz);

      emitSchedChanged();
      await onUpdated();
    } catch (e:any) {
      setMsg(e.message || String(e));
    }
  };

  // --- Exception: block a single slot using micro time-off ---
  const blockOneSlot = async () => {
    try {
      setMsg("");
      const startLocal = toLocalDate(slotStart);
      const endLocal   = new Date(startLocal.getTime() + slotDuration * 60000);
      if (endLocal <= startLocal) throw new Error("Slot duration must be positive.");

      await addTimeOff({
        start: tz ? fromLocalInTZToUTC(slotStart, tz) : startLocal.toISOString(),
        end:   tz ? new Date(fromLocalInTZToUTC(toInputLocal(endLocal), tz)).toISOString() : endLocal.toISOString(),
        note: "Blocked single slot",
      });

      emitSchedChanged();
      await onUpdated();
    } catch (e:any) {
      setMsg(e.message || String(e));
    }
  };

  // --- Exception: replace a single slot (block + one-off opening) ---
  const replaceOneSlot = async () => {
    try {
      setMsg("");
      const startLocal = toLocalDate(slotStart);
      const endLocal   = new Date(startLocal.getTime() + slotDuration * 60000);
      if (endLocal <= startLocal) throw new Error("Slot duration must be positive.");
      if (newSlotMinutes <= 0) throw new Error("Replacement slot must be positive.");
      if (newBufferMinutes < 0) throw new Error("Replacement buffer cannot be negative.");

      // 1) Remove the weekly slot occurrence
      await addTimeOff({
        start: tz ? fromLocalInTZToUTC(slotStart, tz) : startLocal.toISOString(),
        end:   tz ? new Date(fromLocalInTZToUTC(toInputLocal(endLocal), tz)).toISOString() : endLocal.toISOString(),
        note: "Replace weekly slot",
      });

      // 2) Add a one-off opening in that window with new slot/buffer
      await addOpening({
        start: startLocal.toISOString(),
        end: endLocal.toISOString(),
        slot_minutes: newSlotMinutes,
        buffer_minutes: newBufferMinutes,
        note: "One-off replacement",
      });

      emitSchedChanged();
      await onUpdated();
    } catch (e:any) {
      setMsg(e.message || String(e));
    }
  };

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto">
      <div className="min-h-full flex items-center justify-center p-4 bg-black/40">
        <div className="bg-white rounded-xl shadow-xl p-6 min-w-[360px] max-w-[90vw] w-full max-h-[90vh] overflow-y-auto space-y-4">
        <h3 className="text-lg font-semibold">Edit Weekly Availability Rule</h3>

        {/* --- Edit the recurring rule (delete + re-add) --- */}
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="block text-sm mb-1">Start (HH:mm)</label>
            <input className="border rounded px-2 py-1 w-full"
              value={start} onChange={e=>setStart(e.target.value)} placeholder="09:00" />
          </div>
          <div>
            <label className="block text-sm mb-1">End (HH:mm)</label>
            <input className="border rounded px-2 py-1 w-full"
              value={end} onChange={e=>setEnd(e.target.value)} placeholder="17:00" />
          </div>
          <div>
            <label className="block text-sm mb-1">Slot minutes</label>
            <input type="number" min={5} step={5} className="border rounded px-2 py-1 w-full"
              value={slot} onChange={e=>setSlot(Number(e.target.value))} />
          </div>
          <div>
            <label className="block text-sm mb-1">Buffer minutes</label>
            <input type="number" min={0} step={5} className="border rounded px-2 py-1 w-full"
              value={buffer} onChange={e=>setBuffer(Number(e.target.value))} />
          </div>
        </div>

        <div className="flex gap-2">
          <button className="px-4 py-2 bg-black text-white rounded-lg" onClick={save}>Save rule</button>
          <button className="px-4 py-2 bg-gray-200 rounded-lg" onClick={onClose}>Close</button>
        </div>

        {/* --- Single-date exceptions --- */}
        <div className="border-t pt-4 space-y-3">
          <h4 className="font-medium">Exception for this date only</h4>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            <div className="md:col-span-2">
              <label className="block text-sm mb-1">Slot start (local)</label>
              <input type="datetime-local" className="border rounded px-2 py-1 w-full"
                value={slotStart} onChange={e=>setSlotStart(e.target.value)} />
            </div>
            <div>
              <label className="block text-sm mb-1">Slot duration (min)</label>
              <input type="number" min={5} step={5} className="border rounded px-2 py-1 w-full"
                value={slotDuration} onChange={e=>setSlotDuration(Number(e.target.value))} />
            </div>
          </div>

          <div className="flex gap-2">
            <button className="px-3 py-2 bg-red-600 text-white rounded-md" onClick={blockOneSlot}>
              Block this slot (time off)
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <div>
              <label className="block text-sm mb-1">Replacement slot minutes</label>
              <input type="number" min={5} step={5} className="border rounded px-2 py-1 w-full"
                value={newSlotMinutes} onChange={e=>setNewSlotMinutes(Number(e.target.value))} />
            </div>
            <div>
              <label className="block text-sm mb-1">Replacement buffer minutes</label>
              <input type="number" min={0} step={5} className="border rounded px-2 py-1 w-full"
                value={newBufferMinutes} onChange={e=>setNewBufferMinutes(Number(e.target.value))} />
            </div>
          </div>

          <div className="flex gap-2">
            <button className="px-3 py-2 bg-green-600 text-white rounded-md" onClick={replaceOneSlot}>
              Replace this slot (one-off opening)
            </button>
          </div>
        </div>

        {msg && <p className="text-sm text-red-600">{msg}</p>}
        </div>
      </div>
    </div>
  );
}

  // Group modal entry point
  if (selectedEvent && (selectedEvent.resource as any)?.type === 'group' && (editMode as any) === 'group') {
    return (
      <EditGroupModal
        event={selectedEvent}
        onClose={closeModal}
        onUpdated={async () => { if (range) await load(range.start, range.end); else await load(); }}
        tz={tz}
      />
    )
  }



  // ---- Appointment edit modal (reuses your updateAppointment signature) ----
  function EditAppointmentModal({
    event,
    onClose,
    onUpdated,
    tz: tzProp,
  }: {
    event: any
    onClose: () => void
    onUpdated: () => void
    tz?: string
  }) {
    // Only handle appointment events
    if (!(event?.resource?.type === "appointment")) { onClose(); return null }

    const [clientEmail, setClientEmail] = useState(event.resource.client_email || "")
    const clientName: string = event.resource.client_name || "—"
    const lessonName: string = event.resource.lesson_name || ""
    const lessonEmail: string = event.resource.lesson_email || ""
    const [lessonFor, setLessonFor] = useState(lessonName)
    const [lessonPersonId, setLessonPersonId] = useState<number | null>(event.resource.lesson_person_id || null)
    const clientAccountId: number | null = event.resource.client_account_id || null
    const [people, setPeople] = useState<Array<{ id: number; full_name: string; email?: string | null }>>([])

    useEffect(() => {
      (async () => {
        try {
          if (clientAccountId) {
            const detail = await ownerGetClientDetail(clientAccountId)
            setPeople(detail.people || [])
          }
        } catch {}
      })();
    }, [clientAccountId])

    // derive initial time pieces in the owner's timezone if provided
    const startLocalStr = tzProp
      ? toInputLocalInTZ(event.start ? new Date(event.start) : new Date(), tzProp)
      : toInputLocal(event.start ? new Date(event.start) : new Date());
    const initDateStr = startLocalStr.slice(0, 10); // yyyy-mm-dd
    const initTime24 = startLocalStr.slice(11, 16); // HH:mm

    const [dateStr, setDateStr] = useState(initDateStr)
    //const { timeFormat } = useTimeFormat()

    //const [time24, setTime24] = useState(initTime24)

    const initH = Number(initTime24.slice(0,2))
    const initM = Number(initTime24.slice(3,5))
    const [h12, setH12] = useState(((initH + 11) % 12) + 1)
    const [m12, setM12] = useState(initM - (initM % 5))
    const [ampm, setAmPm] = useState<"AM" | "PM">(initH >= 12 ? "PM" : "AM")

    const initialDuration =
      event.end && event.start
        ? Math.round((new Date(event.end).getTime() - new Date(event.start).getTime()) / 60000)
        : 30
    const [duration, setDuration] = useState<number>(initialDuration)
    // status is inferred by backend; not editable here

    // ✅ add state for the optional email message
    const [messageNote, setMessageNote] = useState<string>("")
    const [saving, setSaving] = useState(false)
    const [msg, setMsg] = useState("")

    // Group details
    const [groupId, setGroupId] = useState<string | null>(null)
    const [groupAttendees, setGroupAttendees] = useState<Array<{ appointment_id: string; name: string; status: string; payment_status: string; price_cents?: number | null; owed_cents: number }>>([])
    const [groupLoading, setGroupLoading] = useState(false)

    useEffect(() => {
      (async () => {
        try {
          const detail = await ownerGetAppointmentDetails(event.id)
          const gid = (detail as any).group_id as (string | null | undefined)
          setGroupId(gid || null)
          if (gid) {
            setGroupLoading(true)
            const g = await ownerGetGroupDetails(gid)
            setGroupAttendees(g.attendees.map(a => ({ appointment_id: a.appointment_id, name: a.name, status: a.status, payment_status: a.payment_status, price_cents: a.price_cents, owed_cents: a.owed_cents })))
          } else {
            setGroupAttendees([])
          }
        } catch {}
        finally { setGroupLoading(false) }
      })()
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [event?.id])

    const handleSubmit = async (e: React.FormEvent) => {
      e.preventDefault()
      setMsg("")
      if (!clientEmail || !dateStr) { setMsg("Client email and date are required."); return }

      const h = to24h(h12, ampm);
      const hhmm = `${pad2(h)}:${pad2(m12)}`;

      if (!/^\d{2}:\d{2}$/.test(hhmm)) { setMsg("Please provide a valid time."); return }

      try {
        setSaving(true)
        const start_local = `${dateStr}T${hhmm}:00`
        //await updateAppointment(event.id, {
        const payload: any = {
          client_email: clientEmail.trim(),
          start_local,
          duration_minutes: duration,
          // status not editable here; use Cancel button instead
          // ✅ include the optional message if provided
          message: messageNote.trim() || undefined,
        }
        if (lessonPersonId) payload.lesson_person_id = lessonPersonId
        else if (lessonFor && lessonFor.trim()) payload.lesson_person_name = lessonFor.trim()
        await updateAppointment(event.id, payload)
        setMsg("Updated!")
        onUpdated()
      } catch (e: any) {
        setMsg("Error: " + (e.message || String(e)))
      } finally {
        setSaving(false)
      }
    }

    async function cancelNow() {
      setMsg("")
      try {
        setSaving(true)
        await cancelAppointment(event.id, messageNote.trim() || undefined)
        setMsg("Canceled")
        onUpdated()
      } catch (e: any) {
        setMsg("Error: " + (e.message || String(e)))
      } finally {
        setSaving(false)
      }
    }

    return (
      <div className="fixed inset-0 z-50 overflow-y-auto">
        <div className="min-h-full flex items-center justify-center p-4 bg-black/40">
          <div className="bg-white rounded-xl shadow-xl p-6 min-w-[320px] max-w-[90vw] w-full max-h-[90vh] overflow-y-auto">
          <h3 className="text-lg font-semibold mb-2">Edit Appointment</h3>

          <form onSubmit={handleSubmit} className="space-y-4">
            {/* Group lesson section */}
            {groupId && (
              <div className="rounded-md border p-3 bg-indigo-50">
                <div className="flex items-center justify-between mb-2">
                  <div className="text-sm font-medium">Group lesson</div>
                  <div className="text-xs text-gray-600">Group ID: {groupId.slice(0,8)}…</div>
                </div>
                {groupLoading ? (
                  <div className="text-sm text-gray-600">Loading attendees…</div>
                ) : (
                  <div className="space-y-2">
                    {groupAttendees.length === 0 && <div className="text-sm">No attendees.</div>}
                    {groupAttendees.map((a) => (
                      <div key={a.appointment_id} className="flex items-center justify-between rounded border bg-white px-2 py-1 text-sm">
                        <div className="flex items-center gap-3">
                          <span className="font-medium">{a.name}</span>
                          <span className="text-xs text-gray-600">{a.status}</span>
                          <span className="text-xs">{a.payment_status}</span>
                          {typeof a.price_cents === 'number' && (
                            <span className="text-xs text-gray-600">${(a.price_cents/100).toFixed(2)}</span>
                          )}
                          {a.owed_cents > 0 && (
                            <span className="text-xs text-red-700">Owed ${(a.owed_cents/100).toFixed(2)}</span>
                          )}
                        </div>
                        <button
                          type="button"
                          className="text-red-600 underline"
                          onClick={async () => {
                            try {
                              const pid = Number((a as any).person_id) || 0;
                              const apptId = (a as any).appointment_id as string | undefined;
                              if (pid > 0) {
                                await adminGroupRemoveAttendees(groupId!, [pid])
                              } else if (apptId) {
                                await adminGroupRemoveAttendees(groupId!, [], [apptId])
                              } else {
                                throw new Error('Cannot determine attendee to remove')
                              }
                              const g = await ownerGetGroupDetails(groupId!)
                              setGroupAttendees(g.attendees as any)
                              onUpdated()
                            } catch (e: any) {
                              setMsg(e?.message || 'Failed to remove attendee')
                            }
                          }}
                        >Remove</button>
                      </div>
                    ))}
                    <div className="flex gap-2 pt-2">
                      <button type="button" className="rounded border px-2 py-1 text-xs" onClick={async () => {
                        try {
                          await adminGroupCancel(groupId!)
                          emitSchedChanged();
                          await onUpdated()
                          setMsg('Group canceled')
                        } catch (e:any) { setMsg(e?.message || 'Failed to cancel group') }
                      }}>Cancel group</button>
                    </div>
                  </div>
                )}
              </div>
            )}
            {/* Client details card */}
            <div className="rounded-md border p-3 bg-gray-50">
              <div className="text-sm font-medium mb-2">Client</div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                <div>
                  <label className="block text-xs text-gray-600 mb-1">Name</label>
                  <div className="px-2 py-1 border rounded bg-white select-text">
                    {clientName}
                  </div>
                </div>
                <div>
                  <label className="block text-xs text-gray-600 mb-1">Email</label>
                  <div className="px-2 py-1 border rounded bg-white overflow-x-auto select-text">
                    {clientEmail || "—"}
                  </div>
                </div>
              </div>
            </div>

            {/* Lesson for card */}
            <div className="rounded-md border p-3 bg-gray-50">
              <div className="text-sm font-medium mb-2">Lesson for</div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                <div>
                  <label className="block text-xs text-gray-600 mb-1">Name</label>
                  {people.length ? (
                    <>
                      <select className="border rounded px-2 py-1 w-full" value={lessonPersonId ?? ''}
                        onChange={(e)=>{
                          const v = e.target.value
                          if (v === '') {
                            setLessonPersonId(null);
                            setLessonFor('');
                          } else {
                            const idNum = Number(v);
                            setLessonPersonId(idNum);
                            const p = people.find(x=> x.id === idNum)
                            setLessonFor(p?.full_name || '');
                          }
                        }}>
                        <option value="">(custom)</option>
                        {people.map(p=> (
                          <option key={p.id} value={p.id}>{p.full_name}{p.email ? ` <${p.email}>` : ''}</option>
                        ))}
                      </select>
                      {lessonPersonId === null && (
                        <input
                          className="mt-2 border rounded px-2 py-1 w-full"
                          value={lessonFor}
                          onChange={(e)=>{ setLessonFor(e.target.value); }}
                          placeholder="Enter custom lesson name (e.g., Fluffy Junior)"
                        />
                      )}
                    </>
                  ) : (
                    <input className="border rounded px-2 py-1 w-full"
                      value={lessonFor}
                      onChange={(e)=>{ setLessonFor(e.target.value); setLessonPersonId(null); }}
                      placeholder="e.g., Fluffy Junior" />
                  )}
                </div>
                <div>
                  <label className="block text-xs text-gray-600 mb-1">Email</label>
                  <div className="px-2 py-1 border rounded bg-white overflow-x-auto select-text">
                    {lessonEmail || "—"}
                  </div>
                </div>
              </div>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                <div>
                  <label className="block text-sm font-medium mb-1">Client Email</label>
                  <input
                    type="email"
                    className="border rounded px-2 py-1 w-full"
                    value={clientEmail}
                    onChange={(e) => setClientEmail(e.target.value)}
                    placeholder="client@example.com"
                  />
                </div>

              <div>
                <label className="block text-sm font-medium mb-1">Date</label>
                <input
                  type="date"
                  className="border rounded px-2 py-1 w-full"
                  value={dateStr}
                  onChange={(e) => setDateStr(e.target.value)}
                />
              </div>

              <div className="flex items-end gap-2">
                <div>
                  <label className="block text-xs text-gray-600 mb-1">Hour</label>
                  <select className="border rounded px-2 py-1" value={h12} onChange={e=>setH12(Number(e.target.value))}>
                    {Array.from({length:12},(_,i)=>i+1).map(h=><option key={h} value={h}>{h}</option>)}
                  </select>
                </div>
                <div>
                  <label className="block text-xs text-gray-600 mb-1">Minute</label>
                  <select className="border rounded px-2 py-1" value={m12} onChange={e=>setM12(Number(e.target.value))}>
                    {[0,5,10,15,20,25,30,35,40,45,50,55].map(m=><option key={m} value={m}>{pad2(m)}</option>)}
                  </select>
                </div>
                <div>
                  <label className="block text-xs text-gray-600 mb-1">AM/PM</label>
                  <select className="border rounded px-2 py-1" value={ampm} onChange={e=>setAmPm(e.target.value as "AM"|"PM")}>
                    <option>AM</option><option>PM</option>
                  </select>
                </div>
              </div>


              <div>
                <label className="block text-sm font-medium mb-1">Duration (min)</label>
                <select
                  className="border rounded px-2 py-1 w-full"
                  value={duration}
                  onChange={(e) => setDuration(Number(e.target.value))}
                >
                  <option value={15}>15</option>
                  <option value={30}>30</option>
                  <option value={45}>45</option>
                  <option value={60}>60</option>
                  <option value={90}>90</option>
                </select>
              </div>

              {/* Status is not editable; use Cancel button to cancel */}

              {/* ✅ Put the new textarea INSIDE this grid and span both columns */}
              <div className="md:col-span-2">
                <label className="block text-sm font-medium mb-1">
                  Optional message to include in the email
                </label>
                <textarea
                  className="border rounded px-2 py-1 w-full"
                  rows={2}
                  value={messageNote}
                  onChange={(e) => setMessageNote(e.target.value)}
                  placeholder="Anything you'd like the client to know..."
                />
              </div>
            </div>

            {/* Cancel appointment button (uses optional message above) */}
            <div className="mb-2">
              <button
                type="button"
                className="px-3 py-2 bg-red-600 text-white rounded-md disabled:opacity-50"
                disabled={saving}
                onClick={cancelNow}
              >
                {saving ? "Cancelling…" : "Cancel appointment"}
              </button>
            </div>

            <div className="flex gap-2">
              <button
                disabled={saving}
                className="px-4 py-2 bg-black text-white rounded-lg disabled:opacity-50"
              >
                {saving ? "Saving…" : "Save"}
              </button>
              <button
                type="button"
                className="px-4 py-2 bg-gray-200 rounded-lg"
                onClick={onClose}
              >
                Close
              </button>
              {msg && (
                <span className={`text-sm ${msg.startsWith("Error") ? "text-red-600" : "text-green-700"}`}>
                  {msg}
                </span>
              )}
            </div>
          </form>
          </div>
        </div>
      </div>
    )
  }

  // Edit a single weekly availability occurrence: modify (carve out) or cancel
  function EditAvailabilityOccurrenceModal({ event, onClose, onUpdated, tz }: EditModalProps & { tz?: string }) {
    const [msg, setMsg] = useState("");
    const origStartLocal = tz ? toInputLocalInTZ(new Date(event.start), tz) : toInputLocal(new Date(event.start));
    const origEndLocal   = tz ? toInputLocalInTZ(new Date(event.end), tz) : toInputLocal(new Date(event.end));
    const [newStart, setNewStart] = useState<string>(origStartLocal);
    const [newEnd, setNewEnd]     = useState<string>(origEndLocal);

    const toISO = (s: string) => (tz ? fromLocalInTZToUTC(s, tz) : new Date(s).toISOString());

    const saveChanges = async () => {
      try {
        setMsg("");
        const oS = new Date(toISO(origStartLocal));
        const oE = new Date(toISO(origEndLocal));
        const nS = new Date(toISO(newStart));
        const nE = new Date(toISO(newEnd));
        if (!(oS < oE)) throw new Error("Invalid original window");
        if (!(nS < nE)) throw new Error("End must be after start");
        // must stay within original occurrence
        if (nS < oS || nE > oE) throw new Error("New times must be within the original slot");

        // carve out removed parts using time off
        if (nS > oS) {
          await addTimeOff({ start: oS.toISOString(), end: nS.toISOString(), note: "Adjust weekly slot (start)" });
        }
        if (nE < oE) {
          await addTimeOff({ start: nE.toISOString(), end: oE.toISOString(), note: "Adjust weekly slot (end)" });
        }
        emitSchedChanged();
        await onUpdated();
      } catch (e: any) {
        setMsg(e?.message || String(e));
      }
    };

    const cancelThis = async () => {
      try {
        setMsg("");
        const oS = new Date(toISO(origStartLocal));
        const oE = new Date(toISO(origEndLocal));
        await addTimeOff({ start: oS.toISOString(), end: oE.toISOString(), note: "Cancel weekly slot" });
        emitSchedChanged();
        await onUpdated();
      } catch (e: any) {
        setMsg(e?.message || String(e));
      }
    };

    return (
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40">
      <div className="bg-white rounded-xl shadow-xl p-6 min-w-[360px] max-w-[90vw] max-h-[90vh] overflow-y-auto space-y-3">
          <h3 className="text-lg font-semibold">Edit availability for this date</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <div>
              <label className="block text-sm mb-1">Start</label>
              <input type="datetime-local" className="border rounded px-2 py-1 w-full" value={newStart} onChange={(e)=>setNewStart(e.target.value)} />
            </div>
            <div>
              <label className="block text-sm mb-1">End</label>
              <input type="datetime-local" className="border rounded px-2 py-1 w-full" value={newEnd} onChange={(e)=>setNewEnd(e.target.value)} />
            </div>
          </div>
          {msg && <p className="text-sm text-red-600">{msg}</p>}
          <div className="flex gap-2">
            <button className="px-4 py-2 bg-black text-white rounded" onClick={saveChanges}>Save changes</button>
            <button className="px-4 py-2 bg-red-600 text-white rounded" onClick={cancelThis}>Cancel this slot</button>
            <button className="px-4 py-2 bg-gray-200 rounded" onClick={onClose}>Close</button>
          </div>
        </div>
      </div>
    );
  }



  return (
    <section className="space-y-3">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-medium">Calendar</h2>
        <div className="flex items-center gap-4">
          <label className="flex items-center gap-2 text-sm select-none">
            <input
              type="checkbox"
              className="h-4 w-4"
              checked={showHolidays}
              onChange={(e) => {
                const next = e.target.checked;
                setShowHolidays(next);
                // Reload current range immediately using the toggled value (avoid stale closure)
                if (range) load(range.start, range.end, next); else load(undefined, undefined, next);
              }}
            />
            Show holidays
          </label>
          <div className="text-sm text-gray-600">{loading ? "Loading…" : err ? <span className="text-red-600">{err}</span> : null}</div>
        </div>
      </div>

      <div className="-mx-4 sm:mx-0 overflow-x-auto rounded-xl border">
        <div className="min-w-[640px]">
        <Calendar
          localizer={localizer}
          events={events}
          // Render using timezone-adjusted times if available
          startAccessor={(e: any) => e.displayStart ?? e.start}
          backgroundEvents={bgEvents}
          dayLayoutAlgorithm="no-overlap"
          endAccessor={(e: any) => e.displayEnd ?? e.end}
          titleAccessor="title"
          scrollToTime={new Date(1970, 0, 1, 8, 0, 0)}
          defaultView="week"
          view={view}
          views={["month", "week", "day", "agenda"]}
          onView={(v) => {
            setView(v);
            const next = getVisibleRange(v, date);
            setRange(next);
          }}
          date={date}
          onNavigate={(d) => {
            setDate(d);
            const next = getVisibleRange(view, d);
            setRange(next);
          }}
          onRangeChange={onRangeChange}
          selectable
          onSelectEvent={onSelectEvent}
          onSelectSlot={onSelectSlot}
          style={{ height: "70vh" }}
          eventPropGetter={eventPropGetter}
        />
        </div>
      </div>

      {/* Inline modals */}
      {selectedEvent && editMode === "appointment" && (
        <EditAppointmentModal
          event={selectedEvent}
          onClose={closeModal}
          onUpdated={async () => { closeModal(); if (range) await load(range.start, range.end); else await load() }}
          tz={tz}
        />
      )}
      {selectedEvent && editMode === "opening" && (
        <EditOpeningModal
          event={selectedEvent}
          onClose={closeModal}
          onUpdated={async () => { closeModal(); if (range) await load(range.start, range.end); else await load() }}
        />
      )}
      {selectedEvent && editMode === "time_off" && (
        <EditTimeOffModal
          event={selectedEvent}
          onClose={closeModal}
          onUpdated={async () => { closeModal(); if (range) await load(range.start, range.end); else await load() }}
          tz={tz}
        />
      )}
      {selectedEvent && editMode === ("availability_occurrence" as any) && (
        <EditAvailabilityOccurrenceModal
          event={selectedEvent}
          onClose={closeModal}
          onUpdated={async () => { closeModal(); if (range) await load(range.start, range.end); else await load() }}
          tz={tz}
        />
      )}
    </section>
  )
}
