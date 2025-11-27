// src/app/owner/scheduling/ui/panels/WeeklyAvailabilityPanel.tsx
"use client";

import React, { useEffect, useMemo, useState, useCallback } from "react";
import { listAvailability, deleteAvailability, addAvailability, type AvailabilityRule } from "@/lib/api";

// (Optional) calendar preview
import { Calendar, dateFnsLocalizer, View } from "react-big-calendar";
import { Th, Td } from "../utils/table";
import { format, parse, startOfWeek, getDay, addDays, setHours, setMinutes } from "date-fns";
import { enCA } from "date-fns/locale";

type Props = { tz?: string };

const locales = { "en-CA": enCA };
const localizer = dateFnsLocalizer({
  format,
  parse,
  startOfWeek: () => startOfWeek(new Date(), { weekStartsOn: 1 }), // Monday
  getDay,
  locales,
});

// --- helpers ---------------------------------------------------------------

// backend sends weekday 0..6 (Mon..Sun). Convert that + "HH:MM[:SS]" to a Date in the current week
function toDateForWeekday(weekday: number, hhmm: string): Date {
  const now = new Date();
  const monday = startOfWeek(now, { weekStartsOn: 1 }); // Monday baseline
  const base = addDays(monday, weekday); // 0=Mon
  const [h, m] = (hhmm || "00:00").split(":").map((x) => parseInt(x, 10));
  return setMinutes(setHours(base, Number.isFinite(h) ? h : 0), Number.isFinite(m) ? m : 0);
}

function pad2(n: number) {
  return n < 10 ? `0${n}` : `${n}`;
}

function normTimeInput(v: string) {
  // Accept "H:MM" and "HH:MM[:SS]" → return "HH:MM"
  const m = v.trim().match(/^(\d{1,2}):(\d{2})(?::\d{2})?$/);
  if (!m) return v;
  return `${pad2(+m[1])}:${m[2]}`;
}

const WEEKDAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

// --- component ------


export default function WeeklyAvailabilityPanel({ tz, hidePreview = false }: Props & { hidePreview?: boolean }) {
  const [rules, setRules] = useState<AvailabilityRule[]>([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string>("");

  // form state (keep as strings to allow empty input while typing)
  const [weekday, setWeekday] = useState(0);
  const [startLocal, setStartLocal] = useState<string>("09:00");
  const [endLocal, setEndLocal] = useState<string>("17:00");
  const [slotStr, setSlotStr] = useState<string>("60");
  const [bufferStr, setBufferStr] = useState<string>("0");
  const [weeksStr, setWeeksStr] = useState<string>("8");
  const [success, setSuccess] = useState<string>("");

  const reload = useCallback(async () => {
    setLoading(true);
    setErr("");
    try {
      const data = await listAvailability();
      setRules(data);
    } catch (e: any) {
      setErr(e?.message || "Failed to load availability");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void reload();
  }, [reload]);

  const events = useMemo(
    () =>
      rules.map((r) => {
        const start = toDateForWeekday(r.weekday, r.start_local);
        const end = toDateForWeekday(r.weekday, r.end_local);
        return {
          id: `avail:${r.id}`, // purely UI
          title: `Available (${r.slot_minutes}m${r.buffer_minutes ? ` +${r.buffer_minutes}m buf` : ""})`,
          start,
          end,
          resource: {
            type: "availability" as const,
            ruleId: r.id, // ← carry FULL UUID here
            slot_minutes: r.slot_minutes,
            buffer_minutes: r.buffer_minutes,
          },
        };
      }),
    [rules]
  );

  function isValidHHMM(v: string) {
    const m = v.trim().match(/^([01]?\d|2[0-3]):([0-5]\d)$/);
    return !!m;
  }

  function clamp(n: number, min: number, max: number) {
    if (!Number.isFinite(n)) return min;
    return Math.min(max, Math.max(min, n));
  }

  async function handleAdd(ev: React.FormEvent) {
    ev.preventDefault();
    setErr("");
    try {
      // validate inputs
      const slot = clamp(parseInt((slotStr ?? "").trim(), 10), 0, 120);
      const buffer = clamp(parseInt((bufferStr ?? "").trim(), 10), 0, 20);
      const weeks = clamp(parseInt((weeksStr ?? "").trim(), 10), 0, 100);

      if (!isValidHHMM(startLocal)) {
        throw new Error("Start time must be in HH:MM");
      }
      if (!isValidHHMM(endLocal)) {
        throw new Error("End time must be in HH:MM");
      }
      // Optional: prevent negative values (already clamped), enforce maxes
      if (slot > 120) throw new Error("Slot minutes cannot exceed 120");
      if (buffer > 20) throw new Error("Buffer minutes cannot exceed 20");
      if (weeks > 100) throw new Error("Repeat weeks cannot exceed 100");

      // Add an availability rule so it appears in the list
      await addAvailability(
        {
          weekday,
          start_local: normTimeInput(startLocal),
          end_local: normTimeInput(endLocal),
          slot_minutes: slot,
          buffer_minutes: buffer,
        },
        tz
      );
      await reload();
      setSuccess("Weekly opening added");
      setTimeout(() => setSuccess(""), 3000);
    } catch (e: any) {
      setErr(e?.message || "Failed to create recurring openings");
    }
  }

  async function handleDeleteByRuleId(ruleId: string) {
    setErr("");
    try {
      await deleteAvailability(ruleId, tz); // ← use backend UUID directly
      await reload();
    } catch (e: any) {
      setErr(e?.message || "Failed to delete");
    }
  }

  // If user clicks on an event in the calendar preview, delete via resource.ruleId
  async function onSelectEvent(ev: any) {
    const ruleId = ev?.resource?.ruleId;
    if (!ruleId) return;
    if (confirm("Delete this weekly availability rule?")) {
      await handleDeleteByRuleId(ruleId);
    }
  }

  return (
    <div className="space-y-6">
      {/* Add form with collapsible wrapper */}
      <details open className="bg-gray-50 rounded-xl p-4">
      <summary className="cursor-pointer text-base font-medium mb-3">Weekly opening</summary>
      <form onSubmit={handleAdd} className="grid grid-cols-1 sm:grid-cols-7 gap-3 items-end mt-2">
        <div className="flex flex-col">
          <label className="text-sm font-medium">Weekday</label>
          <select
            className="border rounded px-2 py-1"
            value={weekday}
            onChange={(e) => setWeekday(parseInt(e.target.value, 10))}
          >
            {WEEKDAY_LABELS.map((lbl, idx) => (
              <option value={idx} key={lbl}>
                {lbl}
              </option>
            ))}
          </select>
        </div>

        <div className="flex flex-col">
          <label className="text-sm font-medium">Start (HH:MM)</label>
          <input
            className="border rounded px-2 py-1"
            value={startLocal}
            onChange={(e) => setStartLocal(e.target.value)}
            placeholder="09:00"
            inputMode="numeric"
            pattern="^([01]?\d|2[0-3]):([0-5]\d)$"
          />
        </div>

        <div className="flex flex-col">
          <label className="text-sm font-medium">End (HH:MM)</label>
          <input
            className="border rounded px-2 py-1"
            value={endLocal}
            onChange={(e) => setEndLocal(e.target.value)}
            placeholder="17:00"
            inputMode="numeric"
            pattern="^([01]?\d|2[0-3]):([0-5]\d)$"
          />
        </div>

        <div className="flex flex-col">
          <label className="text-sm font-medium">Slot (min)</label>
          <input
            type="number"
            className="border rounded px-2 py-1"
            value={slotStr}
            onChange={(e) => setSlotStr(e.target.value)}
            inputMode="numeric"
            placeholder="60"
            min={0}
            max={120}
          />
        </div>

        <div className="flex flex-col">
          <label className="text-sm font-medium">Buffer (min)</label>
          <input
            type="number"
            className="border rounded px-2 py-1"
            value={bufferStr}
            onChange={(e) => setBufferStr(e.target.value)}
            inputMode="numeric"
            placeholder="0"
            min={0}
            max={20}
          />
        </div>

        <div className="flex flex-col">
          <label className="text-sm font-medium">Repeat (weeks)</label>
          <input
            type="number"
            className="border rounded px-2 py-1"
            value={weeksStr}
            onChange={(e) => setWeeksStr(e.target.value)}
            inputMode="numeric"
            placeholder="8"
            min={0}
            max={100}
          />
        </div>

        <div className="sm:col-span-2 mt-6 sm:mt-0">
          <button
            type="submit"
            className="w-full border rounded px-3 py-2 bg-black text-white hover:opacity-90"
          >
            Add Weekly Openings
          </button>
        </div>
      </form>
      </details>

  {err && <p className="text-red-600 text-sm">{err}</p>}
  {!err && success && <p className="text-green-600 text-sm">{success}</p>}

      <p className="text-sm text-gray-600">
        Tip: This creates individual weekly openings you can edit/delete independently in the “Quick openings” tab.
      </p>

  {/* Table list similar to One-off opening, with bulk actions */}
  <WeeklyRulesSection
    rules={rules}
    loading={loading}
    onDeleteOne={handleDeleteByRuleId}
    tz={tz}
    onReload={reload}
  />

      {/* Optional: week calendar preview (click event to delete) */}
      {!hidePreview && (
        <div className="border rounded p-2">
          <div className="mb-2 text-sm text-gray-600">
            Calendar preview (click an availability block to delete it)
          </div>
          <Calendar
            localizer={localizer}
            events={events}
            // Mirror timezone-aware accessors (no-op for preview events)
            startAccessor={(e: any) => e.displayStart ?? e.start}
            endAccessor={(e: any) => e.displayEnd ?? e.end}
            scrollToTime={new Date(1970, 0, 1, 8, 0, 0)}
            defaultView={"week" as View}
            views={["week"]}
            style={{ height: 420 }}
            onSelectEvent={onSelectEvent}
          />
        </div>
      )}
    </div>
  );
}

function WeeklyRulesSection({
  rules,
  loading,
  onDeleteOne,
  tz,
  onReload,
}: {
  rules: AvailabilityRule[];
  loading: boolean;
  onDeleteOne: (id: string) => Promise<void> | void;
  tz?: string;
  onReload: () => Promise<void> | void;
}) {
  const [selected, setSelected] = React.useState<Record<string, boolean>>({});
  const [upSlot, setUpSlot] = React.useState<string>("");
  const [upBuf, setUpBuf] = React.useState<string>("");
  const [upStart, setUpStart] = React.useState<string>("");
  const [upEnd, setUpEnd] = React.useState<string>("");

  const toggleAll = (checked: boolean) => {
    const next: Record<string, boolean> = {};
    if (checked) for (const r of rules) next[r.id] = true;
    setSelected(next);
  };
  const anySelected = Object.values(selected).some(Boolean);

  async function bulkDelete() {
    const ids = Object.entries(selected).filter(([, v]) => v).map(([k]) => k);
    if (!ids.length) return;
    if (!confirm(`Delete ${ids.length} weekly opening${ids.length>1? 's' : ''}?`)) return;
    for (const id of ids) {
      await onDeleteOne(id);
    }
    setSelected({});
    await onReload();
  }

  async function bulkUpdate() {
    const ids = Object.entries(selected).filter(([, v]) => v).map(([k]) => k);
    const slot = parseInt(upSlot || "", 10);
    const buf = parseInt(upBuf || "", 10);
    const hasSlot = Number.isFinite(slot);
    const hasBuf = Number.isFinite(buf);
    const hasTimes = Boolean(upStart?.trim() || upEnd?.trim());
    if (!ids.length || (!hasSlot && !hasBuf && !hasTimes)) return;

    // Validate inputs
    const validTime = (v: string) => /^([01]?\d|2[0-3]):([0-5]\d)$/.test(v.trim());
    if (hasTimes) {
      if (upStart && !validTime(upStart)) {
        alert("Start must be HH:MM");
        return;
      }
      if (upEnd && !validTime(upEnd)) {
        alert("End must be HH:MM");
        return;
      }
    }
    if (hasSlot && (slot < 0 || slot > 120)) {
      alert("Slot minutes must be between 0 and 120");
      return;
    }
    if (hasBuf && (buf < 0 || buf > 20)) {
      alert("Buffer minutes must be between 0 and 20");
      return;
    }
    const parts: string[] = [];
    if (hasSlot) parts.push(`${slot} min slot`);
    if (hasBuf) parts.push(`${buf} min buffer`);
    if (hasTimes) parts.push(`times ${upStart || '(keep)'}–${upEnd || '(keep)'} (HH:MM)`);
    if (!confirm(`Update ${ids.length} rule${ids.length>1?'s':''}: ${parts.join(', ')}?`)) return;
    for (const id of ids) {
      const r = rules.find((x) => x.id === id);
      if (!r) continue;
      // delete + re-add with new slot/buffer
      await onDeleteOne(id);
      await addAvailability(
        {
          weekday: r.weekday,
          start_local: (upStart?.trim() ? normTimeInput(upStart.trim()) : r.start_local.slice(0,5)),
          end_local: (upEnd?.trim() ? normTimeInput(upEnd.trim()) : r.end_local.slice(0,5)),
          slot_minutes: hasSlot ? slot : r.slot_minutes,
          buffer_minutes: hasBuf ? buf : r.buffer_minutes,
        },
        tz
      );
    }
    setSelected({});
    setUpSlot(""); setUpBuf(""); setUpStart(""); setUpEnd("");
    await onReload();
  }

  return (
    <details open className="border rounded bg-white overflow-hidden">
      <summary className="cursor-pointer px-4 py-3 bg-gray-50 border-b text-sm font-medium">Weekly openings</summary>
      <div className="flex items-center justify-between px-4 py-2">
        <div className="text-sm text-gray-600">Bulk actions</div>
        <div className="flex items-center gap-2 text-sm">
          <input
            type="text"
            placeholder="Start HH:MM"
            className="w-28 border rounded px-2 py-1"
            value={upStart}
            onChange={(e)=>setUpStart(e.target.value)}
          />
          <input
            type="text"
            placeholder="End HH:MM"
            className="w-28 border rounded px-2 py-1"
            value={upEnd}
            onChange={(e)=>setUpEnd(e.target.value)}
          />
          <input
            type="number"
            className="w-20 border rounded px-2 py-1"
            placeholder="Slot"
            value={upSlot}
            onChange={(e) => setUpSlot(e.target.value)}
          />
          <input
            type="number"
            className="w-24 border rounded px-2 py-1"
            placeholder="Buffer"
            value={upBuf}
            onChange={(e) => setUpBuf(e.target.value)}
          />
          <button className="border rounded px-2 py-1 disabled:opacity-50" onClick={bulkUpdate} disabled={!anySelected}>
            Update selected
          </button>
          <button className="border rounded px-2 py-1 text-red-600 disabled:opacity-50" onClick={bulkDelete} disabled={!anySelected}>
            Delete selected
          </button>
        </div>
      </div>
      <table className="w-full text-sm">
        <thead className="bg-gray-50">
          <tr>
            <Th className="w-16">
              <input type="checkbox" onChange={(e) => toggleAll(e.target.checked)} />
            </Th>
            <Th>Weekday</Th>
            <Th>Start</Th>
            <Th>End</Th>
            <Th>Slot</Th>
            <Th>Buffer</Th>
            <Th className="text-right">Actions</Th>
          </tr>
        </thead>
        <tbody>
          {loading ? (
            <tr><Td colSpan={7}>Loading…</Td></tr>
          ) : rules.length === 0 ? (
            <tr><Td colSpan={7} className="text-gray-600">No weekly availability yet.</Td></tr>
          ) : (
            rules.map((r) => (
              <tr key={r.id} className="border-t">
                <Td className="w-16">
                  <input
                    type="checkbox"
                    checked={!!selected[r.id]}
                    onChange={(e) => setSelected((prev) => ({ ...prev, [r.id]: e.target.checked }))}
                  />
                </Td>
                <Td>{WEEKDAY_LABELS[r.weekday] ?? r.weekday}</Td>
                <Td>{r.start_local.slice(0, 5)}</Td>
                <Td>{r.end_local.slice(0, 5)}</Td>
                <Td>{r.slot_minutes} min</Td>
                <Td>{r.buffer_minutes} min</Td>
                <Td className="text-right">
                  <button
                    className="text-red-600 hover:underline"
                    onClick={() => {
                      if (confirm("Delete this weekly availability rule?")) {
                        void onDeleteOne(r.id);
                      }
                    }}
                  >
                    Delete
                  </button>
                </Td>
              </tr>
            ))
          )}
        </tbody>
      </table>
    </details>
  );
}
