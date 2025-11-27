// src/app/owner/scheduling/ui/panels/TimeOffPanel.tsx
"use client";

import React, { useEffect, useState } from "react";
import {
  listTimeOff,
  listAppointments,
  listOpenings,
  addTimeOff,
  updateTimeOff,
  cancelAppointment,
  deleteOpening,
  deleteTimeOff,
} from "@/lib/api";
import { Th, Td } from "../utils/table";
import { to24h, pad2, localDateTimeToISO, toInputLocalInTZ, fromLocalInTZToUTC } from "../utils/datetime";
import { emitSchedChanged } from "../hooks/schedBus";

type TimeOffItem = {
  id: string;
  start_utc: string;
  end_utc: string;
  start_local?: string | null;
  end_local?: string | null;
  timezone?: string | null;
  note?: string | null;
};

export default function TimeOffPanel({ tz, hideHeaders = false }: { tz?: string; hideHeaders?: boolean }) {
  // date parts
  const [startDate, setStartDate] = useState<string>("");
  const [endDate, setEndDate] = useState<string>("");

  // am/pm times
  const [sH, setSH] = useState(9),
    [sM, setSM] = useState(0),
    [sAP, setSAP] = useState<"AM" | "PM">("AM");
  const [eH, setEH] = useState(5),
    [eM, setEM] = useState(0),
    [eAP, setEAP] = useState<"AM" | "PM">("PM");

  const [note, setNote] = useState<string>("");
  const [saving, setSaving] = useState(false);
  const [msg, setMsg] = useState("");

  const [items, setItems] = useState<TimeOffItem[]>([]);
  const [listMsg, setListMsg] = useState("");
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editStart, setEditStart] = useState<string>("");
  const [editEnd, setEditEnd] = useState<string>("");
  const [editNote, setEditNote] = useState<string>("");

  const reload = async () => {
    setListMsg("");
    try {
      const rows = (await listTimeOff(tz)) as any;
      setItems(rows || []);
    } catch (e: any) {
      setListMsg(`Error loading time off: ${e?.message || String(e)}`);
    }
  };
  useEffect(() => {
    reload();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tz]);
  // useOnSchedChanged(reload);

  const minuteOptions = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55];
  const hhmm = (h: number, m: number, ap: "AM" | "PM") => `${pad2(to24h(h, ap))}:${pad2(m)}`;

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setMsg("");

    const st = hhmm(sH, sM, sAP);
    const et = hhmm(eH, eM, eAP);

    if (!startDate || !st || !endDate || !et) {
      setMsg("Please provide start and end date/time.");
      return;
    }

    try {
      setSaving(true);

      // Interpret the entered date/time in the owner's timezone if provided
      const startISO = tz ? fromLocalInTZToUTC(`${startDate}T${st}`, tz) : localDateTimeToISO(startDate, st);
      const endISO = tz ? fromLocalInTZToUTC(`${endDate}T${et}`, tz) : localDateTimeToISO(endDate, et);
      const start = new Date(startISO);
      const end = new Date(endISO);

      if (end <= start) {
        setMsg("End must be after start.");
        return;
      }

      // Check conflicts
      const [appts, openings] = await Promise.all([listAppointments(tz), listOpenings(tz)]);
      const overlaps = (aS: Date, aE: Date, bS: Date, bE: Date) => aE > bS && aS < bE;

      const hitAppts = (appts as any[]).filter(
        (a: any) =>
          a.status !== "canceled" &&
          overlaps(start, end, new Date(a.start_utc), new Date(a.end_utc))
      );
      const hitOpens = (openings as any[]).filter((o: any) =>
        overlaps(start, end, new Date(o.start_utc), new Date(o.end_utc))
      );

      if (hitAppts.length || hitOpens.length) {
        const parts: string[] = [];
        if (hitAppts.length)
          parts.push(`${hitAppts.length} appointment${hitAppts.length > 1 ? "s" : ""}`);
        if (hitOpens.length)
          parts.push(`${hitOpens.length} opening${hitOpens.length > 1 ? "s" : ""}`);

        const ok = confirm(
          `This time off overlaps ${parts.join(", ")}.\n\nIf you continue I will:\n` +
            (hitAppts.length ? "• cancel the overlapping appointment(s)\n" : "") +
            (hitOpens.length ? "• delete the overlapping opening(s)\n" : "") +
            `\nProceed?`
        );
        if (!ok) {
          setSaving(false);
          return;
        }

        for (const a of hitAppts) await cancelAppointment(a.id);
        for (const o of hitOpens) await deleteOpening(o.id);
      }

      await addTimeOff({ start: startISO, end: endISO, note });
      setMsg("Time off added successfully!");
      // reset
      setStartDate(""); setEndDate("");
      setSH(9); setSM(0); setSAP("AM");
      setEH(5); setEM(0); setEAP("PM");
      setNote("");
      await reload();
    } catch (e: any) {
      setMsg(`Error: ${e.message || String(e)}`);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="space-y-6">
      <details open className="bg-gray-50 rounded-xl p-4">
        <summary className="cursor-pointer text-base font-medium mb-3">Time off</summary>

        <form onSubmit={submit} className="space-y-3 mt-2">
          {/* Row 1: Start (date + time inline) */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-0.5">Start</label>
            <div className="flex flex-wrap items-center gap-2">
              <input
                type="date"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
                className="border rounded-md px-3 py-2"
              />
              <select
                aria-label="Start hour"
                title="Hour"
                className="border rounded px-2 py-1 text-sm"
                value={sH}
                onChange={(e) => setSH(Number(e.target.value))}
              >
                {Array.from({ length: 12 }, (_, i) => i + 1).map((h) => (
                  <option key={h} value={h}>{h}</option>
                ))}
              </select>
              <select
                aria-label="Start minute"
                title="Minute"
                className="border rounded px-2 py-1 text-sm"
                value={sM}
                onChange={(e) => setSM(Number(e.target.value))}
              >
                {minuteOptions.map((m) => (
                  <option key={m} value={m}>{pad2(m)}</option>
                ))}
              </select>
              <select
                aria-label="AM or PM"
                title="AM / PM"
                className="border rounded px-2 py-1 text-sm"
                value={sAP}
                onChange={(e) => setSAP(e.target.value as "AM" | "PM")}
              >
                <option>AM</option><option>PM</option>
              </select>
            </div>
          </div>

          {/* Row 2: End (date + time inline) */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-0.5">End</label>
            <div className="flex flex-wrap items-center gap-2">
              <input
                type="date"
                value={endDate}
                onChange={(e) => setEndDate(e.target.value)}
                className="border rounded-md px-3 py-2"
              />
              <select
                aria-label="End hour"
                title="Hour"
                className="border rounded px-2 py-1 text-sm"
                value={eH}
                onChange={(e) => setEH(Number(e.target.value))}
              >
                {Array.from({ length: 12 }, (_, i) => i + 1).map((h) => (
                  <option key={h} value={h}>{h}</option>
                ))}
              </select>
              <select
                aria-label="End minute"
                title="Minute"
                className="border rounded px-2 py-1 text-sm"
                value={eM}
                onChange={(e) => setEM(Number(e.target.value))}
              >
                {minuteOptions.map((m) => (
                  <option key={m} value={m}>{pad2(m)}</option>
                ))}
              </select>
              <select
                aria-label="AM or PM"
                title="AM / PM"
                className="border rounded px-2 py-1 text-sm"
                value={eAP}
                onChange={(e) => setEAP(e.target.value as "AM" | "PM")}
              >
                <option>AM</option><option>PM</option>
              </select>
            </div>
          </div>

          {/* Note on its own line spanning all columns */}
            <div className="md:col-span-12">
              <label className="block text-sm font-medium text-gray-700 mb-0.5">
                Note (Optional)
              </label>
              <input
                type="text"
                value={note}
                onChange={(e) => setNote(e.target.value)}
                placeholder="e.g., Vacation"
                className="w-full border rounded-md px-3 py-2"
              />
            </div>
          

          <div className="flex items-center gap-3">
            <button
              type="submit"
              disabled={saving}
              className="px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 disabled:opacity-50"
            >
              {saving ? "Adding…" : "Add Time Off"}
            </button>
            {msg && (
              <p className={`text-sm ${msg.includes("Error") ? "text-red-600" : "text-green-600"}`}>
                {msg}
              </p>
            )}
          </div>
        </form>
      </details>

      {/* Existing time off list */}
      <details open className="bg-white rounded-xl border overflow-hidden">
        <summary className="cursor-pointer px-4 py-3 bg-gray-50 border-b flex items-center justify-between">
          <span className="font-medium">Existing Time Off</span>
          {listMsg && <span className="text-sm text-red-600">{listMsg}</span>}
        </summary>
        <table className="w-full text-sm">
          <thead className="bg-gray-50">
            <tr>
              <Th>Start (local)</Th>
              <Th>End (local)</Th>
              <Th>Note</Th>
              <Th className="text-right">Actions</Th>
            </tr>
          </thead>
          <tbody>
            {items.map((t) => {
              const s = new Date(t.start_utc);
              const e = new Date(t.end_utc);
              // Prefill editor using the owner's timezone if available
              const sLocal = tz ? toInputLocalInTZ(s, tz) : new Date(s.getTime() - s.getTimezoneOffset() * 60000).toISOString().slice(0, 16);
              const eLocal = tz ? toInputLocalInTZ(e, tz) : new Date(e.getTime() - e.getTimezoneOffset() * 60000).toISOString().slice(0, 16);
              const isEditing = editingId === t.id;
              return (
                <tr key={t.id} className="border-t">
                  <Td>
                    {isEditing ? (
                      <input type="datetime-local" className="border rounded px-2 py-1"
                        value={editStart} onChange={(e) => setEditStart(e.target.value)} />
                    ) : (
                      s.toLocaleString([], { dateStyle: "medium", timeStyle: "short", ...(tz ? { timeZone: tz } : {}) })
                    )}
                  </Td>
                  <Td>
                    {isEditing ? (
                      <input type="datetime-local" className="border rounded px-2 py-1"
                        value={editEnd} onChange={(e) => setEditEnd(e.target.value)} />
                    ) : (
                      e.toLocaleString([], { dateStyle: "medium", timeStyle: "short", ...(tz ? { timeZone: tz } : {}) })
                    )}
                  </Td>
                  <Td>
                    {isEditing ? (
                      <input type="text" className="border rounded px-2 py-1 w-full"
                        value={editNote} onChange={(e) => setEditNote(e.target.value)} />
                    ) : (
                      t.note || "—"
                    )}
                  </Td>
                  <Td className="text-right space-x-2">
                    {isEditing ? (
                      <>
                        <button
                          className="px-2 py-1 border rounded-md hover:bg-gray-50"
                          onClick={async () => {
                            try {
                              // Convert local inputs (owner-local) to ISO UTC strings
                              const sISO = tz ? fromLocalInTZToUTC(editStart, tz) : new Date(editStart).toISOString();
                              const eISO = tz ? fromLocalInTZToUTC(editEnd, tz) : new Date(editEnd).toISOString();
                              await updateTimeOff(t.id, { start: sISO, end: eISO, note: editNote || undefined });
                              setEditingId(null);
                              await reload();
                            } catch (e: any) {
                              alert(e.message || "Failed to update");
                            }
                          }}
                        >Save</button>
                        <button
                          className="px-2 py-1 border rounded-md hover:bg-gray-50"
                          onClick={() => {
                            setEditingId(null);
                            setEditStart(""); setEditEnd(""); setEditNote("");
                          }}
                        >Cancel</button>
                      </>
                    ) : (
                      <>
                        <button
                          className="px-2 py-1 border rounded-md hover:bg-gray-50"
                          onClick={() => {
                            setEditingId(t.id);
                            setEditStart(sLocal);
                            setEditEnd(eLocal);
                            setEditNote(t.note || "");
                          }}
                        >Edit</button>
                        <button
                          className="px-2 py-1 border rounded-md hover:bg-red-50 text-red-600"
                          onClick={async () => {
                            try {
                              await deleteTimeOff(t.id);
                              emitSchedChanged();
                              await reload();
                            } catch (e: any) {
                              alert(e.message || "Failed to delete");
                            }
                          }}
                        >Delete</button>
                      </>
                    )}
                  </Td>
                </tr>
              );
            })}
            {!items.length && (
              <tr>
                <Td colSpan={4}>No time off yet.</Td>
              </tr>
            )}
          </tbody>
        </table>
      </details>
    </div>
  );
}
