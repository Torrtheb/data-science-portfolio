"use client";

import React, { useEffect, useState } from "react";
import {
  listOpenings,
  addOpening,
  deleteOpening,
  type SpecialOpening,
} from "@/lib/api";
import { Th, Td } from "../utils/table";
import { to24h, pad2, localDateTimeToISO, fromLocalInTZToUTC, toInputLocalInTZ } from "../utils/datetime";
import { emitSchedChanged } from "../hooks/schedBus";

export default function OpeningsPanel({ tz, hideHeaders = false }: { tz?: string; hideHeaders?: boolean }) {
  const [items, setItems] = useState<SpecialOpening[]>([]);
  const [msg, setMsg] = useState("");

  // split inputs
  const [startDate, setStartDate] = useState(""); // yyyy-mm-dd
  const [endDate, setEndDate] = useState(""); // yyyy-mm-dd

  // am/pm times
  const [sH, setSH] = useState(9),
    [sM, setSM] = useState(0),
    [sAP, setSAP] = useState<"AM" | "PM">("AM");
  const [eH, setEH] = useState(5),
    [eM, setEM] = useState(0),
    [eAP, setEAP] = useState<"AM" | "PM">("PM");

  const [slot, setSlot] = useState(30);
  const [buffer, setBuffer] = useState(5);
  const [note, setNote] = useState("");
  const [saving, setSaving] = useState(false);

  // Inline edit state for existing openings
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editStart, setEditStart] = useState<string>(""); // datetime-local
  const [editEnd, setEditEnd] = useState<string>("");     // datetime-local
  const [editSlot, setEditSlot] = useState<number>(30);
  const [editBuffer, setEditBuffer] = useState<number>(0);
  const [editNote, setEditNote] = useState<string>("");

  const reload = async () => {
    setMsg("");
    try {
      const rows = await listOpenings(tz);
      setItems(rows || []);
    } catch (e: any) {
      setMsg(`Error: ${e.message || String(e)}`);
    }
  };
  useEffect(() => {
    reload();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tz]);
  // useOnSchedChanged(reload);

  const hhmm = (h: number, m: number, ap: "AM" | "PM") => `${pad2(to24h(h, ap))}:${pad2(m)}`;
  const minuteOptions = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55];

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setMsg("");
    const st = hhmm(sH, sM, sAP);
    const et = hhmm(eH, eM, eAP);

    if (!startDate || !st || !endDate || !et) {
      setMsg("Please provide start date/time and end date/time.");
      return;
    }
    try {
      setSaving(true);
      const startIso = tz ? fromLocalInTZToUTC(`${startDate}T${st}`, tz) : localDateTimeToISO(startDate, st);
      const endIso = tz ? fromLocalInTZToUTC(`${endDate}T${et}`, tz) : localDateTimeToISO(endDate, et);
      if (new Date(endIso) <= new Date(startIso)) {
        setMsg("End must be after start.");
        setSaving(false);
        return;
      }
      // Guards: slot <=120, buffer <=20, non-negative
      if (slot < 0 || slot > 120) {
        setMsg("Slot minutes must be between 0 and 120.");
        setSaving(false);
        return;
      }
      if (buffer < 0 || buffer > 20) {
        setMsg("Buffer minutes must be between 0 and 20.");
        setSaving(false);
        return;
      }
      await addOpening({
        start: startIso,
        end: endIso,
        slot_minutes: slot,
        buffer_minutes: buffer,
        note: note || null,
      });
      // reset
      setStartDate("");
      setEndDate("");
      setSH(9);
      setSM(0);
      setSAP("AM");
      setEH(5);
      setEM(0);
      setEAP("PM");
      setNote("");
      await reload();
      setMsg("One-off opening added successfully!");
    } catch (e: any) {
      setMsg(`Error: ${e.message || String(e)}`);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="space-y-6">
      <details open className="bg-gray-50 rounded-xl p-4">
        <summary className="cursor-pointer text-base font-medium mb-3">One-off opening</summary>
        <form onSubmit={submit} className="space-y-3 mt-2">
          {/* Row 1: Start (date + time on one line) */}
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
                <option>AM</option>
                <option>PM</option>
              </select>
            </div>
          </div>

          {/* Row 2: End (date + time on one line) */}
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
                <option>AM</option>
                <option>PM</option>
              </select>
            </div>
          </div>

          {/* Row 3: Slot + Buffer */}
          <div className="flex flex-wrap items-end gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Slot Duration
              </label>
              <select
                value={slot}
                onChange={(e) => setSlot(Number(e.target.value))}
                className="border rounded-md px-3 py-2"
              >
                <option value={15}>15 min</option>
                <option value={30}>30 min</option>
                <option value={45}>45 min</option>
                <option value={60}>60 min</option>
                <option value={90}>90 min</option>
                <option value={120}>120 min</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Buffer Time
              </label>
              <select
                value={buffer}
                onChange={(e) => setBuffer(Number(e.target.value))}
                className="border rounded-md px-3 py-2"
              >
                <option value={0}>No buffer</option>
                <option value={5}>5 min</option>
                <option value={10}>10 min</option>
                <option value={15}>15 min</option>
                <option value={20}>20 min</option>
              </select>
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Note (Optional)
            </label>
            <input
              value={note}
              onChange={(e) => setNote(e.target.value)}
              placeholder="e.g., Special event"
              className="w-full border rounded-md px-3 py-2"
            />
          </div>

          <div className="flex items-center gap-3">
            <button
              disabled={saving}
              className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:opacity-50"
            >
              {saving ? "Adding…" : "Add Opening"}
            </button>
            {msg && (
              <p
                className={`text-sm ${
                  msg.includes("Error") ? "text-red-600" : "text-green-600"
                }`}
              >
                {msg}
              </p>
            )}
          </div>
        </form>
      </details>

      {/* existing table stays the same */}
      <details open className="bg-white rounded-xl border overflow-hidden">
        <summary className="cursor-pointer px-4 py-3 bg-gray-50 border-b font-medium">Existing One-off Openings</summary>
        
        <table className="w-full text-sm">
          <thead className="bg-gray-50">
            <tr>
              <Th>Start (local)</Th>
              <Th>End (local)</Th>
              <Th>Slot</Th>
              <Th>Buffer</Th>
              <Th>Note</Th>
              <Th></Th>
            </tr>
          </thead>
          <tbody>
            {items.map((o) => {
              const displayTz = o.timezone ?? tz;
              const s = new Date(o.start_utc);
              const e = new Date(o.end_utc);
              return (
                <tr key={o.id} className="border-t">
                  {editingId === o.id ? (
                    <>
                      <Td>
                        <input
                          type="datetime-local"
                          className="border rounded px-2 py-1 w-full"
                          value={editStart}
                          onChange={(e)=>setEditStart(e.target.value)}
                        />
                      </Td>
                      <Td>
                        <input
                          type="datetime-local"
                          className="border rounded px-2 py-1 w-full"
                          value={editEnd}
                          onChange={(e)=>setEditEnd(e.target.value)}
                        />
                      </Td>
                      <Td>
                        <input
                          type="number"
                          min={5}
                          step={5}
                          max={120}
                          className="border rounded px-2 py-1 w-full"
                          value={editSlot}
                          onChange={(e)=>setEditSlot(Number(e.target.value))}
                        />
                      </Td>
                      <Td>
                        <input
                          type="number"
                          min={0}
                          step={5}
                          max={20}
                          className="border rounded px-2 py-1 w-full"
                          value={editBuffer}
                          onChange={(e)=>setEditBuffer(Number(e.target.value))}
                        />
                      </Td>
                      <Td>
                        <input
                          type="text"
                          className="border rounded px-2 py-1 w-full"
                          value={editNote}
                          onChange={(e)=>setEditNote(e.target.value)}
                        />
                      </Td>
                      <Td className="text-right space-x-2">
                        <button
                          className="px-2 py-1 border rounded-md hover:bg-gray-50"
                          onClick={async ()=>{
                            try {
                              const startISO = tz ? fromLocalInTZToUTC(editStart, tz) : new Date(editStart).toISOString();
                              const endISO = tz ? fromLocalInTZToUTC(editEnd, tz) : new Date(editEnd).toISOString();
                              if (editSlot < 0 || editSlot > 120) {
                                alert("Slot minutes must be between 0 and 120.");
                                return;
                              }
                              if (editBuffer < 0 || editBuffer > 20) {
                                alert("Buffer minutes must be between 0 and 20.");
                                return;
                              }
                              const { updateOpening } = await import("@/lib/api");
                              await updateOpening(o.id, {
                                start: startISO,
                                end: endISO,
                                slot_minutes: editSlot,
                                buffer_minutes: editBuffer,
                                note: editNote || null,
                              });
                              setEditingId(null);
                              emitSchedChanged();
                              await reload();
                            } catch (e:any) {
                              alert(e.message || "Failed to update opening");
                            }
                          }}
                        >Save</button>
                        <button
                          className="px-2 py-1 border rounded-md hover:bg-gray-50"
                          onClick={()=>setEditingId(null)}
                        >Cancel</button>
                      </Td>
                    </>
                  ) : (
                    <>
                      <Td>
                        {s.toLocaleString([], { dateStyle: "medium", timeStyle: "short", ...(displayTz ? { timeZone: displayTz } : {}) })}
                      </Td>
                      <Td>
                        {e.toLocaleString([], { dateStyle: "medium", timeStyle: "short", ...(displayTz ? { timeZone: displayTz } : {}) })}
                      </Td>
                      <Td>{o.slot_minutes}m</Td>
                      <Td>{o.buffer_minutes}m</Td>
                      <Td>{o.note || "—"}</Td>
                      <Td className="text-right space-x-2">
                        <button
                          className="px-2 py-1 border rounded-md hover:bg-gray-50"
                          onClick={()=>{
                            // Prefill edit fields using owner tz if provided
                            const sLocal = tz ? toInputLocalInTZ(s, tz) : new Date(s.getTime()-s.getTimezoneOffset()*60000).toISOString().slice(0,16);
                            const eLocal = tz ? toInputLocalInTZ(e, tz) : new Date(e.getTime()-e.getTimezoneOffset()*60000).toISOString().slice(0,16);
                            setEditStart(sLocal);
                            setEditEnd(eLocal);
                            setEditSlot(o.slot_minutes);
                            setEditBuffer(o.buffer_minutes);
                            setEditNote(o.note || "");
                            setEditingId(o.id);
                          }}
                        >Edit</button>
                        <button
                          className="px-2 py-1 border rounded-md hover:bg-red-50 text-red-600"
                          onClick={async () => {
                            await deleteOpening(o.id);
                            emitSchedChanged();
                            await reload();
                          }}
                        >
                          Delete
                        </button>
                      </Td>
                    </>
                  )}
                </tr>
              );
            })}
            {!items.length && (
              <tr>
                <Td colSpan={6}>No one-off openings yet.</Td>
              </tr>
            )}
          </tbody>
        </table>
      </details>
    </div>
  );
}
