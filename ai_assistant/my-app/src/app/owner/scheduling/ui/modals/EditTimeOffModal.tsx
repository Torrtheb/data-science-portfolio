"use client";

import React, { useState } from "react";
import { updateTimeOff, deleteTimeOff } from "@/lib/api";
import { EditModalProps } from "../types";
import { emitSchedChanged } from "../hooks/schedBus";
import { ensureDate, toInputLocalInTZ, fromLocalInTZToUTC } from "../utils/datetime";

// --- Time off edit modal: now supports Save ---
export default function EditTimeOffModal({ event, onClose, onUpdated, tz }: EditModalProps & { tz?: string }) {
  // Normalize possible string|Date to Date
  const startDate = ensureDate(event.start);
  const endDate   = ensureDate(event.end);

  const [start, setStart] = useState<string>(() => tz ? toInputLocalInTZ(startDate, tz) : startDate.toISOString().slice(0, 16));
  const [end, setEnd]     = useState<string>(() => tz ? toInputLocalInTZ(endDate, tz)   : endDate.toISOString().slice(0, 16));
  const [note, setNote]   = useState<string>(event.resource?.note || "");
  const [saving, setSaving] = useState(false);
  const [msg, setMsg] = useState("");

  const save = async () => {
    setMsg("");
    setSaving(true);
    try {
      await updateTimeOff(event.id, {
        start: tz ? fromLocalInTZToUTC(start, tz) : new Date(start).toISOString(),
        end:   tz ? fromLocalInTZToUTC(end, tz)   : new Date(end).toISOString(),
        note:  note || null,
      });
      emitSchedChanged();
      await onUpdated();
    } catch (e: unknown) {
      setMsg(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  };

  const del = async () => {
    try {
      await deleteTimeOff(event.id);
      emitSchedChanged();
      await onUpdated();
    } catch (e: unknown) {
      alert(e instanceof Error ? e.message : "Failed to delete time off.");
    }
  };

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto">
      <div className="min-h-full flex items-center justify-center p-4 bg-black/40">
        <div
          className="bg-white rounded-xl shadow-xl p-6 min-w-[340px] max-w-[90vw] w-full max-h-[90vh] overflow-y-auto space-y-3 min-h-0"
          style={{ WebkitOverflowScrolling: "touch" }}
        >
          <h3 className="text-lg font-semibold">Edit Time Off</h3>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <div>
            <label className="block text-sm mb-1">Start</label>
            <input
              type="datetime-local"
              className="border rounded px-2 py-1 w-full"
              value={start}
              onChange={(e) => setStart(e.target.value)}
            />
          </div>
          <div>
            <label className="block text-sm mb-1">End</label>
            <input
              type="datetime-local"
              className="border rounded px-2 py-1 w-full"
              value={end}
              onChange={(e) => setEnd(e.target.value)}
            />
          </div>
        </div>

        <div>
          <label className="block text-sm mb-1">Note</label>
          <input
            className="border rounded px-2 py-1 w-full"
            value={note}
            onChange={(e) => setNote(e.target.value)}
          />
        </div>

        {msg && <p className="text-sm text-red-600">{msg}</p>}

        <div className="flex gap-2">
          <button
            disabled={saving}
            className="px-4 py-2 rounded bg-black text-white disabled:opacity-50"
            onClick={save}
          >
            {saving ? "Saving…" : "Save"}
          </button>
          <button className="px-4 py-2 rounded bg-red-600 text-white" onClick={del}>
            Delete
          </button>
          <button className="px-4 py-2 rounded bg-gray-200" onClick={onClose}>
            Close
          </button>
        </div>
        </div>
      </div>
    </div>
  );
}
