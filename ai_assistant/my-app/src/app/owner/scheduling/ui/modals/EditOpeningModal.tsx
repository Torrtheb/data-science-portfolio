"use client";

import React, { useState } from "react";
import { updateOpening, deleteOpening } from "@/lib/api";
import { EditModalProps } from "../types";
import { emitSchedChanged } from "../hooks/schedBus";
import { ensureDate } from "../utils/datetime";


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


// --- Opening edit modal: now supports Save ---
export default function EditOpeningModal({ event, onClose, onUpdated }: EditModalProps) {
  // Normalize possible string|Date to Date to avoid TS/JS surprises
  const startDate = ensureDate(event.start);
  const endDate   = ensureDate(event.end);

  const [start, setStart]   = useState<string>(() => toInputLocal(startDate));
  const [end, setEnd]       = useState<string>(() => toInputLocal(endDate));

  const [slot, setSlot]     = useState<number>(30);
  const [buffer, setBuffer] = useState<number>(5);
  const [note, setNote]     = useState<string>(event.resource?.note || "");
  const [saving, setSaving] = useState(false);
  const [msg, setMsg]       = useState("");

  const save = async () => {
    setMsg("");
    setSaving(true);
    try {
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
      await updateOpening(event.id, {
        start: new Date(start).toISOString(),
        end:   new Date(end).toISOString(),
        slot_minutes: slot,
        buffer_minutes: buffer,
        note: note || null,
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
      await deleteOpening(event.id);
      emitSchedChanged();
      await onUpdated();
    } catch (e: unknown) {
      alert(e instanceof Error ? e.message : "Failed to delete opening.");
    }
  };

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto">
      <div className="min-h-full flex items-center justify-center p-4 bg-black/40">
        <div
          className="bg-white rounded-xl shadow-xl p-6 min-w-[340px] max-w-[90vw] w-full max-h-[90vh] overflow-y-auto space-y-3 min-h-0"
          style={{ WebkitOverflowScrolling: "touch" }}
        >
          <h3 className="text-lg font-semibold">Edit Opening</h3>

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
          <div>
            <label className="block text-sm mb-1">Slot minutes</label>
            <input
              type="number"
              min={5}
              step={5}
              max={120}
              className="border rounded px-2 py-1 w-full"
              value={slot}
              onChange={(e) => setSlot(Number(e.target.value))}
            />
          </div>
          <div>
            <label className="block text-sm mb-1">Buffer minutes</label>
            <input
              type="number"
              min={0}
              step={5}
              max={20}
              className="border rounded px-2 py-1 w-full"
              value={buffer}
              onChange={(e) => setBuffer(Number(e.target.value))}
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
