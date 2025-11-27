"use client";

import React, { useState } from "react";
import { addAvailability } from "@/lib/api";
import { to24h, pad2 } from "../utils/datetime";

// If you already export WEEKDAYS somewhere else, feel free to import it.
// Keeping it local here is simplest and preserves behavior.
const WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

export default function AvailabilityForm({
  onSuccess,
  tz,
}: {
  onSuccess: () => void;
  tz?: string;
}) {
  const [weekday, setWeekday] = useState(0);

  // AM/PM values
  const [sH, setSH] = useState(9); // 1..12
  const [sM, setSM] = useState(0); // 0..55 step 5
  const [sAP, setSAP] = useState<"AM" | "PM">("AM");
  const [eH, setEH] = useState(5);
  const [eM, setEM] = useState(0);
  const [eAP, setEAP] = useState<"AM" | "PM">("PM");

  const [slot, setSlot] = useState(30);
  const [buffer, setBuffer] = useState(5);
  const [saving, setSaving] = useState(false);
  const [err, setErr] = useState<string>("");

  const buildHHMM = () => {
    const s = `${pad2(to24h(sH, sAP))}:${pad2(sM)}`;
    const e = `${pad2(to24h(eH, eAP))}:${pad2(eM)}`;
    return { start: s, end: e };
  };

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      setSaving(true);
      setErr("");
      const { start, end } = buildHHMM();
      if (end <= start) {
        setErr("End time must be after start time.");
        return;
      }
      if (slot <= 0) {
        setErr("Slot duration must be positive.");
        return;
      }
      await addAvailability(
        {
          weekday,
          start_local: start,
          end_local: end,
          slot_minutes: slot,
          buffer_minutes: buffer,
        },
        tz
      );

      onSuccess();
      // reset to defaults
      setSH(9);
      setSM(0);
      setSAP("AM");
      setEH(5);
      setEM(0);
      setEAP("PM");
      setSlot(30);
      setBuffer(5);
    } catch (e: any) {
      setErr(e.message || String(e));
    } finally {
      setSaving(false);
    }
  };

  const minuteOptions = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55];

  return (
    <form onSubmit={submit} className="grid grid-cols-2 md:grid-cols-6 gap-4">
      <div className="md:col-span-1">
        <label className="block text-sm font-medium text-gray-700 mb-1">Day</label>
        <select
          value={weekday}
          onChange={(e) => setWeekday(Number(e.target.value))}
          className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
        >
          {WEEKDAYS.map((d, i) => (
            <option key={i} value={i}>
              {d}
            </option>
          ))}
        </select>
      </div>

      {/* START — AM/PM ONLY */}
      <div className="md:col-span-2">
        <label className="block text-sm font-medium text-gray-700 mb-1">Start</label>
        <div className="flex gap-2">
          <select
            className="border rounded px-2 py-2"
            value={sH}
            onChange={(e) => setSH(Number(e.target.value))}
          >
            {Array.from({ length: 12 }, (_, i) => i + 1).map((h) => (
              <option key={h} value={h}>
                {h}
              </option>
            ))}
          </select>
          <select
            className="border rounded px-2 py-2"
            value={sM}
            onChange={(e) => setSM(Number(e.target.value))}
          >
            {minuteOptions.map((m) => (
              <option key={m} value={m}>
                {pad2(m)}
              </option>
            ))}
          </select>
          <select
            className="border rounded px-2 py-2"
            value={sAP}
            onChange={(e) => setSAP(e.target.value as "AM" | "PM")}
          >
            <option>AM</option>
            <option>PM</option>
          </select>
        </div>
      </div>

      <div className="md:col-span-2">
        <label className="block text-sm font-medium text-gray-700 mb-1">End</label>
        <div className="flex gap-2">
          <select
            className="border rounded px-2 py-2"
            value={eH}
            onChange={(e) => setEH(Number(e.target.value))}
          >
            {Array.from({ length: 12 }, (_, i) => i + 1).map((h) => (
              <option key={h} value={h}>
                {h}
              </option>
            ))}
          </select>
          <select
            className="border rounded px-2 py-2"
            value={eM}
            onChange={(e) => setEM(Number(e.target.value))}
          >
            {minuteOptions.map((m) => (
              <option key={m} value={m}>
                {pad2(m)}
              </option>
            ))}
          </select>
          <select
            className="border rounded px-2 py-2"
            value={eAP}
            onChange={(e) => setEAP(e.target.value as "AM" | "PM")}
          >
            <option>AM</option>
            <option>PM</option>
          </select>
        </div>
      </div>
      {/* END — AM/PM ONLY */}

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-1">Slot Duration</label>
        <select
          value={slot}
          onChange={(e) => setSlot(Number(e.target.value))}
          className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
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
        <label className="block text-sm font-medium text-gray-700 mb-1">Buffer Time</label>
        <select
          value={buffer}
          onChange={(e) => setBuffer(Number(e.target.value))}
          className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
        >
          <option value={0}>No buffer</option>
          <option value={5}>5 min</option>
          <option value={10}>10 min</option>
          <option value={15}>15 min</option>
          <option value={30}>30 min</option>
        </select>
      </div>

      {/* submit hidden to allow Enter key to submit; buttons are elsewhere if you prefer */}
      <button type="submit" className="hidden" aria-hidden />
      {err && (
        <div className="md:col-span-6 text-sm text-red-600" role="alert">
          {err}
        </div>
      )}
    </form>
  );
}
