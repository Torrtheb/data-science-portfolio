// src/components/TimePicker12h.tsx
import React from "react";

type Props = {
  /** "HH:mm" 24h string (e.g. "09:30", "14:05"). Empty string allowed. */
  value: string;
  /** Called with new "HH:mm" (24h) string */
  onChange: (v: string) => void;
  /** Minute step options to display (default 5) */
  minuteStep?: 5 | 10 | 15 | 20 | 30;
  className?: string;
  disabled?: boolean;
  "aria-label"?: string;
};

function clamp(n: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, n));
}

function parseHHMM(v: string) {
  const m = /^(\d{1,2}):(\d{2})$/.exec(v || "");
  let HH = 0, mm = 0;
  if (m) {
    HH = clamp(parseInt(m[1], 10), 0, 23);
    mm = clamp(parseInt(m[2], 10), 0, 59);
  }
  return { HH, mm };
}

function toHHMM(HH: number, mm: number) {
  const h = String(clamp(HH, 0, 23)).padStart(2, "0");
  const m = String(clamp(mm, 0, 59)).padStart(2, "0");
  return `${h}:${m}`;
}

export default function TimePicker12h({
  value,
  onChange,
  minuteStep = 5,
  className,
  disabled,
  "aria-label": ariaLabel,
}: Props) {
  const { HH, mm } = parseHHMM(value);
  const isPM = HH >= 12;
  let hour12 = HH % 12;
  if (hour12 === 0) hour12 = 12;

  const hours = Array.from({ length: 12 }, (_, i) => i + 1);
  const minutes = Array.from({ length: Math.floor(60 / minuteStep) }, (_, i) =>
    String(i * minuteStep).padStart(2, "0")
  );

  const update = (next: { hour12?: number; minutes?: number; isPM?: boolean }) => {
    const h12 = clamp(next.hour12 ?? hour12, 1, 12);
    const mins = clamp(next.minutes ?? mm, 0, 59);
    const pm = next.isPM ?? isPM;

    let hh24 = h12 % 12;
    if (pm) hh24 += 12;
    // 12 AM => 00, 12 PM => 12 (handled by modulo above)
    onChange(toHHMM(hh24, mins));
  };

  return (
    <div className={className} aria-label={ariaLabel}>
      <div className="flex items-center gap-2">
        <select
          disabled={disabled}
          value={hour12}
          onChange={(e) => update({ hour12: parseInt(e.target.value, 10) })}
          className="border rounded-md px-2 py-1"
        >
          {hours.map((h) => (
            <option key={h} value={h}>{h}</option>
          ))}
        </select>
        :
        <select
          disabled={disabled}
          value={String(Math.round(mm / minuteStep) * minuteStep).padStart(2, "0")}
          onChange={(e) => update({ minutes: parseInt(e.target.value, 10) })}
          className="border rounded-md px-2 py-1"
        >
          {minutes.map((m) => (
            <option key={m} value={m}>{m}</option>
          ))}
        </select>
        <select
          disabled={disabled}
          value={isPM ? "PM" : "AM"}
          onChange={(e) => update({ isPM: e.target.value === "PM" })}
          className="border rounded-md px-2 py-1"
        >
          <option>AM</option>
          <option>PM</option>
        </select>
      </div>
    </div>
  );
}
