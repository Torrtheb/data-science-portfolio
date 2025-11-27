"use client";

import React, { useMemo, useState } from "react";
import { ownerUpdateAppointment, updateAppointment, cancelAppointment } from "@/lib/api";
import TimePicker12h from "@/components/TimePicker12h";

// -------------------- helpers --------------------
function pad2(n: number) { return String(n).padStart(2, "0"); }
function toYmd(d: Date) {
  return `${d.getFullYear()}-${pad2(d.getMonth() + 1)}-${pad2(d.getDate())}`;
}
function hhmmFromDate(d: Date) {
  return `${pad2(d.getHours())}:${pad2(d.getMinutes())}`;
}
function formatTo12h(d: Date) {
  let h = d.getHours();
  const m = d.getMinutes();
  const ampm = h >= 12 ? "PM" : "AM";
  h = h % 12; if (h === 0) h = 12;
  return `${h}:${pad2(m)} ${ampm}`;
}
function parse12hToHHMM(s: string) {
  const m = s.match(/^\s*(\d{1,2}):(\d{2})\s*([AP]M)\s*$/i);
  if (!m) throw new Error("Invalid time format");
  let h = parseInt(m[1], 10);
  const mm = m[2];
  const ap = m[3].toUpperCase();
  if (ap === "PM" && h < 12) h += 12;
  if (ap === "AM" && h === 12) h = 0;
  return `${pad2(h)}:${mm}`;
}

type Attendance = "unknown" | "attended" | "late" | "no_show";
type Payment = "unpaid" | "paid" | "refunded" | "waived";

// -------------------- Component --------------------
export default function EditPostDetailsModal({
  appt,
  onClose,
  onSaved,
}: {
  // Using any so we can accept extra fields (owner_note, attendance_status, etc.)
  appt: any;
  onClose: () => void;
  onSaved: () => void;
}) {
  // -------- Base scheduling fields
  const startDt = useMemo(() => new Date(appt.start_utc), [appt.start_utc]);
  const endDt = useMemo(() => new Date(appt.end_utc), [appt.end_utc]);
  const initialDuration = Math.max(15, Math.round((endDt.getTime() - startDt.getTime()) / 60000));

  const [dateYmd, setDateYmd] = useState<string>(toYmd(startDt));
  const [time12h, setTime12h] = useState<string>(formatTo12h(startDt));
  const [durationMin, setDurationMin] = useState<number>(initialDuration);
  const [clientEmail, setClientEmail] = useState<string>(appt.client?.email || "");

  // -------- Post fields
  const [note, setNote] = useState(appt.owner_note || appt.owner_private_note || "");
  const [attendance, setAttendance] = useState<Attendance>(appt.attendance_status || "attended");
  const [lateMin, setLateMin] = useState<number>(appt.late_minutes ?? (appt.late ? 5 : 0));
  const [payment, setPayment] = useState<Payment>(appt.payment_status || "unpaid");
  const [priceUsd, setPriceUsd] = useState<string>(
    typeof appt.price_override_cents === "number" ? (appt.price_override_cents / 100).toFixed(2) : ""
  );

  // (Bundles/wallet attach UI removed per new business rules)

  const [saving, setSaving] = useState(false);
  const [msg, setMsg] = useState("");

  // Owed preview (cash removed; wallet applied automatically elsewhere)
  const owedPreview = undefined as unknown as number | undefined;

  // (Bundle creation removed)

  async function save() {
    setMsg("");
    setSaving(true);
    try {
      // ---- Compute & detect base changes
      const origDate = toYmd(startDt);
      const origHHMM = hhmmFromDate(startDt);
      const hhmm = parse12hToHHMM(time12h);
      const baseChanged =
        clientEmail !== (appt.client?.email || "") ||
        dateYmd !== origDate ||
        hhmm !== origHHMM ||
        durationMin !== initialDuration;

      if (baseChanged) {
        const start_local = `${dateYmd}T${hhmm}:00`;
        await updateAppointment(appt.id, {
          client_email: clientEmail?.trim() || undefined,
          start_local,
          duration_minutes: durationMin,
          allow_override: false,
        });
      }

      // ---- Post fields payload
  const ownerBody: Partial<{
    owner_private_note: string;
    attendance_status: Attendance;
    late_minutes: number;
    payment_status: Payment;
    price_override_cents: number;
  }> = {
    owner_private_note: note || undefined,
    attendance_status: attendance,
    late_minutes: attendance === "late" ? Math.max(1, Number(lateMin) || 0) : 0,
    payment_status: payment,
  };
      if (priceUsd.trim() === "") {
        ownerBody.price_override_cents = 0; // clear override
      } else {
        const pc = Math.round(parseFloat(priceUsd) * 100);
        if (!Number.isNaN(pc)) ownerBody.price_override_cents = pc;
      }

      await ownerUpdateAppointment(appt.id, ownerBody);

      setMsg("Saved!");
      onSaved();
    } catch (e: unknown) {
      setMsg(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  }

  async function cancelNow() {
    setMsg("");
    try {
      setSaving(true);
      await cancelAppointment(appt.id);
      setMsg("Canceled");
      onSaved();
    } catch (e: unknown) {
      setMsg(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto">
      <div className="min-h-full flex items-center justify-center p-4 bg-black/40">
        <div
          className="bg-white rounded-xl shadow-xl p-6 min-w-[360px] max-w-[90vw] w-full max-h-[90vh] overflow-y-auto space-y-4 min-h-0"
          style={{ WebkitOverflowScrolling: "touch" }}
        >
          <h3 className="text-lg font-semibold">Edit appointment</h3>

        {/* Basic info (who & when) */}
        <div className="text-sm text-gray-600">
          {appt.client?.name || "—"} • {appt.client?.email || "—"}
          <br />
          {new Date(appt.start_utc).toLocaleString([], { dateStyle: "medium", timeStyle: "short" })}{" "}
          –{" "}
          {new Date(appt.end_utc).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Client email */}
          <div className="md:col-span-2">
            <label className="block text-sm font-medium mb-1">Client email</label>
            <input
              type="email"
              className="border rounded px-2 py-2 w-full"
              value={clientEmail}
              onChange={(e) => setClientEmail(e.target.value)}
              placeholder="name@example.com"
            />
          </div>

          {/* Date */}
          <div>
            <label className="block text-sm font-medium mb-1">Date</label>
            <input
              type="date"
              className="border rounded px-2 py-2 w-full"
              value={dateYmd}
              onChange={(e) => setDateYmd(e.target.value)}
            />
          </div>

          {/* Time */}
          <div>
            <label className="block text-sm font-medium mb-1">Start time</label>
            <TimePicker12h value={time12h} onChange={setTime12h} />
          </div>

          {/* Duration */}
          <div>
            <label className="block text-sm font-medium mb-1">Duration (min)</label>
            <input
              type="number"
              min={15}
              step={5}
              className="border rounded px-2 py-2 w-full"
              value={durationMin}
              onChange={(e) => setDurationMin(Math.max(15, Number(e.target.value) || 0))}
            />
          </div>
        </div>

        {/* Read-only appointment status inline with schedule */}
        <div className="md:col-span-2 text-sm">
          <span className="text-gray-600">Appointment status:</span>
          <span className="ml-2 inline-block px-2 py-0.5 rounded border capitalize">{String(appt.status || 'booked')}</span>
        </div>

        {/* Post fields */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Status is not editable here; use Cancel button to cancel */}
          <div className="md:col-span-2">
            <label className="block text-sm font-medium mb-1">Private note</label>
            <textarea
              className="border rounded px-2 py-1 w-full"
              rows={3}
              value={note}
              onChange={(e) => setNote(e.target.value)}
            />
          </div>

          <div>
            <label className="block text-sm font-medium mb-1">Attendance</label>
            <select
              className="border rounded px-2 py-2 w-full"
              value={attendance}
              onChange={(e) => setAttendance(e.target.value as Attendance)}
            >
              <option value="unknown">Unknown</option>
              <option value="attended">Attended</option>
              <option value="late">Late</option>
              <option value="no_show">No-show</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium mb-1">Late minutes</label>
            <input
              type="number"
              min={0}
              className="border rounded px-2 py-2 w-full"
              disabled={attendance !== "late"}
              value={lateMin}
              onChange={(e) => setLateMin(Number(e.target.value))}
            />
          </div>

          <div>
            <label className="block text-sm font-medium mb-1">Payment status</label>
            <select
              className="border rounded px-2 py-2 w-full"
              value={payment}
              onChange={(e) => setPayment(e.target.value as Payment)}
            >
              <option value="unpaid">Unpaid</option>
              <option value="paid">Paid</option>
              <option value="refunded">Refunded</option>
              <option value="waived">Waived</option>
            </select>
          </div>

          {/* Price + Amount paid */}
          <div>
            <label className="block text-sm font-medium mb-1">Price (USD) — optional</label>
            <input
              type="number"
              step="0.01"
              min="0"
              className="border rounded px-2 py-2 w-full"
              value={priceUsd}
              onChange={(e) => setPriceUsd(e.target.value)}
              placeholder="e.g., 60.00"
            />
            <p className="text-xs text-gray-500 mt-1">
              Leave blank to clear the override (will fallback to default price by duration).
            </p>
          </div>

          
        </div>

        {msg && (
          <p className={`text-sm ${msg.startsWith("Saved") ? "text-green-700" : "text-red-600"}`}>{msg}</p>
        )}

          {/* Cancel above Save/Close */}
          <div className="flex items-center gap-2">
            <button
              type="button"
              disabled={saving}
              onClick={cancelNow}
              className="px-3 py-2 bg-red-600 text-white rounded-md disabled:opacity-50"
            >
              {saving ? "Cancelling…" : "Cancel appointment"}
            </button>
            <button
              disabled={saving}
              className="px-4 py-2 bg-black text-white rounded-lg disabled:opacity-50"
              onClick={save}
            >
              {saving ? "Saving…" : "Save"}
            </button>
            <button className="px-4 py-2 bg-gray-200 rounded-lg" onClick={onClose}>
              Close
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
