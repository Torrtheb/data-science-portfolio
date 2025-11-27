"use client";

import React, { useMemo, useState } from "react";
import ClientPicker from "@/components/ClientPicker";
import {
  AdminCreateRecurringAppointmentInput,
  AdminCreateRecurringAppointmentResponse,
  ApiError,
  adminCreateRecurringAppointments,
} from "@/lib/api";
import { emitSchedChanged } from "../hooks/schedBus";
import { pad2, to24h } from "../utils/datetime";

type Props = {
  onCreated?: (count: number) => void | Promise<void>;
};

type ConflictInfo = {
  start_local: string;
  conflicts: string[];
};

type ConfirmState = {
  human: string;
  conflicts: ConflictInfo[];
  pending: AdminCreateRecurringAppointmentInput;
};

function asConfirmState(error: unknown): ConfirmState | null {
  if (!(error instanceof ApiError)) return null;
  const detail = error.body?.detail;
  if (typeof detail !== "string" || !detail.startsWith("CONFIRM_REQUIRED:")) return null;
  const raw = detail.slice("CONFIRM_REQUIRED:".length);
  try {
    const parsed = JSON.parse(raw);
    const pending = parsed?.pending_http?.body as AdminCreateRecurringAppointmentInput | undefined;
    if (!pending) return null;
    return {
      human: typeof parsed?.human === "string" ? parsed.human : "Conflicts detected for one or more occurrences.",
      conflicts: Array.isArray(parsed?.conflicts)
        ? parsed.conflicts.map((row: unknown) => {
            const obj = (typeof row === "object" && row !== null) ? (row as Record<string, unknown>) : {};
            const startLocal = typeof obj.start_local === "string" ? obj.start_local : "";
            const conflictList = Array.isArray(obj.conflicts) ? obj.conflicts.map((c) => String(c)) : [];
            return { start_local: startLocal, conflicts: conflictList };
          })
        : [],
      pending,
    };
  } catch {
    return null;
  }
}

function formatLocalLabel(iso: string) {
  if (!iso) return iso;
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  return date.toLocaleString([], { dateStyle: "medium", timeStyle: "short" });
}

export default function OwnerCreateRecurringAppt({ onCreated }: Props) {
  const [clientPick, setClientPick] = useState<{ id: string; name?: string | null; email: string } | null>(null);
  const [clientEmail, setClientEmail] = useState("");
  const [clientName, setClientName] = useState("");

  const [startDate, setStartDate] = useState("");
  const [hour12, setHour12] = useState(4);
  const [minute, setMinute] = useState(0);
  const [ampm, setAmPm] = useState<"AM" | "PM">("PM");

  const [duration, setDuration] = useState(30);
  const [repeatInterval, setRepeatInterval] = useState(1);
  const [weeksCount, setWeeksCount] = useState(6);
  const [message, setMessage] = useState("");

  const [saving, setSaving] = useState(false);
  const [feedback, setFeedback] = useState("");
  const [err, setErr] = useState("");
  const [confirmState, setConfirmState] = useState<ConfirmState | null>(null);

  const minuteOptions = useMemo(() => [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55], []);

  const canSubmit = clientEmail && startDate && !saving;

  const resetForm = () => {
    setClientPick(null);
    setClientEmail("");
    setClientName("");
    setStartDate("");
    setHour12(4);
    setMinute(0);
    setAmPm("PM");
    setDuration(30);
    setRepeatInterval(1);
    setWeeksCount(6);
    setMessage("");
  };

  async function submit(payload: AdminCreateRecurringAppointmentInput) {
    const res: AdminCreateRecurringAppointmentResponse = await adminCreateRecurringAppointments(payload);
    setFeedback(`Created ${res.count} appointment${res.count === 1 ? "" : "s"}.`);
    setErr("");
    setConfirmState(null);
    resetForm();
    emitSchedChanged();
    if (onCreated) await onCreated(res.count);
  }

  const onSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    setErr("");
    setFeedback("");
    setConfirmState(null);

    if (!clientEmail || !startDate) {
      setErr("Client email and start date are required.");
      return;
    }

    const hh = to24h(hour12, ampm);
    const startLocal = `${startDate}T${pad2(hh)}:${pad2(minute)}:00`;
    const weeks = Math.max(1, Math.min(weeksCount, 104));
    const interval = Math.max(1, repeatInterval);
    const fallbackName = clientName.trim() || clientEmail.split("@")[0] || "Client";

    const payload: AdminCreateRecurringAppointmentInput = {
      client_name: fallbackName,
      client_email: clientEmail.trim(),
      start_local: startLocal,
      duration_minutes: duration,
      repeat_every_weeks: interval,
      occurrences: weeks,
      message: message.trim() ? message : undefined,
    };

    try {
      setSaving(true);
      await submit(payload);
    } catch (error) {
      const confirm = asConfirmState(error);
      if (confirm) {
        setConfirmState(confirm);
        setFeedback("");
        setErr("");
      } else {
        setErr(error instanceof Error ? error.message : "Failed to create recurring appointments.");
      }
    } finally {
      setSaving(false);
    }
  };

  const confirmAnyway = async () => {
    if (!confirmState) return;
    try {
      setSaving(true);
      await submit({
        ...confirmState.pending,
        confirm_if_conflicts: true,
      });
    } catch (error) {
      const confirm = asConfirmState(error);
      if (confirm) {
        setConfirmState(confirm);
        setFeedback("");
        setErr("");
      } else {
        setErr(error instanceof Error ? error.message : "Failed to create recurring appointments.");
      }
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="space-y-3">
      <form onSubmit={onSubmit} className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
        <div className="md:col-span-2 xl:col-span-2">
          <label className="block text-sm font-medium text-gray-700">Client *</label>
          <div className="mt-1 space-y-2">
            <ClientPicker
              value={clientPick}
              onChange={(hit) => {
                setClientPick(hit);
                setClientEmail(hit?.email ?? "");
                if (hit?.name) setClientName(hit.name);
              }}
              placeholder="Search by name or email…"
              minChars={1}
              showEmailOnlyInInput={false}
            />
            <input
              className="w-full rounded-md border px-3 py-2 text-sm"
              placeholder="Client email"
              value={clientEmail}
              onChange={(e) => setClientEmail(e.target.value)}
              required
            />
            <input
              className="w-full rounded-md border px-3 py-2 text-sm"
              placeholder="Client name (optional)"
              value={clientName}
              onChange={(e) => setClientName(e.target.value)}
            />
          </div>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700">Start date *</label>
          <input
            type="date"
            className="mt-1 w-full rounded-md border px-3 py-2 text-sm"
            value={startDate}
            onChange={(e) => setStartDate(e.target.value)}
            required
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700">Start time *</label>
          <div className="mt-1 grid grid-cols-3 gap-2">
            <select
              className="rounded-md border px-2 py-2 text-sm"
              value={hour12}
              onChange={(e) => setHour12(Number(e.target.value))}
            >
              {Array.from({ length: 12 }, (_, idx) => idx + 1).map((h) => (
                <option key={h} value={h}>
                  {h}
                </option>
              ))}
            </select>
            <select
              className="rounded-md border px-2 py-2 text-sm"
              value={minute}
              onChange={(e) => setMinute(Number(e.target.value))}
            >
              {minuteOptions.map((m) => (
                <option key={m} value={m}>
                  {pad2(m)}
                </option>
              ))}
            </select>
            <select
              className="rounded-md border px-2 py-2 text-sm"
              value={ampm}
              onChange={(e) => setAmPm(e.target.value as "AM" | "PM")}
            >
              <option value="AM">AM</option>
              <option value="PM">PM</option>
            </select>
          </div>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700">Duration (minutes)</label>
          <select
            className="mt-1 w-full rounded-md border px-3 py-2 text-sm"
            value={duration}
            onChange={(e) => setDuration(Number(e.target.value))}
          >
            {[15, 30, 45, 60, 75, 90, 120].map((d) => (
              <option key={d} value={d}>
                {d}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700">Repeat every</label>
          <select
            className="mt-1 w-full rounded-md border px-3 py-2 text-sm"
            value={repeatInterval}
            onChange={(e) => setRepeatInterval(Number(e.target.value))}
          >
            {[1, 2, 3, 4].map((w) => (
              <option key={w} value={w}>
                {w} week{w > 1 ? "s" : ""}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700">Number of lessons</label>
          <input
            type="number"
            min={1}
            max={104}
            className="mt-1 w-full rounded-md border px-3 py-2 text-sm"
            value={weeksCount}
            onChange={(e) => setWeeksCount(Number(e.target.value))}
          />
          <p className="mt-1 text-xs text-gray-500">Creates this many weekly lessons.</p>
        </div>

        <div className="md:col-span-2 xl:col-span-2">
          <label className="block text-sm font-medium text-gray-700">Message to client (optional)</label>
          <textarea
            className="mt-1 w-full rounded-md border px-3 py-2 text-sm"
            placeholder="Include any notes that should go in the confirmation email."
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            rows={3}
          />
        </div>

        <div className="md:col-span-2 xl:col-span-4 flex flex-wrap items-center gap-3">
          <button
            type="submit"
            className="rounded-md bg-black px-4 py-2 text-sm font-medium text-white disabled:opacity-50"
            disabled={!canSubmit}
          >
            {saving ? "Saving…" : "Create recurring series"}
          </button>
          {feedback && <span className="text-sm text-green-700">{feedback}</span>}
          {err && <span className="text-sm text-red-600">{err}</span>}
        </div>
      </form>

      {confirmState && (
        <div className="space-y-3 rounded-lg border border-amber-300 bg-amber-50 p-4">
          <div className="space-y-1">
            <p className="text-sm text-amber-900">{confirmState.human}</p>
            {confirmState.conflicts.length > 0 && (
              <ul className="space-y-2 text-xs text-amber-900">
                {confirmState.conflicts.map((row, idx) => (
                  <li key={`${row.start_local}-${idx}`}>
                    <div className="font-medium">{formatLocalLabel(row.start_local)}</div>
                    <ul className="ml-5 list-disc space-y-1">
                      {row.conflicts.map((c, jdx) => (
                        <li key={`${idx}-${jdx}`}>{c}</li>
                      ))}
                    </ul>
                  </li>
                ))}
              </ul>
            )}
          </div>
          <div className="flex flex-wrap gap-2">
            <button
              type="button"
              className="rounded-md bg-amber-600 px-4 py-2 text-sm font-medium text-white hover:bg-amber-700 disabled:opacity-50"
              onClick={confirmAnyway}
              disabled={saving}
            >
              {saving ? "Submitting…" : "Confirm and book"}
            </button>
            <button
              type="button"
              className="rounded-md border px-3 py-2 text-sm"
              onClick={() => setConfirmState(null)}
              disabled={saving}
            >
              Cancel
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
