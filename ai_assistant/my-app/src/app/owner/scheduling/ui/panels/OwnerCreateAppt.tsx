"use client";

import React, { useEffect, useMemo, useState } from "react";
import ClientPicker from "@/components/ClientPicker";

import {
  listOwnerClients,
  listAppointments,
  listOpenings,
  listTimeOff,
  cancelAppointment,
  deleteTimeOff,
  deleteOpening,
  adminCreateAppointment,
} from "@/lib/api";
import { ClientRow } from "../types";
import { to24h, pad2, fromLocalInTZToUTC } from "../utils/datetime";
import { emitSchedChanged } from "../hooks/schedBus";

export default function OwnerCreateAppt({ onCreated, tz }: { onCreated: () => void; tz?: string }) {
  // Selected client from typeahead + the email used for booking
  const [pickedClient, setPickedClient] = useState<{ id: string; name?: string | null; email: string } | null>(null);
  const [clientEmail, setClientEmail] = useState<string>(""); // <- bind your “Client email” input to this

  const [clientName, setClientName] = useState("");
  const [date, setDate] = useState(""); // yyyy-mm-dd

  // AM/PM time inputs
  const [h12, setH12] = useState(2); // 1..12
  const [m12, setM12] = useState(30);
  const [ampm, setAmPm] = useState<"AM" | "PM">("PM");

  const [duration, setDuration] = useState(30);
  const [saving, setSaving] = useState(false);
  const [msg, setMsg] = useState("");
  const [message, setMessage] = useState("");
  const [lessonFor, setLessonFor] = useState("");

  // --- client search/picker ---
  const [clients, setClients] = useState<ClientRow[]>([]);
  const [clientQuery, setClientQuery] = useState(""); // user types name/email
  const [loadingClients, setLoadingClients] = useState(false);

  useEffect(() => {
    (async () => {
      try {
        setLoadingClients(true);
        const rows = await listOwnerClients();
        setClients(rows);
      } catch {
        // silently ignore for now (manual email still works)
      } finally {
        setLoadingClients(false);
      }
    })();
  }, []);

  const normalized = (s: string) =>
    (s || "").toLowerCase().normalize("NFKD").replace(/[\u0300-\u036f]/g, "");

  const filteredClients = useMemo(() => {
    const q = normalized(clientQuery.trim());
    if (!q) return clients.slice(0, 30);
    return clients
      .filter((c) => normalized(`${c.name ?? ""} ${c.email ?? ""}`).includes(q))
      .slice(0, 30);
  }, [clients, clientQuery]);

  const pickClient = (c: ClientRow) => {
    const email = c.email ?? "";
    setClientEmail(email); // always a string now
    const derivedName = c.name ?? (email ? email.split("@")[0] : "Client");
    setClientName(derivedName);
    setClientQuery(`${c.name ? c.name + " — " : ""}${email}`);
  };

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setMsg("");
    if (!clientEmail || !date) {
      setMsg("Email and date are required.");
      return;
    }

    try {
      setSaving(true);

      // Build HH:mm from selected format
      const h = to24h(h12, ampm);
      const hhmm = `${pad2(h)}:${pad2(m12)}`;

      if (!/^\d{2}:\d{2}$/.test(hhmm)) {
        setMsg("Please provide a valid time.");
        return;
      }

      const start_local = `${date}T${hhmm}:00`; // owner-local wall time string for API
      const startUtc = tz
        ? new Date(fromLocalInTZToUTC(`${date}T${hhmm}`, tz))
        : new Date(start_local);
      const endUtc = new Date(startUtc.getTime() + duration * 60000);

      // Pull current items to check overlaps
      const [appts, openings, offs] = await Promise.all([
        listAppointments(tz),
        listOpenings(tz),
        listTimeOff(tz),
      ]);

      const overlaps = (aS: Date, aE: Date, bS: Date, bE: Date) => aE > bS && aS < bE;

      const hitAppts = (appts as any[]).filter(
        (a: any) =>
          a.status !== "canceled" &&
          overlaps(startUtc, endUtc, new Date(a.start_utc), new Date(a.end_utc))
      );
      const hitOffs = (offs as any[]).filter((t: any) =>
        overlaps(startUtc, endUtc, new Date(t.start_utc), new Date(t.end_utc))
      );
      const hitOpens = (openings as any[]).filter((o: any) =>
        overlaps(startUtc, endUtc, new Date(o.start_utc), new Date(o.end_utc))
      );

      if (hitAppts.length || hitOffs.length || hitOpens.length) {
        const parts: string[] = [];
        if (hitAppts.length)
          parts.push(`${hitAppts.length} appointment${hitAppts.length > 1 ? "s" : ""}`);
        if (hitOffs.length)
          parts.push(`${hitOffs.length} time-off block${hitOffs.length > 1 ? "s" : ""}`);
        if (hitOpens.length)
          parts.push(`${hitOpens.length} opening${hitOpens.length > 1 ? "s" : ""}`);

        const ok = confirm(
          `This time overlaps ${parts.join(", ")}.\n\nIf you continue I will:\n` +
            (hitAppts.length ? "• cancel the overlapping appointment(s)\n" : "") +
            (hitOffs.length ? "• delete the overlapping time-off block(s)\n" : "") +
            // Do NOT delete openings; appointments occupy time within openings
            `\nProceed?`
        );
        if (!ok) {
          setSaving(false);
          return;
        }

        for (const a of hitAppts) await cancelAppointment(a.id);
        for (const t of hitOffs) await deleteTimeOff(t.id);
        // Do not delete overlapping openings; leave them so availability remains and appointments carve them
      }

      const fallbackName = clientEmail.split("@")[0] || "Client";
      await adminCreateAppointment({
        client_name: clientName.trim() || fallbackName,
        client_email: clientEmail.trim(),
        start_local,
        duration_minutes: duration,
        message: message.trim() ? message : undefined,
        lesson_person_name: lessonFor.trim() || undefined,
      });

      emitSchedChanged();
      setMsg("Created!");
      setClientName("");
      setClientEmail("");
      setDate("");
      setH12(2);
      setM12(30);
      setAmPm("PM");
      setDuration(30);
      setLessonFor("");
      await onCreated();
    } catch (err: any) {
      setMsg("Error: " + (err.message || String(err)));
    } finally {
      setSaving(false);
    }
  };

  return (
    <form onSubmit={submit} className="bg-gray-50 rounded-xl p-3 flex flex-wrap items-end gap-3">
      {/* Client picker (search by name/email) */}
      <div className="space-y-1">
        <label className="block text-sm text-gray-700">Client (search by name or email)</label>
        <ClientPicker
          value={pickedClient}
          onChange={(hit) => {
            setPickedClient(hit);
            setClientEmail(hit?.email ?? "");
          }}
          placeholder="Start typing a client name or email…"
          minChars={1}
          showEmailOnlyInInput={false} // show name+email while searching; we copy only email to the field below
        />
      </div>

      <div>
        <label className="block text-sm text-gray-700">Client email *</label>
        <input
          className="w-full border rounded-md px-3 py-2"
          value={clientEmail}
          onChange={(e) => setClientEmail(e.target.value)}
          placeholder="client@example.com"
          required
        />
      </div>


      <div>
        <label className="block text-xs text-gray-600 mb-1">Date</label>
        <input
          type="date"
          className="border rounded px-2 py-1"
          value={date}
          onChange={(e) => setDate(e.target.value)}
        />
      </div>

      {/* Time input(s) */}
      <div className="flex items-end gap-2">
        <div>
          <label className="block text-xs text-gray-600 mb-1">Hour</label>
          <select
            className="border rounded px-2 py-1"
            value={h12}
            onChange={(e) => setH12(Number(e.target.value))}
          >
            {Array.from({ length: 12 }, (_, i) => i + 1).map((h) => (
              <option key={h} value={h}>
                {h}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label className="block text-xs text-gray-600 mb-1">Minute</label>
          <select
            className="border rounded px-2 py-1"
            value={m12}
            onChange={(e) => setM12(Number(e.target.value))}
          >
            {[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55].map((m) => (
              <option key={m} value={m}>
                {pad2(m)}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label className="block text-xs text-gray-600 mb-1">AM/PM</label>
          <select
            className="border rounded px-2 py-1"
            value={ampm}
            onChange={(e) => setAmPm(e.target.value as "AM" | "PM")}
          >
            <option>AM</option>
            <option>PM</option>
          </select>
        </div>
      </div>

      <div>
        <label className="block text-xs text-gray-600 mb-1">Duration (min)</label>
        <select
          className="border rounded px-2 py-1"
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

      <div className="basis-full">
        <label className="block text-xs text-gray-600 mb-1">
          Optional message to include in the email
        </label>
        <textarea
          className="border rounded px-2 py-1 w-full"
          rows={2}
          placeholder="Anything you’d like to tell the client…"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
        />
      </div>

      <div className="basis-full">
        <label className="block text-xs text-gray-600 mb-1">Lesson for (Person name)</label>
        <input
          className="border rounded px-2 py-1 w-full"
          placeholder="e.g., Fluffy Junior"
          value={lessonFor}
          onChange={(e) => setLessonFor(e.target.value)}
        />
      </div>

      <button
        disabled={saving}
        className="px-3 py-2 bg-black text-white rounded-lg disabled:opacity-50"
      >
        {saving ? "Creating…" : "Create Appointment"}
      </button>
      {msg && (
        <span
          className={`text-sm ${
            msg.startsWith("Error") ? "text-red-600" : "text-green-700"
          }`}
        >
          {msg}
        </span>
      )}
    </form>
  );
}
