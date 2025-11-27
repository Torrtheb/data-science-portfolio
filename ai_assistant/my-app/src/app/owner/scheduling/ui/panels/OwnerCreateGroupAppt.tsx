"use client";

import React, { useEffect, useMemo, useState } from "react";
import ClientPicker from "@/components/ClientPicker";
import { pad2, to24h } from "../utils/datetime";
import {
  adminCreateGroupAppointment,
  adminCreateGroupRecurringAppointments,
  adminGroupAddAttendees,
  ownerResolveClientAccount,
  ownerGetClientDetail,
  type OwnerClientLite,
} from "@/lib/api";
import { emitSchedChanged } from "../hooks/schedBus";

type ResolvedPerson = { account_id: number; person_id: number; label: string };

export default function OwnerCreateGroupAppt() {
  const [clients, setClients] = useState<OwnerClientLite[]>([]);
  const [resolved, setResolved] = useState<Record<string, ResolvedPerson | null>>({}); // key by client_user_id
  const [peopleByClient, setPeopleByClient] = useState<Record<string, Array<{ id: number; full_name: string; email?: string | null }>>>({});
  const [selectedPeople, setSelectedPeople] = useState<Record<string, number[]>>({});

  const [startDate, setStartDate] = useState("");
  const [hour12, setHour12] = useState(4);
  const [minute, setMinute] = useState(0);
  const [ampm, setAmPm] = useState<"AM" | "PM">("PM");
  const [duration, setDuration] = useState(60);
  const [repeat, setRepeat] = useState(false);
  const [repeatWeeks, setRepeatWeeks] = useState(1);
  const [occurrences, setOccurrences] = useState(4);

  const [saving, setSaving] = useState(false);
  const [ok, setOk] = useState<string>("");
  const [err, setErr] = useState<string>("");

  // Stable key for maps (prefer account_id when present)
  const clientKey = (c: OwnerClientLite) => String(((c as any).account_id as number | undefined) ?? c.id);

  // Toggle a person selection for a given client (keyed by account_id when available)
  function togglePerson(clientKey: string, personId: number) {
    setSelectedPeople((prev) => {
      const prevList = prev[clientKey] || [];
      const nextList = prevList.includes(personId)
        ? prevList.filter((id) => id !== personId)
        : [...prevList, personId];
      return { ...prev, [clientKey]: nextList };
    });
  }

  // Resolve selected clients → load account people directly by account_id
  useEffect(() => {
    (async () => {
      const map: Record<string, ResolvedPerson | null> = {};
      const ppl: Record<string, Array<{ id: number; full_name: string; email?: string | null }>> = {};
      const sel: Record<string, number[]> = {};
      for (const c of clients) {
        try {
          const acctId = (c as any).account_id as number | undefined;
          const key = String(acctId ?? c.id);
          // Prefer direct fetch by account_id from search; fallback to resolve if missing
          const detail = acctId
            ? await ownerGetClientDetail(acctId)
            : await (async () => {
                const acct = await ownerResolveClientAccount(c.email || c.name || "");
                return ownerGetClientDetail(acct.account_id);
              })();
          const people = (detail.people || []) as Array<{ id: number; full_name: string; email?: string | null }>;
          ppl[key] = people;
          if (people.length > 0) {
            sel[key] = people.map((p) => p.id); // default: all people selected
            const p = people[0];
            map[key] = { account_id: detail.account_id, person_id: p.id, label: p.full_name || p.email || c.email || c.name || "Person" };
          } else {
            map[key] = null;
            sel[key] = [];
          }
        } catch {
          const acctId = (c as any).account_id as number | undefined;
          const key = String(acctId ?? c.id);
          map[key] = null;
        }
      }
      setResolved(map);
      setPeopleByClient(ppl);
      setSelectedPeople(sel);
    })();
  }, [clients]);

  const canSubmit = useMemo(() => {
    if (saving) return false;
    if (!startDate) return false;
    if (!(duration > 0)) return false;
    if (clients.length === 0) return false;
    // At least one selected person across all chosen clients (keyed by account_id when present)
    return clients.some((c) => (selectedPeople[clientKey(c)] || []).length > 0);
  }, [saving, startDate, duration, clients, selectedPeople]);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setErr("");
    setOk("");
    setSaving(true);
    try {
      if (!canSubmit) return;
      const hh = to24h(hour12, ampm);
      const startLocal = `${startDate}T${pad2(hh)}:${pad2(minute)}:00`;
      // Collect ALL selected people (across all picked clients)
      const personIds = Array.from(new Set(
        Object.values(selectedPeople).flat()
      )).map(Number).filter((n) => Number.isFinite(n) && n > 0);
      if (personIds.length === 0) {
        setErr("Select at least one person to add to the group.");
        return;
      }
      if (!repeat) {
        const res = await adminCreateGroupAppointment({
          start_local: startLocal,
          duration_minutes: duration,
          person_ids: personIds,
          confirm_if_conflicts: true,
        });
        if ((res?.count ?? 0) > 0 && res.group_id) {
          // Ensure all selected people are attached to THIS group
          try { await adminGroupAddAttendees(res.group_id, personIds); } catch {}
          setOk(`Created group with ${personIds.length} attendee${personIds.length === 1 ? '' : 's'}.`);
          emitSchedChanged();
          setClients([]);
        } else if (personIds.length > 0) {
          // Establish a group with the first person, then attach the rest
          const first = personIds[0];
          const one = await adminCreateGroupAppointment({
            start_local: startLocal,
            duration_minutes: duration,
            person_ids: [first],
            allow_override: true,
            confirm_if_conflicts: true,
          });
          if ((one?.count ?? 0) > 0 && one.group_id) {
            try { await adminGroupAddAttendees(one.group_id, personIds); } catch {}
            setOk(`Created group with ${personIds.length} attendee${personIds.length === 1 ? '' : 's'}.`);
            emitSchedChanged();
            setClients([]);
          } else {
            throw new Error("No attendees were created. Check for time conflicts or duplicate bookings.");
          }
        } else {
          throw new Error("Select at least one person to add to the group.");
        }
      } else {
        const res = await adminCreateGroupRecurringAppointments({
          start_local: startLocal,
          duration_minutes: duration,
          repeat_every_weeks: repeatWeeks,
          occurrences: occurrences,
          person_ids: personIds,
          confirm_if_conflicts: true,
        });
        if ((res?.count ?? 0) > 0) {
          setOk(`Created ${res.count} group lesson seat${res.count === 1 ? "" : "s"} across ${res.groups.length} occurrence(s).`);
        } else {
          throw new Error("No group lessons were created. Check for conflicts or duplicates.");
        }
      }
      emitSchedChanged();
      setClients([]);
    } catch (e: any) {
      setErr(e?.message || "Failed to create group lesson.");
    } finally {
      setSaving(false);
    }
  }

  return (
    <form className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4" onSubmit={submit}>
      <div className="md:col-span-2">
        <label className="block text-sm font-medium text-gray-700">Attendees (clients)</label>
        <ClientPicker
          multiple
          value={clients}
          onChange={setClients}
          placeholder="Search clients by name or email…"
          showEmailOnlyInInput
        />
        {clients.length > 0 && (
          <div className="mt-2 text-xs text-gray-600">
            {clients.map((c) => {
              const key = String(((c as any).account_id as number | undefined) ?? c.id);
              return (
              <div key={key} className="py-1">
                <div className="font-medium">{c.email || c.name}</div>
                <div className="ml-3 space-y-1">
                  {(peopleByClient[key] && peopleByClient[key].length > 0) ? (
                    peopleByClient[key].map((p) => (
                      <label key={p.id} className="flex items-center gap-2">
                        <input
                          type="checkbox"
                          checked={(selectedPeople[key] || []).includes(p.id)}
                          onChange={() => togglePerson(key, p.id)}
                        />
                        <span>{p.full_name || p.email || `Person ${p.id}`}</span>
                      </label>
                    ))
                  ) : (
                    <div className="text-red-600">No person on this account</div>
                  )}
                </div>
              </div>
            )})}
          </div>
        )}
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
          <select className="rounded-md border px-2 py-2 text-sm" value={hour12} onChange={(e) => setHour12(Number(e.target.value))}>
            {Array.from({ length: 12 }, (_, idx) => idx + 1).map((h) => (
              <option key={h} value={h}>{h}</option>
            ))}
          </select>
          <select className="rounded-md border px-2 py-2 text-sm" value={minute} onChange={(e) => setMinute(Number(e.target.value))}>
            {[0, 15, 30, 45].map((m) => (
              <option key={m} value={m}>{pad2(m)}</option>
            ))}
          </select>
          <select className="rounded-md border px-2 py-2 text-sm" value={ampm} onChange={(e) => setAmPm(e.target.value as any)}>
            <option value="AM">AM</option>
            <option value="PM">PM</option>
          </select>
        </div>
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700">Duration (minutes)</label>
        <select className="mt-1 w-full rounded-md border px-3 py-2 text-sm" value={duration} onChange={(e) => setDuration(Number(e.target.value))}>
          {[30, 45, 60, 75, 90, 120].map((d) => (
            <option key={d} value={d}>{d}</option>
          ))}
        </select>
      </div>

      <div className="md:col-span-2 flex items-center gap-3">
        <label className="text-sm text-gray-700">Repeat weekly</label>
        <input type="checkbox" checked={repeat} onChange={(e) => setRepeat(e.target.checked)} />
        {repeat && (
          <>
            <label className="text-sm text-gray-700 ml-2">Every</label>
            <input type="number" min={1} max={8} value={repeatWeeks} onChange={(e) => setRepeatWeeks(Math.max(1, Math.min(8, Number(e.target.value) || 1)))} className="w-20 border rounded px-2 py-1 text-sm" />
            <span className="text-sm">week(s)</span>
            <label className="text-sm text-gray-700 ml-3">Occurrences</label>
            <input type="number" min={1} max={24} value={occurrences} onChange={(e) => setOccurrences(Math.max(1, Math.min(24, Number(e.target.value) || 1)))} className="w-24 border rounded px-2 py-1 text-sm" />
          </>
        )}
      </div>

      <div className="md:col-span-2 flex items-center gap-3">
        <button type="submit" className="rounded-md bg-black px-4 py-2 text-sm font-medium text-white disabled:opacity-50" disabled={!canSubmit}>
          {saving ? "Saving…" : "Create group lesson"}
        </button>
        {ok && <span className="text-sm text-green-700">{ok}</span>}
        {err && <span className="text-sm text-red-600">{err}</span>}
      </div>
    </form>
  );
}
