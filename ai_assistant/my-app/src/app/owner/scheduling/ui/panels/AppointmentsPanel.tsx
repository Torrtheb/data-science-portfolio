"use client";

import React, { useEffect, useState } from "react";
import {
  listOwnerClients,
  listAppointments,
  cancelAppointment,
  type Appointment,
} from "@/lib/api";
import { ClientRow } from "../types";
import { Th as _Th, Td as _Td } from "../utils/table";
import { useOnSchedChanged } from "../hooks/schedBus";
import EditPostDetailsModal from "../modals/EditPostDetailsModal";
import OwnerCreateAppt from "./OwnerCreateAppt";
import OwnerCreateRecurringAppt from "./OwnerCreateRecurringAppt";
import ClientPicker from "@/components/ClientPicker"; // ← ADD

type StatusFilter = "all" | "booked" | "completed" | "canceled";

export default function AppointmentsPanel({ tz }: { tz?: string }) {
  // -- New: unified client picking + text query like analytics --
  const [clientPick, setClientPick] = useState<{ id: string; name?: string | null; email: string } | null>(null);
  const [clientFreeText, setClientFreeText] = useState(""); // mirrors what's typed in the picker input

  const [items, setItems] = useState<Appointment[]>([]);
  const [totalCount, setTotalCount] = useState<number>(0);
  const [nextOffset, setNextOffset] = useState<number>(0);
  const [pageSize] = useState<number>(50);
  const [loadingMore, setLoadingMore] = useState<boolean>(false);
  const [editingAppt, setEditingAppt] = useState<Appointment | null>(null);

  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState("");

  // status filter
  const [statusFilter, setStatusFilter] = useState<StatusFilter>("all");

  // NEW: lazy toggle + "loaded" flag for the list
  // Always show list (remove toggle)
  const [showList] = useState(true);
  const [query, setQuery] = useState("");
  // Recurring creator now lives in a collapsible <details>, so no separate toggle state

  // --- Date range filter (owner-local dates) ---
  const [startDate, setStartDate] = useState<string>("");
  const [endDate, setEndDate] = useState<string>("");
  const [dateErr, setDateErr] = useState<string>("");

  function onStartChange(v: string) {
    setStartDate(v);
    setDateErr("");
    if (endDate && v && endDate < v) setDateErr("End date must be on or after start date.");
  }
  function onEndChange(v: string) {
    setEndDate(v);
    setDateErr("");
    if (!startDate && v) setDateErr("Select a start date before choosing an end date.");
    else if (startDate && v && v < startDate) setDateErr("End date must be on or after start date.");
  }

  // client filter
  //const [clients, setClients] = useState<ClientRow[]>([]);
  //const [selectedClientId, setSelectedClientId] = useState<string>("all");

  const now = new Date();

  //const filteredByClient = items.filter((a) => {
   // if (selectedClientId === "all") return true;
  //  return a.client?.id === selectedClientId; // safe; both are strings
  //});

  const keyOf = (c: ClientRow, i: number) =>
    // unique in the list (append #i), even if id/email missing/duplicated
    `${c.id ? `id:${c.id}` : c.email ? `email:${c.email}` : `anon:${i}`}#${i}`;

  const valueOf = (c: ClientRow, i: number) =>
    // stable selection token (no #i for id/email; include index only for true anon)
    c.id ? `id:${c.id}` : c.email ? `email:${c.email}` : `anon:${i}`;

  // ---------- search helpers ----------
  const norm = (s: unknown): string =>
    String(s ?? "")
      .toLowerCase()
      .normalize("NFKD")
      .replace(/[\u0300-\u036f]/g, "");

  function collectSearchHaystack(a: Appointment): string {
    const parts: string[] = [];
    const add = (v: unknown) => {
      const s = String(v ?? "");
      if (s) parts.push(s);
    };

    // Primary client
    add(a.client?.name);
    add(a.client?.email);

    // Common optional shapes (safe if absent)
    const c: any = a.client || {};
    add(c.parent_name);
    add(c.account_name);
    add(c.owner_name);

    // Booker (who made the booking)
    const bookedBy = (a as any).booked_by;
    add(bookedBy?.name);
    add(bookedBy?.email);

    // People in the account / dependents
    const people: any[] = Array.isArray(c.people)
      ? c.people
      : Array.isArray(c.household)
      ? c.household
      : Array.isArray(c.members)
      ? c.members
      : [];
    for (const p of people) {
      add(p?.name);
      add(p?.email);
    }

    return norm(parts.join(" "));
  }

  function matchesQuery(a: Appointment, q: string): boolean {
    const t = norm(q.trim());
    if (!t) return true;
    const tokens = t.split(/\s+/).filter(Boolean);
    if (!tokens.length) return true;
    const hay = collectSearchHaystack(a);
    return tokens.every((tok) => hay.includes(tok));
  }

  // --- Normalize helper ---
  const normalize = (s: unknown): string =>
    String(s ?? "")
      .toLowerCase()
      .normalize("NFKD")
      .replace(/[\u0300-\u036f]/g, "");

  // --- Client predicate (analytics-style) ---
  function matchClient(a: Appointment): boolean {
    // 1) exact pick wins
    if (clientPick?.id) return a.client?.id === clientPick.id;

    // 2) otherwise free-text on name OR email
    const q = normalize(clientFreeText.trim());
    if (!q) return true;
    const name = normalize(a.client?.name);
    const email = normalize(a.client?.email);
    return name.includes(q) || email.includes(q);
  }

  // --- Status/date predicate (kept from your logic) ---
  function matchStatusAndDates(a: Appointment): boolean {
    const st = new Date(a.start_utc);
    const status = a.status || "booked";

    // status filter
    if (statusFilter !== "all") {
      if (statusFilter === "canceled" && status !== "canceled") return false;
      if (statusFilter === "completed" && status !== "completed") return false;
      if (statusFilter === "booked" && !(status === "booked" && st > now)) return false;
    }

    // date range (owner-local dates provided as YYYY-MM-DD)
    if (startDate) {
      const startBoundary = new Date(`${startDate}T00:00:00`);
      if (st < startBoundary) return false;
    }
    if (endDate) {
      const endBoundary = new Date(`${endDate}T23:59:59.999`);
      if (st > endBoundary) return false;
    }

    return true;
  }

  const visibleItems = items.filter(matchClient).filter(matchStatusAndDates);


  //useEffect(() => {
    //(async () => {
     // try {
       // const rows = await listOwnerClients();
        //setClients(rows);
      //} catch {
        // ignore; filtering is optional
      //}
    //})();
 // }, []);

  const reload = async () => {
    try {
      if (dateErr) return; // don't fetch when invalid
      setLoading(true);
      setErr("");

      // Build query for backend date filtering (owner-local YYYY-MM-DD)
      const params = new URLSearchParams();
      if (startDate) params.set("start", startDate);
      if (endDate) params.set("end", endDate);
      if (tz) params.set("tz", tz);
      params.set("limit", String(pageSize));
      params.set("offset", "0");

      const res = await fetch(`/api/back/api/scheduling/appointments?${params.toString()}`, {
        credentials: "include",
        cache: "no-store",
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `Fetch failed (${res.status})`);
      }
      const data: Appointment[] = await res.json();
      setItems(data);
      const total = parseInt(res.headers.get("X-Total-Count") || "0", 10) || 0;
      const next = parseInt(res.headers.get("X-Next-Offset") || "0", 10) || 0;
      setTotalCount(total);
      setNextOffset(next);
    } catch (e: any) {
      setErr(e.message || String(e));
    } finally {
      setLoading(false);
    }
  };

  // Load list on mount
  useEffect(() => {
    reload();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Re-fetch on date filter changes
  useEffect(() => {
    // Reset pagination when filters change
    setNextOffset(0);
    setItems([]);
    setTotalCount(0);
    reload();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [startDate, endDate]);

  // Refresh when schedule changes elsewhere
  useOnSchedChanged(() => {
    reload();
  });

  return (
    <section className="space-y-4">
      {/* Header removed to avoid duplicate page heading */}

      {/* Collapsible creators */}
      <details className="rounded-xl border border-gray-200 bg-white p-4 shadow-sm" open>
        <summary className="cursor-pointer text-base font-medium">Create appointment</summary>
        <div className="mt-3">
          <OwnerCreateAppt onCreated={reload} tz={tz} />
        </div>
      </details>

      <details className="rounded-xl border border-gray-200 bg-white p-4 shadow-sm">
        <summary className="cursor-pointer text-base font-medium">Set recurring appointments</summary>
        <div className="mt-3">
          <OwnerCreateRecurringAppt
            onCreated={async () => {
              await reload();
            }}
          />
        </div>
      </details>

      {/* Appointments list with pagination */}
      <div className="rounded-xl border bg-white shadow-sm">
        <div className="p-3 flex items-center justify-between">
          <div className="text-sm text-gray-700">
            Showing {items.length} of {totalCount || items.length}
          </div>
          <div className="flex items-center gap-2">
            <input
              className="border rounded px-2 py-1 text-sm"
              placeholder="Filter by client name/email…"
              value={clientFreeText}
              onChange={(e) => setClientFreeText(e.target.value)}
            />
          </div>
        </div>

        {err && (
          <div className="px-3 py-2 text-sm text-red-600">{String(err)}</div>
        )}

        <div className="overflow-x-auto">
          <table className="min-w-full text-sm">
            <thead>
              <tr className="bg-gray-50 text-left">
                <th className="px-3 py-2">Date</th>
                <th className="px-3 py-2">Time</th>
                <th className="px-3 py-2">Client</th>
                <th className="px-3 py-2">Status</th>
              </tr>
            </thead>
            <tbody>
              {visibleItems.map((a) => {
                const st = new Date(a.start_utc);
                const en = new Date(a.end_utc);
                const d = st.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
                const range = `${st.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })} – ${en.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })}`;
                const client = a.client?.name || a.client?.email || "—";
                return (
                  <tr key={a.id} className="border-t">
                    <td className="px-3 py-2 whitespace-nowrap">{d}</td>
                    <td className="px-3 py-2 whitespace-nowrap">{range}</td>
                    <td className="px-3 py-2">{client}</td>
                    <td className="px-3 py-2">{a.status}</td>
                  </tr>
                );
              })}
              {visibleItems.length === 0 && !loading && (
                <tr>
                  <td className="px-3 py-6 text-gray-500" colSpan={4}>No appointments match your filters.</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        <div className="p-3 flex items-center justify-between">
          <div className="text-xs text-gray-500">{loading ? "Loading…" : ""}</div>
          <button
            className="text-sm border rounded px-3 py-1 disabled:opacity-50"
            disabled={loadingMore || !nextOffset || nextOffset >= totalCount}
            onClick={async () => {
              try {
                if (!nextOffset || nextOffset >= totalCount) return;
                setLoadingMore(true);
                const params = new URLSearchParams();
                if (startDate) params.set("start", startDate);
                if (endDate) params.set("end", endDate);
                if (tz) params.set("tz", tz);
                params.set("limit", String(pageSize));
                params.set("offset", String(nextOffset));
                const res = await fetch(`/api/back/api/scheduling/appointments?${params.toString()}`, {
                  credentials: "include",
                  cache: "no-store",
                });
                if (!res.ok) throw new Error(await res.text());
                const more: Appointment[] = await res.json();
                setItems((prev) => [...prev, ...more]);
                const hdr = res.headers.get("X-Next-Offset");
                const n = hdr ? parseInt(hdr, 10) : (nextOffset + more.length);
                setNextOffset(n);
              } catch (e: any) {
                setErr(e.message || String(e));
              } finally {
                setLoadingMore(false);
              }
            }}
          >
            {loadingMore ? "Loading…" : nextOffset < totalCount ? "Load more" : "All loaded"}
          </button>
        </div>
      </div>
    </section>
  );
}
