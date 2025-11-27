// src/app/client/appointments/page.tsx
"use client";

import { useEffect, useMemo, useState } from "react";
import { useSession } from "next-auth/react";
import { getClientPaymentsFiltered } from "@/lib/api";

type Status = "booked" | "completed" | "canceled";
type Pay = "unpaid" | "paid" | "refunded" | "waived" | "partial" | "unknown" | "bundle";

const ALL_STATUS: Status[] = ["booked", "completed", "canceled"];
// Hide 'partial' from the filter UI; backend may still return it
const ALL_PAY: Pay[] = ["unpaid", "paid", "refunded", "waived"]; // no 'partial'

type Row = {
  id: string;
  start_utc: string;
  status: Status;
  payment_status: Pay;
  amount_paid_cents: number;     // cash paid
  bundle_applied_cents?: number; // wallet/bundle applied
  price_cents?: number | null;   // may be null → unknown
  duration_minutes: number;
  lesson_person_name?: string | null;
  is_group?: boolean;
};

export default function Page() {
  const { data: session } = useSession();
  const tz = (session?.user as { timezone?: string } | null | undefined)?.timezone;
  const [status, setStatus] = useState<Status[]>([]);
  const [pay, setPay] = useState<Pay[]>([]);
  const [rows, setRows] = useState<Row[]>([]);
  const [summary, setSummary] = useState<{ total_appointments: number; total_paid_cents: number; total_owed_cents: number } | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string>("");

  // Default to current month (client-local)
  const ymdLocal = (d: Date) => {
    const pad = (n: number) => String(n).padStart(2, "0");
    return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
  };
  const now = new Date();
  const firstOfMonth = new Date(now.getFullYear(), now.getMonth(), 1);
  const lastOfMonth = new Date(now.getFullYear(), now.getMonth() + 1, 0);
  const [dateFrom, setDateFrom] = useState<string>(() => ymdLocal(firstOfMonth));
  const [dateTo,   setDateTo]   = useState<string>(() => ymdLocal(lastOfMonth));

  const money = (c?: number | null) =>
    typeof c === "number" ? `$${(c / 100).toFixed(2)}` : "—";

  const toggle = <T,>(arr: T[], v: T) =>
    arr.includes(v) ? arr.filter((x) => x !== v) : [...arr, v];

  async function fetchData() {
    setLoading(true);
    setErr("");
    try {
      const res = await getClientPaymentsFiltered({
        date_from: dateFrom,
        date_to: dateTo,
        status: status.length ? (status as any) : undefined,
      });
      // Optional client-side payment_status filter
      const rows = (res.rows || []) as Array<{
        id: string;
        start_utc: string;
        duration_minutes: number;
        status: Status;
        price_cents?: number | null;
        amount_paid_cents: number;
        bundle_applied_cents?: number;
        payment_status: Pay;
      }>;
      const filteredByPay = pay.length ? rows.filter(r => pay.includes(r.payment_status)) : rows;
      setRows(filteredByPay as Row[]);
      setSummary(res.summary ?? null);
    } catch (e: unknown) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    fetchData();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [JSON.stringify(status), JSON.stringify(pay), dateFrom, dateTo]);

  const filtered = useMemo(() => {
    if (!rows.length) return rows;
    const from = dateFrom ? new Date(`${dateFrom}T00:00:00`) : null;
    const to = dateTo ? new Date(`${dateTo}T23:59:59.999`) : null;
    return rows.filter((r) => {
      const d = new Date(r.start_utc);
      if (from && d < from) return false;
      if (to && d > to) return false;
      return true;
    });
  }, [rows, dateFrom, dateTo]);

  const totals = useMemo(() => {
    // If no payment-status filter is applied and we have a server summary, trust it
    if (pay.length === 0 && summary) {
      return {
        paid: Math.max(0, summary.total_paid_cents || 0),
        owed: Math.max(0, summary.total_owed_cents || 0),
        count: Math.max(0, summary.total_appointments || 0),
      };
    }
    // Otherwise compute from the currently filtered rows
    let paid = 0;
    let owed = 0;
    for (const r of filtered) {
      const expKnown = r.price_cents != null;
      const exp = r.price_cents ?? 0;
      const got = r.amount_paid_cents ?? 0;
      paid += Math.max(0, got);
      if (expKnown) owed += Math.max(0, exp - got);
    }
    return { paid, owed, count: filtered.length };
  }, [filtered, pay.length, summary]);

  // Chip is NEUTRAL only when owed === 0 AND expected price is known
  const Chip = ({ kind, neutral }: { kind: Pay | null; neutral: boolean }) => {
    const k = kind ?? "unknown";
    const cls = neutral
      ? "bg-gray-100 text-gray-700"
      : k === "paid" || k === "bundle"
      ? "bg-green-100 text-green-700"
      : k === "refunded" || k === "waived"
      ? "bg-blue-100 text-blue-700"
      : k === "unpaid"
      ? "bg-red-100 text-red-700"
      : k === "partial"
      ? "bg-yellow-100 text-yellow-800"
      : "bg-gray-100 text-gray-700";
    return <span className={`px-2 py-0.5 rounded text-xs font-medium ${cls}`}>{k}</span>;
  };

  return (
    <div className="space-y-6">
      <h1 className="text-xl font-semibold">My appointments</h1>
      {err && <p className="text-sm text-red-600">{err}</p>}

      {/* Filters */}
      <div className="flex flex-col gap-3">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium">Status:</span>
          {ALL_STATUS.map((s) => (
            <button
              key={s}
              onClick={() => setStatus((cur) => toggle(cur, s))}
              className={`px-2 py-1 rounded border text-sm ${
                status.includes(s) ? "bg-zinc-200 dark:bg-zinc-800" : ""
              }`}
              aria-pressed={status.includes(s)}
            >
              {s}
            </button>
          ))}
        </div>

        <div className="flex items-center gap-2">
          <span className="text-sm font-medium">Payment:</span>
          {ALL_PAY.map((p) => (
            <button
              key={p}
              onClick={() => setPay((cur) => toggle(cur, p))}
              className={`px-2 py-1 rounded border text-sm ${
                pay.includes(p) ? "bg-zinc-200 dark:bg-zinc-800" : ""
              }`}
              aria-pressed={pay.includes(p)}
            >
              {p}
            </button>
          ))}
        </div>

        {/* Date range */}
        <div className="flex flex-wrap items-center gap-3">
          <label className="text-sm">
            From
            <input
              type="date"
              className="ml-2 border rounded px-2 py-1"
              value={dateFrom}
              onChange={(e) => setDateFrom(e.target.value)}
            />
          </label>
          <label className="text-sm">
            To
            <input
              type="date"
              className="ml-2 border rounded px-2 py-1"
              value={dateTo}
              onChange={(e) => setDateTo(e.target.value)}
            />
          </label>

          <button
            className="px-3 py-1 rounded border text-sm"
            onClick={() => {
              setStatus([]);
              setPay([]);
            }}
            title="Clear status/payment filters"
          >
            Clear
          </button>
        </div>
      </div>

      {/* Summary cards (filtered) */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        <div className="rounded-xl border p-3">
          <div className="text-xs text-gray-600">Appointments</div>
          <div className="text-xl font-semibold mt-1">{totals.count}</div>
        </div>
        <div className="rounded-xl border p-3">
          <div className="text-xs text-gray-600">Total Paid</div>
          <div className="text-xl font-semibold mt-1">{money(totals.paid)}</div>
        </div>
        <div className="rounded-xl border p-3">
          <div className="text-xs text-gray-600">Total Due</div>
          <div className="text-xl font-semibold mt-1">{money(totals.owed)}</div>
        </div>
      </div>

      {/* Rows */}
      <div className="overflow-x-auto rounded border">
        <table className="min-w-full text-sm">
          <thead className="bg-gray-50">
            <tr className="text-left">
              <th className="px-3 py-2 font-medium">When</th>
              <th className="px-3 py-2 font-medium">Status</th>
              <th className="px-3 py-2 font-medium">Payment</th>
              <th className="px-3 py-2 font-medium">Amount</th>
            </tr>
          </thead>
          <tbody>
            {loading && (
              <tr>
                <td colSpan={4} className="px-3 py-4 text-center text-gray-500">
                  Loading…
                </td>
              </tr>
            )}

            {!loading && !filtered.length && (
              <tr>
                <td colSpan={4} className="px-3 py-4 text-gray-500">
                  No appointments match your filters.
                </td>
              </tr>
            )}

            {!loading &&
              filtered.map((r) => {
                const start = new Date(r.start_utc);

                const expKnown = r.price_cents != null;
                const exp = r.price_cents ?? 0;
                const paidCash = r.amount_paid_cents ?? 0;
                const paidBundle = r.bundle_applied_cents ?? 0;
                const paidTotal = paidCash + paidBundle;
                const owed = expKnown ? Math.max(0, exp - Math.max(0, paidTotal)) : null;

                // Neutralize color only when we KNOW price and owed === 0
                const neutralChip = expKnown && (owed === 0);
                const isAdminFee = typeof r.id === 'string' && r.id.startsWith('fee:');

                return (
                  <tr key={r.id} className="border-t">
                    <td className="px-3 py-2">
                      <div className="font-medium">
                        {start.toLocaleString([], { dateStyle: "medium", timeStyle: "short", ...(tz ? { timeZone: tz } : {}) })}
                        {!isAdminFee && typeof r.duration_minutes === "number" && (
                          <span className="text-xs text-zinc-500 ml-2">· {r.duration_minutes} min</span>
                        )}
                        {isAdminFee && (
                          <span className="text-xs text-zinc-500 ml-2">· Administration fee</span>
                        )}
                      </div>
                      <div className="text-xs text-zinc-600 flex items-center gap-2">
                        {r.lesson_person_name && <span>For {r.lesson_person_name}</span>}
                        {r.is_group && (
                          <span className="inline-flex items-center px-2 py-[1px] rounded-full text-[10px] font-medium border border-zinc-300 bg-zinc-50">Group</span>
                        )}
                      </div>
                    </td>
                    <td className="px-3 py-2">{r.status ?? "—"}</td>
                    <td className="px-3 py-2">
                      <Chip kind={r.payment_status ?? "unknown"} neutral={neutralChip} />
                    </td>
                    <td className="px-3 py-2">
                      {expKnown ? (
                        r.payment_status === "refunded" ? (
                          <span className="text-blue-700 font-medium">Refunded</span>
                        ) : r.payment_status === "waived" ? (
                          <span className="text-blue-700">Waived</span>
                        ) : r.payment_status === "unpaid" || r.payment_status === "partial" ? (
                          owed! > 0 ? (
                            <span className="font-medium text-red-700">Owed {money(owed!)}</span>
                          ) : (
                            <span className="text-gray-700">—</span>
                          )
                        ) : paidTotal > 0 ? (
                          <span className="text-gray-800">Paid {money(Math.min(paidTotal, exp))}</span>
                        ) : exp > 0 ? (
                          <span className="text-gray-800">{r.payment_status === 'bundle' ? 'Covered by wallet' : 'Price'} {money(exp)}</span>
                        ) : (
                          <span className="text-zinc-500">—</span>
                        )
                      ) : (
                        paidTotal > 0 ? (
                          <span className="text-gray-800">Paid {money(paidTotal)}</span>
                        ) : (
                          <span className="text-zinc-500">—</span>
                        )
                      )}
                    </td>
                  </tr>
                );
              })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
