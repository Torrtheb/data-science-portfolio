"use client";

import React, { useEffect, useMemo, useState } from "react";
import { getClientPaymentsFiltered } from "@/lib/api";

type PaymentKind =
  | "paid"
  | "partial"
  | "unpaid"
  | "bundle"
  | "unknown"
  | "refunded"
  | "waived";

type Row = {
  id: string;
  start_utc: string;
  duration_minutes: number;
  status: "booked" | "completed" | "canceled";
  attendance?: "unknown" | "on_time" | "late" | "no_show" | "attended";
  price_cents?: number | null;
  amount_paid_cents: number;
  payment_status: PaymentKind;
};

type Summary = {
  total_appointments: number;
  late_appointments: number;
  paid_appointments: number;
  unpaid_appointments: number;
  total_expected_cents: number;
  total_paid_cents: number;
  total_owed_cents: number;
};

export default function ClientDashboardAppointments() {
  const [rows, setRows] = useState<Row[]>([]);
  const [summary, setSummary] = useState<Summary | null>(null);
  const [err, setErr] = useState("");
  const [loading, setLoading] = useState(false);

  // Default to last 60 days
  const [dateFrom, setDateFrom] = useState<string>(() =>
    new Date(Date.now() - 60 * 24 * 3600 * 1000).toISOString().slice(0, 10)
  );
  const [dateTo, setDateTo] = useState<string>(() =>
    new Date().toISOString().slice(0, 10)
  );

  async function load() {
    setLoading(true);
    setErr("");
    try {
      const data = await getClientPaymentsFiltered({
        date_from: dateFrom,
        date_to: dateTo,
      });
      setRows((data.rows ?? []) as Row[]);
      setSummary((data.summary ?? null) as Summary | null);
    } catch (e: unknown) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  // initial fetch
  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const money = (c?: number | null) => `$${((c ?? 0) / 100).toFixed(2)}`;

  const Chip = ({ kind }: { kind: PaymentKind }) => {
    // Required colors: paid=green, refunded/waived/bundle=blue, unpaid=red
    const cls =
      kind === "paid"
        ? "bg-green-100 text-green-700"
        : kind === "refunded" || kind === "waived" || kind === "bundle"
        ? "bg-blue-100 text-blue-700"
        : kind === "unpaid"
        ? "bg-red-100 text-red-700"
        : kind === "partial"
        ? "bg-yellow-100 text-yellow-800"
        : "bg-gray-100 text-gray-700";

    const label =
      kind === "paid"
        ? "Paid"
        : kind === "unpaid"
        ? "Unpaid"
        : kind === "partial"
        ? "Partial"
        : kind === "bundle"
        ? "Bundle"
        : kind === "refunded"
        ? "Refunded"
        : kind === "waived"
        ? "Waived"
        : "Unknown";

    return (
      <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${cls}`}>
        {label}
      </span>
    );
  };

  // For the summary card if backend summary is missing or you want a client check
  const computedOwed = useMemo(() => {
    return rows.reduce((sum, r) => {
      const exp = r.price_cents ?? 0;
      const paid = r.amount_paid_cents ?? 0;
      return sum + Math.max(exp - paid, 0);
    }, 0);
  }, [rows]);

  return (
    <div className="space-y-4">
      <h2 className="text-lg font-medium">My Appointments</h2>
      {err && <p className="text-red-600 text-sm">{err}</p>}

      {/* Filters */}
      <div className="flex flex-wrap items-end gap-2">
        <label className="text-sm">
        From
        <input
            type="date"
            className="ml-2 border rounded px-2 py-1"
            value={dateFrom}
            max={dateTo || undefined}
            onChange={(e) => {
            const v = e.target.value;
            setDateFrom(v);
            if (dateTo && v > dateTo) setDateTo(v);
            }}
        />
        </label>
        <label className="text-sm">
        To
        <input
            type="date"
            className="ml-2 border rounded px-2 py-1"
            value={dateTo}
            min={dateFrom || undefined}
            onChange={(e) => {
            const v = e.target.value;
            setDateTo(v);
            if (dateFrom && v < dateFrom) setDateFrom(v);
            }}
        />
        </label>

        <button
          className="px-3 py-1 border rounded text-sm"
          onClick={load}
          disabled={loading}
        >
          {loading ? "Loading…" : "Apply"}
        </button>
      </div>

      {/* Account Summary */}
      {(summary || rows.length) && (
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          <div className="rounded-xl border p-3">
            <div className="text-xs text-gray-600">Appointments</div>
            <div className="text-xl font-semibold mt-1">
              {summary?.total_appointments ?? rows.length}
            </div>
          </div>
          <div className="rounded-xl border p-3">
            <div className="text-xs text-gray-600">Total Paid</div>
            <div className="text-xl font-semibold mt-1">
              {money(summary?.total_paid_cents ?? rows.reduce((s, r) => s + (r.amount_paid_cents ?? 0), 0))}
            </div>
          </div>
          <div className="rounded-xl border p-3">
            <div className="text-xs text-gray-600">Total Due</div>
            <div className="text-xl font-semibold mt-1">
              {money(summary?.total_owed_cents ?? computedOwed)}
            </div>
          </div>
        </div>
      )}

      {/* Table */}
      <div className="overflow-x-auto rounded-xl border">
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

            {!loading && rows.length === 0 && (
              <tr>
                <td colSpan={4} className="px-3 py-4 text-gray-500">
                  No appointments in this range.
                </td>
              </tr>
            )}

            {!loading &&
              rows.map((a) => {
                const start = new Date(a.start_utc);
                const owed = Math.max((a.price_cents ?? 0) - (a.amount_paid_cents ?? 0), 0);
                const showOwed = a.payment_status === "unpaid" || a.payment_status === "partial";

                return (
                  <tr key={a.id} className="border-t">
                    <td className="px-3 py-2">
                      {start.toLocaleString([], { dateStyle: "medium", timeStyle: "short" })}
                      <span className="text-xs text-gray-500 ml-2">· {a.duration_minutes} min</span>
                    </td>
                    <td className="px-3 py-2">
                      {a.status}
                      {a.attendance ? (
                        <span className="ml-2 text-xs text-gray-500">· {a.attendance}</span>
                      ) : null}
                    </td>
                    <td className="px-3 py-2">
                      <Chip kind={a.payment_status} />
                    </td>
                    <td className="px-3 py-2">
                      {showOwed ? (
                        <span className="font-medium text-red-700">Owed {money(owed)}</span>
                      ) : a.amount_paid_cents > 0 ? (
                        <span className="text-gray-800">Paid {money(a.amount_paid_cents)}</span>
                      ) : typeof a.price_cents === "number" ? (
                        <span className="text-gray-800">Price {money(a.price_cents)}</span>
                      ) : (
                        <span className="text-gray-500">—</span>
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
