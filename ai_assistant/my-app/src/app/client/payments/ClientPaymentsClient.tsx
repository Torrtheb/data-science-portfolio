// src/app/client/payments/ClientPaymentsClient.tsx
"use client";

import { useEffect, useState } from "react";
import { getClientAppointmentsWithPayments } from "@/lib/api";

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
  attendance?: "on_time" | "late" | "no_show" | "attended";
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

type ApiOut = { summary?: Summary; rows?: Row[] };

export default function ClientPaymentsClient() {
  const [data, setData] = useState<ApiOut | null>(null);
  const [err, setErr] = useState<string>("");

  useEffect(() => {
    (async () => {
      try {
        const res: ApiOut = await getClientAppointmentsWithPayments();
        setData(res || {});
      } catch (e: unknown) {
        setErr(e instanceof Error ? e.message : String(e));
      }
    })();
  }, []);

  const money = (c?: number | null) => ((c ?? 0) / 100).toFixed(2);

  if (err) return <div className="text-red-600 text-sm">{err}</div>;
  if (!data) return <div>Loading…</div>;

  // Safe defaults in case backend omitted "summary" or "rows"
  const summary: Summary = {
    total_appointments: 0,
    late_appointments: 0,
    paid_appointments: 0,
    unpaid_appointments: 0,
    total_expected_cents: 0,
    total_paid_cents: 0,
    total_owed_cents: 0,
    ...(data.summary || {}),
  };
  const rows: Row[] = data.rows || [];

  return (
    <div className="space-y-4">
      <div className="text-sm bg-gray-50 border rounded p-3">
        <b>Summary</b>: Paid ${money(summary.total_paid_cents)} · Owed ${money(summary.total_owed_cents)} ·{" "}
        Appointments {summary.total_appointments} ({summary.paid_appointments} paid / {summary.unpaid_appointments} unpaid)
      </div>

      <div className="overflow-auto">
        <table className="min-w-[720px] w-full text-sm border rounded">
          <thead className="bg-gray-50">
            <tr>
              <th className="p-2 text-left">When</th>
              <th className="p-2">Dur.</th>
              <th className="p-2">Status</th>
              <th className="p-2">Attendance</th>
              <th className="p-2 text-right">Price</th>
              <th className="p-2 text-right">Paid</th>
              <th className="p-2">Payment</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.id} className="border-t">
                <td className="p-2">{new Date(r.start_utc).toLocaleString()}</td>
                <td className="p-2 text-center">{r.duration_minutes}m</td>
                <td className="p-2 text-center">{r.status}</td>
                <td className="p-2 text-center">{r.attendance ?? "—"}</td>
                <td className="p-2 text-right">
                  {typeof r.price_cents === "number" ? `$${money(r.price_cents)}` : "—"}
                </td>
                <td className="p-2 text-right">${money(r.amount_paid_cents)}</td>
                <td className="p-2 text-center">
                  <span className="inline-block px-2 py-0.5 rounded text-xs bg-gray-100">
                    {r.payment_status}
                  </span>
                </td>
              </tr>
            ))}
            {!rows.length && (
              <tr>
                <td className="p-3 text-gray-500" colSpan={7}>
                  No appointments yet.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
