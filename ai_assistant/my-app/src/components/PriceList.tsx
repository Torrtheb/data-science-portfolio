// src/components/PriceList.tsx
"use client";
import { useEffect, useState } from "react";
import { getPublicPricing } from "@/lib/api";

export default function PriceList() {
  const [rows, setRows] = useState<{ duration_minutes: number; price_cents: number }[]>([]);
  const [adminFeeCents, setAdminFeeCents] = useState<number | null>(null);
  useEffect(() => {
    (async () => {
      const res: unknown = await getPublicPricing();
      const r = res as { options?: { duration_minutes: number; price_cents: number }[]; admin_fee_cents?: number };
      setRows(r.options || []);
      setAdminFeeCents(typeof r.admin_fee_cents === "number" ? r.admin_fee_cents : null);
    })();
  }, []);
  const money = (c: number) => (c / 100).toFixed(2);
  // Filter out durations we don't want to display to clients (e.g., 120)
  const visible = rows.filter((o) => o.duration_minutes !== 120 && o.duration_minutes <= 60);
  if (!visible.length && adminFeeCents == null) return null;
  return (
    <div className="border rounded p-3 text-sm">
      <div className="font-medium mb-1">Session Prices</div>
      {visible.length > 0 && (
        <ul className="list-disc pl-5">
          {visible.map((o) => (
            <li key={o.duration_minutes}>{o.duration_minutes} min — ${money(o.price_cents)}</li>
          ))}
        </ul>
      )}
      {typeof adminFeeCents === "number" && (
        <div className="mt-2 text-xs text-gray-600">Administration fee: ${money(adminFeeCents)}</div>
      )}
    </div>
  );
}
