// src/app/dashboard/pricing/ui/OwnerPricingClient.tsx
"use client";

import { useEffect, useState } from "react";
import {
  ownerListServiceOptions,
  ownerReplaceServiceOptions,
  ownerGetAdminFeeSettings,
  ownerUpdateAdminFeeSettings,
  ownerCreateAdminFeeCharge,
  getOwnerSettings,
  updateOwnerSettings,
} from "@/lib/api";
import ClientPicker from "@/components/ClientPicker";

type ResolvedAccount = {
  account_id: number;
  client_user_id: string;
  client_email: string | null;
  client_name: string | null;
  name: string | null;
  people_count: number;
};

type Row = { duration_minutes: number; price_cents: number; currency?: string; is_active?: boolean };

export default function OwnerPricingClient() {
  const ALLOWED_DURATIONS = [15, 30, 45, 60];
  const [rows, setRows] = useState<Row[]>([]);
  const [priceInputs, setPriceInputs] = useState<Record<number, string>>({});
  const [saving, setSaving] = useState(false);
  const [ok, setOk] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [adminFeeInput, setAdminFeeInput] = useState("15.00");
  const [groupPrice60, setGroupPrice60] = useState<string>("0.00");
  const [lookupQuery, setLookupQuery] = useState("");
  const [lookupLoading, setLookupLoading] = useState(false);
  const [resolvedAccount, setResolvedAccount] = useState<ResolvedAccount | null>(null);
  const [lookupError, setLookupError] = useState<string | null>(null);
  const [chargeError, setChargeError] = useState<string | null>(null);
  const [chargeSuccess, setChargeSuccess] = useState<string | null>(null);
  const [chargeSubmitting, setChargeSubmitting] = useState(false);
  const [clientPick, setClientPick] = useState<{ id: string; account_id: number; name?: string | null; email: string } | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const [data, adminFee, settings] = await Promise.all([
          ownerListServiceOptions(),
          ownerGetAdminFeeSettings().catch(() => ({ admin_fee_cents: 1500 })),
          getOwnerSettings().catch(() => ({ appt_edge_buffer_min: 5, group_price_60_cents: 0 })),
        ]);
        // Normalize to always show allowed durations (backend constraint)
        const map = new Map<number, Row>(data.map(d => [d.duration_minutes, d]));
        // Remove any durations not in our allowed UI set (e.g., 120)
        Array.from(map.keys()).forEach((k) => { if (!ALLOWED_DURATIONS.includes(k)) map.delete(k); });
        ALLOWED_DURATIONS.forEach(d => {
          if (!map.has(d)) map.set(d, { duration_minutes: d, price_cents: 0, currency: "USD", is_active: true });
        });
        const list = Array.from(map.values()).sort((a, b) => a.duration_minutes - b.duration_minutes);
        setRows(list);
        // seed input strings with formatted values (allow free typing later)
        const seeded: Record<number, string> = {};
        list.forEach(r => { seeded[r.duration_minutes] = ((r.price_cents ?? 0) / 100).toFixed(2); });
        setPriceInputs(seeded);
        setAdminFeeInput(((adminFee.admin_fee_cents ?? 1500) / 100).toFixed(2));
        setGroupPrice60((((settings as any).group_price_60_cents ?? 0) / 100).toFixed(2));
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        setErr(message || "Failed to load pricing");
      }
    })();
  }, []);

  // Money input helpers
  const cleanMoneyInput = (raw: string): string => {
    const s = raw.replace(",", ".");
    // keep only digits and first dot
    let out = "";
    let dot = false;
    for (const ch of s) {
      if (ch >= "0" && ch <= "9") out += ch;
      else if ((ch === "." || ch === "·") && !dot) { out += "."; dot = true; }
    }
    if (dot) {
      const [a, b = ""] = out.split(".");
      return a + "." + b.slice(0, 2); // limit to 2 decimals while typing
    }
    return out;
  };

  const normalizeMoneyString = (s: string): string => {
    const cents = parseMoneyToCents(s);
    return (cents / 100).toFixed(2);
  };

  const handlePriceChange = (dur: number, raw: string) => {
    const cleaned = cleanMoneyInput(raw);
    setPriceInputs(prev => ({ ...prev, [dur]: cleaned }));
  };

  const handlePriceBlur = (dur: number) => {
    const normalized = normalizeMoneyString(priceInputs[dur] ?? "0");
    const cents = parseMoneyToCents(normalized);
    setPriceInputs(prev => ({ ...prev, [dur]: normalized }));
    setRows(old => old.map(r => (r.duration_minutes === dur ? { ...r, price_cents: cents } : r)));
  };

  const parseMoneyToCents = (value: string) => {
    const dollars = parseFloat(value || "0");
    if (!Number.isFinite(dollars)) return 0;
    return Math.round(dollars * 100);
  };

  const save = async () => {
    setSaving(true);
    setErr(null);
    setOk(null);
    try {
      const payload = rows
        .filter(r => ALLOWED_DURATIONS.includes(r.duration_minutes))
        .map(r => ({
        duration_minutes: r.duration_minutes,
        price_cents: parseMoneyToCents(priceInputs[r.duration_minutes] ?? ((r.price_cents ?? 0) / 100).toString()),
        currency: r.currency ?? "USD",
        is_active: r.is_active ?? true,
      }));
      const res = await ownerReplaceServiceOptions(payload);
      const list = (res as Row[]).sort((a, b) => a.duration_minutes - b.duration_minutes);
      setRows(list);
      const feeCents = parseMoneyToCents(adminFeeInput);
      await ownerUpdateAdminFeeSettings(feeCents);
      setAdminFeeInput((feeCents / 100).toFixed(2));
      setOk("Saved!");
      setTimeout(() => setOk(null), 2000);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setErr(message || "Save failed");
    } finally {
      setSaving(false);
    }
  };

  const _money = (c: number) => (c / 100).toFixed(2);

  // Update resolved account from client pick (unified client search)
  useEffect(() => {
    if (clientPick) {
      setResolvedAccount({
        account_id: clientPick.account_id,
        client_user_id: clientPick.id,
        client_email: clientPick.email,
        client_name: clientPick.name ?? null,
        name: clientPick.name ?? null,
        people_count: 0,
      });
      setLookupError(null);
    } else {
      setResolvedAccount(null);
    }
  }, [clientPick]);

  async function addFeeToClient() {
    if (!resolvedAccount) {
      setChargeError("Lookup a client first");
      return;
    }
    const amountCents = parseMoneyToCents(adminFeeInput);
    if (amountCents <= 0) {
      setChargeError("Administration fee must be greater than $0.00");
      return;
    }
    setChargeSubmitting(true);
    setChargeError(null);
    setChargeSuccess(null);
    try {
      const charge = await ownerCreateAdminFeeCharge({
        client_account_id: resolvedAccount.account_id,
        amount_cents: amountCents,
        note: "Administration fee",
      });
      const dollars = (charge.amount_cents / 100).toFixed(2);
      setChargeSuccess(`Added $${dollars} admin fee to ${resolvedAccount.name ?? resolvedAccount.client_name ?? resolvedAccount.client_email ?? "client"}.`);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setChargeError(message || "Failed to add administration fee");
    } finally {
      setChargeSubmitting(false);
    }
  }

  return (
    <div className="max-w-xl space-y-4">
      <h2 className="text-xl font-semibold">Appointment Pricing</h2>
      <p className="text-sm text-muted-foreground">
        Set default prices (shown on new appointments).
      </p>

      <div className="space-y-3">
        {rows.map(r => (
          <div key={r.duration_minutes} className="flex items-center justify-between rounded-xl border p-3">
            <div className="font-medium">{r.duration_minutes} minutes</div>
            <div className="flex items-center gap-2">
              <span className="text-sm">$</span>
              <input
                inputMode="decimal"
                pattern="^[0-9]*[.,]?[0-9]{0,2}$"
                placeholder="0.00"
                className="w-28 rounded-lg border px-3 py-1"
                value={priceInputs[r.duration_minutes] ?? ((r.price_cents ?? 0) / 100).toFixed(2)}
                onChange={(e) => handlePriceChange(r.duration_minutes, e.target.value)}
                onBlur={() => handlePriceBlur(r.duration_minutes)}
              />
            </div>
          </div>
        ))}
      </div>

      <div className="space-y-3 rounded-xl border p-3">
        <div className="flex items-center justify-between">
          <div>
            <div className="font-medium">Group lesson (60m)</div>
            <div className="text-xs text-muted-foreground">Per attendee price for 60-minute group lessons.</div>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-sm">$</span>
            <input
              inputMode="decimal"
              className="w-28 rounded-lg border px-3 py-1"
              value={groupPrice60}
              onChange={(e) => setGroupPrice60(e.target.value)}
            />
            <button
              type="button"
              className="rounded-xl border px-3 py-1 text-sm"
              onClick={async () => {
                try {
                  const cents = parseMoneyToCents(groupPrice60);
                  await updateOwnerSettings({ appt_edge_buffer_min: 5, group_price_60_cents: cents });
                  setOk("Group price saved!");
                  setTimeout(() => setOk(null), 2000);
                } catch (e: any) {
                  setErr(e?.message || "Failed to save group price");
                }
              }}
            >
              Save
            </button>
          </div>
        </div>
      </div>

      <div className="space-y-3 rounded-xl border p-3">
        <div className="flex items-center justify-between">
          <div>
            <div className="font-medium">Administration Fee</div>
            <div className="text-xs text-muted-foreground">Default fee applied when charging a client.</div>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-sm">$</span>
            <input
              inputMode="decimal"
              className="w-28 rounded-lg border px-3 py-1"
              value={adminFeeInput}
              onChange={(e) => setAdminFeeInput(e.target.value)}
            />
          </div>
        </div>
        <div className="border-t pt-3 space-y-2">
          <div className="text-sm font-medium">Add fee to a client</div>
          <div className="flex flex-col gap-2">
            <ClientPicker
              value={clientPick}
              onChange={(v) => setClientPick(v)}
              placeholder="Search clients by name or email…"
              showEmailOnlyInInput
            />
          </div>
          {lookupError && <div className="text-xs text-red-600">{lookupError}</div>}
          {resolvedAccount && (
            <div className="rounded-md border bg-gray-50 px-3 py-2 text-xs text-muted-foreground">
              <div className="font-medium text-sm text-gray-700">Selected client</div>
              <div>{resolvedAccount.name ?? resolvedAccount.client_name ?? resolvedAccount.client_email ?? `Account #${resolvedAccount.account_id}`}</div>
              {resolvedAccount.people_count ? (
                <div>{resolvedAccount.people_count} linked people</div>
              ) : null}
            </div>
          )}
          <button
            type="button"
            onClick={addFeeToClient}
            disabled={!resolvedAccount || chargeSubmitting}
            className="rounded-xl bg-black px-3 py-1 text-sm text-white disabled:opacity-50"
          >
            {chargeSubmitting ? "Adding..." : "Add Administration Fee"}
          </button>
          {chargeSuccess && <div className="text-xs text-green-600">{chargeSuccess}</div>}
          {chargeError && <div className="text-xs text-red-600">{chargeError}</div>}
        </div>
      </div>

      <div className="flex items-center gap-3">
        <button
          onClick={save}
          disabled={saving}
          className="rounded-xl bg-black text-white px-4 py-2 disabled:opacity-50"
        >
          {saving ? "Saving..." : "Save"}
        </button>
        {ok && <span className="text-green-600 text-sm">{ok}</span>}
        {err && <span className="text-red-600 text-sm">{err}</span>}
      </div>
    </div>
  );
}
