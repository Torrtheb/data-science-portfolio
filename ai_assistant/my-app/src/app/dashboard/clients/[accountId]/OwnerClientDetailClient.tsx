"use client";

import { useState } from "react";
import { ownerUpdateClient, ownerListBundles, ownerListBundleLedger, ownerTopUpBundle, ownerAdjustWallet } from "@/lib/api";

type Email = { id?: number; email: string; is_primary: boolean; unsubscribed?: boolean };
type Person = { id: number; full_name: string; email?: string | null };

type ClientDetail = {
  account_id: number;
  client_user_id: string;
  client_email?: string | null;
  client_name?: string | null;
  name?: string | null;                // account display name (not edited here)
  phone?: string | null;
  emergency_contact?: string | null;
  emails: Email[];
  people: Person[];
};

export default function OwnerClientDetailClient({ initial }: { initial: ClientDetail }) {
  function QuickWalletButton({ clientId }: { clientId: string }) {
    const [open, setOpen] = useState(false);
    const [loading, setLoading] = useState(false);
    const [wallet, setWallet] = useState<{ id: number; balance_cents: number } | null>(null);
    const [ledger, setLedger] = useState<Array<{ event: string; amount_cents: number; created_at: string }>>([]);
    const [topup, setTopup] = useState("");
    const [walletNote, setWalletNote] = useState("");
    const [wFrom, setWFrom] = useState<string>("");
    const [wTo, setWTo] = useState<string>("");

    async function load(filters?: { date_from?: string; date_to?: string }) {
      setLoading(true);
      try {
        const bundles = await ownerListBundles(clientId);
        const wallets = (bundles || []).filter(b => (b.total_credits || 0) === 0);
        if (wallets.length) {
          const w = wallets[0];
          setWallet({ id: w.id, balance_cents: w.remaining_balance_cents ?? 0 });
          const rows = await ownerListBundleLedger(clientId, w.id, 10, filters);
          setLedger(
            rows.map((x: { event: string; amount_cents: number; created_at: string }) => ({
              event: x.event,
              amount_cents: x.amount_cents,
              created_at: x.created_at,
            }))
          );
        } else {
          setWallet(null);
          setLedger([]);
        }
      } finally {
        setLoading(false);
      }
    }

    return (
      <div className="relative">
        <button type="button" className="px-3 py-1 border rounded text-sm" onClick={async () => {
          if (!open) { await load(); setOpen(true); }
          else { setOpen(false); }
        }}>Wallet</button>
        {open && (
          <>
            {/* Backdrop */}
            <div className="fixed inset-0 bg-black/30 z-40" onClick={() => setOpen(false)} />
            {/* Sidebar Drawer */}
            <aside className="fixed top-0 right-0 h-full w-[min(100vw,420px)] bg-white z-50 shadow-2xl flex flex-col">
              <div className="px-4 py-3 border-b flex items-center justify-between">
                <div className="text-sm font-medium">Client Wallet</div>
                <button type="button" className="text-xs" onClick={() => setOpen(false)}>✕</button>
              </div>
              <div className="p-4 space-y-3 overflow-auto">
                {loading ? (
                  <div className="text-sm text-zinc-600 py-4">Loading…</div>
                ) : !wallet ? (
                  <div className="text-sm text-zinc-600">No wallet found.</div>
                ) : (
                  <>
                    <div className="text-sm">Balance: <b>${((wallet.balance_cents||0)/100).toFixed(2)}</b></div>
                    {/* Date filters */}
                    <div className="flex items-end gap-2">
                      <label className="text-sm">
                        <div className="text-xs text-zinc-600">From</div>
                        <input type="date" className="border rounded px-2 py-1" value={wFrom} onChange={(e)=>setWFrom(e.target.value)} />
                      </label>
                      <label className="text-sm">
                        <div className="text-xs text-zinc-600">To</div>
                        <input type="date" className="border rounded px-2 py-1" value={wTo} onChange={(e)=>setWTo(e.target.value)} />
                      </label>
                      <button type="button" className="px-2 py-1 border rounded text-sm" onClick={async ()=>{
                        await load({ date_from: wFrom || undefined, date_to: wTo || undefined });
                      }}>Search</button>
                      <button type="button" className="px-2 py-1 border rounded text-sm" onClick={async ()=>{
                        setWFrom(""); setWTo("");
                        await load();
                      }}>Reset</button>
                    </div>
                    {/* Adjust */}
                    <div className="flex flex-col gap-2">
                      <div className="flex items-center gap-2">
                        <input
                          className="border rounded px-2 py-1 text-sm w-28"
                          placeholder="Amount $"
                          type="number"
                          step={0.01}
                          min={0}
                          value={topup}
                          onChange={(e)=>setTopup(e.target.value)}
                        />
                        <input
                          className="border rounded px-2 py-1 text-sm flex-1"
                          placeholder="Note (optional)"
                          value={walletNote}
                          onChange={(e)=>setWalletNote(e.target.value)}
                        />
                      </div>
                      <div className="flex items-center gap-2">
                        <button type="button" className="px-2 py-1 border rounded text-sm" onClick={async ()=>{
                          const dollars = parseFloat(topup || '0');
                          const cents = Math.round((Number.isFinite(dollars) ? dollars : 0) * 100);
                          if (!wallet || cents <= 0) return;
                          await ownerTopUpBundle(clientId, wallet.id, cents, walletNote || undefined);
                          setTopup('');
                          setWalletNote('');
                          await load({ date_from: wFrom || undefined, date_to: wTo || undefined });
                        }}>Add</button>
                        <button type="button" className="px-2 py-1 border rounded text-sm" onClick={async ()=>{
                          const dollars = parseFloat(topup || '0');
                          const cents = Math.round((Number.isFinite(dollars) ? dollars : 0) * 100);
                          if (!wallet || cents <= 0) return;
                          await ownerAdjustWallet(clientId, wallet.id, -cents, walletNote || undefined);
                          setTopup('');
                          setWalletNote('');
                          await load({ date_from: wFrom || undefined, date_to: wTo || undefined });
                        }}>Remove</button>
                      </div>
                    </div>
                    {/* Ledger */}
                    <div className="text-xs text-zinc-500">Recent activity</div>
                    <div className="max-h-[50vh] overflow-auto border rounded">
                      {ledger.length ? (
                        <ul className="text-xs divide-y">
                          {ledger.map((l, i) => (
                            <li key={i} className="flex items-center justify-between px-2 py-1">
                              <span>{new Date(l.created_at).toLocaleString([], { dateStyle: 'short', timeStyle: 'short' })}</span>
                              <span>{l.event}</span>
                              <span>{l.amount_cents >= 0 ? `+$${(l.amount_cents/100).toFixed(2)}` : `-$${((-l.amount_cents)/100).toFixed(2)}`}</span>
                            </li>
                          ))}
                        </ul>
                      ) : (
                        <div className="text-xs text-zinc-600 p-2">No activity.</div>
                      )}
                    </div>
                  </>
                )}
              </div>
            </aside>
          </>
        )}
      </div>
    );
  }
  const [form, setForm] = useState({
    phone: initial.phone ?? "",
    emergency_contact: initial.emergency_contact ?? "",
    email1: initial.emails[0]?.email ?? "",
    email1Primary: initial.emails[0]?.is_primary ?? true,
    email1Unsub: initial.emails[0]?.unsubscribed ?? false,
    email2: initial.emails[1]?.email ?? "",
    email2Primary: initial.emails[1]?.is_primary ?? false,
    email2Unsub: initial.emails[1]?.unsubscribed ?? false,
  });
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [savedAt, setSavedAt] = useState<number | null>(null);

  const [people, setPeople] = useState<Person[]>(initial.people || []);
  const [pFullName, setPFullName] = useState("");
  const [pEmail, setPEmail] = useState("");

  function set<K extends keyof typeof form>(k: K, v: (typeof form)[K]) {
    setForm(prev => ({ ...prev, [k]: v }));
  }

  async function onSave(e: React.FormEvent) {
    e.preventDefault();
    setSaving(true);
    setError(null);
    try {
      const emailsPayload: { email: string; is_primary?: boolean; unsubscribed?: boolean }[] = [];
      if (form.email1.trim()) emailsPayload.push({ email: form.email1.trim(), is_primary: form.email1Primary, unsubscribed: form.email1Unsub });
      if (form.email2.trim()) emailsPayload.push({ email: form.email2.trim(), is_primary: form.email2Primary, unsubscribed: form.email2Unsub });

      if (emailsPayload.length > 2) throw new Error("At most two emails are allowed.");

      const updated = await ownerUpdateClient(initial.account_id, {
        phone: form.phone || null,
        emergency_contact: form.emergency_contact || null,
        emails: emailsPayload,
      });

      const e1 = updated.emails?.[0];
      const e2 = updated.emails?.[1];
      setForm(prev => ({
        ...prev,
        phone: updated.phone ?? "",
        emergency_contact: updated.emergency_contact ?? "",
        email1: e1?.email ?? "",
        email1Primary: e1 ? !!e1.is_primary : true,
        email1Unsub: e1 ? !!e1.unsubscribed : false,
        email2: e2?.email ?? "",
        email2Primary: e2 ? !!e2.is_primary : false,
        email2Unsub: e2 ? !!e2.unsubscribed : false,
      }));
      setSavedAt(Date.now());
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Save failed");
    } finally {
      setSaving(false);
    }
  }

  async function addPerson(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    try {
      const r = await fetch(`/api/back/owner/clients/${initial.account_id}/people`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ full_name: pFullName, email: pEmail || null }),
      });
      if (!r.ok) throw new Error(await r.text());
      const created = (await r.json()) as Person;
      setPeople(prev => [created, ...prev]);
      setPFullName("");
      setPEmail("");
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to add person");
    }
  }

  async function deletePerson(id: number) {
    setError(null);
    try {
      const r = await fetch(`/api/back/owner/clients/${initial.account_id}/people/${id}`, {
        method: "DELETE",
      });
      if (!(r.status === 204 || r.ok)) throw new Error(await r.text());
      setPeople(prev => prev.filter(p => p.id !== id));
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to delete person");
    }
  }

  return (
    <section className="space-y-6">
      <form onSubmit={onSave} className="border rounded p-4 space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="font-medium text-lg">Contact</h2>
          <QuickWalletButton clientId={initial.client_user_id} />
        </div>

        <div className="grid gap-3">
          <label className="grid gap-1">
            <span className="text-sm opacity-80">Phone</span>
            <input
              className="border rounded p-2"
              value={form.phone}
              onChange={(e) => set("phone", e.target.value)}
              placeholder="555-555-1234"
            />
          </label>

          <label className="grid gap-1">
            <span className="text-sm opacity-80">Emergency contact</span>
            <input
              className="border rounded p-2"
              value={form.emergency_contact}
              onChange={(e) => set("emergency_contact", e.target.value)}
              placeholder="Name + phone"
            />
          </label>

          <div className="grid gap-2">
            <span className="text-sm font-medium">Emails (max 2)</span>
            <div className="grid sm:grid-cols-[1fr_auto_auto] gap-2 items-center">
              <input
                className="border rounded p-2"
                type="email"
                placeholder="Primary email"
                value={form.email1}
                onChange={(e) => set("email1", e.target.value)}
              />
              <label className="inline-flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={form.email1Primary}
                  onChange={(e) => set("email1Primary", e.target.checked)}
                />
                Primary
              </label>
              <label className="inline-flex items-center gap-2 text-sm text-amber-700">
                <input
                  type="checkbox"
                  checked={form.email1Unsub}
                  onChange={(e) => set("email1Unsub", e.target.checked)}
                />
                Unsubscribed
              </label>
            </div>
            <div className="grid sm:grid-cols-[1fr_auto_auto] gap-2 items-center">
              <input
                className="border rounded p-2"
                type="email"
                placeholder="Secondary email"
                value={form.email2}
                onChange={(e) => set("email2", e.target.value)}
              />
              <label className="inline-flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={form.email2Primary}
                  onChange={(e) => set("email2Primary", e.target.checked)}
                />
                Primary
              </label>
              <label className="inline-flex items-center gap-2 text-sm text-amber-700">
                <input
                  type="checkbox"
                  checked={form.email2Unsub}
                  onChange={(e) => set("email2Unsub", e.target.checked)}
                />
                Unsubscribed
              </label>
            </div>
            <p className="text-xs text-zinc-600">
              Unsubscribed addresses will not receive broadcasts. They stay on file for reference but are skipped when sending.
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <button className="border rounded px-4 py-2" disabled={saving}>
            {saving ? "Saving..." : "Save changes"}
          </button>
          {savedAt && !error && <span className="text-sm text-green-700">Saved.</span>}
          {error && <span className="text-sm text-red-600">{error}</span>}
        </div>
      </form>

      <section className="border rounded p-4 space-y-3">
        <h2 className="font-medium">People on this account</h2>

        <ul className="space-y-2">
          {people.length ? (
            people.map((p) => (
              <li key={p.id} className="flex items-center justify-between border rounded p-2">
                <span>{p.full_name}{p.email ? ` (${p.email})` : ""}</span>
                <button
                  className="text-sm text-red-600 underline"
                  onClick={() => deletePerson(p.id)}
                >
                  Delete
                </button>
              </li>
            ))
          ) : (
            <li className="opacity-70">None</li>
          )}
        </ul>

        <form onSubmit={addPerson} className="grid gap-2 pt-2 sm:grid-cols-[1fr_1fr_auto]">
          <input
            className="border rounded p-2"
            placeholder="Full name"
            value={pFullName}
            onChange={(e) => setPFullName(e.target.value)}
            required
          />
          <input
            className="border rounded p-2"
            placeholder="Email (optional)"
            value={pEmail}
            onChange={(e) => setPEmail(e.target.value)}
          />
          <button className="border rounded px-4 py-2">Add</button>
        </form>
      </section>
    </section>
  );
}
