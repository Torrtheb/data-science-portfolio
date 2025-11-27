"use client";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { useSession } from "next-auth/react";
import TimezoneSelect from "@/components/TimezoneSelect";
import { getClientProfile, updateClientProfile, getClientWallet, ownerResolveClientAccount, ownerListBundles, ownerListBundleLedger, ownerTopUpBundle, ownerAdjustWallet } from "@/lib/api";

type Person = { id: number; full_name: string; email?: string | null };
type ClientEmail = { id: number; email: string; is_primary: boolean };
type Profile = {
  account_id: number;
  name?: string | null;
  phone?: string | null;
  emergency_contact?: string | null;
  emails: ClientEmail[];
  people: Person[];
};

export default function ProfileClient({
  initialTimezone,
  role = "CLIENT",
}: {
  initialTimezone: string;
  role?: "OWNER" | "CLIENT";
}) {
  const router = useRouter();
  const { data: session, status: _sessionStatus, update } = useSession();

  // ---- Shared: timezone state ----
  const [tz, setTz] = useState(initialTimezone);
  const [savingTz, setSavingTz] = useState(false);

  // ---- Client-only profile state ----
  const [profile, setProfile] = useState<Profile | null>(null);
  const [pending, setPending] = useState(role !== "OWNER"); // OWNER doesn't need to load profile
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Wallet state (CLIENT only)
  const [wallet, setWallet] = useState<{ balance_cents: number; transactions: Array<{ event: string; amount_cents: number; appointment_id: string; note?: string | null; created_at: string }>; appointments_count: number } | null>(null);
  const [walletLimit, setWalletLimit] = useState<number>(10);
  const [loadingMore, setLoadingMore] = useState<boolean>(false);

  // local edit fields (CLIENT only)
  const [phone, setPhone] = useState("");
  const [emergency, setEmergency] = useState("");
  const [email1, setEmail1] = useState("");
  const [email1Primary, setEmail1Primary] = useState(true);
  const [email2, setEmail2] = useState("");
  const [email2Primary, setEmail2Primary] = useState(false);

  // people add form (CLIENT only)
  const [fullName, setFullName] = useState("");
  const [personEmail, setPersonEmail] = useState("");

  // --- Save timezone (shared) ---
  async function saveTimezone(e: React.FormEvent) {
    e.preventDefault();
    setSavingTz(true);
    try {
      const r = await fetch("/api/account/timezone", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ timezone: tz }),
      });
      if (!r.ok) throw new Error(await r.text());

      // Update the JWT/session so UI reflects the new tz immediately
      await update({ timezone: tz });

      // Soft refresh server components (if any) that read session on the server
      router.refresh();
    } catch (err: unknown) {
      alert(err instanceof Error ? err.message : "Failed to save timezone");
    } finally {
      setSavingTz(false);
    }
  }

  // --- Load client profile (CLIENT only) ---
  async function loadAll() {
    if (role === "OWNER") return; // no-op for owners
    setPending(true);
    setError(null);
    try {
      const p = await getClientProfile();
      // Load wallet in parallel (don’t block profile)
      getClientWallet(walletLimit)
        .then(setWallet)
        .catch(() => setWallet({ balance_cents: 0, transactions: [], appointments_count: 0 }));

      // initial form state
      setPhone(p.phone ?? "");
      setEmergency(p.emergency_contact ?? "");

      const e1 = p.emails[0];
      const e2 = p.emails[1];
      const signupEmail = session?.user?.email ?? "";
      // Autofill with sign-up email if no saved emails exist yet
      setEmail1(e1?.email ?? signupEmail);
      setEmail1Primary(e1 ? !!e1.is_primary : true);
      setEmail2(e2?.email ?? "");
      setEmail2Primary(e2 ? !!e2.is_primary : false);

      if (!p.people || !Array.isArray(p.people)) p.people = [];
      setProfile(p);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to load profile");
      setProfile(null);
    } finally {
      setPending(false);
    }
  }

  useEffect(() => {
    if (role === "CLIENT") void loadAll();
    else setPending(false); // OWNER: nothing to load
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [role, session?.user?.email]);

  // When walletLimit increases (via "Show 10 more"), fetch more
  useEffect(() => {
    if (role !== "CLIENT") return;
    let cancelled = false;
    setLoadingMore(true);
    (async () => {
      try {
        const data = await getClientWallet(walletLimit);
        if (!cancelled) setWallet(data);
      } finally {
        if (!cancelled) setLoadingMore(false);
      }
    })();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [walletLimit]);

  // --- Save client profile (CLIENT only) ---
  async function saveProfile(e: React.FormEvent) {
    e.preventDefault();
    if (!profile) return;
    setSaving(true);
    setError(null);
    try {
      const emailsPayload: { email: string; is_primary?: boolean }[] = [];
      if (email1.trim()) emailsPayload.push({ email: email1.trim(), is_primary: email1Primary });
      if (email2.trim()) emailsPayload.push({ email: email2.trim(), is_primary: email2Primary });

      const updated = await updateClientProfile({
        phone: phone || null,
        emergency_contact: emergency || null,
        emails: emailsPayload,
      });
      setProfile(updated);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to save");
    } finally {
      setSaving(false);
    }
  }

  // --- People editing (CLIENT only) ---
  async function addPerson(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    try {
      const res = await fetch("/api/back/api/me/people", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ full_name: fullName, email: personEmail || null }),
      });
      if (!res.ok) throw new Error((await res.text()) || `Failed to add person (${res.status})`);
      const person = (await res.json()) as Person;
      setProfile((p) => (p ? { ...p, people: [...(p.people || []), person] } : p));
      setFullName("");
      setPersonEmail("");
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to add person");
    }
  }

  async function deletePerson(id: number) {
    setError(null);
    try {
      const res = await fetch(`/api/back/api/me/people/${id}`, { method: "DELETE" });
      if (!res.ok) throw new Error((await res.text()) || `Failed to delete person (${res.status})`);
      setProfile((p) => (p ? { ...p, people: (p.people || []).filter((x) => x.id !== id) } : p));
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to delete person");
    }
  }

  // ---- OWNER view: only timezone card ----
  // Keep as a custom hook-style util to avoid hooks lint (not used)
  function useOwnerWalletManager() {
    const [open, setOpen] = useState(false);
    const [q, setQ] = useState("");
    const [resolving, setResolving] = useState(false);
    const [clientId, setClientId] = useState<string | null>(null);
    const [wallet, setWallet] = useState<{ id: number; balance_cents: number } | null>(null);
    const [ledger, setLedger] = useState<Array<{ event: string; amount_cents: number; created_at: string }>>([]);
    const [topup, setTopup] = useState("");
    // Note removed per request

    async function resolveAndLoad() {
      setResolving(true);
      try {
        const r = await ownerResolveClientAccount(q.trim());
        const cid = r.client_user_id as string;
        setClientId(cid);
        const bundles = await ownerListBundles(cid);
        const wallets = (bundles || []).filter(b => (b.total_credits || 0) === 0);
        if (wallets.length) {
          const w = wallets[0];
          setWallet({ id: w.id, balance_cents: w.remaining_balance_cents ?? 0 });
          const rows = await ownerListBundleLedger(cid, w.id, 5);
          setLedger(rows.map((x: { event: string; amount_cents: number; created_at: string }) => ({ event: x.event, amount_cents: x.amount_cents, created_at: x.created_at })));
        } else {
          setWallet(null);
          setLedger([]);
        }
      } catch (e: unknown) {
        alert(e instanceof Error ? e.message : "Client not found");
      } finally {
        setResolving(false);
      }
    }

    return (
      <section className="border rounded-lg p-4 space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-medium">Client Wallet (owner)</h2>
          <button className="border rounded px-2 py-1 text-sm" onClick={() => setOpen(v=>!v)}>{open ? 'Hide' : 'Open'}</button>
        </div>
        {open && (
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <input className="border rounded p-2 flex-1" placeholder="Search name or email" value={q} onChange={(e)=>setQ(e.target.value)} />
              <button className="border rounded px-3 py-2" onClick={resolveAndLoad} disabled={resolving}>{resolving ? 'Loading…' : 'Select'}</button>
            </div>
            {clientId && (
              <div className="space-y-2">
                <div className="text-sm">Balance: <b>${((wallet?.balance_cents ?? 0)/100).toFixed(2)}</b></div>
                <div className="flex flex-col gap-2">
                  <div className="flex items-center gap-2">
                    <input className="border rounded px-2 py-1 text-sm w-28" placeholder="Amount $" type="number" step="0.01" min={0} value={topup} onChange={(e)=>setTopup(e.target.value)} />
                  </div>
                  <div className="flex items-center gap-2">
                    <button className="border rounded px-2 py-1 text-sm" onClick={async ()=>{
                      const dollars = parseFloat(topup || '0');
                      const cents = Math.round((Number.isFinite(dollars) ? dollars : 0)*100);
                      if (!wallet || cents <= 0 || !clientId) return;
                      try {
                        await ownerTopUpBundle(clientId, wallet.id, cents);
                        setTopup('');
                        await resolveAndLoad();
                      } catch(e: unknown) {
                        alert(e instanceof Error ? e.message : 'Failed to add funds');
                      }
                    }}>Add</button>
                    <button className="border rounded px-2 py-1 text-sm" onClick={async ()=>{
                      const dollars = parseFloat(topup || '0');
                      const cents = Math.round((Number.isFinite(dollars) ? dollars : 0)*100);
                      if (!wallet || cents <= 0 || !clientId) return;
                      try {
                        await ownerAdjustWallet(clientId, wallet.id, -cents);
                        setTopup('');
                        await resolveAndLoad();
                      } catch(e: unknown) {
                        alert(e instanceof Error ? e.message : 'Failed to remove funds');
                      }
                    }}>Remove</button>
                  </div>
                </div>
                <div className="text-sm text-zinc-600">Recent</div>
                <div className="border rounded">
                  {ledger.length ? (
                    <ul className="divide-y text-sm">
                      {ledger.map((l, i) => (
                        <li key={i} className="flex items-center justify-between px-3 py-2">
                          <span>{new Date(l.created_at).toLocaleString([], { dateStyle: 'short', timeStyle: 'short' })}</span>
                          <span>{l.event}</span>
                          <span>{l.amount_cents >= 0 ? `+$${(l.amount_cents/100).toFixed(2)}` : `-$${((-l.amount_cents)/100).toFixed(2)}`}</span>
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <div className="text-sm text-zinc-500 px-3 py-2">No activity.</div>
                  )}
                </div>
              </div>
            )}
          </div>
        )}
      </section>
    );
  }
  if (role === "OWNER") {
    return (
      <section className="p-6 space-y-6 max-w-xl">
        <h1 className="text-2xl font-semibold">My Profile</h1>

        <form onSubmit={saveTimezone} className="border rounded-lg p-4 space-y-3">
          <h2 className="text-lg font-medium">My Account Timezone</h2>
          <TimezoneSelect value={tz} onChange={setTz} showUseSystemButton />
          <button className="border rounded px-4 py-2" disabled={savingTz}>
            {savingTz ? "Saving…" : "Save timezone"}
          </button>
        </form>

        {/* Owner-only wallet manager removed per request */}
      </section>
    );
  }

  // ---- CLIENT view (existing UI) ----
  if (pending) return <div className="p-6">Loading…</div>;

  if (error) {
    return (
      <main className="p-6 max-w-xl space-y-4">
        <p className="text-red-600">{error}</p>
        <button onClick={loadAll} className="underline">Retry</button>
      </main>
    );
  }

  if (!profile) return <div className="p-6">No profile data.</div>;

  return (
    <section className="p-6 space-y-6 max-w-xl">
      <h1 className="text-2xl font-semibold">My Profile</h1>

      <form onSubmit={saveProfile} className="border rounded-lg p-4 space-y-3">
        <div className="grid gap-3">
          <label className="grid gap-1">
            <span className="text-sm opacity-80">Phone</span>
            <input className="border rounded p-2" value={phone} onChange={(e)=>setPhone(e.target.value)} />
          </label>
          <label className="grid gap-1">
            <span className="text-sm opacity-80">Emergency contact</span>
            <input className="border rounded p-2" value={emergency} onChange={(e)=>setEmergency(e.target.value)} />
          </label>

          <div className="grid gap-2">
            <span className="text-sm font-medium">Emails (max 2)</span>
            <div className="grid sm:grid-cols-[1fr_auto] gap-2">
              <input className="border rounded p-2" type="email" placeholder="Primary email" value={email1} onChange={(e)=>setEmail1(e.target.value)} />
              <label className="inline-flex items-center gap-2 text-sm">
                <input type="checkbox" checked={email1Primary} onChange={(e)=>setEmail1Primary(e.target.checked)} />
                Primary
              </label>
            </div>
            <div className="grid sm:grid-cols-[1fr_auto] gap-2">
              <input className="border rounded p-2" type="email" placeholder="Secondary email" value={email2} onChange={(e)=>setEmail2(e.target.value)} />
              <label className="inline-flex items-center gap-2 text-sm">
                <input type="checkbox" checked={email2Primary} onChange={(e)=>setEmail2Primary(e.target.checked)} />
                Primary
              </label>
            </div>
          </div>
        </div>
        <button className="border rounded px-4 py-2" disabled={saving}>
          {saving ? "Saving..." : "Save profile"}
        </button>
      </form>

      {/* --- My Account Timezone --- */}
      <form onSubmit={saveTimezone} className="border rounded-lg p-4 space-y-3">
        <h2 className="text-lg font-medium">My Account Timezone</h2>
        <TimezoneSelect value={tz} onChange={setTz} showUseSystemButton />
        <button className="border rounded px-4 py-2" disabled={savingTz}>
          {savingTz ? "Saving…" : "Save timezone"}
        </button>
      </form>

      <section className="border rounded-lg p-4 space-y-3">
        <h2 className="text-lg font-medium">My Wallet</h2>
        <div className="text-sm">Current balance: <b>${((wallet?.balance_cents ?? 0) / 100).toFixed(2)}</b></div>
        <div className="text-sm">Appointments (booked/completed): <b>{wallet?.appointments_count ?? 0}</b></div>
        <div className="text-sm text-zinc-600">Recent activity</div>
        <div className="border rounded max-h-64 overflow-auto">
          {(wallet?.transactions?.length || 0) > 0 ? (
            <ul className="divide-y">
              {wallet!.transactions.map((t, idx) => (
                <li key={idx} className="flex items-center justify-between px-3 py-2 text-sm">
                  <span>{new Date(t.created_at).toLocaleString([], { dateStyle: 'short', timeStyle: 'short' })}</span>
                  <span>{t.event}</span>
                  <span>{t.amount_cents >= 0 ? `+$${(t.amount_cents/100).toFixed(2)}` : `-$${((-t.amount_cents)/100).toFixed(2)}`}</span>
                </li>
              ))}
            </ul>
          ) : (
            <div className="text-sm text-zinc-500 px-3 py-2">No wallet activity yet.</div>
          )}
          {loadingMore && (
            <div className="flex items-center justify-center py-2">
              <span className="inline-block h-4 w-4 rounded-full border-2 border-zinc-300 border-t-transparent animate-spin" />
            </div>
          )}
        </div>
        {/* Show more button (fetches 10 more); show end state when no more */}
        {(wallet?.transactions?.length || 0) >= walletLimit && (
          <button
            className="px-3 py-1 border rounded text-sm"
            onClick={() => setWalletLimit((n) => n + 10)}
            disabled={loadingMore}
          >
            {loadingMore ? 'Loading…' : 'Show 10 more'}
          </button>
        )}
        {(wallet?.transactions?.length || 0) > 0 && (wallet?.transactions?.length || 0) < walletLimit && (
          <div className="text-xs text-zinc-500">No more activity</div>
        )}
        <div className="text-xs text-zinc-500">Only the business owner can add funds to your wallet.</div>
      </section>

      <section className="border rounded-lg p-4 space-y-3">
        <h2 className="text-lg font-medium">People on my account</h2>
        <ul className="space-y-2">
          {(profile.people || []).map((p) => (
            <li key={p.id} className="flex items-center justify-between border rounded p-2">
              <span>{p.full_name}{p.email ? ` (${p.email})` : ""}</span>
              <button onClick={() => deletePerson(p.id)} className="text-red-600 underline">Delete</button>
            </li>
          ))}
          {(!profile.people || profile.people.length === 0) && (
            <li className="text-sm opacity-70">No people added yet.</li>
          )}
        </ul>

        <form onSubmit={addPerson} className="space-y-2 pt-2">
          <input className="border rounded p-2 w-full" placeholder="Full name" value={fullName} onChange={e => setFullName(e.target.value)} required />
          <input className="border rounded p-2 w-full" placeholder="Email (optional)" value={personEmail} onChange={e => setPersonEmail(e.target.value)} />
          <button className="border rounded px-4 py-2">Add person</button>
        </form>
      </section>
    </section>
  );
}
