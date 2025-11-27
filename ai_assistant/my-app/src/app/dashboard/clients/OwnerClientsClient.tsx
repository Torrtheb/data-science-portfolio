"use client";
import { useState } from "react";

type Client = {
  account_id: number;
  client_user_id: string;
  client_email?: string | null;
  client_name?: string | null;
  people_count: number;
  name?: string | null;
};

export default function OwnerClientsClient({ initial }: { initial: Client[] }) {
  const [clients, setClients] = useState<Client[]>(initial);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<number | null>(null);

  const [email, setEmail] = useState("");
  const [name, setName] = useState("");

  async function addClient(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setPending(true);
    try {
      // 1) Provision NextAuth user first (must succeed)
      const provRes = await fetch("/api/admin/provision-client", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, name }),
      });
      if (!provRes.ok) {
        const msg = await provRes.text().catch(() => "");
        throw new Error(msg || `Provision failed (${provRes.status})`);
      }

      // 2) Attach in FastAPI
      const createdRes = await fetch("/api/back/owner/clients", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, name: name || null }),
      });
      if (!createdRes.ok) {
        const msg = await createdRes.text().catch(() => "");
        throw new Error(msg || `Attach failed (${createdRes.status})`);
      }
      const created: Client = await createdRes.json();

      setClients(prev => [created, ...prev]);
      setEmail("");
      setName("");
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to add client");
    } finally {
      setPending(false);
    }
  }


    async function removeClient(account_id: number) {
    if (!confirm("Remove this client?")) return;
    setError(null);
    setDeletingId(account_id);
    try {
        const email = clients.find(c => c.account_id === account_id)?.client_email || null;

        // 1) Remove mapping in FastAPI (handles 204 cleanly)
        const res = await fetch(`/api/back/owner/clients/${account_id}`, { method: "DELETE" });
        if (!(res.status === 204 || res.ok)) {
        const txt = await res.text().catch(() => "");
        throw new Error(txt || `Delete failed (${res.status})`);
        }

        // 2) Also remove auth user in Prisma (if we know their email)
        if (email) {
        await fetch("/api/admin/deprovision-client", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ email }),
        }).catch(() => {});
        }

        // 3) Update UI
        setClients(prev => prev.filter(c => c.account_id !== account_id));
    } catch (err: unknown) {
        setError(err instanceof Error ? err.message : "Failed to remove client");
    } finally {
        setDeletingId(null);
    }
    }


  return (
    <section className="space-y-4">
      <form onSubmit={addClient} className="border rounded-lg p-4 space-y-3">
        <h2 className="text-lg font-medium">Add client</h2>
        <div className="grid sm:grid-cols-2 gap-2">
          <input
            className="border rounded p-2"
            type="email"
            required
            placeholder="Client email"
            value={email}
            onChange={e => setEmail(e.target.value)}
          />
          <input
            className="border rounded p-2"
            placeholder="Client name (optional)"
            value={name}
            onChange={e => setName(e.target.value)}
          />
        </div>
        <button className="border rounded px-4 py-2" disabled={pending}>
          {pending ? "Adding..." : "Add client"}
        </button>
        {error && <p className="text-sm text-red-600">{error}</p>}
      </form>

      <div className="border rounded-lg">
        <div className="px-4 py-3 border-b font-medium">Existing clients ({clients.length})</div>
        <ul className="divide-y">
          {clients.map(c => (
            <li key={c.account_id} className="p-4 flex items-center justify-between">
              <div>
                <div className="font-medium">
                  {c.name ?? c.client_name ?? c.client_email ?? `Account #${c.account_id}`}
                </div>
                <div className="text-sm opacity-80">Account ID: {c.account_id}</div>
                {(c.client_email || c.client_name) && (
                  <div className="text-sm opacity-80">
                    {c.client_name ? `${c.client_name} — ` : ""}{c.client_email ?? ""}
                  </div>
                )}
                <div className="text-sm">People: {c.people_count}</div>

              </div>
              <div className="flex items-center gap-3">
                <a className="text-sm underline" href={`/dashboard/clients/${c.account_id}`}>Details</a>
                <button
                  className="text-sm text-red-600 underline"
                  onClick={() => removeClient(c.account_id)}
                  disabled={pending || deletingId === c.account_id}
                >
                  {deletingId === c.account_id ? "Removing..." : "Remove"}
                </button>
              </div>
            </li>
          ))}
          {clients.length === 0 && (
            <li className="p-4 text-sm opacity-70">No clients yet.</li>
          )}
        </ul>
      </div>
    </section>
  );
}
