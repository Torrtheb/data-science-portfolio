"use client";

import React, { useEffect, useMemo, useState } from "react";
import { useSession } from "next-auth/react";
import { listOwnerClients, ownerBroadcastEmail, type OwnerClientLite } from "@/lib/api";
import { ClientRow } from "../types";
import ClientPicker from "@/components/ClientPicker"; // ← ADD

export default function BroadcastEmailPanel() {
  const [clients, setClients] = useState<ClientRow[]>([]);
  const [filter, setFilter] = useState("");
  const [selected, setSelected] = useState<Record<string, boolean>>({});
  const [selectAll, setSelectAll] = useState(true); // default: all
  const [subject, setSubject] = useState("");
  const [body, setBody] = useState("");
  const [sending, setSending] = useState(false);
  const [msg, setMsg] = useState("");
  const { status } = useSession();
  const [clientPick, setClientPick] = useState<OwnerClientLite | null>(null);

  const keyOf = (c: ClientRow, i: number) => {
    if (c.id) return `id:${c.id}`;
    if (c.email) return `email:${c.email}`;
    return `anon:${i}`; // unique fallback per position
  };

  // normalize strings for robust searching (lower, remove accents)
  const normalize = (s: string) =>
    (s || "")
      .toLowerCase()
      .normalize("NFKD")
      .replace(/[\u0300-\u036f]/g, ""); // strip diacritics

  useEffect(() => {
    if (status !== "authenticated") return;
    (async () => {
      try {
        const rows = await listOwnerClients();
        setClients(rows);
        const all: Record<string, boolean> = {};
        rows.forEach((c: ClientRow, i: number) => {
          all[keyOf(c, i)] = true;
        });
        setSelected(all);
        setSelectAll(true);
      } catch (e: any) {
        setMsg(e.message || String(e));
      }
    })();
  }, [status]);

  // Restore/save draft (subject/body) in localStorage
  useEffect(() => {
    try {
      const s = localStorage.getItem("owner.msg.subject");
      const b = localStorage.getItem("owner.msg.body");
      if (s !== null) setSubject(s);
      if (b !== null) setBody(b);
    } catch {}
     
  }, []);
  useEffect(() => {
    try { localStorage.setItem("owner.msg.subject", subject); } catch {}
  }, [subject]);
  useEffect(() => {
    try { localStorage.setItem("owner.msg.body", body); } catch {}
  }, [body]);

  const visible = useMemo(() => {
    if (clientPick?.id) {
      return clients.filter((c) => c.id === clientPick.id);
    }
    const q = normalize(filter.trim());
    if (!q) return clients;
    const tokens = q.split(/\s+/).filter(Boolean);
    return clients.filter((c) => {
      const hay = normalize(`${c.name ?? ""} ${c.email ?? ""}`);
      return tokens.every((t) => hay.includes(t));
    });
  }, [clients, filter, clientPick]);



  const toggleOne = (k: string) => {
    setSelected((prev) => {
      const next = { ...prev, [k]: !prev[k] };
      const allTrue =
        clients.length > 0 && clients.every((c, i) => !!next[keyOf(c, i)]);
      setSelectAll(allTrue);
      return next;
    });
  };

  const setAll = (v: boolean) => {
    const map: Record<string, boolean> = {};
    clients.forEach((c, i) => {
      map[keyOf(c, i)] = v;
    });
    setSelected(map);
    setSelectAll(v);
  };

  const doSend = async () => {
    setMsg("");
    if (!subject.trim()) {
      setMsg("Subject is required.");
      return;
    }

    setSending(true);
    try {
      const chosenIds = clients
        .filter((c, i) => selected[keyOf(c, i)] && !!c.id)
        .map((c) => c.id!);

      const payload: any = {
        subject: subject.trim(),
        text: body,
      };
      if (!selectAll) payload.client_user_ids = chosenIds;

      const res: any = await ownerBroadcastEmail(payload);
      setMsg(`Sent to ${res.recipients} client(s).`);
    } catch (e: any) {
      const raw = e?.message || String(e);
      if (raw.toLowerCase().includes("dev") && raw.toLowerCase().includes("email")) {
        setMsg("Email sending is disabled in this dev environment. Drafts can be exported or sent once SMTP is configured.");
      } else if (raw.toLowerCase().includes("confirm_send")) {
        setMsg("Broadcast blocked: confirm send is required. Please try again (we now auto-confirm), or check SMTP/dev settings.");
      } else {
        setMsg(`Error: ${raw}`);
      }
    } finally {
      setSending(false);
    }
  };
  const selectedCount = useMemo(() => {
    if (selectAll) return clients.length;
    return clients.reduce((acc, c, i) => acc + (selected[keyOf(c, i)] ? 1 : 0), 0);
  }, [selectAll, clients, selected]);

  return (
    <section className="space-y-4">
      <h2 className="text-xl font-medium">Broadcast Email</h2>

      {/* Recipients first, then composer */}
      <div className="bg-white rounded-xl border overflow-hidden">
        <div className="px-4 py-3 border-b flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="font-medium">Recipients</div>
            <span className="text-xs rounded-full bg-gray-100 px-2 py-1 text-gray-700">
              {selectAll ? `All (${clients.length})` : `${selectedCount} selected`}
            </span>
          </div>
          <label className="text-sm flex items-center gap-2">
            <input type="checkbox" checked={selectAll} onChange={(e) => setAll(e.target.checked)} />
            Select all
          </label>
        </div>

        <div className="p-4 space-y-3 border-b">
          <ClientPicker
            value={clientPick}
            onChange={(hit) => {
              setClientPick(hit);
              setFilter(hit?.email ?? "");
              if (hit) {
                setSelected((prev) => {
                  const next = { ...prev };
                  const row = clients.find(
                    (c) => c.id === hit.id || (c.email && hit.email && c.email.toLowerCase() === hit.email.toLowerCase())
                  );
              if (row) {
                    const key = row.id ? `id:${row.id}` : row.email ? `email:${row.email}` : null;
                    if (key) next[key] = true;
                  }
                  setSelectAll(false);
                  return next;
                });
              }
            }}
            placeholder="Type a client to add as recipient…"
            minChars={1}
            showEmailOnlyInInput={true}
          />

          <div className="flex items-center gap-2">
            <input
              placeholder="Or free-text filter name/email…"
              className="w-full border rounded-md px-3 py-2"
              value={filter}
              onChange={(e) => {
                setFilter(e.target.value);
                if (clientPick) setClientPick(null);
              }}
            />
            <button
              type="button"
              className="border rounded-md px-3 py-2 text-sm"
              onClick={() => { setFilter(""); setClientPick(null); }}
            >
              Clear
            </button>
          </div>
        </div>

        <div className="max-h-[40vh] overflow-auto" style={{ WebkitOverflowScrolling: 'touch' }}>
          {visible.map((c, i) => {
            const k = keyOf(c, i);
            return (
              <label key={k} className="flex items-center gap-2 px-4 py-2 border-b">
                <input type="checkbox" checked={!!selected[k]} onChange={() => toggleOne(k)} />
                <div>
                  <div className="text-sm">{c.name || "—"}</div>
                  <div className="text-xs text-gray-500">{c.email || "—"}</div>
                </div>
              </label>
            );
          })}
          {!visible.length && (
            <div className="px-4 py-6 text-sm text-gray-500">No clients match that filter.</div>
          )}
        </div>
      </div>

      <div className="bg-gray-50 rounded-xl p-4 space-y-3 relative">
        <div>
          <label className="block text-sm text-gray-700 mb-1">Subject *</label>
          <input
            className="w-full border rounded-md px-3 py-2"
            value={subject}
            onChange={(e) => setSubject(e.target.value)}
            placeholder="e.g., Schedule update for next week"
          />
        </div>
        <div>
          <label className="block text-sm text-gray-700 mb-1">Message (plain text)</label>
          <textarea
            className="w-full border rounded-md px-3 py-2 min-h-[200px]"
            value={body}
            onChange={(e) => setBody(e.target.value)}
            placeholder={`Hi there,\n\nJust a quick update...`}
          />
        </div>
        {/* Sticky action bar */}
        <div className="flex flex-wrap items-center gap-3 sticky bottom-2 bg-gray-50/80 backdrop-blur supports-[backdrop-filter]:bg-gray-50/60 border rounded-lg p-2">
          <button
            disabled={sending}
            className="px-4 py-2 bg-black text-white rounded-lg disabled:opacity-50"
            onClick={() => doSend()}
          >
            {sending ? "Sending…" : "Send"}
          </button>
          
          {msg && (
            <span className={`text-sm ${msg.startsWith("Error") ? "text-red-600" : "text-green-700"}`}>
              {msg}
            </span>
          )}
          <span className="ml-auto text-xs text-gray-500">
            {selectAll ? `Sending to all (${clients.length})` : `Sending to ${selectedCount}`}
          </span>
        </div>
      </div>
    </section>
  );
}
