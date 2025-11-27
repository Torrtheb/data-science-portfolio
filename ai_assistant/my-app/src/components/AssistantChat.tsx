// src/components/AssistantChat.tsx
"use client";
import { useEffect, useMemo, useRef, useState } from "react";
import { useChatStore, ChatMsg, Thread } from "@/store/chatStore";
import ClientPicker from "@/components/ClientPicker";
import { listOwnerClients } from "@/lib/api";

const EMPTY_MSGS: ChatMsg[] = [];


function tryParseEmailDraft(raw: string): {
  draft_id: string;
  to: string;
  to_name?: string;
  subject: string;
  text: string;
  recipients?: { email: string; name?: string | null }[];
} | null {
  try {
    const parsed = JSON.parse(raw);
    if (parsed?.marker === "email_draft" && parsed?.payload?.draft_id) {
      const p = parsed.payload;
      const out = {
        draft_id: p.draft_id,
        to: p.to,
        to_name: p.to_name,
        subject: p.subject,
        text: p.text,
        recipients: Array.isArray(p.recipients) ? p.recipients : [],
      };
      if ((!out.recipients || out.recipients.length === 0) && out.to) {
        out.recipients = [{ email: out.to, name: out.to_name }];
      }
      return out;
    }
  } catch {}
  if (raw.startsWith("[[TOOL:email_draft]]")) {
    try {
      const json = raw.replace("[[TOOL:email_draft]]", "");
      const p = JSON.parse(json);
      if (p?.draft_id) {
        const out = {
          draft_id: p.draft_id,
          to: p.to,
          to_name: p.to_name,
          subject: p.subject,
          text: p.text,
          recipients: Array.isArray(p.recipients) ? p.recipients : [],
        };
        if ((!out.recipients || out.recipients.length === 0) && out.to) {
          out.recipients = [{ email: out.to, name: out.to_name }];
        }
        return out;
      }
    } catch {}
  }
  return null;
}


function tryParsePendingMarker(raw: string): { draft_id: string; to: string; to_name?: string; subject: string; text: string } | null {
  if (typeof raw !== "string") return null;
  if (!raw.startsWith("PENDING_EMAIL_SEND:")) return null;
  const json = raw.slice("PENDING_EMAIL_SEND:".length).trim();
  try {
    const p = JSON.parse(json);
    if (p?.draft_id) return p;
  } catch {}
  return null;
}

  function _splitEmails(s: string): { email: string; name?: string | null }[] {
    return (s || "")
      .split(/[\n,;]+/)
      .map((t) => t.trim())
      .filter(Boolean)
      .map((email) => ({ email }));
  }

  function _joinRecipients(recs?: { email: string; name?: string | null }[]) {
    if (!Array.isArray(recs) || recs.length === 0) return "";
    return recs.map((r) => r.email).join(", ");
  }


function EmailDraftEditor({ data }: { data: unknown }) {
  type Rec = { email: string; name?: string | null };
  type EmailDraftLike = Partial<{
    draft_id: string;
    to: string;
    to_name: string;
    subject: string;
    text: string;
    recipients: Rec[];
  }>;
  const d = data as EmailDraftLike;

  const [subject, setSubject] = useState(d?.subject ?? "");
  const [text, setText] = useState(d?.text ?? "");

  const [recipients, setRecipients] = useState<Rec[]>(() => {
    const arr = Array.isArray(d?.recipients) ? (d.recipients as Rec[]) : [];
    if (arr.length > 0) return arr;
    if (d?.to) return [{ email: String(d.to), name: d?.to_name ?? null }];
    return [] as Rec[];
  });

  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [sent, setSent] = useState(false);
  const friendlyEmailError = (raw: string, status?: number) => {
    const low = (raw || "").toLowerCase();
    if (low.includes("dev") && low.includes("email")) {
      return "Email sending is disabled in this dev environment. The draft is kept in Outbox—enable SMTP/production settings to send.";
    }
    if (low.includes("sorry, email sending") || low.includes("5.7.0")) {
      return "Email sending is disabled by the SMTP provider for this account. Configure SMTP or send manually.";
    }
    if (status === 503) {
      return "Email sending is unavailable right now. Please try again later or send manually.";
    }
    const firstLine = (raw || "").split("\n")[0].trim();
    if (firstLine.length > 240) return firstLine.slice(0, 240) + "…";
    return firstLine || "Failed to send.";
  };

  // Owner clients and selection (broadcast-style)
  type ClientRow = { id?: string; name?: string; email?: string };
  const [clients, setClients] = useState<ClientRow[]>([]);
  const [filter, setFilter] = useState("");
  const [selected, setSelected] = useState<Record<string, boolean>>({});
  const [selectAll, setSelectAll] = useState(false);
  const [clientPick, setClientPick] = useState<{ id: string; name?: string | null; email: string } | null>(null);

  const addMessage = useChatStore((s) => s.addMessage);
  const sessionId  = useChatStore((s) => s.activeId);

  // Helpers
  const emailish = (s: string) => /\S+@\S+\.\S+/.test(s);
  const dedupe = (arr: Rec[]) => {
    const seen = new Set<string>();
    return arr.filter((r) => {
      const key = (r.email || "").toLowerCase();
      if (!key || seen.has(key)) return false;
      seen.add(key);
      return true;
    });
  };

  const keyOf = (c: ClientRow, i: number) => {
    if (c.id) return `id:${c.id}`;
    if (c.email) return `email:${c.email}`;
    return `anon:${i}`;
  };

  // Load owner clients and preselect any that match initial recipients
  useEffect(() => {
    (async () => {
      try {
        const rows = await listOwnerClients();
        setClients(rows);
        if (rows.length) {
          const recEmails = new Set((recipients || []).map((r) => (r.email || "").toLowerCase()));
          const map: Record<string, boolean> = {};
          rows.forEach((c, i) => {
            const em = (c.email || "").toLowerCase();
            if (em && recEmails.has(em)) map[keyOf(c, i)] = true;
          });
          setSelected(map);
          setSelectAll(rows.length > 0 && rows.every((c, i) => !!map[keyOf(c, i)]));
        }
      } catch {
        // ignore if unavailable; manual recipients remain
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const normalize = (s: string) => (s || "").toLowerCase().normalize("NFKD").replace(/[\u0300-\u036f]/g, "");
  const visible = useMemo(() => {
    if (clientPick?.id) return clients.filter((c) => c.id === clientPick.id);
    const q = normalize(filter.trim());
    if (!q) return clients;
    const tokens = q.split(/\s+/).filter(Boolean);
    return clients.filter((c) => {
      const hay = normalize(`${c.name ?? ""} ${c.email ?? ""}`);
      return tokens.every((t) => hay.includes(t));
    });
  }, [clients, filter, clientPick]);

  const selectedCount = useMemo(() => {
    let n = 0;
    clients.forEach((c, i) => { if (selected[keyOf(c, i)]) n++; });
    return n;
  }, [clients, selected]);

  const toggleOne = (k: string) => {
    setSelected((prev) => ({ ...prev, [k]: !prev[k] }));
    setSelectAll(false);
  };
  const setAll = (v: boolean) => {
    const map: Record<string, boolean> = {};
    clients.forEach((c, i) => { map[keyOf(c, i)] = v; });
    setSelected(map);
    setSelectAll(v);
  };

  // Sync recipients array from selected clients + any non-client emails originally in draft
  useEffect(() => {
    const fromSelected: Rec[] = [];
    clients.forEach((c, i) => {
      const k = keyOf(c, i);
      if (selected[k] && c.email) fromSelected.push({ email: c.email, name: c.name ?? null });
    });
    const clientEmails = new Set(clients.map((c) => (c.email || "").toLowerCase()).filter(Boolean));
    const extras = (Array.isArray(d?.recipients) ? d.recipients : [])
      .filter((r) => r?.email && !clientEmails.has(r.email.toLowerCase()));
    setRecipients(dedupe([...fromSelected, ...extras]));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [clients, selected]);

  // local chips/dropdown removed in favor of broadcast-like UI

  // Approve & send
  const approveAndSend = async () => {
    if (!d?.draft_id) return;
    setBusy(true);
    setErr(null);
    try {
      const body: {
        approve: true;
        subject: string;
        text: string;
        recipients: Rec[];
        replace_recipients: boolean;
      } = {
        approve: true,
        subject,
        text,
        recipients,              // ← list of {email,name?}
        replace_recipients: true // ← make these authoritative
      };

      const res = await fetch(`/api/back/api/outbox/${encodeURIComponent(d.draft_id!)}/send`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        const msg = await res.text().catch(() => "");
        throw new Error(friendlyEmailError(msg, res.status));
      }

      if (sessionId) {
        addMessage({ role: "ai", content: "✅ Message sent successfully.", ts: Date.now() }, sessionId);
      }
      setSent(true);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Failed to send.";
      setErr(friendlyEmailError(msg));
    } finally {
      setBusy(false);
    }
  };

  const reject = async () => {
    if (!d?.draft_id) return;
    const reason = prompt("Reason?");
    if (reason == null) return;
    setBusy(true);
    setErr(null);
    try {
      const res = await fetch(`/api/back/api/outbox/${encodeURIComponent(d.draft_id!)}/reject`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ reason }),
      });
      if (!res.ok) throw new Error(`Reject failed (${res.status})`);
    } catch (e: unknown) {
      setErr(e instanceof Error ? e.message : "Failed to reject.");
    } finally {
      setBusy(false);
    }
  };

  // Recipients UI (search bar + checkbox list)
  return (
    <div className={`rounded-lg border p-3 bg-white shadow-sm space-y-3 ${sent ? "opacity-70 pointer-events-none" : ""}`}>
      <div className="text-xs text-gray-500">{sent ? "Email sent" : "Email draft"}</div>
      <div className="bg-white rounded-md border overflow-hidden">
        <div className="px-3 py-2 border-b flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="text-sm font-medium">Recipients</div>
            <span className="text-xs rounded-full bg-gray-100 px-2 py-0.5 text-gray-700">
              {selectAll ? `All (${clients.length})` : `${selectedCount} selected`}
            </span>
          </div>
          <label className="text-xs flex items-center gap-2">
            <input type="checkbox" checked={selectAll} onChange={(e) => setAll(e.target.checked)} />
            Select all
          </label>
        </div>

        <div className="p-3 space-y-2 border-b">
          <ClientPicker
            value={clientPick}
            onChange={(hit) => {
              setClientPick(hit);
              setFilter(hit?.email ?? "");
              if (hit) {
                setSelected((prev) => {
                  const next = { ...prev };
                  const idx = clients.findIndex((c) => c.id === hit.id || ((c.email || "").toLowerCase() === (hit.email || "").toLowerCase()));
                  if (idx >= 0) next[keyOf(clients[idx], idx)] = true;
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
              className="w-full border rounded-md px-3 py-1.5 text-sm"
              value={filter}
              onChange={(e) => { setFilter(e.target.value); if (clientPick) setClientPick(null); }}
            />
            <button type="button" className="border rounded-md px-2.5 py-1.5 text-xs" onClick={() => { setFilter(""); setClientPick(null); }}>
              Clear
            </button>
          </div>
        </div>

        <div className="max-h-48 overflow-auto">
          {visible.map((c, i) => {
            const k = keyOf(c, i);
            return (
              <label key={k} className="flex items-center gap-2 px-3 py-2 border-b">
                <input type="checkbox" checked={!!selected[k]} onChange={() => toggleOne(k)} />
                <div>
                  <div className="text-sm">{c.name || "—"}</div>
                  <div className="text-xs text-gray-500">{c.email || "—"}</div>
                </div>
              </label>
            );
          })}
          {!visible.length && (
            <div className="px-3 py-4 text-xs text-gray-500">No clients match that filter.</div>
          )}
        </div>
      </div>

      <div className="grid gap-2">
        <label className="text-xs text-gray-500">Subject</label>
        <input
          className="border rounded p-2 text-sm"
          value={subject}
          onChange={(e) => setSubject(e.target.value)}
          placeholder="Subject"
          disabled={busy || sent}
        />
      </div>

      <div className="grid gap-2">
        <label className="text-xs text-gray-500">Body</label>
        <textarea
          className="border rounded p-2 text-sm min-h-28"
          value={text}
          onChange={(e) => setText(e.target.value)}
          disabled={busy || sent}
        />
      </div>

      {err && <div className="text-sm text-red-600">{err}</div>}

      <div className="flex gap-2">
        <button
          onClick={approveAndSend}
          disabled={busy || sent}
          className="px-3 py-1 rounded bg-black text-white text-sm disabled:opacity-50"
        >
          {sent ? "Sent ✅" : busy ? "Sending…" : "Send"}
        </button>

        <button
          onClick={reject}
          disabled={busy || sent}
          className="px-3 py-1 rounded border text-sm disabled:opacity-50"
        >
          Reject
        </button>
      </div>
    </div>
  );
}

export default function AssistantChat() {
  const [pending, setPending] = useState(false);
  const esRef = useRef<EventSource | null>(null);
  const enterLock = useRef(false);
  const seenPendingRef = useRef(false);
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const [inputError, setInputError] = useState<string | null>(null);
  const MAX_INPUT_CHARS = 2000; // keep in sync with backend MAX_CHAT_Q_CHARS
  const lastSentRef = useRef<string>("");

  // --- Each selector returns ONE field (no object literal) ---
  const hydrated   = useChatStore((s) => s.hydrated);
  const sessionId  = useChatStore((s) => s.activeId);
  const q          = useChatStore((s) => s.drafts[sessionId ?? ""] ?? "");
  const messages   = useChatStore((s) => s.byThread[sessionId ?? ""] ?? EMPTY_MSGS);

  // actions (split individually to keep typing exact)
  const setMessages        = useChatStore((s) => s.setMessages);
  const addMessage         = useChatStore((s) => s.addMessage);
  const mutateMessages     = useChatStore((s) => s.mutateMessages);
  const resetThread        = useChatStore((s) => s.resetThread);
  const setSessionId       = useChatStore((s) => s.setSessionId);
  const newThread          = useChatStore((s) => s.newThread);
  const setDraft           = useChatStore((s) => s.setDraft);
  const _listThreads        = useChatStore((s) => s.listThreads);
  const bootstrapFromServer= useChatStore((s) => s.bootstrapFromServer);
  const getMessages        = useChatStore((s) => s.getMessages);
  // SSE UX: banner for transient errors/archived notices
  const [banner, setBanner] = useState<string | null>(null);

  // --- SAFETY FUSE (unchanged) ---
  useEffect(() => {
    if (!hydrated) {
      const t = setTimeout(() => {
        if (!useChatStore.getState().hydrated) {
          useChatStore.setState({ hydrated: true });
        }
      }, 300);
      return () => clearTimeout(t);
    }
  }, [hydrated]);

  // ✅ Bootstrap: prefer server list; only create if server has none
  useEffect(() => {
    if (!hydrated) return;

    (async () => {
      try {
        await bootstrapFromServer();

        const state = useChatStore.getState();
        const threadsNow: Thread[] = state.listThreads(); // already sorted
        const sid = state.activeId;
        const knownIds = new Set(Object.keys(state.threads));

        if (sid && knownIds.has(sid)) return;

        if (!sid || !knownIds.has(sid)) {
          if (threadsNow.length > 0) {
            state.setSessionId(threadsNow[0].id);
            return;
          } else {
            const createdId = await state.newThread("New chat");
            state.setSessionId(createdId);
            return;
          }
        }
      } catch {
        // ignore
      }
    })();
  }, [hydrated, bootstrapFromServer]);

  // ✅ History fetch (do not gate on threads[sessionId])
  useEffect(() => {
    if (!hydrated || !sessionId) return;

    if (getMessages(sessionId).length > 0) return;

    fetch(`/api/back/api/agent/history?session=${encodeURIComponent(sessionId)}`, { cache: "no-store" })
      .then((r) => (r.ok ? r.json() : { messages: [] }))
      .then((data) => {
        const msgs = (data as { messages?: unknown[] } | undefined)?.messages;
        const restored: ChatMsg[] = Array.isArray(msgs)
          ? msgs
              .filter((m: unknown) => (m as { role?: string })?.role !== "tool")
              .map((m: unknown) => ({
                role: ((m as { role?: string })?.role === "user") ? "user" : "ai",
                content: String((m as { content?: unknown })?.content ?? ""),
                ts: Date.now(),
              }))
          : [];
        if (useChatStore.getState().getMessages(sessionId).length === 0) {
          setMessages(restored, sessionId);
        }
      })
      .catch(() => {});
  }, [hydrated, sessionId, getMessages, setMessages]);

  // Auto-scroll to newest message on updates
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    // Scroll after paint; smooth if already near bottom
    requestAnimationFrame(() => {
      try {
        const nearBottom = (el.scrollHeight - el.scrollTop - el.clientHeight) < 40;
        if (nearBottom) {
          el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
        } else {
          el.scrollTop = el.scrollHeight;
        }
      } catch {}
    });
  }, [messages, pending]);

  async function hardResetChat() {
    esRef.current?.close();
    esRef.current = null;

    if (!sessionId) return;

    setMessages([], sessionId);
    setDraft(""); // <-- use the action to clear draft

    await fetch(`/api/back/api/agent/reset?session=${encodeURIComponent(sessionId)}`, {
      method: "DELETE",
    }).catch(() => {});
    resetThread(sessionId);
  }

  async function send() {
    const text = q.trim();
    if (!text || pending) return;
    if (text.length > MAX_INPUT_CHARS) {
      setInputError(`Message too long. Limit is ${MAX_INPUT_CHARS} characters.`);
      return;
    }
    setInputError(null);

    let tid = sessionId;
    if (!tid) {
      const created = await newThread("New chat");
      setSessionId(created);
      tid = created;
    }
    if (!tid) return;

    if (enterLock.current) return;
    enterLock.current = true;
    setTimeout(() => (enterLock.current = false), 250);

    // ⬇️ reset the marker-detection flag for this turn (PUT IT HERE)
    seenPendingRef.current = false;

    esRef.current?.close();
    setPending(true);

    addMessage({ role: "user", content: text, ts: Date.now() }, tid);

    const url = `/api/back/api/agent/chat?session=${encodeURIComponent(tid)}&q=${encodeURIComponent(text)}`;
    lastSentRef.current = q;
    const es = new EventSource(url);
    esRef.current = es;

    let assistantBuffer = "";

    es.onopen = () => {
      addMessage({ role: "ai", content: "", ts: Date.now() }, tid!);
    };


    es.onmessage = (e) => {
      console.log("[SSE data]", e.data);
      assistantBuffer += e.data;

      // Strip markers from visible text
      const stripMarkers = (s: string) => s
        .replace(/PENDING_EMAIL_SEND:\s*\{[\s\S]*?\}/g, "")
        .replace(/UI:EMAIL_DRAFT:\s*\{[\s\S]*?\}/g, "")
        .replace(/\[\[TOOL:email_draft\]\]\s*\{[\s\S]*?\}/g, "");

      // Detect embedded UI or PENDING marker once per turn, then insert a UI draft message
      if (!seenPendingRef.current && (assistantBuffer.includes("UI:EMAIL_DRAFT:") || assistantBuffer.includes("PENDING_EMAIL_SEND:") || assistantBuffer.includes("[[TOOL:email_draft]]") || (assistantBuffer.includes('"marker"') && assistantBuffer.includes('"email_draft"')))) {
        let payloadJson: string | null = null;
        const mUI = assistantBuffer.match(/UI:EMAIL_DRAFT:\s*(\{[\s\S]*?\})/);
        if (mUI) payloadJson = mUI[1];
        if (!payloadJson) {
          const m = assistantBuffer.match(/PENDING_EMAIL_SEND:\s*(\{[\s\S]*?\})/);
          if (m) payloadJson = m[1];
        }
        if (!payloadJson) {
          const mTool = assistantBuffer.match(/\[\[TOOL:email_draft\]\]\s*(\{[\s\S]*?\})/);
          if (mTool) payloadJson = mTool[1];
        }
        if (!payloadJson) {
          // JSON fallback marker: {"marker":"email_draft","payload":{...}}
          try {
            const braces = assistantBuffer.match(/\{[\s\S]*\}/g);
            if (braces && braces.length) {
              for (let k = braces.length - 1; k >= 0; k--) {
                try {
                  const j = JSON.parse(braces[k]);
                  if (j && j.marker === "email_draft" && j.payload) {
                    payloadJson = JSON.stringify(j.payload);
                    break;
                  }
                } catch {}
              }
            }
          } catch {}
        }
        if (payloadJson) {
          console.log("[UI] inserting email draft card from stream marker");
          addMessage({ role: "ai", content: `UI:EMAIL_DRAFT:${payloadJson}`, ts: Date.now() }, tid!);
          seenPendingRef.current = true;
        }
      }

      // Update the visible AI bubble, but NEVER overwrite a freshly inserted UI:EMAIL_DRAFT card.
      mutateMessages((prev: ChatMsg[]) => {
        const copy = prev.slice();
        const clean = stripMarkers(assistantBuffer);
        // When we've inserted a draft card this turn, hide the duplicate
        // Subject/Body details and "Would you like me to send" text from the bubble.
        const stripDraftDetails = (s: string) => {
          if (!seenPendingRef.current) return s;
          let out = s;
          const subjRe = /(\n|^)\s*(\*\*Subject:\*\*|Subject:)\s*[\s\S]*$/i;
          const bodyRe = /(\n|^)\s*(\*\*Body:\*\*|Body:)\s*[\s\S]*$/i;
          const askRe  = /\n+\s*Would you like me to send[\s\S]*$/i;
          // Cut at the earliest of Subject/Body sections
          const subjIdx = out.search(subjRe);
          const bodyIdx = out.search(bodyRe);
          let cut = -1;
          if (subjIdx >= 0) cut = subjIdx;
          if (bodyIdx >= 0) cut = cut >= 0 ? Math.min(cut, bodyIdx) : bodyIdx;
          if (cut >= 0) out = out.slice(0, cut).trimEnd();
          // Also strip trailing send question if present
          out = out.replace(askRe, "").trimEnd();
          return out;
        };
        const visible = stripDraftDetails(clean);
        // Find the last AI message that is NOT a UI:EMAIL_DRAFT card
        let idx = -1;
        for (let k = copy.length - 1; k >= 0; k--) {
          const m = copy[k];
          if (m?.role === "ai" && typeof m.content === "string" && !m.content.startsWith("UI:EMAIL_DRAFT:")) {
            idx = k; break;
          }
        }
        if (idx >= 0) {
          copy[idx] = { ...copy[idx], content: visible };
          return copy;
        }
        // If all AI messages are cards (or none exist), append a new text bubble
        return [...copy, { role: "ai", content: visible, ts: Date.now() }];
      }, tid!);
    };

    // Tool events suppressed (debug-only)

    // Backend may emit a dedicated 'error' event (distinct from network errors)
    // when the agent pipeline fails gracefully. Surface this as a clear banner
    // while preserving the ability to retry.
    es.addEventListener("error", (e) => {
      try {
        const data = String((e as MessageEvent).data || "").trim();
        const msg =
          data ||
          "Sorry, something went wrong while processing your request. Please try again.";
        setBanner(msg);
      } catch {
        setBanner(
          "Sorry, something went wrong while processing your request. Please try again."
        );
      }
    });


    const close = () => {
      es.close();
      if (esRef.current === es) esRef.current = null;
      setPending(false);
    };
    // Soft-archive: backend will emit a dedicated 'archived' event and then close
    es.addEventListener("archived", (e) => {
      try {
        const obj = JSON.parse(String((e as MessageEvent).data || ""));
        const msg = obj?.message || "This conversation is archived. Start a new chat to continue.";
        addMessage({ role: "ai", content: msg, ts: Date.now() }, tid!);
        setBanner("This conversation is archived. Start a new chat to continue.");
      } catch {
        addMessage({ role: "ai", content: "This conversation is archived. Start a new chat to continue.", ts: Date.now() }, tid!);
        setBanner("This conversation is archived. Start a new chat to continue.");
      }
      close();
    });
    es.addEventListener("done", close);
    es.addEventListener("rate_limit", (e) => {
      try {
        const obj = JSON.parse(String((e as MessageEvent).data || ""));
        const msg = obj?.message || "Rate limit exceeded. Please try again in a minute.";
        setBanner(msg);
      } catch {
        setBanner("Rate limit exceeded. Please try again in a minute.");
      }
      close();
    });
    es.onerror = () => {
      setBanner("Connection lost. Please resend or try again.");
      // Restore the last prompt into the input box so user can quickly resend
      if (lastSentRef.current) setDraft(lastSentRef.current);
      close();
    };

    setDraft(""); // clear input via action
  }

  // ---- RENDER ----
  return (
    <div className="rounded-xl border p-4 space-y-3">
      {banner && (
        <div className="rounded-md bg-yellow-50 border border-yellow-200 px-3 py-2 text-xs text-yellow-900 flex items-center justify-between gap-3">
          <span>{banner}</span>
          <button
            className="border rounded px-2 py-1 text-[11px]"
            onClick={() => {
              if (!pending && lastSentRef.current) {
                setDraft(lastSentRef.current);
                send();
              }
            }}
          >
            Retry
          </button>
        </div>
      )}
      <div className="flex items-center justify-between">
        <div className="font-medium">AI Scheduling Assistant</div>
        <div className="flex items-center gap-3">
          <button
            className="text-sm text-gray-600 hover:text-gray-900"
            onClick={hardResetChat}
            disabled={pending}
            title="Clear messages in this chat (keeps the same chat id)"
          >
            Clear
          </button>
        </div>
      </div>

      {!hydrated ? (
        <div className="rounded-xl border p-4">
          <div className="h-24 animate-pulse text-gray-500">Loading chat…</div>
        </div>
      ) : (
        <>
          <div className="flex gap-2">
            <textarea
              className="border rounded p-2 flex-1 whitespace-pre-wrap break-words resize-none min-h-[42px] max-h-[96px] overflow-y-auto"
              placeholder='Try: "What is my schedule today?"'
              value={q}
              onChange={(e) => setDraft(e.target.value)}
              maxLength={MAX_INPUT_CHARS}
              rows={1}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey && !pending) {
                  e.preventDefault();
                  send();
                }
              }}
              disabled={pending}
            />
            <button
              onClick={send}
              className="px-4 py-2 rounded bg-black text-white"
              disabled={pending || !q.trim()}
            >
              {pending ? "Sending…" : "Send"}
            </button>
          </div>
          <div className="flex items-center justify-between text-xs">
            <div className={`text-red-600 ${inputError ? "opacity-100" : "opacity-0"}`}>{inputError || "_"}</div>
            <div className="text-gray-500">{q.length}/{MAX_INPUT_CHARS}</div>
          </div>

          <div ref={scrollRef} className="border rounded p-3 h-72 overflow-auto whitespace-pre-wrap break-words space-y-2 bg-white/50 text-sm">
            {messages.length === 0 ? (
              <div className="text-gray-500">
                Ask about availability or bookings. I’ll call tools for you.
              </div>
            ) : (
              <>
            {messages.map((m: ChatMsg, i: number) => {
                  // Email editor from AI marker
                  if (m.role === "ai" && typeof m.content === "string") {
                    // Direct UI marker at start
                    if (m.content.startsWith("UI:EMAIL_DRAFT:")) {
                      try {
                        const json = m.content.slice("UI:EMAIL_DRAFT:".length).trim();
                        const parsed = JSON.parse(json);
                        console.log("[UI] rendering email draft editor from UI marker");
                        return <EmailDraftEditor key={i} data={parsed} />;
                      } catch {}
                    }
                    // Robust: marker embedded inside a longer message (history reload cases)
                    const mUI = m.content.match(/UI:EMAIL_DRAFT:\s*(\{[\s\S]*?\})/);
                    if (mUI) {
                      try {
                        const parsed = JSON.parse(mUI[1]);
                        return <EmailDraftEditor key={i} data={parsed} />;
                      } catch {}
                    }
                    // Pending-send marker (start or embedded)
                    const pending = tryParsePendingMarker(m.content) || (() => {
                      const mm = m.content.match(/PENDING_EMAIL_SEND:\s*(\{[\s\S]*?\})/);
                      if (!mm) return null;
                      try { return JSON.parse(mm[1]); } catch { return null; }
                    })();
                    if (pending) return <EmailDraftEditor key={i} data={pending} />;
                    const parsedDraft = tryParseEmailDraft(m.content);
                    if (parsedDraft) return <EmailDraftEditor key={i} data={parsedDraft} />;
                  }

                  // Default text bubble (Markdown)
                  return (
                    <div key={i} className={m.role === "user" ? "text-black" : "text-blue-700"}>
                      {m.role === "user" ? "You: " : "AI: "}
                      {typeof m.content === "string"
                        ? (m.role === "ai" ? renderMarkdown(m.content) : m.content)
                        : JSON.stringify(m.content)}
                    </div>
                  );
                })}
              </>
            )}
          </div>

        </>
      )}
    </div>
  );
}
// Minimal Markdown renderer: headings, bold/italic, code blocks, lists, images
function renderMarkdown(text: string) {
  // Escape HTML
  const esc = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

  // Extract code blocks
  const codeBlocks: string[] = [];
  let safe = text.replace(/```([\s\S]*?)```/g, (_, code) => {
    codeBlocks.push(esc(code));
    return `[[CODE_BLOCK_${codeBlocks.length - 1}]]`;
  });

  // Images ![alt](url)
  safe = safe.replace(/!\[([^\]]*)\]\((https?:\/\/[^)\s]+)\)/g, (_m, alt, url) => (
    `<img src="${url}" alt="${esc(alt)}" style="max-height:220px;max-width:100%;object-fit:cover;border-radius:8px;display:block;margin:6px 0;" />`
  ));

  // Bold and italics
  safe = safe
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\*([^*]+)\*/g, "<em>$1</em>");

  // Headings (##, #)
  safe = safe
    .replace(/^###\s+(.+)$/gm, '<h3 style="margin:6px 0 4px 0;font-weight:600;">$1</h3>')
    .replace(/^##\s+(.+)$/gm, '<h2 style="margin:8px 0 6px 0;font-weight:700;">$1</h2>')
    .replace(/^#\s+(.+)$/gm, '<h1 style="margin:10px 0 6px 0;font-weight:700;">$1</h1>');

  // Lists
  safe = safe.replace(/^(?:-\s+.+\n?)+/gm, (block) => {
    const items = block
      .trim()
      .split(/\n/)
      .map((l) => l.replace(/^[-*]\s+/, "").trim())
      .map((li) => `<li>${li}</li>`) 
      .join("");
    return `<ul style="margin:6px 0 6px 18px;list-style:disc;">${items}</ul>`;
  });

  // Paragraphs: split on blank lines
  const html = safe
    .split(/\n{2,}/)
    .map((para) => para.match(/^<h\d|^<ul|^<img/) ? para : `<p style="margin:6px 0;">${para.replace(/\n/g, '<br/>')}</p>`)
    .join("");

  // Rehydrate code blocks
  const finalHtml = html.replace(/\[\[CODE_BLOCK_(\d+)\]\]/g, (_, i) => (
    `<pre style="background:#f6f8fa;padding:10px;border-radius:6px;overflow:auto;"><code>${codeBlocks[Number(i)]}</code></pre>`
  ));

  return <span dangerouslySetInnerHTML={{ __html: finalHtml }} />;
}
