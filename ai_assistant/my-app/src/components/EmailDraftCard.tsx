"use client";

import React from "react";

type DraftPayload = {
  draft_id: string;
  to: string;
  to_name?: string;
  subject: string;
  text: string;
};

export function EmailDraftCard({
  payload,
  onSendViaChat,    // Option A: send a chat message "SEND_EMAIL: {...}"
  onSendViaApi,     // Option B: POST /api/outbox/:id/send
  onCancel,
}: {
  payload: DraftPayload;
  onSendViaChat?: (jsonMessage: string) => void;
  onSendViaApi?: (draftId: string, body: Omit<DraftPayload, "draft_id">) => Promise<void>;
  onCancel: () => void;
}) {
  const [to, setTo] = React.useState(payload.to);
  const [toName, setToName] = React.useState(payload.to_name ?? "");
  const [subject, setSubject] = React.useState(payload.subject);
  const [text, setText] = React.useState(payload.text);
  const [sending, setSending] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const friendlyEmailError = (raw: string) => {
    const low = (raw || "").toLowerCase();
    if (low.includes("dev") && low.includes("email")) {
      return "Email sending is disabled in this dev environment. The draft stays in Outbox—enable SMTP/production settings to send.";
    }
    if (low.includes("sorry, email sending") || low.includes("5.7.0")) {
      return "Email sending is disabled by the SMTP provider for this account. Configure SMTP or send manually.";
    }
    const first = (raw || "").split("\n")[0].trim();
    if (first.length > 200) return first.slice(0, 200) + "…";
    return first || "Failed to send.";
  };

  async function handleSend() {
    setError(null);
    setSending(true);
    try {
      if (onSendViaApi) {
        // Option B: call your backend REST route
        await onSendViaApi(payload.draft_id, { to, to_name: toName, subject, text });
      } else if (onSendViaChat) {
        // Option A: route back through the graph with a special chat message
        const overrides = {
          draft_id: payload.draft_id,
          approve: true,
          to,
          to_name: toName,
          subject,
          text,
        };
        onSendViaChat(`SEND_EMAIL: ${JSON.stringify(overrides)}`);
      }
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Failed to send.";
      setError(friendlyEmailError(msg));
    } finally {
      setSending(false);
    }
  }

  return (
    <div className="rounded-xl border p-4 space-y-3 bg-white shadow-sm">
      <div className="text-sm font-medium">Email draft</div>

      <div className="grid gap-2">
        <label className="text-xs text-gray-500">To</label>
        <input className="border rounded p-2" value={to} onChange={e=>setTo(e.target.value)} />
      </div>

      <div className="grid gap-2">
        <label className="text-xs text-gray-500">To name (optional)</label>
        <input className="border rounded p-2" value={toName} onChange={e=>setToName(e.target.value)} />
      </div>

      <div className="grid gap-2">
        <label className="text-xs text-gray-500">Subject</label>
        <input className="border rounded p-2" value={subject} onChange={e=>setSubject(e.target.value)} />
      </div>

      <div className="grid gap-2">
        <label className="text-xs text-gray-500">Body</label>
        <textarea className="border rounded p-2 h-40" value={text} onChange={e=>setText(e.target.value)} />
      </div>

      {error && <div className="text-sm text-red-600">{error}</div>}

      <div className="flex gap-2">
        <button
          className="rounded-lg px-4 py-2 bg-black text-white disabled:opacity-50"
          disabled={sending}
          onClick={handleSend}
        >
          {sending ? "Sending…" : "Send"}
        </button>
        <button className="rounded-lg px-4 py-2 border" onClick={onCancel}>
          Cancel
        </button>
      </div>
    </div>
  );
}
