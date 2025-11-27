// src/app/owner/assistant/OutboxListClient.tsx
"use client";
import { useEffect, useState } from "react";
import { listOutboxDrafts, OutboxDraft } from "@/lib/api";

export default function OutboxListClient() {
  const [items, setItems] = useState<OutboxDraft[]>([]);
  const [status, setStatus] = useState<OutboxDraft["status"] | "">("");

  useEffect(() => {
    listOutboxDrafts({ status: status || undefined, limit: 50 }).then(setItems).catch(console.error);
  }, [status]);

  return (
    <div className="space-y-3">
      <div className="flex gap-2 items-center">
        <span className="text-sm">Filter:</span>
        <select
          className="border rounded p-1 text-sm"
          value={status}
          onChange={(e)=> setStatus(e.target.value as OutboxDraft["status"] | "")}
        >
          <option value="">All</option>
          <option value="pending">Pending</option>
          <option value="approved">Approved</option>
          <option value="sent">Sent</option>
          <option value="rejected">Rejected</option>
        </select>
      </div>

      <div className="grid gap-2">
        {items.map(it => (
          <div key={it.id} className="border rounded p-3 bg-white">
            <div className="flex justify-between">
              <div className="font-medium">{it.subject}</div>
              <span className="text-xs px-2 py-1 rounded bg-gray-100">{it.status}</span>
            </div>
            <div className="text-sm text-gray-600">
              {it.recipients?.length
                ? it.recipients.map(r => r.email).join(", ")
                : (it.to || "")}
            </div>
            <div className="text-sm mt-1 line-clamp-2">{it.text}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
