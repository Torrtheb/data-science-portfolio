// src/lib/agentConversations.ts
import { apiFetch } from "@/lib/api";

const BASE = "/api/back/api/agent";

export type ConversationItem = {
  id: string;
  title: string;
  created_at: string;
  updated_at?: string;
  last_preview?: string;
};

export async function listConversations(): Promise<ConversationItem[]> {
  const r = await apiFetch(`${BASE}/conversations`, { cache: "no-store" });
  if (!r.ok) throw new Error(await r.text());
  const j = await r.json();
  return j.items ?? [];
}

export async function createConversation(title?: string): Promise<{ id: string; title?: string }> {
  const r = await apiFetch(`${BASE}/conversations${title ? `?title=${encodeURIComponent(title)}` : ""}`, {
    method: "POST",
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function renameConversation(id: string, title: string) {
  const r = await apiFetch(`${BASE}/conversations/${encodeURIComponent(id)}?title=${encodeURIComponent(title)}`, {
    method: "PATCH",
  });
  if (!r.ok) throw new Error(await r.text());
}

export async function deleteConversation(id: string) {
  const r = await apiFetch(`${BASE}/conversations/${encodeURIComponent(id)}`, { method: "DELETE" });
  if (!r.ok) throw new Error(await r.text());
}
