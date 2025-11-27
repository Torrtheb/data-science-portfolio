// src/store/chatStore.ts
"use client";

import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";
import { createConversation, listConversations, renameConversation as svRename, deleteConversation as svDelete, ConversationItem } from "@/lib/agentConversations";

// --- Preview helpers (add) ---
// --- Preview helpers ---
const stripMarkdownToText = (s: string) =>
  s
    .replace(/```[\s\S]*?```/g, "")
    .replace(/`[^`]*`/g, "")
    .replace(/!\[[^\]]*\]\([^)]+\)/g, "")
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
    .replace(/(^|\s)[#>*_~\-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();

const shorten = (s: string, max = 160) => (s.length <= max ? s : s.slice(0, max - 1) + "…");

// Accept any role except "tool" for preview (some code may save "assistant")
const isPreviewCandidate = (role: unknown) => {
  if (!role) return false;
  const r = String(role);
  return r.toLowerCase() !== "tool";
};


export type ChatRole = "user" | "ai" | "tool";



export type ChatMsg = {
  role: ChatRole;
  content: string;
  ts: number; // epoch ms
  name?: string; // for tool messages
};

export type Thread = {
  id: string;
  title: string;        // owner-editable
  createdAt: number;
  updatedAt: number;
  lastPreview?: string; // first line of last message
};

type State = {
  hydrated: boolean;
  threads: Record<string, Thread>;
  activeId: string | null;
  byThread: Record<string, ChatMsg[]>;
  drafts: Record<string, string>;
};

type Actions = {
  ensureActive: () => string;
  getMessages: (id?: string) => ChatMsg[];
  getDraft: () => string;
  setDraft: (s: string) => void;

  bootstrapFromServer: () => Promise<void>;
  newThread: (title?: string) => Promise<string>;
  renameThread: (id: string, title: string) => Promise<void>;
  deleteThread: (id: string) => Promise<void>;
  setSessionId: (id: string) => void;

  setMessages: (msgs: ChatMsg[], threadId?: string) => void;
  addMessage: (m: ChatMsg, threadId?: string) => void;
  mutateMessages: (fn: (prev: ChatMsg[]) => ChatMsg[], threadId?: string) => void;
  resetThread: (threadId?: string) => void;

  listThreads: () => Thread[];
  sessionId: string | null;
  getLastPreview: (threadId: string, max?: number) => string;


  // (If you already added server sync helpers, keep them here; not required for this fix.)
};

const STORAGE_NAME = "assistant-chat-v2";
const DEFAULT_TITLE = "New chat";

export const useChatStore = create<State & Actions>()(
  persist(
    (set, get) => ({
      hydrated: false,

      threads: {},
      activeId: null,
      byThread: {},
      drafts: {},

    ensureActive: () => {
    const s = get();

    // 1) If we already have an active id in memory, use it.
    if (s.activeId) return s.activeId;

    // 2) If not hydrated yet, DO NOT create a new id. Try legacy key.
    if (!s.hydrated) {
        try {
        const saved = localStorage.getItem("assistant_thread_id");
        if (saved) return saved;
        } catch {}
        return ""; // caller should wait until hydrated
    }

    // 3) Hydrated and still nothing: DO NOT create a local UUID.
    //    We return empty so the UI can create a server conversation on first send().
    return "";
    },


    setSessionId: (id) => {
    const now = Date.now();
    set((s) => {
        const exists = !!s.threads[id];
        return {
        activeId: id,
        threads: exists
            ? s.threads
            : { ...s.threads, [id]: { id, title: DEFAULT_TITLE, createdAt: now, updatedAt: now } },
        byThread: exists ? s.byThread : { ...s.byThread, [id]: [] },
        drafts: exists ? s.drafts : { ...s.drafts, [id]: "" },
        };
    });
    try { localStorage.setItem("assistant_thread_id", id); } catch {}
    },
    getLastPreview: (threadId, max = 160) => {
    const msgs = get().byThread?.[threadId] ?? [];
    for (let i = msgs.length - 1; i >= 0; i--) {
        const m = msgs[i] as Partial<ChatMsg> | undefined;
        const c = String(m?.content ?? "");
        if (isPreviewCandidate(m?.role) && c.trim()) {
        return shorten(stripMarkdownToText(c), max);
        }
    }
    return "";
    },




    bootstrapFromServer: async () => {
    // Fetch existing conversations from server once (idempotent)
    const items = await listConversations(); // [{ id, title, created_at, updated_at?, last_preview? }]
    const now = Date.now();

    set((s) => {
        const threads = { ...s.threads };
        const byThread = { ...s.byThread };
        const drafts = { ...s.drafts };

        for (const it of items as ConversationItem[]) {
          const created = new Date(it.created_at ?? Date.now()).getTime() || now;
          const updated = new Date(it.updated_at ?? it.created_at ?? Date.now()).getTime() || created;
          const existing = threads[it.id];
          threads[it.id] = {
            id: it.id,
            title: it.title || DEFAULT_TITLE,
            createdAt: existing?.createdAt ?? created,
            updatedAt: updated,
            lastPreview: it.last_preview ?? existing?.lastPreview,
          };
          if (!byThread[it.id]) byThread[it.id] = [];
          if (!(it.id in drafts)) drafts[it.id] = "";
        }

        // If no active, pick the most recent server thread
        let activeId = s.activeId;
        if (!activeId && items.length) {
        activeId = items[0].id; // server returns DESC by updated_at in our handler
        try { localStorage.setItem("assistant_thread_id", activeId); } catch {}
        }

        return { threads, byThread, drafts, activeId: activeId ?? s.activeId };
    });
    },

    newThread: async (title) => {
    // Create on server and use its canonical id
    const conv = await createConversation(title?.trim() || DEFAULT_TITLE);
    const id = conv.id;
    const now = Date.now();

    set((s) => ({
        activeId: id,
        threads: {
        ...s.threads,
        [id]: { id, title: title?.trim() || DEFAULT_TITLE, createdAt: now, updatedAt: now },
        },
        byThread: { ...s.byThread, [id]: [] },
        drafts: { ...s.drafts, [id]: "" },
    }));

    try { localStorage.setItem("assistant_thread_id", id); } catch {}
    return id;
    },

    renameThread: async (id, title) => {
    const name = title.trim();
    if (!name) return;
    await svRename(id, name);
    set((s) => {
        const th = s.threads[id]; if (!th) return {};
        return {
        threads: { ...s.threads, [id]: { ...th, title: name, updatedAt: Date.now() } },
        };
    });
    },

    deleteThread: async (id) => {
    await svDelete(id);
    set((s) => {
        const { [id]: _a, ...restThreads } = s.threads;
        const { [id]: _b, ...restBy } = s.byThread;
        const { [id]: _c, ...restDrafts } = s.drafts;

        let activeId = s.activeId === id ? null : s.activeId;
        if (!activeId) {
        const remaining = Object.values(restThreads).sort((a, b) => b.updatedAt - a.updatedAt);
        activeId = remaining[0]?.id ?? null;
        }
        if (activeId) {
        try { localStorage.setItem("assistant_thread_id", activeId); } catch {}
        }
        return { threads: restThreads, byThread: restBy, drafts: restDrafts, activeId };
    });
    },


      getMessages: (id) => {
        const tid = id ?? get().activeId ?? "";
        return get().byThread[tid] ?? [];
      },

      getDraft: () => {
        const tid = get().activeId ?? "";
        return get().drafts[tid] ?? "";
      },

      setDraft: (sval) => {
        const tid = get().activeId ?? "";
        set((s) => ({ drafts: { ...s.drafts, [tid]: sval } }));
      },

    setMessages: (msgs, threadId) => {
    const tid = threadId ?? get().activeId ?? "";
    set((s) => {
        // compute preview from the last meaningful (non-tool) message
        let preview = "";
        for (let i = msgs.length - 1; i >= 0; i--) {
        const m = msgs[i] as Partial<ChatMsg> | undefined;
        const c = String(m?.content ?? "");
        if (isPreviewCandidate(m?.role) && c.trim()) {
            preview = shorten(stripMarkdownToText(c), 160);
            break;
        }
        }

        const th =
        s.threads[tid] ?? { id: tid, title: DEFAULT_TITLE, createdAt: Date.now(), updatedAt: Date.now() };

        return {
        threads: { ...s.threads, [tid]: { ...th, lastPreview: preview, updatedAt: Date.now() } },
        byThread: { ...s.byThread, [tid]: msgs },
        };
    });
    },


      addMessage: (m, threadId) => {
        const tid = threadId ?? get().activeId ?? "";
        const prev = get().byThread[tid] ?? [];
        get().setMessages([...prev, m], tid);
      },

      mutateMessages: (fn, threadId) => {
        const tid = threadId ?? get().activeId ?? "";
        const prev = get().byThread[tid] ?? [];
        get().setMessages(fn(prev), tid);
      },

      resetThread: (threadId) => {
        const tid = threadId ?? get().activeId ?? "";
        set((s) => ({
          byThread: { ...s.byThread, [tid]: [] },
          threads: {
            ...s.threads,
            [tid]: {
              ...(s.threads[tid] ?? { id: tid, title: DEFAULT_TITLE, createdAt: Date.now() }),
              updatedAt: Date.now(),
              lastPreview: "",
            },
          },
        }));
      },

      listThreads: () => {
        const dict = get().threads;
        return Object.values(dict).sort((a, b) => b.updatedAt - a.updatedAt);
      },

      get sessionId() {
        return get().activeId;
      },
    }),
    {
      name: STORAGE_NAME,
      version: 1,
      storage: createJSONStorage(() => localStorage),
    onRehydrateStorage: () => {
    return (_state, _error) => {
        try {
        // Mark hydrated so UI can wait
        useChatStore.setState({ hydrated: true });

        // Restore active pointer if missing
        const saved = localStorage.getItem("assistant_thread_id");
        if (saved && !useChatStore.getState().activeId) {
            useChatStore.getState().setSessionId(saved);
        }

        // ⬇️ Backfill lastPreview for threads missing it (from persisted byThread)
        const st = useChatStore.getState();
        const patched: Record<string, Thread> = {};
        for (const [id, th] of Object.entries(st.threads)) {
            if (!th.lastPreview) {
            const preview = st.getLastPreview(id, 160);
            if (preview) {
                patched[id] = { ...th, lastPreview: preview };
            }
            }
        }
        if (Object.keys(patched).length) {
            useChatStore.setState((s) => ({ threads: { ...s.threads, ...patched } }));
        }
        } catch {}
    };
    },



      partialize: (s) => ({
        threads: s.threads,
        activeId: s.activeId,
        byThread: s.byThread,
        drafts: s.drafts,
      }),
    }
  )
);

export const useActiveSessionId = () => useChatStore((s) => s.activeId);
