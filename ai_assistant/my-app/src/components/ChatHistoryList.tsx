// src/components/ChatHistoryList.tsx
"use client";
import { useMemo, useState } from "react";
import { useChatStore } from "@/store/chatStore";

type Props = { maxVisible?: number };

function _toSnippet(s?: string, max = 160): string {
  if (!s) return "";
  const trimmed = s.replace(/\s+/g, " ").trim();
  if (trimmed.length <= max) return trimmed;
  return trimmed.slice(0, max - 1) + "…";
}

export default function ChatHistoryList({ maxVisible = 5 }: Props) {
  // ====== store selectors (hooks) — always called, in the same order ======
  // Subscribe to base state and derive list via useMemo to keep selector pure
  const threadsDict     = useChatStore((s) => s.threads);
  const sessionId      = useChatStore((s) => s.activeId);
  const setSessionId   = useChatStore((s) => s.setSessionId);
  const newThread      = useChatStore((s) => s.newThread);
  const renameThread   = useChatStore((s) => s.renameThread);
  const deleteThread   = useChatStore((s) => s.deleteThread);
  const getLastPreview = useChatStore((s) => s.getLastPreview);
  const hydrated       = useChatStore((s) => s.hydrated);

  // ====== local state (hooks) — always called, in the same order ======
  const [from, setFrom] = useState<string>("");
  const [to, setTo] = useState<string>("");
  const [showAll, setShowAll] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [draftTitle, setDraftTitle] = useState("");

  // ====== derived data (hooks) — always called, in the same order ======
  const all = useMemo(() => Object.values(threadsDict).sort((a, b) => b.updatedAt - a.updatedAt), [threadsDict]);

  const filtered = useMemo(() => {
    const fromMs = from ? new Date(from + "T00:00:00").getTime() : -Infinity;
    const toMs = to ? new Date(to + "T23:59:59.999").getTime() : Infinity;
    return all.filter((t) => t.updatedAt >= fromMs && t.updatedAt <= toMs);
  }, [all, from, to]);

  const visible = showAll ? filtered : filtered.slice(0, maxVisible);

  // flags for UI branching (no early returns)
  const isLoading = !hydrated;
  const isEmpty = hydrated && all.length === 0;

  async function onNew() {
    const id = await newThread("New chat");
    setSessionId(id);
  }

  // ====== single return; branch inside JSX only ======
  return (
    <div className="space-y-3">
      {/* Loading state */}
      {isLoading && (
        <div className="animate-pulse text-sm text-gray-500 border rounded p-3">
          Loading conversations…
        </div>
      )}

      {/* Empty state (only when hydrated) */}
      {!isLoading && isEmpty && (
        <>
          <div className="flex items-center justify-between">
            <button className="text-sm border rounded px-2 py-1" onClick={onNew}>+ New</button>
          </div>
          <div className="text-sm text-gray-500 border rounded p-3">No previous chats yet.</div>
        </>
      )}

      {/* Controls + List + Footer (only when hydrated and not empty) */}
      {!isLoading && !isEmpty && (
        <>
          {/* Controls */}
          <div className="flex items-center justify-between">
            <button className="text-sm border rounded px-2 py-1" onClick={onNew}>+ New</button>
            <div className="flex items-end gap-2">
              <div>
                <label className="block text-xs text-gray-600 mb-1">From</label>
                <input
                  type="date"
                  value={from}
                  onChange={(e) => setFrom(e.target.value)}
                  className="text-sm border rounded px-2 py-1"
                />
              </div>
              <div>
                <label className="block text-xs text-gray-600 mb-1">To</label>
                <input
                  type="date"
                  value={to}
                  onChange={(e) => setTo(e.target.value)}
                  className="text-sm border rounded px-2 py-1"
                />
              </div>
              <button
                className="text-sm border rounded px-2 py-1"
                onClick={() => { setFrom(""); setTo(""); }}
              >
                Clear
              </button>
            </div>
          </div>

          {/* List */}
          <div className="space-y-2">
            {visible.map((t) => {
              const isActive = t.id === sessionId;
              const isEditing = editingId === t.id;

              return (
                <div
                  key={t.id}
                  className={`border rounded p-2 hover:bg-gray-50 transition ${isActive ? "border-black" : "border-gray-200"}`}
                >
                  <div className="flex items-start gap-2">
                    {/* Text/content column */}
                    <button
                      onClick={() => setSessionId(t.id)}
                      className="min-w-0 flex-1 text-left"
                      title="Open chat"
                    >
                      {!isEditing ? (
                        <>
                          <div className="font-medium leading-5 truncate">
                            {t.title || "(untitled)"}
                          </div>
                          <div className="text-[11px] text-gray-500">
                            {new Date(t.updatedAt).toLocaleString()}
                          </div>

                          {/* Preview: one-line snippet (first few words) under date */}
                          {(() => {
                            const raw = (t.lastPreview?.trim() || getLastPreview(t.id, 80) || "");
                            const firstLine = raw.split("\n")[0]?.trim() || "";
                            return firstLine ? (
                              <div className="text-xs text-gray-600 mt-0.5 line-clamp-1">{firstLine}</div>
                            ) : null;
                          })()}
                        </>
                      ) : (
                        <input
                          className="text-sm border rounded px-2 py-1 w-full"
                          value={draftTitle}
                          onChange={(e) => setDraftTitle(e.target.value)}
                          onBlur={async () => {
                            const title = draftTitle.trim();
                            if (title && title !== t.title) {
                              await renameThread(t.id, title);
                            }
                            setEditingId(null);
                          }}
                          onKeyDown={async (e) => {
                            if (e.key === "Enter") {
                              e.preventDefault();
                              const title = draftTitle.trim();
                              if (title && title !== t.title) {
                                await renameThread(t.id, title);
                              }
                              setEditingId(null);
                            }
                            if (e.key === "Escape") {
                              e.preventDefault();
                              setEditingId(null);
                            }
                          }}
                        />
                      )}
                    </button>

                    {/* Actions column */}
                    <div className="shrink-0 flex flex-col items-end gap-1 self-stretch">
                      {!isEditing ? (
                        <button
                          className="text-xs text-gray-600 hover:text-gray-900 px-2 py-1"
                          onClick={(e) => {
                            e.stopPropagation();
                            setDraftTitle(t.title || "");
                            setEditingId(t.id);
                          }}
                          title="Rename"
                        >
                          Rename
                        </button>
                      ) : null}

                      <button
                        className="text-xs text-red-600 hover:text-red-700 px-2 py-1"
                        onClick={async (e) => {
                          e.stopPropagation();
                          const ok = confirm("Delete this chat?");
                          if (ok) await deleteThread(t.id);
                        }}
                        title="Delete"
                      >
                        Delete
                      </button>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Footer */}
          <div className="flex items-center justify-between text-xs text-gray-600">
            <div>
              Showing {visible.length} of {filtered.length} filtered
              {filtered.length !== all.length ? ` (total ${all.length})` : ""}.
            </div>
            {filtered.length > maxVisible && (
              <button className="underline" onClick={() => setShowAll((v) => !v)}>
                {showAll ? "Show less" : `Show last ${maxVisible}`}
              </button>
            )}
          </div>
        </>
      )}
    </div>
  );
}
