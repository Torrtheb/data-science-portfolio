// src/components/ChatThreadSwitcher.tsx
"use client";
import { useEffect, useMemo, useRef, useState } from "react";
import { useChatStore } from "@/store/chatStore";

export default function ChatThreadSwitcher({
  onSwitched,
  onNewChat,
}: {
  onSwitched?: () => void;
  onNewChat?: () => void;
}) {
  const listThreads  = useChatStore((s) => s.listThreads);
  const sessionId    = useChatStore((s) => s.activeId);
  const setSessionId = useChatStore((s) => s.setSessionId);
  const newThread    = useChatStore((s) => s.newThread);
  const renameThread = useChatStore((s) => s.renameThread);
  const deleteThread = useChatStore((s) => s.deleteThread);

  const threads = listThreads();
  const [editing, setEditing] = useState<string | null>(null);
  const [draftTitle, setDraftTitle] = useState("");
  const inputRef = useRef<HTMLInputElement | null>(null);

  const activeTitle = useMemo(() => {
    const t = threads.find((t) => t.id === sessionId);
    return t?.title ?? "New chat";
  }, [threads, sessionId]);

  useEffect(() => {
    if (editing && inputRef.current) inputRef.current.focus();
  }, [editing]);

  const startRename = () => {
    if (!sessionId) return;
    setDraftTitle(activeTitle);
    setEditing(sessionId);
  };

  const saveRename = async () => {
    if (!sessionId) return;
    const name = draftTitle.trim();
    if (name && name !== activeTitle) {
      await renameThread(sessionId, name);
    }
    setEditing(null);
  };

  const cancelRename = () => setEditing(null);

  return (
    // Let the bar wrap if space is tight, and keep buttons from growing
    <div className="flex flex-wrap items-center gap-2">
      {/* Thread selector: cap width so it doesn't push buttons off-screen */}
      <select
        className="text-sm border rounded px-2 py-1 bg-white w-full sm:w-auto max-w-[420px]"
        value={sessionId ?? ""}
        onChange={(e) => {
          setSessionId(e.target.value);
          onSwitched?.();
        }}
      >
        {threads.length === 0 ? (
          <option value="">No chats</option>
        ) : (
          threads.map((t) => (
            <option key={t.id} value={t.id}>
              {t.title || "(untitled)"}
            </option>
          ))
        )}
      </select>

      {/* Rename */}
      {editing === sessionId ? (
        <input
          ref={inputRef}
          className="text-sm border rounded px-2 py-1 w-full sm:w-auto"
          value={draftTitle}
          onChange={(e) => setDraftTitle(e.target.value)}
          onBlur={saveRename}
          onKeyDown={async (e) => {
            if (e.key === "Enter") { e.preventDefault(); await saveRename(); }
            if (e.key === "Escape") { e.preventDefault(); cancelRename(); }
          }}
          placeholder={activeTitle}
        />
      ) : (
        <button
          className="shrink-0 text-sm text-gray-600 hover:text-gray-900"
          onClick={startRename}
          disabled={!sessionId}
          title="Rename chat"
        >
          Rename
        </button>
      )}

      {/* New chat */}
      <button
        className="shrink-0 text-sm px-2 py-1 border rounded"
        onClick={async () => {
          const id = await newThread("New chat");
          setSessionId(id);
          onNewChat?.();
        }}
        title="Start a new chat"
      >
        + New
      </button>

      {/* Delete active */}
      <button
        className="shrink-0 text-sm text-red-600 hover:text-red-700"
        onClick={async () => {
          if (!sessionId) return;
          const ok = confirm("Delete this chat?");
          if (ok) await deleteThread(sessionId);
        }}
        disabled={!sessionId}
        title="Delete this chat"
      >
        Delete
      </button>
    </div>
  );
}
