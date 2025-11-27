export async function exportSession(sessionId: string, fmt: "json"|"csv"|"md"|"txt"|"pdf") {
  const base = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";
  const url = `${base}/api/chat/sessions/${encodeURIComponent(sessionId)}/export?format=${fmt}`;
  const res = await fetch(url, {
    method: "GET",
  });
  if (!res.ok) throw new Error(`Export failed: ${res.status}`);
  // trigger browser download
  const blob = await res.blob();
  const a = document.createElement("a");
  const ext = fmt === "pdf" ? "pdf" : fmt;
  a.href = URL.createObjectURL(blob);
  a.download = `chat.${ext}`;
  a.click();
  URL.revokeObjectURL(a.href);
}
