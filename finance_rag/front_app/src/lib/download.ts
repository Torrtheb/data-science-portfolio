// src/lib/download.ts
export type ExportFormat = "csv" | "json" | "md" | "txt" | "pdf";

import { buildUrl } from "./backend";

// Keep these keys aligned with ChatPro/session management
const SESSION_KEY = "finassist.sessionId.v1";
const SESSION_TOKEN_KEY = "finassist.sessionToken.v1";

function filenameFromContentDisposition(cd?: string | null, fallback = "chat_export"): string {
  if (!cd) return fallback;
  // RFC 6266 – look for filename*= or filename=
  // Example: attachment; filename="abc.csv"
  const matchStar = cd.match(/filename\*\s*=\s*[^']+'[^']*'([^;]+)/i);
  if (matchStar?.[1]) return decodeURIComponent(matchStar[1]).replace(/["]/g, "");
  const match = cd.match(/filename\s*=\s*"([^"]+)"|filename\s*=\s*([^;]+)/i);
  if (match?.[1]) return match[1];
  if (match?.[2]) return match[2];
  return fallback;
}

export async function downloadSessionExport(
  sessionId: string,
  format: ExportFormat,
  opts?: { apiKey?: string; backendBase?: string }
): Promise<void> {
  const endpoint =
    format === "pdf"
      ? `/chat/${encodeURIComponent(sessionId)}/export.pdf`
      : format === "csv"
      ? `/chat/${encodeURIComponent(sessionId)}/export.csv`
      : format === "md"
      ? `/chat/${encodeURIComponent(sessionId)}/export.md`
      : format === "txt"
      ? `/chat/${encodeURIComponent(sessionId)}/export.txt`
      : `/chat/${encodeURIComponent(sessionId)}/export.json`;

  const url = buildUrl(endpoint);

  const headers: Record<string, string> = {};
  if (opts?.apiKey) headers["x-api-key"] = opts.apiKey;
  try {
    const token = localStorage.getItem(SESSION_TOKEN_KEY) || "";
    const sid = localStorage.getItem(SESSION_KEY) || "";
    if (token) headers["Authorization"] = `Bearer ${token}`;
    else if (sid) headers["x-session-id"] = sid;
  } catch {}

  const res = await fetch(url, {
    method: "GET",
    headers,
  });

  if (!res.ok) {
    // Try to surface server error details if JSON
    let reason = `${res.status} ${res.statusText}`;
    try {
      const j = await res.clone().json();
      reason = j?.detail || JSON.stringify(j);
    } catch {
      try {
        reason = await res.text();
      } catch {}
    }
    throw new Error(`Export failed: ${reason}`);
  }

  const blob = await res.blob();
  const cd = res.headers.get("Content-Disposition");
  const fallbackName = `conversation_${sessionId}.${format}`;
  const filename = filenameFromContentDisposition(cd, fallbackName);

  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  // Cleanup
  setTimeout(() => {
    URL.revokeObjectURL(link.href);
    link.remove();
  }, 0);
}
