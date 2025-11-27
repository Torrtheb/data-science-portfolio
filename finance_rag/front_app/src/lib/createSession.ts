// src/lib/createSession.ts (Next.js version)
export async function createSession() {
  const base = process.env.NEXT_PUBLIC_API_BASE;

  if (!base) {
    throw new Error(
      "NEXT_PUBLIC_API_BASE is not set. Add it to your .env.local (e.g., http://127.0.0.1:8000) and restart dev server."
    );
  }

  const res = await fetch(`${base}/api/chat/sessions`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
    },
    body: JSON.stringify({ title: "New session" }),
  });

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`createSession failed: ${res.status} ${res.statusText} – ${text}`);
  }
  return res.json();
}
