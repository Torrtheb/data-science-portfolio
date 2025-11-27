// src/lib/csrf.ts
// Lightweight CSRF helper for the frontend: ensures a stable token cookie
// and attaches it to mutating fetch requests as an `X-CSRF-Token` header.

export function getOrInitCsrfToken(): string {
  if (typeof document === "undefined") return "";
  const name = "x-csrf-token";
  const m = document.cookie.match(new RegExp("(?:^|;\\s*)" + name + "=([^;]+)"));
  if (m && m[1]) return decodeURIComponent(m[1]);
  const token = crypto.getRandomValues(new Uint8Array(16)).reduce((s, b) => s + b.toString(16).padStart(2, "0"), "");
  // Lax so same-site navigations carry it; max-age 1 year
  document.cookie = `${name}=${encodeURIComponent(token)}; Path=/; Max-Age=31536000; SameSite=Lax`;
  return token;
}

export type ApiFetchInit = RequestInit & { csrf?: boolean };

// Wrap fetch to auto-attach CSRF header for unsafe methods when desired
export async function apiFetch(input: RequestInfo | URL, init: ApiFetchInit = {}): Promise<Response> {
  const method = (init.method || "GET").toUpperCase();
  const isUnsafe = method !== "GET" && method !== "HEAD" && method !== "OPTIONS";
  const needsCsrf = init.csrf === true || isUnsafe;
  const headers = new Headers(init.headers || {});
  if (needsCsrf && typeof window !== "undefined") {
    const token = getOrInitCsrfToken();
    headers.set("X-CSRF-Token", token);
  }
  return fetch(input, { ...init, headers });
}

