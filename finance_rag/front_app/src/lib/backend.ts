// src/lib/backend.ts
const stripTrailingSlash = (v: string) => v.replace(/\/$/, '');
const cleanEnvUrl = (v?: string | null) => {
  const s = (v ?? '').trim();
  // Treat these strings as unset
  if (!s || s === 'undefined' || s === 'null' || s === '/') return '';
  return stripTrailingSlash(s);
};

const BASE =
  cleanEnvUrl(process.env.NEXT_PUBLIC_API_BASE_URL) ||
  cleanEnvUrl(process.env.NEXT_PUBLIC_BACKEND_URL) ||
  cleanEnvUrl(process.env.NEXT_PUBLIC_API_BASE) ||
  '';

export const BACKEND_ORIGIN =
  cleanEnvUrl(process.env.NEXT_PUBLIC_BACKEND_ORIGIN) ||
  `http://${typeof window !== 'undefined' ? window.location.hostname : 'localhost'}:8080`;

const defaultHeaders: Record<string, string> = {
  'content-type': 'application/json',
};

export function buildUrl(path: string): string {
  if (/^https?:\/\//i.test(path)) return path;
  const p = path.startsWith('/api') ? path : `/api${path.startsWith('/') ? path : '/' + path}`;
  if (!BASE) return p;
  return `${BASE}${p}`;
}

function withTimeout(
  sourceSignal?: AbortSignal | null,
  ms = 30000
): { signal?: AbortSignal; clear: () => void } {
  const ctrl = new AbortController();
  const id = setTimeout(() => ctrl.abort('timeout'), ms);

  if (sourceSignal) {
    const onAbort = () => ctrl.abort((sourceSignal as any).reason);
    sourceSignal.addEventListener('abort', onAbort, { once: true });
  }

  return { signal: ctrl.signal, clear: () => clearTimeout(id) };
}

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.text();
    if (res.status === 429) throw new Error(`Rate limit hit. Try again in a few seconds.\n${body}`);
    throw new Error(`HTTP ${res.status}: ${body}`);
  }
  return res.json() as Promise<T>;
}

export async function postJSON<T>(path: string, body?: any, init?: RequestInit): Promise<T> {
  const { signal, clear } = withTimeout(init?.signal);
  try {
    const reqInit: RequestInit = {
      method: 'POST',
      headers: { ...defaultHeaders, ...(init?.headers ?? {}) },
      cache: 'no-store',
      ...(init ?? {}),
    };

    if (signal) reqInit.signal = signal;
    if (body !== undefined) reqInit.body = JSON.stringify(body); // do NOT set if undefined

    const res = await fetch(buildUrl(path), reqInit);
    return handle<T>(res);
  } finally {
    clear();
  }
}

export async function getJSON<T>(path: string, params?: Record<string, any>, init?: RequestInit): Promise<T> {
  const usp = new URLSearchParams();
  if (params) for (const [k, v] of Object.entries(params)) if (v !== undefined && v !== null) usp.set(k, String(v));
  const q = usp.toString();

  const { signal, clear } = withTimeout(init?.signal);
  try {
    const reqInit: RequestInit = {
      cache: 'no-store',
      headers: { ...defaultHeaders, ...(init?.headers ?? {}) },
      ...(init ?? {}),
    };
    if (signal) reqInit.signal = signal;

    const res = await fetch(buildUrl(q ? `${path}?${q}` : path), reqInit);
    return handle<T>(res);
  } finally {
    clear();
  }
}

// Convenience specific to chat
export async function createSession(title?: string) {
  return postJSON<{ id: string; title?: string; created_at: string; updated_at: string; messages: any[] }>(
    '/chat/sessions',
    { title }
  );
}
export async function getSession(sessionId: string) {
  return getJSON(`/chat/sessions/${sessionId}`);
}
export async function appendMessage(
  sessionId: string,
  msg: { role: 'user' | 'assistant' | 'system' | 'tool'; content: string; tool_calls?: any[] }
) {
  return postJSON(`/chat/sessions/${sessionId}/messages`, msg);
}
