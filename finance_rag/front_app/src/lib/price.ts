// src/lib/price.ts

import { buildUrl } from './backend';

const TIMEOUT_MS = 20_000;

async function fetchWithTimeout(input: RequestInfo | URL, init: RequestInit = {}) {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), TIMEOUT_MS);
  try {
    return await fetch(input, { cache: 'no-store', ...init, signal: controller.signal });
  } finally {
    clearTimeout(id);
  }
}

function isLikelySymbolError(msg: string | undefined) {
  if (!msg) return false;
  const m = msg.toLowerCase();
  return m.includes('symbol') || m.includes('figi') || m.includes('invalid');
}

async function readBodySafe(res: Response): Promise<any> {
  try {
    return await res.json();
  } catch {
    try {
      const t = await res.text();
      try { return JSON.parse(t); } catch { return { detail: t || `HTTP ${res.status}` }; }
    } catch {
      return { detail: `HTTP ${res.status}` };
    }
  }
}

/**
 * Fetch a live/last price for a symbol or query.
 * Returns: {symbol, name, price, currency, change, changePercent, aliases?: string[]}
 */
export async function fetchPrice(q: string) {
  const url = buildUrl(`/api/price?q=${encodeURIComponent(q)}`);
  const res = await fetchWithTimeout(url);
  if (!res.ok) {
    const body = await readBodySafe(res);
    throw new Error(body?.detail?.message ?? body?.detail ?? body?.error ?? `HTTP ${res.status}`);
  }
  return res.json();
}

/**
 * Fetch candles from your TD route, keeping your current API shape.
 * Returns (TD): { s:'ok', t:number[], c:number[], o:number[], h:number[], l:number[], symbol? }
 */
export async function fetchCandles(
  symbol: string,
  opts: { interval?: '1day' | '1week' | '1month'; outputsize?: number } = {}
) {
  return fetchCandlesTD(symbol, opts);
}

/**
 * Same as fetchCandles, with slightly different defaults that you already used.
 * - Strips dotted suffixes (AAPL stays AAPL, AC.TO -> AC for TD)
 * - Retries once without hints if TD reports "symbol/figi invalid"
 * Returns (TD): { s:'ok', t:number[], c:number[], o,h,l, symbol }
 */
export async function fetchCandlesTD(
  symbol: string,
  opts: { interval?: '1day' | '1week' | '1month'; outputsize?: number } = {}
) {
  const interval: '1day' | '1week' | '1month' = (opts.interval ?? '1day');
  const outputsize: number = (opts.outputsize ?? 365);

  const symRaw: string = (symbol ?? '').trim().toUpperCase();
const dot = symRaw.indexOf('.');
const base: string = dot === -1 ? symRaw : symRaw.slice(0, dot);

  const url1 = buildUrl(
    `/api/td/candles?symbol=${encodeURIComponent(String(base))}&interval=${interval}&outputsize=${outputsize}`
  );

  let res = await fetchWithTimeout(url1);
  let body = await readBodySafe(res);

  const tdError =
    (res.ok && body && typeof body === 'object' && body.status === 'error') ||
    !res.ok;

  if (tdError) {
    const msg: string | undefined = body?.message || body?.detail || body?.error;
    if (isLikelySymbolError(msg)) {
      const url2 = buildUrl(
        `/api/td/candles?symbol=${encodeURIComponent(String(base))}&interval=${interval}&outputsize=${outputsize}`
      );
      res = await fetchWithTimeout(url2);
      body = await readBodySafe(res);
    }
  }

  if (!res.ok || (body && body.status === 'error')) {
    const msg = body?.message || body?.detail || `Twelve Data fetch failed (${res.status})`;
    throw new Error(msg);
  }

  return body; // -> { s:'ok', t:number[], c:number[], o,h,l, symbol }
}

/* -----------------------------------------------------------
   Safer wrapper that falls back to /api/market/candles
   if /api/td/candles fails (e.g., method/permission issues).
----------------------------------------------------------- */
export async function fetchCandlesSafe(
  symbol: string,
  opts: { interval?: '1day' | '1week' | '1month'; outputsize?: number } = {}
) {
  try {
    return await fetchCandlesTD(symbol, opts); 
  } catch (err) {
    const interval: '1day' | '1week' | '1month' = (opts.interval ?? '1day');
    const resolution = interval === '1week' ? 'W' : interval === '1month' ? 'M' : 'D';
    const url = buildUrl(
      `/api/market/candles?symbol=${encodeURIComponent(symbol)}&resolution=${resolution}`
    );
    const r = await fetchWithTimeout(url);
    const body = await readBodySafe(r);
    if (!r.ok) {
      throw new Error(body?.detail || body?.error || `HTTP ${r.status}`);
    }
    return body; 
  }
}
