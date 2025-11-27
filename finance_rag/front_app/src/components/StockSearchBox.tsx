// src/components/StockSearchBox.tsx
'use client';

import { useRef, useState } from 'react';
import { buildUrl } from '@/lib/backend';

export default function StockSearchBox() {
  const [q, setQ] = useState('');
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  // simple 300ms debounce for Enter/click
  const timerRef = useRef<number | null>(null);
  function scheduleRun() {
    if (timerRef.current) window.clearTimeout(timerRef.current);
    timerRef.current = window.setTimeout(() => {
      timerRef.current = null;
      void run();
    }, 300);
  }

  async function run() {
    const text = q.trim();
    if (!text || loading) return;

    // analytics: submit attempt
    try {
      window.dispatchEvent(
        new CustomEvent('analytics', { detail: { event: 'stock_search_submit', q: text } })
      );
    } catch {
      // no-op if analytics listener doesn't exist
    }

    setErr(null);
    setLoading(true);
    try {
      // Try /api/price first, then /api/quote as a fallback
      const endpoints = [
        buildUrl(`/api/price?q=${encodeURIComponent(text)}`),
        buildUrl(`/api/quote?q=${encodeURIComponent(text)}`),
      ];

      let data: any | null = null;
      for (const url of endpoints) {
        const r = await fetch(url, { cache: 'no-store' });
        if (!r.ok) continue;
        const js = await r.json().catch(() => null);
        if (js && (typeof js.price === 'number' || typeof js.c === 'number')) {
          data = {
            symbol: (js.symbol || js.ticker || text).toUpperCase(),
            price: typeof js.price === 'number' ? js.price : (typeof js.c === 'number' ? js.c : undefined),
            currency: js.currency || js.cur || js.ccy || '',
            change: js.change ?? js.d ?? undefined,
            changePercent: js.changePercent ?? js.dp ?? js.change_pct ?? undefined,
            aliases: Array.isArray(js.aliases) ? js.aliases : undefined, 
            ...js,
          };

          break;
        }
      }

      if (!data) throw new Error('No live price found.');

      // analytics: success
      try {
        window.dispatchEvent(
          new CustomEvent('analytics', { detail: { event: 'stock_search_success', symbol: data.symbol, q: text } })
        );
      } catch {}

      // Broadcast to the right sidebar results panel (and anyone else)
      window.dispatchEvent(
        new CustomEvent('finassist:add-tool', { detail: { tool: 'live_price', result: data } })
      );

      // Optional: notify other listeners something was selected
      window.dispatchEvent(new CustomEvent('stock:selected', { detail: data }));
    } catch (e: any) {
      setErr('Could not fetch price. Ticker may be outside of Finnhub plan, or invalid.');
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="rounded-2xl border bg-white shadow-sm p-4">
      <h2 className="text-sm font-semibold mb-2">Stock Prices Search</h2>
      <div className="flex items-center gap-2">
        <input
          value={q}
          onChange={(e) => { setQ(e.target.value); setErr(null); }}
          onKeyDown={(e) => { if (e.key === 'Enter') scheduleRun(); }}
          placeholder="Type a symbol or company… e.g., AAPL, MSFT, Tesla"
          className="text-sm rounded-lg border px-3 py-2 w-full md:w-96"
          aria-label="Search for a stock"
        />
        <button
          onClick={scheduleRun}
          disabled={loading || !q.trim()}
          className="text-sm px-3 py-2 rounded-xl border font-semibold text-gray-900 hover:bg-gray-50 disabled:opacity-50"
          title="Get quote"
          aria-label="Get quote"
        >
          {loading ? '...' : 'Get'}
        </button>
      </div>
      {err && <div className="mt-2 text-xs text-red-600">{err}</div>}
      <p className="mt-1 text-[11px] text-gray-500">
        Tip: you can also ask in chat (e.g., “price of MSFT”) and it’ll add a live card below.
      </p>
    </section>
  );
}
