// src/components/PriceWidget.tsx
'use client';

import { useMemo, useState } from 'react';
import { fetchPrice } from '@/lib/price';
import { logUi } from '@/lib/log';

type PriceResp =
  | number
  | {
      symbol?: string;
      price?: number | string;
      currency?: string;
      timestamp?: number | string;
      last?: number | string;
      close?: number | string;
      name?: string;
      change?: number;
      changePercent?: number;
    }
  | {
      data?: {
        price?: number | string;
        symbol?: string;
        currency?: string;
        timestamp?: number | string;
      };
    };

function normalizePrice(resp: PriceResp) {
  if (typeof resp === 'number') {
    return { price: resp, currency: undefined as string | undefined, ts: undefined as number | undefined, symbol: undefined as string | undefined, name: undefined as string | undefined };
  }
  const obj = (resp as any)?.data ?? resp;
  const priceRaw = (obj as any).price ?? (obj as any).last ?? (obj as any).close;
  const price = typeof priceRaw === 'string' ? parseFloat(priceRaw) : (priceRaw as number | undefined);
  const currency = (obj as any).currency as string | undefined;
  const symbol = (obj as any).symbol as string | undefined;
  const name = (obj as any).name as string | undefined;

  let ts: number | undefined;
  const t = (obj as any).timestamp;
  if (typeof t === 'number') ts = t > 2_000_000_000 ? Math.floor(t / 1000) : t;
  else if (typeof t === 'string') { const parsed = Date.parse(t); if (!Number.isNaN(parsed)) ts = Math.floor(parsed / 1000); }

  return { price, currency, ts, symbol, name };
}

function fmtCurrency(v: number | undefined, currency?: string) {
  if (v == null || Number.isNaN(v)) return '—';
  try {
    return new Intl.NumberFormat(undefined, {
      style: currency ? 'currency' : 'decimal',
      currency: currency || undefined,
      maximumFractionDigits: 6,
    }).format(v);
  } catch {
    return `${currency ?? ''} ${v}`;
  }
}

function fmtTime(ts?: number) {
  if (!ts) return '';
  const d = new Date(ts * 1000);
  return d.toLocaleString();
}

export default function PriceWidget() {
  const [symbol, setSymbol] = useState('AAPL');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<ReturnType<typeof normalizePrice> | null>(null);

  const subtitle = useMemo(() => {
    if (!data) return '';
    const when = fmtTime(data.ts);
    return when ? `as of ${when}` : '';
  }, [data]);

  async function fetchIt() {
    const s = symbol.trim();
    if (!s) return;
    setLoading(true);
    setError(null);
    setData(null);
    try {
      const resp = await fetchPrice(s);
      setData(normalizePrice(resp));
    } catch (e: any) {
      const msg = e?.message || 'Failed to fetch price';
      setError(msg);
      void logUi({ level: 'error', msg: 'price_widget_failed', meta: { symbol: s, err: msg } });
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="rounded-2xl shadow p-4 border max-w-xl mx-auto">
      <div className="flex items-end gap-2">
        <div className="flex-1">
          <label className="text-sm text-gray-600">Symbol</label>
          <input
            value={symbol}
            onChange={(e) => setSymbol(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter') fetchIt(); }}
            placeholder="AAPL"
            className="w-full border rounded-xl px-3 py-2"
          />
        </div>
        <button
          onClick={fetchIt}
          disabled={loading || !symbol.trim()}
          className="px-4 py-2 rounded-xl border bg-white hover:bg-gray-50 disabled:opacity-50"
        >
          {loading ? 'Loading…' : 'Get Price'}
        </button>
      </div>

      {error && (
        <div className="mt-3 text-sm text-red-600 bg-red-50 border border-red-200 rounded-xl p-2">
          {error}
        </div>
      )}

      {data && (
        <div className="mt-4 border rounded-2xl p-4 bg-gray-50">
          <div className="text-xs text-gray-500">{data.symbol || symbol.toUpperCase()}</div>
          <div className="text-2xl font-semibold">{fmtCurrency(data.price, data.currency)}</div>
          {subtitle && <div className="text-xs text-gray-500 mt-1">{subtitle}</div>}
          {data.name && <div className="text-xs text-gray-500 mt-1">{data.name}</div>}
        </div>
      )}
    </div>
  );
}
