// src/components/PriceCard.tsx
'use client';
import React, { useEffect, useMemo, useState } from 'react';
import StockSparkline from './StockSparkline';
import { fetchCandlesTD } from '@/lib/price';

type Alias = { symbol: string; name?: string };
type Props = {
  symbol: string;
  name?: string;
  currency?: string;
  price?: number;
  change?: number;
  changePercent?: number;
  aliases?: Alias[];
  /** compact mini card mode */
  compact?: boolean;
};

/** Smallest → largest (Day removed), keep YTD */
type PeriodKey = 'WEEK' | '1M' | '6M' | 'YTD' | '1Y' | 'MAX';

function daysSinceJan1(): number {
  const now = new Date();
  const start = new Date(now.getFullYear(), 0, 1);
  // +2-day buffer so sparse markets still fill the chart
  return Math.max(1, Math.ceil((now.getTime() - start.getTime()) / 86400000)) + 2;
}

export default function PriceCard({
  symbol,
  name,
  currency,
  price,
  change,
  changePercent,
  aliases,
  compact = false,
}: Props) {
  const [period, setPeriod] = useState<PeriodKey>('1M');
  const [times, setTimes] = useState<number[]>([]);
  const [closes, setCloses] = useState<number[]>([]);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  /**
   * Map each period to Twelve Data params
   * NOTE: constrain to fetchCandlesTD's allowed intervals:
   *   interval: "1day" | "1week" | "1month"
   */
  const tdParams = useMemo(() => {
    switch (period) {
      case 'WEEK':
        // Daily bars ~10 days to cover ~7 trading days (with buffer)
        return { interval: '1day' as const, outputsize: 10 };
      case '1M':
        return { interval: '1day' as const, outputsize: 32 };
      case '6M':
        return { interval: '1day' as const, outputsize: 185 };
      case 'YTD':
        return { interval: '1day' as const, outputsize: daysSinceJan1() };
      case '1Y':
        return { interval: '1day' as const, outputsize: 365 };
      case 'MAX':
        return { interval: '1week' as const, outputsize: 1500 };
    }
  }, [period]);

  useEffect(() => {
    let ok = true;
    if (!symbol || !tdParams) { setTimes([]); setCloses([]); return; }
    const sym = String(symbol).trim().toUpperCase();

    setLoading(true);
    setErr(null);
    fetchCandlesTD(sym, tdParams)
      .then((d) => {
        if (!ok) return;
        setTimes(d.t || []);
        setCloses(d.c || []);
      })
      .catch((e) => {
        if (!ok) return;
        const msg = (e?.message || e || 'Failed to load').toString();
        setErr(msg);
      })
      .finally(() => { if (ok) setLoading(false); });

    return () => { ok = false; };
  }, [symbol, tdParams]);

  const ch = typeof change === 'number' ? change.toFixed(2) : '';
  const chp = typeof changePercent === 'number' ? changePercent.toFixed(2) : '';
  const hasDelta = ch !== '' || chp !== '';
  const isUp = Number(ch || 0) >= 0;

  const LABELS: Record<PeriodKey, string> = {
    WEEK: 'Week',
    '1M': '1M',
    '6M': '6M',
    YTD: 'YTD',
    '1Y': '1Y',
    MAX: 'Total',
  };

  const ORDER: PeriodKey[] = ['WEEK', '1M', '6M', 'YTD', '1Y', 'MAX'];

  return (
    <div className={`rounded-2xl border bg-white shadow-sm h-full w-full min-w-0 ${compact ? 'p-3' : 'p-4'}`}>
      <div className="flex items-baseline justify-between gap-3">
        <div className="min-w-0">
          <div className="text-sm text-gray-500 truncate">
            {(symbol || '').toUpperCase()}
            {name && name.toUpperCase() !== (symbol || '').toUpperCase() ? (
              <span className="text-gray-400"> — {name}</span>
            ) : null}
          </div>
          <div className={`${compact ? 'text-sm' : 'text-base'} font-semibold truncate`}>
            {name || symbol}
          </div>
        </div>
        <div className="text-right">
          <div className={`${compact ? 'text-lg' : 'text-xl'} font-semibold`}>
            {price !== undefined ? price.toLocaleString(undefined, { maximumFractionDigits: 4 }) : '—'}
            {currency ? <span className="text-sm text-gray-500">&nbsp;{currency}</span> : null}
          </div>
          {hasDelta && (
            <div className={['text-xs', isUp ? 'text-emerald-600' : 'text-rose-600'].join(' ')}>
              {isUp ? '▲' : '▼'} {ch} {chp ? `(${chp}%)` : ''}
            </div>
          )}
        </div>
      </div>

      {/* Chart */}
      <div className="mt-3 overflow-hidden rounded-xl border bg-white">
        <div className={`w-full ${compact ? 'h-28 md:h-32' : 'h-44 md:h-56'}`}>
          {loading ? (
            <div className="w-full h-full flex items-center justify-center text-xs text-gray-500">
              Loading chart…
            </div>
          ) : err ? (
            <div className="w-full h-full flex items-center justify-center">
              <div className="text-xs text-rose-600 bg-rose-50 border border-rose-200 rounded px-2 py-1">
                {err}
              </div>
            </div>
          ) : (times.length > 1 && closes.length > 1) ? (
            <StockSparkline className="w-full h-full" times={times} closes={closes} showAxes />
          ) : (
            <div className="w-full h-full flex items-center justify-center text-xs text-gray-500">
              No chart data for this period.
            </div>
          )}
        </div>
      </div>

      {/* Period buttons (hide on compact) */}
      {!compact && (
        <div className="mt-3 flex flex-wrap gap-1">
          {ORDER.map((p) => (
            <button
              key={p}
              onClick={() => setPeriod(p)}
              aria-pressed={p === period}
              className={[
                'text-xs px-2 py-1 rounded-lg border',
                p === period ? 'bg-gray-900 text-white' : 'bg-gray-50 hover:bg-gray-100',
              ].join(' ')}
              title={LABELS[p]}
            >
              {LABELS[p]}
            </button>
          ))}
        </div>
      )}

      {/* Aliases */}
      {!compact && aliases?.length ? (
        <div className="mt-2 text-[11px] text-gray-500">
          Also found:&nbsp;
          {aliases.slice(0, 4).map((a, i) => (
            <button
              key={a.symbol}
              onClick={() => {
                window.dispatchEvent(new CustomEvent('finassist:add-tool', {
                  detail: { tool: 'live_price', result: { query: a.symbol, symbol: a.symbol } }
                }));
              }}
              className="underline underline-offset-2 decoration-dotted hover:decoration-solid mr-1"
              title={a.name ? `${a.name} (${a.symbol})` : a.symbol}
            >
              <span className="font-medium">{a.symbol}</span>{a.name ? ` (${a.name})` : ''}
              {i < Math.min(aliases.length, 4) - 1 ? ',' : ''}
            </button>
          ))}
        </div>
      ) : null}
    </div>
  );
}
