// src/components/CandlesChart.tsx
'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { postJSON } from '@/lib/backend';
import { logUi } from '@/lib/log';
import type {
  DeepPartial,
  ChartOptions,
  CandlestickData,
  UTCTimestamp,
  IChartApi,
  ISeriesApi,
} from 'lightweight-charts';

type Props = {
  symbol: string;
  /** initial resolution, e.g. 'D', '60', '5', 'W' */
  resolution?: string;
  /** initial lookback days (default 180) */
  lookbackDays?: number;
  className?: string;
};

type Candle = { time: number; open: number; high: number; low: number; close: number };

/** Treat big epoch values as ms; normalize to seconds */
function toSeconds(ts: number): number {
  return ts > 1e11 ? Math.floor(ts / 1000) : Math.floor(ts);
}

/** Normalize several common response shapes into Candle[] */
function normalizeCandles(resp: any): Candle[] {
  if (!resp) return [];

  if (Array.isArray(resp.candles)) {
    return resp.candles.map((c: any) => ({
      time: toSeconds(Number(c.time)),
      open: Number(c.open),
      high: Number(c.high),
      low: Number(c.low),
      close: Number(c.close),
    }));
  }

  if (Array.isArray(resp.t) && Array.isArray(resp.o) && Array.isArray(resp.h) && Array.isArray(resp.l) && Array.isArray(resp.c)) {
    const n = Math.min(resp.t.length, resp.o.length, resp.h.length, resp.l.length, resp.c.length);
    const out: Candle[] = [];
    for (let i = 0; i < n; i++) {
      out.push({
        time: toSeconds(Number(resp.t[i])),
        open: Number(resp.o[i]),
        high: Number(resp.h[i]),
        low: Number(resp.l[i]),
        close: Number(resp.c[i]),
      });
    }
    return out;
  }

  if (Array.isArray(resp)) {
    return resp.map((c: any) => ({
      time: toSeconds(Number(c.time ?? c.t)),
      open: Number(c.open ?? c.o),
      high: Number(c.high ?? c.h),
      low: Number(c.low ?? c.l),
      close: Number(c.close ?? c.c),
    }));
  }

  return [];
}

/* -------------------- Range presets -------------------- */
/** UI keys for timeline buttons */
type RangeKey = 'day' | 'week' | 'month' | 'year' | 'total';

/** Map each button to a server-friendly resolution + lookbackDays */
const PRESETS: Record<RangeKey, { label: string; resolution: string; lookbackDays: number }> = {
  // intraday (5-min bars) — pull a couple days to ensure a full latest session
  day:   { label: 'Day',   resolution: '5',  lookbackDays: 2   },
  // hourly bars ~2 weeks to cover 7 trading days comfortably
  week:  { label: 'Week',  resolution: '60', lookbackDays: 14  },
  // daily bars ~1 month
  month: { label: 'Month', resolution: 'D',  lookbackDays: 31  },
  // daily bars ~1 year
  year:  { label: 'Year',  resolution: 'D',  lookbackDays: 370 },
  // weekly bars “all time” (tune as you prefer)
  total: { label: 'Total', resolution: 'W',  lookbackDays: 365 * 20 },
};

/** Try to infer which range best matches incoming props, else default to 'day' */
function inferInitialRange(res?: string, days?: number): RangeKey {
  const r = (res || '').toUpperCase();
  if ((r === '5' || r === '15') && (days ?? 0) <= 3) return 'day';
  if (r === '60' && (days ?? 0) <= 21) return 'week';
  if (r === 'D' && (days ?? 0) <= 45) return 'month';
  if (r === 'D' && (days ?? 0) > 45 && (days ?? 0) < 500) return 'year';
  if (r === 'W' || (days ?? 0) >= 500) return 'total';
  return 'day';
}

/* ------------------------------------------------------- */

export default function CandlesChart({
  symbol,
  resolution: initialResolution = 'D',
  lookbackDays: initialLookbackDays = 180,
  className,
}: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);

  const [candles, setCandles] = useState<Candle[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  // UI state: selected range + the actual parameters we send
  const [range, setRange] = useState<RangeKey>(() =>
    inferInitialRange(initialResolution, initialLookbackDays)
  );
  const [res, setRes] = useState<string>(() => PRESETS[inferInitialRange(initialResolution, initialLookbackDays)].resolution);
  const [days, setDays] = useState<number>(() => PRESETS[inferInitialRange(initialResolution, initialLookbackDays)].lookbackDays);

  // Update res/days when range changes
  useEffect(() => {
    const p = PRESETS[range];
    setRes(p.resolution);
    setDays(p.lookbackDays);
  }, [range]);

  // Compute from/to (unix seconds) from the *current* lookbackDays
  const { from, to } = useMemo(() => {
    const nowSec = Math.floor(Date.now() / 1000);
    const fromSec = nowSec - days * 86400;
    return { from: fromSec, to: nowSec };
  }, [days]);

  // Fetch candles when inputs change
  useEffect(() => {
    let cancelled = false;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 20_000);

    (async () => {
      setLoading(true);
      setError(null);
      try {
        // postJSON prefixes `/api` automatically
        const resp = await postJSON<any>(
          '/market/candles',
          { symbol, resolution: res, from, to },
          { signal: controller.signal }
        );

        if (resp?.error) {
          throw new Error(resp?.note || resp?.detail || resp?.error);
        }

        const data = normalizeCandles(resp).sort((a, b) => a.time - b.time);
        if (!cancelled) setCandles(data);
      } catch (e: any) {
        let msg = 'Failed to load candles';
        if (e?.name === 'AbortError') msg = 'Request timed out';
        if (e?.message) msg = e.message;
        if (!cancelled) {
          setError(msg);
          void logUi({
            level: 'error',
            msg: 'candles_chart_fetch_failed',
            meta: { symbol, resolution: res, from, to, err: msg },
          });
        }
      } finally {
        clearTimeout(timer);
        if (!cancelled) setLoading(false);
      }
    })();

    return () => {
      cancelled = true;
      clearTimeout(timer);
      try { controller.abort(); } catch {}
    };
  }, [symbol, res, from, to]);

  // Create chart once
  useEffect(() => {
    if (!containerRef.current) return;
    let disposed = false;

    (async () => {
      const mod = await import('lightweight-charts');
      if (disposed || !containerRef.current) return;

      const { createChart, CrosshairMode } = mod;
      const options: DeepPartial<ChartOptions> = {
        autoSize: true,
        layout: { textColor: '#111827', background: { color: '#ffffff' } },
        grid: {
          horzLines: { color: 'rgba(17,24,39,0.06)' },
          vertLines: { color: 'rgba(17,24,39,0.06)' },
        },
        crosshair: { mode: CrosshairMode.Normal },
        rightPriceScale: { borderVisible: true },
        timeScale: { borderVisible: true, secondsVisible: false },
      };

      const chart = createChart(containerRef.current, options);
      const series = chart.addCandlestickSeries({
        upColor: '#22c55e',
        downColor: '#ef4444',
        borderUpColor: '#16a34a',
        borderDownColor: '#dc2626',
        wickUpColor: '#16a34a',
        wickDownColor: '#dc2626',
        priceLineVisible: true,
      });

      chartRef.current = chart;
      seriesRef.current = series;

      // Crosshair → set tooltip state (X/Y markers)
      chart.subscribeCrosshairMove((param) => {
        const el = containerRef.current!;
        if (!param?.time || !param?.point) {
          el?.setAttribute('data-tip', '');
          return;
        }
        const price = param.seriesData.get(series) as any;
        const dt = new Date((param.time as number) * 1000);
        const timeStr = dt.toLocaleString();
        const priceStr = price && typeof price.close === 'number' ? price.close.toFixed(2) : '';
        el?.setAttribute('data-tip', `${param.point.x}|${param.point.y}|${timeStr}|${priceStr}`);
      });

      // Resize observer for good measure (autoSize helps, but this stabilizes)
      const ro = new ResizeObserver(() => {
        if (!containerRef.current || !chartRef.current) return;
        const { clientWidth, clientHeight } = containerRef.current;
        chartRef.current.applyOptions({ width: clientWidth, height: Math.max(240, clientHeight) });
      });
      ro.observe(containerRef.current);

      // Cleanup
      return () => {
        ro.disconnect();
      };
    })();

    return () => {
      disposed = true;
      if (chartRef.current) chartRef.current.remove?.();
      chartRef.current = null;
      seriesRef.current = null;
    };
  }, []);

  // Update data when candles change
  useEffect(() => {
    if (!seriesRef.current || !candles?.length) return;
    const data: CandlestickData[] = candles.map((c) => ({
      time: toSeconds(c.time) as UTCTimestamp,
      open: c.open,
      high: c.high,
      low: c.low,
      close: c.close,
    }));
    seriesRef.current.setData(data);
    chartRef.current?.timeScale().fitContent();
  }, [candles]);

  /* ----------- Tooltip (X/Y markers) derived from data-tip ----------- */
  const tip = (() => {
    const raw = containerRef.current?.getAttribute('data-tip') || '';
    if (!raw) return null;
    const [sx, sy, timeStr, priceStr] = raw.split('|');
    const x = Number(sx), y = Number(sy);
    if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
    return { x, y, timeStr, priceStr };
  })();

  const buttons = (['day', 'week', 'month', 'year', 'total'] as RangeKey[]).map((k) => ({
    k,
    label: PRESETS[k].label,
  }));

  return (
    <div className={`rounded-2xl shadow p-4 border ${className ?? ''}`}>
      <div className="flex items-center justify-between mb-3 gap-2">
        <div className="font-semibold">
          {symbol.toUpperCase()} — {PRESETS[range].label}
        </div>
        <div className="flex items-center gap-1">
          {buttons.map((b) => (
            <button
              key={b.k}
              onClick={() => setRange(b.k)}
              className={[
                'text-xs px-2 py-1 rounded-lg border',
                range === b.k ? 'bg-gray-900 text-white border-gray-900' : 'hover:bg-gray-50'
              ].join(' ')}
              aria-pressed={range === b.k}
              title={`${b.label} view`}
            >
              {b.label}
            </button>
          ))}
        </div>
      </div>

      {loading && <div className="text-sm text-gray-500">Loading candles…</div>}
      {error && (
        <div className="text-sm text-red-600 bg-red-50 border border-red-200 rounded-xl p-2">
          {error}
        </div>
      )}

      <div className="relative">
        <div ref={containerRef} className="w-full h-[360px] rounded-xl overflow-hidden border" />

        {/* floating crosshair tooltip */}
        {tip && tip.priceStr && tip.timeStr && (
          <div
            className="pointer-events-none absolute text-[11px] leading-tight whitespace-pre rounded-md border bg-white/95 shadow px-2 py-1"
            style={{
              left: Math.max(8, tip.x + 8),
              top: Math.max(8, tip.y + 8),
            }}
          >
            {tip.timeStr}
            {'\n'}${tip.priceStr}
          </div>
        )}
      </div>

      {!loading && !error && (!candles || candles.length === 0) && (
        <div className="text-sm text-gray-500 mt-2">No data.</div>
      )}
    </div>
  );
}
