// src/app/analytics/page.tsx
'use client';

import useSWR, { useSWRConfig } from 'swr';
import { useMemo, useState } from 'react';
import { buildUrl } from '@/lib/backend';
import {
  LineChart, Line, CartesianGrid, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend,
  BarChart, Bar, Label,
} from 'recharts';

// Shared accent color for lines + bars
const ACCENT = '#4f46e5'; // indigo-600

const fetcher = (url: string) =>
  fetch(url, { cache: 'no-store' }).then(r => {
    if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
    return r.json();
  });

const nfmt = (n: number | undefined | null, digits = 0) =>
  (n === undefined || n === null) ? '—' :
  Intl.NumberFormat(undefined, { maximumFractionDigits: digits }).format(n);

// Currency (USD) – rounded to nearest whole dollar
const $fmt0 = (n: number | undefined | null) =>
  (n === undefined || n === null) ? '—' :
  Intl.NumberFormat(undefined, {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(Math.round(n));

const pfmt = (x: number | undefined | null) =>
  (x === undefined || x === null) ? '—' :
  Intl.NumberFormat(undefined, { style: 'percent', maximumFractionDigits: 1 }).format(x);

function timeLabel(ts: number, spanDays: number) {
  const d = new Date(ts);
  if (spanDays <= 1) return d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
  return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}

// Shorten tool names for ticks; full name in tooltip
function shortName(name: string, max = 24) {
  if (!name) return '';
  return name.length > max ? name.slice(0, max - 1) + '…' : name;
}

type RangeOpt = 1 | 3 | 7 | 30 | 90 | 0; // 0 = all time
const DEFAULT_DAYS: RangeOpt = 7;

export default function AnalyticsPage() {
  const [days, setDays] = useState<RangeOpt>(DEFAULT_DAYS);
  const { mutate } = useSWRConfig();

  const keySummary   = buildUrl(`/api/analytics/summary?days=${days}`);
  const keySeries    = buildUrl(`/api/analytics/series?days=${days}`);
  const keyBreakdown = buildUrl(`/api/analytics/breakdown?days=${days}`);
  const keyTop       = buildUrl(`/api/analytics/top?days=${days}`);

  const { data: summary,   error: err1 } = useSWR(keySummary, fetcher,   { refreshInterval: 15000 });
  const { data: series,    error: err2 } = useSWR(keySeries, fetcher,    { refreshInterval: 15000 });
  const { data: breakdown, error: err3 } = useSWR(keyBreakdown, fetcher, { refreshInterval: 15000 });
  const { data: top,       error: err4 } = useSWR(keyTop, fetcher,       { refreshInterval: 15000 });

  // Tokens line
  const chartData = useMemo(() => {
    const raw: any[] = Array.isArray(series?.points) ? series.points : [];
    return raw.map(p => {
      const tVal = typeof p.t === 'number' ? p.t : Date.parse(p.t);
      return { ...p, t: tVal };
    });
  }, [series]);

  // Tool usage (top 10)
  const toolBars = useMemo(() => {
    const arr: any[] = Array.isArray(top?.tools) ? top.tools : [];
    return arr.sort((a, b) => (b.count || 0) - (a.count || 0)).slice(0, 10);
  }, [top]);

  const lat = breakdown?.latency || {};
  const err = err1 || err2 || err3 || err4;

  // Reset: default range + revalidate all analytics keys in cache
  const handleResetAll = async () => {
    setDays(DEFAULT_DAYS);
    await mutate((key) => typeof key === 'string' && key.includes('/api/analytics/'), undefined, { revalidate: true });
  };

  return (
    <main className="p-6 md:p-8 lg:p-10 max-w-6xl mx-auto space-y-6">
      <header className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <h1 className="text-2xl font-semibold">Analytics</h1>

        <div className="flex items-center gap-2">
          <RangeButton label="Today" active={days === 1} onClick={() => setDays(1)} />
          <RangeButton label="3d"    active={days === 3} onClick={() => setDays(3)} />
          <RangeButton label="7d"    active={days === 7} onClick={() => setDays(7)} />
          <RangeButton label="30d"   active={days === 30} onClick={() => setDays(30)} />
          <RangeButton label="90d"   active={days === 90} onClick={() => setDays(90)} />
          <RangeButton label="All"   active={days === 0} onClick={() => setDays(0)} />
          <button
            className="ml-2 rounded-xl border px-3 py-1 text-sm hover:bg-gray-50"
            onClick={handleResetAll}
            aria-label="Reset to default range and refresh"
            title="Reset to default range and refresh"
          >
            Reset
          </button>
        </div>
      </header>

      {/* Summary cards: cost gets a bit more room */}
      <section className="grid grid-cols-2 md:grid-cols-6 gap-4 items-stretch">
        <Card title={`Turns (${days === 0 ? 'all' : days + 'd'})`}>{nfmt(summary?.turns)}</Card>

        {/* widen cost on md+ */}
        <div className="md:col-span-2">
          <Card title={`Cost (${days === 0 ? 'all' : days + 'd'})`}>
            <span className="break-words leading-tight">{ $fmt0(summary?.cost_usd) }</span>
          </Card>
        </div>

         <Card title="RAG rate">{pfmt(summary?.rag_rate)}</Card>
        <Card title="LLM error rate">{pfmt(summary?.error_rate)}</Card>
        <Card title="Tool failure rate">{pfmt(summary?.tool_error_rate)}</Card>
      </section>


      {/* Tokens over time */}
      <section className="rounded-2xl border bg-white p-4">
        <h2 className="text-lg font-semibold mb-3">Tokens over time</h2>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 10, right: 24, bottom: 0, left: 28 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="t"
                type="number"
                scale="time"
                domain={['dataMin', 'dataMax']}
                tickFormatter={(v) => timeLabel(v as number, days || 30)}
                minTickGap={28}
              />
              <YAxis tickFormatter={(v) => nfmt(v as number)} width={96} tickMargin={10} />
              <Tooltip labelFormatter={(v) => new Date(v as number).toLocaleString()} formatter={(value: any, name: string) => [nfmt(value), name]} />
              <Legend />
              <Line type="monotone" dataKey="tokens" name="Tokens" dot={false} strokeWidth={1.5} stroke={ACCENT} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </section>

      {/* Latency cards */}
      <section className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card title="Avg latency">{nfmt(lat?.avg_ms)} ms</Card>
        <Card title="Median latency">{nfmt(lat?.median_ms)} ms</Card>
        <Card title="P95 latency">{nfmt(lat?.p95_ms)} ms</Card>
        <Card title="P99 latency">{nfmt(lat?.p99_ms)} ms</Card>
      </section>

      {/* Tool usage — horizontal bars (bigger & unclipped) */}
      <section className="rounded-2xl border bg-white p-4">
        <h2 className="text-lg font-semibold mb-3">Tool usage (top 10)</h2>
        {/* bigger height + generous margins; labels won’t clip */}
          <div className="h-96 overflow-visible">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={toolBars}
                layout="vertical"
                margin={{ top: 12, right: 32, bottom: 24, left: 40 }}
                barSize={20}
                barCategoryGap={10}
                barGap={4}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <YAxis
                  type="category"
                  dataKey="name"
                  width={240}               
                  tick={{ fontSize: 12 }}
                  tickLine={false}
                  axisLine={false}
                  tickFormatter={(v: string) => shortName(v, 30)}
                >
                  <Label value="Tools" angle={-90} position="insideLeft" offset={-10} />
                </YAxis>
                <XAxis type="number" allowDecimals={false}>
                  <Label value="Uses" position="insideBottom" offset={-6} />
                </XAxis>
                <Tooltip
                  labelFormatter={(_lbl, payload: any) => payload?.[0]?.payload?.name ?? ''}
                  formatter={(v: any) => [nfmt(v), 'Uses']}
                />
                <Bar dataKey="count" name="Uses" fill={ACCENT} />
              </BarChart>
            </ResponsiveContainer>
          </div>

        {toolBars.length === 0 && (
          <div className="text-xs text-gray-500 mt-2">No tool usage recorded in this range.</div>
        )}
      </section>

      {err && <div className="text-sm text-red-600">{err.message || 'Failed to load analytics.'}</div>}
    </main>
  );
}

function RangeButton({ label, active, onClick }: { label: string; active?: boolean; onClick: () => void }) {
  return (
    <button
      className={[
        "rounded-xl border px-3 py-1 text-sm",
        active ? "bg-gray-900 text-white border-gray-900" : "hover:bg-gray-50"
      ].join(' ')}
      onClick={onClick}
      aria-pressed={!!active}
    >
      {label}
    </button>
  );
}

function Card({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-2xl border bg-white p-4">
      <div className="text-xs text-gray-500">{title}</div>
      {/* prevent overflow; allow wrap; compact line-height */}
      <div className="text-2xl font-semibold leading-tight break-words">{children}</div>
    </div>
  );
}
