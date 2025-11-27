// src/components/ToolResults.tsx
'use client';

import { useMemo, useState, useEffect } from 'react';
import type React from 'react';
import PriceCard from './PriceCard';
import { buildUrl } from '@/lib/backend';
import { trackTool, SESSION_KEY as ANALYTICS_SESSION_KEY } from '@/lib/analytics';

const FE_ANALYTICS = process.env.NEXT_PUBLIC_ANALYTICS_FE === '1';
const TOOL_LABELS: Record<string, string> = {
  live_price: 'Live Price',
  profile: 'Company Profile',
  company_profile: 'Company Profile',
  recommendation: 'Analyst Trends',
  recommendation_trends: 'Analyst Trends',
  news: 'Recent News',
  company_news: 'Recent News',
  world_bank: 'World Bank',
};

function labelForTool(raw?: string) {
  const k = String(raw || '').toLowerCase();
  return TOOL_LABELS[k] || (k ? k.replace(/_/g, ' ') : 'tool');
}

// conservative URL guard (keeps http/https and protocol-relative)
function sanitizeUrl(u?: string | null): string | undefined {
  if (!u) return undefined;
  const s = String(u).trim();
  if (/^https?:\/\//i.test(s) || /^\/\//.test(s)) return s;
  return undefined;
}

type Props = { items: any[] };
type PriceCardProps = React.ComponentProps<typeof PriceCard>;

/* ============================ Small utils ============================ */

function isPlainObject(x: unknown): x is Record<string, unknown> {
  return !!x && typeof x === 'object' && !Array.isArray(x);
}

function clamp(n: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, n));
}

function truncate(s: string, max = 6000) {
  if (s.length <= max) return s;
  return s.slice(0, max) + '\n… (trimmed)';
}

function formatResult(result: unknown): string {
  try {
    let out = '';
    if (typeof result === 'string') out = result;
    else if (result == null) out = 'null';
    else if (typeof result === 'number' || typeof result === 'boolean') out = String(result);
    else out = JSON.stringify(result, null, 2);

    const SOFT_LIMIT = 6000; 
    if (out.length > SOFT_LIMIT) {
      out = `${out.slice(0, SOFT_LIMIT)}\n…(truncated)`;
    }
    return out;
  } catch {
    return String(result);
  }
}


/* ---------- Guard for live price payload ---------- */
function isPriceCardProps(x: unknown): x is PriceCardProps {
  if (!isPlainObject(x)) return false;
  const o = x as Record<string, unknown>;
  return (
    typeof o.symbol === 'string' &&
    (typeof o.price === 'number' || o.price === undefined) &&
    (typeof o.change === 'number' || o.change === undefined) &&
    (typeof o.changePercent === 'number' || o.changePercent === undefined) &&
    (typeof o.currency === 'string' || o.currency === undefined)
  );
}

/* ---------- Small UI bits ---------- */
function Badge({ children, tone = 'default' }: { children: React.ReactNode; tone?: 'default' | 'ok' | 'err' }) {
  const toneCls =
    tone === 'ok'
      ? 'bg-green-100 text-green-700 border-green-200'
      : tone === 'err'
      ? 'bg-red-100 text-red-700 border-red-200'
      : 'bg-gray-100 text-gray-700 border-gray-200';
  return (
    <span className={`inline-flex items-center rounded px-2 py-0.5 text-xs border ${toneCls}`}>
      {children}
    </span>
  );
}

function ToolButton({ label, onClick }: { label: string; onClick: () => Promise<void> | void }) {
  const [loading, setLoading] = useState(false);
  return (
    <button
      className="text-xs px-2 py-1 rounded-lg border hover:bg-gray-50 disabled:opacity-50"
      disabled={loading}
      onClick={async () => {
        setLoading(true);
        try { await onClick(); } finally { setLoading(false); }
      }}
    >
      {loading ? '…' : label}
    </button>
  );
}

function ActionsRow({ symbol }: { symbol: string }) {
  return (
    <div className="rounded-2xl border bg-white shadow-sm p-3 flex flex-wrap gap-2 items-center">
      <span className="text-xs text-gray-600">
        Actions for <strong>{symbol}</strong>:
      </span>
      <ToolButton label="Company Profile" onClick={() => fetchTool('profile', symbol)} />
      <ToolButton label="Analyst Trends" onClick={() => fetchTool('recommendation', symbol)} />
      <ToolButton label="Recent News" onClick={() => fetchTool('news', symbol)} />
    </div>
  );
}

/* ---------- Generic formatters ---------- */
function fmtMoney(n?: number | null, digits = 0) {
  if (typeof n !== 'number' || Number.isNaN(n)) return 'n/a';
  return n.toLocaleString(undefined, { style: 'currency', currency: 'USD', maximumFractionDigits: digits });
}
function fmtDateFromEpochSec(t?: number | null) {
  if (!t || typeof t !== 'number') return '';
  const d = new Date(t * 1000);
  return d.toLocaleString();
}

/* ---------- Type guards for pretty payloads ---------- */
function isProfile(x: any) {
  return x && typeof x === 'object' && typeof x.symbol === 'string';
}
function isRecoRow(x: any) {
  return x && typeof x === 'object' && ['strongBuy', 'buy', 'hold', 'sell', 'strongSell'].some(k => k in x);
}
function normalizeReco(input: any): any | null {
  if (Array.isArray(input)) {
    const rows = input.filter(isRecoRow);
    if (rows.length === 0) return null;
    return rows[0] ?? rows[rows.length - 1];
  }
  return isRecoRow(input) ? input : null;
}
function isNewsArray(x: any): x is any[] {
  return Array.isArray(x);
}
function isNewsObject(x: any): x is { items: any[] } {
  return x && typeof x === 'object' && Array.isArray(x.items);
}
function normalizeNews(input: any): { symbol?: string; items: any[] } {
  if (isNewsArray(input)) return { items: input };
  if (isNewsObject(input)) return { symbol: (input as any).symbol, items: input.items };
  return { items: [] };
}

/* ---------- World Bank CSV helpers ---------- */

function parseCSV(text: string): { columns: string[]; rows: string[][] } {
  const rows: string[][] = [];
  let row: string[] = [];
  let cur = '';
  let inQuotes = false;

  const pushCell = () => { row.push(cur); cur = ''; };
  const pushRow = () => {
    if (row.length > 0 && !(row.length === 1 && row[0] === '')) rows.push(row);
    row = [];
  };

  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (inQuotes) {
      if (ch === '"') {
        const next = text[i + 1];
        if (next === '"') { cur += '"'; i++; } else { inQuotes = false; }
      } else {
        cur += ch;
      }
    } else {
      if (ch === '"') inQuotes = true;
      else if (ch === ',') pushCell();
      else if (ch === '\n' || ch === '\r') {
        if (ch === '\r' && text[i + 1] === '\n') i++;
        pushCell(); pushRow();
      } else cur += ch;
    }
  }
  if (cur.length > 0 || row.length > 0) { pushCell(); pushRow(); }

  if (rows.length === 0) return { columns: [], rows: [] };

  const [header, ...data] = rows as [string[], ...string[][]];
  const columns = header.map(h => (h ?? '').trim());
  const dataRows = data.filter(r => r.some(c => c !== ''));

  return { columns, rows: dataRows };
}

function toNum(x: string | null | undefined): number | null {
  if (x == null || x.trim() === '') return null;
  const n = Number(x);
  return Number.isFinite(n) ? n : null;
}

type WBRow = { date: number; value: number | null; iso3?: string; indicator?: string; country?: string; };

function extractWorldBankSeries(csvText: string) {
  const { columns, rows } = parseCSV(csvText);
  if (!columns.length || !rows.length) {
    return { series: [] as WBRow[], indicatorName: undefined as string | undefined, countryName: undefined as string | undefined };
  }

  const idx = (name: string) => columns.indexOf(name);

  const iDate = idx('date');
  const iVal = idx('value');
  const iIso3 = idx('countryiso3code');
  const iIndName = idx('indicator.value');
  const iCountry = idx('country.value');

  const series: WBRow[] = rows
    .map(r => {
      const obj: WBRow = {
        date: Number(r[iDate] ?? NaN),
        value: toNum(r[iVal]),
      };
      if (iIso3 >= 0) obj.iso3 = r[iIso3]!;
      if (iIndName >= 0) obj.indicator = r[iIndName]!;
      if (iCountry >= 0) obj.country = r[iCountry]!;
      return obj;
    })
    .filter(r => Number.isFinite(r.date))
    .sort((a, b) => a.date - b.date);

  const indicatorName = series.find(s => s.indicator)?.indicator;
  const countryName = series.find(s => s.country)?.country;

  return { series, indicatorName, countryName };
}

function Sparkline({ data, width = 280, height = 60 }: { data: (number | null)[]; width?: number; height?: number }) {
  const clean = data.map(v => (typeof v === 'number' ? v : null)).filter(v => v != null) as number[];
  if (clean.length < 2) return <svg width={width} height={height} />;
  const min = Math.min(...clean);
  const max = Math.max(...clean);
  const dx = width / (data.length - 1 || 1);
  const scaleY = (v: number) => (max === min ? height / 2 : height - ((v - min) / (max - min)) * (height - 8) - 4);

  const pts: string[] = [];
  data.forEach((v, i) => { if (v != null) pts.push(`${(i * dx).toFixed(2)},${scaleY(v).toFixed(2)}`); });

  return (
    <svg width={width} height={height} aria-label="sparkline">
      <polyline points={pts.join(' ')} fill="none" stroke="currentColor" strokeWidth={1.5} />
    </svg>
  );
}



function WorldBankCSVCard({ item }: { item: any }) {
  const csvText: string | undefined = item?.result?.text || item?.text;

  if (!csvText) {
    return (
      <div className="rounded-2xl border p-4 bg-white shadow-sm w-full h-full">
        <div className="text-sm font-semibold">No data</div>
        <div className="text-xs text-gray-600">No World Bank data returned.</div>
      </div>
    );
  }

  const isStandardLike = csvText.includes(',countryiso3code,') && csvText.includes('\n');

  if (!isStandardLike) {
    return (
      <div className="rounded-2xl border p-4 bg-white shadow-sm w-full h-full">
        <div className="text-sm font-semibold">Unrecognized data format</div>
        <div className="text-xs text-gray-600">Data received but not in the expected schema.</div>
      </div>
    );
  }

  const { series, indicatorName, countryName } = extractWorldBankSeries(csvText);

  if (!series.length) {
    return (
      <div className="rounded-2xl border p-4 bg-white shadow-sm w-full h-full">
        <div className="text-sm font-semibold">Couldn’t parse series</div>
        <div className="text-xs text-gray-600">Please try a different indicator/country.</div>
      </div>
    );
  }

  // Dataset bounds
  const minYear = Math.min(...series.map(s => s.date));
  const maxYear = Math.max(...series.map(s => s.date));

  // Preferred range from MacroSnapshot (clamped to bounds)
  const meta = item?.meta || {};
  const prefStart = Number.isFinite(meta.startYear) ? Number(meta.startYear) : undefined;
  const prefEnd   = Number.isFinite(meta.endYear)   ? Number(meta.endYear)   : undefined;

  const s = prefStart !== undefined ? Math.max(minYear, Math.min(prefStart, maxYear)) : minYear;
  const e = prefEnd   !== undefined ? Math.max(s,       Math.min(prefEnd,   maxYear)) : maxYear;

  const filtered = series.filter(sv => sv.date >= s && sv.date <= e);

  const latest = [...filtered].reverse().find(sv => sv.value != null);
  const countryLabel = countryName ?? series[0]?.iso3 ?? '';
  const showTitle = countryLabel || 'World Bank';

  // If more than 10 values in range, make the table area scrollable
  const rowsScrollable = filtered.length > 10;
  const tableWrapperCls = rowsScrollable
    ? 'mt-3 overflow-y-auto max-h-64 border rounded-lg'
    : 'mt-3 overflow-x-auto';

  return (
    <div className="rounded-2xl border p-4 bg-white shadow-sm w-full h-full">
      {/* Title = country name, no extra "tool:" text */}
      <div className="text-base font-semibold mb-1">{showTitle}</div>

      <div className="text-xs text-gray-500 mb-1">
        {indicatorName || 'Indicator'} · {s}–{e}
      </div>

      <div className="flex items-end gap-3">
        <div className="text-2xl font-semibold">
          {latest?.value != null ? latest.value.toFixed(2) : '—'}
          <span className="text-gray-500 text-sm ml-2">{latest?.date ?? ''}</span>
        </div>
      </div>

      {/* Sparkline */}
      <div className="mt-2 text-gray-700">
        <Sparkline data={filtered.map(sv => sv.value)} />
      </div>

      {/* Table — all rows in the selected range; scroll if >10 */}
      <div className={tableWrapperCls}>
        <table className="w-full text-sm">
          <thead className="sticky top-0 bg-white">
            <tr className="text-left border-b">
              <th className="py-1 pr-3">Year</th>
              <th className="py-1 pr-3">Value</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((r, i) => (
              <tr key={i} className="border-b last:border-0">
                <td className="py-1 pr-3">{r.date}</td>
                <td className="py-1 pr-3">{r.value != null ? r.value.toFixed(3) : '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}




/* ---------- Pretty mini-cards ---------- */
function ProfileCardMini({ data }: { data: any }) {
  return (
    <div className="rounded-2xl border p-4 bg-white shadow-sm w-full h-full">
      <div className="flex items-center gap-2 mb-2">
        {data.logo ? <img src={data.logo} alt="" className="w-6 h-6 rounded-sm" /> : null}
        <div className="text-sm font-semibold">
          {(data.name || data.symbol)} <span className="text-gray-500">({String(data.symbol || '').toUpperCase()})</span>
        </div>
      </div>
      <div className="text-xs text-gray-700 space-y-1">
        <div><span className="text-gray-500">Exchange:</span> {data.exchange || 'n/a'}</div>
        <div><span className="text-gray-500">Currency:</span> {data.currency || 'n/a'}</div>
        <div><span className="text-gray-500">Market Cap:</span> {fmtMoney(data.market_cap, 0)}</div>
        <div><span className="text-gray-500">IPO:</span> {data.ipo || 'n/a'}</div>
      </div>
    </div>
  );
}

function RecoTrendsCard({ data }: { data: any }) {
  const Row = ({ label, val, tone }: { label: string; val: number; tone?: 'g' | 'y' | 'r' }) => {
    const cls =
      tone === 'g' ? 'bg-green-100 text-green-700 border-green-200'
      : tone === 'y' ? 'bg-yellow-100 text-yellow-700 border-yellow-200'
      : tone === 'r' ? 'bg-red-100 text-red-700 border-red-200'
      : 'bg-gray-100 text-gray-700 border-gray-200';
    return (
      <span className={`inline-flex items-center rounded px-2 py-0.5 text-xs border ${cls}`}>{label}: {val ?? 0}</span>
    );
  };
  return (
    <div className="rounded-2xl border p-4 bg-white shadow-sm w-full h-full">
      <div className="text-sm font-semibold mb-1">
        Analyst recommendation trends {data.symbol ? <>— {String(data.symbol).toUpperCase()}</> : null}
        {data.period ? <span className="text-gray-500 text-xs"> ({data.period})</span> : null}
      </div>
      <div className="flex flex-wrap gap-2">
        <Row label="Strong Buy" val={data.strongBuy} tone="g" />
        <Row label="Buy"       val={data.buy}       tone="g" />
        <Row label="Hold"      val={data.hold}      tone="y" />
        <Row label="Sell"      val={data.sell}      tone="r" />
        <Row label="Strong Sell" val={data.strongSell} tone="r" />
      </div>
    </div>
  );
}

function NewsListCard({ data }: { data: { symbol?: string; items: any[] } }) {
  const uniq: any[] = [];
  const seen = new Set<string>();
  for (const n of data.items || []) {
    const key = String(n.url || n.link || n.headline || n.title || '').trim().toLowerCase();
    if (!key || seen.has(key)) continue;
    seen.add(key);
    uniq.push(n);
  }
  data = { ...data, items: uniq };
  
  return (
    <div className="rounded-2xl border p-4 bg-white shadow-sm w-full h-full">
      <div className="text-sm font-semibold mb-2">Company news {data.symbol ? <>— {String(data.symbol).toUpperCase()}</> : null}</div>
      {data.items.length === 0 ? (
        <div className="text-xs text-gray-500">No recent news.</div>
      ) : (
        <ul className="space-y-1">
          {data.items.slice(0, 10).map((n, i) => (
            <li key={i} className="text-xs">
              <a
                href={sanitizeUrl(n.url) || '#'}
                target="_blank"
                rel="noreferrer noopener"
                className="underline underline-offset-2"
                title={n.source ? `${n.source}${n.datetime ? ' • ' + fmtDateFromEpochSec(n.datetime) : ''}` : undefined}
              >
                {n.headline || n.title || '(no headline)'}
              </a>
              {n.source ? <span className="text-gray-500"> — {n.source}</span> : null}
              {n.datetime ? <span className="text-gray-400"> · {fmtDateFromEpochSec(n.datetime)}</span> : null}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

/* ============================ Main component ============================ */
export default function ToolResults({ items }: Props) {
  useEffect(() => {
    if (!Array.isArray(items) || items.length === 0) return;
    const sid = (typeof window !== 'undefined' && localStorage.getItem(ANALYTICS_SESSION_KEY)) || null;

    items.forEach((it) => {
      const toolName = String(it?.tool || it?.name || 'tool');
      const ok = typeof it?.ok === 'boolean' ? it.ok : !it?.error;
      const latency = typeof it?.elapsed_ms === 'number' ? it.elapsed_ms : undefined;

      const payload = {
        tool_name: toolName,
        ...(sid ? { session_id: sid } : {}),
        ...(latency !== undefined ? { latency_ms: latency } : {}),
        ...(ok !== undefined ? { ok } : {}),
        ...(it?.error ? { error: String(it.error) } : {}),
      } as const;

      if (FE_ANALYTICS) trackTool(payload);
    });
  }, [items]);

  const { mainLive, miniLives, others } = useMemo(() => {
    const arr = Array.isArray(items) ? items : [];
    const liveAll = arr.filter((it) => it?.tool === 'live_price');

    // unique by symbol (case-insensitive), keeping first occurrence
    const seen = new Set<string>();
    const uniqLive: any[] = [];
    for (const it of liveAll) {
      const sym = String(it?.result?.symbol || '').toUpperCase();
      if (!sym || seen.has(sym)) continue;
      seen.add(sym);
      uniqLive.push(it);
    }

    // Split: 1 big + up to 3 mini
    const mainLive = uniqLive[0] || null;
    const miniLives = uniqLive.slice(1, 4);

    // Keep only the most recent 2 “other” action outputs (split screen)
    const others = arr.filter((it) => it?.tool !== 'live_price').slice(0, 2);

    return { mainLive, miniLives, others };
  }, [items]);

  if (!mainLive && others.length === 0) {
    return (
      <section className="border rounded-2xl bg-white w-full">
        <div className="px-3 py-2 border-b bg-gray-50">
          <h2 className="text-sm font-semibold">Details</h2>
        </div>
        <div className="p-3 text-xs text-gray-500">No tool results yet.</div>
      </section>
    );
  }

  const livePayload: any | undefined = mainLive?.result;
  const liveSymbol = livePayload?.symbol ? String(livePayload.symbol).toUpperCase() : '';

  return (
    <section className="border rounded-2xl bg-white w-full">
      <div className="px-3 py-2 border-b bg-gray-50">
        <h2 className="text-sm font-semibold">Details</h2>
      </div>

      <div className="p-3 space-y-3">
        {/* Actions row ABOVE the chart */}
        {liveSymbol ? <ActionsRow symbol={liveSymbol} /> : null}

        {/* Main big chart */}
        {(() => {
          if (!livePayload || !liveSymbol) return null;

          const cur = (livePayload.currency as string | undefined) ?? undefined;
          const mainProps: PriceCardProps = {
            symbol: liveSymbol,
            ...(livePayload.price !== undefined ? { price: livePayload.price } : {}),
            ...(livePayload.change !== undefined ? { change: livePayload.change } : {}),
            ...(livePayload.changePercent !== undefined ? { changePercent: livePayload.changePercent } : {}),
            ...(cur ? { currency: cur } : {}),
            ...(Array.isArray(livePayload.aliases) ? { aliases: livePayload.aliases } : {}),
          };

          if (!isPriceCardProps(mainProps)) return null;
          return <PriceCard {...mainProps} />;
        })()}

        {/* Mini charts row (up to 3) */}
        {miniLives.length > 0 && (
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            {miniLives.map((it, idx) => {
              const p: any = it?.result || {};
              const sym = String(p.symbol || '').toUpperCase();
              if (!sym) return null;

              const cur = (it?._raw?.currency as string | undefined) ?? (p.currency as string | undefined) ?? undefined;
              const miniProps: PriceCardProps = {
                symbol: sym,
                ...(p.price !== undefined ? { price: p.price } : {}),
                ...(p.change !== undefined ? { change: p.change } : {}),
                ...(p.changePercent !== undefined ? { changePercent: p.changePercent } : {}),
                ...(cur ? { currency: cur } : {}),
                compact: true,
              };

              if (!isPriceCardProps(miniProps)) return null;
              return <PriceCard key={`${sym}:${idx}`} {...miniProps} />;
            })}
          </div>
        )}

        {/* Pretty renderers for other tool outputs (max 2; split screen) */}
        {others.length > 0 && (
          <div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {others.map((it, i) => {
                const tool = String(it?.tool || it?.name || 'tool').toLowerCase();
                const result = it?.result ?? it?.observation ?? it?.data ?? it?.error ?? it;

                // World Bank CSV pretty card
                if (it?.__type === 'mcp_result' && typeof it?.result?.text === 'string') {
                  return <WorldBankCSVCard key={`wb:${i}`} item={it} />;
                }


                if (tool === 'company_profile' && isProfile(result)) {
                  return <ProfileCardMini key={`${tool}:${i}`} data={result} />;
                }
                if (tool === 'recommendation_trends') {
                  const row = normalizeReco(result);
                  if (row) return <RecoTrendsCard key={`${tool}:${i}`} data={row} />;
                }
                if (tool === 'company_news') {
                  const news = normalizeNews(result);
                  return <NewsListCard key={`${tool}:${i}`} data={news} />;
                }

                // Fallback: compact raw payload with consistent label + collapsible details
                const rawKey = it?.name || it?.tool || 'tool';
                const displayName = labelForTool(rawKey);
                const ok = typeof it?.ok === 'boolean' ? it.ok : !it?.error;
                const elapsed = typeof it?.elapsed_ms === 'number' ? it.elapsed_ms : undefined;

                return (
                  <div key={`${rawKey}:${i}`} className="rounded-2xl border p-4 bg-white shadow-sm w-full h-full">
                    <div className="flex items-center gap-2 text-xs text-gray-600 mb-1">
                      <Badge>{displayName}</Badge>
                      {ok ? <Badge tone="ok">ok</Badge> : <Badge tone="err">error</Badge>}
                      {typeof elapsed === 'number' && <span>{elapsed.toFixed(0)} ms</span>}
                    </div>

                    <details className="text-xs">
                      <summary className="cursor-pointer text-gray-700">Show raw data</summary>
                      <pre className="text-[11px] bg-gray-50 rounded-lg p-2 overflow-auto max-h-48 max-w-full whitespace-pre-wrap break-words break-all mt-1">
                        {formatResult(result)}
                      </pre>
                    </details>
                  </div>
                );

              })}
            </div>
          </div>
        )}
      </div>
    </section>
  );
}

/* ---------- Helpers for action buttons ---------- */
async function fetchTool(kind: 'profile' | 'recommendation' | 'news', symbol: string) {
  const url =
    kind === 'profile'
      ? buildUrl(`/api/profile?symbol=${encodeURIComponent(symbol)}`)
      : kind === 'recommendation'
      ? buildUrl(`/api/reco-trends?symbol=${encodeURIComponent(symbol)}`)
      : buildUrl(`/api/news?symbol=${encodeURIComponent(symbol)}&days=7&limit=10`);

  const t0 = performance.now();
  let ok = true;
  let error: string | null = null;

  try {
    const res = await fetch(url, { cache: 'no-store' });
    if (!res.ok) {
      const text = await res.text().catch(() => '');
      throw new Error(text || `HTTP ${res.status}`);
    }
    const json = await res.json();

    const detail =
      kind === 'profile'
        ? { tool: 'company_profile', result: json }
        : kind === 'recommendation'
        ? { tool: 'recommendation_trends', result: json }
        : { tool: 'company_news', result: json };

    window.dispatchEvent(new CustomEvent('finassist:add-tool', { detail }));

    window.dispatchEvent(
      new CustomEvent('finassist:add-source', {
        detail: {
          type: 'tool',
          name: kind,
          title: kind === 'profile' ? 'Company Profile' : kind === 'recommendation' ? 'Analyst Trends' : 'Recent News',
          meta: { symbol },
        },
      })
    );
  } catch (e: any) {
    ok = false;
    error = e?.message || String(e);
    throw e;
  } finally {
    try {
      const ms = Math.round(performance.now() - t0);
      const sid = localStorage.getItem(ANALYTICS_SESSION_KEY) || undefined;
      const payload: any = {
        tool_name: kind,
        args: { symbol },
        latency_ms: ms,
        ok,
        error: error ?? null,
      };
      if (sid) payload.session_id = sid;
      if (FE_ANALYTICS) trackTool(payload);
    } catch {}
  }
}
