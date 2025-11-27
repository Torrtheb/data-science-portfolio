// src/components/CalcPanel.tsx
'use client';

import React, { useMemo, useState } from 'react';
import { postJSON, buildUrl } from '@/lib/backend';
import { logUi } from '@/lib/log';

type FieldType = 'number' | 'percent' | 'integer' | 'csv' | 'select';
type SelectOption = { label: string; value: string };

type FieldConfig = {
  key: string;
  label: string;
  type: FieldType;
  placeholder?: string;
  min?: number;
  step?: number;
  widthClass?: string;
  options?: SelectOption[];
  default?: string | number;
};

type CalcConfig = {
  key: string;
  label: string;
  endpoint: string; 
  method?: 'POST' | 'GET';
  fields: FieldConfig[];
};

type CurrencyCode = 'USD' | 'EUR' | 'CAD' | 'NONE';

const INPUT_BASE =
  'h-9 w-36 sm:w-40 px-2 text-sm rounded-lg border border-gray-300 focus:outline-none focus:ring-2 focus:ring-black/10';
const LABEL_BASE = 'text-xs text-gray-600';
const BTN_BASE =
  'h-9 px-3 text-sm rounded-lg bg-black text-white hover:opacity-90 disabled:opacity-50';
const CARD_BASE = 'rounded-2xl border shadow-sm bg-white p-3';
const HIDE_RESULT_KEYS = new Set([
  'markdown','summary','notes','explanation',
  'compounding_frequency','inflation_rate','rate','years'
]);
const NUMBER_LOCALE = 'en-US';

const CALCS: CalcConfig[] = [
  {
    key: 'simple-interest',
    label: 'Simple Interest',
    endpoint: '/api/calc/simple-interest',
    method: 'POST',
    fields: [
      { key: 'principal', label: 'Principal', type: 'number', min: 0.01, step: 0.01, default: 1000 },
      { key: 'rate_percent', label: 'Rate (%)', type: 'number', min: 0, placeholder: 'e.g. 7', default: 7 },
      { key: 'years', label: 'Years', type: 'number', min: 0.01, step: 0.01, default: 3 },
      { key: 'inflation_rate_percent', label: 'Inflation (%)', type: 'number', min: 0, placeholder: 'optional', default: '' },
    ],
  },
  {
    key: 'compound-interest',
    label: 'Compound Interest',
    endpoint: '/api/calc/compound-interest',
    method: 'POST',
    fields: [
      { key: 'principal', label: 'Principal', type: 'number', min: 0.01, step: 0.01, default: 1000 },
      { key: 'rate_percent', label: 'Rate (%)', type: 'number', min: 0, placeholder: 'e.g. 7', default: 7 },
      { key: 'years', label: 'Years', type: 'number', min: 0.01, step: 0.01, default: 3 },
      { key: 'compounding_per_year', label: 'Compounds/Year', type: 'integer', min: 1, step: 1, default: 12 },
      { key: 'inflation_rate_percent', label: 'Inflation (%)', type: 'number', min: 0, placeholder: 'optional', default: '' },
    ],
  },
  {
    key: 'cagr',
    label: '(CAGR) Compound Annual Growth Rate ',
    endpoint: '/api/calc/cagr',
    method: 'POST',
    fields: [
      { key: 'start_value', label: 'Start', type: 'number', min: 0, step: 0.01, default: 1000 },
      { key: 'end_value', label: 'End', type: 'number', min: 0, step: 0.01, default: 1500 },
      { key: 'years', label: 'Years', type: 'number', min: 0.01, step: 0.01, default: 3 },
    ],
  },
  {
    key: 'amortization',
    label: 'Loan Amortization',
    endpoint: '/api/calc/loan-amortization',
    method: 'POST',
    fields: [
      { key: 'principal', label: 'Principal', type: 'number', min: 0.01, step: 0.01, default: 300000 },
      { key: 'annual_rate_percent', label: 'Annual Rate (%)', type: 'number', min: 0, default: 5 },
      { key: 'years', label: 'Years', type: 'integer', min: 1, step: 1, default: 25 },
      { key: 'payments_per_year', label: 'Payments/Year', type: 'integer', min: 1, step: 1, default: 12 },
    ],
  },
  {
    key: 'npv',
    label: 'Net Present Value (NPV)',
    endpoint: '/api/calc/npv',
    method: 'POST',
    fields: [
      { key: 'rate_percent_per_period', label: 'Rate/Period (%)', type: 'number', min: 0, step: 0.01, default: 7 },
      {
        key: 'cashflows',
        label: 'Cashflows (CSV)',
        type: 'csv',
        placeholder: 'e.g. -1000, 300, 400, 500',
        widthClass: 'w-64 sm:w-72',
        default: '-1000, 300, 400, 500',
      },
    ],
  },
  {
    key: 'investment-return',
    label: 'Investment Return',
    endpoint: '/api/calc/investment-return',
    method: 'POST',
    fields: [
      { key: 'principal', label: 'Principal', type: 'number', min: 0.01, step: 0.01, default: 10000 },
      { key: 'rate_percent', label: 'Rate (%)', type: 'number', min: 0.0, default: 7 },
      { key: 'years', label: 'Years', type: 'number', min: 0.01, step: 0.01, default: 10 },
      { key: 'compounds_per_year', label: 'Compounds/Year', type: 'integer', min: 1, step: 1, default: 12 },
      { key: 'contribution_per_period', label: 'Contribution/Period', type: 'number', min: 0, step: 0.01, default: 200 },
      { key: 'contribution_frequency_per_year', label: 'Contribution Freq/Year', type: 'integer', min: 1, step: 1, default: 12 },
      {
        key: 'contribution_timing',
        label: 'Contribution Timing',
        type: 'select',
        options: [
          { label: 'End of Period', value: 'end' },
          { label: 'Beginning of Period', value: 'begin' },
        ],
        default: 'end',
      },
      { key: 'inflation_rate_percent', label: 'Inflation (%)', type: 'number', min: 0, default: '' },
    ],
  },
];

/* ---------- Utils ---------- */
// CSV validation/cleaning (for NPV)
const CSV_NUM_RE = /^\s*-?\d+(?:\.\d+)?(?:\s*,\s*-?\d+(?:\.\d+)?)*\s*$/;
function isValidCsvNumberList(s: string): boolean { return !!s.trim() && CSV_NUM_RE.test(s); }
function sanitizeCsv(s: string): string { return s.replace(/[;\n\r]+/g, ',').replace(/\s*,\s*/g, ', ').trim(); }
function parseCsvNumbers(s: string) { return s.split(',').map(t => parseFloat(t.trim())).filter(x => Number.isFinite(x)); }

function toNumberOrString(v: string, type: FieldType) {
  if (type === 'percent') return v.trim();
  if (type === 'integer') { const n = parseInt(v, 10); return Number.isFinite(n) ? n : 0; }
  if (type === 'number')  { const n = parseFloat(v);   return Number.isFinite(n) ? n : 0; }
  return v;
}

// money / percent key detection (plural-friendly, extra variants)
function isMoneyKey(k: string) {
  // value/amount/payment(s)/interest/contribution(s)/balance/npv/pv/fv/paid/principal/invest(ed)/return (not return_percentage)/loss/cost/fee
  return /(value|amount|payment(?!s_per_year)|payments|interest|contribution|contributions|balance|npv|pv|fv|paid|principal|invest(ed)?|return(?!_percentage)|loss|cost|fee)/i.test(k);
}
function isPercentKey(k: string) {
  // includes IRR/ROI just in case
  return /(percent|percentage|rate|cagr|return_percentage|irr|roi)/i.test(k);
}


// result ordering
const RESULT_PICKS: Record<string, string[]> = {
  'simple-interest': ['interest', 'total_amount', 'real_value', 'purchasing_power_loss'],
  'compound-interest': ['total_amount', 'interest', 'real_value', 'purchasing_power_loss'],
  cagr: ['cagr'],
  amortization: ['payment', 'monthly_payment', 'total_interest', 'total_paid', 'num_payments'],
  npv: ['npv'],
  'investment-return': ['future_value', 'total_invested', 'total_return', 'return_percentage', 'real_value', 'purchasing_power_loss'],
};

// friendly labels
const NICE_LABEL: Record<string, string> = {
  future_value: 'Future Value',
  total_amount: 'Future Value',
  present_value: 'Present Value',
  total_interest: 'Total Interest',
  interest: 'Interest',
  interest_earned: 'Interest Earned',
  total_invested: 'Total Invested',
  total_return: 'Total Return',
  return_percentage: 'Return (%)',
  monthly_payment: 'Monthly Payment',
  payment: 'Periodic Payment',
  num_payments: 'Number of Payments',
  cagr: 'CAGR',
  npv: 'Net Present Value',
  real_value: 'Inflation-Adjusted Future Value',
  purchasing_power_loss: 'Purchasing Power Loss',
};
function niceLabel(key: string) {
  return NICE_LABEL[key] || key.replace(/_/g, ' ').replace(/\b\w/g, m => m.toUpperCase());
}

const TIMEOUT_MS = 20_000;

/* ---------- Inline Stock Price helpers ---------- */
type QuoteData = {
  symbol?: string; price?: number; change?: number; changePercent?: number;
  currency?: string; exchange?: string; [k: string]: any;
};
function normalizePricePayload(js: any, fallback: string): QuoteData {
  return {
    symbol: String(js.symbol || js.ticker || fallback || '').toUpperCase(),
    price: typeof js.price === 'number' ? js.price : (typeof js.c === 'number' ? js.c : undefined),
    change: typeof js.change === 'number' ? js.change : (typeof js.d === 'number' ? js.d : undefined),
    changePercent: typeof js.changePercent === 'number' ? js.changePercent : (typeof js.dp === 'number' ? js.dp : js.change_pct),
    currency: js.currency || js.cur || js.ccy || '',
    exchange: js.exchange || js.exch || js.market || '',
    ...js,
  };
}

// Parse numbers out of strings like "CA$ 1 234,56", "$54,713.58", "1753.77"
function parseMoneyish(x: any): number | null {
  if (typeof x === 'number' && Number.isFinite(x)) return x;
  if (typeof x !== 'string') return null;
  let s = x.replace(/\u00A0|\u202F/g, ' ').trim(); 
  s = s.replace(/[^\d,\.\-\s]/g, '').replace(/\s+/g, '');
  if (s.includes(',') && s.includes('.')) s = s.replace(/,/g, '');
  else if (s.includes(',') && !s.includes('.')) s = s.replace(',', '.');
  if (!/^-?\d+(\.\d+)?$/.test(s)) return null;
  const n = parseFloat(s);
  return Number.isFinite(n) ? n : null;
}

function normalizeEquityQuery(raw: string): { query: string; note?: string } {
  const original = raw.trim();
  const s = original.toUpperCase().replace(/\s+/g, ' ');
  const squished = s.replace(/\s+/g, '');
  const m1 = s.match(/^(?:TSX|TSE|TO):\s*([A-Z.\-]+)$/); if (m1) return { query: `${m1[1]}.TO`, note: 'TSX symbol' };
  const m2 = s.match(/^([A-Z.\-]+):\s*(?:TSX|TSE|TO)$/); if (m2) return { query: `${m2[1]}.TO`, note: 'TSX symbol' };
  if (/[A-Z.\-]+\.TO$/.test(s)) return { query: s, note: 'TSX (.TO)' };
  const ALIASES: Record<string, string> = { 'AIR CANADA': 'AC.TO','AIRCANADA': 'AC.TO','RBC': 'RY.TO','ROYAL BANK': 'RY.TO','SHOPIFY': 'SHOP.TO' };
  if (ALIASES[s]) return { query: ALIASES[s], note: 'Alias→TSX' };
  if (ALIASES[squished]) return { query: ALIASES[squished], note: 'Alias→TSX' };
  return { query: original };
}

async function fetchJsonWithTimeout(url: string, timeoutMs = TIMEOUT_MS) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(url, { cache: 'no-store', signal: controller.signal });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } finally { clearTimeout(timer); }
}
async function fetchQuoteInline(query: string) {
  const { query: q } = normalizeEquityQuery(query);
  const endpoints = [ buildUrl(`/api/price?q=${encodeURIComponent(q)}`), buildUrl(`/api/quote?q=${encodeURIComponent(q)}`) ];
  for (const url of endpoints) {
    try { const js = await fetchJsonWithTimeout(url); const data = normalizePricePayload(js, q);
      if (typeof data.price === 'number') return { ...data, _resolved: q }; } catch {}
  }
  throw new Error('price_lookup_failed');
}

function PriceCard({ data }: { data: QuoteData & { _resolved?: string } }) {
  const priceStr = typeof data.price === 'number'
    ? Number(data.price).toLocaleString(undefined, { maximumFractionDigits: 4 })
    : '—';
  const ch = typeof data.change === 'number' ? data.change : null;
  const cp = typeof data.changePercent === 'number' ? data.changePercent : null;
  const delta = ch !== null && cp !== null ? `${ch >= 0 ? '▲' : '▼'} ${Math.abs(ch).toFixed(2)} (${cp.toFixed(2)}%)` : '';
  const curr = data.currency ? ` ${data.currency}` : '';
  const sym = (data.symbol || data._resolved || '').toUpperCase();
  return (
    <div className="rounded-xl border bg-gray-50 p-3">
      <div className="text-xs text-gray-600 mb-1">Stock price</div>
      <div className="text-sm font-semibold">
        {sym || '—'} <span className="font-normal">·</span> {priceStr}
        <span className="text-gray-600">{curr}</span>
      </div>
      {delta && <div className={`text-xs ${ch! >= 0 ? 'text-green-600' : 'text-red-600'}`}>{delta}</div>}
      <div className="text-[11px] text-gray-500 mt-1">
        {data.exchange ? `Exchange: ${data.exchange}` : null}
        {data._resolved && data._resolved !== sym ? (<span className="ml-2">Resolved as <code>{data._resolved}</code></span>) : null}
      </div>
    </div>
  );
}

function CalcSelect({
  value,
  onChange,
  options,
  label = 'Select calculator',
}: {
  value: string;
  onChange: (v: string) => void;
  options: { label: string; value: string }[];
  label?: string;
}) {
  const [open, setOpen] = React.useState(false);
  const btnRef = React.useRef<HTMLButtonElement | null>(null);

  React.useEffect(() => {
    function onDocClick(e: MouseEvent) {
      if (!btnRef.current) return;
      const target = e.target as Node;
      if (btnRef.current.contains(target)) return;
      const menu = document.getElementById('calc-select-menu');
      if (menu && menu.contains(target)) return;
      setOpen(false);
    }
    document.addEventListener('mousedown', onDocClick);
    return () => document.removeEventListener('mousedown', onDocClick);
  }, []);

  const current = options.find(o => o.value === value);

  return (
    <div className="relative">
      <span className="sr-only">{label}</span>
      <button
        ref={btnRef}
        type="button"
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-label={label}
        onClick={() => setOpen(o => !o)}
        className={`
          ${INPUT_BASE}
          ${['cagr', 'npv'].includes(value) ? 'h-11 w-64 sm:w-72' : 'h-9 w-36 sm:w-40'}
          px-3 text-left whitespace-normal break-words leading-snug
        `}
      >
        <div className="line-clamp-2">{current?.label ?? 'Choose…'}</div>
      </button>

      {open && (
        <div
          id="calc-select-menu"
          role="listbox"
          className="absolute z-20 mt-1 min-w-full max-w-[90vw] rounded-xl border bg-white shadow-lg max-h-64 overflow-auto"
        >
          {options.map(opt => {
            const active = opt.value === value;
            return (
              <div
                key={opt.value}
                role="option"
                aria-selected={active}
                onClick={() => { onChange(opt.value); setOpen(false); }}
                className={`px-3 py-2 cursor-pointer whitespace-normal break-words ${active ? 'bg-gray-100' : 'hover:bg-gray-50'}`}
              >
                <div className="text-sm leading-snug">{opt.label}</div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}


/* ---------- Component ---------- */
export default function CalcPanel() {
  const firstCalc = CALCS.length > 0 ? CALCS[0] : undefined;
  const [activeKey, setActiveKey] = useState<string>(firstCalc ? firstCalc.key : '');
  const [values, setValues] = useState<Record<string, string>>(buildInitialValues(firstCalc));
  const [lastPayload, setLastPayload] = useState<Record<string, any> | null>(null);
  const [currency, setCurrency] = useState<CurrencyCode>('USD');

  const active = useMemo<CalcConfig | undefined>(() => CALCS.find(c => c.key === activeKey), [activeKey]);

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const [stockQuery, setStockQuery] = useState<string>('');
  const [stockLoading, setStockLoading] = useState<boolean>(false);
  const [stockError, setStockError] = useState<string | null>(null);
  const [stockResult, setStockResult] = useState<(QuoteData & { _resolved?: string }) | null>(null);
  const [csvError, setCsvError] = useState<string | null>(null);

  function changeCalc(newKey: string) {
    setActiveKey(newKey);
    const cfg = CALCS.find((c) => c.key === newKey);
    setValues(buildInitialValues(cfg));
    setResult(null);
    setError(null);
    setLastPayload(null);
  }

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!active) return;
    setLoading(true);
    setError(null);
    setResult(null);

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);

    if (active.key === 'npv' && csvError) {
      setLoading(false);
      setError('Please fix the cashflows input before calculating.');
      return;
    }

    try {
      const payload: Record<string, any> = {};
      for (const f of active.fields) {
        const rawVal = (values[f.key] ?? '').toString();
        if (rawVal.trim() === '') continue;
        if (f.type === 'csv') payload[f.key] = parseCsvNumbers(rawVal);
        else if (f.type === 'select') payload[f.key] = rawVal;
        else payload[f.key] = toNumberOrString(rawVal, f.type);
      }
      for (const [k, v] of Object.entries(payload)) {
        if (typeof v === 'number' && !Number.isFinite(v)) throw new Error(`Invalid number for "${k}".`);
      }
      setLastPayload(payload);

      const path = active.endpoint.replace(/^\/api/, '');
      const json = await postJSON<any>(path, payload, { signal: controller.signal });
      const data = json?.data ?? json;

      if (data && typeof data === 'object') {
        for (const k of ['markdown', 'summary', 'notes', 'explanation']) if (k in data) delete (data as any)[k];
      }

      setResult(data);

      if (json?.source) {
        window.dispatchEvent(new CustomEvent('finassist:add-source', { detail: json.source }));
      }
    } catch (err: any) {
      const msg = err?.name === 'AbortError' ? 'Request timed out.' : (err?.message ?? String(err));
      setError(msg);
      void logUi({ level: 'error', msg: 'calc_submit_failed', meta: { calc: active.key, err: msg } });
    } finally {
      clearTimeout(timer);
      setLoading(false);
    }
  }

  async function onStockSubmit(e: React.FormEvent) {
    e.preventDefault();
    const q = (stockQuery || '').trim();
    if (!q) return;
    setStockLoading(true);
    setStockError(null);
    setStockResult(null);
    try { setStockResult(await fetchQuoteInline(q)); }
    catch { setStockError('Could not fetch a live quote.'); }
    finally { setStockLoading(false); }
  }

  if (!active) {
    return (<div className={CARD_BASE}><div className="text-sm text-gray-500">No calculators configured.</div></div>);
  }

  // ---- currency-aware formatters (depend on `currency`) ----
  const fmtCurrency = (n: number) => {
    const safe = Number.isFinite(n) ? n : 0;
    if (currency === 'NONE') {
      return new Intl.NumberFormat(NUMBER_LOCALE, {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
      }).format(safe);
    }
    const base = new Intl.NumberFormat(NUMBER_LOCALE, {
      style: 'currency',
      currency,
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(safe);
    return `${base} ${currency}`;
  };


  const fmtPercent = (n: number) =>
    new Intl.NumberFormat(NUMBER_LOCALE, { style: 'percent', minimumFractionDigits: 2, maximumFractionDigits: 2 }).format(n / 100);

  const fmtNumber = (n: number) =>
    new Intl.NumberFormat(NUMBER_LOCALE, { minimumFractionDigits: 2, maximumFractionDigits: 2 }).format(Number.isFinite(n) ? n : 0);

  // fields that are always currency for specific calcs
  const FORCE_MONEY: Record<string, Set<string>> = {
    amortization: new Set(['payment','monthly_payment','total_interest','total_paid']),
    'investment-return': new Set([
      'future_value','total_invested','total_return','real_value','purchasing_power_loss',
      'total_contributions', 'ending_balance', 'starting_balance'
    ]),
    'simple-interest': new Set(['interest','total_amount','real_value','purchasing_power_loss']),
    'compound-interest': new Set(['interest','total_amount','real_value','purchasing_power_loss']),
  };


  // percent-only fields
  const FORCE_PERCENT: Record<string, Set<string>> = {
    'investment-return': new Set(['return_percentage']),
    cagr: new Set(['cagr']),
  };

  // ---------- THE function you were unsure about: keep it ONCE, here ----------
  function buildDisplayRows(
    calcKey: string,
    payload: Record<string, any>,
    _result: Record<string, any>
  ) {
    const picks = RESULT_PICKS[calcKey] || [];
    const rows: Array<{ label: string; value: string }> = [];
    const payloadKeys = new Set(Object.keys(payload || {}));

    const moneyForced = FORCE_MONEY[calcKey] ?? new Set<string>();
    const percentForced = FORCE_PERCENT[calcKey] ?? new Set<string>();

    const toNumeric = (v: any): number | null => {
      if (typeof v === 'number' && Number.isFinite(v)) return v;
      if (typeof v === 'string') {
        // quick path for simple numeric strings
        const simple = v.trim();
        if (/^-?\d+(\.\d+)?$/.test(simple)) {
          const n = parseFloat(simple);
          return Number.isFinite(n) ? n : null;
        }
        // fallback to robust parser that handles "$1,234", "CA$ 1 234,56", etc.
        const n2 = parseMoneyish(v);
        return n2;
      }
      return null;
    };

    const addRow = (k: string, v: any) => {
      if (payloadKeys.has(k) || v == null || HIDE_RESULT_KEYS.has(k)) return;

      const label = niceLabel(k);
      const n = toNumeric(v);
      const moneyOverride = moneyForced.has(k) || isMoneyKey(k);
      const percentOverride = percentForced.has(k) || isPercentKey(k);

      if (n !== null) {
        // If it’s a number, route by type
        const formatted = percentOverride ? fmtPercent(n) : (moneyOverride ? fmtCurrency(n) : fmtNumber(n));
        rows.push({ label, value: formatted });
        return;
      }

      // Non-numeric string → try money one last time, else keep as-is
      if (typeof v === 'string' && v.trim() !== '') {
        const fallbackNum = parseMoneyish(v);
        if (fallbackNum !== null && moneyOverride) {
          rows.push({ label, value: fmtCurrency(fallbackNum) });
        } else {
          rows.push({ label, value: v });
        }
      }
    };

    // Add preferred fields first
    for (const k of picks) if (k in _result) addRow(k, _result[k]);

    // Fill extras (avoid arrays/objects)
    if (rows.length < 4) {
      for (const [k, vv] of Object.entries(_result)) {
        if (picks.includes(k) || payloadKeys.has(k) || HIDE_RESULT_KEYS.has(k)) continue;
        if (Array.isArray(vv) || (typeof vv === 'object' && vv !== null)) continue;
        addRow(k, vv);
        if (rows.length >= 6) break;
      }
    }
    return rows;
  }


  // ---------------------------------------------------------------------------

  const rows = result && lastPayload ? buildDisplayRows(active.key, lastPayload, result) : [];

  return (
    <div className={CARD_BASE}>
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 mb-3">
        <div className="font-semibold">{active.label}</div>
        <div className="flex items-center gap-2">
          <label className={LABEL_BASE}>Select</label>
            <CalcSelect
              value={activeKey}
              onChange={changeCalc} 
              options={CALCS.map(c => ({ label: c.label, value: c.key }))}
            />

          <label className={`${LABEL_BASE} ml-2`}>Currency</label>
          <select
            value={currency}
            onChange={(e) => setCurrency(e.target.value as CurrencyCode)}
            className={`${INPUT_BASE} h-9 w-32`}
            aria-label="Select currency"
          >
            <option value="USD">USD</option>
            <option value="EUR">EUR</option>
            <option value="CAD">CAD</option>
            <option value="NONE">None</option>
          </select>
        </div>
      </div>

      {/* Dynamic Form */}
      <form onSubmit={onSubmit} className="space-y-3">
        <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
          {active.fields.map((f) => {
            const value = values[f.key] ?? '';
            const commonProps = {
              value,
              onChange: (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) =>
                setValues((p) => ({ ...p, [f.key]: e.target.value })),
              className: `${INPUT_BASE} ${f.widthClass ?? ''}`,
              placeholder: f.placeholder,
            } as const;

            if (f.type === 'select') {
              return (
                <div key={f.key} className="flex flex-col gap-1">
                  <label className={LABEL_BASE}>{f.label}</label>
                  <select {...(commonProps as any)}>
                    {(f.options ?? []).map((opt) => (
                      <option key={opt.value} value={opt.value}>{opt.label}</option>
                    ))}
                  </select>
                </div>
              );
            }

            if (f.type === 'csv') {
              const isNpvCashflows = active.key === 'npv' && f.key === 'cashflows';
              return (
                <div key={f.key} className="flex flex-col gap-1">
                  <label className={LABEL_BASE}>{f.label}</label>
                  <input
                    type="text"
                    value={value}
                    onChange={(e) => {
                      const raw = sanitizeCsv(e.target.value);
                      setValues((p) => ({ ...p, [f.key]: raw }));
                      if (isNpvCashflows) {
                        setCsvError(isValidCsvNumberList(raw) ? null :
                          'Use only numbers separated by commas (e.g., -1000, 300, 400, 500).');
                      }
                    }}
                    onBlur={(e) => {
                      const raw = sanitizeCsv(e.target.value);
                      if (isNpvCashflows && !isValidCsvNumberList(raw)) {
                        setCsvError('Use only numbers separated by commas (e.g., -1000, 300, 400, 500).');
                      }
                    }}
                    inputMode="decimal"
                    spellCheck={false}
                    autoCorrect="off"
                    className={`${INPUT_BASE} ${f.widthClass ?? ''} ${
                      isNpvCashflows && csvError ? 'border-red-400 focus:ring-red-200' : ''
                    }`}
                    placeholder={f.placeholder}
                  />
                  {isNpvCashflows && csvError && (
                    <div className="text-[11px] text-red-600">{csvError}</div>
                  )}
                </div>
              );
            }

            if (f.type === 'percent') {
              return (
                <div key={f.key} className="flex flex-col gap-1">
                  <label className={LABEL_BASE}>{f.label}</label>
                  <input type="text" {...(commonProps as any)} />
                </div>
              );
            }

            return (
              <div key={f.key} className="flex flex-col gap-1">
                <label className={LABEL_BASE}>{f.label}</label>
                <input type="number" min={f.min as number | undefined} step={f.step as number | undefined} {...(commonProps as any)} />
              </div>
            );
          })}
        </div>

        <div className="flex items-center gap-2">
          <button className={BTN_BASE} disabled={loading}>{loading ? 'Calculating…' : 'Calculate'}</button>
          {error && <div className="text-xs text-red-600 bg-red-50 border border-red-200 rounded-lg px-2 py-1">{error}</div>}
        </div>
      </form>

      {/* Formatted results */}
      {rows.length > 0 && (
        <div className="mt-4 space-y-2">
          <div className="text-xs text-gray-600">Result</div>
          <div className="rounded-xl border bg-gray-50 p-3">
            {active.key === 'cagr' ? (
              <dl className="space-y-2">
                {rows.map((r) => (
                  <div key={r.label} className="flex flex-col sm:flex-row sm:justify-between sm:items-center gap-1">
                    <dt className="text-xs text-gray-600">{r.label}</dt>
                    <dd className="text-sm font-medium break-words">{r.value}</dd>
                  </div>
                ))}
              </dl>
            ) : (
              <dl className="grid grid-cols-1 sm:grid-cols-2 gap-y-2 gap-x-6">
                {rows.map((r) => (
                  <div key={r.label} className="flex items-center justify-between">
                    <dt className="text-xs text-gray-600">{r.label}</dt>
                    <dd className="text-sm font-medium">{r.value}</dd>
                  </div>
                ))}
              </dl>
            )}
          </div>
        </div>
      )}

      {/* Raw JSON fallback */}
      {result && rows.length === 0 && (
        <div className="mt-4">
          <div className="text-xs text-gray-600 mb-1">Result</div>
          <pre className="text-xs bg-gray-50 rounded-xl border p-2 overflow-auto max-h-64">
            {JSON.stringify(result, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}

// helpers used above
function buildInitialValues(cfg: CalcConfig | undefined) {
  const init: Record<string, string> = {};
  if (!cfg) return init;
  for (const f of cfg.fields) init[f.key] = String(f.default ?? '');
  return init;
}
