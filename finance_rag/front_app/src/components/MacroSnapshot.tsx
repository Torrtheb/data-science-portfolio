// src/components/MacroSnapshot.tsx
'use client';

import React, { useMemo, useState } from 'react';
import { callMcpTool } from '@/lib/fetcher';

type Props = {
  addItem: (item: any) => void;
  addSources: (srcs: any[]) => void;
};

const TOOL_NAME = 'get_indicator_for_country';
const MIN_YEAR = 1960;
const MAX_YEAR = 2024;

export default function MacroSnapshot({ addItem, addSources }: Props) {
  const [countryId, setCountryId] = useState('Canada');         
  const [indicatorId, setIndicatorId] = useState('FP.CPI.TOTL.ZG');
  const [startYear, setStartYear] = useState<number>(2015);
  const [endYear, setEndYear] = useState<number>(2024);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // --- validation helpers ---
  const countryValid = useMemo(() => {
    const s = (countryId || '').trim();
    return /^[A-Za-zÀ-ÖØ-öø-ÿ .,'-]{2,60}$/.test(s);
  }, [countryId]);

  const startOutOfRange = startYear < MIN_YEAR || startYear > MAX_YEAR;
  const endOutOfRange   = endYear   < MIN_YEAR || endYear   > MAX_YEAR;

  // prepare a normalized, clamped range for submission
  const clampedStart = Math.max(MIN_YEAR, Math.min(startYear || MIN_YEAR, MAX_YEAR));
  const clampedEnd   = Math.max(MIN_YEAR, Math.min(endYear   || MAX_YEAR, MAX_YEAR));
  const normalizedStart = Math.min(clampedStart, clampedEnd);
  const normalizedEnd   = Math.max(clampedStart, clampedEnd);
  const yearsValid = !startOutOfRange && !endOutOfRange;

  const canSubmit = countryValid && yearsValid && !loading;

  async function run() {
    setLoading(true);
    setError(null);
    try {
      if (!countryValid) {
        throw new Error('Please enter a valid country name (e.g., China, United States).');
      }
      if (!yearsValid) {
        throw new Error(`Years out of range. Please choose ${MIN_YEAR}–${MAX_YEAR}.`);
      }

      const country = countryId.trim();

      const r = await callMcpTool('world_bank', TOOL_NAME, {
        country_id: country,
        indicator_id: indicatorId,
      });

      // Try to detect “country not recognized” cases from the MCP result payload
      const text: string = r?.result?.text ?? '';
      const looksMissing =
        !text ||
        /country\s+not\s+found|invalid\s+country|not\s+recognized|no\s+data\s+found/i.test(text);

      if (looksMissing) {
        // bubble a specific error we can show nicely
        const err = new Error('Country not recognized');
        (err as any).__kind = 'COUNTRY_NOT_RECOGNIZED';
        throw err;
      }

      // Submit with normalized range
      addItem({
        __type: 'mcp_result',
        serverKey: 'world_bank',
        tool: TOOL_NAME,
        result: r.result,
        meta: {
          startYear: normalizedStart,
          endYear: normalizedEnd,
          countryId: country,
          indicatorId,
        },
      });

      if (Array.isArray(r.sources) && r.sources.length) addSources(r.sources);
    } catch (e: any) {
      // Friendly messages
      if (e?.__kind === 'COUNTRY_NOT_RECOGNIZED') {
        setError('Country not recognized. Try a full name like “United States” instead of “USA”.');
      } else {
        const msg =
          typeof e?.message === 'string'
            ? e.message
            : 'Failed to fetch World Bank data. Please check the country and try again.';
        setError(msg);
      }
    } finally {
      setLoading(false);
    }
  }

  // helpers to clamp on blur so the field never stays invalid
  const clampStartOnBlur = () => {
    if (startOutOfRange) setStartYear(Math.max(MIN_YEAR, Math.min(startYear || MIN_YEAR, MAX_YEAR)));
  };
  const clampEndOnBlur = () => {
    if (endOutOfRange) setEndYear(Math.max(MIN_YEAR, Math.min(endYear || MAX_YEAR, MAX_YEAR)));
  };

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-end gap-3">
        {/* Country input */}
        <div className="flex flex-col">
          <label className="text-xs text-gray-500 mb-1">Country</label>
          <input
            className={`border rounded px-3 py-2 text-sm w-52 md:w-72 ${!countryValid ? 'border-red-400' : ''}`}
            value={countryId}
            onChange={(e)=>setCountryId(e.target.value)}
            placeholder="e.g., China, United States, Côte d’Ivoire"
          />
          {!countryValid && (
            <div className="text-[11px] text-red-600 mt-1">
              Please enter a valid country name (2–60 chars; letters, spaces, . , ' -).
            </div>
          )}
        </div>

        {/* Indicator select */}
        <div className="flex flex-col">
          <label className="text-xs text-gray-500 mb-1">Indicator</label>
          <select
            className="border rounded px-3 py-2 text-sm min-w-72"
            value={indicatorId}
            onChange={(e)=>setIndicatorId(e.target.value)}
            title="World Bank indicator_id"
          >
            <option value="FP.CPI.TOTL.ZG">Inflation (CPI, % y/y)</option>
            <option value="SL.UEM.TOTL.ZS">Unemployment rate (%)</option>
            <option value="NY.GDP.PCAP.PP.KD">GDP per capita (PPP, const)</option>
          </select>
        </div>

        {/* Year range (hard-guarded to 1960–2024) */}
        <div className="flex flex-col">
          <label className="text-xs text-gray-500 mb-1">Start year</label>
          <input
            type="number"
            className={`border rounded px-3 py-2 text-sm w-28 ${startOutOfRange ? 'border-red-400' : ''}`}
            value={startYear}
            min={MIN_YEAR}
            max={MAX_YEAR}
            onChange={(e)=>setStartYear(Number(e.target.value || MIN_YEAR))}
            onBlur={clampStartOnBlur}
          />
          {startOutOfRange && (
            <div className="text-[11px] text-red-600 mt-1">
              Years out of range. Use {MIN_YEAR}–{MAX_YEAR}.
            </div>
          )}
        </div>
        <div className="flex flex-col">
          <label className="text-xs text-gray-500 mb-1">End year</label>
          <input
            type="number"
            className={`border rounded px-3 py-2 text-sm w-28 ${endOutOfRange ? 'border-red-400' : ''}`}
            value={endYear}
            min={MIN_YEAR}
            max={MAX_YEAR}
            onChange={(e)=>setEndYear(Number(e.target.value || MAX_YEAR))}
            onBlur={clampEndOnBlur}
          />
          {endOutOfRange && (
            <div className="text-[11px] text-red-600 mt-1">
              Years out of range. Use {MIN_YEAR}–{MAX_YEAR}.
            </div>
          )}
        </div>

        <button
          onClick={run}
          disabled={!canSubmit}
          className="rounded-lg border px-4 py-2 text-sm hover:bg-gray-50 disabled:opacity-50"
        >
          {loading ? 'Loading…' : 'Fetch data'}
        </button>
      </div>

      <div className="text-xs text-gray-500">
        Using tool:&nbsp;<code>{TOOL_NAME}</code>
      </div>

      {error && <div className="text-sm text-red-600">{error}</div>}
    </div>
  );
}
