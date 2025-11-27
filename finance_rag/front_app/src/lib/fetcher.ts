// src/lib/fetcher.ts

const API_BASE =
  (process.env.NEXT_PUBLIC_API_BASE_URL ||
    process.env.NEXT_PUBLIC_BACKEND_URL ||
    '').replace(/\/+$/, ''); 

function apiUrl(path: string): string {
  // Normalize input
  const p = path.startsWith('/') ? path : `/${path}`;

  if (p.startsWith('/api/') || p.startsWith('/_debug') || p === '/openapi.json') {
    return `${API_BASE}${p}`;
  }
  return `${API_BASE}/api${p}`;
}

// ---- Basic helpers ----------------------------------------------------------
export async function fetcher<T = any>(url: string): Promise<T> {
  const full = url.startsWith('http') ? url : apiUrl(url);
  const res = await fetch(full, { cache: 'no-store' });
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
  return res.json();
}

export class HttpError extends Error {
  status: number;
  payload: any;
  constructor(message: string, status: number, payload?: any) {
    super(message);
    this.status = status;
    this.payload = payload;
  }
}

export async function fetchJSON<T = any>(
  url: string,
  init: RequestInit = {},
  timeoutMs = 20000,
  retries = 1
): Promise<T> {
  const full = url.startsWith('http') ? url : apiUrl(url);
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(full, { ...init, signal: controller.signal, cache: 'no-store' });
    const text = await res.text();
    let json: any = null;
    try { json = text ? JSON.parse(text) : null; } catch { /* ignore parse errors */ }
    if (!res.ok) throw new HttpError(`HTTP ${res.status}`, res.status, json ?? text);
    return json as T;
  } catch (e: any) {
    if (retries > 0 && (e?.name === 'AbortError' || (e?.status && e.status >= 500))) {
      return fetchJSON<T>(url, init, timeoutMs, retries - 1);
    }
    throw e;
  } finally {
    clearTimeout(id);
  }
}

// Convenience POST/GET that use apiUrl()
export async function postJSON<T = any>(
  path: string,
  body?: any,
  headers?: Record<string, string>
) {
  const res = await fetch(apiUrl(path), {
    method: 'POST',
    headers: { 'content-type': 'application/json', ...(headers ?? {}) },
    body: body == null ? null : JSON.stringify(body),
    cache: 'no-store',
  } satisfies RequestInit);
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
  return res.json() as Promise<T>;
}

export async function getJSON<T = any>(
  path: string,
  headers?: Record<string, string>
) {
  const res = await fetch(apiUrl(path), {
    headers: headers ?? {},
    cache: 'no-store',
  } satisfies RequestInit);
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
  return res.json() as Promise<T>;
}


// ---- Chat endpoints ---------------------------------------------------------
export async function postChatJSON(
  body: { question?: string; messages?: Array<{ role: 'user'|'assistant'|'system'|'tool'; content: string; tool_calls?: any[] }> },
  headers?: Record<string,string>
) {
  return postJSON('/chat', body, headers);
}

export async function chat(question: string) {
  return postChatJSON({ question });
}

// ---- Calculators (all under /api/calc/*) -----------------------------------
export async function calcCAGR(begin: number, end: number, years: number) {
  return postJSON('/calc/cagr', { begin, end, years }) as Promise<{ data: { cagr: number } }>;
}

export async function calcPMT(rate: number, periods: number, pv: number, fv = 0, when: 'begin'|'end'='end') {
  return postJSON('/calc/pmt', { rate, periods, pv, fv, when }) as Promise<{ data: { pmt: number } }>;
}

export async function calcFV(pv: number, rate: number, periods: number, pmt = 0, when: 'begin'|'end'='end') {
  return postJSON('/calc/fv', { pv, rate, periods, pmt, when }) as Promise<{ data: { fv: number } }>;
}

export async function calcNPV(rate: number, cashflows: number[]) {
  return postJSON('/calc/npv', { rate, cashflows }) as Promise<{ data: { npv: number } }>;
}

// ---- Generic calc helper + extras ------------------------------------------
export type PercentLike = number | string;
export type SimpleInterestPayload = { principal: number | string; rate_percent: PercentLike; years: number; inflation_rate_percent?: PercentLike; };
export type CompoundInterestPayload = { principal: number | string; rate_percent: PercentLike; years: number; compounding_per_year: number; inflation_rate_percent?: PercentLike; };
export type InvestmentReturnPayload = {
  principal: number | string;
  rate_percent: PercentLike;
  years: number;
  compounds_per_year: number;
  contribution_per_period: number;
  contribution_frequency_per_year: number;
  contribution_timing: 'end' | 'begin';
  inflation_rate_percent?: PercentLike;
};
export type InvestmentReturnStringsPayload = {
  principal: number | string;
  rate_percent: PercentLike;
  years: number;
  compound?: string;
  regular_addition?: number;
  regular_addition_every?: string;
  addition_timing?: 'end' | 'begin';
  inflation_rate_percent?: PercentLike;
};
export type LoanAmortizationPayload = {
  principal: number | string;
  annual_rate_percent: PercentLike;
  years: number;
  payments_per_year?: number;
};

export type ToolSource = { type: 'tool'; name: string; title?: string; meta?: Record<string, any> };
export type ToolResult<TData = any> = { tool: string; ok: boolean; data: TData & { markdown?: string }; source?: ToolSource };

export const postCalc = <T = any>(path: string, body: any) =>
  postJSON<T>(path.startsWith('/calc/') ? path : `/calc/${path.replace(/^\/+/, '')}`, body);

export const calcSimpleInterest        = (payload: SimpleInterestPayload)        => postCalc<ToolResult>('/calc/simple-interest', payload);
export const calcCompoundInterest      = (payload: CompoundInterestPayload)      => postCalc<ToolResult>('/calc/compound-interest', payload);
export const calcInvestmentReturn      = (payload: InvestmentReturnPayload)      => postCalc<ToolResult>('/calc/investment-return', payload);
export const calcInvestmentReturnStrings = (payload: InvestmentReturnStringsPayload) => postCalc<ToolResult>('/calc/investment-return-strings', payload);
export const calcLoanAmortization      = (payload: LoanAmortizationPayload)      => postCalc<ToolResult>('/calc/loan-amortization', payload);

// ---- MCP helpers ------------------------------------------------------------
export async function fetchMcpTools(serverKey: string) {
  try {
    const q = serverKey ? `?server_key=${encodeURIComponent(serverKey)}` : '';
    return await getJSON<{ tools: any[] }>(`/mcp/tools${q}`);
  } catch (e: any) {
    if (typeof e?.message === 'string' && /404/.test(e.message)) {
      const dbg = await getJSON<any>('/_debug/mcp_ping');
      return { tools: (dbg?.sample ?? []).map((t: any) => ({ name: t?.name, ...t })) };
    }
    throw e;
  }
}

export async function callMcpTool(serverKey: string, name: string, args: any) {
  try {
    return await postJSON<{ result: any; sources?: any[] }>(
      '/mcp/call',
      { server_key: serverKey, tool: name, arguments: args }
    );
  } catch (e: any) {
    if (typeof e?.message === 'string' && /404/.test(e.message)) {
      return await postJSON<{ result: any; sources?: any[] }>(
        '/_debug/mcp_call',
        { server_key: serverKey, name, arguments: args }
      );
    }
    throw e;
  }
}
