// src/lib/types.ts

export type SourceItem = {
  snippet?: string;
  provider?: string;
  id?: number;
  n?: number;
  title?: string;
  url?: string;
};

export type ToolItem = {
  /** Optional standardized fields for richer tool rendering */
  name?: string;
  ok?: boolean;
  elapsed_ms?: number;
  tool?: string;
  args?: unknown;
  observation?: string;
  result?: unknown;
  error?: string;
};

export type ChatMessage = {
  role: 'system' | 'user' | 'assistant';
  content: string;
};

export type ChatResponse = {
  answer: string;
  sources: SourceItem[];
  tools: ToolItem[];
};

// --- Streaming frames (if/when the backend emits structured SSE frames) ---
export type ChatStreamToken = { type: 'token'; content: string };
export type ChatStreamDone = { type: 'done'; sources: SourceItem[]; tools: ToolItem[] };
export type ChatStreamError = { type: 'error'; message: string };
export type ChatStreamFrame = ChatStreamToken | ChatStreamDone | ChatStreamError;

// --- Finance data ---
export type PriceSnapshot = {
  symbol: string;
  price: number | null;
  open: number | null;
  high: number | null;
  low: number | null;
  prev_close: number | null;
  change: number | null;
  change_pct: number | null;
  ts: number | null;
};

export type CandlePayload = {
  c: number[]; h: number[]; l: number[]; o: number[]; t: number[]; v: number[];
  s?: string; error?: string; note?: string; 
};

// --- Calculator request bodies (if you call your backend calculators) ---
export type SimpleInterestBody = {
  principal: number;
  rate: number | string; 
  years: number;
  inflation_rate_percent?: number | string;
};

export type CompoundInterestBody = {
  principal: number;
  rate: number | string;
  years: number;
  compounds_per_year?: number;
  contribution_per_period?: number;
  contribution_frequency_per_year?: number;
  contribution_timing?: 'end' | 'begin';
  inflation_rate_percent?: number | string;
};

export type AmortizationBody = {
  principal: number;
  annual_rate: number | string;
  years: number;
  payments_per_year?: number;
};

export type InvestmentReturnBody = {
  principal: number;
  annual_rate_percent: number | string;
  years: number;
};

export type NpvBody = {
  rate_percent_per_period: number | string;
  cashflows: number[];
};

// --- Generic API error shape for consistent UI rendering ---
export type ApiError = {
  status?: number;
  error?: string;
  detail?: string;
  path?: string;
};

