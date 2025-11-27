// src/lib/analytics.ts
import { buildUrl } from '@/lib/backend';

async function post(path: string, payload: any) {
  try {
    await fetch(buildUrl(path), {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      cache: 'no-store',
      body: JSON.stringify(payload),
    });
  } catch {
  }
}

export async function track(payload: any) {
  return post('/api/analytics/ingest', payload);
}

/** Optional helpers (nice to have) */
export const trackTurn = (p: {
  session_id?: string;
  role?: 'assistant' | 'user' | 'system';
  content?: string;
  model?: string;
  tokens_in?: number;
  tokens_out?: number;
  cost_usd?: number;
  latency_ms?: number;
  had_rag?: boolean;
  tools_used?: any[];
  error?: string | null;
}) => track({ type: 'turn', ...p });

export const trackTool = (p: {
  session_id?: string;
  tool_name: string;
  args?: Record<string, any>;
  latency_ms?: number;
  ok?: boolean;
  error?: string | null;
}) => track({ type: 'tool', ...p });

export const SESSION_KEY = 'finassist.sessionId.v1';
