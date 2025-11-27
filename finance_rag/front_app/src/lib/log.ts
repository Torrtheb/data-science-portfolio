// src/lib/log.ts
import { buildUrl } from '@/lib/backend';

export type UiLog = {
  level: 'info' | 'warn' | 'error';
  msg: string;
  meta?: Record<string, unknown>;
};

export async function logUi(event: UiLog) {
  try {
    await fetch(buildUrl('/api/log'), {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ ...event, t: Date.now() }),
      keepalive: true,
    });
  } catch {
  }
}
