// src/app/page.tsx
'use client';
import { useEffect, useState } from 'react';
import ChatPro from '@/components/ChatPro';
import CalcPanel from '@/components/CalcPanel';
import StockSearchBox from '@/components/StockSearchBox';
import StockResultsPanel from '@/components/StockResultsPanel';
import MacroSnapshot from '../components/MacroSnapshot';
import ToolResults from '@/components/ToolResults';
import SourcesDrawer from '@/components/SourcesDrawer';

export default function Page() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [macroItems, setMacroItems] = useState<any[]>([]);
  const [macroSources, setMacroSources] = useState<any[]>([]);
  const [sourcesOpen, setSourcesOpen] = useState(false);
  const [booting, setBooting] = useState(false);
  const [bootError, setBootError] = useState<string | null>(null);
  const SESSION_KEY = 'finassist.sessionId.v1';
  const SESSION_TOKEN_KEY = 'finassist.sessionToken.v1';

  const API_BASE =
    process.env.NEXT_PUBLIC_API_BASE ||
    (typeof window !== 'undefined' ? window.location.origin : '');
  const [bootNonce, setBootNonce] = useState(0);

  useEffect(() => {
    let cancelled = false;
    setBooting(true);
    setBootError(null);

    async function boot() {
      try {
        const stored =
          typeof window !== 'undefined'
            ? (localStorage.getItem(SESSION_KEY) || localStorage.getItem('fa_session_id'))
            : null;
        const token = typeof window !== 'undefined' ? localStorage.getItem(SESSION_TOKEN_KEY) : null;
        if (stored) {
          setSessionId(stored);
          const r = await fetch(`${API_BASE}/api/chat/sessions/${stored}`, {
            headers: {
              ...(token ? { Authorization: `Bearer ${token}` } : {}),
            },
          });
          if (!r.ok) {
            const created = await createSession();
            if (!cancelled) {
              setSessionId(created.id);
              localStorage.setItem(SESSION_KEY, created.id);
              localStorage.setItem('fa_session_id', created.id);
              if ((created as any)?.token) localStorage.setItem(SESSION_TOKEN_KEY, (created as any).token);
            }
          }
          return;
        }

        const created = await createSession();
        if (!cancelled) {
          setSessionId(created.id);
          localStorage.setItem(SESSION_KEY, created.id);
          localStorage.setItem('fa_session_id', created.id);
          if ((created as any)?.token) localStorage.setItem(SESSION_TOKEN_KEY, (created as any).token);
        }
      } catch (e: any) {
        if (!cancelled) {
          setBootError(e?.message || 'Could not reach the backend. Please try again.');
          setSessionId(null);
        }
      } finally {
        if (!cancelled) setBooting(false);
      }
    }

    async function createSession() {
      const res = await fetch(`${API_BASE}/api/chat/sessions`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ title: 'New session' }),
      });
      if (!res.ok) {
        const txt = await res.text().catch(() => '');
        throw new Error(`createSession failed: ${res.status} ${res.statusText} – ${txt}`);
      }
      return res.json() as Promise<{ id: string }>;
    }

    boot();
    return () => { cancelled = true; };
  }, [API_BASE, bootNonce]);

  const addMacroSources = (srcs: any[]) => {
    setMacroSources(prev => {
      const map = new Map<string, any>((prev || []).map(s => [s.id || s.display, s]));
      for (const s of srcs) {
        map.set(s.id || s.display, s);
      }
      return Array.from(map.values());
    });
  };

  return (
    <main className="p-6 md:p-8 lg:p-10 max-w-none">
      <header className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold">FinAssist</h1>
        <div className="text-xs text-gray-500">Next.js + FastAPI</div>
      </header>

      <section className="space-y-8">
        {bootError && (
          <div className="rounded-xl border border-red-200 bg-red-50 p-3 text-sm text-red-700 flex items-start justify-between gap-3">
            <div>
              <div className="font-semibold">Cannot connect to the chat backend.</div>
              <div className="text-red-700/90">{bootError}</div>
            </div>
            <button
              onClick={() => { setBootError(null); setBooting(true); setBootNonce((n) => n + 1); }}
              className="shrink-0 rounded-lg border border-red-300 bg-white px-3 py-1 text-xs font-semibold text-red-700 hover:bg-red-100"
              disabled={booting}
            >
              {booting ? 'Retrying…' : 'Retry'}
            </button>
          </div>
        )}

        <div className="rounded-2xl border border-gray-200 bg-white p-4">
          <h2 className="text-xl font-semibold mb-3">Chat</h2>
          <ChatPro sessionId={sessionId ?? undefined} />
        </div>

        <div className="rounded-2xl border border-gray-200 bg-white p-4">
          <h2 className="text-xl font-semibold mb-3">Financial calculator</h2>
          <CalcPanel />
        </div>

        <div className="rounded-2xl border border-gray-200 bg-white p-4">
          <h2 className="text-xl font-semibold mb-3">Stock prices</h2>
          <div className="mb-4">
            <StockSearchBox />
          </div>
          <StockResultsPanel />
        </div>

        {/* World Bank Statistics card */}
        <div className="rounded-2xl border border-gray-200 bg-white p-4">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-xl font-semibold">World Bank Statistics</h2>
            <div className="flex items-center gap-2">
              <button
                aria-label="Toggle sources"
                onClick={() => setSourcesOpen(o => !o)}
                className="rounded-lg border px-2 py-1 text-xs hover:bg-gray-50"
              >
                {sourcesOpen ? 'Hide sources' : 'Show sources'}
              </button>
              <div className="text-xs text-gray-500">MCP: world_bank</div>
            </div>
          </div>

          <MacroSnapshot
            addItem={(it: any) => setMacroItems(prev => [it, ...prev])}
            addSources={addMacroSources}
          />

          {/* Results */}
          <div className="mt-3">
            <ToolResults items={macroItems} />
          </div>

          {/* Sources for these actions */}
          <div className="mt-3">
            <SourcesDrawer open={sourcesOpen} onOpenChange={setSourcesOpen} sources={macroSources} />
          </div>
        </div>
      </section>
    </main>
  );
}
