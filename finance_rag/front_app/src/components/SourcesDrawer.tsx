// src/components/SourcesDrawer.tsx
'use client';

import { useEffect, useMemo, useRef } from 'react';
import type { SourceItem } from '@/lib/types';

type Props = {
  open: boolean;
  onOpenChange: (v: boolean) => void;
  sources: SourceItem[];
};

// Be flexible with fields coming from the backend
type AnySource = SourceItem & {
  id?: string | number;
  n?: number;
  url?: string | null;
  href?: string | null;     
  title?: string | null;
  display?: string | null;   
  citation?: string | null;  
  provider?: string | null;
  snippet?: string | null;
  type?: string | null;   
  kind?: string | null;  
  origin?: string | null; 
  source?: string | null;
};

function originHost(u?: string | null): string | null {
  try {
    if (!u) return null;
    const url = new URL(u);
    return url.hostname.replace(/^www\./, '');
  } catch {
    return null;
  }
}

function getHref(s: AnySource): string | null {
  return (s.href ?? s.url ?? null) || null;
}

function isFileHref(u?: string | null) {
  return typeof u === 'string' && /^file:/i.test(u);
}

function linkableHref(s: AnySource): string | null {
  const h = getHref(s);
  if (!h) return null;
  return isFileHref(h) ? null : h; 
}

function primaryText(s: AnySource): string {
  return (
    s.display?.trim() ||
    s.citation?.trim() ||
    s.title?.trim() ||
    s.url?.trim() ||
    'Source'
  );
}

function stripPageRefs(text: string): string {
  // Remove page markers like "| Page 12", "Page 12-14", or "p. 12"
  let cleaned = text.replace(/\|\s*page[^|]+$/i, '');
  cleaned = cleaned.replace(/\bpp?\s*\.?\s*\d+(?:\s*[-–]\s*\d+)?/gi, '');
  cleaned = cleaned.replace(/\bpage\s+\d+(?:\s*[-–]\s*\d+)?/gi, '');
  cleaned = cleaned.replace(/\s{2,}/g, ' ');
  return cleaned.replace(/,\s*$/, '').trim();
}

function escapeHtml(t: string) {
  return t
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

function normalizeVendorDisplay(s: AnySource): AnySource {
  const blob = `${s.provider || ""} ${s.source || ""} ${s.display || ""} ${s.title || ""}`;
  const isFinnhub = /finnhub/i.test(blob) || /finnhub\.io/i.test(String(s.href || s.url || ""));
  if (isFinnhub) {
    return {
      ...s,
      provider: "Finnhub",
      href: s.href && /finnhub\.io\/api\//i.test(s.href) ? "https://finnhub.io/" : s.href ?? s.url ?? null,
      display: s.display || s.title || "Finnhub",
      title: s.title || s.display || "Finnhub",
    };
  }
  return s;
}

function sourceNumber(s: AnySource, fallback: number): number {
  const fromId = Number(String(s.id ?? '').replace(/[^\d]/g, ''));
  const fromN = Number(String(s.n ?? '').replace(/[^\d]/g, ''));
  if (Number.isFinite(fromN) && fromN > 0) return fromN;
  if (Number.isFinite(fromId) && fromId > 0) return fromId;
  return fallback;
}

/** Minimal markdown-for-citations: only *emphasis* allowed */
function citationHtml(text: string) {
  const safe = escapeHtml(text || '');
  return safe.replace(/\*(.+?)\*/g, '<em>$1</em>');
}

/** Nice little favicon helper (uses the site’s icon via browser) */
function faviconUrl(u?: string | null) {
  const host = originHost(u || undefined);
  if (!host) return null;
  return `https://${host}/favicon.ico`;
}

/** Identify tool sources regardless of which field the backend uses */
function isToolSource(s: AnySource): boolean {
  const fields = [
    (s.type || ''),
    (s.kind || ''),
    (s.origin || ''),
    (s.provider || ''),
    (s.source || ''),
  ].join('|').toLowerCase();
  if (fields.includes('tool')) return true;
  // Treat pseudo-links like tool://calc.cagr as tools
  const h = getHref(s);
  if (h && /^tool:/i.test(h)) return true;
  return false;
}

function isBook(s: AnySource): boolean {
  if (isToolSource(s)) return false;
  if ((s.type || '').toLowerCase() === 'book') return true;
  const hasLink = !!getHref(s);
  const hasCitationLike = !!(s.display || s.citation);
  return !hasLink && hasCitationLike;
}

// Exported so buttons can show the same count we render here
export function dedupeSourcesForDisplay(sources: SourceItem[]): AnySource[] {
  const seen = new Set<string>();
  const out: AnySource[] = [];
  for (let i = 0; i < (sources?.length || 0); i++) {
    const raw = normalizeVendorDisplay(sources[i] as AnySource);
    const cleanedDisplay = stripPageRefs(raw.display || raw.title || '');
    const normalized: AnySource = {
      ...raw,
      display: (cleanedDisplay || raw.display || null) as string | null,
    };
    const href = linkableHref(raw);
    const key = (href || raw.title || String(i)).trim().toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(normalized);
  }
  return out;
}

export default function SourcesDrawer({ open, onOpenChange, sources }: Props) {
  const closeBtnRef = useRef<HTMLButtonElement | null>(null);

  // Close on Escape
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onOpenChange(false);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, onOpenChange]);

  // Focus the close button when opening (simple focus management)
  useEffect(() => {
    if (open) {
      closeBtnRef.current?.focus();
    }
  }, [open]);

  // Dedupe sources by link/title for cleaner display
  const items = useMemo(() => dedupeSourcesForDisplay(sources), [sources]);

  const has = items.length > 0;

  return (
    <div
      className={[
        'fixed inset-0 z-40 transition',
        open ? 'pointer-events-auto' : 'pointer-events-none',
      ].join(' ')}
      aria-hidden={!open}
    >
      {/* backdrop */}
      <div
        className={[
          'absolute inset-0 bg-black/30 transition-opacity',
          open ? 'opacity-100' : 'opacity-0',
        ].join(' ')}
        onClick={() => onOpenChange(false)}
      />

      {/* panel */}
      <aside
        className={[
          'absolute right-0 top-0 h-full w-full max-w-md bg-white shadow-2xl border-l',
          'transition-transform duration-300',
          open ? 'translate-x-0' : 'translate-x-full',
        ].join(' ')}
        role="dialog"
        aria-modal="true"
        aria-labelledby="sources-title"
      >
        <div className="p-4 border-b flex items-center justify-between">
          <div id="sources-title" className="font-semibold">Sources</div>
          <button
            ref={closeBtnRef}
            onClick={() => onOpenChange(false)}
            className="text-sm px-3 py-1 rounded-lg border hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            Close
          </button>
        </div>

        <div className="p-4 space-y-3 overflow-auto h-[calc(100%-56px)]">
          {!has && (
            <div className="text-sm text-gray-500">
              Sources will appear here once the assistant uses the knowledge base or external tools
              to answer your questions.
            </div>
          )}

          {items.map((s, i) => {
            const num = i + 1; // sequential numbering for stability
            const href = linkableHref(s);
            const host = href ? originHost(href) : null;
            const tool = isToolSource(s);
            const book = !tool && isBook(s);
            const rawText = primaryText(s);
            const text = stripPageRefs(rawText);
            const fav = faviconUrl(href || undefined);
            const key = `${href || s.title || i}:${num}`;

            return (
              <div key={key} id={`source-${num}`} className="rounded-xl border p-3 scroll-m-4">
                <div className="flex items-center gap-2 text-xs text-gray-500 mb-1 min-w-0">
                  <span className="inline-flex h-5 min-w-5 items-center justify-center rounded bg-gray-100 px-1 font-mono text-gray-700">
                    {num}
                  </span>

                  {s.provider && (
                    <span className="rounded bg-gray-100 px-2 py-0.5">{s.provider}</span>
                  )}

                  {tool ? (
                    <span className="rounded bg-blue-100 text-blue-800 px-2 py-0.5">
                      Tool
                    </span>
                  ) : book ? (
                    <span className="rounded bg-amber-100 text-amber-800 px-2 py-0.5">
                      Book
                    </span>
                  ) : host ? (
                    <span className="inline-flex items-center gap-1 truncate">
                      {fav ? <img src={fav} alt="" className="w-3.5 h-3.5 rounded-sm" /> : null}
                      <span className="truncate">{host}</span>
                    </span>
                  ) : null}
                </div>

                <div className="font-medium break-words">
                  {href && !book ? (
                    <a
                      href={href}
                      target="_blank"
                      rel="noreferrer noopener"
                      className="underline underline-offset-2 break-words"
                      title={host || undefined}
                    >
                      {text}
                    </a>
                  ) : (
                    <span
                      className="whitespace-pre-line"
                      // Only <em> allowed via citationHtml; everything else escaped
                      dangerouslySetInnerHTML={{ __html: citationHtml(text) }}
                    />
                  )}
                </div>

                {s.snippet && (
                  <p className="text-sm text-gray-600 mt-1 line-clamp-4">{s.snippet}</p>
                )}
              </div>
            );
          })}
        </div>
      </aside>
    </div>
  );
}
