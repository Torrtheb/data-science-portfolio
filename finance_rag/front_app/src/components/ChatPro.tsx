// src/components/ChatPro.tsx
'use client';

import React, { useEffect, useMemo, useRef, useState } from 'react';
import 'highlight.js/styles/github.css';
import { trackTurn, trackTool, SESSION_KEY as ANALYTICS_SESSION_KEY } from '@/lib/analytics';
const FE_ANALYTICS = process.env.NEXT_PUBLIC_ANALYTICS_FE === '1';

import type { SourceItem } from '@/lib/types';
import { buildUrl } from '@/lib/backend';
import { downloadSessionExport } from '@/lib/download';
import SourcesDrawer, { dedupeSourcesForDisplay } from './SourcesDrawer';
import MarkdownMath from './MarkdownMath';

type ChatRole = 'user' | 'assistant' | 'system';
type ChatMsg = { role: ChatRole; content: string };
type ChatProProps = { sessionId?: string | undefined };

type Usage = {
  model?: string;
  input_tokens?: number;
  output_tokens?: number;
  total_tokens?: number;
  cost_usd?: number;
  pricing_per_m_tokens?: { input?: number; output?: number };
};

type DonePayload = { sources?: SourceItem[]; tools?: any[]; usage?: Usage };
type SSEHandle = { cancel: () => void };

const STORAGE_KEY = 'finassist.chat.v1';
const ENABLE_PRICE_DIRECT = false;

// persistent session id + headers
const SESSION_KEY = 'finassist.sessionId.v1';
const SESSION_TOKEN_KEY = 'finassist.sessionToken.v1';

function sessionHeaders(sessionId?: string): Record<string, string> {
  const h: Record<string, string> = { 'content-type': 'application/json' };
  const sid = sessionId || localStorage.getItem(SESSION_KEY) || '';
  const tok = localStorage.getItem(SESSION_TOKEN_KEY) || '';
  // When a token is present, skip X-Session-Id to avoid mismatch with signed sid.
  if (tok) {
    h['Authorization'] = `Bearer ${tok}`;
  } else if (sid) {
    h['X-Session-Id'] = sid;
  }
  return h;
}

/* ========================= Sanitizers & helpers ========================= */

const INLINE_MATH_SAFE = /\$(?!\s*[+\-]?\d)([\s\S]*?)(?<!\\)\$/g;

function updateLastAssistant(
  setMessages: React.Dispatch<React.SetStateAction<ChatMsg[]>>,
  updater: (prevText: string) => string
) {
  setMessages((prev) => {
    const copy = prev.slice();
    for (let i = copy.length - 1; i >= 0; i--) {
      const m = copy[i];
      if (m && m.role === 'assistant') {
        copy[i] = { role: 'assistant', content: updater(m.content) };
        return copy;
      }
    }
    copy.push({ role: 'assistant', content: updater('') });
    return copy;
  });
}
/** Strip placeholder junk the model sometimes leaks. */
function stripPlaceholders(s: string): string {
  if (!s) return s;
  return s
    .replace(/\$?\$?_?BLOCK_|\b_BLOCK_\$?\$?\b/gi, '')
    .replace(/@@(?:KTXSLOT|MATHBLK|MATHINL)\d+@@/g, '')
    .replace(/\$[DI]\$/g, '')
    .replace(/\bMB_\d+_*\b/gi, '')
    .replace(/\bMATH\d*\b/gi, '')
    .replace(/__MB_\d+__/g, '')
    .replace(/\bMB_\d+____MB_\d+\b/g, '');
}

/** Mask code and math while we operate on plain text safely. */
function maskSegments(md: string) {
  const slots: string[] = [];
  let s = md.replace(/```[\s\S]*?```/g, (m) => `__CODE_${slots.push(m) - 1}__`);
  s = s.replace(/\$\$([\s\S]*?)\$\$/g, (_m, b) => `__MB_${slots.push(`$$${b}$$`) - 1}__`);
  s = s.replace(INLINE_MATH_SAFE, (_m, b) => `__MI_${slots.push(`$${b}$`) - 1}__`);
  return { s, slots };
}
function unmaskSegments(s: string, slots: string[]) {
  return s
    .replace(/__CODE_(\d+)__/g, (_m, i) => slots[Number(i)] ?? '')
    .replace(/__MB_(\d+)__/g, (_m, i) => slots[Number(i)] ?? '')
    .replace(/__MI_(\d+)__/g, (_m, i) => slots[Number(i)] ?? '');
}

/** Remove stray backslashes before bare digits only.
 *  Keep "\$123" escaped so $ stays as currency, not math.
 */
function unescapeStrayNumberBackslashes(md: string): string {
  if (!md) return md;
  const { s: masked, slots } = maskSegments(md);

  // Drop "\" only if the next non-space is NOT "$" and a number follows.
  const t = masked.replace(/\\(?=(?!\s*\$)\s*[+\-]?\d)/g, '');

  return unmaskSegments(t, slots);
}



/** Repairs common LaTeX breakages INSIDE math only. */
function repairMathBodiesGenerically(md: string): string {
  if (!md) return md;

  const fixBody = (body: string) => {
    let b = String(body);

    b = b
      .replace(/\s*\n+\s*/g, ' ') 
      .replace(/[ \t]{2,}/g, ' ');
    b = b.replace(/\\frac\s+([A-Za-z0-9])\s+([A-Za-z0-9])/g, (_m, a, c) => `\\frac{${a}}{${c}}`);
    b = b
      .replace(/\\frac\s*\{([^}]+)\}\s+([A-Za-z0-9]+)/g, (_m, a, c) => `\\frac{${a}}{${c}}`)
      .replace(/\\frac\s+([A-Za-z0-9]+)\s*\{([^}]+)\}/g, (_m, a, c) => `\\frac{${a}}{${c}}`);
    b = b.replace(/\^([A-Za-z]{2,}|\d{2,})/g, (_m, pow) => `^{${pow}}`);
    b = b.replace(/\\(?=\d)/g, '');

    // Finance-specific safety net:
    b = b.replace(/(\\text\{[^}]*compound interest[^}]*\}\s*=\s*)A\s+P/gi, '$1A - P');
    // Generic fallback right after an equals: "= A P" → "= A - P"
    b = b.replace(/(=\s*)A\s+P\b/g, '$1A - P');

    return b.trim();
  };

  return md
    .replace(/\$\$([\s\S]*?)\$\$/g, (_m, body) => `$$${fixBody(body)}$$`)
    .replace(INLINE_MATH_SAFE, (_m, body) => `$${fixBody(body)}$`);

}


/** Make headings/lists start on their own lines (never touches code blocks). */
function enforceHeadingsAndStepsNewlines(md: string): string {
  if (!md) return md;
  const parts = md.split(/(```[\s\S]*?```)/g);
  const fixed = parts.map((seg, i) => {
    if (i % 2 === 1) return seg ?? ''; // keep code as-is
    let s = seg ?? '';

    // a) Headings on their own lines
    s = s.replace(/([^\n])\s*(#{1,6}\s+[^\n]+)/g, (_m, pre, hd) => `${pre}\n${hd}`);
    s = s.replace(/\n{2,}(#{1,6}\s+)/g, '\n\n$1');
    s = s.replace(/(#{1,6}\s+[^\n]+)(?!\n)/g, '$1\n');

    // b) Steps/lists on fresh lines
    s = s.replace(/([^\n])\s+((?:\d{1,3}[.)]|[-*+])\s+)/g, (_m, pre, marker) => `${pre}\n${marker}`);

    return s;
  });
  return fixed.join('');
}

/** Collapse huge blank runs outside code/math; keep 1 empty line at most. */
function collapseExcessNewlinesOutsideBlocks(md: string): string {
  if (!md) return md;
  const slots: string[] = [];
  let t = md;

  // Mask code fences & math
  t = t.replace(/```[\s\S]*?```/g, (m) => `__CODE_${slots.push(m) - 1}__`);
  t = t.replace(/\$\$([\s\S]*?)\$\$/g, (_m, body) => `__MB_${slots.push(`$$${body}$$`) - 1}__`);
  t = t.replace(INLINE_MATH_SAFE, (_m, body) => `__MI_${slots.push(`$${body}$`) - 1}__`);

  // Stronger compaction:
  t = t
    .replace(/[ \t]+\n/g, '\n')  
    .replace(/\n[ \t]*\n+/g, '\n\n') 
    .replace(
      /(^[^\n:\r]{1,40}:[^\n]*?)\n[ \t]*\n(?=^[^\n:\r]{1,40}:[^\n]*?$)/gm,
      '$1\n'
    );

  // Unmask
  t = t
    .replace(/__CODE_(\d+)__/g, (_m, i) => slots[Number(i)] ?? '')
    .replace(/__MB_(\d+)__/g,   (_m, i) => slots[Number(i)] ?? '')
    .replace(/__MI_(\d+)__/g,   (_m, i) => slots[Number(i)] ?? '');

  // Trim leading/trailing blank lines
  return t.replace(/^\s*\n/, '').replace(/\n\s*$/, '');
}

/** Rejoin split words like "Int erest", "Exa mple". */
/** Safer patch: only mend real line-break shards, never normal spaces. */
/** Safer patch: only mend real line-break shards, never normal spaces. */
function fixMidwordSplits(md: string): string {
  if (!md) return md;
  const parts = md.split(/(```[\s\S]*?```)/g);
  return parts.map((seg, i) => {
    if (i % 2 === 1) return seg ?? '';  
    let s = seg ?? '';

    // Join hyphenation across newlines: "com-\npute" → "com-pute"
    s = s.replace(/-\n([A-Za-z])/g, '-$1');

    // Newline inside a word → space: "simp\nle" → "simp le"
    s = s.replace(/([A-Za-z])\n([a-z])/g, '$1 $2');

    return s;
  }).join('');
}

/** Turn consecutive "Key: Value" lines into a tight markdown list (outside code/math). */
export function listifyKeyValueBlocks(md?: string): string {
  const src = md ?? "";
  if (!src) return src;
  const { s: masked, slots } = maskSegments(src);

  const isKV = (ln: string) =>
    /^[ \t]*[A-Za-z][A-Za-z0-9 /().%+°-]{0,40}:[ \t]*\S/.test(ln);

  const isHeaderWithColonOnly = (ln: string) =>
    /^[ \t]*[A-Za-z][A-Za-z0-9 /().%+°-]{1,40}:[ \t]*$/.test(ln);

  const lines: string[] = masked.split("\n");
  const out: string[] = [];

  let inBlock = false;

  for (let i = 0; i < lines.length; i++) {
    const ln = lines[i] ?? "";

    if (isKV(ln)) {
      // Start (or continue) a KV block
      if (!inBlock) {
        // If the previous line is a "Header:" with no value, convert it to a bold heading.
        if (out.length > 0) {
          const prev = out[out.length - 1] ?? "";
          if (isHeaderWithColonOnly(prev)) {
            out[out.length - 1] = `**${prev.replace(/:\s*$/, "")}**`;
          }
          // Ensure exactly one blank line before the list
          if ((out[out.length - 1] ?? "").trim() !== "") out.push("");
        }
        inBlock = true;
      }

      // Emit as a list item (strip leading spaces, prefix "- ")
      out.push("- " + ln.replace(/^\s+/, ""));

      // Skip one blank line *between* KV items if the next non-blank is KV
      while (i + 2 < lines.length) {
        const maybeBlank = lines[i + 1] ?? "";
        const maybeNext = lines[i + 2] ?? "";
        if (/^\s*$/.test(maybeBlank) && isKV(maybeNext)) {
          i += 1;
          continue;
        }
        break;
      }
      continue;
    }

    // Non-KV line
    if (inBlock) {
      inBlock = false;
      const prevOut = out[out.length - 1] ?? "";
      const needSep = prevOut.trim() !== "";

      if (ln.trim() === "") {
        // The input line is already blank: ensure at most one blank line total.
        if (needSep) out.push("");
        // Skip pushing this blank `ln`; we've already added the single spacer if needed.
        continue;
      } else {
        // Next line is non-blank: ensure a single spacer, then push the line.
        if (needSep) out.push("");
        out.push(ln);
        continue;
      }
    }

    // Normal passthrough
    out.push(ln);
  }

  // Join still-masked text
  let res = out.join("\n");

  // Final pass (still masked): collapse runs of blank lines to a single blank line.
  {
    let prevBlank = false;
    const compact: string[] = [];
    for (const l of res.split("\n")) {
      const line = l ?? "";
      const isBlank = line.trim() === "";
      if (isBlank && prevBlank) continue;
      compact.push(line);
      prevBlank = isBlank;
    }
    res = compact.join("\n");
  }

  // Unmask + trim
  return unmaskSegments(res, slots ?? []).replace(/^\s+|\s+$/g, "");

}


/** Normalize \[...\] / \(...\) / ```math to $$ / $ */
function normalizeMath(md: string): string {
  if (!md) return md;
  // Keep $$ blocks blocky, but don't inject extra blank lines around them.
  return md
    .replace(/```math\s*([\s\S]*?)```/g, (_m, p1) => `$$\n${String(p1).trim()}\n$$`)
    .replace(/\\\[\s*([\s\S]*?)\s*\\\]/gs, (_m, p1) => `$$\n${String(p1).trim()}\n$$`)
    .replace(/\\\(\s*([\s\S]*?)\s*\\\)/g, (_m, p1) => `$${String(p1).trim()}$`);
}


/** Convert LaTeX links to Markdown links in non-code segments. */
function normalizeLatexLinksForChat(md: string): string {
  if (!md) return md;
  const parts = md.split('```');
  for (let i = 0; i < parts.length; i += 2) {
    let s = parts[i] ?? '';
    s = s.replace(/\\href\{([^}]+)\}\{([^}]+)\}/g, '[$2]($1)');
    s = s.replace(/\\url\{([^}]+)\}/g, '$1');
    s = s.replace(/\\\((https?:\/\/[^)]+)\\\)/g, '$1');
    s = s.replace(/\\\[(https?:\/\/[^\]]+)\\\]/g, '$1');
    s = s.replace(/\$(https?:\/\/[^$]+)\$/g, '$1');
    parts[i] = s;
  }
  return parts.join('```');
}

/** Gentle cleanup of LaTeX-ish noise (does NOT delete single dollars). */
function sanitizeLatexNoise(md: string): string {
  if (!md) return md;
  let t = md;

  // LaTeX envs → $$...$$ blocks
  t = t.replace(
    /\\begin\{(?:align\*?|equation\*?|gather\*?)\}([\s\S]*?)\\end\{(?:align\*?|equation\*?|gather\*?)\}/g,
    (_m, body) => `\n$$\n${String(body || '').trim()}\n$$\n`
  );

  // Remove \text{} outside math
  t = t.replace(/(?<!\$)\\text\{([^}]*)\}(?!\$)/g, (_m, p1) => p1);

  // Clean stray \left/\right outside math
  t = t.replace(/(?<!\$)\\(?:left|right)[\(\)\[\]\{\}](?!\$)/g, '');

  // Close accidental quadruple $$
  t = t.replace(/\$\$\s*\$\$/g, '$$');

  return t;
}

/** Currency-escape outside math; keep $...$ / $$...$$ intact. */
function escapeCurrencyDollars(md: string): string {
  if (!md) return md;
  const placeholders: string[] = [];
  let t = md.replace(/\$\$([\s\S]*?)\$\$/g, (_m, p1) => `__MB_${placeholders.push(p1) - 1}__`);
  t = t.replace(INLINE_MATH_SAFE, (_m, p1) => `__MI_${placeholders.push(p1) - 1}__`);

  // Escape $ that clearly starts a number (currency)
  t = t.replace(/(?<!\$)\$(?=[()+-]?\s*\d)/g, '\\$');

  // Unmask
  t = t.replace(/__MB_(\d+)__/g, (_m, i) => `$$${placeholders[Number(i)]}$$`);
  t = t.replace(/__MI_(\d+)__/g, (_m, i) => `$${placeholders[Number(i)]}$`);
  return t;
}

/** Smart fixer for mixed currency/math and placeholder junk in plain text. */
function sanitizeFormulas(md: string): string {
  if (!md) return md;

  // Quick exit if nothing to fix
  if (!/(\$\$|\$|\\text\{|MB_|__MB_|MATH|\\\[|\\\()/.test(md)) return md;

  // Mask code & math
  type Slot = { kind: 'code' | 'mathD' | 'mathI'; body: string };
  const slots: Slot[] = [];
  const slot = (kind: Slot['kind'], body: string) => `__SLOT_${slots.push({ kind, body }) - 1}__`;

  let t = md
    .replace(/```[\s\S]*?```/g, (m) => slot('code', m))
    .replace(/\$\$([\s\S]*?)\$\$/g, (_m, b) => slot('mathD', `$$${String(b)}$$`))
    .replace(INLINE_MATH_SAFE, (_m, b) => slot('mathI', `$${String(b)}$`));

  // Plain text fixes
  t = stripPlaceholders(t);

  // Turn \text{Simple Interest} (in text) → Simple Interest
  t = t.replace(/\\text\{([^}]*)\}/g, '$1');

  // Collapse $$$ → $$ (we’ll balance later anyway)
  t = t.replace(/\${3,}/g, '$$');

  // If we see $$ before a number in text (e.g., r$$150), it’s currency
  t = t.replace(/\$\$(?=\s*[+\-]?\d)/g, '\\$');

  // Escape lone $ that starts a number (currency)
  t = t.replace(/(?<!\$)\$(?=\s*[+\-]?\d)/g, '\\$');

  // Rejoin split words
  t = t.replace(/([A-Za-z])\n([a-z])/g, '$1 $2').replace(/-\n([A-Za-z])/g, '-$1');
  // Keep arithmetic fragments together: "years ×\n12" → "years × 12", "rate /\n12" → "rate / 12"
  t = t.replace(/×\s*\n\s*/g, '× ').replace(/\/\s*\n\s*/g, '/ ');

  // Unmask original code/math
  t = t.replace(/__SLOT_(\d+)__/g, (_m, i) => slots[Number(i)]?.body ?? '');

  // Balance any odd counts of $$ or $
  const count = (s: string, re: RegExp) => (s.match(re) || []).length;
  const displayDollars = count(t, /\$\$/g);
  if (displayDollars % 2 === 1) {
    t = t.replace(/\$\$(?!.*\$\$)/, '\\$\\$'); 
  }
  const inlineDollars = count(t, /(?<!\$)\$(?!\$)/g);
  if (inlineDollars % 2 === 1) {
    t = t.replace(/(?<!\$)\$(?!.*(?<!\$)\$)/, '\\$'); 
  }

  // $$ 123 $$ → 123 (don’t wrap a plain number in display math)
  t = t.replace(/\$\$\s*([+\-]?\d[\d.,]*)\s*\$\$/g, '$1');

  return t;
}

/** Insert a space when long numbers and words get glued together (outside code/math). */
function fixNumberWordGlueOutsideBlocks(md: string): string {
  if (!md) return md;
  const { s: masked, slots } = maskSegments(md);

  // 1) Number immediately followed by a word (e.g., "1,000over") → "1,000 over"
  let t = masked.replace(/(\d[\d,]*)(?=[A-Za-z]{2,})/g, '$1 ');

  // 2) Word immediately followed by a number (e.g., "years10") → "years 10"
  t = t.replace(/([A-Za-z]{2,})(?=\d[\d,]*)/g, '$1 ');

  // 3) Collapse any double spaces produced by the two passes above
  t = t.replace(/[ ]{2,}/g, ' ');

  return unmaskSegments(t, slots);
}

/** Ensure ",word" → ", word" outside code/math (won't touch "1,000"). */
function ensureSpaceAfterCommaBeforeLetters(md: string): string {
  if (!md) return md;
  const { s, slots } = maskSegments(md);
  const t = s.replace(/,([A-Za-z])/g, ', $1');
  return unmaskSegments(t, slots);
}

/** Turn the stray finance tokenization "(A P)" (incl. line break) into "(A - P)". */
function fixAPParenEverywhere(md: string): string {
  if (!md) return md;
  const { s, slots } = maskSegments(md);
  const t = s.replace(/\(\s*A\s+P\s*\)/g, '(A - P)');
  return unmaskSegments(t, slots);
}

/** Normalize streaming noise and trim silly whitespace. */
function sanitizeLLM(text: string): string {
  if (!text) return text;
  return text
    .replace(/\r\n/g, '\n')
    .replace(/[ \t]+\n/g, '\n')
    .replace(/^\s*\*?\s*(Resolved\s*Query|ResolvedQuery|Normalized\s*Query)\s*:[^\n]*\n?/gim, '');
}

/** Minimal patches:
 *  - ensure a space after commas before letters: "39.94,with" → "39.94, with"
 *  - fix "(A P)" (even if split across lines) → "(A - P)"
 *  - unjam a frequent finance phrase the model sometimes glues together
 */
function tinyChatPatches(md: string): string {
  if (!md) return md;

  // Work only outside code/math
  const { s: masked, slots } = maskSegments(md);

  let t = masked;

  // 1) Comma immediately followed by a letter → insert a space
  t = t.replace(/,([A-Za-z])/g, ', $1');

  // 2) Fix "(A P)" even when the P is on the next line
  t = t
    .replace(/\(\s*A\s*[\n\r]+\s*P\s*\)/g, '(A - P)') 
    .replace(/\(\s*A\s+P\s*\)/g, '(A - P)');  

  // 3) The model sometimes glues this specific phrase
  t = t.replace(/\bwithatotalcontributionof\b/gi, 'with a total contribution of');

  return unmaskSegments(t, slots);
}

/** Insert missing spaces around numbers and the single-letter article 'a' (outside code/math). */
function fixStuckArticlesAndNumbers(md: string): string {
  if (!md) return md;
  const { s: masked, slots } = maskSegments(md);
  let t = masked;

  // number+letter → number + space + letter   (e.g., 1000with → 1000 with)
  t = t.replace(/(\d)(?=[A-Za-z])/g, '$1 ');

  // letter+number → letter + space + number   (e.g., with7 → with 7)
  t = t.replace(/([A-Za-z])(?=\d)/g, '$1 ');

  // word glued to the single-letter article 'a' right before a number:
  t = t.replace(/([A-Za-z])(a)(?=\d)/gi, '$1 a');

  return unmaskSegments(t, slots);
}


/** Insert a space when numbers (optionally with $ / %) run into words, outside code/math. */
function fixGluedNumberWordPairs(md: string): string {
  if (!md) return md;
  const { s, slots } = maskSegments(md); 

  let t = s;

  // $10,000with → $10,000 with   |   10,000with → 10,000 with
  t = t.replace(/(\\?\$?[+\-]?\d[\d,]*(?:\.\d+)?)(?=[A-Za-z])/g, '$1 ');

  // 7%annual → 7% annual
  t = t.replace(/(%)(?=[A-Za-z])/g, '$1 ');

  return unmaskSegments(t, slots);
}

/** Insert a missing space when "at" is glued to the article 'a' before a number: "ata7" → "at a7". */
function fixAtArticleBeforeNumber(md: string): string {
  if (!md) return md;
  const { s, slots } = maskSegments(md);
  // Only when 'at' is immediately followed by 'a' and then a number
  const t = s.replace(/\bat(?=a\s*[+\-]?\d)/gi, 'at ');
  return unmaskSegments(t, slots);
}

/** Strip any remaining backslashes that appear immediately before a bare number (NOT currency) outside math/code. */
function stripBackslashesBeforeBareNumbers(md: string): string {
  if (!md) return md;
  const { s, slots } = maskSegments(md);
  // \58,000  or  \\12.5  ->  58,000 / 12.5   (but keeps "\$10,000" intact)
  const t = s.replace(/\\+(?=(?!\s*\$)\s*[+\-]?\d[\d,]*(?:\.\d+)?\b)/g, '');
  return unmaskSegments(t, slots);
}

function finalPlaintextPolish(md: string): string {
  if (!md) return md;
  const { s, slots } = maskSegments(md);
  let t = s;

  // 1) \\\58,000  →  58,000   (but keep \$)
  t = t.replace(/\\+(?=(?!\s*\$)\s*[+\-]?\d[\d,]*(?:\.\d+)?\b)/g, '');

  // 2) number (or $number) glued to a word: 10,000for → 10,000 for   |   $10,000with → $10,000 with
  t = t.replace(/(\\?\$?[+\-]?\d[\d,]*(?:\.\d+)?)(?=[A-Za-z])/g, '$1 ');

  // 3) percent glued to a word: 7%annual → 7% annual
  t = t.replace(/%(?=[A-Za-z])/g, '% ');

  // 4) comma directly before a letter: 39.94,with → 39.94, with
  t = t.replace(/,([A-Za-z])/g, ', $1');

  // 5) “ata” right before a number or % → “at a”
  t = t.replace(/\bata(?=\s*(?:%|[+\-]?\d))/gi, 'at a');

  // 6) Generic: any word immediately followed by 'a' + number/percent → insert space before 'a'
  //    e.g., "witha7"→"with a7" (then rule 2/3 yields "with a 7" / "with a %")
  t = t.replace(/\b([A-Za-z]+)a(?=\s*(?:%|[+\-]?\d))/g, '$1 a');

  // 7) (A P) (same line or split line) → (A - P)
  t = t.replace(/\(\s*A\s*(?:[\n\r]+|\s+)\s*P\s*\)/g, '(A - P)');
  // Fix plain "(A P)" → "(A - P)" outside math too
t = t.replace(/\(\s*A\s+P\s*\)/g, '(A - P)');


  return unmaskSegments(t, slots);
}


/** Main one-pass formatter used for every assistant token and final text. */
function formatLLMForRender(md: string): string {
  if (!md) return md;
  let t = md;

  t = normalizeLatexLinksForChat(t);
  t = sanitizeLLM(t);
  t = sanitizeFormulas(t); 
  t = normalizeMath(t);
  t = unescapeStrayNumberBackslashes(t);
  t = finalPlaintextPolish(t);
  t = listifyKeyValueBlocks(t);
  t = collapseExcessNewlinesOutsideBlocks(t);
  t = fixMidwordSplits(t);
  t = tinyChatPatches(t);

  return t;
}

/** Rewrite citations like [S4] to markdown links that target the drawer. */
function rewriteCitations(text: string, sources: (SourceItem & any)[]): string {
  if (!text) return text;
  const map = new Map<string, number>();
  for (const s of sources || []) {
    const n = Number(String(s?.n ?? '').replace(/[^\d]/g, ''));
    if (Number.isFinite(n) && n > 0) {
      map.set(String(n), n);
    }
    const idNum = Number(String(s?.id ?? '').replace(/[^\d]/g, ''));
    if (Number.isFinite(idNum) && idNum > 0) {
      map.set(String(idNum), n || idNum);
    }
  }

  return (text || '').replace(/\[S(\d+)\]/g, (_m, g1) => {
    const num = map.get(String(g1)) ?? Number(g1);
    const safeNum = Number.isFinite(num) && num > 0 ? num : Number(g1);
    const href = `#source-${safeNum}`;
    return `[${safeNum}](${href})`;
  });
}



/* ========================= Networking & SSE ========================= */

const TIMEOUT_MS = 20000;
async function fetchWithTimeout(input: RequestInfo | URL, init: RequestInit = {}) {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), TIMEOUT_MS);
  try {
    return await fetch(input, { cache: 'no-store', ...init, signal: controller.signal });
  } finally {
    clearTimeout(id);
  }
}

function createTokenBucket(max = 5, periodMs = 10_000) {
  let tokens = max;
  let last = Date.now();
  return () => {
    const now = Date.now();
    tokens = Math.min(max, tokens + ((now - last) / periodMs) * max);
    last = now;
    if (tokens >= 1) { tokens -= 1; return true; }
    return false;
  };
}

function formatUSD(n?: number) {
  if (n === undefined || n === null || isNaN(n)) return '';
  return n < 0.01 ? `$${n.toFixed(4)}` : n < 1 ? `$${n.toFixed(3)}` : `$${n.toFixed(2)}`;
}

function extractPriceQuery(text: string): string | null {
  const original = text.trim().replace(/\s+/g, ' ');
  const s = original.toLowerCase();

  let m = s.match(/^(?:what\s+is\s+)?(?:the\s+)?(?:stock\s+price|price|quote)\s+(?:of|for)\s+(.+)$/i);
  if (m?.[1]) return original.slice(s.indexOf(m[1])).replace(/[?.!,]$/,'').trim();
  m = s.match(/^(?:price|quote)\s+(.+)$/i);
  if (m?.[1]) return original.slice(s.indexOf(m[1])).replace(/[?.!,]$/,'').trim();
  m = s.match(/^(.+?)(?:\s+stock)?\s+(?:price|quote)(?:\s+now|\s+today)?$/i);
  if (m?.[1]) return original.slice(s.indexOf(m[1])).replace(/[?.!,]$/,'').trim();
  const p4 = s.match(/(?:^|\b)(price|quote)\b.*\b([A-Za-z.\-]{1,10})\b/i);
  if (p4?.[2]) {
    const tok = p4[2];
    const start = s.indexOf(tok);
    return original.slice(start, start + tok.length).replace(/[?.!,]$/,'').trim();
  }
  return null;
}

function normalizeEquityQuery(raw: string): { query: string; note?: string } {
  const original = raw.trim();
  const s = original.toUpperCase().replace(/\s+/g, ' ');
  const squished = s.replace(/\s+/g, '');

  let m = s.match(/^(?:TSX|TSE|TO):\s*([A-Z.\-]+)$/);
  if (m) return { query: `${m[1]}.TO`, note: 'TSX symbol' };
  m = s.match(/^([A-Z.\-]+):\s*(?:TSX|TSE|TO)$/);
  if (m) return { query: `${m[1]}.TO`, note: 'TSX symbol' };
  if (/[A-Z.\-]+\.TO$/.test(s)) return { query: s, note: 'TSX (.TO)' };

  const ALIASES: Record<string, string> = {
    'AIR CANADA': 'AC.TO',
    'AIRCANADA': 'AC.TO',
    'RBC': 'RY.TO',
    'ROYAL BANK': 'RY.TO',
    'SHOPIFY': 'SHOP.TO',
  };
  if (ALIASES[s]) return { query: ALIASES[s], note: 'Alias→TSX' };
  if (ALIASES[squished]) return { query: ALIASES[squished], note: 'Alias→TSX' };
  return { query: original };
}

async function fetchQuoteInline(query: string) {
  const { query: q } = normalizeEquityQuery(query);
  const endpoints = [
    buildUrl(`/api/price?q=${encodeURIComponent(q)}`),
    buildUrl(`/api/quote?q=${encodeURIComponent(q)}`),
  ];
  for (const url of endpoints) {
    try {
      const r = await fetchWithTimeout(url);
      if (!r.ok) continue;
      const js = await r.json().catch(() => null);
      if (!js) continue;
      const data = normalizePricePayload(js, q);
      if (typeof data.price === 'number') return { ...data, _resolved: q };
    } catch {}
  }
  throw new Error('price_lookup_failed');
}

function renderInlinePriceCard(
  data: ReturnType<typeof normalizePricePayload> & { _resolved?: string }
) {
  const priceStr = Number(data.price).toLocaleString(undefined, { maximumFractionDigits: 4 });
  const ch = typeof data.change === 'number' ? data.change : null;
  const cp = typeof data.changePercent === 'number' ? data.changePercent : null;
  const delta =
    ch !== null && cp !== null
      ? ` ${ch >= 0 ? '▲' : '▼'} ${Math.abs(ch).toFixed(2)} (${cp.toFixed(2)}%)`
      : '';
  const curr = data.currency ? ` ${data.currency}` : '';
  const sym = (data.symbol || data._resolved || '').toUpperCase();

  return [
    '',
    '---',
    '**Stock price**',
    '',
    `**${sym}**  ·  **${priceStr}${curr}**${delta}`,
    data.exchange ? `Exchange: ${data.exchange}` : '',
    '---',
    '',
  ]
    .filter(Boolean)
    .join('\n');
}

function normalizePricePayload(js: any, fallback: string) {
  return {
    symbol: String(js.symbol || js.ticker || fallback || '').toUpperCase(),
    price: typeof js.price === 'number' ? js.price : (typeof js.c === 'number' ? js.c : undefined),
    change: typeof js.change === 'number' ? js.change : (typeof js.d === 'number' ? js.d : undefined),
    changePercent:
      typeof js.changePercent === 'number'
        ? js.changePercent
        : (typeof js.dp === 'number' ? js.dp : js.change_pct),
    currency: js.currency || js.cur || js.ccy || '',
    ...js,
  };
}

async function ensureBackendSession(id?: string): Promise<string> {
  let sid = (id || localStorage.getItem(SESSION_KEY) || '').trim();
  if (!sid) {
    sid = (globalThis.crypto?.randomUUID?.() ?? Math.random().toString(36).slice(2));
    localStorage.setItem(SESSION_KEY, sid);
    try { localStorage.removeItem(SESSION_TOKEN_KEY); } catch {}
  }
  try {
    const res = await fetch(buildUrl('/api/chat/sessions'), {
      method: 'POST',
      headers: sessionHeaders(sid),
      body: JSON.stringify({ id: sid, title: 'My Chat' }),
    });
    if (res.ok) {
      const js = await res.json().catch(() => ({}));
      if (js?.token) {
        localStorage.setItem(SESSION_TOKEN_KEY, js.token);
      }
    }
  } catch {}
  return sid;
}

async function loadHistory(sid: string): Promise<ChatMsg[]> {
  const r = await fetch(buildUrl(`/api/chat/sessions/${encodeURIComponent(sid)}`), {
    headers: sessionHeaders(sid),
    cache: 'no-store',
  });
  if (!r.ok) return [];
  const js = await r.json();
  const hist = (js?.messages || []).map((m: any) => ({ role: m.role as ChatRole, content: String(m.content ?? '') }));
  return hist;
}

async function persistMessage(sid: string, role: ChatRole, content: string, extras?: any) {
  try {
    await fetch(buildUrl(`/api/chat/sessions/${encodeURIComponent(sid)}/messages`), {
      method: 'POST',
      headers: sessionHeaders(sid),
      body: JSON.stringify({ role, content, ...(extras || {}) }),
    });
  } catch {}
}

function startSSE(
  url: string,
  body: unknown,
  onToken: (token: string) => void,
  onDone: (done: DonePayload) => void,
  onError: (msg: string) => void,
  timeoutMs = 60000,
  headers?: Record<string, string>,
): SSEHandle {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  (async () => {
    try {
      const res = await fetch(url, {
        method: 'POST',
        headers: headers && Object.keys(headers).length ? headers : { 'content-type': 'application/json' },
        body: JSON.stringify(body),
        signal: controller.signal,
      });

      if (!res.ok || !res.body) {
        const txt = await res.text().catch(() => '');
        onError(txt || `HTTP ${res.status}`);
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const parts = buffer.split('\n\n');
        buffer = parts.pop() || '';

        for (const chunk of parts) {
          const lines = chunk.split('\n').map((l) => l.trim());
          for (const line of lines) {
            if (!line.startsWith('data:')) continue;
            const jsonStr = line.slice(5).trim();
            if (!jsonStr) continue;
            let payload: any;
            try {
              payload = JSON.parse(jsonStr);
            } catch {
              continue;
            }
            if (typeof payload?.token === 'string') {
              onToken(payload.token);
            } else if (payload?.done) {
              onDone({
                sources: payload.sources,
                tools: payload.tools,
                usage: payload.usage,
              });
            } else if (payload?.error) {
              onError(String(payload.error));
            }
          }
        }
      }
    } catch (e: any) {
      if (e?.name === 'AbortError') {
        onError('Request timed out.');
      } else {
        onError(String(e?.message || e));
      }
    } finally {
      clearTimeout(timer);
    }
  })();

  return {
    cancel: () => {
      try {
        clearTimeout(timer);
        controller.abort();
      } catch {}
    },
  };
}

/* ========================= Component ========================= */

export default function ChatPro({ sessionId: sessionIdProp }: ChatProProps) {
  const [sessionId, setSessionId] = useState<string>('');
  const [messages, setMessages] = useState<ChatMsg[]>([]);
  const [input, setInput] = useState('');
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastQuestion, setLastQuestion] = useState<string | null>(null);

  const [sources, setSources] = useState<SourceItem[]>([]);
  const [showSources, setShowSources] = useState(false);
  const [usage, setUsage] = useState<Usage | null>(null);

  const endRef = useRef<HTMLDivElement | null>(null);
  const answerRef = useRef<string>('');
  const cancelRef = useRef<null | (() => void)>(null);
  const tryTake = useRef(createTokenBucket()).current;
  const authRetryingRef = useRef(false);

  useEffect(() => {
    (async () => {
      const sid = await ensureBackendSession(sessionIdProp);
      setSessionId(sid);
      try { localStorage.setItem(ANALYTICS_SESSION_KEY, sid); } catch {}
      const hist = await loadHistory(sid);
      if (Array.isArray(hist) && hist.length) setMessages(hist);
      else {
        try {
          const raw = localStorage.getItem(STORAGE_KEY);
          if (raw) {
            const parsed = JSON.parse(raw) as ChatMsg[];
            if (Array.isArray(parsed)) setMessages(parsed);
          }
        } catch {}
      }
    })();
  }, []);

  useEffect(() => {
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(messages)); } catch {}
  }, [messages]);

  useEffect(() => () => { try { cancelRef.current?.(); } catch {} }, []);

  useEffect(() => {
    const id = setTimeout(() => {
      endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
    }, 0);
    return () => clearTimeout(id);
  }, [messages, streaming]);

  const canSend = useMemo(() => {
    const t = input.trim();
    return t.length >= 2 && t.length <= 5000 && !streaming;
  }, [input, streaming]);

  const usageSummary = useMemo(() => {
    if (!usage) return '';
    const it = usage.input_tokens ?? 0;
    const ot = usage.output_tokens ?? 0;
    const tt = usage.total_tokens ?? (it + ot);
    const cost = usage.cost_usd;
    const pieces = [`Tokens: ${it.toLocaleString()} in + ${ot.toLocaleString()} out = ${tt.toLocaleString()}`];
    if (typeof cost === 'number') pieces.push(`Cost: ${formatUSD(cost)}`);
    return pieces.join(' · ');
  }, [usage]);

  const dedupedSources = useMemo(() => dedupeSourcesForDisplay(uniqSources(sources)), [sources]);

  function storeSessionLocally(id: string, token?: string | null) {
    setSessionId(id);
    try {
      localStorage.setItem(SESSION_KEY, id);
      localStorage.setItem(ANALYTICS_SESSION_KEY, id);
      if (token) localStorage.setItem(SESSION_TOKEN_KEY, token);
      else localStorage.removeItem(SESSION_TOKEN_KEY);
    } catch {}
  }

  async function createSessionAndStore(id?: string, title = 'My Chat') {
    const provisionalId = id || (globalThis.crypto?.randomUUID?.() ?? Math.random().toString(36).slice(2));
    try { localStorage.removeItem(SESSION_TOKEN_KEY); } catch {}

    let newId = provisionalId;
    let token: string | null = null;
    try {
      const res = await fetch(buildUrl('/api/chat/sessions'), {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ id: provisionalId, title }),
      });
      if (res.ok) {
        const js = await res.json().catch(() => ({}));
        newId = js.sessionId || js.id || provisionalId;
        token = js.token || null;
      }
    } catch {}

    storeSessionLocally(newId, token);
    return { id: newId, token };
  }

  function isAuthError(msg: string): boolean {
    const m = (msg || '').toLowerCase();
    return m.includes('401') || m.includes('403') || m.includes('unauthorized') || m.includes('forbidden');
  }

  async function refreshSessionAfterAuthFailure(message?: string) {
    if (authRetryingRef.current) return;
    authRetryingRef.current = true;
    try {
      await createSessionAndStore(undefined, 'My Chat');
      setError(message || 'Session refreshed. Please resend your question.');
    } catch (e: any) {
      setError(e?.message || message || 'Session expired. Please reload.');
    } finally {
      setStreaming(false);
      authRetryingRef.current = false;
    }
  }

  async function newChat() {
    try { cancelRef.current?.(); } catch {}
    setMessages([]);
    setSources([]);
    setShowSources(false);
    setInput('');
    setError(null);
    setUsage(null);
    answerRef.current = '';
    try { localStorage.removeItem(STORAGE_KEY); } catch {}

    await createSessionAndStore(undefined, 'My Chat');
  }

  async function fetchChatOnce(question: string) {
    const res = await fetch(buildUrl('/api/chat'), {
      method: 'POST',
      headers: sessionHeaders(sessionId),
      body: JSON.stringify({ messages: [{ role: 'user', content: question }] }),
      cache: 'no-store',
    });
    if (!res.ok) {
      const txt = await res.text().catch(() => '');
      throw new Error(txt || `HTTP ${res.status}`);
    }
    return res.json();
  }

  async function send(qOverride?: string) {
    const questionRaw = typeof qOverride === 'string' ? qOverride : input;
    const question = (questionRaw || '').trim();
    if (!question) { setError('Please enter a question.'); return; }
    if (question.length > 5000) { setError('That message is too long (5000 char limit).'); return; }
    if (!qOverride && !canSend) return;
    if (!tryTake()) { setError('Please wait a few seconds before sending again.'); return; }
    try { cancelRef.current?.(); } catch {}

    setLastQuestion(question);
    setInput('');
    setError(null);
    setShowSources(false);
    setUsage(null);

    // Shortcut for live price (optional)
    const priceQuery = extractPriceQuery(question);
    if (ENABLE_PRICE_DIRECT && priceQuery) {
      setMessages((prev) => [...prev, { role: 'user', content: question }, { role: 'assistant', content: '' }]);
      setStreaming(true);
      try {
        const data = await fetchQuoteInline(priceQuery);
        const card = renderInlinePriceCard(data);
        updateLastAssistant(setMessages, () => card);
        if (sessionId) {
          await persistMessage(sessionId, 'user', question);
          await persistMessage(sessionId, 'assistant', card);
        }
        setSources([{ provider: 'Finnhub', href: 'https://finnhub.io/', title: 'Finnhub', type: 'web' } as any]);
        setUsage({ model: 'n/a (direct quote)', input_tokens: 0, output_tokens: 0, total_tokens: 0, cost_usd: 0 });
        try {
          const sid = sessionId;
          const toolPayload: any = { tool_name: 'live_price_inline', args: { query: priceQuery }, ok: true };
          if (sid) toolPayload.session_id = sid;
          if (FE_ANALYTICS) trackTool(toolPayload);
          const turnPayload: any = {
            role: 'assistant',
            content: (card || '').slice(0, 1500),
            model: 'n/a (direct quote)',
            tokens_in: 0, tokens_out: 0, cost_usd: 0,
            latency_ms: 0, had_rag: false,
            tools_used: [{ name: 'live_price_inline' }],
          };
          if (sid) turnPayload.session_id = sid;
          if (FE_ANALYTICS) trackTurn(turnPayload);
        } catch {}
      } catch {
        const safe = `> I couldn't fetch a live quote for "${priceQuery}".  
> It might not be included in Finnhub's free API. Try a ticker like 'TSLA' or 'AAPL'.`;
        updateLastAssistant(setMessages, () => safe);
        if (sessionId) {
          await persistMessage(sessionId, 'user', question);
          await persistMessage(sessionId, 'assistant', safe);
        }
      } finally {
        setStreaming(false);
      }
      return;
    }

    // Normal chat stream
    setMessages((prev) => [...prev, { role: 'user', content: question }, { role: 'assistant', content: '' }]);
    setStreaming(true);
    answerRef.current = '';

    try {
      const sid = sessionId || localStorage.getItem(SESSION_KEY) || undefined;
      const payload: any = { role: 'user', content: question.slice(0, 1500) };
      if (sid) payload.session_id = sid;
      if (FE_ANALYTICS) trackTurn(payload);
    } catch {}

    const history: ChatMsg[] = [...messages, { role: 'user', content: question }];

    const url = buildUrl('/api/chat/stream');
    const body = { messages: history };
    const t0 = performance.now();

    const handle = startSSE(
      url,
      body,
      // onToken: stream with idempotent formatter
      (token: string) => {
        answerRef.current = (answerRef.current || '') + token; // append only
        updateLastAssistant(setMessages, () => answerRef.current);
      },
      // onDone
      (done: DonePayload) => {
        setStreaming(false);
        const mergedSources = uniqSources([...(sources || []), ...(done.sources || [])]);
        setSources(mergedSources);
        answerRef.current = formatLLMForRender(answerRef.current || '');
        const rewritten = rewriteCitations(answerRef.current, mergedSources);
        updateLastAssistant(setMessages, () => rewritten);

        if (done.usage && typeof done.usage === 'object') {
          const u = done.usage as any;
          const input_tokens = Number(u.input_tokens ?? u.prompt_tokens ?? 0);
          const output_tokens = Number(u.output_tokens ?? u.completion_tokens ?? 0);
          const total_tokens = Number(u.total_tokens ?? (input_tokens + output_tokens));
          const cost_usd =
            typeof u.cost_usd === 'number'
              ? u.cost_usd
              : (typeof u.total_cost === 'number' ? u.total_cost : undefined);

          setUsage({
            model: u.model || undefined,
            input_tokens,
            output_tokens,
            total_tokens,
            cost_usd,
            pricing_per_m_tokens: u.pricing_per_m_tokens || undefined,
          });
        } else {
          setUsage(null);
        }

        try {
          const ms = Math.round(performance.now() - t0);
          const u: any = done?.usage || {};
          trackTurn({
            session_id: sessionId,
            role: 'assistant',
            content: (answerRef.current || '').slice(0, 1500),
            model: u?.model,
            tokens_in: Number(u.input_tokens ?? u.prompt_tokens ?? 0),
            tokens_out: Number(u.output_tokens ?? u.completion_tokens ?? 0),
            cost_usd: typeof u.cost_usd === 'number' ? u.cost_usd : undefined,
            latency_ms: ms,
            had_rag: (done?.sources?.length ?? 0) > 0,
            tools_used: Array.isArray(done?.tools) ? done!.tools : [],
          });
        } catch {}
      },
      // onError
      (msg: string) => {
        if (isAuthError(msg)) {
          refreshSessionAfterAuthFailure('Session expired. Reconnected with a fresh session.');
          return;
        }
        if ((msg || '').toLowerCase().includes('timed out')) {
          fetchChatOnce(question)
            .then((js) => {
              const mergedSources = uniqSources([...(sources || []), ...(js.sources || [])]);
              setSources(mergedSources);
              const u = js.usage || {};
              const input_tokens = Number(u.input_tokens ?? u.prompt_tokens ?? 0);
              const output_tokens = Number(u.output_tokens ?? u.completion_tokens ?? 0);
              const total_tokens = Number(u.total_tokens ?? (input_tokens + output_tokens));
              const cost_usd =
                typeof u.cost_usd === 'number'
                  ? u.cost_usd
                  : (typeof u.total_cost === 'number' ? u.total_cost : undefined);
              setUsage({
                model: u.model || undefined,
                input_tokens,
                output_tokens,
                total_tokens,
                cost_usd,
                pricing_per_m_tokens: u.pricing_per_m_tokens || undefined,
              });
              answerRef.current = formatLLMForRender(String(js.text || ''));
              const rewritten = rewriteCitations(answerRef.current, mergedSources);
              updateLastAssistant(setMessages, () => rewritten);
              setStreaming(false);
            })
            .catch((e) => {
              const errMsg = e?.message || msg || 'Something went wrong.';
              if (isAuthError(errMsg)) {
                refreshSessionAfterAuthFailure('Session expired. Reconnected with a fresh session.');
              } else {
                setError(errMsg);
                setStreaming(false);
              }
            });
        } else {
          setError(msg || 'Something went wrong.');
          setStreaming(false);
        }
      },
      120000,
      sessionHeaders(sessionId)
    );

    cancelRef.current = handle.cancel;
  }

  function onKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  }

  return (
    <div className="rounded-2xl border shadow-sm bg-white flex flex-col">
      {/* Top controls */}
      <div className="px-3 pt-3 flex flex-col gap-2 sm:flex-row sm:flex-wrap sm:items-center">
        <button
          onClick={() => setShowSources((s) => !s)}
          className="px-4 py-2 rounded-xl border font-semibold text-gray-900 hover:bg-gray-50 disabled:opacity-50 w-full sm:w-auto"
          disabled={streaming || dedupedSources.length === 0}
          title={dedupedSources.length ? 'View sources' : 'No sources'}
        >
          View Sources {dedupedSources.length ? `(${dedupedSources.length})` : ''}
        </button>

        <div className="flex flex-wrap gap-2 w-full sm:w-auto">
          <button
            onClick={async () => {
              if (!sessionId) return;
              try { await downloadSessionExport(sessionId, 'json'); }
              catch (e: any) { alert(e?.message || 'Export failed'); }
            }}
            className="px-3 py-2 rounded-xl border text-gray-700 hover:bg-gray-50 disabled:opacity-50 flex-1 sm:flex-none"
            disabled={!sessionId}
            title="Download conversation as JSON"
          >
            Export JSON
          </button>
          <button
            onClick={async () => {
              if (!sessionId) return;
              try { await downloadSessionExport(sessionId, 'pdf'); }
              catch (e: any) { alert(e?.message || 'Export failed'); }
            }}
            className="px-3 py-2 rounded-xl border text-gray-700 hover:bg-gray-50 disabled:opacity-50 flex-1 sm:flex-none"
            disabled={!sessionId}
            title="Download conversation as PDF"
          >
            PDF
          </button>
          <button
            onClick={async () => {
              if (!sessionId) return;
              try { await downloadSessionExport(sessionId, 'csv'); }
              catch (e: any) { alert(e?.message || 'Export failed'); }
            }}
            className="px-3 py-2 rounded-xl border text-gray-700 hover:bg-gray-50 disabled:opacity-50 flex-1 sm:flex-none"
            disabled={!sessionId}
            title="Download conversation as CSV"
          >
            CSV
          </button>
        </div>

        <button
          onClick={newChat}
          className="px-4 py-2 rounded-xl border font-semibold text-gray-900 hover:bg-gray-50 w-full sm:w-auto"
          title="Start a new chat"
        >
          New chat
        </button>

        <div className="sm:ml-auto text-xs text-gray-600 pt-1 sm:pt-0">
          {streaming ? (
            <span className="animate-pulse text-gray-500">Streaming…</span>
          ) : usage ? (
            <span title={usage.model ? `Model: ${usage.model}` : undefined}>
              {usageSummary}
            </span>
          ) : null}
        </div>
      </div>

      {/* Messages */}
      <div className="px-3 py-4 space-y-3 overflow-auto" style={{ maxHeight: '65vh', scrollBehavior: 'smooth' }}>
        {messages.length === 0 && (
          <div className="text-sm text-gray-500">
            Ask me about investing, fees, asset allocation.
          </div>
        )}

        {messages.map((m, i) => (
          <MessageBubble
            key={i}
            msg={m}
            onCitationClick={(target) => {
              setShowSources(true);
              // Delay scrolling until drawer renders
              setTimeout(() => {
                const el = document.getElementById(target);
                if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
              }, 150);
            }}
          />
        ))}

        {error && (
          <div className="text-sm text-red-600 bg-red-50 border border-red-200 rounded-xl p-3 flex flex-wrap items-start justify-between gap-3">
            <div>{error}</div>
            {lastQuestion && !streaming && (
              <button
                onClick={() => send(lastQuestion)}
                className="text-xs font-semibold text-red-700 bg-white border border-red-200 rounded-lg px-3 py-1 hover:bg-red-100"
              >
                Retry last message
              </button>
            )}
          </div>
        )}

        <div ref={endRef} />
      </div>

      {/* Composer */}
      <div className="p-3 border-t">
        <div className="flex items-end gap-2">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder="Type your question… (Shift+Enter for newline)"
            className="flex-1 resize-none min-h-[48px] max-h-[160px] rounded-xl border px-3 py-2 focus:outline-none focus:ring-2 focus:ring-black/10"
          />
          <button
            onClick={() => send()}
            disabled={!canSend}
            className="px-4 py-2 rounded-xl bg-black text-white disabled:opacity-50"
          >
            {streaming ? 'Sending…' : 'Send'}
          </button>
        </div>
        <div className="mt-1 text-[11px] text-gray-400">Enter to send · Shift+Enter for newline</div>
      </div>

      <SourcesDrawer open={showSources} onOpenChange={setShowSources} sources={dedupedSources} />
    </div>
  );
}

/* ---- Message bubble (Markdown + KaTeX via MarkdownMath) ---- */
function MessageBubble({ msg, onCitationClick }: { msg: ChatMsg; onCitationClick?: (target: string) => void }) {
  const isUser = msg.role === 'user';
  const safe = isUser ? (msg.content || '') : formatLLMForRender(msg.content || '');

  const handleClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!onCitationClick) return;
    const target = (e.target as HTMLElement | null)?.closest('a');
    if (target && typeof target.getAttribute === 'function') {
      const href = target.getAttribute('href') || '';
      const m = href.match(/^#source-(\d+)/);
      if (m && m[1]) {
        e.preventDefault();
        onCitationClick(`source-${m[1]}`);
      }
    }
  };

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`} onClick={handleClick}>
      <div
        className={[
          'rounded-2xl px-3 py-2 max-w-[85%]',
          isUser ? 'bg-gray-900 text-white' : 'bg-gray-50 text-gray-900',
        ].join(' ')}
      >
        {isUser ? (
          <pre className="whitespace-pre-wrap font-sans text-sm m-0">{safe}</pre>
        ) : (
          <MarkdownMath
            text={safe}
            className={
              'prose prose-sm max-w-none chat-markdown ' +
              'prose-p:my-0 prose-li:my-0 prose-ul:my-0 prose-ol:my-0 ' +
              'prose-pre:my-2 prose-h1:mb-2 prose-h2:mb-2 prose-h3:mb-1'
            }
          />
        )}
      </div>
    </div>
  );
}

/* ========================= Source utils ========================= */

function normalizeSourceKey(s: Partial<SourceItem> & any): string {
  const href = (s.href || '').trim().toLowerCase();
  if (href) return `href:${href}`;
  const url = (s.url || '').trim().toLowerCase();
  if (url) return `url:${url}`;
  const display = (s.display || '').trim().toLowerCase();
  if (display) return `display:${display}`;
  const title = (s.title || '').trim().toLowerCase();
  if (title) return `title:${title}`;
  return 'unknown';
}

function rewriteVendorHref(s: any): string | null {
  const rawHref = (s?.href ?? s?.url ?? null) as string | null;
  const vendorBlob = `${s?.provider || ''} ${s?.source || ''} ${s?.title || ''} ${s?.display || ''} ${rawHref || ''}`;
  const isFinnhub = /finnhub/i.test(vendorBlob);
  if (!isFinnhub) return rawHref;
  if (!rawHref || /finnhub\.io\/api\//i.test(rawHref)) return 'https://finnhub.io/';
  return rawHref;
}

function uniqSources(list: (SourceItem & any)[] | undefined | null): (SourceItem & any)[] {
  const out: (SourceItem & any)[] = [];
  const seen = new Set<string>();
  for (const s of (list || [])) {
    const fixedHref = rewriteVendorHref(s);
    const key = normalizeSourceKey({ ...s, href: fixedHref });
    if (seen.has(key)) continue;
    seen.add(key);
    const existingNum = Number(String(s.n ?? s.id ?? '').replace(/[^\d]/g, ''));
    out.push({
      title: s.title || undefined,
      url: s.url || undefined,
      href: fixedHref ?? null,
      display: s.display || s.citation || s.title || s.url || 'Source',
      snippet: s.snippet,
      provider: s.provider,
      type: s.type,
      id: s.id ?? (Number.isFinite(existingNum) && existingNum > 0 ? `S${existingNum}` : undefined),
      n: Number.isFinite(existingNum) && existingNum > 0 ? existingNum : undefined,
    } as any);
  }
  return out.map((s, i) => {
    if (Number.isFinite(s.n) && s.n > 0) return s;
    const n = i + 1;
    return { ...s, id: s.id || `S${n}`, n };
  });
}
