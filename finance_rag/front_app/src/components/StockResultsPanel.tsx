// src/components/StockResultsPanel.tsx
'use client';

import { useEffect, useState } from 'react';
import ToolResults from './ToolResults';

type ToolItem = { tool: string; result: any };

export default function StockResultsPanel() {
  const [items, setItems] = useState<ToolItem[]>([]);

  useEffect(() => {
    function onAdd(e: Event) {
      const detail = (e as CustomEvent).detail as ToolItem | undefined;
      if (!detail || !detail.tool) return;
      setItems((prev) => pushTool(prev, detail));
    }
    window.addEventListener('finassist:add-tool', onAdd as EventListener);
    return () => window.removeEventListener('finassist:add-tool', onAdd as EventListener);
  }, []);

  return <ToolResults items={items} />;
}

/* ----------------- helpers ----------------- */
function pushTool(prev: ToolItem[], incoming: ToolItem): ToolItem[] {
  const maxTotal = 12;
  const tool = String(incoming.tool || '').toLowerCase();
  const sym = String(incoming?.result?.symbol || '').toUpperCase();

  let next = prev.slice();

  if (tool === 'live_price') {
    next.unshift(incoming);
    let countForSym = 0;
    next = next.filter((it) => {
      if (String(it.tool).toLowerCase() !== 'live_price') return true;
      const s = String(it?.result?.symbol || '').toUpperCase();
      if (s !== sym) return true;
      countForSym += 1;
      return countForSym <= 2;
    });
  } else {
    const idx = next.findIndex(
      (it) => String(it.tool).toLowerCase() === tool &&
              String(it?.result?.symbol || '').toUpperCase() === sym
    );
    if (idx >= 0) next[idx] = incoming;
    else next.unshift(incoming);
  }
  if (next.length > maxTotal) next = next.slice(0, maxTotal);
  return next;
}
