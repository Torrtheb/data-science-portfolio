// src/components/ClientPicker.tsx
"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import { searchOwnerClients } from "@/lib/api";

type OwnerClientLite = { id: string; account_id: number; name?: string | null; email: string };

type CommonProps = {
  placeholder?: string;
  minChars?: number;          // start searching after this many chars
  className?: string;
  disabled?: boolean;
  showEmailOnlyInInput?: boolean;  // <-- NEW: controls the text shown in the box on select
};

type SingleProps = CommonProps & {
  multiple?: false;
  value: OwnerClientLite | null;
  onChange: (val: OwnerClientLite | null) => void;
};

type MultiProps = CommonProps & {
  multiple: true;
  value: OwnerClientLite[];
  onChange: (val: OwnerClientLite[]) => void;
};

export default function ClientPicker(props: SingleProps | MultiProps) {
  const {
    placeholder = "Search by name or email…",
    minChars = 2,
    className,
    disabled,
    showEmailOnlyInInput = false,
  } = props;

  const [q, setQ] = useState("");
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<OwnerClientLite[]>([]);
  const timer = useRef<number | null>(null);
  const listRef = useRef<HTMLDivElement | null>(null);

  const isMulti = (p: SingleProps | MultiProps): p is MultiProps => "multiple" in p && !!p.multiple;

  const selectedMap = useMemo(() => {
    if (!isMulti(props)) return new Map<string, true>();
    const m = new Map<string, true>();
    props.value.forEach((v) => m.set(v.id, true));
    return m;
  }, [props]);

  // Debounced search
  useEffect(() => {
    if (timer.current) window.clearTimeout(timer.current);
    if (q.trim().length < minChars) {
      setResults([]);
      setOpen(false);
      return;
    }
    setLoading(true);
    timer.current = window.setTimeout(async () => {
      try {
        const rows = await searchOwnerClients(q.trim());
        setResults(rows);
        setOpen(true);
      } catch (e) {
        console.error("Client search failed", e);
        setResults([]);
        setOpen(false);
      } finally {
        setLoading(false);
      }
    }, 250);
    return () => {
      if (timer.current) window.clearTimeout(timer.current);
    };
  }, [q, minChars]);

  function choose(item: OwnerClientLite) {
    // What do we show inside the input box after selection?
    const textForBox = showEmailOnlyInInput
      ? (item.email ?? "")
      : ((item.name ?? "") + (item.name ? " " : "") + (item.email ?? "")).trim();

    if (isMulti(props)) {
      if (selectedMap.has(item.id)) {
        props.onChange(props.value.filter((v) => v.id !== item.id));
      } else {
        props.onChange([...props.value, item]);
      }
      setQ(textForBox);
      setOpen(true); // keep open to add more
    } else {
      props.onChange(item);
      setQ(textForBox);
      setOpen(false);
    }
  }

  function clearSelection() {
    if (isMulti(props)) props.onChange([]);
    else props.onChange(null);
    setQ("");
    setOpen(false);
  }

  // Close on outside click
  useEffect(() => {
    function onDocClick(ev: MouseEvent) {
      if (!listRef.current) return;
      if (!listRef.current.contains(ev.target as Node)) setOpen(false);
    }
    if (open) document.addEventListener("mousedown", onDocClick);
    return () => document.removeEventListener("mousedown", onDocClick);
  }, [open]);

  return (
    <div className={`relative ${className || ""}`}>
      <div className="flex items-center gap-2">
        <input
          disabled={disabled}
          value={q}
          onChange={(e) => setQ(e.target.value)}
          onFocus={() => results.length && setOpen(true)}
          placeholder={placeholder}
          className="w-full rounded-md border px-3 py-2 text-sm"
        />
        {(isMulti(props) ? props.value.length > 0 : !!props.value) && (
          <button
            type="button"
            onClick={clearSelection}
            className="text-xs rounded-md border px-2 py-1 hover:bg-gray-50"
          >
            Clear
          </button>
        )}
      </div>

      {open && (
        <div
          ref={listRef}
          className="absolute z-30 mt-1 w-full rounded-md border bg-white shadow max-h-72 overflow-auto"
        >
          {loading && <div className="px-3 py-2 text-xs text-gray-500">Searching…</div>}

          {!loading && results.length === 0 && (
            <div className="px-3 py-2 text-xs text-gray-500">No matches</div>
          )}

          {!loading &&
            results.map((r, idx) => {
              const _label = `${r.name ?? ""}${r.name ? " " : ""}${r.email ?? ""}`.trim() || "(unknown)";
              const picked = selectedMap.has(r.id);
              return (
                <button
                  key={`${r.id || r.email || "row"}-${idx}`} // defensive unique key
                  type="button"
                  className={`block w-full px-3 py-2 text-left text-sm hover:bg-gray-50 ${
                    picked ? "bg-gray-50" : ""
                  }`}
                  onMouseDown={(e) => e.preventDefault()}
                  onClick={() => choose(r)}
                >
                  <div className="font-medium">{r.name || "—"}</div>
                  <div className="text-xs text-gray-500">{r.email || "—"}</div>
                </button>
              );
            })}
        </div>
      )}
    </div>
  );
}
