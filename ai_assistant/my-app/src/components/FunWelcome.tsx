"use client";
import React, { useState } from "react";
import { getFunWelcome, type FunWelcome } from "@/lib/api";

export default function FunWelcome() {
  const [item, setItem] = useState<FunWelcome | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string>("");
  const [src, setSrc] = useState<"random" | "cat" | "dog" | "fox">("random");

  async function fetchOne(srcIn?: "cat" | "dog" | "fox" | "random", fresh = false) {
    setLoading(true);
    setErr("");
    setItem(null);
    try {
      const r = await getFunWelcome(srcIn ?? src, fresh);
      if (!/^https?:\/\//i.test(r.url)) throw new Error("Bad URL");
      setItem(r);
    } catch (e: unknown) {
      setErr(e instanceof Error ? e.message : "Failed to load");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="p-3 flex items-center justify-between gap-3">
      <div className="space-y-2">
        <div className="font-medium">Feeling down?</div>
        <div className="text-sm text-gray-600">Click for a cute animal picture.</div>
        <div className="flex gap-2">
          <button
            className="px-3 py-1 rounded bg-blue-600 text-white"
            onClick={() => fetchOne(src)}
          >
            Click here
          </button>
          <select
            className="border rounded px-2 py-1 text-sm"
            value={src}
            onChange={(e) => setSrc(e.target.value as "random" | "cat" | "dog" | "fox")}
          >
            <option value="random">Random</option>
            <option value="cat">Cat</option>
            <option value="dog">Dog</option>
            <option value="fox">Fox</option>
          </select>
        </div>
        <div>
          <button className="text-sm underline" onClick={() => fetchOne(src, true)}>Show another</button>
        </div>
        {loading && <div className="text-sm text-gray-500">Loading…</div>}
        {err && <div className="text-sm text-red-600">Error: {err}</div>}
      </div>

      {item?.kind === "image" && (
        <img
          src={item.url}
          alt={item.alt || "Cute animal"}
          loading="lazy"
          decoding="async"
          style={{ maxHeight: 220, maxWidth: 320, objectFit: "cover", borderRadius: 8 }}
        />
      )}
    </div>
  );
}
