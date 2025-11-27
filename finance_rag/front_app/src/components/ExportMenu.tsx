// src/components/ExportMenu.tsx
"use client";

import React, { useState } from "react";
import { downloadSessionExport, type ExportFormat } from "@/lib/download";

export function ExportMenu({ sessionId }: { sessionId: string }) {
  const [busy, setBusy] = useState<ExportFormat | null>(null);

  async function handleExport(fmt: ExportFormat) {
    try {
      setBusy(fmt);
      await downloadSessionExport(sessionId, fmt, {
      });
    } catch (e) {
      const msg = (e as Error)?.message || "Export failed";
      // Replace with your toast if you have one
      alert(msg);
    } finally {
      setBusy(null);
    }
  }

  const Btn = ({
    fmt,
    label,
  }: {
    fmt: ExportFormat;
    label: string;
  }) => (
    <button
      type="button"
      onClick={() => handleExport(fmt)}
      disabled={busy !== null}
      aria-busy={busy === fmt}
      aria-label={`Export conversation as ${label}`}
      className="px-3 py-2 rounded-xl border shadow-sm text-sm hover:bg-gray-50 disabled:opacity-50"
    >
      {busy === fmt ? `Exporting ${label}…` : label}
    </button>
  );

  return (
    <div className="flex gap-2 flex-wrap items-center">
      <Btn fmt="pdf"  label="PDF" />
      <Btn fmt="md"   label="Markdown" />
      <Btn fmt="csv"  label="CSV" />
      <Btn fmt="json" label="JSON" />
      <Btn fmt="txt"  label="Text" />
    </div>
  );
}
