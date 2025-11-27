// src/components/TimezoneSelect.tsx
"use client";

import React from "react";

type Props = {
  value: string;
  onChange: (v: string) => void;
  showUseSystemButton?: boolean;
};

export default function TimezoneSelect({ value, onChange, showUseSystemButton }: Props) {
  const sysTz = Intl.DateTimeFormat().resolvedOptions().timeZone;

  // Build a list (example uses supportedValuesOf if present + sorts)
  const supported = (Intl as unknown as { supportedValuesOf?: (k: string) => string[] })
    .supportedValuesOf?.("timeZone");
  const options = (supported && supported.length ? supported : [sysTz, "UTC"]).slice().sort();

  return (
    <div className="grid gap-2">
      <label className="text-sm opacity-80">Timezone</label>
      <select
        className="border rounded p-2"
        value={value}
        onChange={(e) => onChange(e.target.value)}
      >
        {options.map((z) => (
          <option key={z} value={z}>{z}</option>
        ))}
      </select>

      {showUseSystemButton && (
        <button
          type="button"
          className="text-sm underline self-start"
          onClick={() => onChange(sysTz)}
          disabled={value === sysTz}
          title={sysTz}
        >
          Use my computer time zone
        </button>
      )}
    </div>
  );
}
