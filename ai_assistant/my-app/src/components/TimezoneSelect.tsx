// src/components/TimezoneSelect.tsx
"use client";

import * as React from "react";
import { getTimeZones, TimeZone } from "@vvo/tzdb";

type Props = {
  value?: string;
  onChange: (iana: string) => void;
  className?: string;
  showUseSystemButton?: boolean; // shows the "Use my computer time zone" button
  label?: string;                // optional label text
};

// tzdb stores offsets in minutes. tzdb uses positive = west of UTC.
// We’ll render as UTC±HH:MM (standard convention for humans).
function fmtOffset(minutesWestOfUTC: number): string {
  // minutesWestOfUTC: e.g., New York in winter ~ 300 (5 hours west)
  const sign = minutesWestOfUTC === 0 ? "±" : minutesWestOfUTC > 0 ? "-" : "+";
  const abs = Math.abs(minutesWestOfUTC);
  const hh = String(Math.floor(abs / 60)).padStart(2, "0");
  const mm = String(abs % 60).padStart(2, "0");
  return `UTC${sign}${hh}:${mm}`;
}

function systemTimeZone(): string {
  try {
    return Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC";
  } catch {
    return "UTC";
  }
}

const US_CA = new Set(["US", "CA"]);

export default function TimezoneSelect({
  value,
  onChange,
  className,
  showUseSystemButton = true,
  label = "Time zone",
}: Props) {
  const sys = React.useMemo(systemTimeZone, []);

  // Build the list once
  const zones = React.useMemo(() => {
    const all: TimeZone[] = getTimeZones();
    // Filter to US + CA
    const filtered = all.filter((z) => US_CA.has(z.countryCode));
    // De-dup by canonical tz name
    const map = new Map<string, TimeZone>();
    for (const z of filtered) {
      if (!map.has(z.name)) map.set(z.name, z);
    }
    // Sort by country, then offset, then name
    const arr = Array.from(map.values()).sort((a, b) => {
      const aCountry = a.countryCode;
      const bCountry = b.countryCode;
      if (aCountry !== bCountry) {
        // Canada first? flip here if you want CA first
        if (aCountry === "US" && bCountry === "CA") return -1;
        if (aCountry === "CA" && bCountry === "US") return 1;
      }
      const off = a.currentTimeOffsetInMinutes - b.currentTimeOffsetInMinutes;
      if (off) return off;
      return a.name.localeCompare(b.name);
    });

    const us = arr.filter((z) => z.countryCode === "US");
    const ca = arr.filter((z) => z.countryCode === "CA");
    return { us, ca };
  }, []);

  // If no value is provided, initialize to the system tz once.
  React.useEffect(() => {
    if (!value) onChange(sys);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="flex flex-col gap-2">
      {label && <label className="text-sm font-medium">{label}</label>}

      <div className="flex items-center gap-2">
        <select
          value={value || sys}
          onChange={(e) => onChange(e.target.value)}
          className={className || "border rounded-md px-3 py-2 w-full"}
        >
          <optgroup label="United States">
            {zones.us.map((z) => (
              <option key={z.name} value={z.name}>
                {z.name} ({fmtOffset(z.currentTimeOffsetInMinutes)})
              </option>
            ))}
          </optgroup>
          <optgroup label="Canada">
            {zones.ca.map((z) => (
              <option key={z.name} value={z.name}>
                {z.name} ({fmtOffset(z.currentTimeOffsetInMinutes)})
              </option>
            ))}
          </optgroup>
        </select>

        {showUseSystemButton && (
          <button
            type="button"
            onClick={() => onChange(sys)}
            className="whitespace-nowrap border rounded px-3 py-2 text-sm"
            title={`Detected: ${sys}`}
          >
            Use my computer time zone
          </button>
        )}
      </div>

      <p className="text-xs text-gray-500">
        Detected: <span className="font-mono">{sys}</span>
      </p>
    </div>
  );
}
