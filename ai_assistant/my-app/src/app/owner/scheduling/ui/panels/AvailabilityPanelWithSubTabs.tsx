"use client";

import React, { useEffect, useMemo, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import WeeklyAvailabilityPanel from "./WeeklyAvailabilityPanel";
import OpeningsPanel from "./OpeningsPanel";
import TimeOffPanel from "./TimeOffPanel";
import { SubTabButton } from "../components/Tabs";

type SubTabKey = "weekly" | "openings" | "timeoff";

type Props = {
  tz?: string;
  initialSubTab?: SubTabKey;
  disableUrlSync?: boolean;
  hideTabs?: boolean;
};

export default function AvailabilityPanelWithSubTabs({
  tz,
  initialSubTab,
  disableUrlSync = false,
  hideTabs = false,
}: Props) {
  const router = useRouter();
  const sp = useSearchParams();
  const [active, setActive] = useState<SubTabKey>(initialSubTab ?? "weekly");

  useEffect(() => {
    if (disableUrlSync) return;
    const urlTab = (sp.get("subtab") as SubTabKey) || "weekly";
    if (urlTab !== active) {
      setActive(urlTab);
    }
  }, [disableUrlSync, sp, active]);

  useEffect(() => {
    if (!disableUrlSync) return;
    if (!initialSubTab) return;
    if (initialSubTab !== active) {
      setActive(initialSubTab);
    }
  }, [disableUrlSync, initialSubTab, active]);

  const handleSelect = (k: SubTabKey) => {
    setActive(k);
    if (disableUrlSync) return;
    const params = new URLSearchParams(sp?.toString() ?? "");
    params.set("subtab", k);
    router.replace(`?${params.toString()}`);
  };

  const body = useMemo(() => {
    switch (active) {
      case "openings":
        return <OpeningsPanel tz={tz} hideHeaders={hideTabs} />;
      case "timeoff":
        return <TimeOffPanel tz={tz} hideHeaders={hideTabs} />;
      case "weekly":
      default:
        // Forward hideTabs to hide the preview in sidebar card context
        return <WeeklyAvailabilityPanel tz={tz} hidePreview={hideTabs} />;
    }
  }, [active, tz]);

  return (
    <div className="space-y-6">
      {!hideTabs && (
        <div className="flex gap-2">
          <SubTabButton active={active === "weekly"} onClick={() => handleSelect("weekly")}>
            Weekly availability
          </SubTabButton>
          <SubTabButton active={active === "openings"} onClick={() => handleSelect("openings")}>
            Quick openings
          </SubTabButton>
          <SubTabButton active={active === "timeoff"} onClick={() => handleSelect("timeoff")}>
            Time off
          </SubTabButton>
        </div>
      )}
      <div>{body}</div>
    </div>
  );
}
