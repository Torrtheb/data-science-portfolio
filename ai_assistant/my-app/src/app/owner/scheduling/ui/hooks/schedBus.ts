// src/app/owner/scheduling/ui/hooks/schedBus.ts
import { useEffect } from "react";

const SCHED_EVENT = "sched-changed";

export function emitSchedChanged() {
  document.dispatchEvent(new CustomEvent(SCHED_EVENT));
}

export function useOnSchedChanged(handler: () => void) {
  useEffect(() => {
    const fn = () => handler();
    document.addEventListener(SCHED_EVENT, fn);
    return () => document.removeEventListener(SCHED_EVENT, fn);
  }, [handler]);
}
