// src/app/owner/scheduling/ui/OwnerSchedulingClient.tsx
"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import { SessionProvider, useSession } from "next-auth/react";
import AvailabilityPanelWithSubTabs from "./panels/AvailabilityPanelWithSubTabs";
import HomePanel from "./panels/HomePanel";
import AppointmentsPanel from "./panels/AppointmentsPanel";
import OwnerCreateGroupAppt from "./panels/OwnerCreateGroupAppt";

type SidebarAction = "availability" | "opening" | "timeoff" | "appointment" | "group";

const ACTION_OPTIONS: Array<{
  id: SidebarAction;
  label: string;
  description: string;
}> = [
  {
    id: "availability",
    label: "Weekly opening",
    description: "Set availability for specific days and recurring windows.",
  },
  {
    id: "opening",
    label: "One-off opening",
    description: "Offer a standalone slot outside your regular schedule.",
  },
  {
    id: "timeoff",
    label: "Time off",
    description: "Block personal time so clients cannot book.",
  },
  {
    id: "appointment",
    label: "Appointment",
    description: "Book a lesson directly on behalf of a client.",
  },
  {
    id: "group" as any,
    label: "Group lesson",
    description: "Book a group lesson for multiple clients.",
  },
];

export default function OwnerSchedulingClient() {
  return (
    <SessionProvider>
      <OwnerSchedulingInner />
    </SessionProvider>
  );
}

function OwnerSchedulingInner() {
  const { data: session } = useSession();
  const tz = session?.user?.timezone;
  const [menuOpen, setMenuOpen] = useState(false);
  const [sidebarAction, setSidebarAction] = useState<SidebarAction | null>(null);
  const menuRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!menuOpen) return;
    const handleClick = (event: MouseEvent) => {
      if (!menuRef.current) return;
      if (!menuRef.current.contains(event.target as Node)) {
        setMenuOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, [menuOpen]);

  useEffect(() => {
    if (!menuOpen) return;
    const handleKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setMenuOpen(false);
      }
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [menuOpen]);

  useEffect(() => {
    if (!sidebarAction) return;
    const handleKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setSidebarAction(null);
      }
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [sidebarAction]);

  return (
    <div className="relative p-4 sm:p-6 space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <h1 className="text-xl sm:text-2xl font-semibold">Owner Scheduling</h1>
        <div className="relative" ref={menuRef}>
          <button
            type="button"
            onClick={() => setMenuOpen((v) => !v)}
            className="inline-flex items-center gap-2 rounded-full border border-gray-300 bg-white px-4 py-2 text-sm font-medium shadow-sm hover:bg-gray-50"
          >
            <span className="text-lg leading-none">+</span>
            <span>Add</span>
          </button>
          {menuOpen && (
            <div className="absolute right-0 z-20 mt-2 w-80 overflow-hidden rounded-lg border bg-white shadow-lg">
              <div className="py-2">
                {ACTION_OPTIONS.map(({ id, label, description }) => (
                  <button
                    key={id}
                    type="button"
                    onClick={() => {
                      setSidebarAction(id);
                      setMenuOpen(false);
                    }}
                    className="flex w-full flex-col items-start gap-1 px-4 py-2 text-left hover:bg-gray-50"
                  >
                    <span className="text-sm font-medium text-gray-900">{label}</span>
                    <span className="text-xs text-gray-500">{description}</span>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      <HomePanel tz={tz} />

      {sidebarAction && (
        <ActionSidebar
          action={sidebarAction}
          tz={tz}
          onClose={() => setSidebarAction(null)}
        />
      )}
    </div>
  );
}

function ActionSidebar({
  action,
  tz,
  onClose,
}: {
  action: SidebarAction;
  tz?: string;
  onClose: () => void;
}) {
  const content = useMemo(() => {
    switch (action) {
      case "availability":
        return {
          title: "Weekly opening",
          description: "Create availability blocks and recurring windows without leaving the calendar.",
          body: (
            <AvailabilityPanelWithSubTabs
              tz={tz}
              initialSubTab="weekly"
              disableUrlSync
              hideTabs
            />
          ),
        };
      case "opening":
        return {
          title: "One-off opening",
          description: "Offer a specific day and time for bookings.",
          body: (
            <AvailabilityPanelWithSubTabs
              tz={tz}
              initialSubTab="openings"
              disableUrlSync
              hideTabs
            />
          ),
        };
      case "timeoff":
        return {
          title: "Time off",
          description: "Hold time on your calendar so clients cannot book over it.",
          body: (
            <AvailabilityPanelWithSubTabs
              tz={tz}
              initialSubTab="timeoff"
              disableUrlSync
              hideTabs
            />
          ),
        };
      case "appointment":
        return {
          title: "Appointment",
          description: "Book a lesson directly and see upcoming sessions.",
          body: <AppointmentsPanel tz={tz} />,
        };
      case "group":
        return {
          title: "Group lesson",
          description: "Create a group lesson for multiple clients.",
          body: <OwnerCreateGroupAppt />,
        };
      default:
        return null;
    }
  }, [action, tz]);

  if (!content) return null;

  // Centered modal (pop-up) instead of sidebar
  return (
    <div className="fixed inset-0 z-50 overflow-y-auto">
      <div className="min-h-full flex items-center justify-center p-4 bg-black/40" onClick={onClose}>
        <div
          className="w-full max-w-4xl rounded-xl bg-white shadow-2xl border max-h-[90vh] flex flex-col min-h-0"
          onClick={(e) => e.stopPropagation()}
        >
          <div className="flex items-start justify-between border-b px-10 py-4">
            <div className="space-y-1">
              <h2 className="text-xl font-semibold text-gray-900">{content.title}</h2>
              {content.description && (
                <p className="text-sm text-gray-500">{content.description}</p>
              )}
            </div>
            <button
              type="button"
              onClick={onClose}
              className="rounded-md p-2 text-gray-500 hover:bg-gray-100 hover:text-gray-700"
              aria-label="Close modal"
            >
              <span className="text-xl leading-none" aria-hidden="true">&times;</span>
            </button>
          </div>
          <div className="flex-1 min-h-0 overflow-y-auto px-10 py-6" style={{ WebkitOverflowScrolling: 'touch' }}>
            {content.body}
          </div>
        </div>
      </div>
    </div>
  );
}
