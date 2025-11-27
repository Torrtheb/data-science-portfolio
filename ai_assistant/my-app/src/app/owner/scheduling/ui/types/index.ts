// src/app/owner/scheduling/ui/types/index.ts
export type ClientRow = { id?: string; name?: string; email?: string };

export type TabKey = "home" | "availability" | "appointments" | "messages";
export type AvailabilitySubTab = "weekly" | "openings" | "timeoff";

export type EditModalProps = {
  event: RbcEvent;
  onClose: () => void;
  onUpdated: () => void;
};

// Keep this in sync with your calendar event usage
export type RbcEvent = {
  id: string;
  title: string;
  start: Date;
  end: Date;
  resource?: {
    type: "appointment" | "opening" | "time_off" | "availability";
    status?: "booked" | "completed" | "canceled" | null;
    client_name?: string | null;
    client_email?: string | null;
    note?: string | null;
    owner_note?: string | null;
    client_note?: string | null;
    paid?: boolean | null;
    late?: boolean | null;
    no_show?: boolean | null;
    amount_paid_cents?: number | null;
    labels?: string[] | null;
  };
};


// …keep your other types here too (EditModalProps, etc.)
