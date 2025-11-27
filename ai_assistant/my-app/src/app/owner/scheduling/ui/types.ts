export * from "./types/index";

export type RbcResourceType = "appointment" | "opening" | "time_off" | "availability";
export type ApptStatus = "booked" | "completed" | "canceled";

export type RbcEvent = {
  id: string;
  title?: string;
  start: Date | string;    // RBC gives Date; your API may give string during mapping
  end: Date | string;
  resource?: {
    type: RbcResourceType;
    status?: ApptStatus | null;
    client_name?: string | null;
    client_email?: string | null;
    note?: string | null;
    // owner-only fields:
    owner_note?: string | null;
    client_note?: string | null;
    paid?: boolean | null;
    late?: boolean | null;
    no_show?: boolean | null;
    amount_paid_cents?: number | null;
    labels?: string[] | null;
  };
};

export type EditModalProps = {
  event: RbcEvent;
  onClose: () => void;
  onUpdated: () => void; // e.g. refresh after save/delete
};
