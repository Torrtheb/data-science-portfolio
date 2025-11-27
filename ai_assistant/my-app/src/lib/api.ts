// src/lib/api.ts


// ---- API error helper (friendly messages) ----
export class ApiError extends Error {
  status: number;
  body: unknown;
  constructor(status: number, body: unknown) {
    const b = body as { detail?: string; message?: string } | string | undefined;
    const msg =
      (typeof b === "string" ? b : b?.detail || b?.message) || "Request failed";
    super(msg);
    this.status = status;
    this.body = body;
  }
}
async function ensureOk(r: Response) {
  if (r.ok) return;
  const text = await r.text();
  try {
    throw new ApiError(r.status, JSON.parse(text));
  } catch {
    throw new ApiError(r.status, { detail: text });
  }
}

const _API = process.env.NEXT_PUBLIC_API_BASE_URL || ""; // keep "" for same-origin + Next rewrites
const _withCreds = (init: RequestInit = {}): RequestInit => ({
  credentials: "include",
  ...init,
  headers: {
    "Content-Type": "application/json",
    ...(init.headers || {}),
  },
});

const _back = (path: string, init?: RequestInit) =>
  fetch(`/api/back${path}`, {
    ...init,
    headers: {
      'content-type': 'application/json',
      ...(init?.headers || {}),
    },
    // credentials not needed when hitting same-origin /api/back/*
  });


import { getOrInitCsrfToken } from "@/lib/csrf";

// ---------- Base + fetch helper ----------
export function getBaseUrl() {
  if (typeof window !== "undefined") return ""; // browser can use relative
  if (process.env.NEXTAUTH_URL) return process.env.NEXTAUTH_URL;
  if (process.env.VERCEL_URL) return `https://${process.env.VERCEL_URL}`;
  return "http://localhost:3000"; // dev fallback
}

export async function apiFetch(input: string, init: RequestInit = {}) {
  const base = getBaseUrl();
  const method = (init.method || "GET").toUpperCase();
  const isUnsafe = method !== "GET" && method !== "HEAD" && method !== "OPTIONS";
  const headers = new Headers(init.headers || {});
  if (isUnsafe && typeof window !== "undefined") {
    const tok = getOrInitCsrfToken();
    if (tok) headers.set("X-CSRF-Token", tok);
  }
  return fetch(`${base}${input}`, { ...init, headers });
}

const BASE = "/api/back/api";

function withTZ(path: string, tz?: string) {
  if (!tz) return path;
  const u = new URL(path, typeof window === "undefined" ? "http://localhost:3000" : window.location.origin);
  u.searchParams.set("tz", tz);
  // keep relative path for the browser
  return u.pathname + u.search;
}


// ---------- Agent Tools ----------
export async function listAgentTools() {
  const res = await apiFetch(`${BASE}/agent/tools`, { cache: "no-store" });
  if (!res.ok) throw new Error(`List tools failed: ${res.status}`);
  return res.json() as Promise<Array<{ name: string; description: string; parameters: unknown }>>;
}

export async function callAgentTool(tool: string, args: Record<string, unknown>) {
  const res = await apiFetch(`${BASE}/agent/tools/call`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ tool, arguments: args }),
  });
  if (!res.ok) throw new Error((await res.text()) || `Call ${tool} failed: ${res.status}`);
  return res.json();
}

// ---------- Scheduling: Availability ----------
export type AvailabilityRule = {
  id: string;
  weekday: number;
  start_local: string; // "HH:MM:SS" from backend
  end_local: string;
  slot_minutes: number;
  buffer_minutes: number;
};

export async function listAvailability(): Promise<AvailabilityRule[]> {
  const r = await apiFetch(`${BASE}/scheduling/availability`, { cache: "no-store" });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function addAvailability(
  input: {
    weekday: number;
    start_local: string; // "HH:MM"
    end_local: string;   // "HH:MM"
    slot_minutes: number;
    buffer_minutes: number;
  },
  tz?: string
) {
  const r = await apiFetch(withTZ(`${BASE}/scheduling/availability`, tz), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function deleteAvailability(ruleId: string, tz?: string) {
  const r = await apiFetch(withTZ(`${BASE}/scheduling/availability/${encodeURIComponent(ruleId)}`, tz), {
    method: "DELETE",
  });
  if (!r.ok) throw new Error(await r.text());
}

export async function bulkDeleteAvailability(opts?: { weekday?: number }) {
  const qs = opts?.weekday !== undefined ? `?weekday=${opts.weekday}` : "";
  const r = await apiFetch(`${BASE}/scheduling/availability${qs}`, { method: "DELETE" });
  if (!r.ok) throw new Error(await r.text());
  return r.json() as Promise<{ ok: boolean; deleted: number }>;
}

// ---------- Scheduling: Time Off ----------
export type TimeOff = {
  id: string;
  start_utc: string;
  end_utc: string;
  start_local?: string | null;
  end_local?: string | null;
  timezone?: string | null;
  note?: string | null;
};

export async function listTimeOff(tz?: string): Promise<TimeOff[]> {
  const r = await apiFetch(withTZ(`${BASE}/scheduling/timeoff`, tz), { cache: "no-store" });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function addTimeOff(input: {
  start: string; // ISO with tz
  end: string;   // ISO with tz
  note?: string | null;
}) {
  const r = await apiFetch(`${BASE}/scheduling/timeoff`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function updateTimeOff(id: string, input: {
  start?: string; // ISO with tz or owner-local
  end?: string;   // ISO with tz or owner-local
  note?: string | null;
}) {
  const r = await apiFetch(`${BASE}/scheduling/timeoff/${encodeURIComponent(id)}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

// (duplicate guard) — updateTimeOff is defined above

export async function deleteTimeOff(id: string) {
  const r = await apiFetch(`${BASE}/scheduling/timeoff/${encodeURIComponent(id)}`, { method: "DELETE" });
  if (!r.ok) throw new Error(await r.text());
}

// ---------- Scheduling: Appointments ----------
export type Appointment = {
  id: string;
  start_utc: string;
  end_utc: string;
  start_local?: string | null;
  end_local?: string | null;
  timezone?: string | null;
  status: "booked" | "completed" | "canceled";
  client?: { id: string; name?: string | null; email?: string | null } | null;
  client_account_id?: number | null;
  person?: { id?: number; name?: string | null; email?: string | null } | null;
  group_id?: string | null;

  // --- metadata (owner/admin) ---
  owner_note?: string | null;
  client_note?: string | null;
  cancel_reason?: string | null;

  // legacy booleans kept for compatibility with existing UI
  paid?: boolean | null;
  late?: boolean | null;
  no_show?: boolean | null;
  amount_paid_cents?: number | null;
  price_override_cents?: number | null;
  labels?: string[] | null;

  // NEW: richer fields for the editor
  attendance_status?: "unknown" | "attended" | "late" | "no_show";
  late_minutes?: number | null;
  payment_status?: "unpaid" | "paid" | "refunded" | "waived" | null;
  paid_at?: string | null;
  bundle_id?: number | null;
};




// --- For filtered list view (lightweight rows) ---
export type AppointmentLite = {
  id: string;
  start_utc: string; // ISO
  end_utc: string;   // ISO
  status: "booked" | "completed" | "canceled" | string;
  client?: { id?: string; name?: string | null; email?: string | null } | null;
};


export async function listAppointments(tz?: string): Promise<Appointment[]> {
  const r = await apiFetch(withTZ(`${BASE}/scheduling/appointments`, tz), { cache: "no-store" });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function cancelAppointment(id: string, message?: string) {
  const qs = message ? `?message=${encodeURIComponent(message)}` : "";
  const r = await apiFetch(`${BASE}/scheduling/appointments/${encodeURIComponent(id)}/cancel${qs}`, {
    method: "POST",
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}


export async function updateAppointment(
  apptId: string,
  payload: {
    client_email?: string;
    start_local?: string;
    duration_minutes?: number;
    status?: "booked" | "completed" | "canceled";
    allow_override?: boolean;
    message?: string;
    lesson_person_id?: number;
    lesson_person_name?: string;
    // --- owner/admin-side metadata (your backend can ignore unknowns safely) ---
    owner_note?: string;
    paid?: boolean;
    late?: boolean;
    no_show?: boolean;
    amount_paid_cents?: number;
    labels?: string[];
  }
) {
  const r = await apiFetch(`${BASE}/scheduling/appointments/${encodeURIComponent(apptId)}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  await ensureOk(r);
  return r.json();
}


// ---------- Owner: appointment metadata (post-visit) ----------
export type AttendanceStatus = "unknown" | "attended" | "late" | "no_show";
export type PaymentStatus = "unpaid" | "paid" | "refunded" | "waived";

export async function ownerUpdateAppointment(
  apptId: string,
  payload: {
    status?: "booked" | "completed" | "canceled";
    attendance_status?: AttendanceStatus;
    late_minutes?: number;
    owner_private_note?: string;
    payment_status?: PaymentStatus;
    paid_at?: string;
    bundle_id?: number | 0;
    amount_paid_cents?: number;          // ← NEW
    price_override_cents?: number | null; // ← NEW  
  },
  opts?: { apply_wallet_now?: boolean; restore_wallet_now?: boolean }
) {
  const u = new URL(`${BASE}/scheduling/appointments/${encodeURIComponent(apptId)}/owner`, typeof window === 'undefined' ? 'http://localhost:3000' : window.location.origin)
  if (opts?.apply_wallet_now) u.searchParams.set('apply_wallet_now', 'true')
  if (opts?.restore_wallet_now) u.searchParams.set('restore_wallet_now', 'true')
  const r = await apiFetch(u.pathname + u.search, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  await ensureOk(r);
  return r.json() as Promise<{ ok: boolean; notice?: string }>;
}

// Owner: hard delete an appointment (restores any wallet effects first)
export async function ownerDeleteAppointment(apptId: string) {
  const r = await apiFetch(`${BASE}/scheduling/appointments/${encodeURIComponent(apptId)}`, {
    method: 'DELETE',
  })
  await ensureOk(r)
  return r.json() as Promise<{ ok: boolean }>
}

// Owner: get appointment details for owner drawer
export async function ownerGetAppointmentDetails(apptId: string): Promise<{
  id: string;
  owner_private_note?: string | null;
  attendance_status?: "unknown" | "attended" | "late" | "no_show" | null;
  late_minutes?: number;
  payment_status?: "unpaid" | "paid" | "refunded" | "waived" | null;
  price_override_cents?: number | null;
  start_utc: string;
  end_utc: string;
  cancel_reason?: string | null;
  group_id?: string | null;
}> {
  const r = await apiFetch(`${BASE}/scheduling/appointments/${encodeURIComponent(apptId)}/owner`, { cache: "no-store" });
  await ensureOk(r);
  return r.json();
}

export async function ownerGetGroupDetails(groupId: string): Promise<{
  group_id: string;
  start_utc: string;
  end_utc: string;
  attendees: Array<{
    appointment_id: string;
    person_id?: number | null;
    name: string;
    status: "booked" | "completed" | "canceled" | string;
    payment_status: "paid" | "partial" | "unpaid" | "bundle" | "unknown";
    price_cents?: number | null;
    paid_cash_cents: number;
    bundle_applied_cents: number;
    owed_cents: number;
  }>;
}> {
  const r = await apiFetch(`${BASE}/scheduling/appointments/group/${encodeURIComponent(groupId)}`, { cache: "no-store" });
  await ensureOk(r);
  return r.json();
}


// ---------- Owner: Prepaid bundles ----------
export type PrepaidBundle = {
  id: number;
  client_id: string;
  name: string;
  total_credits: number;
  remaining_credits: number;
  remaining_balance_cents?: number | null;
  price_cents: number;
  currency: string; // "USD" etc.
  expires_at?: string | null;
  created_at?: string | null;
};

export async function ownerListBundles(clientId: string): Promise<PrepaidBundle[]> {
  const r = await apiFetch(`${BASE}/scheduling/owner/clients/${encodeURIComponent(clientId)}/bundles`, { cache: "no-store" });
  await ensureOk(r);
  return r.json();
}

export async function ownerListBundleLedger(clientId: string, bundleId: number, limit = 5, filters?: { date_from?: string; date_to?: string }) {
  const params = new URLSearchParams();
  params.set('limit', String(limit));
  if (filters?.date_from) params.set('date_from', filters.date_from);
  if (filters?.date_to) params.set('date_to', filters.date_to);
  const r = await apiFetch(`${BASE}/scheduling/owner/clients/${encodeURIComponent(clientId)}/bundles/${bundleId}/ledger?${params.toString()}`, { cache: 'no-store' })
  await ensureOk(r)
  return r.json() as Promise<Array<{ event: string; delta_credits: number; amount_cents: number; appointment_id?: string; note?: string | null; created_at: string }>>
}

export async function ownerCreateBundle(input: {
  client_id: string;
  name: string;
  total_credits: number;  // keep for compat; use 0 for wallet
  price_cents: number;    // deposit amount
  currency: string;
  expires_at?: string | null;
}) {
  const r = await apiFetch(`${BASE}/scheduling/owner/clients/${encodeURIComponent(input.client_id)}/bundles`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
  await ensureOk(r);
  return r.json() as Promise<PrepaidBundle>;
}

export async function ownerMigrateBundlesToWallet(clientId: string) {
  const r = await apiFetch(`${BASE}/scheduling/owner/clients/${encodeURIComponent(clientId)}/bundles/migrate-to-wallet`, {
    method: 'POST'
  })
  await ensureOk(r)
  return r.json() as Promise<{ ok: boolean; migrated: number; deposited_cents: number }>
}

export async function ownerTopUpBundle(clientId: string, bundleId: number, amount_cents: number, note?: string) {
  const r = await apiFetch(`${BASE}/scheduling/owner/clients/${encodeURIComponent(clientId)}/bundles/${bundleId}/topup`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ amount_cents, note })
  })
  await ensureOk(r)
  return r.json() as Promise<{ ok: boolean }>
}

export async function ownerAdjustWallet(
  clientId: string,
  bundleId: number,
  amount_cents: number,
  note?: string,
): Promise<{ ok: boolean; balance_cents: number }> {
  const r = await apiFetch(`${BASE}/scheduling/owner/clients/${encodeURIComponent(clientId)}/bundles/${bundleId}/adjust`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ amount_cents, note }),
  })
  await ensureOk(r)
  return r.json()
}



export async function bookAppointment(input: {
  client_name: string;
  client_email: string;
  start_local: string; // naive local ISO, e.g. "2025-09-16T09:00:00"
  message?: string;
}) {
  const r = await apiFetch(`${BASE}/scheduling/book`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function adminCreateAppointment(input: {
  client_name: string;
  client_email: string;
  start_local: string;   // naive local ISO "YYYY-MM-DDTHH:mm:ss" or with tz
  duration_minutes?: number;
  status?: "booked" | "completed" | "canceled";
  allow_override?: boolean;
  message?: string;
  lesson_person_id?: number;
  lesson_person_name?: string;
}) {
  const r = await apiFetch(`${BASE}/scheduling/appointments/admin-create`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json() as Promise<{ ok: boolean; appointment_id: string; status: string }>;
}

export type AdminCreateRecurringAppointmentInput = {
  client_name: string;
  client_email: string;
  start_local: string;
  duration_minutes?: number;
  status?: "booked" | "completed" | "canceled";
  repeat_every_weeks?: number;
  occurrences?: number;
  until_date?: string;
  allow_override?: boolean;
  confirm_if_conflicts?: boolean;
  message?: string;
  lesson_person_id?: number;
  lesson_person_name?: string;
};

export type AdminCreateRecurringAppointmentResponse = {
  ok: boolean;
  count: number;
  appointments: Array<{
    appointment_id: string;
    status: string;
    start_local: string;
  }>;
};

export async function adminCreateRecurringAppointments(
  input: AdminCreateRecurringAppointmentInput
): Promise<AdminCreateRecurringAppointmentResponse> {
  const r = await apiFetch(`${BASE}/scheduling/appointments/admin-create/recurring`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
  await ensureOk(r);
  return r.json();
}

// --- NEW: owner broadcast email ---
// Map backend ClientAccountSummary[] → plain { id, name?, email? }[]
export async function listOwnerClients(): Promise<Array<{ id: string; name?: string; email?: string }>> {
  const res = await apiFetch(`${BASE}/scheduling/owner/clients`, { cache: "no-store" });
  await ensureOk(res);
  const rows = (await res.json()) as Array<{
    account_id: number;
    client_user_id: string;
    client_email?: string | null;
    client_name?: string | null;
    name?: string | null;        // account display name (unused in UI)
    people_count: number;
  }>;

  // IMPORTANT: coerce null → undefined so it matches UI's ClientRow (name?: string, email?: string)
  return rows.map((r) => ({
    id: r.client_user_id,
    name: (r.client_name ?? r.name) ?? undefined,
    email: (r.client_email ?? undefined) ?? undefined,
  }));
}


// --- Filtered owner appointments (when + client filters) ---
export async function listOwnerAppointmentsByFilter(params: {
  when?: "past" | "today" | "future";
  client_ids?: string[];
  client_emails?: string[];
  tz?: string; // optional override if you want to pin a specific tz
}): Promise<AppointmentLite[]> {
  const u = new URL(
    `${BASE}/scheduling/appointments`,
    typeof window === 'undefined' ? 'http://localhost:3000' : window.location.origin,
  );
  if (params.when) u.searchParams.set("when", params.when);
  if (params.tz)   u.searchParams.set("tz", params.tz);
  (params.client_ids ?? []).forEach(id => u.searchParams.append("client_ids", id));
  (params.client_emails ?? []).forEach(e => u.searchParams.append("client_emails", e));
  const r = await apiFetch(u.pathname + u.search, { cache: "no-store" });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}


export async function ownerBroadcastEmail(payload: {
  subject: string;
  text: string;
  client_user_ids?: string[];
}) {
  const body = {
    ...payload,
    confirm_send: true, // required by backend for non-preview sends
  };
  const res = await apiFetch(`${BASE}/scheduling/owner/email/broadcast`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    // Try to parse backend detail for friendlier messaging.
    let friendly = `Broadcast failed (${res.status}).`;
    try {
      const data = await res.json();
      const detail = typeof data?.detail === "string" ? data.detail : "";
      if (detail.includes("confirm_send")) {
        friendly = "Broadcast blocked: confirm_send is required. Please try again (we auto-confirm), or enable SMTP in production.";
      } else if (detail) {
        friendly = detail;
      }
    } catch {
      const txt = await res.text().catch(() => "");
      if (txt.includes("confirm_send")) {
        friendly = "Broadcast blocked: confirm_send is required. Please try again (we auto-confirm), or enable SMTP in production.";
      } else if (txt) {
        friendly = txt;
      }
    }
    throw new Error(friendly);
  }
  return res.json();
}


// ---------- Scheduling: Special Openings ----------
export type SpecialOpening = {
  id: string;
  start_utc: string;
  end_utc: string;
  start_local?: string | null;
  end_local?: string | null;
  timezone?: string | null;
  slot_minutes: number;
  buffer_minutes: number;
  note?: string | null;
};

export async function listOpenings(tz?: string): Promise<SpecialOpening[]> {
  const r = await apiFetch(withTZ(`${BASE}/scheduling/openings`, tz), { cache: "no-store" });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function addOpening(input: {
  start: string;        // ISO with tz
  end: string;          // ISO with tz
  slot_minutes: number;
  buffer_minutes: number;
  note?: string | null;
  allow_overlap?: boolean;
}) {
  const r = await apiFetch(`${BASE}/scheduling/openings`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function updateOpening(
  id: string,
  payload: {
    start?: string; // ISO with tz OR owner-local naive
    end?: string;
    slot_minutes?: number;
    buffer_minutes?: number;
    note?: string | null;
    allow_overlap?: boolean;
  }
) {
  const r = await apiFetch(`${BASE}/scheduling/openings/${encodeURIComponent(id)}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function deleteOpening(id: string) {
  const r = await apiFetch(`${BASE}/scheduling/openings/${encodeURIComponent(id)}`, { method: "DELETE" });
  if (!r.ok) throw new Error(await r.text());
}

// Recurring weekly openings (create as individual specials)
export async function addRecurringOpenings(input: {
  weekday: number;         // 0=Mon..6=Sun
  start_hhmm: string;      // HH:MM (owner-local)
  end_hhmm: string;        // HH:MM (owner-local)
  slot_minutes: number;
  buffer_minutes?: number;
  weeks?: number;          // default handled by backend
  start_date?: string;     // YYYY-MM-DD (owner-local) optional
  note?: string | null;
}, tz?: string) {
  const r = await apiFetch(withTZ(`${BASE}/scheduling/openings/recurring`, tz), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json() as Promise<SpecialOpening[]>;
}

// ---------- Slots + Calendar ----------
export type Slot = { start: string; end: string }; // owner-local ISO strings from backend

export async function getSlotsForDate(yyyyMmDd: string, tz?: string): Promise<Slot[]> {
  const r = await apiFetch(withTZ(`${BASE}/scheduling/slots`, tz), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ date: yyyyMmDd }),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function getSlotsDebug(yyyyMmDd: string, tz?: string) {
  const r = await apiFetch(withTZ(`${BASE}/scheduling/slots/debug?date=${encodeURIComponent(yyyyMmDd)}`, tz), {
    cache: "no-store",
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json() as Promise<{
    owner_id: string;
    date: string;
    weekly_pairs: { start: string; end: string }[];
    special_pairs: { start: string; end: string; opening_id: string }[];
    timeoffs: { start_utc: string; end_utc: string; note?: string | null }[];
    final_slots: { start: string; end: string }[];
  }>;
}

// ---------- Calendar range ----------
export type DayCalendar = {
  rules: { start: string; end: string; slot_minutes: number; buffer_minutes: number }[];
  openings: { start: string; end: string; slot_minutes: number; buffer_minutes: number; note?: string | null }[];
  timeoffs: { start: string; end: string; note?: string | null }[];
  appointments: { start: string; end: string; status: string; client_id?: string | null; person_id?: number | null }[];
  slots: Slot[];
};

export async function getCalendarRange(start: string, days = 14): Promise<Record<string, DayCalendar>> {
  const r = await apiFetch(`${BASE}/scheduling/calendar/range?start=${encodeURIComponent(start)}&days=${days}`, {
    cache: "no-store",
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

// ---------- Owner: Holidays (range) ----------
export type OwnerHoliday = { date: string; name: string; start_utc: string; end_utc: string };
export async function getOwnerHolidays(start: string, end: string, tz?: string): Promise<OwnerHoliday[]> {
  const path = withTZ(`${BASE}/scheduling/owner/holidays?start=${encodeURIComponent(start)}&end=${encodeURIComponent(end)}`, tz);
  const r = await apiFetch(path, { cache: "no-store" });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

// Public: holidays
export async function getPublicHolidays(start: string, end: string, tz?: string): Promise<OwnerHoliday[]> {
  const path = withTZ(`${BASE}/scheduling/public/holidays?start=${encodeURIComponent(start)}&end=${encodeURIComponent(end)}`, tz);
  const r = await apiFetch(path, { cache: "no-store" });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

// ---------- Fun: Cute animal welcome ----------
export type FunWelcome = { kind: "image"; url: string; alt: string; source: "cat" | "dog" | "fox" };
export async function getFunWelcome(source?: "cat" | "dog" | "fox" | "random", fresh?: boolean): Promise<FunWelcome> {
  const params = new URLSearchParams();
  if (source) params.set("source", source);
  if (fresh) params.set("fresh", "1");
  const qs = params.toString();
  const r = await apiFetch(`${BASE}/fun/welcome${qs ? `?${qs}` : ""}`, { cache: "no-store" });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

// ---------- Owner snapshot + filtered appointments ----------
export type SnapshotEvent = {
  id: string;
  type: "appointment" | "opening" | "time_off";
  title: string;
  start: string;
  end: string;
  status?: string | null;
};
export type OwnerSnapshot = {
  tz: string;
  start: string;
  end: string;
  events: SnapshotEvent[];
};

export async function getOwnerSnapshot(
  scope: "today" | "week" | "month" = "week",
  tz?: string
): Promise<OwnerSnapshot> {
  const r = await apiFetch(withTZ(`${BASE}/scheduling/owner/snapshot?scope=${scope}`, tz), {
    cache: "no-store",
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}


// src/lib/api.ts
export async function getOwnerSettings(): Promise<{
  appt_edge_buffer_min: number;
  auto_apply_wallet_on_book: boolean;
  wallet_deposits_as_paid: boolean;
  group_price_60_cents?: number;
}> {
  const r = await apiFetch(`${BASE}/scheduling/owner/settings`, { cache: "no-store" });
  await ensureOk(r);
  return r.json();
}

export async function updateOwnerSettings(p: {
  appt_edge_buffer_min: number;
  auto_apply_wallet_on_book?: boolean;
  wallet_deposits_as_paid?: boolean;
  group_price_60_cents?: number;
}) {
  const r = await apiFetch(`${BASE}/scheduling/owner/settings`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(p),
  });
  await ensureOk(r);
  return r.json();
}


export async function getOwnerAppointments(
  filter:
    | "today"
    | "this_week"
    | "this_month"
    | "cancelled"
    | "completed_last_week"
    | "completed_last_month"
    | "completed_all_time",
  tz?: string
) {
  const r = await apiFetch(withTZ(`${BASE}/scheduling/owner/appointments?filter=${filter}`, tz), {
    cache: "no-store",
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json() as Promise<{
    rows: {
      id: string;
      title: string;
      start: string;
      end: string;
      status: string;
      client_name?: string | null;
      client_email?: string | null;
    }[];
  }>;
}


// ---------- Client Profile ----------
export type ClientEmail = { id: number; email: string; is_primary: boolean };
export type ClientProfile = {
  account_id: number;
  name?: string | null;
  phone?: string | null;
  emergency_contact?: string | null;
  emails: ClientEmail[];
  people: { id: number; full_name: string; email?: string | null }[];
};
export type ClientRow = {
  id: string;
  name?: string | null;
  email?: string | null;
};

export async function getClientProfile(): Promise<ClientProfile> {
  const r = await apiFetch(`${BASE}/me/client-profile`, { cache: "no-store" });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function updateClientProfile(input: {
  name?: string | null;
  phone?: string | null;
  emergency_contact?: string | null;
  emails?: { email: string; is_primary?: boolean }[]; // up to 2
}): Promise<ClientProfile> {
  const r = await apiFetch(`${BASE}/me/client-profile`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

// ---------- Owner: Client detail update ----------
export async function ownerUpdateClient(
  accountId: number,
  input: { name?: string | null; phone?: string | null; emergency_contact?: string | null;
           emails?: { email: string; is_primary?: boolean; unsubscribed?: boolean }[]; }
) {
  // Use the owner detail endpoint which returns full ClientAccountDetail
  const r = await apiFetch(`${BASE}/scheduling/owner/clients/${accountId}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

// ====================== Client-facing Scheduling (users) ======================
export type PublicSlot = {
  start: string; // ISO (owner-local)
  end: string;
};

export type UserAppt = {
  id: string;
  start_utc: string;
  end_utc: string;
  status: "booked" | "completed" | "canceled";
  payment_status?: "unpaid" | "paid" | "refunded" | "waived" | "partial" | "unknown" | null;
  client_email?: string | null;
  client_name?: string | null;
};

export async function fetchPublicSlots(dateISO: string, tz?: string) {
  const path = withTZ(`${BASE}/scheduling/public/slots?d=${encodeURIComponent(dateISO)}`, tz);
  const r = await apiFetch(path, { cache: "no-store" });
  if (!r.ok) throw new Error(await r.text());
  return (await r.json()) as PublicSlot[];
}

export async function listMyAppointments(scope: "upcoming" | "history" | "all" = "upcoming", tz?: string): Promise<UserAppt[]> {
  // Correct prefix is /api/client
  const path = withTZ(`${BASE}/client/my-appointments?scope=${scope}`, tz);
  const r = await apiFetch(path, { cache: "no-store" });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}


export async function myBookAppointment(input: {
  start_local: string
  duration_minutes?: number
  message?: string;
  lesson_person_name?: string;
}) {
  const r = await apiFetch(`${BASE}/client/my/appointments`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
  await ensureOk(r);
  return r.json();
}

export async function myBookRecurringAppointments(input: {
  start_local: string;
  duration_minutes: number;
  repeat_every_weeks?: number;
  occurrences?: number;
  until_date?: string; // YYYY-MM-DD
  message?: string;
  lesson_person_name?: string;
}) {
  const r = await apiFetch(`${BASE}/client/my/appointments/recurring`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
  await ensureOk(r);
  return r.json() as Promise<{ ok: boolean; count: number; appointments: Array<{ appointment_id: string; status: string; start_local: string }>; conflicts: Array<{ start_local: string; reason: string }> }>;
}



export async function myCancelAppointment(id: string, message?: string) {
  const r = await apiFetch(`${BASE}/client/my/appointments/${id}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ status: "canceled", ...(message ? { message } : {}) }),
  });
  await ensureOk(r);
  return r.json();
}

export async function myRescheduleAppointment(
  id: string,
  payload: { start_local: string; duration_minutes?: number; message?: string }
) {
  const r = await apiFetch(`${BASE}/client/my/appointments/${id}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  await ensureOk(r);
  return r.json();
}


// src/lib/api.ts
export async function clientListAppointments() {
  const r = await apiFetch(`${BASE}/client/appointments`, { cache: "no-store" });
  await ensureOk(r);
  const data = await r.json() as {
    rows: Array<{
      id: string;
      start_utc: string;
      end_utc: string;
      status: "booked" | "completed" | "canceled" | null;
      payment_status: "unpaid" | "paid" | "refunded" | "waived" | null;
      amount_paid_cents?: number | null;
      duration_minutes?: number | null;
    }>;
  };
  return data.rows;
}



// --- Public pricing/options ---
export type ServiceOption = {
  id: number
  duration_minutes: number
  price_cents: number
  currency: string
  is_active: boolean
}

// Returns: [{ start, end, duration_minutes, price_cents, currency }]
export async function listPublicSlotsPriced(dayISO: string, duration?: number) {
  const qs = new URLSearchParams({ day: dayISO })
  if (duration) qs.set("duration_minutes", String(duration))
  const r = await apiFetch(`${BASE}/scheduling/public/slots-priced?${qs.toString()}`, { cache: "no-store" })
  await ensureOk(r)
  return r.json() as Promise<Array<{ start: string; end: string; duration_minutes: number; price_cents: number; currency: string }>>
}

async function jsonOrText(r: Response) {
  // Throw on non-2xx first so callers don't accidentally parse HTML error pages
  if (!r.ok) {
    const t = await r.text();
    throw new ApiError(r.status, { detail: t });
  }
  const ct = r.headers.get("content-type") || "";
  const text = await r.text();
  if (ct.includes("application/json")) {
    try {
      return JSON.parse(text);
    } catch {
      throw new ApiError(r.status || 500, { detail: `Invalid JSON: ${text.slice(0, 200)}` });
    }
  }
  // Non-JSON success (probably HTML) -> surface the body so you can see it
  throw new ApiError(r.status || 500, { detail: text || "Non-JSON response" });
}

// --- Owner analytics ---
export async function getOwnerPaymentsSummary(args: {
  start: string;
  end: string;
  client_account_ids?: Array<string | number>;   // <-- array
}) {
  const params = new URLSearchParams({ start: args.start, end: args.end });
  for (const id of args.client_account_ids ?? []) {
    params.append("client_account_ids", String(id));      // <-- repeatable param
  }
  const r = await apiFetch(`${BASE}/owner/analytics/payments/summary?${params.toString()}`, { cache: "no-store" });
  await ensureOk(r);
  return r.json();
}

export async function getClientAppointmentsWithPayments() {
  const r = await apiFetch(`${BASE}/client/payments`, { cache: "no-store" });
  return jsonOrText(r);
}

export async function getClientPaymentsFiltered(opts?: {
  date_from?: string; // YYYY-MM-DD
  date_to?: string;   // YYYY-MM-DD
  status?: Array<"booked" | "completed" | "canceled">;
}) {
  const u = new URL(
    `${BASE}/client/payments`,
    typeof window === 'undefined' ? 'http://localhost:3000' : window.location.origin,
  );
  if (opts?.date_from) u.searchParams.set("date_from", opts.date_from);
  if (opts?.date_to) u.searchParams.set("date_to", opts.date_to);
  for (const s of opts?.status ?? []) u.searchParams.append("status", s);
  const r = await apiFetch(u.pathname + u.search, { cache: "no-store" });
  return jsonOrText(r) as Promise<{
    summary: {
      total_appointments: number;
      late_appointments: number;
      paid_appointments: number;
      unpaid_appointments: number;
      total_expected_cents: number;
      total_paid_cents: number;
      total_owed_cents: number;
    };
    rows: Array<{
      id: string;
      start_utc: string;
      duration_minutes: number;
      status: "booked" | "completed" | "canceled";
      lesson_person_name?: string | null;
      is_group?: boolean;
      attendance?: "unknown" | "on_time" | "late" | "no_show" | "attended";
      price_cents?: number | null;
      amount_paid_cents: number;
      bundle_applied_cents?: number;
      payment_status: "paid" | "partial" | "unpaid" | "bundle" | "unknown" | "refunded" | "waived";
    }>;
  }>;
}

// --- Client wallet info ---
export async function getClientWallet(limit = 20, filters?: { date_from?: string; date_to?: string }): Promise<{ balance_cents: number; transactions: Array<{ event: string; amount_cents: number; appointment_id: string; note?: string | null; created_at: string }>; appointments_count: number }>{
  const params = new URLSearchParams();
  params.set('limit', String(limit));
  if (filters?.date_from) params.set('date_from', filters.date_from);
  if (filters?.date_to) params.set('date_to', filters.date_to);
  const r = await apiFetch(`${BASE}/client/wallet?${params.toString()}`, { cache: "no-store" });
  await ensureOk(r);
  return r.json();
}


// ---- Financial analytics DTOs (frontend) ----
export type PaymentStatusFull = "unpaid" | "partial" | "paid" | "refunded" | "waived" | "bundle" | "unknown";

export type AppointmentFinancialRow = {
  id: string;
  start_utc: string;
  end_utc: string;
  client_account_id?: number | null;
  client_label?: string | null;
  lesson_person_name?: string | null;
  lesson_person_email?: string | null;
  is_group?: boolean;
  status: "booked" | "completed" | "canceled";
  duration_minutes: number;
  attendance_status?: AttendanceStatus | null;
  cancel_reason?: string | null;
  price_cents?: number | null;
  paid_cash_cents: number;
  bundle_applied_cents: number;
  owed_cents: number;
  payment_status: PaymentStatusFull;
};

export type FinancialSummary = {
  total_appointments: number;
  total_expected_cents: number;
  total_paid_cents: number;
  total_cash_cents: number;
  total_bundle_cents: number;
  total_owed_cents: number;
  total_wallet_balance_cents?: number;
  total_no_show?: number;
};

function buildQuery(params: Record<string, string | number | (string|number)[] | undefined>) {
  const usp = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v === undefined || v === null) continue;
    if (Array.isArray(v)) {
      for (const item of v) usp.append(k, String(item));
    } else {
      usp.set(k, String(v));
    }
  }
  return usp.toString();
}

// Owner: list per-appointment financial rows
export async function ownerListFinancialAppointments(args: {
  start: string; // YYYY-MM-DD
  end: string;   // YYYY-MM-DD
  status?: Array<"booked" | "completed" | "canceled">;
  payment_status?: Array<PaymentStatusFull | "paid" | "unpaid" | "refunded" | "waived" | "partial" | "bundle">;
  client_account_id?: number;
}): Promise<AppointmentFinancialRow[]> {
  const qs = buildQuery({
    start: args.start,
    end: args.end,
    status: args.status,
    payment_status: args.payment_status,
    client_account_id: args.client_account_id,
  });
  const r = await apiFetch(`${BASE}/owner/analytics/appointments?${qs}`, { cache: "no-store" });
  await ensureOk(r);
  return r.json() as Promise<AppointmentFinancialRow[]>;
}

// Owner: financial summary for current filters
export async function ownerGetFinancialSummary(args: {
  start: string; // YYYY-MM-DD
  end: string;   // YYYY-MM-DD
  status?: Array<"booked" | "completed" | "canceled">;
  payment_status?: Array<PaymentStatusFull | "paid" | "unpaid" | "refunded" | "waived" | "partial" | "bundle">;
  client_account_id?: number;
}): Promise<FinancialSummary> {
  const qs = buildQuery({
    start: args.start,
    end: args.end,
    status: args.status,
    payment_status: args.payment_status,
    client_account_id: args.client_account_id,
  });
  const r = await apiFetch(`${BASE}/owner/analytics/summary?${qs}`, { cache: "no-store" });
  await ensureOk(r);
  return r.json() as Promise<FinancialSummary>;
}

// Owner: update single appointment financials (inline edit)
export async function ownerUpdateAppointmentFinancials(
  apptId: string,
  payload: {
    payment_status?: "unpaid" | "paid" | "refunded" | "waived";
    amount_paid_cents?: number;
    price_override_cents?: number | null;
    paid_at?: string | null;
    owner_private_note?: string;
  },
): Promise<{ ok: true }> {
  const r = await apiFetch(`${BASE}/owner/analytics/appointments/${encodeURIComponent(apptId)}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  await ensureOk(r);
  return r.json();
}

// Client: filtered summary for account header cards
export async function clientGetAppointmentsSummary(args: {
  start: string; // YYYY-MM-DD
  end: string;   // YYYY-MM-DD
  status?: Array<"booked" | "completed" | "canceled">;
  payment_status?: Array<PaymentStatusFull | "paid" | "unpaid" | "refunded" | "waived" | "partial" | "bundle">;
}): Promise<FinancialSummary> {
  const qs = buildQuery({
    start: args.start,
    end: args.end,
    status: args.status,
    payment_status: args.payment_status,
  });
  const r = await apiFetch(`${BASE}/client/appointments/summary?${qs}`, { cache: "no-store" });
  await ensureOk(r);
  return r.json() as Promise<FinancialSummary>;
}



// --- Public pricing/options ---
export async function listPublicServiceOptions(): Promise<ServiceOption[]> {
  const r = await apiFetch(`${BASE}/scheduling/public/service-options`, { cache: "no-store" });
  return jsonOrText(r);
}

// Returns: { options: [{ duration_minutes, price_cents }] }
export async function getPublicPricing() {
  const r = await apiFetch(`${BASE}/scheduling/public/pricing`, { cache: "no-store" });
  return jsonOrText(r);
}


export async function clientListAppointmentsFiltered(opts?: {
  status?: Array<"booked" | "completed" | "canceled">;
  payment_status?: Array<"unpaid" | "paid" | "refunded" | "waived" | "partial" | "unknown">;
  date_from?: string; // YYYY-MM-DD
  date_to?: string;   // YYYY-MM-DD
}) {
  const params = new URLSearchParams();
  for (const s of opts?.status ?? []) params.append("status", s);
  for (const p of opts?.payment_status ?? []) params.append("payment_status", p);
  if (opts?.date_from) params.set("date_from", opts.date_from);
  if (opts?.date_to) params.set("date_to", opts.date_to);
  const qs = params.toString();
  const r = await apiFetch(`${BASE}/client/appointments${qs ? `?${qs}` : ""}`, { cache: "no-store" });
  await ensureOk(r);
  const data = (await r.json()) as {
    rows: Array<{
      id: string;
      start_utc: string;
      end_utc: string;
      status: "booked" | "completed" | "canceled" | null;
      payment_status: "unpaid" | "paid" | "refunded" | "waived" | null;
      amount_paid_cents?: number | null;
      duration_minutes?: number | null;
    }>;
  };
  return data.rows;
}


// --- Search owner clients (name/email icontains) ---
export type OwnerClientLite = { id: string; account_id: number; name?: string | null; email: string };

export type OwnerClientAccountSummary = {
  account_id: number;
  client_user_id: string;
  client_email?: string | null;
  client_name?: string | null;
  name?: string | null;          // account display name (unused in UI)
  people_count: number;
};

export async function searchOwnerClients(q: string): Promise<OwnerClientLite[]> {
  const query = q?.trim();
  if (!query) return [];
  const r = await apiFetch(
    `${BASE}/scheduling/owner/clients?search=${encodeURIComponent(query)}`,
    { cache: "no-store" }
  );
  await ensureOk(r);
  const rows = (await r.json()) as OwnerClientAccountSummary[];

  // 🔁 Map analytics response → picker shape
  return rows.map((row) => ({
    id: row.client_user_id,                               // unique id (required by picker)
    account_id: row.account_id,                           // include to fetch details directly
    name: row.client_name ?? row.name ?? null,            // prefer user name, fallback to account name
    email: row.client_email ?? "",                        // normalize to string
  }));
}

export async function ownerListClientAccounts(search?: string): Promise<OwnerClientAccountSummary[]> {
  const qs = search ? `?search=${encodeURIComponent(search)}` : "";
  const r = await apiFetch(`${BASE}/scheduling/owner/clients${qs}`, { cache: "no-store" }); // <-- FIXED PATH
  await ensureOk(r);
  return r.json();
}


// src/lib/api.ts  (append near other owner APIs)

// ---- Owner: service options (pricing) ----
export async function ownerListServiceOptions() {
  const r = await apiFetch(`${BASE}/scheduling/owner/service-options`, { cache: "no-store" });
  await ensureOk(r);
  return r.json() as Promise<Array<{
    id: number;
    duration_minutes: number;
    price_cents: number;
    currency: string;
    is_active: boolean;
  }>>;
}

export async function ownerReplaceServiceOptions(options: Array<{
  duration_minutes: number;
  price_cents: number;
  currency?: string;
  is_active?: boolean;
}>) {
  const payload = options.map(o => ({
    duration_minutes: o.duration_minutes,
    price_cents: o.price_cents,
    currency: o.currency ?? "USD",
    is_active: o.is_active ?? true,
  }));
  const r = await apiFetch(`${BASE}/scheduling/owner/service-options`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  await ensureOk(r);
  return r.json();
}

// ---- Group lessons (owner) ----
export async function adminCreateGroupAppointment(input: {
  start_local: string;         // "YYYY-MM-DDTHH:mm:ss"
  duration_minutes: number;
  person_ids: number[];
  status?: "booked" | "completed" | "canceled";
  allow_override?: boolean;
  confirm_if_conflicts?: boolean;
  message?: string;
}) {
  const r = await apiFetch(`${BASE}/scheduling/appointments/admin-create-group`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
  await ensureOk(r);
  return r.json() as Promise<{ ok: boolean; group_id: string; count: number; appointment_ids: string[] }>;
}

export async function adminUpdateGroupTime(groupId: string, input: { start_local: string; duration_minutes: number; confirm_if_conflicts?: boolean; }) {
  const r = await apiFetch(`${BASE}/scheduling/appointments/group/${encodeURIComponent(groupId)}/time`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
  await ensureOk(r);
  return r.json();
}

export async function adminGroupAddAttendees(groupId: string, personIds: number[]) {
  const r = await apiFetch(`${BASE}/scheduling/appointments/group/${encodeURIComponent(groupId)}/attendees`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ person_ids: personIds }),
  });
  await ensureOk(r);
  return r.json() as Promise<{ ok: boolean; added: number }>;
}

export async function adminGroupRemoveAttendees(groupId: string, personIds: number[], appointmentIds?: string[]) {
  const body: any = { person_ids: personIds };
  if (appointmentIds && appointmentIds.length) body.appointment_ids = appointmentIds;
  const r = await apiFetch(`${BASE}/scheduling/appointments/group/${encodeURIComponent(groupId)}/attendees`, {
    method: "DELETE",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  await ensureOk(r);
  return r.json() as Promise<{ ok: boolean; removed: number }>;
}

export async function adminGroupCancel(groupId: string) {
  const r = await apiFetch(`${BASE}/scheduling/appointments/group/${encodeURIComponent(groupId)}/cancel`, {
    method: "PUT",
  });
  await ensureOk(r);
  return r.json() as Promise<{ ok: boolean; canceled: number }>;
}

export async function adminCreateGroupRecurringAppointments(input: {
  start_local: string;
  duration_minutes: number;
  repeat_every_weeks?: number;
  occurrences?: number;
  until_date?: string;
  person_ids: number[];
  allow_override?: boolean;
  confirm_if_conflicts?: boolean;
  message?: string;
}) {
  const r = await apiFetch(`${BASE}/scheduling/appointments/admin-create-group/recurring`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
  await ensureOk(r);
  return r.json() as Promise<{ ok: boolean; count: number; groups: Array<{ group_id: string; count: number; start_local: string }> }>;
}

export type AdminFeeStatus = "unpaid" | "bundle" | "refunded" | "waived" | "paid";

export type AdminFeeCharge = {
  id: number;
  owner_id: string;
  client_account_id: number;
  client_user_id?: string | null;
  amount_cents: number;
  status: AdminFeeStatus;
  paid_cash_cents: number;
  bundle_applied_cents: number;
  note?: string | null;
  created_at: string;
  updated_at: string;
  client_label?: string | null;
};

export async function ownerGetAdminFeeSettings(): Promise<{ admin_fee_cents: number }> {
  const r = await apiFetch(`${BASE}/scheduling/owner/admin-fee`, { cache: "no-store" });
  await ensureOk(r);
  return r.json();
}

export async function ownerUpdateAdminFeeSettings(admin_fee_cents: number): Promise<{ admin_fee_cents: number }> {
  const r = await apiFetch(`${BASE}/scheduling/owner/admin-fee`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ admin_fee_cents }),
  });
  await ensureOk(r);
  return r.json();
}

export async function ownerCreateAdminFeeCharge(args: {
  client_account_id: number;
  amount_cents?: number;
  note?: string;
}): Promise<AdminFeeCharge> {
  const r = await apiFetch(`${BASE}/scheduling/owner/admin-fee/charges`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(args),
  });
  await ensureOk(r);
  return r.json();
}

export async function ownerListAdminFeeCharges(params?: {
  status?: AdminFeeStatus[];
  client_account_id?: number;
  limit?: number;
}): Promise<AdminFeeCharge[]> {
  const qs = buildQuery({
    status: params?.status,
    client_account_id: params?.client_account_id,
    limit: params?.limit,
  });
  const r = await apiFetch(`${BASE}/scheduling/owner/admin-fee/charges${qs ? `?${qs}` : ""}`, { cache: "no-store" });
  await ensureOk(r);
  return r.json();
}

export async function ownerUpdateAdminFeeCharge(
  chargeId: number,
  payload: {
    status?: AdminFeeStatus;
    paid_cash_cents?: number;
    note?: string;
    apply_wallet?: boolean;
  },
): Promise<AdminFeeCharge> {
  const r = await apiFetch(`${BASE}/scheduling/owner/admin-fee/charges/${chargeId}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  await ensureOk(r);
  return r.json();
}

export async function ownerDeleteAdminFeeCharge(chargeId: number): Promise<{ ok: true }> {
  const r = await apiFetch(`${BASE}/scheduling/owner/admin-fee/charges/${chargeId}`, {
    method: "DELETE",
  });
  await ensureOk(r);
  return r.json();
}


export async function getOwnerFinancialAppointments(args: {
  start: string; // YYYY-MM-DD
  end: string;   // YYYY-MM-DD
  status?: Array<"booked" | "completed" | "canceled">;
  payment_status?: Array<"paid" | "partial" | "unpaid" | "bundle" | "unknown">;
  client_account_id?: number;
}) {
  const u = new URL(
    `${BASE}/owner/analytics/appointments`,
    typeof window === 'undefined' ? 'http://localhost:3000' : window.location.origin,
  );
  u.searchParams.set("start", args.start);
  u.searchParams.set("end", args.end);
  for (const s of args.status ?? []) u.searchParams.append("status", s);
  for (const p of args.payment_status ?? []) u.searchParams.append("payment_status", p);
  if (args.client_account_id) u.searchParams.set("client_account_id", String(args.client_account_id));
  const r = await apiFetch(u.pathname + u.search, { cache: "no-store" });
  await ensureOk(r);
  return r.json() as Promise<{
    rows: Array<{
      id: string;
      start_utc: string;
      end_utc: string;
      client_account_id?: number | null;
      client_label: string;
      status: "booked" | "completed" | "canceled";
      duration_minutes: number;
      price_cents?: number | null;
      paid_cash_cents: number;
      bundle_applied_cents: number;
      owed_cents: number;
      payment_status: "paid" | "partial" | "unpaid" | "bundle" | "unknown";
    }>;
  }>;
}


// --- Owner: resolve client account by query ---
type ResolveCandidatesBody = {
  detail?: {
    candidates?: unknown;
    message?: string;
  };
};

type ErrorWithCandidates = Error & { candidates?: unknown };

export async function ownerResolveClientAccount(query: string) {
  const u = new URL("/api/back/api/scheduling/owner/clients/resolve", window.location.origin);
  u.searchParams.set("query", query);
  const r = await fetch(u.toString(), { credentials: "include" });
  const text = await r.text();
  let body: unknown;
  try { body = text ? JSON.parse(text) : null; } catch { body = text; }

  if (r.ok) return body as {
    account_id: number;
    client_user_id: string;
    client_email: string | null;
    client_name: string | null;
    name: string | null;
    people_count: number;
  };

  // 409 with candidates → throw a specific error we can handle
  if (
    r.status === 409 &&
    typeof body === "object" &&
    body !== null &&
    "detail" in body &&
    typeof (body as ResolveCandidatesBody).detail === "object" &&
    (body as ResolveCandidatesBody).detail?.candidates
  ) {
    const msg = (body as ResolveCandidatesBody).detail?.message || "Multiple matches";
    const err: ErrorWithCandidates = new Error(msg);
    err.candidates = (body as ResolveCandidatesBody).detail?.candidates;
    throw err;
  }
  throw new ApiError(r.status, body);
}

// Owner: get client detail including people
export async function ownerGetClientDetail(accountId: number) {
  // Correct path under scheduling router
  const r = await apiFetch(`${BASE}/scheduling/owner/clients/${accountId}`, { cache: "no-store" });
  await ensureOk(r);
  return r.json() as Promise<{
    account_id: number;
    client_user_id: string;
    client_email?: string | null;
    client_name?: string | null;
    name?: string | null;
    emails?: Array<{ id: number; email: string; is_primary: boolean }>;
    people: { id: number; full_name: string; email?: string | null }[];
  }>;
}


// src/lib/api.ts

// src/lib/api.ts

export type Recipient = { email: string; name: string | null };

export type OutboxDraft = {
  id: string;
  owner_user_id: string;
  to: string | null;
  to_name: string | null;
  recipients: Recipient[];
  subject: string;
  text: string;
  status: "pending" | "approved" | "rejected" | "sent";
  created_at?: string;
  approved_at?: string | null;
  sent_at?: string | null;
  rejected_reason?: string | null;
};

async function _json<T>(res: Response): Promise<T> {
  if (!res.ok) throw new Error(await res.text().catch(() => res.statusText));
  return (await res.json()) as T;
}

export async function listOutboxDrafts(params?: {
  status?: "pending" | "approved" | "rejected" | "sent";
  q?: string;
  limit?: number;
  offset?: number;
}): Promise<OutboxDraft[]> {
  const u = new URL(`/api/back/api/outbox`, window.location.origin);
  if (params?.status) u.searchParams.set("status", params.status);
  if (params?.q) u.searchParams.set("q", params.q);
  if (typeof params?.limit === "number") u.searchParams.set("limit", String(params.limit));
  if (typeof params?.offset === "number") u.searchParams.set("offset", String(params.offset));
  const r = await fetch(u.toString(), { cache: "no-store" });
  return _json<OutboxDraft[]>(r);
}

export async function getOutboxDraft(draftId: string): Promise<OutboxDraft> {
  const r = await fetch(`/api/back/api/outbox/${encodeURIComponent(draftId)}`, { cache: "no-store" });
  return _json<OutboxDraft>(r);
}

export async function updateOutboxDraft(
  draftId: string,
  body: {
    to?: string;
    to_name?: string | null;
    recipients?: { email: string; name?: string | null }[];
    replace_recipients?: boolean;
    subject?: string;
    text?: string;
  }
): Promise<OutboxDraft> {
  const r = await fetch(`/api/back/api/outbox/${encodeURIComponent(draftId)}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return _json<OutboxDraft>(r);
}

export async function sendOutboxDraft(
  draftId: string,
  body: {
    approve?: boolean;
    to?: string;
    to_name?: string | null;
    recipients?: { email: string; name?: string | null }[];
    replace_recipients?: boolean;
    subject?: string;
    text?: string;
  } = { approve: true }
): Promise<OutboxDraft> {
  const r = await fetch(`/api/back/api/outbox/${encodeURIComponent(draftId)}/send`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ approve: true, ...body }),
  });
  return _json<OutboxDraft>(r);
}

export async function rejectOutboxDraft(draftId: string, reason: string): Promise<OutboxDraft> {
  const r = await fetch(`/api/back/api/outbox/${encodeURIComponent(draftId)}/reject`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ reason }),
  });
  return _json<OutboxDraft>(r);
}
