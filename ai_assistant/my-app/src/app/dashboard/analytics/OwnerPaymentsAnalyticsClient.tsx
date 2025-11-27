// src/app/dashboard/analytics/OwnerPaymentsAnalyticsClient.tsx
"use client";

import React, { forwardRef, useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import {
  ownerListClientAccounts,
  type OwnerClientAccountSummary,
  ownerListFinancialAppointments,
  ownerGetFinancialSummary,
  ownerUpdateAppointment,
  updateAppointment,
  ownerUpdateClient,
  ownerListBundles,
  ownerTopUpBundle,
  ownerAdjustWallet,
  ownerDeleteAppointment,
  ownerListBundleLedger,
  type PrepaidBundle,
  type AppointmentFinancialRow,
  type FinancialSummary,
  type PaymentStatusFull,
  type PaymentStatus,
  ownerListAdminFeeCharges,
  ownerUpdateAdminFeeCharge,
  ownerDeleteAdminFeeCharge,
  type AdminFeeCharge,
  type AdminFeeStatus,
  type AttendanceStatus,
  ownerGetAppointmentDetails,
  ownerGetClientDetail,
  ownerGetGroupDetails,
  adminGroupRemoveAttendees,
  adminGroupCancel,
  cancelAppointment,
} from "@/lib/api";
import { ApiError } from "@/lib/api";
import moment from "moment-timezone";
import { useSession } from "next-auth/react";

// Small wallet quick-view for owners on analytics page
function QuickWalletButton({ clientId, tz, onChanged }: { clientId: string; tz?: string; onChanged?: () => void }) {
  const [open, setOpen] = React.useState(false);
  const [loading, setLoading] = React.useState(false);
  const [wallet, setWallet] = React.useState<PrepaidBundle | null>(null);
  const [ledger, setLedger] = React.useState<Array<{ event: string; amount_cents: number; created_at: string }> | null>(null);
  const [topup, setTopup] = React.useState<string>("");
  const [wFrom, setWFrom] = React.useState<string>("");
  const [wTo, setWTo] = React.useState<string>("");

  async function load(filters?: { date_from?: string; date_to?: string }) {
    setLoading(true);
    try {
      const list = await ownerListBundles(clientId);
      const wallets = (list || []).filter((b) => (b.total_credits || 0) === 0);
      setWallet(wallets.length ? wallets[0] : null);
      if (wallets.length) {
        const rows = await ownerListBundleLedger(clientId, wallets[0].id, 10, filters);
        setLedger(rows);
      } else {
        setLedger([]);
      }
    } catch {
      setWallet(null);
      setLedger([]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="relative">
      <button type="button" className="px-3 py-1 border rounded text-sm" onClick={async () => {
        if (!open) { await load(); setOpen(true); } else { setOpen(false); }
      }}>Wallet</button>
      {open && (
        <>
          <div className="fixed inset-0 bg-black/30 z-40" onClick={() => setOpen(false)} />
          <aside className="fixed top-0 right-0 h-full w-[min(100vw,420px)] bg-white z-50 shadow-2xl flex flex-col">
            <div className="px-4 py-3 border-b flex items-center justify-between">
              <div className="text-sm font-medium">Client Wallet</div>
              <button type="button" className="text-xs" onClick={() => setOpen(false)}>✕</button>
            </div>
            <div className="p-4 space-y-3 overflow-auto">
              {loading ? (
                <div className="text-sm text-zinc-600 py-4">Loading…</div>
              ) : !wallet ? (
                <div className="text-sm text-zinc-600">No wallet found.</div>
              ) : (
                <>
                  <div className="text-sm">Balance: <b>{money(wallet.remaining_balance_cents ?? 0)}</b></div>
                  <div className="flex items-end gap-2">
                    <label className="text-sm">
                      <div className="text-xs text-zinc-600">From</div>
                      <input type="date" className="border rounded px-2 py-1" value={wFrom} onChange={(e)=>setWFrom(e.target.value)} />
                    </label>
                    <label className="text-sm">
                      <div className="text-xs text-zinc-600">To</div>
                      <input type="date" className="border rounded px-2 py-1" value={wTo} onChange={(e)=>setWTo(e.target.value)} />
                    </label>
                    <button type="button" className="px-2 py-1 border rounded text-sm" onClick={async ()=>{
                      await load({ date_from: wFrom || undefined, date_to: wTo || undefined });
                    }}>Search</button>
                    <button type="button" className="px-2 py-1 border rounded text-sm" onClick={async ()=>{
                      setWFrom(""); setWTo("");
                      await load();
                    }}>Reset</button>
                  </div>
                  <div className="flex flex-col gap-2">
                    <div className="flex items-center gap-2">
                      <input
                        placeholder="Amount $"
                        className="border rounded px-2 py-1 text-sm w-28"
                        type="number"
                        step="0.01"
                        min={0}
                        value={topup}
                        onChange={(e) => setTopup(e.target.value)}
                      />
                    </div>
                    <div className="flex items-center gap-2">
                      <button
                        type="button"
                        className="px-2 py-1 border rounded text-sm"
                        onClick={async () => {
                          const dollars = parseFloat(topup || "0");
                          const cents = Math.round((Number.isFinite(dollars) ? dollars : 0) * 100);
                          if (!wallet || cents <= 0) return;
                          try {
                            await ownerTopUpBundle(clientId, wallet.id, cents);
                            setTopup("");
                            await load({ date_from: wFrom || undefined, date_to: wTo || undefined });
                            onChanged?.();
                          } catch (e: unknown) {
                            alert(e instanceof Error ? e.message : "Failed to add funds");
                          }
                        }}
                      >Add</button>
                      <button
                        type="button"
                        className="px-2 py-1 border rounded text-sm"
                        onClick={async () => {
                          const dollars = parseFloat(topup || "0");
                          const cents = Math.round((Number.isFinite(dollars) ? dollars : 0) * 100);
                          if (!wallet || cents <= 0) return;
                          try {
                            await ownerAdjustWallet(clientId, wallet.id, -cents);
                            setTopup("");
                            await load({ date_from: wFrom || undefined, date_to: wTo || undefined });
                            onChanged?.();
                          } catch (e: unknown) {
                            alert(e instanceof Error ? e.message : "Failed to remove funds");
                          }
                        }}
                      >Remove</button>
                    </div>
                  </div>
                  <div className="text-xs text-zinc-500">Recent activity</div>
                  <div className="max-h-[50vh] overflow-auto border rounded">
                    {(ledger || []).length ? (
                      <ul className="text-xs divide-y">
                        {ledger!.map((l, i) => (
                          <li key={i} className="flex items-center justify-between px-2 py-1">
                            <span>{new Date(l.created_at).toLocaleString([], { dateStyle: 'short', timeStyle: 'short', ...(tz ? { timeZone: tz } : {}) })}</span>
                            <span>{l.event}</span>
                            <span>{l.amount_cents >= 0 ? `+${money(l.amount_cents)}` : `-${money(-l.amount_cents)}`}</span>
                          </li>
                        ))}
                      </ul>
                    ) : (
                      <div className="text-xs text-zinc-600 p-2">No activity.</div>
                    )}
                  </div>
                </>
              )}
            </div>
          </aside>
        </>
      )}
    </div>
  );
}

type Props = {
  initialStart?: string;
  initialEnd?: string;
};



function money(cents?: number | null) {
  const n = Number(cents ?? 0);
  return `$${(n / 100).toFixed(2)}`;
}

const ALL_STATUSES: Array<"booked" | "completed" | "canceled"> = ["booked", "completed", "canceled"];
const ALL_PAYMENT: PaymentStatusFull[] = ["paid", "bundle", "refunded", "waived", "unpaid"];
const ADMIN_FEE_STATUSES: AdminFeeStatus[] = ["unpaid", "bundle", "paid", "waived", "refunded"];


// Prevent background scroll while the drawer is open
function useBodyScrollLock(locked: boolean) {
  useEffect(() => {
    const prev = document.body.style.overflow;
    if (locked) document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = prev;
    };
  }, [locked]);
}

function Drawer({
  open,
  onClose,
  width = 420,
  title = "Edit",
  children,
}: {
  open: boolean;
  onClose: () => void;
  width?: number;
  title?: string;
  children: React.ReactNode;
}) {
  useBodyScrollLock(open);

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    if (open) window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;

  // Portal to avoid z-index/overflow issues
  return createPortal(
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/30 z-40"
        onClick={onClose}
        aria-hidden="true"
      />

      {/* Panel */}
      <aside
        role="dialog"
        aria-modal="true"
        className="fixed top-0 right-0 h-full bg-white z-50 shadow-2xl flex flex-col"
        style={{ width: "min(100vw, " + width + "px)" }}
      >
        <div className="px-4 py-3 border-b flex items-center justify-between">
          <h2 className="text-sm font-medium">{title}</h2>
          <button
            aria-label="Close"
            className="text-zinc-600 hover:text-black text-sm"
            onClick={onClose}
          >
            ✕
          </button>
        </div>
        <div className="p-4 overflow-auto">{children}</div>
      </aside>
    </>,
    document.body
  );
}

// Small attendee details popover row component
function AttendeeRow({ attendee, groupId, reload, refreshGroup }: {
  attendee: { appointment_id: string; person_id?: number | null; name: string; status: string; payment_status: string; price_cents?: number | null; owed_cents: number };
  groupId: string;
  reload: () => Promise<void>;
  refreshGroup: () => Promise<void>;
}) {
  const [open, setOpen] = React.useState(false);
  return (
    <div className="rounded border bg-white">
      <div className="flex items-center justify-between px-2 py-1 text-xs">
        <div className="flex items-center gap-2">
          <span className="font-medium">{attendee.name}</span>
          <span className="text-[10px] text-gray-600">{attendee.status}</span>
          <span className="text-[10px]">{attendee.payment_status}</span>
        </div>
        <button type="button" className="underline" onClick={() => setOpen(v => !v)}>{open ? 'Hide' : 'Details'}</button>
      </div>
      {open && (
        <div className="px-2 pb-2 text-[11px] space-y-2 border-t">
          <div className="flex items-center justify-between">
            <div>Price</div>
            <div>{typeof attendee.price_cents === 'number' ? `$${(attendee.price_cents/100).toFixed(2)}` : '-'}</div>
          </div>
          <div className="flex items-center justify-between">
            <div>Cash paid</div>
            <div>{typeof (attendee as any).paid_cash_cents === 'number' ? `$${(Number((attendee as any).paid_cash_cents)/100).toFixed(2)}` : '$0.00'}</div>
          </div>
          <div className="flex items-center justify-between">
            <div>Wallet applied</div>
            <div>{typeof (attendee as any).bundle_applied_cents === 'number' ? `$${(Number((attendee as any).bundle_applied_cents)/100).toFixed(2)}` : '$0.00'}</div>
          </div>
          <div className="flex items-center justify-between">
            <div>Owed</div>
            <div className={attendee.owed_cents > 0 ? 'text-red-700 font-medium' : ''}>{`$${(attendee.owed_cents/100).toFixed(2)}`}</div>
          </div>
          <div className="flex items-center gap-2 pt-1">
            <button
              type="button"
              className="inline-flex items-center gap-1 rounded border border-amber-500 px-2 py-1 text-[10px] font-medium text-amber-700 hover:bg-amber-50"
              onClick={async () => {
                try {
                  await ownerUpdateAppointment(attendee.appointment_id, { attendance_status: 'no_show' as any });
                  await refreshGroup();
                  await reload();
                } catch (e:any) {
                  alert(e?.message || 'Failed to mark no show');
                }
              }}
            >Mark no‑show</button>
            <button
              type="button"
              className="inline-flex items-center gap-1 rounded border border-emerald-500 px-2 py-1 text-[10px] font-medium text-emerald-700 hover:bg-emerald-50"
              onClick={async () => {
                try {
                  await ownerUpdateAppointment(attendee.appointment_id, { attendance_status: 'attended' as any });
                  await refreshGroup();
                  await reload();
                } catch (e:any) {
                  alert(e?.message || 'Failed to mark attended');
                }
              }}
            >Mark attended</button>
            <button
              type="button"
              className="inline-flex items-center gap-1 rounded border border-red-500 px-2 py-1 text-[10px] font-medium text-red-600 hover:bg-red-50"
              onClick={async () => {
                try {
                  const pid = Number((attendee as any).person_id) || 0;
                  const apptId = (attendee as any).appointment_id as string | undefined;
                  if (pid > 0) {
                    await adminGroupRemoveAttendees(groupId, [pid]);
                  } else if (apptId) {
                    await adminGroupRemoveAttendees(groupId, [], [apptId]);
                  } else {
                    throw new Error('Cannot determine attendee to remove');
                  }
                  await refreshGroup();
                  await reload();
                } catch (e:any) {
                  alert(e?.message || 'Failed to remove attendee');
                }
              }}
            >Remove</button>
          </div>
        </div>
      )}
    </div>
  )
}



function StatusChip({ kind }: { kind: PaymentStatusFull }) {
  const label =
    kind === "paid" ? "Paid" :
    kind === "bundle" ? "Bundle" :
    kind === "refunded" ? "Refunded" :
    kind === "waived" ? "Waived" :
    kind === "unpaid" ? "Unpaid" :
    kind === "partial" ? "Partial" :
    "Unknown";

  const cls =
    kind === "paid" || kind === "bundle" ? "bg-green-100 text-green-800 border-green-200" :
    kind === "refunded" || kind === "waived" ? "bg-blue-100 text-blue-800 border-blue-200" :
    kind === "unpaid" ? "bg-red-100 text-red-800 border-red-200" :
    kind === "partial" ? "bg-yellow-100 text-yellow-800 border-yellow-200" :
    "bg-gray-100 text-gray-700 border-gray-200";

  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium border ${cls}`}>
      {label}
    </span>
  );
}

// --- small helpers for UX ---
// --- small helpers for UX ---
function useClickOutside<T extends HTMLElement>(onOutside: () => void) {
  const ref = useRef<T | null>(null);
  useEffect(() => {
    function onDocClick(e: MouseEvent) {
      if (!ref.current) return;
      if (e.target instanceof Node && !ref.current.contains(e.target)) onOutside();
    }
    document.addEventListener("mousedown", onDocClick);
    return () => document.removeEventListener("mousedown", onDocClick);
  }, [onOutside]);
  return ref;
}

// eslint-disable-next-line @typescript-eslint/no-unused-vars
function Popover({
  anchorRef,
  children,
  onClose,
  className = "",
}: {
  anchorRef: React.RefObject<HTMLElement>;
  children: React.ReactNode;
  onClose: () => void;
  className?: string;
}) {
  const [style, setStyle] = useState<React.CSSProperties>({});
  const popRef = useClickOutside<HTMLDivElement>(onClose);

  useEffect(() => {
    function position() {
      const anchor = anchorRef.current;
      if (!anchor) return;
      const rect = anchor.getBoundingClientRect();
      setStyle({
        position: "fixed",
        top: rect.bottom + 6,
        left: Math.max(8, rect.right - 360), // popover ~360px wide
        zIndex: 50,
      });
    }
    position();
    window.addEventListener("resize", position);
    window.addEventListener("scroll", position, true);
    return () => {
      window.removeEventListener("resize", position);
      window.removeEventListener("scroll", position, true);
    };
  }, [anchorRef]);

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div
      ref={popRef}
      style={style}
      className={`w-[360px] rounded-xl border bg-white shadow-xl p-3 ${className}`}
      role="dialog"
      aria-modal="true"
    >
      {children}
    </div>
  );
}


// --- UI helpers (drop-in) ---
function FieldLabel({ children, htmlFor }: { children: React.ReactNode; htmlFor?: string }) {
  return (
    <label htmlFor={htmlFor} className="block text-[11px] leading-4 text-zinc-500">
      {children}
    </label>
  );
}

function MoneyInput({
  id,
  value,
  onChange,
  placeholder,
}: {
  id: string;
  value: string | undefined;
  onChange: (v: string) => void;
  placeholder?: string;
}) {
  return (
    <div className="flex items-center gap-1 border rounded px-2 py-1">
      <span className="text-xs">$</span>
      <input
        id={id}
        className="w-24 bg-transparent text-xs outline-none"
        type="number"
        inputMode="decimal"
        step="0.01"
        min={0}
        placeholder={placeholder ?? "0.00"}
        value={value ?? ""}
        onChange={(e) => onChange(e.target.value)}
      />
    </div>
  );
}

const TinyButton = forwardRef<
  HTMLButtonElement,
  React.ButtonHTMLAttributes<HTMLButtonElement> & { kind?: "primary" | "neutral" | "ghost" }
>(function TinyButtonBase({ children, kind = "neutral", ...props }, ref) {
  const base = "px-2 py-1 rounded text-xs";
  const cls =
    kind === "primary"
      ? "bg-blue-600 text-white hover:bg-blue-700"
      : kind === "ghost"
      ? "hover:bg-gray-100 dark:hover:bg-zinc-800"
      : "border hover:bg-gray-50 dark:hover:bg-zinc-800";
  return (
    <button ref={ref} className={`${base} ${cls}`} {...props}>
      {children}
    </button>
  );
});




export default function OwnerPaymentsAnalyticsClient({ initialStart, initialEnd }: Props) {
  const { data: session } = useSession();
  const tz = (session?.user as { timezone?: string } | null | undefined)?.timezone;
  // --- date filters
  const ymdLocal = (d: Date) => {
    const pad = (n: number) => String(n).padStart(2, "0");
    return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
  };
  const now = new Date();
  const firstOfMonth = new Date(now.getFullYear(), now.getMonth(), 1);
  const lastOfMonth = new Date(now.getFullYear(), now.getMonth() + 1, 0);
  const [start, setStart] = useState<string>(initialStart ?? ymdLocal(firstOfMonth));
  const [end, setEnd] = useState<string>(initialEnd ?? ymdLocal(lastOfMonth));
  const [dateError, setDateError] = useState<string>("");
  // --- other filters
  const [status, setStatus] = useState<Array<"booked" | "completed" | "canceled">>(["booked", "completed"]);
  const [payment, setPayment] = useState<PaymentStatusFull[]>([]);
  const [accounts, setAccounts] = useState<OwnerClientAccountSummary[]>([]);
  const [search, setSearch] = useState("");
  const [clientAccountId, setClientAccountId] = useState<number | "">("");

  // --- data
  const [rows, setRows] = useState<AppointmentFinancialRow[]>([]);
  const [summary, setSummary] = useState<FinancialSummary | null>(null);
  const [loading, setLoading] = useState(false);
  const [accountsLoading, setAccountsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [openSuggest, setOpenSuggest] = useState(false);
  const [highlight, setHighlight] = useState(0);
  const [openRowId, setOpenRowId] = useState<string | null>(null);
  const [openGroupId, setOpenGroupId] = useState<string | null>(null);
  const [groupLoading, setGroupLoading] = useState(false);
  const [groupAttendees, setGroupAttendees] = useState<Array<{ appointment_id: string; person_id?: number | null; name: string; status: string; payment_status: string; price_cents?: number | null; owed_cents: number }>>([]);
  const _anchorRefs = useRef<Record<string, HTMLButtonElement | null>>({});
  const [peopleByAccount, setPeopleByAccount] = useState<Record<number, Array<{ id: number; full_name: string; email?: string | null }>>>({});
  const [walletByClient, setWalletByClient] = useState<Record<string, number>>({}); // balance_cents
  const [adminFees, setAdminFees] = useState<AdminFeeCharge[]>([]);
  const [adminFeesLoading, setAdminFeesLoading] = useState(false);
  const [adminFeesError, setAdminFeesError] = useState<string | null>(null);
  const [adminFeeNotice, setAdminFeeNotice] = useState<string | null>(null);
  const [feeUpdating, setFeeUpdating] = useState<Record<number, boolean>>({});
  const [attendanceUpdating, setAttendanceUpdating] = useState<Record<string, boolean>>({});




  // --- edit state (per-row)
  type Edit = {
    // Appointment status (booked/completed/canceled)
    status?: "booked" | "completed" | "canceled";
    payment_status?: "unpaid" | "paid" | "refunded" | "waived";
    amount_paid_dollars?: string;         // input in dollars
    price_override_dollars?: string | ""; // input in dollars ("" = keep)

    // New editable fields
    client_name?: string;
    client_email?: string;
    date?: string;             // YYYY-MM-DD (owner local)
    time?: string;             // HH:MM (owner local)
    duration_minutes?: string; // as string for input
    owner_private_note?: string;
    attendance_status?: "unknown" | "attended" | "late" | "no_show";
    lesson_person_name?: string;
    cancel_reason?: string;
    error?: string;            // inline error to show in drawer
    errors?: Partial<{
      client_email: string;
      date: string;
      time: string;
      duration_minutes: string;
    }>;

  };
  const [editing, setEditing] = useState<Record<string, Edit>>({});

  const noShowCount = useMemo(
    () => rows.filter((r) => r.attendance_status === "no_show").length,
    [rows]
  );

  function setEdit(id: string, patch: Partial<Edit>) {
    setEditing((cur) => ({ ...cur, [id]: { ...cur[id], ...patch } }));
  }

  function labelForAccount(a: OwnerClientAccountSummary) {
    return a.name || a.client_name || a.client_email || `Account #${a.account_id}`;
    }

  function clientUserIdForRow(r: AppointmentFinancialRow): string | null {
    const acct = accounts.find(a => a.account_id === (r.client_account_id ?? -1));
    return acct?.client_user_id ?? null;
  }

  // Lightweight wallet balance preload for visible rows
  useEffect(() => {
    const ids = Array.from(new Set(
      rows.map(r => clientUserIdForRow(r)).filter(Boolean) as string[]
    ));
    (async () => {
      for (const cid of ids) {
        if (walletByClient[cid] !== undefined) continue;
        try {
          const list = await ownerListBundles(cid);
          const wallets = (list || []).filter(b => (b.total_credits || 0) === 0);
          const bal = wallets.length ? (wallets[0].remaining_balance_cents ?? 0) : 0;
          setWalletByClient(prev => ({ ...prev, [cid]: bal }));
        } catch {
          setWalletByClient(prev => ({ ...prev, [cid]: 0 }));
        }
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rows]);

  // Hydrate drawer fields whenever a row is opened
  useEffect(() => {
    (async () => {
      if (!openRowId) return;
      try {
        const d = await ownerGetAppointmentDetails(openRowId);
        setEdit(openRowId, {
          owner_private_note: d.owner_private_note ?? "",
          attendance_status: (d.attendance_status as AttendanceStatus) ?? "attended",
          ...(typeof d.price_override_cents === 'number' ? { price_override_dollars: (d.price_override_cents / 100).toFixed(2) } : {}),
        });
      } catch {}
    })();
  }, [openRowId]);

  function onStartChange(v: string) {
    setStart(v);
    setDateError("");
    if (end && v && end < v) setDateError("End date must be on or after start date.");
  }
  function onEndChange(v: string) {
    setEnd(v);
    setDateError("");
    if (!start && v) {
        setDateError("Select a start date before choosing an end date.");
    } else if (start && v && v < start) {
        setDateError("End date must be on or after start date.");
    }
  }

  

  // Simple rank: exact email/name starts-with first
  function rankMatches(a: OwnerClientAccountSummary, q: string) {
    const s = q.toLowerCase();
    const fields = [
        a.client_email?.toLowerCase() || "",
        a.client_name?.toLowerCase() || "",
        a.name?.toLowerCase() || "",
    ];
    if (!s) return 999;
    if (fields.some(f => f === s)) return 0;            // exact
    if (fields.some(f => f.startsWith(s))) return 1;    // starts-with
    return 2;                                           // contains/other
    }

  const suggestions = useMemo(() => {
    const q = search.trim();
    if (!q) return [];
    const ranked = [...accounts].sort((a, b) => {
        const ra = rankMatches(a, q);
        const rb = rankMatches(b, q);
        if (ra !== rb) return ra - rb;
        // stable tiebreakers
        const la = labelForAccount(a).toLowerCase();
        const lb = labelForAccount(b).toLowerCase();
        return la.localeCompare(lb);
    });
    return ranked.slice(0, 8); // top 8
    }, [accounts, search]);

  function startEdit(r: AppointmentFinancialRow) {
    // derive local date/time for inputs
    const dtTz = tz ? moment.utc(r.start_utc).tz(tz) : moment(r.start_utc);
    const localDate = dtTz.format("YYYY-MM-DD");
    const localTime = dtTz.format("HH:mm");

    // Attempt to get client details from loaded accounts for convenience
    const account = accounts.find(a => a.account_id === (r.client_account_id ?? -1));
    setEditing((cur) => ({
      ...cur,
      [r.id]: {
        status: r.status,
        payment_status:
          r.payment_status === "refunded" || r.payment_status === "waived" || r.payment_status === "paid" || r.payment_status === "unpaid"
            ? (r.payment_status as Edit["payment_status"])
            : undefined,
        amount_paid_dollars: ((r.paid_cash_cents ?? 0) / 100).toFixed(2),
        price_override_dollars: typeof r.price_cents === "number" ? (r.price_cents / 100).toFixed(2) : "",
        client_name: account?.client_name ?? account?.name ?? undefined,
        client_email: account?.client_email ?? undefined,
        date: localDate,
        time: localTime,
        duration_minutes: String(r.duration_minutes ?? ""),
        owner_private_note: undefined, // not included in this list payload
        attendance_status: (r.attendance_status as Edit["attendance_status"]) ?? "attended",
        late_minutes: undefined,
        
        error: undefined,
      },
    }));
  }

  function cancelEdit(id: string) {
    setEditing((cur) => {
      const next = { ...cur };
      delete next[id];
      return next;
    });
  }

    // 1) Replace dollarsToCents with this more robust parser
    function dollarsToCents(v?: string | null) {
    // Accepts "$1,234.50", "1 234.50", "1234,50" (if you paste with comma),
    // trims spaces, removes currency symbols, and normalizes commas.
    if (v == null) return undefined;
    const raw = String(v).trim();
    if (raw === "") return undefined;

    // Remove everything except digits, dot, comma, and minus
    let cleaned = raw.replace(/[^\d.,-]/g, "");

    // If both comma and dot exist, assume comma = thousands, dot = decimal → drop commas
    if (cleaned.includes(",") && cleaned.includes(".")) {
        cleaned = cleaned.replace(/,/g, "");
    } else if (cleaned.includes(",") && !cleaned.includes(".")) {
        // If only comma present, treat comma as decimal separator
        cleaned = cleaned.replace(",", ".");
    }

    const n = Number(cleaned);
    if (!Number.isFinite(n)) return undefined;
    return Math.round(n * 100);
    }


    // 2) Replace saveEdit with this version (only the function body)
  async function saveEdit(id: string): Promise<boolean> {
    const e = editing[id];
    if (!e) return false;

    // Basic inline validation (field-level)
    const errs: Record<string, string> = {};
    if ((e.date && !e.time)) errs.time = "Start time required when changing date.";
    if ((!e.date && e.time)) errs.date = "Date required when changing time.";
    if (e.duration_minutes && Number(e.duration_minutes) <= 0) errs.duration_minutes = "Duration must be positive.";
    if (e.client_email && e.client_email.trim() && !/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(e.client_email.trim())) {
      errs.client_email = "Invalid email format.";
    }
    if (Object.keys(errs).length) {
      setEdit(id, { errors: errs, error: "Please fix the highlighted fields." });
      return false;
    } else {
      // clear previous errors
      setEdit(id, { errors: {}, error: undefined });
    }

    const ownerPatch: Partial<{
      payment_status: PaymentStatus;
      price_override_cents: number;
      owner_private_note: string;
      attendance_status: AttendanceStatus;
    }> = {};
    if (e.payment_status) {
      const allowed: PaymentStatus[] = ["paid", "unpaid", "refunded", "waived"];
      if (allowed.includes(e.payment_status as PaymentStatus)) {
        ownerPatch.payment_status = e.payment_status as PaymentStatus;
      }
    }

    // cash input removed; wallet handles payments automatically

    // price_override_dollars → price_override_cents
    // If explicitly "", omit to keep current server value
    if (e.price_override_dollars !== "" && e.price_override_dollars != null) {
        const overrideCents = dollarsToCents(e.price_override_dollars);
        if (overrideCents != null) {
        ownerPatch.price_override_cents = Math.max(0, overrideCents);
        }
    }

    // owner fields
    if (e.owner_private_note !== undefined) ownerPatch.owner_private_note = e.owner_private_note;
    if (e.attendance_status) ownerPatch.attendance_status = e.attendance_status;
    // bundle attach removed

    // Appointment-level updates (email/time/duration)
    const apptPut: Partial<{
      client_email: string;
      status: "booked" | "completed" | "canceled";
      lesson_person_name: string;
      start_local: string;
      duration_minutes: number;
    }> = {};
    if (e.client_email && e.client_email.trim()) apptPut.client_email = e.client_email.trim();
    if (e.status) apptPut.status = e.status;
    if (e.lesson_person_name && e.lesson_person_name.trim()) apptPut.lesson_person_name = e.lesson_person_name.trim();
    if (e.date && e.time) {
        // Construct naive local ISO string (backend interprets as owner-local)
        apptPut.start_local = `${e.date}T${e.time}:00`;
    }
    if (e.duration_minutes && e.duration_minutes.trim()) {
        const dm = Number(e.duration_minutes);
        if (Number.isFinite(dm) && dm > 0) apptPut.duration_minutes = Math.round(dm);
    }

    try {
        // 1) Owner patch (attendance/payment/note/bundle)
        if (Object.keys(ownerPatch).length) {
            await ownerUpdateAppointment(id, ownerPatch);
        }

        // 2) Appointment schedule/client updates
        if (Object.keys(apptPut).length) {
            // attempt without override; on conflict, prompt and retry
            try {
              await updateAppointment(id, { ...apptPut, allow_override: false });
            } catch (err: unknown) {
              if (err instanceof ApiError && err.status === 409) {
                const ok = window.confirm("This change conflicts with existing events. Proceed anyway?");
                if (ok) {
                  await updateAppointment(id, { ...apptPut, allow_override: true });
                } else {
                  setEdit(id, { error: "Conflicts detected. Adjust the time or confirm override." });
                  return false;
                }
              } else {
                throw err;
              }
            }
        }

        // 3) Optional: client name → update account (if available)
        if (e.client_name && e.client_name.trim()) {
            const r = rows.find(r => r.id === id);
            const acct = r ? accounts.find(a => a.account_id === (r.client_account_id ?? -1)) : undefined;
            if (acct?.account_id) {
                await ownerUpdateClient(acct.account_id, { name: e.client_name.trim() });
            }
        }

        // 4) Legacy bundle refresh removed

        await refresh();
        cancelEdit(id);
        return true;
    } catch (err: any) {
        window.alert(err?.message || "Failed to save");
        return false;
    }
    }


  async function quickSetAttendance(id: string, status: AttendanceStatus) {
    setAttendanceUpdating((cur) => ({ ...cur, [id]: true }));
    try {
      await ownerUpdateAppointment(id, { attendance_status: status });
      const prevRow = rows.find((r) => r.id === id);
      const prevWasNoShow = prevRow?.attendance_status === "no_show";
      const nextNoShow = Math.max(0, noShowCount - (prevWasNoShow ? 1 : 0) + (status === "no_show" ? 1 : 0));
      setRows((cur) => cur.map((r) => (r.id === id ? { ...r, attendance_status: status } : r)));
      setSummary((cur) => (cur ? { ...cur, total_no_show: nextNoShow } : cur));
      setEdit(id, { attendance_status: status });
    } catch (err: unknown) {
      window.alert(err instanceof Error ? err.message : "Failed to update attendance");
    } finally {
      setAttendanceUpdating((cur) => {
        const next = { ...cur };
        delete next[id];
        return next;
      });
    }
  }


  // --- fetch clients
  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        setAccountsLoading(true);
        const rows = await ownerListClientAccounts(search.trim() || undefined);
        if (!alive) return;
        setAccounts(rows);
      } catch (e: unknown) {
        if (!alive) return;
        setError(e instanceof Error ? e.message : "Failed to load clients");
      } finally {
        if (!alive) return;
        setAccountsLoading(false);
      }
    })();
    return () => {
      alive = false;
    };
  }, [search]);

  async function refresh() {
    if (dateError) return;   
    setLoading(true);
    setError(null);
    setAdminFeesLoading(true);
    setAdminFeesError(null);
    setAdminFeeNotice(null);
    try {
      const clientAccountFilter = clientAccountId === "" ? undefined : Number(clientAccountId);
      const [rs, sum] = await Promise.all([
        ownerListFinancialAppointments({
          start,
          end,
          status,
          payment_status: payment.length ? payment : undefined,
          client_account_id: clientAccountFilter,
        }),
        ownerGetFinancialSummary({
          start,
          end,
          status,
          payment_status: payment.length ? payment : undefined,
          client_account_id: clientAccountFilter,
        }),
      ]);
      setRows(rs);
      setSummary(sum);
      try {
        const fees = await ownerListAdminFeeCharges({
          client_account_id: clientAccountFilter,
          limit: 100,
        });
        setAdminFees(fees);
      } catch (feeErr) {
        const message = feeErr instanceof Error ? feeErr.message : String(feeErr);
        setAdminFees([]);
        setAdminFeesError(message || "Failed to load admin fee charges");
      }
    } catch (error) {
      setRows([]);
      setSummary(null);
      const message = error instanceof Error ? error.message : String(error);
      setError(message || "Failed to load analytics");
      setAdminFees([]);
      setAdminFeesError(message || "Failed to load admin fee charges");
    } finally {
      setLoading(false);
      setAdminFeesLoading(false);
    }
  }

  async function updateAdminFee(chargeId: number, payload: {
    status?: AdminFeeStatus;
    paid_cash_cents?: number;
    note?: string;
    apply_wallet?: boolean;
  }) {
    setFeeUpdating(prev => ({ ...prev, [chargeId]: true }));
    try {
      const updated = await ownerUpdateAdminFeeCharge(chargeId, payload);
      setAdminFees(prev => prev.map(f => (f.id === chargeId ? updated : f)));
      setAdminFeeNotice(`Updated admin fee for ${updated.client_label ?? `charge #${updated.id}`}`);
      setAdminFeesError(null);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setAdminFeesError(message || "Failed to update administration fee");
    } finally {
      setFeeUpdating(prev => ({ ...prev, [chargeId]: false }));
    }
  }

  async function deleteAdminFee(chargeId: number) {
    setFeeUpdating(prev => ({ ...prev, [chargeId]: true }));
    try {
      await ownerDeleteAdminFeeCharge(chargeId);
      // Be defensive: coerce to number in case a string sneaks in from callers
      const idNum = Number(chargeId);
      setAdminFees(prev => prev.filter(f => Number(f.id) !== idNum));
      setAdminFeeNotice(`Deleted admin fee charge #${chargeId}`);
      setAdminFeesError(null);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setAdminFeesError(message || "Failed to delete administration fee");
    } finally {
      setFeeUpdating(prev => ({ ...prev, [chargeId]: false }));
    }
  }

  useEffect(() => {
    void refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [start, end, status, payment, clientAccountId]);

  // Settings removed

  const selectedAccount = useMemo(
    () => (clientAccountId === "" ? null : accounts.find(a => a.account_id === Number(clientAccountId))) ,
    [clientAccountId, accounts]
  );
  const selectedLabel = useMemo(() => {
    if (!selectedAccount) return "All clients";
    return selectedAccount.name || selectedAccount.client_name || selectedAccount.client_email || `Client #${selectedAccount.account_id}`;
  }, [selectedAccount]);

  // Hide legacy bundle management inside the edit drawer
  // legacy bundle UI removed

  // ----- UI -----
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold">Payments — {selectedLabel}</h1>
        {selectedAccount && (
          <QuickWalletButton clientId={String(selectedAccount.client_user_id || "")} tz={tz} onChanged={() => void refresh()} />
        )}
      </div>


      {/* Filters */}
      <div className="grid gap-3 md:grid-cols-3">
        <div className="flex items-center gap-2">
        <label className="text-sm w-16">Start</label>
        <input
            type="date"
            value={start}
            onChange={(e) => onStartChange(e.target.value)}
            className="border rounded px-2 py-1 text-sm w-full"
        />
        </div>
        <div className="flex items-center gap-2">
        <label className="text-sm w-16">End</label>
        <input
            type="date"
            value={end}
            onChange={(e) => onEndChange(e.target.value)}
            className="border rounded px-2 py-1 text-sm w-full"
            disabled={!start}
            min={start || undefined}
        />
        </div>
        {dateError && (
            <div className="md:col-span-3 text-xs text-red-600">{dateError}</div>
        )}


        <div className="flex items-center justify-end gap-2">
          <span className="text-xs text-zinc-500">{selectedLabel}</span>
          <button className="px-3 py-1 rounded border text-sm" onClick={refresh} disabled={loading}>
            {loading ? "Loading..." : "Refresh"}
          </button>
        </div>
      </div>

      {/* Clients */}
      <div className="border rounded p-3 space-y-3">
        <div className="text-sm font-medium">Clients</div>

        <div className="flex items-start gap-2">
        {/* Search with suggestions */}
        <div className="relative w-full">
            <input
            value={search}
            onChange={(e) => {
                setSearch(e.target.value);
                setOpenSuggest(true);
                setHighlight(0);
            }}
            onFocus={() => {
                if (search.trim()) setOpenSuggest(true);
            }}
            onBlur={() => {
                // small delay so click can land
                setTimeout(() => setOpenSuggest(false), 120);
            }}
            onKeyDown={async (e) => {
                if (e.key === "ArrowDown") {
                e.preventDefault();
                if (!suggestions.length) return;
                setHighlight((i) => Math.min(i + 1, suggestions.length - 1));
                } else if (e.key === "ArrowUp") {
                e.preventDefault();
                if (!suggestions.length) return;
                setHighlight((i) => Math.max(i - 1, 0));
                } else if (e.key === "Enter") {
                e.preventDefault();
                const pick = suggestions[highlight];
                if (!pick) return;
                setClientAccountId(pick.account_id);
                setSearch(labelForAccount(pick));
                setOpenSuggest(false);
                } else if (e.key === "Escape") {
                setOpenSuggest(false);
                }
            }}
            placeholder="Search name or email"
            className="border rounded px-2 py-1 text-sm w-full"
            aria-autocomplete="list"
            aria-expanded={openSuggest}
            aria-controls="client-suggest"
            role="combobox"
            />

            {/* Dropdown */}
            {openSuggest && (
            <div
                id="client-suggest"
                role="listbox"
                className="absolute z-20 mt-1 w-full rounded border bg-white shadow-lg max-h-64 overflow-auto text-sm"
            >
                {accountsLoading && (
                <div className="px-3 py-2 text-zinc-500">Searching…</div>
                )}

                {!accountsLoading && suggestions.length === 0 && search.trim() && (
                <div className="px-3 py-2 text-zinc-500">No matches</div>
                )}

                {!accountsLoading &&
                suggestions.map((a, idx) => {
                    const active = idx === highlight;
                    const label = labelForAccount(a);
                    return (
                    <div
                        key={a.account_id}
                        role="option"
                        aria-selected={active}
                        className={`px-3 py-2 cursor-pointer ${
                        active ? "bg-blue-600 text-white" : "hover:bg-gray-50"
                        }`}
                        onMouseEnter={() => setHighlight(idx)}
                        onMouseDown={(e) => {
                        e.preventDefault(); // keep focus so onBlur delay works
                        setClientAccountId(a.account_id);
                        setSearch(label);
                        setOpenSuggest(false);
                        }}
                    >
                        <div className="font-medium truncate">{label}</div>
                        <div className={`text-xs truncate ${active ? "text-white/90" : "text-zinc-600"}`}>
                        {a.client_email || "-"}
                        </div>
                    </div>
                    );
                })}
            </div>
            )}
        </div>

        {/* Existing select still useful as a secondary chooser */}
        <select
            value={clientAccountId}
            onChange={(e) => setClientAccountId(e.target.value === "" ? "" : Number(e.target.value))}
            className="border rounded px-2 py-1 text-sm"
            title="Pick client account"
        >
            <option value="">All</option>
            {accounts.map((a) => (
            <option key={a.account_id} value={a.account_id}>
                {labelForAccount(a)}
            </option>
            ))}
        </select>

        <button
            className="px-2 py-1 border rounded text-sm"
            disabled={accountsLoading}
            onClick={() => {
            setClientAccountId("");
            setSearch("");
            setOpenSuggest(false);
            }}
        >
            Clear
        </button>
        </div>

      </div>

      {/* Settings removed */}

      {/* Status + Payment filters */}
      <div className="border rounded p-3 grid gap-3 md:grid-cols-2">
        <div>
          <div className="text-sm font-medium mb-2">Appointment status</div>
          <div className="flex flex-wrap gap-2">
            {ALL_STATUSES.map((s) => (
              <label key={s} className="flex items-center gap-2 text-sm border rounded px-2 py-1">
                <input
                  type="checkbox"
                  checked={status.includes(s)}
                  onChange={() =>
                    setStatus((cur) => (cur.includes(s) ? cur.filter((x) => x !== s) : [...cur, s]))
                  }
                />
                <span className="capitalize">{s}</span>
              </label>
            ))}
          </div>
        </div>
        <div>
          <div className="text-sm font-medium mb-2">Payment status</div>
          <div className="flex flex-wrap gap-2">
            {ALL_PAYMENT.map((p) => (
              <label key={p} className="flex items-center gap-2 text-sm border rounded px-2 py-1">
                <input
                  type="checkbox"
                  checked={payment.includes(p)}
                  onChange={() =>
                    setPayment((cur) => (cur.includes(p) ? cur.filter((x) => x !== p) : [...cur, p]))
                  }
                />
                <span className="capitalize">{p}</span>
              </label>
            ))}
          </div>
        </div>
      </div>

      {/* Totals (without Expected) */}
      {summary && (
        <div className="grid gap-2 md:grid-cols-2 lg:grid-cols-5">
          <div className="rounded border bg-gray-50 dark:bg-zinc-800/40 p-2 text-xs">
            <div className="text-zinc-600 dark:text-zinc-300">Missed appointments</div>
            <div className="text-base font-semibold text-zinc-900 dark:text-zinc-100">{summary.total_no_show ?? noShowCount}</div>
            <div className="text-[10px] text-zinc-500 dark:text-zinc-400">Matches current filters</div>
          </div>
          <div className="rounded border bg-gray-50 dark:bg-zinc-800/40 p-2 text-xs">
            <div className="text-zinc-500">Appointments</div>
            <div className="text-base font-semibold">{summary.total_appointments}</div>
          </div>
          <div className="rounded border bg-gray-50 dark:bg-zinc-800/40 p-2 text-xs">
            <div className="text-zinc-500">Paid</div>
            {(() => {
              const adminPaid = (adminFees || []).reduce((sum, f) => sum + Number(f.paid_cash_cents || 0) + Number(f.bundle_applied_cents || 0), 0);
              const total = Number(summary.total_paid_cents || 0) + adminPaid;
              return <div className="text-base font-semibold">{money(total)}</div>;
            })()}
            <div className="text-[10px] text-zinc-500">Includes admin fees · Wallet {money(summary.total_bundle_cents)}</div>
          </div>
          <div className="rounded border bg-gray-50 dark:bg-zinc-800/40 p-2 text-xs">
            <div className="text-zinc-500">Bundle balance</div>
            <div className="text-base font-semibold">{money(summary.total_wallet_balance_cents ?? 0)}</div>
            <div className="text-[10px] text-zinc-500">Unapplied wallet funds</div>
          </div>
          <div className="rounded border bg-gray-50 dark:bg-zinc-800/40 p-2 text-xs">
            <div className="text-zinc-500">Owed</div>
            {(() => {
              // Include outstanding admin-fee charges in the owed card
              const extra = (adminFees || [])
                .filter(f => f.status === 'unpaid')
                .reduce((sum, f) => sum + Math.max(0, (f.amount_cents || 0) - (f.paid_cash_cents || 0) - (f.bundle_applied_cents || 0)), 0);
              const total = (summary.total_owed_cents || 0) + extra;
              return <div className="text-base font-semibold">{money(total)}</div>;
            })()}
            <div className="text-[10px] text-zinc-500">Includes admin fees</div>
          </div>
        </div>
      )}

      {error && (
        <div className="text-sm text-red-600 border border-red-300 bg-red-50 rounded p-2">{error}</div>
      )}

      {/* Table */}
      <div className="overflow-x-auto rounded border">
        <table className="min-w-full text-sm">
            <thead className="bg-gray-50 dark:bg-zinc-800/40 text-xs font-medium border-b">
            <tr>
                <th className="px-3 py-2 text-left w-48">When • Client</th>
                <th className="px-3 py-2 text-left">Status</th>
                <th className="px-3 py-2 text-right">Duration</th>
                <th className="px-3 py-2 text-right">Price</th>
                <th className="px-3 py-2 text-right">Wallet</th>
                <th className="px-3 py-2 text-right">Owed</th>
                <th className="px-3 py-2 text-right w-48">Actions</th>
            </tr>
            </thead>
            <tbody>
            {rows.map((r, idx) => {
                const isEditing = !!editing[r.id];
                const e = editing[r.id];
                const currentAttendance = (e?.attendance_status ?? r.attendance_status ?? "attended") as AttendanceStatus;
                const attendanceBusy = !!attendanceUpdating[r.id];
                const attendanceLabel =
                  currentAttendance === "no_show"
                    ? "No show"
                    : currentAttendance === "late"
                    ? "Late"
                    : currentAttendance === "unknown"
                    ? "Unknown"
                    : "Attended";
                const attendanceChipClass =
                  currentAttendance === "no_show"
                    ? "border-red-200 bg-red-50 text-red-600 dark:border-red-400/60 dark:bg-red-500/10 dark:text-red-200"
                    : currentAttendance === "late"
                    ? "border-amber-200 bg-amber-50 text-amber-700 dark:border-amber-400/60 dark:bg-amber-500/10 dark:text-amber-200"
                    : currentAttendance === "unknown"
                    ? "border-zinc-300 bg-zinc-50 text-zinc-600 dark:border-zinc-500/60 dark:bg-zinc-700/40 dark:text-zinc-200"
                    : "border-emerald-200 bg-emerald-50 text-emerald-600 dark:border-emerald-400/60 dark:bg-emerald-500/10 dark:text-emerald-200";
                const statusForDrawer = (e?.status ?? r.status) as "booked" | "completed" | "canceled";
                const cancelReasonDisplay = (e?.cancel_reason ?? r.cancel_reason ?? "").trim();
                return (
                <tr
                    key={r.id}
                    className={`border-t hover:bg-gray-50 dark:hover:bg-zinc-800/30 ${
                    idx % 2 ? "bg-white dark:bg-transparent" : "bg-gray-50/40 dark:bg-zinc-900/10"
                    }`}
                >
                    <td className="px-3 py-2">
                    <div className="font-medium">
                        {new Date(r.start_utc).toLocaleString([], {
                        dateStyle: "medium",
                        timeStyle: "short",
                        ...(tz ? { timeZone: tz } : {}),
                        })}
                    </div>
                    <div className="text-xs text-zinc-600 truncate flex items-center gap-2">
                        <span>
                          {r.client_label || (r.client_account_id ? `Account #${r.client_account_id}` : "Client")}
                        </span>
                        {(r as any).lesson_person_name && (
                          <span>· {(r as any).lesson_person_name}</span>
                        )}
                        {(r as any).is_group && (
                          <span className="inline-flex items-center px-2 py-[1px] rounded-full text-[10px] font-medium border border-zinc-300 bg-zinc-50">Group</span>
                        )}
                        {(() => {
                          const cid = clientUserIdForRow(r);
                          const bal = cid ? walletByClient[cid] : undefined;
                          if (typeof bal === 'number') {
                            const label = bal > 0 ? `Wallet ${money(bal)}` : `Wallet $0.00`;
                            return (
                              <span className="inline-flex items-center px-2 py-[1px] rounded-full text-[10px] font-medium border border-zinc-300 bg-zinc-50">
                                {label}
                              </span>
                            );
                          }
                          return null;
                        })()}
                        {(currentAttendance === "no_show" || currentAttendance === "late" || currentAttendance === "unknown") && (
                          <span className={`inline-flex items-center gap-1 px-2 py-[1px] rounded-full text-[10px] font-semibold border ${attendanceChipClass}`}>
                            {currentAttendance === "no_show" ? "🚩 No show" : attendanceLabel}
                          </span>
                        )}
                    </div>
                    </td>
                    <td className="px-3 py-2">
                    <StatusChip kind={r.payment_status} />
                    </td>
                    <td className="px-3 py-2 text-right">{r.duration_minutes}m</td>
                    <td className="px-3 py-2 text-right">
                    {typeof r.price_cents === "number" ? money(r.price_cents) : "-"}
                    </td>
                    <td className="px-3 py-2 text-right">{money(r.bundle_applied_cents)}</td>
                    <td
                    className={`px-3 py-2 text-right ${
                        r.owed_cents > 0 ? "text-red-700 font-medium" : ""
                    }`}
                    >
                    {money(r.owed_cents)}
                    </td>
                      <td className="px-3 py-2 text-right">
                        {!isEditing ? (
                          <TinyButton
                            onClick={async () => {
                              startEdit(r);
                              setOpenRowId(r.id);
                              try {
                                const d = await ownerGetAppointmentDetails(r.id);
                                const gid = (d as any).group_id as (string | null | undefined);
                                setOpenGroupId(gid || null);
                                if (gid) {
                                  setGroupLoading(true);
                                  try {
                                    const g = await ownerGetGroupDetails(gid);
                                    setGroupAttendees(g.attendees as any);
                                  } finally {
                                    setGroupLoading(false);
                                  }
                                } else {
                                  setGroupAttendees([]);
                                }
                                setEdit(r.id, {
                                  owner_private_note: d.owner_private_note ?? "",
                                  attendance_status: (d.attendance_status as AttendanceStatus) ?? "attended",
                                  cancel_reason: d.cancel_reason ?? r.cancel_reason ?? "",
                                  ...(typeof d.price_override_cents === 'number' ? { price_override_dollars: (d.price_override_cents / 100).toFixed(2) } : {}),
                                });
                              } catch {}
                            }}
                          >
                            Edit
                          </TinyButton>
                        ) : (
                          <>
                            <TinyButton onClick={async () => {
                              setOpenRowId(r.id);
                              try {
                                const d = await ownerGetAppointmentDetails(r.id);
                                setEdit(r.id, {
                                  owner_private_note: d.owner_private_note ?? "",
                                  attendance_status: (d.attendance_status as AttendanceStatus) ?? "attended",
                                  ...(typeof d.price_override_cents === 'number' ? { price_override_dollars: (d.price_override_cents / 100).toFixed(2) } : {}),
                                });
                                if (r.client_account_id && !peopleByAccount[r.client_account_id]) {
                                  const detail = await ownerGetClientDetail(r.client_account_id);
                                  setPeopleByAccount(prev => ({ ...prev, [r.client_account_id as number]: detail.people || [] }));
                                }
                              } catch {}
                            }}>Edit</TinyButton>

                            <Drawer
                              open={openRowId === r.id}
                              onClose={() => {
                                setOpenRowId(null);
                                // If you prefer discarding edits on close:
                                // cancelEdit(r.id);
                              }}
                              width={420} // tweak as you like
                              title="Edit appointment"
                            >
                              <form
                                className="space-y-3"
                                onSubmit={async (ev) => {
                                  ev.preventDefault();
                                  const ok = await saveEdit(r.id);
                                  if (ok) setOpenRowId(null);
                                }}
                              >
                                {/* Removed total owed summary per request */}
                                {/* Group lesson summary */}
                                {openGroupId && (
                                  <div className="rounded-md border p-2 bg-indigo-50">
                                    <div className="flex items-center justify-between mb-1">
                                      <div className="text-xs font-medium text-indigo-900">Group lesson</div>
                                      <div className="text-[10px] text-indigo-800">Group ID: {openGroupId.slice(0,8)}…</div>
                                    </div>
                                    {groupLoading ? (
                                      <div className="text-xs text-zinc-700">Loading attendees…</div>
                                    ) : (
                                      <div className="space-y-1">
                                        {groupAttendees.length === 0 && <div className="text-xs">No attendees.</div>}
                                    {groupAttendees.map((a) => (
                                      <AttendeeRow
                                        key={a.appointment_id}
                                        attendee={a}
                                        groupId={openGroupId!}
                                        reload={refresh}
                                        refreshGroup={async () => {
                                          const g = await ownerGetGroupDetails(openGroupId!);
                                          setGroupAttendees(g.attendees as any);
                                        }}
                                      />
                                    ))}
                                        <div className="flex gap-2 pt-1">
                                          <button
                                            type="button"
                                            className="rounded border px-2 py-1 text-xs"
                                            onClick={async () => {
                                              try {
                                                if (!openGroupId) return;
                                                await adminGroupCancel(openGroupId);
                                                await refresh();
                                                alert('Group canceled');
                                              } catch (e: any) {
                                                alert(e?.message || 'Failed to cancel group');
                                              }
                                            }}
                                          >Cancel group</button>
                                        </div>
                                      </div>
                                    )}
                                  </div>
                                )}
                                {e?.error && (
                                  <div className="text-xs text-red-600 border border-red-300 bg-red-50 rounded p-2">
                                    {e.error}
                                  </div>
                                )}
                                {/* Client section */}
                                <div className="border rounded p-2 space-y-2">
                                  <div className="text-xs font-medium text-zinc-600">Client</div>
                                  <div className="grid grid-cols-2 gap-3">
                                    <div className="text-left">
                                      <FieldLabel htmlFor={`client-name-${r.id}`}>Name</FieldLabel>
                                      <input
                                        id={`client-name-${r.id}`}
                                        className="border rounded px-2 py-1 text-xs w-full"
                                        type="text"
                                        value={e.client_name ?? ""}
                                        onChange={(ev) => setEdit(r.id, { client_name: ev.target.value })}
                                        placeholder="Client name (account)"
                                      />
                                    </div>
                                    <div className="text-left">
                                      <FieldLabel htmlFor={`client-email-${r.id}`}>Email</FieldLabel>
                                      <input
                                        id={`client-email-${r.id}`}
                                        className={`border rounded px-2 py-1 text-xs w-full ${e.errors?.client_email ? 'border-red-500' : ''}`}
                                        type="email"
                                        value={e.client_email ?? ""}
                                        onChange={(ev) => setEdit(r.id, { client_email: ev.target.value })}
                                        placeholder="email@example.com"
                                      />
                                      {e.errors?.client_email && (
                                        <div className="text-[11px] text-red-600 mt-1">{e.errors.client_email}</div>
                                      )}
                                    </div>
                                  </div>
                                </div>

                                {/* Lesson for (editable name) */}
                                <div className="border rounded p-2 space-y-2">
                                  <div className="text-xs font-medium text-zinc-600">Lesson for</div>
                                  <div className="text-left grid grid-cols-2 gap-3">
                                    <div>
                                      <FieldLabel>Person</FieldLabel>
                                      {r.client_account_id && peopleByAccount[r.client_account_id] && peopleByAccount[r.client_account_id]!.length ? (
                                        <select
                                          className="border rounded px-2 py-1 text-xs w-full"
                                          value={(() => {
                                            const name = e.lesson_person_name ?? (r.lesson_person_name || "");
                                            const hit = peopleByAccount[r.client_account_id!].find(p => p.full_name === name);
                                            return hit ? String(hit.id) : "";
                                          })()}
                                          onChange={(ev) => {
                                            const v = ev.target.value;
                                            if (v === "") setEdit(r.id, { lesson_person_name: "" });
                                            else {
                                              const p = peopleByAccount[r.client_account_id!].find(pp => String(pp.id) === v);
                                              setEdit(r.id, { lesson_person_name: p?.full_name || "" });
                                            }
                                          }}
                                        >
                                          <option value="">(custom)</option>
                                          {peopleByAccount[r.client_account_id].map(p => (
                                            <option key={p.id} value={p.id}>{p.full_name}{p.email ? ` <${p.email}>` : ''}</option>
                                          ))}
                                        </select>
                                      ) : (
                                        <input
                                          className="border rounded px-2 py-1 text-xs w-full"
                                          placeholder="e.g., Fluffy Junior"
                                          value={e.lesson_person_name ?? (r.lesson_person_name || "")}
                                          onChange={(ev) => setEdit(r.id, { lesson_person_name: ev.target.value })}
                                        />
                                      )}
                                    </div>
                                    <div>
                                      <FieldLabel>Email</FieldLabel>
                                      <div className="border rounded px-2 py-1 text-xs bg-gray-50">
                                        {r.lesson_person_email || "—"}
                                      </div>
                                    </div>
                                  </div>
                                </div>

                                {statusForDrawer === "canceled" && cancelReasonDisplay && (
                                  <div className="border rounded p-2 space-y-1 bg-red-50/40">
                                    <div className="text-xs font-medium text-red-700">Cancellation reason</div>
                                    <p className="text-xs text-red-800 whitespace-pre-wrap">{cancelReasonDisplay}</p>
                                  </div>
                                )}

                                {/* Schedule section */}
                                <div className="border rounded p-2 space-y-2">
                                  <div className="text-xs font-medium text-zinc-600">Schedule</div>
                                  <div className="grid grid-cols-3 gap-3">
                                    <div className="text-left">
                                      <FieldLabel htmlFor={`date-${r.id}`}>Date</FieldLabel>
                                      <input
                                        id={`date-${r.id}`}
                                        type="date"
                                        className={`border rounded px-2 py-1 text-xs w-full ${e.errors?.date ? 'border-red-500' : ''}`}
                                        value={e.date ?? ""}
                                        onChange={(ev) => setEdit(r.id, { date: ev.target.value })}
                                      />
                                      {e.errors?.date && (
                                        <div className="text-[11px] text-red-600 mt-1">{e.errors.date}</div>
                                      )}
                                    </div>
                                    <div className="text-left">
                                      <FieldLabel htmlFor={`time-${r.id}`}>Start time</FieldLabel>
                                      <input
                                        id={`time-${r.id}`}
                                        type="time"
                                        className={`border rounded px-2 py-1 text-xs w-full ${e.errors?.time ? 'border-red-500' : ''}`}
                                        value={e.time ?? ""}
                                        onChange={(ev) => setEdit(r.id, { time: ev.target.value })}
                                      />
                                      {e.errors?.time && (
                                        <div className="text-[11px] text-red-600 mt-1">{e.errors.time}</div>
                                      )}
                                    </div>
                                    <div className="text-left">
                                      <FieldLabel htmlFor={`dur-${r.id}`}>Duration (min)</FieldLabel>
                                      <input
                                        id={`dur-${r.id}`}
                                        type="number"
                                        min={5}
                                        step={5}
                                        className={`border rounded px-2 py-1 text-xs w-full ${e.errors?.duration_minutes ? 'border-red-500' : ''}`}
                                        value={e.duration_minutes ?? ""}
                                        onChange={(ev) => setEdit(r.id, { duration_minutes: ev.target.value })}
                                      />
                                      {e.errors?.duration_minutes && (
                                        <div className="text-[11px] text-red-600 mt-1">{e.errors.duration_minutes}</div>
                                      )}
                                    </div>
                                  </div>
                                  {/* Appointment status (read-only) + Cancel button */}
                                  <div className="text-left">
                                    <FieldLabel htmlFor={`appt-status-${r.id}`}>Appointment status</FieldLabel>
                                    <div className="flex items-center gap-2">
                                      <span id={`appt-status-${r.id}`} className="inline-block border rounded px-2 py-1 text-xs capitalize">
                                        {(e.status ?? r.status) as string}
                                      </span>
                                      <button
                                        type="button"
                                        className="inline-flex items-center rounded px-2 py-1 text-xs bg-red-600 text-white disabled:opacity-50"
                                        disabled={(e.status ?? r.status) === 'canceled'}
                                        onClick={async () => {
                                          try {
                                            await cancelAppointment(r.id)
                                            await refresh()
                                          } catch (err: any) {
                                            setError(err?.message || 'Failed to cancel appointment')
                                          }
                                        }}
                                      >
                                        Cancel appointment
                                      </button>
                                    </div>
                                  </div>
                                </div>

                                {/* Attendance and notes */}
                                <div className="border rounded p-2 space-y-2">
                                  <div className="flex items-center justify-between text-xs font-medium text-zinc-600">
                                    <span>Visit details</span>
                                    <span className={`inline-flex items-center gap-1 px-2 py-[1px] rounded-full text-[10px] font-semibold border ${attendanceChipClass}`}>
                                      {currentAttendance === "no_show" ? "🚩 No show" : attendanceLabel}
                                    </span>
                                  </div>
                                  <div className="grid grid-cols-3 gap-3 items-end">
                                    <div className="text-left col-span-1">
                                      <FieldLabel htmlFor={`att-${r.id}`}>Attendance</FieldLabel>
                                      <select
                                        id={`att-${r.id}`}
                                        className="border rounded px-2 py-1 text-xs w-full"
                                        value={e.attendance_status ?? "attended"}
                                        onChange={(ev) => setEdit(r.id, { attendance_status: ev.target.value as any })}
                                      >
                                        <option value="unknown">Unknown</option>
                                        <option value="attended">Attended</option>
                                        <option value="late">Late</option>
                                        <option value="no_show">No show</option>
                                      </select>
                                      <div className="flex items-center gap-2 mt-2">
                                        {currentAttendance !== "no_show" ? (
                                          <button
                                            type="button"
                                            className="inline-flex items-center gap-1 rounded border border-red-500 px-2 py-1 text-xs font-medium text-red-600 hover:bg-red-50 disabled:opacity-60 disabled:cursor-not-allowed"
                                            onClick={() => quickSetAttendance(r.id, "no_show")}
                                            disabled={attendanceBusy}
                                          >
                                            🚩 Mark no show
                                          </button>
                                        ) : (
                                          <button
                                            type="button"
                                            className="inline-flex items-center gap-1 rounded border border-emerald-500 px-2 py-1 text-xs font-medium text-emerald-600 hover:bg-emerald-50 disabled:opacity-60 disabled:cursor-not-allowed"
                                            onClick={() => quickSetAttendance(r.id, "attended")}
                                            disabled={attendanceBusy}
                                          >
                                            Mark attended
                                          </button>
                                        )}
                                      </div>
                                    </div>
                                    {/* Late minutes removed */}
                                    <div className="text-left col-span-3">
                                      <FieldLabel htmlFor={`note-${r.id}`}>Private note</FieldLabel>
                                      <textarea
                                        id={`note-${r.id}`}
                                        className="border rounded px-2 py-1 text-xs w-full"
                                        rows={3}
                                        value={e.owner_private_note ?? ""}
                                        onChange={(ev) => setEdit(r.id, { owner_private_note: ev.target.value })}
                                        placeholder="Only visible to you"
                                      />
                                    </div>
                                  </div>
                                </div>

                                {/* Payment section */}
                                <div className="border rounded p-2 space-y-2">
                                  <div className="text-xs font-medium text-zinc-600">Payment</div>
                                  <div className="grid grid-cols-2 gap-3">
                                    <div className="text-left">
                                      <FieldLabel htmlFor={`status-${r.id}`}>Status</FieldLabel>
                                      <select
                                        id={`status-${r.id}`}
                                        className="border rounded px-2 py-1 text-xs w-full"
                                        value={e.payment_status ?? "unpaid"}
                                        onChange={(ev) =>
                                          setEdit(r.id, { payment_status: ev.target.value as any })
                                        }
                                      >
                                        <option value="paid">Paid</option>
                                        <option value="refunded">Refunded</option>
                                        <option value="waived">Waived</option>
                                        <option value="unpaid">Unpaid</option>
                                      </select>
                                    </div>

                                    {/* Cash input removed; wallet use is automatic */}

                                    <div className="text-left col-span-2">
                                      <FieldLabel htmlFor={`override-${r.id}`}>Custom Price (optional)</FieldLabel>
                                      <MoneyInput
                                        id={`override-${r.id}`}
                                        value={e.price_override_dollars}
                                        onChange={(v) => setEdit(r.id, { price_override_dollars: v })}
                                        placeholder="(keep)"
                                      />
                                      <div className="text-[11px] text-zinc-500 mt-1">
                                        Leave blank to keep current price.
                                      </div>
                                    </div>
                                  </div>
                                </div>

                                

                                <div className="flex items-center justify-between gap-2 pt-1">
                                  <div>
                                    <TinyButton
                                      type="button"
                                      kind="ghost"
                                      onClick={async () => {
                                        const sure = window.confirm('Permanently delete this appointment? This restores any wallet funds and cannot be undone.');
                                        if (!sure) return;
                                        try {
                                          await ownerDeleteAppointment(r.id);
                                          cancelEdit(r.id);
                                          setOpenRowId(null);
                                          await refresh();
                                        } catch (err: any) {
                                          window.alert(err?.message || 'Failed to delete appointment');
                                        }
                                      }}
                                    >
                                      Delete
                                    </TinyButton>
                                  </div>
                                  <div className="flex items-center gap-2">
                                    <TinyButton type="button" onClick={() => { cancelEdit(r.id); setOpenRowId(null); }}>
                                      Cancel
                                    </TinyButton>
                                    <TinyButton kind="primary" type="submit">Save</TinyButton>
                                  </div>
                                </div>
                              </form>
                            </Drawer>
                          </>
                        )}
                      </td>


                </tr>
                );
            })}

            {!rows.length && (
                <tr>
                <td
                    colSpan={7}
                    className="px-3 py-4 text-center text-sm text-zinc-500"
                >
                    No matching appointments for the selected filters.
                </td>
                </tr>
            )}
            </tbody>
        </table>
        </div>

      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold">Administration Fees</h2>
          {adminFeesLoading && <span className="text-xs text-zinc-500">Loading…</span>}
        </div>
        {adminFeeNotice && (
          <div className="rounded border border-green-200 bg-green-50 px-3 py-2 text-xs text-green-700">
            {adminFeeNotice}
          </div>
        )}
        {adminFeesError && (
          <div className="rounded border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700">
            {adminFeesError}
          </div>
        )}

        <div className="overflow-x-auto rounded border">
          {adminFees.length ? (
            <table className="min-w-full text-sm">
              <thead className="bg-gray-50 dark:bg-zinc-800/40 text-xs font-medium border-b">
                <tr>
                  <th className="px-3 py-2 text-left">Created</th>
                  <th className="px-3 py-2 text-left">Client</th>
                  <th className="px-3 py-2 text-right">Amount</th>
                  <th className="px-3 py-2 text-right">Wallet</th>
                  <th className="px-3 py-2 text-right">Cash</th>
                  <th className="px-3 py-2 text-right">Owed</th>
                  <th className="px-3 py-2 text-left">Status</th>
                  <th className="px-3 py-2 text-right">Actions</th>
                </tr>
              </thead>
              <tbody>
                {adminFees.map((fee) => {
                  const computed = Math.max(fee.amount_cents - fee.bundle_applied_cents - fee.paid_cash_cents, 0);
                  const outstanding = (fee.status === "waived" || fee.status === "refunded") ? 0 : computed;
                  const updating = !!feeUpdating[fee.id];
                  const deletable = (fee.status === "unpaid" && fee.paid_cash_cents === 0 && fee.bundle_applied_cents === 0);
                  return (
                    <tr key={fee.id} className="border-t">
                      <td className="px-3 py-2 text-xs text-zinc-600">
                        {new Date(fee.created_at).toLocaleString([], { dateStyle: "medium", timeStyle: "short", ...(tz ? { timeZone: tz } : {}) })}
                      </td>
                      <td className="px-3 py-2 text-sm">
                        {fee.client_label || `Account #${fee.client_account_id}`}
                      </td>
                      <td className="px-3 py-2 text-right">{money(fee.amount_cents)}</td>
                      <td className="px-3 py-2 text-right">{money(fee.bundle_applied_cents)}</td>
                      <td className="px-3 py-2 text-right">{money(fee.paid_cash_cents)}</td>
                      <td className={`px-3 py-2 text-right ${outstanding > 0 ? "text-red-700 font-medium" : ""}`}>
                        {money(outstanding)}
                      </td>
                      <td className="px-3 py-2">
                        <select
                          className="rounded border px-2 py-1 text-xs"
                          value={fee.status}
                          disabled={updating}
                          onChange={(ev) => {
                            const next = ev.target.value as AdminFeeStatus;
                            if (next === fee.status) return;
                            void updateAdminFee(fee.id, { status: next });
                          }}
                        >
                          {ADMIN_FEE_STATUSES.map((s) => (
                            <option key={s} value={s}>{s}</option>
                          ))}
                        </select>
                      </td>
                      <td className="px-3 py-2 text-right space-x-2">
                        <button
                          type="button"
                          className="rounded border px-2 py-1 text-xs"
                          disabled={updating || outstanding <= 0}
                          onClick={() => { void updateAdminFee(fee.id, { apply_wallet: true }); }}
                        >
                          Apply wallet
                        </button>
                        <button
                          type="button"
                          className="rounded border px-2 py-1 text-xs text-red-700"
                          disabled={updating || !deletable}
                          title={deletable ? "Delete this unpaid admin fee" : "Only unpaid fees with no payments can be deleted"}
                          onClick={() => {
                            if (confirm("Delete this admin fee charge? This cannot be undone.")) {
                              void deleteAdminFee(fee.id);
                            }
                          }}
                        >
                          Delete
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          ) : (
            <div className="px-4 py-6 text-sm text-zinc-500">
              {adminFeesLoading ? "Loading…" : "No administration fee charges yet."}
            </div>
          )}
        </div>
      </div>

    </div>
  );
}
