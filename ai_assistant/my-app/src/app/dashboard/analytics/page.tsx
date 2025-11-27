import { auth } from "@/auth";
import OwnerPaymentsAnalyticsClient from "./OwnerPaymentsAnalyticsClient";

export default async function Page() {
  const session = await auth();
  type UserWithRole = { role?: string };
  if (!session?.user || ((session.user as UserWithRole).role !== "OWNER")) {
    return <div className="p-6">Forbidden.</div>;
  }
  // Default range: current month (owner-local unknown here; use server local date)
  const now = new Date();
  const start = new Date(now.getFullYear(), now.getMonth(), 1);
  const end = new Date(now.getFullYear(), now.getMonth() + 1, 0);
  const toISO = (d: Date) => d.toISOString().slice(0,10);
  return (
    <div className="p-6 space-y-4">
      <h1 className="text-xl font-semibold">Appointment details</h1>
      <OwnerPaymentsAnalyticsClient initialStart={toISO(start)} initialEnd={toISO(end)} />
    </div>
  );
}
