import Link from "next/link";
import { auth } from "@/auth";

export default async function DashboardNav() {
  const session = await auth();
  const role = typeof session?.user?.role === "string" ? session.user.role.toUpperCase() : undefined;

  return (
    <nav className="flex items-center gap-3 md:gap-4 overflow-x-auto whitespace-nowrap no-scrollbar">
      {role === "OWNER" && (
        <>
          <Link href="/dashboard/clients" className="text-sm underline">Clients</Link>
          <Link href="/owner/scheduling" className="text-sm underline">Scheduling</Link>
          <Link href="/owner/messages" className="text-sm underline">Messaging</Link>
          <Link href="/dashboard/analytics" className="text-sm underline">Appointment details</Link>
          <Link href="/dashboard/pricing" className="text-sm underline">Pricing</Link>
      </>
    )}

      {role === "CLIENT" && (
        <>
          <Link href="/profile" className="text-sm underline">Profile</Link>
          <Link href="/scheduling" className="text-sm underline">
            My Scheduling
          </Link>
          <Link href="/client/appointments" className="text-sm underline">
            My Appointments
          </Link>
          {/* <Link href="/client/payments" className="text-sm underline">My Payments</Link> */}
        </>
      )}

    </nav>
  );
}
