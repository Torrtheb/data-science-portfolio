// src/app/scheduling/page.tsx
import { auth } from "@/auth";
import { redirect } from "next/navigation";
import UserSchedulingClient from "./ui/UserSchedulingClient";

export default async function UserSchedulingPage() {
  const session = await auth();
  if (!session?.user) redirect("/login?error=CredentialsSignin");

  const tz = (session.user as { timezone?: string } | null | undefined)?.timezone ?? "America/Toronto";
  return (
    <main className="p-4 sm:p-6 space-y-4">
      <h1 className="text-2xl font-semibold">My Schedule</h1>
      <UserSchedulingClient initialTimezone={tz} />
    </main>
  );
}
