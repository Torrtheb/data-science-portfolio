// src/app/profile/page.tsx
import { auth } from "@/auth";
import ProfileClient from "./ProfileClient";
import { redirect } from "next/navigation";

export default async function ProfilePage() {
  const session = await auth();
  if (!session?.user) redirect("/login");

  const role = (session.user as { role?: "OWNER" | "CLIENT" } | null | undefined)?.role as
    | "OWNER"
    | "CLIENT"
    | undefined;
  const tz = (session.user as { timezone?: string } | null | undefined)?.timezone ?? "America/Toronto";

  // IMPORTANT CHANGE: do NOT redirect owners anymore.
  // We want both OWNER and CLIENT to open /profile to change their timezone.
  return <ProfileClient initialTimezone={tz} role={role} />;
}
